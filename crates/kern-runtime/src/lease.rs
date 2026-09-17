//! The lease shell. The pool decides on the host which pages and slots
//! a sequence gets and which bytes move; the runtime runs those moves on
//! the compute stream and the pool's remaps on a thread of its own.
//!
//! Every lease operation begins with `poll`: land the remaps the thread
//! has finished, their fresh chunks zeroed on the stream so a page or a
//! slot comes out of the pool as it did at load, then hand the thread the
//! plan the pool has pending, to run once both streams have passed
//! everything enqueued so far. One remap is in flight at a time; a
//! [`Denied::Remapping`] tells the caller to ask again once it lands. A
//! leased slot is zeroed on the stream (a fresh sequence's recurrent
//! state); a checkpoint, a restore and a fork move whole pages and whole
//! slots, as the pool's [`Copies`] say.

use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::Arc;
use std::thread::JoinHandle;

use cudarc::driver::{sys, CudaContext};

use crate::device::{landed, record, Mapper};
use crate::error::{bail, cuda_check};
use crate::{Error, Result, Runtime};
use kern_pool::{Checkpoint, Copies, Denied, Lease, Remap};

/// What the remap thread is told: a plan to run once both streams have
/// passed `after` (recorded events, destroyed by the thread), or to stop.
enum Job {
    Run { plan: Remap, after: [u64; 2] },
    Stop,
}

/// The remap thread and the channels to it.
pub(crate) struct Remaps {
    jobs: Sender<Job>,
    done: Receiver<Result<Remap>>,
    thread: Option<JoinHandle<()>>,
}

impl Remaps {
    /// Start the thread that owns `mapper` from here on.
    pub(crate) fn spawn(ctx: Arc<CudaContext>, mapper: Mapper) -> Result<Remaps> {
        let (jobs, job_rx) = mpsc::channel();
        let (done_tx, done) = mpsc::channel();
        let thread = std::thread::Builder::new()
            .name("kern-remap".into())
            .spawn(move || remap_thread(ctx, mapper, job_rx, done_tx))
            .map_err(|e| Error::Cuda(format!("spawning the remap thread: {e}")))?;
        Ok(Remaps { jobs, done, thread: Some(thread) })
    }

    /// Tell the thread to stop and wait for it; the arenas go with it.
    pub(crate) fn stop(&mut self) {
        let _ = self.jobs.send(Job::Stop);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// Runs plans until told to stop, then lets the arenas and chunks go.
fn remap_thread(ctx: Arc<CudaContext>, mut mapper: Mapper, jobs: Receiver<Job>, done: Sender<Result<Remap>>) {
    if let Err(e) = ctx.bind_to_thread() {
        let _ = done.send(Err(e.into()));
        return;
    }
    while let Ok(job) = jobs.recv() {
        match job {
            Job::Run { plan, after } => {
                let evs = after.map(|e| e as sys::CUevent);
                let r = evs
                    .iter()
                    .try_for_each(|&ev| cuda_check(unsafe { sys::cuEventSynchronize(ev) }, "cuEventSynchronize"))
                    .and_then(|()| mapper.run(&plan));
                for ev in evs {
                    unsafe { sys::cuEventDestroy_v2(ev) };
                }
                if done.send(r.map(|()| plan)).is_err() {
                    break;
                }
            }
            Job::Stop => break,
        }
    }
    drop(mapper);
}

impl Runtime {
    /// Lease the pages `tokens` slots need and, when the manifest has
    /// per-sequence states, a slot in each of them, all or nothing
    /// ([`Error::Denied`] says why not). The lease is the only source of
    /// `slot_mapping` values, page-table rows and line indices for the
    /// sequence; the slot is zeroed on the stream (a fresh sequence's
    /// recurrent state), and everything returns when the lease drops.
    pub fn lease(&mut self, tokens: usize) -> Result<Lease> {
        self.ctx.bind_to_thread()?;
        self.poll()?;
        let lease = match self.pool.lease(tokens) {
            Ok(l) => l,
            Err(d) => return self.denied(d),
        };
        for (name, st) in &self.manifest.states {
            let Some(range) = lease.seq_bytes(st.bytes_per_seq).filter(|_| st.is_per_seq()) else { continue };
            let s = self.states.get_mut(name).unwrap();
            let mut view = s.view(range)?;
            self.stream.memset_zeros(&mut view)?;
        }
        Ok(lease)
    }

    /// A slot in each per-sequence state and no pages, zeroed on the
    /// stream: this rank's share of a sequence whose positions live on
    /// another rank (a tensor-parallel peer holds a slice of every row's
    /// recurrent state). Empty when the manifest has no per-sequence state.
    pub fn lease_slot(&mut self) -> Result<Lease> {
        self.ctx.bind_to_thread()?;
        self.poll()?;
        let lease = match self.pool.lease_slot() {
            Ok(l) => l,
            Err(d) => return self.denied(d),
        };
        for (name, st) in &self.manifest.states {
            let Some(range) = lease.seq_bytes(st.bytes_per_seq).filter(|_| st.is_per_seq()) else { continue };
            let s = self.states.get_mut(name).unwrap();
            let mut view = s.view(range)?;
            self.stream.memset_zeros(&mut view)?;
        }
        Ok(lease)
    }

    /// The first `len` tokens of `lease` as a [`Checkpoint`] its
    /// sequence keeps running past: pages shared, the per-sequence state
    /// copied into a fresh slot ([`Error::Denied`] when none is free).
    pub fn checkpoint(&mut self, lease: &mut Lease, len: usize) -> Result<Checkpoint> {
        self.ctx.bind_to_thread()?;
        self.poll()?;
        let (cp, copies) = match self.pool.checkpoint(lease, len) {
            Ok(x) => x,
            Err(d) => return self.denied(d),
        };
        self.copy(&copies)?;
        Ok(cp)
    }

    /// The first `len` tokens of a finished sequence as a [`Checkpoint`]:
    /// its pages past `len` return, its state slot moves over as it is.
    pub fn retire(&mut self, lease: Lease, len: usize) -> Checkpoint {
        self.pool.retire(lease, len)
    }

    /// A sequence continuing from the first `len` tokens of `cp` with room
    /// for `tokens`: shares those pages (copying the one `len` ends
    /// inside), copies the state into a fresh slot, and hands out a lease
    /// that names positions from `len` on. `len` is the checkpoint's own
    /// length, or any whole number of its pages when it holds no state.
    /// [`Error::Denied`] as for [`Runtime::lease`].
    pub fn lease_from(&mut self, cp: &Checkpoint, len: usize, tokens: usize) -> Result<Lease> {
        self.ctx.bind_to_thread()?;
        self.poll()?;
        let (lease, copies) = match self.pool.restore(cp, len, tokens) {
            Ok(x) => x,
            Err(d) => return self.denied(d),
        };
        self.copy(&copies)?;
        Ok(lease)
    }

    /// A sequence branched off the first `len` tokens of `parent`, which
    /// keeps running: the whole pages shared, the page `len` ends inside
    /// copied, the parent's state copied into a fresh slot. With a
    /// recurrent state `len` is the parent's position — the state is the
    /// parent's as of now. The lease names positions from `len` on.
    pub fn fork(&mut self, parent: &mut Lease, len: usize, tokens: usize) -> Result<Lease> {
        self.ctx.bind_to_thread()?;
        self.poll()?;
        let (lease, copies) = match self.pool.fork(parent, len, tokens) {
            Ok(x) => x,
            Err(d) => return self.denied(d),
        };
        self.copy(&copies)?;
        Ok(lease)
    }

    /// Run a pool decision's byte moves on the stream: whole pages of every
    /// paged state, whole slots of every per-sequence state.
    fn copy(&mut self, c: &Copies) -> Result<()> {
        let unit = self.pool.unit();
        for (name, st) in &self.manifest.states {
            let s = &self.states[name];
            let moves: Vec<(u64, u64, u64)> = if st.is_per_seq() {
                c.slot.iter().map(|&(a, b)| (a as u64, b as u64, st.bytes_per_seq)).collect()
            } else if st.bytes_per_token > 0 {
                c.pages.iter().map(|&(a, b)| (a as u64, b as u64, unit * st.bytes_per_token)).collect()
            } else {
                Vec::new()
            };
            for (a, b, bytes) in moves {
                let src = s.view((a * bytes) as usize..((a + 1) * bytes) as usize)?;
                let mut dst = s.view((b * bytes) as usize..((b + 1) * bytes) as usize)?;
                self.stream.memcpy_dtod(&src, &mut dst)?;
            }
        }
        Ok(())
    }

    /// A denial, after handing the remap it may have planned to the thread.
    pub(crate) fn denied<T>(&mut self, d: Denied) -> Result<T> {
        self.poll()?;
        Err(d.into())
    }

    /// Land the remaps the thread has run — their fresh chunks zeroed on
    /// the stream, so a page or a slot comes out of the pool as it did at
    /// load — and hand the thread the plan the pool has pending: it runs
    /// once the stream has passed everything enqueued so far.
    pub(crate) fn poll(&mut self) -> Result<()> {
        let mut parking = std::mem::take(&mut self.parking);
        for (cp, ev) in parking.drain(..) {
            if landed(ev)? {
                unsafe { sys::cuEventDestroy_v2(ev) };
                drop(cp);
            } else {
                self.parking.push((cp, ev));
            }
        }
        loop {
            match self.remaps.done.try_recv() {
                Ok(Ok(plan)) => {
                    self.zero_fresh(&plan)?;
                    self.pool.complete(plan);
                }
                Ok(Err(e)) => return Err(e),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => bail!(Cuda, "the remap thread is gone"),
            }
        }
        if let Some(plan) = self.pool.take_pending() {
            let after = [record(&self.stream)? as u64, record(&self.xfer)? as u64];
            self.remap_count += 1;
            tracing::debug!(unmap = plan.unmap.len(), map = plan.map.len(), "remap");
            if self.remaps.jobs.send(Job::Run { plan, after }).is_err() {
                bail!(Cuda, "the remap thread is gone");
            }
        }
        Ok(())
    }

    /// Zero every chunk `plan` mapped.
    pub(crate) fn zero_fresh(&mut self, plan: &Remap) -> Result<()> {
        let chunk = self.pool.chunk() as usize;
        for &(a, p, _) in &plan.map {
            let s = &self.states[&self.pool.pooled()[a].state];
            let mut v = s.view(p * chunk..(p + 1) * chunk)?;
            self.stream.memset_zeros(&mut v)?;
        }
        Ok(())
    }

    /// Remaps planned so far.
    pub fn remaps(&self) -> u64 {
        self.remap_count
    }
}
