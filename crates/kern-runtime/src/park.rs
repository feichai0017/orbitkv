//! The host tier shell: checkpoints parked in pinned DRAM and woken
//! straight into the lease of the sequence continuing from them.
//! [`kern_pool::Host`] decides where a checkpoint's bytes go; this is the
//! runtime running the copies.
//!
//! Two streams: every program and every pool copy runs on the compute
//! stream in order; the host tier's copies ([`Runtime::park`] out,
//! [`Runtime::wake`] in) run on a transfer stream, and the compute stream
//! never waits for them. The transfer stream starts each batch of copies
//! after everything the compute stream has enqueued; a parked checkpoint
//! stays held (its pages and slot out of the pool) until its copy has
//! landed, and a woken lease is a [`Waking`] until [`Runtime::awake`]
//! finds its copy landed, so no program can read pages still in flight.
//! Parking is two steps, [`Runtime::room`] then [`Runtime::park`], so a
//! caller parking several checkpoints as one unit can find room for all
//! of them before any byte moves.

use std::sync::Arc;

use cudarc::driver::sys;

use crate::device::{copy_2d, landed, record, wait_then_destroy, Pinned, Space};
use crate::error::bail;
use crate::{Error, Result, Runtime};
use kern_pool::{runs, Checkpoint, Copies, Denied, Host, Kind, Lease, Parked};

/// The host tier's block is handed out in these units.
const HOST_GRAIN: u64 = 1 << 16;

/// A checkpoint with host room found for it and nothing copied yet: what
/// [`Runtime::room`] hands out and [`Runtime::park`] spends. Dropping it
/// frees the room and drops the checkpoint.
pub struct Room {
    cp: Checkpoint,
    parked: Parked,
    plan: Copies<i32, u64>,
}

impl Room {
    /// Give the room back and keep the checkpoint.
    pub fn into_checkpoint(self) -> Checkpoint {
        self.cp
    }
}

/// A lease being woken: its pages are taken, the parked prefix's bytes
/// still on the way in. [`Runtime::awake`] hands the lease out once they
/// have landed; dropping it earlier waits for them first, so the pages
/// never return to the pool with a copy still writing them.
pub struct Waking {
    lease: Option<Lease>,
    event: sys::CUevent,
}

// The event is only ever queried through the runtime that recorded it,
// on whatever thread drives that runtime; the runtime itself moves
// between threads the same way.
unsafe impl Send for Waking {}

impl Drop for Waking {
    fn drop(&mut self) {
        if !self.event.is_null() {
            unsafe {
                sys::cuEventSynchronize(self.event);
                sys::cuEventDestroy_v2(self.event);
            }
        }
    }
}

impl Runtime {
    /// Reserve `bytes` of page-locked host memory for parked checkpoints
    /// (once; ~100 ms per GiB on GB300).
    pub fn reserve_host(&mut self, bytes: u64) -> Result<()> {
        self.ctx.bind_to_thread()?;
        if self.host.is_some() {
            bail!(Api, "the host tier is reserved already");
        }
        let pinned = Pinned::alloc(bytes, self.gpu as i32)?;
        let (page_bytes, slot_bytes, _) = self.host_layout();
        self.host = Some((Arc::new(Host::new(bytes, HOST_GRAIN, page_bytes, slot_bytes)), pinned));
        Ok(())
    }

    /// (bytes used, bytes reserved) of the host tier, when there is one.
    pub fn host_tier(&self) -> Option<(u64, u64)> {
        self.host.as_ref().map(|(h, p)| (h.used(), p.bytes()))
    }

    /// Bytes per host page (every paged state's page back to back) and per
    /// host slot, and each pooled state's offset inside its one.
    fn host_layout(&self) -> (u64, u64, Vec<u64>) {
        let (mut page, mut slot) = (0u64, 0u64);
        let offsets = self
            .pool
            .pooled()
            .iter()
            .map(|a| match a.kind {
                Kind::Page => {
                    page += a.object;
                    page - a.object
                }
                Kind::Slot => {
                    slot += a.object;
                    slot - a.object
                }
            })
            .collect();
        (page, slot, offsets)
    }

    /// Host room for `cp` — its pages not parked already (an earlier turn
    /// of the same session shares them there) and its slot — with nothing
    /// copied yet: the first half of a park, so a caller parking several
    /// checkpoints as one unit can find room for all of them before any
    /// byte moves. `Err(cp)` hands the checkpoint back when the block
    /// cannot hold it. Dropping a [`Room`] frees the room and the
    /// checkpoint with it; [`Room::into_checkpoint`] keeps the checkpoint.
    pub fn room(&mut self, cp: Checkpoint) -> Result<std::result::Result<Room, Checkpoint>> {
        self.ctx.bind_to_thread()?;
        self.poll()?;
        let Some((host, _)) = &self.host else {
            bail!(Api, "no host tier: reserve_host first");
        };
        match host.park(&cp) {
            Ok((parked, plan)) => Ok(Ok(Room { cp, parked, plan })),
            Err(Denied::HostFull) => Ok(Err(cp)),
            Err(d) => Err(Error::Denied(d)),
        }
    }

    /// Copy a checkpoint into the room found for it. The checkpoint is
    /// held until the copies, on the transfer stream, have landed; then
    /// its device pages and slot return.
    pub fn park(&mut self, room: Room) -> Result<Parked> {
        self.ctx.bind_to_thread()?;
        let Room { cp, parked, plan } = room;
        self.transfer(&plan.pages, plan.slot, true)?;
        self.parking.push((cp, record(&self.xfer)?));
        Ok(parked)
    }

    /// A sequence continuing from the first `len` tokens of a parked
    /// checkpoint with room for `tokens`: one lease, sized for the
    /// sequence, with those tokens' pages copied back in from the host
    /// and, when `len` is the whole checkpoint, its state into a fresh
    /// slot (a parked state is usable at its length only; a shorter
    /// `len` is a whole number of pages of a stateless one). A slot-only
    /// checkpoint wakes to a slot-only lease. The copies run on the
    /// transfer stream; [`Runtime::awake`] hands out the lease once they
    /// have landed, its parked pages twinned with their host copies, so
    /// its next checkpoint parks them for free. [`Error::Denied`] as for
    /// [`Runtime::lease`].
    pub fn wake(&mut self, p: &Parked, len: usize, tokens: usize) -> Result<Waking> {
        self.ctx.bind_to_thread()?;
        self.poll()?;
        let Some((host, _)) = &self.host else {
            bail!(Api, "no host tier: reserve_host first");
        };
        let unit = self.pool.unit() as usize;
        if len == 0 || len > p.tokens() || (len != p.tokens() && (p.has_slot() || !len.is_multiple_of(unit))) {
            bail!(Api, "waking {len} tokens of a parked checkpoint of {} ({p:?})", p.tokens());
        }
        let (lease, plan) = match Arc::clone(host).restore(p, &self.pool, len, tokens) {
            Ok(x) => x,
            Err(d) => return self.denied(d),
        };
        let pairs: Vec<(i32, u64)> = plan.pages.iter().map(|&(o, page)| (page, o)).collect();
        self.transfer(&pairs, plan.slot.map(|(o, s)| (s, o)), false)?;
        Ok(Waking { lease: Some(lease), event: record(&self.xfer)? })
    }

    /// Whether a wake's copies have landed. Does not block.
    pub fn landed(&self, w: &Waking) -> Result<bool> {
        self.ctx.bind_to_thread()?;
        landed(w.event)
    }

    /// The lease of a wake whose copies have landed; `Err(w)` while they
    /// are still in flight. Does not block.
    pub fn awake(&self, mut w: Waking) -> Result<std::result::Result<Lease, Waking>> {
        self.ctx.bind_to_thread()?;
        if !landed(w.event)? {
            return Ok(Err(w));
        }
        unsafe { sys::cuEventDestroy_v2(w.event) };
        w.event = std::ptr::null_mut();
        Ok(Ok(w.lease.take().expect("a waking lease")))
    }

    /// (device page, host offset) pages and a (device slot, host offset)
    /// slot between the pooled states and the host block, on the transfer
    /// stream after everything the compute stream has enqueued: one
    /// strided copy per run of consecutive pages per state.
    fn transfer(&self, pages: &[(i32, u64)], slot: Option<(i32, u64)>, to_host: bool) -> Result<()> {
        let Some((_, pinned)) = &self.host else {
            bail!(Api, "no host tier: reserve_host first");
        };
        let base = pinned.ptr();
        let (page_bytes, slot_bytes, offsets) = self.host_layout();
        wait_then_destroy(&self.xfer, record(&self.stream)?)?;
        let stream = self.xfer.cu_stream();
        for (a, ar) in self.pool.pooled().iter().enumerate() {
            let st = &self.states[&ar.state];
            match ar.kind {
                Kind::Page => {
                    for (p, o, n) in runs(pages, page_bytes) {
                        let dev = (st.ptr + p as u64 * ar.object, ar.object, Space::Device);
                        let hst = (base + o + offsets[a], page_bytes, Space::Host);
                        let (dst, src) = if to_host { (hst, dev) } else { (dev, hst) };
                        copy_2d(stream, dst, src, ar.object, n as u64)?;
                    }
                }
                Kind::Slot => {
                    if let Some((s, o)) = slot {
                        let dev = (st.ptr + s as u64 * ar.object, ar.object, Space::Device);
                        let hst = (base + o + offsets[a], slot_bytes, Space::Host);
                        let (dst, src) = if to_host { (hst, dev) } else { (dev, hst) };
                        copy_2d(stream, dst, src, ar.object, 1)?;
                    }
                }
            }
        }
        Ok(())
    }
}
