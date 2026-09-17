//! The harness surface: whole-buffer and whole-state access, a
//! program's call list replayed in part, and event-bracketed timing.
//! Nothing here is on the serving path and every call synchronizes; the
//! harness driving it is `kern test` (`docs/test.md`).
//!
//! Whole-state access is for the layout the runtime loaded with: once a
//! remap has moved chunks, a pooled state has holes and the bytes at an
//! offset belong to no page.

use std::collections::BTreeMap;

use cudarc::driver::sys;

use crate::compile::Dense;
use crate::device::{alloc_uninit, DeviceBuf, Events};
use crate::error::{bail, cuda_check};
use crate::{Error, Result, Runtime};

/// Device bytes a harness set aside: a buffer's live prefix or a state's
/// image, copied device to device and handed back to any runtime on the
/// same device. Opaque to the host; what a harness keeps instead of a
/// `Vec<u8>` so a snapshot or a state sync never crosses the bus.
pub struct Scratch(pub(crate) DeviceBuf);

impl Scratch {
    pub fn bytes(&self) -> usize {
        self.0.bytes as usize
    }
}

impl Runtime {
    /// Whole-state access is for the layout the runtime loaded with; once
    /// a remap has moved chunks, a pooled state has holes.
    pub(crate) fn whole_state(&self, name: &str) -> Result<()> {
        let pooled = self.pool.pooled().iter().any(|a| a.state == name);
        if pooled && self.pool.remapped() {
            bail!(Api, "state `{name}` has been remapped since load; whole-state access is for the initial layout");
        }
        Ok(())
    }

    pub fn call_count(&self, program: &str) -> Result<usize> {
        match self.programs.get(program) {
            Some(p) => Ok(p.call_ranges.len()),
            None => bail!(Api, "no program `{program}`"),
        }
    }

    /// Whole allocation of any buffer, regardless of class.
    pub fn read_buffer(&self, name: &str) -> Result<Vec<u8>> {
        self.ctx.bind_to_thread()?;
        let Some(b) = self.buffers.get(name) else {
            bail!(Api, "no buffer `{name}`");
        };
        Ok(self.stream.clone_dtoh(b)?)
    }

    /// The first `bytes` of any buffer (the live prefix at a var value
    /// below the allocation bound).
    pub fn read_buffer_prefix(&self, name: &str, bytes: usize) -> Result<Vec<u8>> {
        self.ctx.bind_to_thread()?;
        let Some(b) = self.buffers.get(name) else {
            bail!(Api, "no buffer `{name}`");
        };
        if bytes as u64 > b.bytes {
            bail!(Api, "buffer `{name}`: prefix {bytes} exceeds allocation {}", b.bytes);
        }
        let view = b.view(0..bytes)?;
        Ok(self.stream.clone_dtoh(&view)?)
    }

    /// Overwrite a prefix of any buffer, regardless of class (synchronous).
    pub fn write_buffer(&mut self, name: &str, data: &[u8]) -> Result<()> {
        self.ctx.bind_to_thread()?;
        let Some(b) = self.buffers.get_mut(name) else {
            bail!(Api, "no buffer `{name}`");
        };
        if data.len() as u64 > b.bytes {
            bail!(Api, "buffer `{name}`: got {} bytes, buffer is {}", data.len(), b.bytes);
        }
        let mut view = b.view(0..data.len())?;
        self.stream.memcpy_htod(data, &mut view)?;
        self.stream.synchronize()?;
        Ok(())
    }

    /// Overwrite `data.len()` bytes of a state starting at `offset`
    /// (synchronous). States are opaque to the runtime; this is how a
    /// harness puts a state back to a snapshot before replaying a cut.
    pub fn write_state_at(&mut self, name: &str, offset: usize, data: &[u8]) -> Result<()> {
        self.ctx.bind_to_thread()?;
        self.whole_state(name)?;
        let Some(s) = self.states.get_mut(name) else {
            bail!(Api, "no state `{name}`");
        };
        let end = offset + data.len();
        if end as u64 > s.bytes {
            bail!(Api, "state `{name}`: write [{offset}, {end}) exceeds allocation {}", s.bytes);
        }
        let mut view = s.view(offset..end)?;
        self.stream.memcpy_htod(data, &mut view)?;
        self.stream.synchronize()?;
        Ok(())
    }

    /// Zero every state (synchronous): a fresh sequence from position 0,
    /// the way the runtime was loaded.
    pub fn zero_states(&mut self) -> Result<()> {
        self.ctx.bind_to_thread()?;
        for name in self.manifest.states.keys() {
            self.whole_state(name)?;
        }
        for s in self.states.values_mut() {
            let mut v = s.view(0..s.bytes as usize)?;
            self.stream.memset_zeros(&mut v)?;
        }
        self.stream.synchronize()?;
        Ok(())
    }

    /// `len` bytes of a state from `offset` (synchronous).
    pub fn read_state_at(&self, name: &str, offset: usize, len: usize) -> Result<Vec<u8>> {
        self.ctx.bind_to_thread()?;
        self.whole_state(name)?;
        let Some(s) = self.states.get(name) else {
            bail!(Api, "no state `{name}`");
        };
        if (offset + len) as u64 > s.bytes {
            bail!(Api, "state `{name}`: read [{offset}, {}) exceeds allocation {}", offset + len, s.bytes);
        }
        Ok(self.stream.clone_dtoh(&s.view(offset..offset + len)?)?)
    }

    /// Bytes allocated for a state (its initial layout).
    pub fn state_bytes(&self, name: &str) -> Result<usize> {
        match self.states.get(name) {
            Some(s) => Ok(s.bytes as usize),
            None => bail!(Api, "no state `{name}`"),
        }
    }

    /// Device scratch of `bytes`, uninitialized: filled before it is read.
    fn scratch(&self, bytes: usize) -> Result<Scratch> {
        self.ctx.bind_to_thread()?;
        Ok(Scratch(alloc_uninit(&self.stream, bytes as u64)?))
    }

    /// Bytes `at` of a buffer into new scratch, device to device (synchronous).
    pub fn save_buffer(&self, name: &str, at: std::ops::Range<usize>) -> Result<Scratch> {
        let Some(b) = self.buffers.get(name) else {
            bail!(Api, "no buffer `{name}`");
        };
        if at.end as u64 > b.bytes {
            bail!(Api, "buffer `{name}`: bytes {at:?} exceed allocation {}", b.bytes);
        }
        let n = at.len();
        let into = self.scratch(n)?;
        self.dtod(into.0.view(0..n)?.ptr(), b.view(at)?.ptr(), n)?;
        Ok(into)
    }

    /// The first `at.len()` bytes of scratch into bytes `at` of a buffer, device to device (synchronous).
    pub fn load_buffer(&mut self, name: &str, at: std::ops::Range<usize>, from: &Scratch) -> Result<()> {
        let Some(b) = self.buffers.get(name) else {
            bail!(Api, "no buffer `{name}`");
        };
        if at.end as u64 > b.bytes {
            bail!(Api, "buffer `{name}`: bytes {at:?} exceed allocation {}", b.bytes);
        }
        let n = at.len();
        self.dtod(b.view(at)?.ptr(), from.0.view(0..n)?.ptr(), n)
    }

    /// A state's whole allocation into new scratch, device to device (synchronous).
    pub fn save_state(&self, name: &str) -> Result<Scratch> {
        self.whole_state(name)?;
        let Some(s) = self.states.get(name) else {
            bail!(Api, "no state `{name}`");
        };
        let n = s.bytes as usize;
        let into = self.scratch(n)?;
        self.dtod(into.0.view(0..n)?.ptr(), s.view(0..n)?.ptr(), n)?;
        Ok(into)
    }

    /// Scratch over a state's whole allocation, device to device (synchronous).
    pub fn load_state(&mut self, name: &str, from: &Scratch) -> Result<()> {
        self.whole_state(name)?;
        let Some(s) = self.states.get(name) else {
            bail!(Api, "no state `{name}`");
        };
        let n = s.bytes as usize;
        self.dtod(s.view(0..n)?.ptr(), from.0.view(0..n)?.ptr(), n)
    }

    /// Bytes `at` of scratch, on the host.
    pub fn read_scratch(&self, from: &Scratch, at: std::ops::Range<usize>) -> Result<Vec<u8>> {
        self.ctx.bind_to_thread()?;
        Ok(self.stream.clone_dtoh(&from.0.view(at)?)?)
    }

    fn dtod(&self, dst: u64, src: u64, bytes: usize) -> Result<()> {
        self.ctx.bind_to_thread()?;
        cuda_check(
            unsafe { sys::cuMemcpyDtoDAsync_v2(dst, src, bytes, self.stream.cu_stream()) },
            "cuMemcpyDtoDAsync",
        )?;
        self.stream.synchronize()?;
        Ok(())
    }

    /// Whole allocation of a state.
    pub fn read_state(&self, name: &str) -> Result<Vec<u8>> {
        self.ctx.bind_to_thread()?;
        self.whole_state(name)?;
        let Some(s) = self.states.get(name) else {
            bail!(Api, "no state `{name}`");
        };
        Ok(self.stream.clone_dtoh(&s.view(0..s.bytes as usize)?)?)
    }

    /// Execute calls `[lo, hi)` of a program eagerly, then synchronize.
    pub fn run_range(&self, program: &str, vars: &BTreeMap<String, u64>, lo: usize, hi: usize) -> Result<()> {
        let Some(prog) = self.programs.get(program) else {
            bail!(Api, "no program `{program}`");
        };
        let n = prog.call_ranges.len();
        if lo > hi || hi > n {
            bail!(Api, "program `{program}`: call range [{lo}, {hi}) outside 0..{n}");
        }
        self.require_peers()?;
        let vars = Dense::check(&self.manifest, vars, &prog.vars)?;
        self.ctx.bind_to_thread()?;
        if lo < hi {
            let (l0, _) = prog.call_ranges[lo];
            let (_, l1) = prog.call_ranges[hi - 1];
            for l in &prog.launches[l0..l1] {
                self.launch(l, &vars).map_err(|e| Error::Call { context: l.ctx.clone(), source: Box::new(e) })?;
            }
        }
        self.stream.synchronize()?;
        Ok(())
    }

    /// Per-call GPU time in ms (eager, event-bracketed) for calls
    /// `[lo, hi)`, minimum over `iters` replays of just that range, so a
    /// cut can be timed without the rest of the program. Note this
    /// attributes launch gaps to the call that follows them.
    pub fn time_range(
        &self,
        program: &str,
        vars: &BTreeMap<String, u64>,
        lo: usize,
        hi: usize,
        iters: usize,
    ) -> Result<Vec<f32>> {
        let Some(prog) = self.programs.get(program) else {
            bail!(Api, "no program `{program}`");
        };
        if lo > hi || hi > prog.call_ranges.len() {
            bail!(Api, "program `{program}`: call range [{lo}, {hi}) outside 0..{}", prog.call_ranges.len());
        }
        self.require_peers()?;
        let vars = Dense::check(&self.manifest, vars, &prog.vars)?;
        self.ctx.bind_to_thread()?;
        let n = hi - lo;
        let events = Events::new(n + 1)?;
        let mut best = vec![f32::INFINITY; n];
        for _ in 0..iters.max(1) {
            events.record(0, &self.stream)?;
            for (di, &(l0, l1)) in prog.call_ranges[lo..hi].iter().enumerate() {
                for l in &prog.launches[l0..l1] {
                    self.launch(l, &vars).map_err(|e| Error::Call { context: l.ctx.clone(), source: Box::new(e) })?;
                }
                events.record(di + 1, &self.stream)?;
            }
            self.stream.synchronize()?;
            for (di, b) in best.iter_mut().enumerate() {
                *b = b.min(events.elapsed_ms(di, di + 1)?);
            }
        }
        Ok(best)
    }

    /// Median wall time per replay of a captured program, in ms, over
    /// `iters` back-to-back graph launches.
    pub fn time_captured(&self, program: &str, vars: &BTreeMap<String, u64>, iters: usize) -> Result<f32> {
        let exec = self.graph(program, vars)?;
        self.ctx.bind_to_thread()?;
        let iters = iters.max(1);
        let events = Events::new(iters + 1)?;
        events.record(0, &self.stream)?;
        for i in 0..iters {
            cuda_check(unsafe { sys::cuGraphLaunch(exec, self.stream.cu_stream()) }, "cuGraphLaunch")?;
            events.record(i + 1, &self.stream)?;
        }
        self.stream.synchronize()?;
        let mut ts: Vec<f32> = (0..iters).map(|i| events.elapsed_ms(i, i + 1)).collect::<Result<_>>()?;
        ts.sort_by(|a, b| a.total_cmp(b));
        Ok(ts[iters / 2])
    }
}
