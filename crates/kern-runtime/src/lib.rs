//! Thin verifier-driven executor for kern manifests.
//!
//! The runtime knows nothing about models. It loads a verified manifest,
//! resolves each declared kernel against the cubins in a directory, allocates
//! every buffer/state, assembles weight buffers out of a checkpoint's
//! tensors (a [`Tensors`]: safetensors blobs, or a weight cache's buckets
//! mapped from another process), and replays the program's call list. The only kernels it understands
//! natively are `extern:` ops (currently `extern:cublaslt_bf16_tn`).
//!
//! Names stop at load time: device pointers are static once buffers, states
//! and scratch are allocated, so [`Runtime::load`] lowers every program into
//! a flat launch list (see `compile`) whose slots are finished values or
//! var-indexed expressions. The name-keyed maps that remain exist only on
//! the caller API surface (`write_input("token_ids")`, `run("decode")`);
//! the execution path performs no name lookups.
//!
//! Same-name Triton kernels ship multiple constexpr instances with different
//! ABIs across modules; resolution picks the instance whose
//! `cuFuncGetParamInfo` layout matches the manifest's declared params — the
//! phase-2 ABI check doubles as instance selection.
//!
//! Every entry point binds the runtime's CUDA context to the calling
//! thread first, so one thread may drive several runtimes (a tray) and a
//! runtime may be loaded on one thread and driven from another.
//!

mod compare;
mod compile;
mod cubin;
mod cublas;
mod device;
mod error;
mod exec;
mod harness;
mod host_weights;
mod lease;
mod load;
mod park;
mod peers;
pub mod profile;
mod weights;

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use cudarc::cublaslt::CudaBlasLT;
use cudarc::driver::{sys, CudaContext, CudaStream, HostSlice, PinnedHostSlice};
use kern_manifest::types::{BufferKind, Manifest, Provision, State};
use kern_manifest::Verified;

pub use compare::{At, Cmp, Logit, TOP_MAX};
use compile::{CompiledProgram, Dense};
use cublas::Blas;
use device::{alloc, DeviceBuf, Pinned};
pub use device::{device_uuid, Mapped, PeerHandle};
use error::{bail, cuda_check};
pub use error::{Error, Result};
pub use harness::Scratch;
pub use host_weights::HostWeights;
use kern_pool::{page_unit, row_tokens, Checkpoint, Host, Pool};
use lease::Remaps;
pub use load::Resident;
pub use park::{Room, Waking};
use peers::PeerSlot;
pub use weights::{dtype_named, Blob, Safetensors, Tensor, Tensors};

/// The CUDA API this binary binds (`13000` is 13.0): fixed by the `cudarc`
/// feature at build time, so a driver older than it fails to load the
/// libraries rather than at a random symbol later.
pub const CUDA_API: u32 = sys::CUDA_VERSION;

/// What the caller will hold in the pooled states at once: pages for
/// `tokens` tokens of every paged state and `seqs` sequences' slots of every
/// per-sequence state (plus the null slot and a spare). A number the caller
/// knows — its batch, its serving bound — never the manifest's var bounds,
/// which say what a step may address, not how many sequences live.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Capacity {
    /// Tokens of paged state; `None` takes what the device has left.
    pub tokens: Option<u64>,
    pub seqs: u64,
}

/// This rank's place in every group the manifest's `topology` declares.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Topology {
    pub groups: BTreeMap<String, GroupRank>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupRank {
    pub index: u64,
    pub size: u64,
}

impl Topology {
    /// A single group, e.g. `Topology::one("ep", 2, 4)`.
    pub fn one(group: &str, index: u64, size: u64) -> Topology {
        Topology { groups: BTreeMap::from([(group.to_string(), GroupRank { index, size })]) }
    }
}

pub struct Runtime {
    /// The manifest as loaded; verified, so nothing here checks it again.
    pub manifest: Verified,
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    /// The host tier's copies run here, off the compute stream.
    xfer: Arc<CudaStream>,
    /// The host tier: its accounting and its pinned block, once reserved.
    host: Option<(Arc<Host>, Pinned)>,
    /// Checkpoints whose park copies are still in flight: held here so
    /// their pages and slots stay out of the pool until the copy lands.
    parking: Vec<(Checkpoint, sys::CUevent)>,
    blt: CudaBlasLT,
    /// cuBLAS handle (with its own workspace) for the f32-result GEMM built-in.
    blas: Blas,
    /// What every `index_into` domain resolves against: the token slots a
    /// paged state's arena spans, the sequence slots a per-sequence one's.
    provision: Provision,
    /// Owner of the states' token slots: hands them out as leases.
    pool: Arc<Pool>,
    /// The remap thread: runs the pool's plans off the serving thread.
    remaps: Remaps,
    remap_count: u64,
    /// Name-keyed because names are the caller API (`write_input`,
    /// `read_output`, weight binding); execution never looks these up —
    /// their device pointers are baked into `programs`.
    buffers: BTreeMap<String, DeviceBuf>,
    /// No kernels may read mapped host bytes before checkpoint binding completes.
    host_weights_ready: bool,
    /// Weight buffers holding their checkpoint bytes: taken from a
    /// [`Resident`] at load, or copied by `load_weights`.
    filled: BTreeSet<String>,
    /// Bytes of them taken from a [`Resident`].
    kept: u64,
    states: BTreeMap<String, DeviceBuf>,
    /// Persistent pinned staging, one per input buffer: H2D from pageable
    /// memory degrades to a synchronous driver-staged copy (tens of µs per
    /// call); through page-locked staging it is a true async DMA. The pinned
    /// slice's event guards reuse across steps.
    staging: BTreeMap<String, PinnedHostSlice<u8>>,
    /// Programs lowered to flat launch lists at load.
    programs: BTreeMap<String, CompiledProgram>,
    /// Owners of the impl-private scratch allocations whose pointers are
    /// baked into `programs`.
    #[allow(dead_code)]
    scratch: Vec<DeviceBuf>,
    /// Per kernel: the module each impl step resolved to (introspection).
    resolution: Vec<(String, Vec<String>)>,
    n_modules: usize,
    /// (program, dense var values) -> instantiated CUDA graph. Grid dims
    /// and scalar args are baked in at capture, so one program holds one
    /// graph per var assignment it was captured at (a batched decode keeps
    /// one per batch bucket).
    graphs: BTreeMap<(String, Dense), sys::CUgraphExec>,
    /// Debug override: [`Runtime::issue`] launches every program eagerly,
    /// whatever its manifest says.
    eager: bool,
    /// CUDA device ordinal, for the virtual-memory calls.
    gpu: usize,
    /// This rank's index per group (empty without a topology).
    ranks: BTreeMap<String, u64>,
    /// Every `peer` buffer, by name.
    peers: BTreeMap<String, PeerSlot>,
    /// Peer mappings kept alive as long as the addresses in `peers`.
    #[allow(dead_code)]
    imports: Vec<DeviceBuf>,
    /// The harness's comparison kernels, loaded on first use.
    compare: compare::CompareCell,
}

impl Drop for Runtime {
    fn drop(&mut self) {
        let _ = self.ctx.bind_to_thread();
        for exec in self.graphs.values() {
            unsafe { sys::cuGraphExecDestroy(*exec) };
        }
        // The thread owns the arenas: nothing on either stream may still
        // touch them when it unmaps.
        let _ = self.stream.synchronize();
        let _ = self.xfer.synchronize();
        for (_, ev) in self.parking.drain(..) {
            unsafe { sys::cuEventDestroy_v2(ev) };
        }
        self.remaps.stop();
    }
}

impl Runtime {
    /// The CUDA device ordinal this runtime is on.
    pub fn gpu(&self) -> usize {
        self.gpu
    }

    pub fn module_count(&self) -> usize {
        self.n_modules
    }

    /// (name, class, allocated bytes) for every buffer.
    pub fn buffer_sizes(&self) -> Vec<(&str, BufferKind, u64)> {
        self.manifest.buffers.iter().map(|(n, b)| (n.as_str(), b.kind, self.buffers[n].bytes)).collect()
    }

    /// (name, declaration, allocated bytes) for every state.
    pub fn state_sizes(&self) -> Vec<(&str, &State, u64)> {
        self.manifest.states.iter().map(|(n, s)| (n.as_str(), s, self.states[n].bytes)).collect()
    }

    /// Per kernel: the module each impl step resolved to, in step order.
    pub fn op_resolution(&self) -> Vec<(String, Vec<String>)> {
        self.resolution.clone()
    }

    /// Check `data` (a prefix of buffer `name`) against the buffer's declared
    /// domain, if any, at the given var values. Symbol-dependent bounds
    /// need `vars`; pass the values the next run will use.
    fn check_domain(&self, name: &str, data: &[u8], vars: &BTreeMap<String, u64>) -> Result<()> {
        let Some(b) = self.manifest.buffers.get(name) else {
            bail!(Api, "no buffer `{name}`");
        };
        let Some(d) = &b.domain else { return Ok(()) };
        let r = d
            .resolve(&self.manifest, vars, &self.provision)
            .map_err(|e| Error::Domain(format!("buffer `{name}`: {e}")))?;
        let vals = kern_manifest::values::to_f64(b.dtype, data);
        let fmt_bound = |v: Option<f64>| v.map_or("∞".to_string(), |x| format!("{x}"));
        for (i, &v) in vals.iter().enumerate() {
            if !r.contains(v) {
                bail!(Domain, "buffer `{name}`[{i}] = {v} outside declared [{}, {}]", fmt_bound(r.lo), fmt_bound(r.hi));
            }
            if r.monotone && i > 0 && v < vals[i - 1] {
                bail!(Domain, "buffer `{name}` is declared monotone but [{i}] = {v} < [{}] = {}", i - 1, vals[i - 1]);
            }
        }
        Ok(())
    }

    /// Write an input buffer. The domain check needs the var values the
    /// next run will use; `write_input` checks against var upper bounds
    /// (the loosest valid reading), `write_input_at` against exact values.
    pub fn write_input(&mut self, name: &str, data: &[u8]) -> Result<()> {
        let vars_max: BTreeMap<_, _> = self.manifest.vars.iter().map(|(s, v)| (s.clone(), v.max)).collect();
        self.write_input_at(name, data, &vars_max)
    }

    pub fn write_input_at(&mut self, name: &str, data: &[u8], vars: &BTreeMap<String, u64>) -> Result<()> {
        self.ctx.bind_to_thread()?;
        let Some(b) = self.manifest.buffers.get(name) else {
            bail!(Api, "no buffer `{name}`");
        };
        if b.kind != BufferKind::Input {
            bail!(Api, "buffer `{name}` is {}, not input", b.kind);
        }
        if data.len() as u64 > self.buffers[name].bytes {
            bail!(Api, "input `{name}`: got {} bytes, buffer is {}", data.len(), self.buffers[name].bytes);
        }
        self.check_domain(name, data, vars)?;
        let dst = self.buffers.get_mut(name).unwrap();
        let pinned = self.staging.get_mut(name).unwrap();
        // The staging waits on its own event before the overwrite and the
        // guard records it after the copy; only the bytes written go over,
        // a page table sized for 256 sequences at 1M tokens is 8 MiB.
        pinned.as_mut_slice()?[..data.len()].copy_from_slice(data);
        let (src, _record) = unsafe { pinned.stream_synced_slice(&self.stream) };
        unsafe { cudarc::driver::result::memcpy_htod_async(dst.ptr, &src[..data.len()], self.stream.cu_stream()) }?;
        Ok(())
    }

    pub fn read_output(&self, name: &str) -> Result<Vec<u8>> {
        self.ctx.bind_to_thread()?;
        let Some(b) = self.manifest.buffers.get(name) else {
            bail!(Api, "no buffer `{name}`");
        };
        if b.kind != BufferKind::Output {
            bail!(Api, "buffer `{name}` is {}, not output", b.kind);
        }
        Ok(self.stream.clone_dtoh(&self.buffers[name])?)
    }

    /// Whether the manifest has per-sequence states (sequence slots at all).
    pub fn has_seq_state(&self) -> bool {
        self.pool.has_slots()
    }

    /// Sequence slots that exist now in every per-sequence state, slot 0
    /// among them (0 without one).
    pub fn seq_slots(&self) -> usize {
        self.pool.slots()
    }

    /// Sequence slots held by a lease or a checkpoint.
    pub fn seq_slots_used(&self) -> usize {
        self.pool.slots_used()
    }

    /// The line-table inputs (`index_into` a per-sequence state, shaped
    /// `[lines, seqs]` or `[lines, seqs, w]`), e.g. a hybrid model's
    /// `gdn.line_index`.
    pub fn seq_tables(&self) -> impl Iterator<Item = &str> {
        self.pool.seq_tables()
    }

    /// Pages the states hold in total.
    pub fn pages_total(&self) -> usize {
        self.pool.total()
    }

    /// Pages currently leased.
    pub fn pages_used(&self) -> usize {
        self.pool.used()
    }

    /// Longest sequence one page-table row can address (whole pages).
    pub fn max_seq_tokens(&self) -> usize {
        self.pool.max_seq_tokens()
    }

    /// The page-table inputs (`index_into` a state, constant row width),
    /// e.g. `block_table` and a speculative manifest's `draft_block_table`.
    pub fn page_tables(&self) -> impl Iterator<Item = &str> {
        self.pool.tables()
    }

    /// Token slots the paged states hold now (whole pages).
    pub fn capacity(&self) -> u64 {
        self.pool.total() as u64 * self.pool.unit()
    }

    /// The bounds `index_into` domains resolve against.
    pub fn provision(&self) -> Provision {
        self.provision
    }

    /// Page unit in tokens (1 if no state is paged).
    pub fn page(&self) -> u64 {
        self.pool.unit()
    }

    /// Wait for everything enqueued on both streams.
    pub fn synchronize(&self) -> Result<()> {
        self.ctx.bind_to_thread()?;
        self.stream.synchronize()?;
        self.xfer.synchronize()?;
        Ok(())
    }
}

/// Device memory left untouched when the states are fitted to the device:
/// the driver's own allocations after load (captured graphs, module
/// lazy-loading, cuBLASLt algorithm state) and a margin for a neighbour.
pub(crate) const HEADROOM: u64 = 1 << 30;

/// Tokens one sequence of `m` can reach — the narrowest page table's row,
/// in whole pages — or `None` when nothing is paged per token. What a
/// single-sequence caller wants as its state capacity.
pub fn seq_capacity(m: &Manifest) -> Option<u64> {
    row_tokens(m, page_unit(m))
}
