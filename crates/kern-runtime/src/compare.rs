//! Device-side comparison for the harness: two byte ranges on this device
//! compared element by element, a state's changed blocks against an image
//! of it, and the end-to-end statistics of `logits` rows — each one kernel
//! launch and a few bytes back, so nothing a verdict needs crosses the bus
//! as data. The definitions are the host functions in kern-test
//! (`compare`, `changed_blocks`, `logit_row`); the kernels reproduce every
//! count and maximum exactly and the floating-point sums to rounding. The
//! PTX is bundled like `profile.ptx`: generated from `compare.cu` with
//! CUDA 13.0 (`nvcc --ptx -arch=compute_80 -O3 compare.cu -o compare.ptx`)
//! so the comparison never depends on the host's toolchain.

use std::cell::OnceCell;
use std::ops::Range;
use std::sync::Arc;

use cudarc::driver::{sys, CudaFunction, CudaModule, DeviceSlice, LaunchConfig, PushKernelArg};
use kern_manifest::types::DType;

use crate::device::{alloc, BufView};
use crate::error::bail;
use crate::harness::Scratch;
use crate::{Result, Runtime};

/// One operand of a comparison: bytes the runtime owns or scratch a
/// harness set aside on this device.
#[derive(Clone)]
pub enum At<'a> {
    /// A byte range of a buffer.
    Buffer(&'a str, Range<usize>),
    /// A byte range of a state's allocation.
    State(&'a str, Range<usize>),
    /// A byte range of scratch.
    Scratch(&'a Scratch, Range<usize>),
}

/// Element-wise counts over two ranges, as kern-test's `compare` defines
/// them: elements, bit-different ones, of those the signed zeros and the
/// pairs with a NaN on one side; the largest ulp distance over the
/// `measured` float pairs and the largest |Δ|.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Cmp {
    pub n: u64,
    pub n_diff: u64,
    pub signed_zero: u64,
    pub nan_one_side: u64,
    pub measured: u64,
    pub max_ulp: u64,
    pub max_abs: f64,
}

/// One logits row of A against B: its [`Cmp`], both argmaxes, where A's
/// argmax ranks in B (1-based), how many of A's `top` most likely tokens
/// are among B's, A's top two values and KL(A‖B).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Logit {
    pub cmp: Cmp,
    pub argmax_a: u32,
    pub argmax_b: u32,
    pub rank_in_b: u32,
    pub top: u32,
    pub top1: f64,
    pub top2: f64,
    pub kl: f64,
}

/// The largest `top` the row kernel keeps in shared memory.
pub const TOP_MAX: usize = 64;
const BLOCK: usize = 64;

pub(crate) struct Kernels {
    _module: Arc<CudaModule>,
    cmp: CudaFunction,
    changed: CudaFunction,
    logits: CudaFunction,
    sm_count: u32,
}

fn kind(dt: DType) -> i32 {
    match dt {
        DType::Bf16 => 0,
        DType::F16 => 1,
        DType::F32 => 2,
        DType::Fp8E4m3 => 3,
        DType::Fp8E8m0 => 4,
        DType::I8 => 5,
        DType::U8 => 6,
        DType::I32 => 7,
        DType::U32 => 8,
        DType::I64 => 9,
        DType::U64 => 10,
    }
}

fn cmp_of(v: &[u64], n: u64) -> Cmp {
    Cmp {
        n,
        n_diff: v[0],
        signed_zero: v[1],
        nan_one_side: v[2],
        measured: v[3],
        max_ulp: v[4],
        max_abs: f64::from_bits(v[5]),
    }
}

/// Byte ranges of the 64-byte blocks whose bit is set, adjacent blocks
/// merged, the last clipped to `bytes`.
/// The set bits as merged block ranges; a word of zeros (32 unchanged
/// blocks, the common case over a large buffer) costs one test.
fn ranges_of(bits: &[u32], bytes: usize) -> Vec<Range<usize>> {
    let mut out: Vec<Range<usize>> = Vec::new();
    for (w, &word) in bits.iter().enumerate().filter(|(_, w)| **w != 0) {
        let mut rest = word;
        while rest != 0 {
            let i = w * 32 + rest.trailing_zeros() as usize;
            rest &= rest - 1;
            let (lo, hi) = (i * BLOCK, ((i + 1) * BLOCK).min(bytes));
            match out.last_mut() {
                Some(r) if r.end == lo => r.end = hi,
                _ => out.push(lo..hi),
            }
        }
    }
    out
}

fn u64s(bytes: &[u8]) -> Vec<u64> {
    bytes.as_chunks::<8>().0.iter().map(|c| u64::from_le_bytes(*c)).collect()
}

impl Runtime {
    fn compare_kernels(&self) -> Result<&Kernels> {
        if let Some(k) = self.compare.get() {
            return Ok(k);
        }
        self.ctx.bind_to_thread()?;
        let module = self.ctx.load_module(cudarc::nvrtc::Ptx::from_src(include_str!("compare.ptx")))?;
        let k = Kernels {
            cmp: module.load_function("cmp_reduce")?,
            changed: module.load_function("changed_blocks")?,
            logits: module.load_function("logit_rows")?,
            sm_count: self.ctx.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)? as u32,
            _module: module,
        };
        Ok(self.compare.get_or_init(|| k))
    }

    fn view_at(&self, at: At) -> Result<BufView> {
        match at {
            At::Buffer(name, r) => {
                let Some(b) = self.buffers.get(name) else {
                    bail!(Api, "no buffer `{name}`");
                };
                if r.end as u64 > b.bytes {
                    bail!(Api, "buffer `{name}`: bytes {r:?} exceed allocation {}", b.bytes);
                }
                b.view(r)
            }
            At::State(name, r) => {
                self.whole_state(name)?;
                let Some(s) = self.states.get(name) else {
                    bail!(Api, "no state `{name}`");
                };
                if r.end as u64 > s.bytes {
                    bail!(Api, "state `{name}`: range [{}, {}) exceeds allocation {}", r.start, r.end, s.bytes);
                }
                s.view(r)
            }
            At::Scratch(s, r) => s.0.view(r),
        }
    }

    fn pair(&self, a: At, b: At) -> Result<(BufView, BufView)> {
        let (x, y) = (self.view_at(a)?, self.view_at(b)?);
        if x.len() != y.len() {
            bail!(Api, "comparing {} bytes against {}", x.len(), y.len());
        }
        Ok((x, y))
    }

    /// `a` against `b` element by element as `dtype` (synchronous).
    pub fn compare(&self, dtype: DType, a: At, b: At) -> Result<Cmp> {
        let k = self.compare_kernels()?;
        let (x, y) = self.pair(a, b)?;
        let n = (x.len() / dtype.bytes() as usize) as u64;
        let out = alloc(&self.stream, 48)?;
        let cfg = LaunchConfig { grid_dim: (k.sm_count * 8, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.stream
                .launch_builder(&k.cmp)
                .arg(&x.ptr())
                .arg(&y.ptr())
                .arg(&n)
                .arg(&kind(dtype))
                .arg(&out.ptr)
                .launch(cfg)
        }?;
        let v = u64s(&self.stream.clone_dtoh(&out.view(0..48)?)?);
        Ok(cmp_of(&v, n))
    }

    /// The 64-byte blocks where `a` and `b` differ, as merged byte ranges
    /// (synchronous): how a state's write-set is found against an image
    /// of it without reading either.
    pub fn changed(&self, a: At, b: At) -> Result<Vec<Range<usize>>> {
        let k = self.compare_kernels()?;
        let (x, y) = self.pair(a, b)?;
        let bytes = x.len();
        let words = bytes.div_ceil(BLOCK).div_ceil(32).max(1);
        let bits = alloc(&self.stream, (words * 4) as u64)?;
        let blocks = bytes.div_ceil(BLOCK) as u32;
        let grid = blocks.div_ceil(256).clamp(1, k.sm_count * 8);
        let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.stream
                .launch_builder(&k.changed)
                .arg(&x.ptr())
                .arg(&y.ptr())
                .arg(&(bytes as u64))
                .arg(&bits.ptr)
                .launch(cfg)
        }?;
        let v: Vec<u32> = self
            .stream
            .clone_dtoh(&bits.view(0..words * 4)?)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        Ok(ranges_of(&v, bytes))
    }

    /// Every row of `cols` elements of `a` against the same row of `b`
    /// (synchronous); `top` at most [`TOP_MAX`].
    pub fn logits(&self, dtype: DType, cols: usize, top: usize, a: At, b: At) -> Result<Vec<Logit>> {
        let k = self.compare_kernels()?;
        let (x, y) = self.pair(a, b)?;
        let row = cols * dtype.bytes() as usize;
        if cols == 0 || top > TOP_MAX || x.len() % row != 0 {
            bail!(Api, "logits rows of {cols} elements, top {top}, over {} bytes", x.len());
        }
        let rows = x.len() / row;
        let out = alloc(&self.stream, (rows * 96) as u64)?;
        let cfg = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.stream
                .launch_builder(&k.logits)
                .arg(&x.ptr())
                .arg(&y.ptr())
                .arg(&(cols as u32))
                .arg(&kind(dtype))
                .arg(&(top as i32))
                .arg(&out.ptr)
                .launch(cfg)
        }?;
        let v = u64s(&self.stream.clone_dtoh(&out.view(0..rows * 96)?)?);
        Ok(v.as_chunks::<12>()
            .0
            .iter()
            .map(|r| Logit {
                cmp: cmp_of(r, cols as u64),
                argmax_a: r[9] as u32,
                argmax_b: r[10] as u32,
                rank_in_b: (r[11] >> 32) as u32,
                top: r[11] as u32,
                top1: f64::from_bits(r[6]),
                top2: f64::from_bits(r[7]),
                kl: f64::from_bits(r[8]),
            })
            .collect())
    }
}

pub(crate) type CompareCell = OnceCell<Kernels>;
