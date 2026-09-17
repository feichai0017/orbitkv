//! Load-time lowering. Everything name-shaped in a program dies here:
//! op launches resolve to CUDA functions, buffer/state/scratch references
//! to device addresses (static once allocated), var names to indices into
//! the dense vars. What execution replays is a flat launch list whose slots
//! are either finished values or var-indexed expressions — no name lookups,
//! no wiring, and no panics left for the hot path.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use cudarc::driver::{result as cu, sys, CudaStream};
use kern_manifest::types::{
    Arg, Call, Dim, Expr, FieldSrc, LaunchArg, Manifest, Op, Pack, ParamType, TensorMap, TmaDType, Var,
};
use std::os::raw::c_void;

use crate::cubin::{param_sizes, LoadedModule, MulticastScan};
use crate::device::{alloc, DeviceBuf};
use crate::error::{bail, cuda_check, Error, Result};

/// A scalar expression with var names resolved to indices into the dense
/// vars (manifest var order). Division by zero is rejected at compile
/// time; overflow stays a runtime error (value-dependent).
#[derive(Clone)]
pub(crate) enum CExpr {
    Const(u64),
    Var(usize),
    CeilDiv(Box<CExpr>, u64),
    Mul(Box<CExpr>, u64),
}

/// A program's var values in manifest var order, the index space every
/// compiled expression reads. Built by [`Dense::check`] and nowhere else:
/// each value is inside its declared bound, and a var the program does
/// not read is `Var::MIN`, so two calls that differ only in vars the
/// program never reads are one `Dense` and one graph key.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Dense(Vec<u64>);

impl Dense {
    /// The caller's values for the vars `used` marks, checked against
    /// their declared bounds; the rest at the minimum.
    pub(crate) fn check(m: &Manifest, vars: &BTreeMap<String, u64>, used: &[bool]) -> Result<Dense> {
        m.vars
            .iter()
            .zip(used)
            .map(|((var, decl), &used)| {
                if !used {
                    return Ok(Var::MIN);
                }
                let Some(&v) = vars.get(var) else {
                    bail!(Api, "var `{var}` not provided");
                };
                if v < Var::MIN || v > decl.max {
                    bail!(Api, "var `{var}` = {v} outside declared [{}, {}]", Var::MIN, decl.max);
                }
                Ok(v)
            })
            .collect::<Result<Vec<_>>>()
            .map(Dense)
    }

    /// `var=value` in manifest var order, for error messages.
    pub(crate) fn describe(&self, m: &Manifest) -> String {
        m.vars.keys().zip(&self.0).map(|(s, v)| format!("{s}={v}")).collect::<Vec<_>>().join(", ")
    }
}

impl CExpr {
    pub(crate) fn eval(&self, vars: &Dense) -> Result<u64> {
        match self {
            CExpr::Const(c) => Ok(*c),
            CExpr::Var(i) => Ok(vars.0[*i]),
            CExpr::CeilDiv(e, c) => Ok(e.eval(vars)?.checked_add(c - 1).ok_or_else(overflow)? / c),
            CExpr::Mul(e, c) => e.eval(vars)?.checked_mul(*c).ok_or_else(overflow),
        }
    }

    /// Mark the vars this expression reads.
    fn mark(&self, used: &mut [bool]) {
        match self {
            CExpr::Const(_) => {}
            CExpr::Var(i) => used[*i] = true,
            CExpr::CeilDiv(e, _) | CExpr::Mul(e, _) => e.mark(used),
        }
    }
}

fn overflow() -> Error {
    Error::Manifest("expression eval: arithmetic overflow".into())
}

/// A launch value: the low bytes of `val` are what the param slot receives;
/// `bytes` is the remaining buffer size for pointer args (0 for scalars),
/// used by the extern gemm.
#[derive(Clone, Copy)]
pub(crate) struct RVal {
    pub(crate) val: u64,
    pub(crate) bytes: u64,
}

/// One launch parameter, lowered.
#[derive(Clone)]
pub(crate) enum Slot {
    /// Known at load time: a device pointer (base + offset) or a literal.
    Const(RVal),
    /// Var-dependent scalar, evaluated against the dense vars per run.
    Expr(CExpr),
    /// A byte aggregate assembled per run from its fields.
    Pack(Arc<PackPlan>),
}

/// A `bytes<n>` param's image: `(offset, width, value)` per scalar field and
/// `(offset, descriptor)` per tensormap field, the rest zero. Pointers,
/// literals and descriptors are finished at load; var-dependent fields are
/// evaluated per run.
pub(crate) struct PackPlan {
    size: usize,
    fields: Vec<(usize, usize, Slot)>,
    maps: Vec<(usize, TmaBlob)>,
}

impl PackPlan {
    /// The image for one run: every field's low `width` bytes, little-endian, at its offset.
    pub(crate) fn image(&self, vars: &Dense) -> Result<Vec<u8>> {
        let mut out = vec![0u8; self.size];
        for (at, blob) in &self.maps {
            out[*at..*at + 128].copy_from_slice(&blob.0);
        }
        for (at, width, slot) in &self.fields {
            let v = match slot {
                Slot::Const(rv) => rv.val,
                Slot::Expr(e) => e.eval(vars)?,
                Slot::Pack(_) => bail!(Manifest, "a pack field cannot be a pack"),
            };
            out[*at..*at + *width].copy_from_slice(&v.to_le_bytes()[..*width]);
        }
        Ok(out)
    }
}

/// A `CUtensorMap` image, copied into a pack image at its (64-byte aligned) field offset.
pub(crate) struct TmaBlob([u8; 128]);

pub(crate) enum LaunchKind {
    Cubin {
        func: sys::CUfunction,
        block: [u32; 3],
        grid: [CExpr; 3],
        shared_mem: Option<CExpr>,
        cluster: Option<[u32; 3]>,
        /// Launched with programmatic stream serialization: the kernel
        /// waits on its predecessor itself (`griddepcontrol.wait`).
        pdl: bool,
    },
    /// `extern:cublaslt_bf16_tn` / `..._acc` (beta 0.0 / 1.0); 6 args, or
    /// 7 with C's row stride, or 8 with C's and A's row strides.
    Gemm { beta: f32 },
    /// `extern:cublas_bf16_tn_f32`: same operands, f32 result (cublasGemmEx).
    GemmF32,
}

pub(crate) struct Launch {
    pub(crate) kind: LaunchKind,
    pub(crate) slots: Vec<Slot>,
    /// Error context: which call and impl launch this came from.
    pub(crate) ctx: String,
}

pub(crate) struct CompiledProgram {
    /// The manifest's `graph`: driven through a captured CUDA graph.
    pub(crate) graph: bool,
    pub(crate) launches: Vec<Launch>,
    /// Launch index range `[lo, hi)` of every call, in call order (a
    /// multi-launch impl contributes several launches).
    pub(crate) call_ranges: Vec<(usize, usize)>,
    /// Per manifest var, whether any launch reads it: a run's vars must
    /// carry exactly these; the others are no part of the program.
    pub(crate) vars: Vec<bool>,
}

impl Launch {
    /// Mark the vars the launch's grid, shared memory and args read.
    fn mark(&self, used: &mut [bool]) {
        if let LaunchKind::Cubin { grid, shared_mem, .. } = &self.kind {
            grid.iter().chain(shared_mem.iter()).for_each(|e| e.mark(used));
        }
        for s in &self.slots {
            match s {
                Slot::Const(_) => {}
                Slot::Expr(e) => e.mark(used),
                Slot::Pack(p) => {
                    for (_, _, f) in &p.fields {
                        if let Slot::Expr(e) = f {
                            e.mark(used);
                        }
                    }
                }
            }
        }
    }
}

/// One launch of an op implementation, resolved against the loaded modules.
enum LaunchImpl {
    Cubin {
        func: sys::CUfunction,
        module: String,
        path: PathBuf,
        entry: String,
    },
    GemmBf16Tn {
        beta: f32,
    },
    /// cublasGemmEx with an f32 result (`extern:cublas_bf16_tn_f32`).
    GemmBf16TnF32,
}

/// An op implementation, resolved: one entry per launch, plus the private
/// scratch buffers the impl declared (allocated once at var max, reused
/// every call — contents are dead outside a single call).
pub(crate) struct ResolvedOp {
    launches: Vec<LaunchImpl>,
    pub(crate) scratch: BTreeMap<String, DeviceBuf>,
}

impl ResolvedOp {
    /// The module each launch resolved to, in launch order (introspection).
    pub(crate) fn launch_modules(&self) -> Vec<String> {
        self.launches
            .iter()
            .map(|s| match s {
                LaunchImpl::Cubin { module, .. } => module.clone(),
                LaunchImpl::GemmBf16Tn { .. } => "runtime built-in (cublasLt)".into(),
                LaunchImpl::GemmBf16TnF32 => "runtime built-in (cublasGemmEx, f32 out)".into(),
            })
            .collect()
    }
}

/// Byte size of a shaped declaration at var upper bounds.
pub(crate) fn shaped_bytes(
    what: &str,
    shape: &[Dim],
    dtype_bytes: u64,
    vars_max: &BTreeMap<String, u64>,
) -> Result<u64> {
    let mut elems = 1u64;
    for d in shape {
        let n = match d {
            Dim::Const(c) => *c,
            Dim::Var(s) => {
                *vars_max.get(s).ok_or_else(|| Error::Manifest(format!("{what}: unknown var `{s}` in shape")))?
            }
        };
        elems = elems.checked_mul(n).ok_or_else(|| Error::Manifest(format!("{what}: size overflow")))?;
    }
    elems.checked_mul(dtype_bytes).ok_or_else(|| Error::Manifest(format!("{what}: size overflow")))
}

/// Resolve every op: match each launch's entry + declared param layout
/// against the loaded modules (or an extern built-in), pin the module's
/// hash, opt into >48KB dynamic shared memory, allocate scratch.
pub(crate) fn resolve_ops(
    manifest: &Manifest,
    modules: &[LoadedModule],
    kernels_dir: &Path,
    stream: &Arc<CudaStream>,
    vars_max: &BTreeMap<String, u64>,
) -> Result<BTreeMap<String, ResolvedOp>> {
    let mut ops = BTreeMap::new();
    for (name, op) in &manifest.ops {
        let mut launches = Vec::new();
        for (li, l) in op.imp.launches.iter().enumerate() {
            let k = match l {
                kern_manifest::types::Launch::Extern(e) => {
                    let ext = e.entry.strip_prefix("extern:").unwrap_or(&e.entry);
                    match ext {
                        "cublaslt_bf16_tn" => launches.push(LaunchImpl::GemmBf16Tn { beta: 0.0 }),
                        "cublaslt_bf16_tn_acc" => launches.push(LaunchImpl::GemmBf16Tn { beta: 1.0 }),
                        "cublas_bf16_tn_f32" => launches.push(LaunchImpl::GemmBf16TnF32),
                        _ => bail!(Manifest, "op `{name}` launch #{li}: unsupported extern `{ext}`"),
                    }
                    continue;
                }
                kern_manifest::types::Launch::Kernel(k) => k,
            };
            // Identity: the launch's module pins a sha256 (the verifier
            // resolved the name; the source is a label). Only artifacts with
            // that hash are candidates; same-named instances inside one are
            // told apart by param layout.
            let md = &manifest.modules[&k.module];
            let sha = md.sha256.to_lowercase();
            if !modules.iter().any(|m| m.sha == sha) {
                bail!(
                    KernelArtifact,
                    "op `{name}` launch #{li}: module `{}` ({} @{}) is not among the artifacts in {} — \
                     the source is a label, the hash is the identity; put an artifact with that \
                     sha256 there (`kern kernels`, or tools/extract_kernels.sh <manifest> <dump dirs> {})",
                    k.module,
                    md.source,
                    &sha[..12],
                    kernels_dir.display(),
                    kernels_dir.display()
                );
            }
            let want: Vec<usize> = l.params_of(op).iter().map(|p| p.size_bytes() as usize).collect();
            let entry =
                CString::new(k.entry.as_str()).map_err(|e| Error::Manifest(format!("op `{name}` entry: {e}")))?;
            let mut resolved = None;
            let mut seen = Vec::new();
            for m in modules.iter().filter(|m| m.sha == sha) {
                let Ok(func) = (unsafe { cu::module::get_function(m.module, entry.clone()) }) else {
                    continue;
                };
                let got = param_sizes(func)?;
                if got == want {
                    resolved = Some(LaunchImpl::Cubin {
                        func,
                        module: format!("{}@{}", m.label, &m.sha[..8]),
                        path: m.path.clone(),
                        entry: k.entry.clone(),
                    });
                    break;
                }
                seen.push(format!("{}@{}: {got:?}", m.label, &m.sha[..8]));
            }
            let Some(r) = resolved else {
                bail!(
                    KernelArtifact,
                    "op `{name}` launch #{li} ({}): module `{}` @{} has no instance matching the \
                     declared param layout {want:?}; saw {seen:?}",
                    k.entry,
                    k.module,
                    &sha[..12]
                );
            };
            // Opt in to >48KB dynamic shared memory where the launch needs it.
            if let (LaunchImpl::Cubin { func, .. }, Some(sm)) = (&r, &k.shared_mem) {
                let bytes = sm.eval(vars_max).map_err(|e| Error::Manifest(format!("op `{name}`: {e}")))?;
                if bytes > 48 * 1024 {
                    cuda_check(
                        unsafe {
                            sys::cuFuncSetAttribute(
                                *func,
                                sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                bytes as i32,
                            )
                        },
                        "cuFuncSetAttribute",
                    )?;
                }
            }
            launches.push(r);
        }
        let mut scratch = BTreeMap::new();
        for (sname, sd) in &op.imp.scratch {
            let bytes = shaped_bytes(&format!("op `{name}` scratch `{sname}`"), &sd.shape, sd.dtype.bytes(), vars_max)?;
            scratch.insert(sname.clone(), alloc(stream, bytes)?);
        }
        ops.insert(name.clone(), ResolvedOp { launches, scratch });
    }
    Ok(ops)
}

/// What a call binds that a plain single-GPU manifest has none of: this
/// rank's index per group, and which buffers hold peer addresses.
pub(crate) struct Ranks<'a> {
    pub(crate) ranks: &'a BTreeMap<String, u64>,
    pub(crate) peer_buffers: &'a BTreeSet<String>,
}

/// Lower every program's call list into a flat launch list. Every launch
/// that receives a peer buffer is SASS-scanned for multicast TMA first.
pub(crate) fn compile_programs(
    manifest: &Manifest,
    ops: &BTreeMap<String, ResolvedOp>,
    buffers: &BTreeMap<String, DeviceBuf>,
    states: &BTreeMap<String, DeviceBuf>,
    ranks: &Ranks,
) -> Result<BTreeMap<String, CompiledProgram>> {
    let vars: BTreeMap<&str, usize> = manifest.vars.keys().enumerate().map(|(i, s)| (s.as_str(), i)).collect();
    let mut scan = MulticastScan::new();
    let mut programs = BTreeMap::new();
    for (pname, p) in &manifest.programs {
        let calls = &p.calls;
        let mut launches = Vec::new();
        let mut call_ranges = Vec::with_capacity(calls.len());
        for (ci, c) in calls.iter().enumerate() {
            let cctx = call_ctx(ci, c);
            let (Some(op), Some(rop)) = (manifest.ops.get(&c.op), ops.get(&c.op)) else {
                bail!(Manifest, "program `{pname}` {cctx}: unknown op");
            };
            let lo = launches.len();
            compile_call(c, op, rop, &cctx, buffers, states, &vars, ranks, &mut scan, &mut launches)
                .map_err(|e| Error::Call { context: format!("program `{pname}` {cctx}"), source: Box::new(e) })?;
            call_ranges.push((lo, launches.len()));
        }
        let mut used = vec![false; manifest.vars.len()];
        launches.iter().for_each(|l| l.mark(&mut used));
        programs.insert(pname.clone(), CompiledProgram { graph: p.graph, launches, call_ranges, vars: used });
    }
    Ok(programs)
}

/// Error context locating one entry of a program's call list.
fn call_ctx(i: usize, c: &Call) -> String {
    match &c.label {
        Some(l) => format!("call #{i} `{l}` (op `{}`)", c.op),
        None => format!("call #{i} (op `{}`)", c.op),
    }
}

#[allow(clippy::too_many_arguments)]
fn compile_call(
    c: &Call,
    op: &Op,
    rop: &ResolvedOp,
    cctx: &str,
    buffers: &BTreeMap<String, DeviceBuf>,
    states: &BTreeMap<String, DeviceBuf>,
    vars: &BTreeMap<&str, usize>,
    ranks: &Ranks,
    scan: &mut MulticastScan,
    launches: &mut Vec<Launch>,
) -> Result<()> {
    if c.args.len() != op.params.len() {
        bail!(Manifest, "op takes {} args, call passes {}", op.params.len(), c.args.len());
    }
    // Lower the interface args once; each launch then wires its own params
    // from these, its scratch, and its private literals. `peer` marks the
    // args a kernel can derive a peer address from.
    let mut vals = Vec::with_capacity(c.args.len());
    let mut peer = Vec::with_capacity(c.args.len());
    for (arg, pty) in c.args.iter().zip(&op.params) {
        vals.push(match pty {
            ParamType::Buf { .. } | ParamType::State { .. } => Slot::Const(pointer_arg(arg, buffers, states)?),
            ParamType::Scalar(_) => scalar_arg(arg, vars, ranks.ranks)?,
            ParamType::Bytes(_) => bail!(Manifest, "interface params cannot be byte aggregates"),
        });
        peer.push(matches!(arg, Arg::Buf { buf, .. } if ranks.peer_buffers.contains(buf)));
    }
    for (li, (l, imp)) in op.imp.launches.iter().zip(&rop.launches).enumerate() {
        let wiring = l.args_of(op);
        let mut slots = Vec::with_capacity(wiring.len());
        let mut touches_peer = false;
        for la in wiring.iter() {
            match la {
                LaunchArg::Param { param } => {
                    touches_peer |= peer.get(*param).copied().unwrap_or(false);
                }
                LaunchArg::Pack { pack } => {
                    for f in &pack.fields {
                        if let FieldSrc::Param { param } | FieldSrc::TensorMap { tensormap: TensorMap { param, .. } } =
                            &f.src
                        {
                            touches_peer |= peer.get(*param).copied().unwrap_or(false);
                        }
                    }
                }
                _ => {}
            }
            slots.push(match la {
                LaunchArg::Param { param } => vals
                    .get(*param)
                    .cloned()
                    .ok_or_else(|| Error::Manifest(format!("launch #{li}: forwarded param #{param} out of range")))?,
                LaunchArg::Scratch { scratch } => {
                    let Some(b) = rop.scratch.get(scratch) else {
                        bail!(Manifest, "launch #{li}: unknown scratch `{scratch}`");
                    };
                    Slot::Const(RVal { val: b.ptr, bytes: b.bytes })
                }
                LaunchArg::I32 { i32: v } => lit(*v as u32 as u64),
                LaunchArg::I64 { i64: v } => lit(*v as u64),
                LaunchArg::U8 { u8: v } => lit(*v as u64),
                LaunchArg::F32 { f32: v } => lit(v.to_bits() as u64),
                LaunchArg::Rank { rank } => lit(rank_index(ranks.ranks, rank)?),
                LaunchArg::Pack { pack } => {
                    Slot::Pack(Arc::new(pack_plan(pack, &vals, c, op, rop, ranks.ranks, vars, li)?))
                }
            });
        }
        let kind = match imp {
            LaunchImpl::GemmBf16Tn { .. } | LaunchImpl::GemmBf16TnF32 => {
                if touches_peer {
                    bail!(Manifest, "launch #{li}: a peer buffer reaches the extern gemm; runtime built-ins never receive peer memory");
                }
                let max_args = if matches!(imp, LaunchImpl::GemmBf16Tn { .. }) { 8 } else { 7 };
                if !(6..=max_args).contains(&slots.len()) {
                    bail!(
                        Manifest,
                        "launch #{li}: extern gemm takes 6..={max_args} args (a, w, c, m, n, k, optional C/A row strides), got {}",
                        slots.len()
                    );
                }
                match imp {
                    LaunchImpl::GemmBf16Tn { beta } => LaunchKind::Gemm { beta: *beta },
                    _ => LaunchKind::GemmF32,
                }
            }
            LaunchImpl::Cubin { func, path, entry, .. } => {
                let Some(k) = l.kernel() else {
                    bail!(Manifest, "launch #{li}: a cubin resolved for an extern launch");
                };
                if touches_peer {
                    let bad = scan.offending(path, entry)?;
                    if !bad.is_empty() {
                        bail!(
                            KernelArtifact,
                            "launch #{li}: `{entry}` in {} receives a peer buffer but issues multicast TMA, \
                             which wedges the GPU at a peer address: {}",
                            path.display(),
                            bad.join(" | ")
                        );
                    }
                }
                LaunchKind::Cubin {
                    func: *func,
                    block: k.block,
                    grid: [
                        compile_expr(&k.grid[0], vars)?,
                        compile_expr(&k.grid[1], vars)?,
                        compile_expr(&k.grid[2], vars)?,
                    ],
                    shared_mem: k.shared_mem.as_ref().map(|e| compile_expr(e, vars)).transpose()?,
                    cluster: k.cluster,
                    pdl: k.pdl,
                }
            }
        };
        launches.push(Launch { kind, slots, ctx: format!("{cctx} launch #{li} (`{}`)", l.entry()) });
    }
    Ok(())
}

/// Lower a pack's fields against the op's lowered interface values.
#[allow(clippy::too_many_arguments)]
fn pack_plan(
    pack: &Pack,
    vals: &[Slot],
    c: &Call,
    op: &Op,
    rop: &ResolvedOp,
    ranks: &BTreeMap<String, u64>,
    vars: &BTreeMap<&str, usize>,
    li: usize,
) -> Result<PackPlan> {
    let mut fields = Vec::with_capacity(pack.fields.len());
    let mut maps = Vec::new();
    for (k, f) in pack.fields.iter().enumerate() {
        if let FieldSrc::TensorMap { tensormap: t } = &f.src {
            let at = f.at as usize;
            if !at.is_multiple_of(64) || at + 128 > pack.size as usize {
                bail!(
                    Manifest,
                    "launch #{li}: pack field #{k}: tensormap at {at} does not fit the {} byte image",
                    pack.size
                );
            }
            maps.push((at, tensor_map_blob(t, vals, c, li)?));
            continue;
        }
        let (slot, natural) = match &f.src {
            FieldSrc::Param { param } => match (vals.get(*param), op.params.get(*param)) {
                (Some(s), Some(p)) => (s.clone(), p.size_bytes() as u32),
                _ => bail!(Manifest, "launch #{li}: pack field #{k}: interface param #{param} out of range"),
            },
            FieldSrc::Scratch { scratch } => match rop.scratch.get(scratch) {
                Some(b) => (Slot::Const(RVal { val: b.ptr, bytes: b.bytes }), 8),
                None => bail!(Manifest, "launch #{li}: pack field #{k}: unknown scratch `{scratch}`"),
            },
            FieldSrc::I32 { i32: v } => (lit(*v as u32 as u64), 4),
            FieldSrc::I64 { i64: v } => (lit(*v as u64), 8),
            FieldSrc::F32 { f32: v } => (lit(v.to_bits() as u64), 4),
            FieldSrc::U8 { u8: v } => (lit(*v as u64), 1),
            FieldSrc::Var { var } => (Slot::Expr(CExpr::Var(var_index(vars, var)?)), 4),
            FieldSrc::Expr { expr } => (Slot::Expr(compile_expr(expr, vars)?), 4),
            FieldSrc::Rank { rank } => (lit(rank_index(ranks, rank)?), 4),
            FieldSrc::TensorMap { .. } => unreachable!("handled above"),
        };
        let width = f.width.unwrap_or(natural) as usize;
        let at = f.at as usize;
        if width == 0 || width > 8 || at + width > pack.size as usize {
            bail!(
                Manifest,
                "launch #{li}: pack field #{k}: {width} bytes at {at} do not fit the {} byte image",
                pack.size
            );
        }
        fields.push((at, width, slot));
    }
    Ok(PackPlan { size: pack.size as usize, fields, maps })
}

/// Encode a tensormap field over the call's finished pointer for its interface param.
fn tensor_map_blob(t: &TensorMap, vals: &[Slot], c: &Call, li: usize) -> Result<TmaBlob> {
    let Some(Slot::Const(rv)) = vals.get(t.param) else {
        bail!(Manifest, "launch #{li}: tensormap over interface param #{} which is not a pointer", t.param);
    };
    let Some(buf) = c.args.get(t.param) else {
        bail!(Manifest, "launch #{li}: malformed tensormap over interface param #{}", t.param);
    };
    let fp = span_footprint(t).ok_or_else(|| {
        Error::Manifest(format!("launch #{li}: malformed tensormap over interface param #{}", t.param))
    })?;
    if fp > rv.bytes {
        bail!(
            Manifest,
            "launch #{li}: tensormap over interface param #{} addresses {fp} bytes but {buf} has {} bytes left",
            t.param,
            rv.bytes
        );
    }
    encode_tensor_map(t, *rv)
        .map_err(|e| Error::Cuda(format!("launch #{li}: tensormap over interface param #{}: {e}", t.param)))
}

/// Bytes a tensormap addresses from its base; a spanning outermost dim
/// counts once (the runtime extends it to the buffer at encode time).
fn span_footprint(t: &TensorMap) -> Option<u64> {
    match t.dims.last() {
        Some(0) => {
            let inner = TensorMap {
                dims: t.dims[..t.dims.len() - 1].to_vec(),
                strides: t.strides[..t.strides.len() - 1].to_vec(),
                ..t.clone()
            };
            inner.footprint()
        }
        _ => t.footprint(),
    }
}

/// Lower a buffer/state arg to its finished pointer value.
fn pointer_arg(arg: &Arg, buffers: &BTreeMap<String, DeviceBuf>, states: &BTreeMap<String, DeviceBuf>) -> Result<RVal> {
    let (map, name, offset, what) = match arg {
        Arg::Buf { buf, offset } => (buffers, buf, *offset, "buffer"),
        Arg::State { state, offset } => (states, state, *offset, "state"),
        _ => bail!(Manifest, "expected buffer/state arg, got {arg}"),
    };
    let Some(b) = map.get(name) else {
        bail!(Manifest, "unknown {what} `{name}`");
    };
    offset_into(b, offset, || format!("{what} `{name}`"))
}

fn offset_into(b: &DeviceBuf, offset: u64, what: impl Fn() -> String) -> Result<RVal> {
    // A pooled state's pointer spans its reservation, not the pages and
    // slots made at load: a tensormap that fills the state is encoded once,
    // and must reach the slots a remap makes later.
    let Some(bytes) = b.span().checked_sub(offset) else {
        bail!(Manifest, "offset {offset} outside {} ({} bytes)", what(), b.span());
    };
    Ok(RVal { val: b.ptr + offset, bytes })
}

/// Lower a scalar arg: literals and ranks finish now, vars and expressions
/// become dense-indexed expressions.
fn scalar_arg(arg: &Arg, vars: &BTreeMap<&str, usize>, ranks: &BTreeMap<String, u64>) -> Result<Slot> {
    Ok(match arg {
        Arg::Var { var } => Slot::Expr(CExpr::Var(var_index(vars, var)?)),
        Arg::Expr { expr } => Slot::Expr(compile_expr(expr, vars)?),
        Arg::I32 { i32: v } => lit(*v as u32 as u64),
        Arg::I64 { i64: v } => lit(*v as u64),
        Arg::U8 { u8: v } => lit(*v as u64),
        Arg::F32 { f32: v } => lit(v.to_bits() as u64),
        Arg::Rank { rank } => lit(rank_index(ranks, rank)?),
        Arg::Buf { .. } | Arg::State { .. } => bail!(Manifest, "expected scalar arg, got {arg}"),
    })
}

fn rank_index(ranks: &BTreeMap<String, u64>, group: &str) -> Result<u64> {
    match ranks.get(group) {
        Some(&i) => Ok(i),
        None => bail!(Manifest, "no rank for topology group `{group}`"),
    }
}

fn lit(val: u64) -> Slot {
    Slot::Const(RVal { val, bytes: 0 })
}

fn compile_expr(e: &Expr, vars: &BTreeMap<&str, usize>) -> Result<CExpr> {
    Ok(match e {
        Expr::Const(c) => CExpr::Const(*c),
        Expr::Var(var) => CExpr::Var(var_index(vars, var)?),
        Expr::CeilDiv { ceil_div: (inner, c) } => {
            if *c == 0 {
                bail!(Manifest, "expression: division by zero");
            }
            CExpr::CeilDiv(Box::new(compile_expr(inner, vars)?), *c)
        }
        Expr::Mul { mul: (inner, c) } => CExpr::Mul(Box::new(compile_expr(inner, vars)?), *c),
    })
}

fn var_index(vars: &BTreeMap<&str, usize>, var: &str) -> Result<usize> {
    match vars.get(var) {
        Some(&i) => Ok(i),
        None => bail!(Manifest, "unknown var `{var}`"),
    }
}

/// `cuTensorMapEncodeTiled` for a manifest tensormap at a finished device
/// address. Interleave none, element strides 1.
fn encode_tensor_map(t: &TensorMap, rv: RVal) -> Result<TmaBlob> {
    use sys::CUtensorMapDataType as Dt;
    use sys::CUtensorMapL2promotion as L2;
    use sys::CUtensorMapSwizzle as Sw;
    let ptr = rv.val;
    if !ptr.is_multiple_of(16) {
        bail!(Manifest, "device address {ptr:#x} is not 16-byte aligned");
    }
    let dtype = match t.dtype {
        TmaDType::U8 => Dt::CU_TENSOR_MAP_DATA_TYPE_UINT8,
        TmaDType::U16 => Dt::CU_TENSOR_MAP_DATA_TYPE_UINT16,
        TmaDType::U32 => Dt::CU_TENSOR_MAP_DATA_TYPE_UINT32,
        TmaDType::I32 => Dt::CU_TENSOR_MAP_DATA_TYPE_INT32,
        TmaDType::U64 => Dt::CU_TENSOR_MAP_DATA_TYPE_UINT64,
        TmaDType::I64 => Dt::CU_TENSOR_MAP_DATA_TYPE_INT64,
        TmaDType::F16 => Dt::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        TmaDType::Bf16 => Dt::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        TmaDType::F32 => Dt::CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        TmaDType::Tf32 => Dt::CU_TENSOR_MAP_DATA_TYPE_TFLOAT32,
        TmaDType::U4 => Dt::CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,
        TmaDType::U4packed => Dt::CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
    };
    let swizzle = match t.swizzle {
        0 => Sw::CU_TENSOR_MAP_SWIZZLE_NONE,
        32 => Sw::CU_TENSOR_MAP_SWIZZLE_32B,
        64 => Sw::CU_TENSOR_MAP_SWIZZLE_64B,
        128 => Sw::CU_TENSOR_MAP_SWIZZLE_128B,
        v => bail!(Manifest, "swizzle {v} is not one of 0, 32, 64, 128"),
    };
    let l2 = match t.l2_promotion {
        0 => L2::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        64 => L2::CU_TENSOR_MAP_L2_PROMOTION_L2_64B,
        128 => L2::CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        256 => L2::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        v => bail!(Manifest, "l2_promotion {v} is not one of 0, 64, 128, 256"),
    };
    let oob = if t.oob_nan {
        sys::CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
    } else {
        sys::CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    };
    let rank = t.dims.len() as u32;
    let mut dims: Vec<u64> = t.dims.clone();
    if dims.last() == Some(&0) {
        // Span: as many outermost slices as the buffer holds past the base.
        let inner = span_footprint(t).unwrap_or(0);
        let stride = *t.strides.last().unwrap_or(&1);
        dims[rank as usize - 1] = (rv.bytes.saturating_sub(inner)) / stride + 1;
    }
    let strides: Vec<u64> = t.strides.clone();
    let boxes: Vec<u32> = t.box_.clone();
    let elem_strides: Vec<u32> = vec![1; t.dims.len()];
    let mut map = sys::CUtensorMap { opaque: [0; 16] };
    cuda_check(
        unsafe {
            sys::cuTensorMapEncodeTiled(
                &mut map,
                dtype,
                rank,
                ptr as *mut c_void,
                dims.as_ptr(),
                strides.as_ptr(),
                boxes.as_ptr(),
                elem_strides.as_ptr(),
                sys::CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
                swizzle,
                l2,
                oob,
            )
        },
        "cuTensorMapEncodeTiled",
    )?;
    let bytes: [u8; 128] = unsafe { std::mem::transmute(map.opaque) };
    Ok(TmaBlob(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_image_places_fields_and_zeroes_the_rest() {
        let plan = PackPlan {
            size: 24,
            fields: vec![
                (0, 8, Slot::Const(RVal { val: 0xf767e4600400, bytes: 0 })),
                (8, 4, Slot::Const(RVal { val: 512, bytes: 0 })),
                (12, 4, Slot::Expr(CExpr::Var(0))),
                (16, 8, Slot::Expr(CExpr::Mul(Box::new(CExpr::Var(0)), 512))),
            ],
            maps: vec![],
        };
        let img = plan.image(&Dense(vec![3])).unwrap();
        assert_eq!(&img[..8], &0xf767e4600400u64.to_le_bytes());
        assert_eq!((&img[8..12], &img[12..16]), (&512u32.to_le_bytes()[..], &3u32.to_le_bytes()[..]));
        assert_eq!(&img[16..24], &1536u64.to_le_bytes());
        assert_eq!(plan.image(&Dense(vec![7])).unwrap()[12], 7);
    }

    #[test]
    fn span_footprint_counts_the_inner_slice_once() {
        let t = TensorMap {
            param: 0,
            dtype: TmaDType::Bf16,
            dims: vec![512, 64, 0],
            strides: vec![1152, 73728],
            box_: vec![64, 64, 1],
            swizzle: 128,
            l2_promotion: 0,
            oob_nan: false,
        };
        assert_eq!((span_footprint(&t), t.footprint()), (Some(1024 + 63 * 1152), None));
    }
}
