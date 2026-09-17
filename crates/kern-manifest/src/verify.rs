//! Load-time verification: refuse any manifest that is not provably
//! self-consistent, before anything touches the GPU. All errors are
//! collected and reported together, rustc-style.
//!
//! Checks:
//!   1. schema_version
//!   2. vars: max > 0
//!   3. states: exactly one of bytes_per_token / bytes / bytes_per_seq is non-zero
//!   4. buffers: shapes resolve, byte sizes don't overflow at var upper
//!      bounds; a declared domain is well-formed (bound kinds vs dtype,
//!      `index_into` resolves, min <= max at the var corners)
//!   5. modules: sha256 shape, source well-formed (registry refs parse)
//!   6. ops (interface + implementation):
//!      - scratch shapes resolve, every scratch is used
//!      - per launch: module ref resolves; geometry present iff not
//!        `extern:`; CUDA block/grid limits at var upper bounds (and grid
//!        non-zero at lower bounds), shared-mem opt-in cap, arg/param arity,
//!        per-position wiring types (a forwarded interface param must match
//!        kind/dtype, a launch may not write through an interface `in`
//!        param, literal types)
//!      - impl dataflow: scratch never read before a launch wrote it; every
//!        interface `out` param written by some launch
//!   7. calls: op refs resolve, arg/param arity and per-position type match
//!      against the interface, var ranges fit scalar params
//!   8. dataflow per program: no read-before-write, no writes to input,
//!      weight or peer buffers, every output / carry buffer written by some
//!      program; a program is `once` or has a `batch`, not both; a batch
//!      has `groups >= 1` and constant `rows >= 1` or a declared var
//!   9. no unused declarations; a `fill` sits on an input or output of an
//!      integer dtype, input roles on inputs and output roles on outputs
//!      (whether the fills add up to a serving contract is
//!      [`crate::protocol`]'s question, not this one's)
//!  10. topology: group sizes > 0, every group used; a `peer` buffer is
//!      `u64[group size]` `of` an exported buffer or a state; `export`,
//!      `of` and `group` only where they mean something; `{"rank": g}`
//!      binds only to i32/i64 params; an op with an extern launch never
//!      receives a peer buffer (runtime built-ins may not touch peer memory)
//!
//! What this deliberately cannot check: kernel *behavior*, and the
//! *semantics* of interface params (that a replacement implementation
//! interprets position #3 as the same row stride). A cubin that lies about
//! what it touches is inside the trust boundary; the manifest only makes
//! the lie explicit and diffable. Cross-checking launch param layouts against
//! `cuFuncGetParamInfo` is a load-time (phase 2) concern in the runtime
//! crate, since it needs the CUDA driver.

use crate::types::*;
use std::collections::{BTreeMap, BTreeSet};

const MAX_GRID_X: u64 = (1 << 31) - 1;
const MAX_GRID_YZ: u64 = 65_535;
const MAX_BLOCK_THREADS: u64 = 1024;
const MAX_BLOCK_Z: u32 = 64;
/// Per-block dynamic shared memory after the `cuFuncSetAttribute` opt-in the
/// runtime performs for any launch declaring `shared_mem`: 227 KiB on
/// sm90/sm100/sm103 datacenter parts.
const MAX_DYN_SHARED_MEM: u64 = 232_448;
/// Blocks per thread-block cluster (non-portable maximum on sm_90+).
const MAX_CLUSTER_BLOCKS: u64 = 16;

/// Sized like buffers: dtype bytes x dims at var upper bounds.
fn shaped_size(
    what: &str,
    dtype: DType,
    shape: &[Dim],
    vars_max: &BTreeMap<String, u64>,
    used_vars: &mut BTreeSet<String>,
    errs: &mut Vec<String>,
) -> Option<u64> {
    if shape.is_empty() {
        errs.push(format!("{what}: shape must not be empty"));
        return None;
    }
    let mut dim_err = false;
    let mut size: Option<u64> = Some(dtype.bytes());
    for dim in shape {
        let extent = match dim {
            Dim::Const(0) => {
                errs.push(format!("{what}: zero-sized dimension"));
                dim_err = true;
                None
            }
            Dim::Const(c) => Some(*c),
            Dim::Var(s) => match vars_max.get(s) {
                Some(mx) => {
                    used_vars.insert(s.clone());
                    Some(*mx)
                }
                None => {
                    errs.push(format!("{what}: unknown var `{s}` in shape"));
                    dim_err = true;
                    None
                }
            },
        };
        size = match (size, extent) {
            (Some(a), Some(e)) => a.checked_mul(e),
            _ => None,
        };
    }
    if size.is_none() && !dim_err {
        errs.push(format!("{what}: byte size overflows u64 at var upper bounds"));
    }
    size
}

/// A manifest that passed [`verify`]: every reference resolves, every
/// launch fits the device limits at the var bounds, every program's
/// dataflow is sound. Constructed by `verify` alone; a runtime loads only
/// one of these, so nothing downstream checks again. Derefs to the
/// [`Manifest`].
#[derive(Debug, Clone)]
pub struct Verified(Manifest);

impl Verified {
    /// Parse and verify; a parse error is reported as the one diagnostic.
    pub fn from_json(s: &str) -> Result<Verified, VerifyErrors> {
        let m = Manifest::from_json(s).map_err(|e| VerifyErrors(vec![e.to_string()]))?;
        verify(m)
    }
}

impl std::ops::Deref for Verified {
    type Target = Manifest;
    fn deref(&self) -> &Manifest {
        &self.0
    }
}

/// Every diagnostic [`verify`] collected, reported together rustc-style.
/// Derefs to the individual messages.
#[derive(Debug)]
pub struct VerifyErrors(pub Vec<String>);

impl std::fmt::Display for VerifyErrors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("manifest failed verification:")?;
        for e in &self.0 {
            write!(f, "\n  - {e}")?;
        }
        Ok(())
    }
}

impl std::error::Error for VerifyErrors {}

impl std::ops::Deref for VerifyErrors {
    type Target = [String];
    fn deref(&self) -> &[String] {
        &self.0
    }
}

/// Verify `m`; the manifest comes back as a [`Verified`] or not at all.
pub fn verify(m: Manifest) -> Result<Verified, VerifyErrors> {
    let errs = diagnostics(&m);
    if errs.is_empty() {
        Ok(Verified(m))
    } else {
        Err(VerifyErrors(errs))
    }
}

/// Every rule, every violation.
fn diagnostics(m: &Manifest) -> Vec<String> {
    let mut errs: Vec<String> = Vec::new();
    let mut used_vars: BTreeSet<String> = BTreeSet::new();
    let mut used_buffers: BTreeSet<String> = BTreeSet::new();
    let mut used_states: BTreeSet<String> = BTreeSet::new();
    let mut used_modules: BTreeSet<String> = BTreeSet::new();
    let mut used_ops: BTreeSet<String> = BTreeSet::new();
    let mut used_groups: BTreeSet<String> = BTreeSet::new();

    // 1. format
    if m.schema_version != SCHEMA_VERSION {
        errs.push(format!("unsupported schema_version {} (this runtime reads {SCHEMA_VERSION})", m.schema_version));
    }

    // 2. vars
    for (name, v) in &m.vars {
        if v.max < Var::MIN {
            errs.push(format!("var `{name}`: max must be >= {}", Var::MIN));
        }
    }
    let vars_max: BTreeMap<String, u64> = m.vars.iter().map(|(k, v)| (k.clone(), v.max)).collect();
    let vars_min: BTreeMap<String, u64> = m.vars.keys().map(|k| (k.clone(), Var::MIN)).collect();

    // 10a. topology
    if let Some(t) = &m.topology {
        for (name, size) in &t.groups {
            if *size == 0 {
                errs.push(format!("topology group `{name}`: size must be > 0"));
            }
        }
    }
    let group_ctx = |g: &str, errs: &mut Vec<String>, used_groups: &mut BTreeSet<String>, ctx: &str| -> Option<u64> {
        match m.group_size(g) {
            Some(sz) => {
                used_groups.insert(g.to_string());
                Some(sz)
            }
            None => {
                errs.push(match &m.topology {
                    Some(_) => format!("{ctx}: unknown topology group `{g}`"),
                    None => format!("{ctx}: group `{g}` but the manifest declares no topology"),
                });
                None
            }
        }
    };

    // 3. states
    for (name, st) in &m.states {
        let set = [st.bytes_per_token, st.bytes, st.bytes_per_seq].iter().filter(|&&b| b > 0).count();
        match set {
            0 => errs.push(format!("state `{name}`: one of bytes_per_token / bytes / bytes_per_seq must be > 0")),
            1 => {}
            _ => errs.push(format!("state `{name}`: bytes_per_token, bytes and bytes_per_seq are exclusive")),
        }
    }

    // 4. buffers
    let mut buf_sizes: BTreeMap<&str, u64> = BTreeMap::new();
    for (name, b) in &m.buffers {
        if let Some(sz) =
            shaped_size(&format!("buffer `{name}`"), b.dtype, &b.shape, &vars_max, &mut used_vars, &mut errs)
        {
            buf_sizes.insert(name, sz);
        }
        if let Some(d) = &b.domain {
            check_domain(name, b, d, m, &vars_max, &vars_min, &mut used_vars, &mut errs);
        }
        let ctx = format!("buffer `{name}`");
        // 9b. fill
        if let Some(fill) = b.fill {
            let output = matches!(fill, Fill::Tokens | Fill::Count | Fill::Error);
            match (b.kind, output) {
                (BufferKind::Input, false) | (BufferKind::Output, true) => {}
                (BufferKind::Input | BufferKind::Output, _) => errs.push(format!(
                    "{ctx}: fill `{fill}` is {} but the buffer is {}",
                    if output { "read from an output" } else { "written into an input" },
                    b.kind
                )),
                (kind, _) => errs.push(format!("{ctx}: a fill sits on an input or output, not a {kind} buffer")),
            }
            if !matches!(b.dtype, DType::I32 | DType::I64) {
                errs.push(format!("{ctx}: fill `{fill}` needs an i32 or i64 buffer, not {}", b.dtype));
            }
        }
        if b.placement == Placement::Host {
            if b.kind != BufferKind::Weight || b.export {
                errs.push(format!("{ctx}: host placement requires a non-exported immutable weight"));
            }
            if b.bind
                .iter()
                .any(|s| matches!(s.tensor, TensorSource::Ranked { .. }) || matches!(s.rows, Some(Rows::Ranked { .. })))
            {
                errs.push(format!("{ctx}: shared host weights cannot select tensors or rows by rank"));
            }
        }
        // 9c. bind: a weight is its segments, nothing else has any
        match (b.kind, b.bind.is_empty()) {
            (BufferKind::Weight, true) => {
                errs.push(format!("{ctx}: a weight buffer binds at least one checkpoint tensor (`bind`)"))
            }
            (BufferKind::Weight, false) => {
                for (i, s) in b.bind.iter().enumerate() {
                    let sctx = format!("{ctx}: bind[{i}]");
                    match &s.tensor {
                        TensorSource::Named(name) => {
                            if name.is_empty() {
                                errs.push(format!("{sctx}: empty tensor name"));
                            }
                        }
                        TensorSource::Ranked { group, tensors } => {
                            match group_ctx(group, &mut errs, &mut used_groups, &sctx) {
                                Some(n) if n as usize == tensors.len() => {}
                                Some(n) => errs.push(format!(
                                    "{sctx}: group `{group}` needs {n} tensor names, got {}",
                                    tensors.len()
                                )),
                                None => {}
                            }
                            if tensors.iter().any(String::is_empty) {
                                errs.push(format!("{sctx}: empty tensor name"));
                            }
                        }
                    }
                    let rows: Vec<[u64; 2]> = match &s.rows {
                        None => vec![],
                        Some(Rows::Range(r)) => vec![*r],
                        Some(Rows::Ranked { group, ranges }) => {
                            match group_ctx(group, &mut errs, &mut used_groups, &sctx) {
                                Some(n) if n as usize == ranges.len() => {}
                                Some(n) => errs.push(format!(
                                    "{sctx}: group `{group}` needs {n} row ranges, got {}",
                                    ranges.len()
                                )),
                                None => {}
                            }
                            ranges.clone()
                        }
                    };
                    let ranges = rows.iter().map(|r| ("rows", *r)).chain(s.cols.map(|c| ("cols", c)));
                    for (axis, [from, to]) in ranges {
                        if from >= to {
                            errs.push(format!("{sctx}: {axis} [{from}, {to}) is empty"));
                        }
                    }
                }
            }
            (kind, false) => {
                errs.push(format!("{ctx}: `bind` names checkpoint tensors for a weight, not a {kind} buffer"))
            }
            (_, true) => {}
        }
        // 10b. export / peer
        if b.kind == BufferKind::Peer {
            if b.export {
                errs.push(format!("{ctx}: a peer buffer cannot itself be exported"));
            }
            if b.dtype != DType::U64 {
                errs.push(format!("{ctx}: a peer buffer holds device addresses, dtype must be u64, not {}", b.dtype));
            }
            if b.domain.is_some() {
                errs.push(format!("{ctx}: a peer buffer's contents are runtime-filled addresses; it takes no domain"));
            }
            match &b.of {
                None => errs.push(format!(
                    "{ctx}: a peer buffer must name the exported buffer or state it holds addresses `of`"
                )),
                Some(of) if of == name => errs.push(format!("{ctx}: a peer buffer cannot be `of` itself")),
                Some(of) => match (m.buffers.get(of), m.states.get(of)) {
                    (Some(_), Some(_)) => errs.push(format!("{ctx}: `of` `{of}` is both a buffer and a state")),
                    (Some(target), None) => {
                        used_buffers.insert(of.clone());
                        if target.kind == BufferKind::Peer {
                            errs.push(format!("{ctx}: `of` `{of}` is itself a peer buffer"));
                        } else if !target.export {
                            errs.push(format!("{ctx}: `of` buffer `{of}` is not exported"));
                        }
                    }
                    (None, Some(_)) => {
                        used_states.insert(of.clone());
                    }
                    (None, None) => errs.push(format!("{ctx}: `of` unknown buffer/state `{of}`")),
                },
            }
            match &b.group {
                None => errs
                    .push(format!("{ctx}: a peer buffer must name the topology `group` its addresses are indexed by")),
                Some(g) => {
                    if let Some(sz) = group_ctx(g, &mut errs, &mut used_groups, &ctx) {
                        match b.shape.as_slice() {
                            [Dim::Const(n)] if *n == sz => {}
                            _ => errs.push(format!(
                                "{ctx}: a peer buffer over group `{g}` has shape [{sz}], one address per member"
                            )),
                        }
                    }
                }
            }
        } else {
            if b.of.is_some() {
                errs.push(format!("{ctx}: `of` only applies to peer buffers"));
            }
            if b.group.is_some() {
                errs.push(format!("{ctx}: `group` only applies to peer buffers"));
            }
        }
    }

    // 5. modules
    for (name, md) in &m.modules {
        let ctx = format!("module `{name}`");
        if md.sha256.len() != 64 || !md.sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
            errs.push(format!("{ctx}: sha256 `{}` is not 64 hex chars", md.sha256));
        }
        if md.source.is_empty() {
            errs.push(format!("{ctx}: empty source"));
        }
        if let Some(Err(e)) = RegistryRef::parse(&md.source) {
            errs.push(format!("{ctx}: {e}"));
        }
    }

    // 6. ops: interface + implementation
    // Per op: (interface param, footprint bytes, context) of every tensormap
    // pack field, checked against the bound buffer at each call (rule 7).
    let mut op_tensormaps: BTreeMap<&str, Vec<(usize, u64, String)>> = BTreeMap::new();
    for (oname, op) in &m.ops {
        let imp = &op.imp;

        for (i, p) in op.params.iter().enumerate() {
            if matches!(p, ParamType::Bytes(_)) {
                errs.push(format!(
                    "op `{oname}`: interface param #{i} is a byte aggregate; packs are launch-private, \
                     the interface takes the buffers and scalars they are assembled from"
                ));
            }
        }

        for (sname, s) in &imp.scratch {
            shaped_size(
                &format!("op `{oname}` scratch `{sname}`"),
                s.dtype,
                &s.shape,
                &vars_max,
                &mut used_vars,
                &mut errs,
            );
        }

        if imp.launches.is_empty() {
            errs.push(format!("op `{oname}`: implementation has no launches"));
        }

        // Impl-level dataflow over slots: interface `out` params and scratch
        // start unwritten; `in`/`inout` interface params are caller-provided.
        let mut iface_written: Vec<bool> = op.params.iter().map(|p| p.dir() != Some(Dir::Out)).collect();
        let mut scratch_written: BTreeSet<&str> = BTreeSet::new();
        let mut scratch_used: BTreeSet<&str> = BTreeSet::new();

        for (li, launch) in imp.launches.iter().enumerate() {
            let ctx = format!("op `{oname}` launch #{li} ({})", launch.entry());
            match launch {
                Launch::Extern(e) => {
                    if !e.entry.starts_with("extern:") {
                        errs.push(format!(
                            "{ctx}: a launch without a module must be a runtime built-in (`extern:<name>`)"
                        ));
                    }
                }
                Launch::Kernel(k) => {
                    if k.entry.is_empty() {
                        errs.push(format!("{ctx}: empty entry"));
                    }
                    if k.entry.starts_with("extern:") {
                        errs.push(format!("{ctx}: an extern entry has no module or launch geometry"));
                    }
                    if m.modules.contains_key(&k.module) {
                        used_modules.insert(k.module.clone());
                    } else {
                        errs.push(format!("{ctx}: unknown module `{}`", k.module));
                    }
                    let block = &k.block;
                    let threads: u64 = block.iter().map(|&x| x as u64).product();
                    if block.contains(&0) || threads > MAX_BLOCK_THREADS {
                        errs.push(format!(
                            "{ctx}: block {block:?} exceeds {MAX_BLOCK_THREADS} threads or has a zero dim"
                        ));
                    }
                    if block[2] > MAX_BLOCK_Z {
                        errs.push(format!("{ctx}: block.z {} > {MAX_BLOCK_Z}", block[2]));
                    }
                    for (axis, e) in ["x", "y", "z"].iter().zip(&k.grid) {
                        let ectx = format!("{ctx}: grid.{axis}");
                        check_expr(e, m, &mut used_vars, &mut errs, &ectx);
                        match e.eval(&vars_max) {
                            Ok(v) => {
                                let limit = if *axis == "x" { MAX_GRID_X } else { MAX_GRID_YZ };
                                if v > limit {
                                    errs.push(format!("{ectx}: {v} exceeds CUDA limit {limit} at var upper bounds"));
                                }
                            }
                            Err(err) => errs.push(format!("{ectx}: {err}")),
                        }
                        if let Ok(0) = e.eval(&vars_min) {
                            errs.push(format!("{ectx}: evaluates to 0 at var lower bounds"));
                        }
                    }
                    if let Some(e) = &k.shared_mem {
                        let ectx = format!("{ctx}: shared_mem");
                        check_expr(e, m, &mut used_vars, &mut errs, &ectx);
                        if let Ok(v) = e.eval(&vars_max) {
                            if v > MAX_DYN_SHARED_MEM {
                                errs.push(format!(
                                    "{ectx}: {v} bytes exceeds opt-in limit {MAX_DYN_SHARED_MEM} at var upper bounds"
                                ));
                            }
                        }
                    }
                    if let Some(cl) = &k.cluster {
                        let blocks: u64 = cl.iter().map(|&x| x as u64).product();
                        if cl.contains(&0) || blocks > MAX_CLUSTER_BLOCKS {
                            errs.push(format!(
                                "{ctx}: cluster {cl:?} has a zero dim or more than {MAX_CLUSTER_BLOCKS} blocks"
                            ));
                        } else {
                            for (axis, (e, &c)) in ["x", "y", "z"].iter().zip(k.grid.iter().zip(cl)) {
                                for (vars, at) in [(&vars_max, "upper"), (&vars_min, "lower")] {
                                    if let Ok(v) = e.eval(vars) {
                                        if v % c as u64 != 0 {
                                            errs.push(format!(
                                                "{ctx}: grid.{axis} = {v} at var {at} bounds is not a multiple of cluster.{axis} = {c}"
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if launch.is_extern() && launch.params_of(op).iter().any(|p| matches!(p, ParamType::Bytes(_))) {
                errs.push(format!("{ctx}: an extern launch takes pointers and scalars, not a pack"));
            }

            let params = launch.params_of(op);
            let args = launch.args_of(op);
            if args.len() != params.len() {
                errs.push(format!("{ctx}: takes {} params, got {} args", params.len(), args.len()));
                continue;
            }
            for (j, (arg, param)) in args.iter().zip(params).enumerate() {
                let actx = format!("{ctx}: arg #{j}");
                match arg {
                    LaunchArg::Param { param: i } => {
                        let Some(iface) = op.params.get(*i) else {
                            errs.push(format!(
                                "{actx}: interface param #{i} out of range ({} interface params)",
                                op.params.len()
                            ));
                            continue;
                        };
                        // Kind and dtype must match; a launch may not write
                        // through an interface param declared `in`.
                        let compatible = match (iface, param) {
                            (ParamType::Buf { dtype: a, .. }, ParamType::Buf { dtype: b, .. }) => a == b,
                            (ParamType::State { .. }, ParamType::State { .. }) => true,
                            (ParamType::Scalar(a), ParamType::Scalar(b)) => a == b,
                            _ => false,
                        };
                        if !compatible {
                            errs.push(format!(
                                "{actx}: interface param `{iface}` does not match launch param `{param}`"
                            ));
                            continue;
                        }
                        if let (Some(idir), Some(ldir)) = (iface.dir(), param.dir()) {
                            if matches!(ldir, Dir::Out | Dir::InOut) && idir == Dir::In {
                                errs.push(format!("{actx}: launch writes through interface `in` param #{i}"));
                            }
                            if matches!(ldir, Dir::In | Dir::InOut) && !iface_written[*i] {
                                errs.push(format!(
                                    "{actx}: interface `out` param #{i} read before any launch wrote it"
                                ));
                            }
                            if matches!(ldir, Dir::Out | Dir::InOut) {
                                iface_written[*i] = true;
                            }
                        }
                    }
                    LaunchArg::Scratch { scratch } => {
                        // Borrow the declaration's key: the wiring may be a
                        // defaulted (owned) list that dies with this launch.
                        let Some((scratch, sdecl)) = imp.scratch.get_key_value(scratch) else {
                            errs.push(format!("{actx}: unknown scratch `{scratch}`"));
                            continue;
                        };
                        scratch_used.insert(scratch.as_str());
                        let ParamType::Buf { dtype, dir } = param else {
                            errs.push(format!("{actx}: scratch `{scratch}` bound to non-buffer param `{param}`"));
                            continue;
                        };
                        if sdecl.dtype != *dtype {
                            errs.push(format!(
                                "{actx}: scratch `{scratch}` has dtype {} but param expects {}",
                                sdecl.dtype, dtype
                            ));
                        }
                        if matches!(dir, Dir::In | Dir::InOut) && !scratch_written.contains(scratch.as_str()) {
                            errs.push(format!("{actx}: scratch `{scratch}` is read before any launch wrote it"));
                        }
                        if matches!(dir, Dir::Out | Dir::InOut) {
                            scratch_written.insert(scratch.as_str());
                        }
                    }
                    LaunchArg::I32 { .. } if matches!(param, ParamType::Scalar(ScalarType::I32)) => {}
                    LaunchArg::I64 { .. } if matches!(param, ParamType::Scalar(ScalarType::I64)) => {}
                    LaunchArg::F32 { .. } if matches!(param, ParamType::Scalar(ScalarType::F32)) => {}
                    LaunchArg::U8 { .. } if matches!(param, ParamType::Scalar(ScalarType::U8)) => {}
                    LaunchArg::Rank { rank } => {
                        if !matches!(param, ParamType::Scalar(ScalarType::I32 | ScalarType::I64)) {
                            errs.push(format!("{actx}: a rank binds only to an i32 or i64 param, not `{param}`"));
                        }
                        group_ctx(rank, &mut errs, &mut used_groups, &actx);
                    }
                    LaunchArg::Pack { pack } => {
                        match param {
                            ParamType::Bytes(n) if *n == pack.size => {}
                            ParamType::Bytes(n) => {
                                errs.push(format!("{actx}: pack of {} bytes bound to a `bytes<{n}>` param", pack.size))
                            }
                            _ => errs.push(format!("{actx}: a pack binds only to a `bytes<n>` param, not `{param}`")),
                        }
                        for e in pack.check(|i| op.params.get(i).map(|p| p.size_bytes() as u32)) {
                            errs.push(format!("{actx}: {e}"));
                        }
                        for (k, f) in pack.fields.iter().enumerate() {
                            let fctx = format!("{actx}: field #{k}");
                            match &f.src {
                                FieldSrc::Param { param: i } => {
                                    let Some(iface) = op.params.get(*i) else {
                                        errs.push(format!(
                                            "{fctx}: interface param #{i} out of range ({} interface params)",
                                            op.params.len()
                                        ));
                                        continue;
                                    };
                                    // A pointer field is read or written per
                                    // the interface direction; the kernel is
                                    // trusted like a tensormap's.
                                    if matches!(iface.dir(), Some(Dir::Out | Dir::InOut)) {
                                        iface_written[*i] = true;
                                    }
                                }
                                FieldSrc::Scratch { scratch } => {
                                    let Some((scratch, _)) = imp.scratch.get_key_value(scratch) else {
                                        errs.push(format!("{fctx}: unknown scratch `{scratch}`"));
                                        continue;
                                    };
                                    scratch_used.insert(scratch.as_str());
                                    scratch_written.insert(scratch.as_str());
                                }
                                FieldSrc::Var { var } => {
                                    if m.vars.contains_key(var) {
                                        used_vars.insert(var.clone());
                                    } else {
                                        errs.push(format!("{fctx}: unknown var `{var}`"));
                                    }
                                }
                                FieldSrc::Expr { expr } => check_expr(expr, m, &mut used_vars, &mut errs, &fctx),
                                FieldSrc::Rank { rank } => {
                                    group_ctx(rank, &mut errs, &mut used_groups, &fctx);
                                }
                                FieldSrc::TensorMap { tensormap: t } => {
                                    let i = t.param;
                                    let Some(iface) = op.params.get(i) else {
                                        errs.push(format!(
                                            "{fctx}: interface param #{i} out of range ({} interface params)",
                                            op.params.len()
                                        ));
                                        continue;
                                    };
                                    let (ParamType::Buf { dir, .. } | ParamType::State { dir }) = iface else {
                                        errs.push(format!(
                                            "{fctx}: tensormap over interface param #{i} (`{iface}`), which is not a buffer or state"
                                        ));
                                        continue;
                                    };
                                    for e in t.check() {
                                        errs.push(format!("{fctx}: {e}"));
                                    }
                                    // The descriptor reads or writes per the
                                    // interface direction; the kernel is
                                    // trusted not to store through an `in`
                                    // buffer's descriptor.
                                    if matches!(dir, Dir::Out | Dir::InOut) {
                                        iface_written[i] = true;
                                    }
                                    if let Some(fp) = t.footprint() {
                                        op_tensormaps.entry(oname.as_str()).or_default().push((i, fp, fctx.clone()));
                                    }
                                }
                                FieldSrc::I32 { .. }
                                | FieldSrc::I64 { .. }
                                | FieldSrc::F32 { .. }
                                | FieldSrc::U8 { .. } => {}
                            }
                        }
                    }
                    arg => {
                        errs.push(format!("{actx}: {arg} does not match launch param `{param}`"));
                    }
                }
            }
        }

        for (i, (p, written)) in op.params.iter().zip(&iface_written).enumerate() {
            if !written {
                errs.push(format!("op `{oname}`: interface `{p}` param #{i} is never written by any launch"));
            }
        }
        for sname in imp.scratch.keys() {
            if !scratch_used.contains(sname.as_str()) {
                errs.push(format!("op `{oname}`: scratch `{sname}` is never used"));
            }
        }
    }

    // 7 + 8. programs
    if m.programs.is_empty() {
        errs.push("no programs declared".to_string());
    }
    let initially_written: BTreeSet<&str> = m
        .buffers
        .iter()
        .filter(|(_, b)| {
            // Carry buffers hold another program's output; whether that
            // program ran first is the caller's sequencing contract, so
            // per-program dataflow treats them as initially written.
            // Peer buffers are filled by the runtime when the group's
            // handles are imported, before any program runs.
            matches!(b.kind, BufferKind::Input | BufferKind::Weight | BufferKind::Carry | BufferKind::Peer)
        })
        .map(|(n, _)| n.as_str())
        .collect();
    let mut actually_written: BTreeSet<&str> = BTreeSet::new();

    for (pname, p) in &m.programs {
        if p.once && p.batch.is_some() {
            errs.push(format!(
                "program `{pname}`: `once` (run after load) and `batch` (driven per step) are exclusive"
            ));
        }
        if p.graph && p.batch.is_none() {
            errs.push(format!("program `{pname}`: `graph` (captured per call shape) needs a `batch` (a program a serving loop drives)"));
        }
        if let Some(batch) = &p.batch {
            if batch.groups == 0 {
                errs.push(format!("program `{pname}`: batch.groups must be >= 1"));
            }
            match &batch.rows {
                Dim::Const(0) => errs.push(format!("program `{pname}`: batch.rows must be >= 1")),
                Dim::Const(_) => {}
                Dim::Var(v) => {
                    if m.vars.contains_key(v) {
                        used_vars.insert(v.clone());
                    } else {
                        errs.push(format!("program `{pname}`: batch.rows names unknown var `{v}`"));
                    }
                }
            }
            if let Some(v) = &batch.span {
                if m.vars.contains_key(v) {
                    used_vars.insert(v.clone());
                } else {
                    errs.push(format!("program `{pname}`: batch.span names unknown var `{v}`"));
                }
            }
        }
        let mut written = initially_written.clone();
        for (i, c) in p.calls.iter().enumerate() {
            let ctx = match &c.label {
                Some(l) => format!("program `{pname}` call #{i} ({l})"),
                None => format!("program `{pname}` call #{i}"),
            };
            let Some(op) = m.ops.get(&c.op) else {
                errs.push(format!("{ctx}: unknown op `{}`", c.op));
                continue;
            };
            used_ops.insert(c.op.clone());
            let has_extern = op.imp.launches.iter().any(|l| l.is_extern());

            if c.args.len() != op.params.len() {
                errs.push(format!("{ctx}: op `{}` takes {} params, got {} args", c.op, op.params.len(), c.args.len()));
                continue;
            }
            for (j, (arg, param)) in c.args.iter().zip(&op.params).enumerate() {
                let actx = format!("{ctx}: arg #{j}");
                match (arg, param) {
                    (Arg::Buf { buf, offset }, ParamType::Buf { dtype, dir }) => {
                        used_buffers.insert(buf.clone());
                        let Some(b) = m.buffers.get(buf) else {
                            errs.push(format!("{actx}: unknown buffer `{buf}`"));
                            continue;
                        };
                        if b.dtype != *dtype {
                            errs.push(format!(
                                "{actx}: buffer `{buf}` has dtype {} but param expects {}",
                                b.dtype, dtype
                            ));
                        }
                        if *offset > 0 {
                            if offset % b.dtype.bytes() != 0 {
                                errs.push(format!(
                                    "{actx}: offset {offset} into buffer `{buf}` is not {}-aligned for {}",
                                    b.dtype.bytes(),
                                    b.dtype
                                ));
                            }
                            if let Some(&sz) = buf_sizes.get(buf.as_str()) {
                                if *offset >= sz {
                                    errs.push(format!(
                                        "{actx}: offset {offset} is outside buffer `{buf}` ({sz} bytes at var upper bounds)"
                                    ));
                                }
                            }
                        }
                        if matches!(dir, Dir::In | Dir::InOut) && !written.contains(buf.as_str()) {
                            errs.push(format!("{actx}: buffer `{buf}` is read before ever being written"));
                        }
                        if let (Some(tms), Some(&sz)) = (op_tensormaps.get(c.op.as_str()), buf_sizes.get(buf.as_str()))
                        {
                            for (_, fp, tctx) in tms.iter().filter(|(p, _, _)| *p == j) {
                                let avail = sz.saturating_sub(*offset);
                                if *fp > avail {
                                    errs.push(format!(
                                        "{actx}: {tctx} addresses {fp} bytes but buffer `{buf}` has {avail} bytes past offset {offset} at var upper bounds"
                                    ));
                                }
                            }
                        }
                        if b.kind == BufferKind::Peer && has_extern {
                            errs.push(format!(
                                "{actx}: peer buffer `{buf}` passed to op `{}`, which has an extern launch; runtime built-ins never receive peer memory",
                                c.op
                            ));
                        }
                        if matches!(dir, Dir::Out | Dir::InOut) {
                            if matches!(b.kind, BufferKind::Input | BufferKind::Weight | BufferKind::Peer) {
                                errs.push(format!("{actx}: op writes to read-only {} buffer `{buf}`", b.kind));
                            }
                            written.insert(buf.as_str());
                            actually_written.insert(buf.as_str());
                        }
                    }
                    (Arg::State { state, .. }, ParamType::State { .. }) => {
                        // state offsets are provider layout arithmetic over a
                        // runtime-scaled pool; there is no static bound to
                        // check them against.
                        used_states.insert(state.clone());
                        if !m.states.contains_key(state) {
                            errs.push(format!("{actx}: unknown state `{state}`"));
                        }
                    }
                    (Arg::Var { var }, ParamType::Scalar(st)) => {
                        used_vars.insert(var.clone());
                        match m.vars.get(var) {
                            None => errs.push(format!("{actx}: unknown var `{var}`")),
                            Some(v) => {
                                if *st == ScalarType::F32 {
                                    errs.push(format!("{actx}: var `{var}` cannot bind to an f32 param"));
                                } else if !scalar_fits(*st, v.max) {
                                    errs.push(format!("{actx}: var `{var}` max {} exceeds {st} range", v.max));
                                }
                            }
                        }
                    }
                    (Arg::Expr { expr }, ParamType::Scalar(st)) => {
                        check_expr(expr, m, &mut used_vars, &mut errs, &actx);
                        if *st == ScalarType::F32 {
                            errs.push(format!("{actx}: an expression cannot bind to an f32 param"));
                        } else if let Ok(v) = expr.eval(&vars_max) {
                            if !scalar_fits(*st, v) {
                                errs.push(format!(
                                    "{actx}: expression reaches {v} at var upper bounds, exceeding {st} range"
                                ));
                            }
                        }
                    }
                    (Arg::I32 { .. }, ParamType::Scalar(ScalarType::I32))
                    | (Arg::I64 { .. }, ParamType::Scalar(ScalarType::I64))
                    | (Arg::F32 { .. }, ParamType::Scalar(ScalarType::F32))
                    | (Arg::U8 { .. }, ParamType::Scalar(ScalarType::U8)) => {}
                    (Arg::Rank { rank }, ParamType::Scalar(ScalarType::I32 | ScalarType::I64)) => {
                        group_ctx(rank, &mut errs, &mut used_groups, &actx);
                    }
                    (arg, param) => {
                        errs.push(format!("{actx}: {arg} does not match param `{param}`"));
                    }
                }
            }
        }
    }
    // Outputs and carries must be produced by *some* program — a
    // prefill-style program whose only effect is state mutation legitimately
    // writes none itself.
    for (bname, b) in &m.buffers {
        if matches!(b.kind, BufferKind::Output | BufferKind::Carry) && !actually_written.contains(bname.as_str()) {
            errs.push(format!("{} buffer `{bname}` is never written by any program", b.kind));
        }
    }

    // 9. unused declarations
    for name in m.buffers.keys() {
        if !used_buffers.contains(name) {
            errs.push(format!("buffer `{name}` is never used by any program"));
        }
    }
    for name in m.ops.keys() {
        if !used_ops.contains(name) {
            errs.push(format!("op `{name}` is never called"));
        }
    }
    for name in m.modules.keys() {
        if !used_modules.contains(name) {
            errs.push(format!("module `{name}` is never launched"));
        }
    }
    for name in m.states.keys() {
        if !used_states.contains(name) {
            errs.push(format!("state `{name}` is never used by any program"));
        }
    }
    for name in m.vars.keys() {
        if !used_vars.contains(name) {
            errs.push(format!("var `{name}` is never used"));
        }
    }
    if let Some(t) = &m.topology {
        for name in t.groups.keys() {
            if !used_groups.contains(name) {
                errs.push(format!("topology group `{name}` is never used (no peer buffer or rank arg names it)"));
            }
        }
    }

    errs
}

fn scalar_fits(st: ScalarType, v: u64) -> bool {
    match st {
        ScalarType::I32 => v <= i32::MAX as u64,
        ScalarType::I64 => v <= i64::MAX as u64,
        ScalarType::U8 => v <= u8::MAX as u64,
        ScalarType::F32 => false,
    }
}

/// A domain is a prior on contents; the verifier only proves it is
/// well-formed against the declaration it decorates (never that any kernel
/// honours it).
#[allow(clippy::too_many_arguments)]
fn check_domain(
    name: &str,
    b: &Buffer,
    d: &Domain,
    m: &Manifest,
    vars_max: &BTreeMap<String, u64>,
    vars_min: &BTreeMap<String, u64>,
    used_vars: &mut BTreeSet<String>,
    errs: &mut Vec<String>,
) {
    let ctx = format!("buffer `{name}` domain");
    let is_float = matches!(b.dtype, DType::Bf16 | DType::F16 | DType::F32 | DType::Fp8E4m3 | DType::Fp8E8m0);
    if d.index_into.is_some() && (d.min.is_some() || d.max.is_some()) {
        errs.push(format!("{ctx}: `index_into` and `min`/`max` are mutually exclusive"));
    }
    if d.index_into.is_none() && d.min.is_none() && d.max.is_none() && !d.monotone {
        errs.push(format!("{ctx}: empty (declare bounds, `index_into`, or `monotone`)"));
    }
    if d.stride == 0 {
        errs.push(format!("{ctx}: `stride` must be > 0"));
    }
    if d.stride > 1 && d.index_into.is_none() {
        errs.push(format!("{ctx}: `stride` only applies with `index_into`"));
    }
    if let Some(t) = &d.index_into {
        if is_float {
            errs.push(format!("{ctx}: a {} buffer cannot index anything", b.dtype));
        }
        match (m.buffers.contains_key(t), m.states.contains_key(t)) {
            (false, false) => errs.push(format!("{ctx}: `index_into` unknown buffer/state `{t}`")),
            (true, true) => errs.push(format!("{ctx}: `index_into` `{t}` is both a buffer and a state")),
            (false, true) if m.states[t].is_per_seq() && !m.states[t].bytes_per_seq.is_multiple_of(d.stride.max(1)) => {
                errs.push(format!(
                    "{ctx}: `index_into` per-sequence state `{t}` in lines of {} bytes, which do not divide its {} bytes per sequence",
                    d.stride, m.states[t].bytes_per_seq
                ))
            }
            (true, false) if t == name => errs.push(format!("{ctx}: a buffer cannot index itself")),
            _ => {}
        }
    }
    if d.monotone && b.shape.len() != 1 {
        errs.push(format!("{ctx}: `monotone` requires a one-dimensional buffer"));
    }
    for (which, bound) in [("min", &d.min), ("max", &d.max)] {
        let Some(bound) = bound else { continue };
        match bound {
            Bound::Float(_) if !is_float => {
                errs.push(format!("{ctx}: float `{which}` on a {} buffer", b.dtype));
            }
            Bound::Expr(e) => check_expr(e, m, used_vars, errs, &format!("{ctx}: `{which}`")),
            _ => {}
        }
    }
    if let (Some(lo), Some(hi)) = (&d.min, &d.max) {
        // Must hold at every var value the bounds can take; both corners
        // suffice for the monotone expression set.
        for vars in [vars_min, vars_max] {
            if let (Ok(lo), Ok(hi)) = (lo.eval(vars), hi.eval(vars)) {
                if lo > hi {
                    errs.push(format!("{ctx}: min {lo} > max {hi}"));
                    break;
                }
            }
        }
    }
}

fn check_expr(e: &Expr, m: &Manifest, used_vars: &mut BTreeSet<String>, errs: &mut Vec<String>, ctx: &str) {
    match e {
        Expr::Const(_) => {}
        Expr::Var(var) => {
            if m.vars.contains_key(var) {
                used_vars.insert(var.clone());
            } else {
                errs.push(format!("{ctx}: unknown var `{var}`"));
            }
        }
        Expr::CeilDiv { ceil_div: (inner, c) } => {
            if *c == 0 {
                errs.push(format!("{ctx}: division by zero"));
            }
            check_expr(inner, m, used_vars, errs, ctx);
        }
        Expr::Mul { mul: (inner, c) } => {
            if *c == 0 {
                errs.push(format!("{ctx}: multiplication by constant zero"));
            }
            check_expr(inner, m, used_vars, errs, ctx);
        }
    }
}
