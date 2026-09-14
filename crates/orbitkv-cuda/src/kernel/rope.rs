//! Fused RoPE (rotary position embedding) — interleaved-pair convention.
//!
//! Replaces flux2's 6-op RoPE chain (split / slice / squeeze / neg / concat /
//! merge_dims / 4× cast / mul / add) with a single kernel launch per call.
//! ~120 RoPE calls per forward pass at full DiT depth.
//!
//! Convention: `repeat_interleave_real=True` (Flux 2 / diffusers), so adjacent
//! dim pairs rotate together. For an input `[a0, b0, a1, b1, ...]` and per-
//! position `(cos, sin)`, the output is
//!   `out[2j]   = x[2j]   * cos[2j]   - x[2j+1] * sin[2j]`
//!   `out[2j+1] = x[2j+1] * cos[2j+1] + x[2j]   * sin[2j+1]`
//!
//! Layout: x `(S, H, D)`, cos/sin `(S, D)` (broadcast across H).

use std::{fmt::Display, sync::Arc};

use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use orbitkv_compiler::{
    dtype::DType,
    egglog_utils::list_to_egglog,
    op::{CustomOp, HLIROp, LLIROp},
    prelude::{FxHashMap, FxHashSet, GraphTensor, NodeIndex, ShapeTracker, Symbol},
    shape::Expression,
};

use crate::compile_module_image_for_current_device;
use crate::kernel::KernelOp;

#[derive(Debug, Clone)]
pub struct RoPEKernel {
    pub s: usize,
    pub h: usize,
    pub d: usize,
}

const TPB: usize = 64;

impl KernelOp for RoPEKernel {
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
        compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
        String,
        (Expression, Expression, Expression),
        (Expression, Expression, Expression),
        Expression,
        FxHashMap<Symbol, CudaSlice<u8>>,
    ) {
        let s = self.s;
        let h = self.h;
        let d = self.d;
        assert!(d.is_multiple_of(2), "RoPE head_dim must be even");
        let kernel = format!(
            include_str!("rope/interleaved.cu.in"),
            TPB = TPB,
            d = d,
            h = h,
            s = s,
        );

        let (module, func) = if let Some((m, f)) = compile_cache.get(&kernel) {
            (m.clone(), f.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("rope_kernel").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        (
            func,
            module,
            kernel,
            (
                Expression::from(s * h),
                Expression::from(1usize),
                Expression::from(1usize),
            ),
            (
                Expression::from(TPB),
                Expression::from(1usize),
                Expression::from(1usize),
            ),
            Expression::from(0usize),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        Expression::from(self.s * self.h * self.d)
    }

    fn output_bytes(&self) -> Expression {
        self.output_size() * 4
    }

    fn output_dtype(&self) -> DType {
        DType::F32
    }

    fn bytes_loaded(&self) -> Expression {
        // x: full (S,H,D); cos/sin: (S,D) read H times each but cached.
        Expression::from(self.s * self.h * self.d * 4 + self.s * self.d * 4 * 2)
    }

    fn bytes_stored(&self) -> Expression {
        self.output_size() * 4
    }

    fn flops(&self) -> Expression {
        // 4 per output element (mul, neg/load, mul, add).
        Expression::from(self.s * self.h * self.d * 4)
    }

    fn kernel_name(&self) -> &'static str {
        "RoPE"
    }
}

#[derive(Debug, Clone)]
pub struct RoPECustom(pub RoPEKernel);

impl CustomOp for RoPECustom {
    fn to_llir_op(&self) -> LLIROp {
        LLIROp::new::<dyn KernelOp>(Box::new(self.0.clone()) as Box<dyn KernelOp>)
    }
}

/// Apply RoPE: `x` shape `(S, H, D)` F32, `cos`/`sin` shape `(S, D)` F32.
/// Returns `(S, H, D)` F32.
pub fn apply_rope(x: GraphTensor, cos: GraphTensor, sin: GraphTensor) -> GraphTensor {
    assert_eq!(x.dtype, DType::F32, "RoPE x must be F32");
    let cos = if cos.dtype == DType::F32 {
        cos
    } else {
        cos.cast(DType::F32)
    };
    let sin = if sin.dtype == DType::F32 {
        sin
    } else {
        sin.cast(DType::F32)
    };
    let x_dims = x.dims();
    assert_eq!(x_dims.len(), 3, "RoPE x must be 3-D (S, H, D)");
    let s = x_dims[0].to_usize().expect("RoPE: S must be static");
    let h = x_dims[1].to_usize().expect("RoPE: H must be static");
    let d = x_dims[2].to_usize().expect("RoPE: D must be static");
    let cos_dims = cos.dims();
    let sin_dims = sin.dims();
    assert_eq!(cos_dims.len(), 2, "RoPE cos must be 2-D (S, D)");
    assert_eq!(sin_dims.len(), 2, "RoPE sin must be 2-D (S, D)");
    assert_eq!(cos_dims[0].to_usize().unwrap(), s, "RoPE cos S mismatch");
    assert_eq!(cos_dims[1].to_usize().unwrap(), d, "RoPE cos D mismatch");
    assert_eq!(sin_dims[0].to_usize().unwrap(), s, "RoPE sin S mismatch");
    assert_eq!(sin_dims[1].to_usize().unwrap(), d, "RoPE sin D mismatch");

    let kern = RoPEKernel { s, h, d };
    let cx = unsafe { &mut *x.graph_ref };
    cx.custom_op(RoPECustom(kern), vec![x, cos, sin], (s, h, d), DType::F32)
}

// ═══════════════════════════════════════════════════════════
// Half-rotation RoPE (Llama 3 convention), dtype-aware, dynamic S.
//
// Rotates the two halves of each head: for j in 0..D/2
//   out[j]       = x[j]       * cos[j] - x[j + D/2] * sin[j]
//   out[j + D/2] = x[j + D/2] * cos[j] + x[j]       * sin[j]
//
// The input is read from a row of a (possibly wider) projection output via
// `pitch` (row stride in elements) and `offset` (column offset), so q and k
// can be roped straight out of a fused QKV GEMM without materializing
// slices. cos/sin are (S, D/2) F32; x/out are `dtype` (F32 or 16-bit, math
// in F32). S is dynamic: the kernel derives everything from blockIdx, so
// only the grid expression carries the dyn dim.
// ═══════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct RoPEHalfKernel {
    pub s: Expression,
    pub h: usize,
    pub d: usize,
    /// Input row stride in elements (e.g. 6144 for a fused QKV row).
    pub pitch: usize,
    /// Column offset of this head group within the input row.
    pub offset: usize,
    /// Source projection dtype.
    pub dtype: DType,
    /// Materialized rotation dtype. Keeping this distinct permits the common
    /// F32-projection → BF16-attention boundary to round in this kernel.
    pub output_dtype: DType,
}

impl Default for RoPEHalfKernel {
    fn default() -> Self {
        Self {
            s: 1.into(),
            h: 1,
            d: 2,
            pitch: 2,
            offset: 0,
            dtype: DType::F32,
            output_dtype: DType::F32,
        }
    }
}

impl Display for RoPEHalfKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RoPEHalf")
    }
}

impl HLIROp for RoPEHalfKernel {
    fn to_egglog(&self, inputs: &[(NodeIndex, String)]) -> String {
        assert_eq!(inputs.len(), 3, "RoPEHalf has x, cos, and sin inputs");
        format!(
            "(Op (KernelRoPEHalf {} {} {} {} {} {} ({:?}) ({:?})) {})",
            self.s.to_egglog(),
            Expression::from(self.h).to_egglog(),
            Expression::from(self.d).to_egglog(),
            Expression::from(self.h * self.d).to_egglog(),
            Expression::from(self.pitch).to_egglog(),
            Expression::from(self.offset).to_egglog(),
            self.dtype,
            self.output_dtype,
            list_to_egglog(&[&inputs[0].1, &inputs[1].1, &inputs[2].1], "ICons", "INil"),
        )
    }
}

impl EgglogOp for RoPEHalfKernel {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelRoPEHalf",
            &[
                ("s", EXPRESSION),
                ("h", EXPRESSION),
                ("d", EXPRESSION),
                ("out_width", EXPRESSION),
                ("pitch", EXPRESSION),
                ("offset", EXPRESSION),
                ("input_dtype", DTYPE),
                ("output_dtype", DTYPE),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        3
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![Rule::raw(include_str!("rope/interleaved_rewrite.egg"))]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        _list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        let s = extract_expr(egraph, kind_children[0], expr_cache).unwrap();
        let h = extract_expr(egraph, kind_children[1], expr_cache)
            .unwrap()
            .to_usize()
            .expect("RoPEHalf h must be static");
        let d = extract_expr(egraph, kind_children[2], expr_cache)
            .unwrap()
            .to_usize()
            .expect("RoPEHalf d must be static");
        let pitch = extract_expr(egraph, kind_children[4], expr_cache)
            .unwrap()
            .to_usize()
            .expect("RoPEHalf pitch must be static");
        let offset = extract_expr(egraph, kind_children[5], expr_cache)
            .unwrap()
            .to_usize()
            .expect("RoPEHalf offset must be static");
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                s,
                h,
                d,
                pitch,
                offset,
                dtype: extract_dtype(egraph, kind_children[6]),
                output_dtype: extract_dtype(egraph, kind_children[7]),
            }) as Box<dyn KernelOp>),
            input_enodes,
        )
    }
}

impl KernelOp for RoPEHalfKernel {
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
        compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
        String,
        (Expression, Expression, Expression),
        (Expression, Expression, Expression),
        Expression,
        FxHashMap<Symbol, CudaSlice<u8>>,
    ) {
        let h = self.h;
        let d = self.d;
        let pitch = self.pitch;
        let offset = self.offset;
        assert!(d.is_multiple_of(2), "RoPE head_dim must be even");
        let half = d / 2;
        let in_ty = crate::cuda_dtype(self.dtype);
        let out_ty = crate::cuda_dtype(self.output_dtype);
        let includes = crate::kernel::hlir::dtype_includes(&[self.dtype, self.output_dtype]);
        let kernel = format!(
            include_str!("rope/half_split.cu.in"),
            TPB = TPB,
            d = d,
            h = h,
            half = half,
            in_ty = in_ty,
            includes = includes,
            offset = offset,
            out_ty = out_ty,
            pitch = pitch,
        );

        let (module, func) = if let Some((m, f)) = compile_cache.get(&kernel) {
            (m.clone(), f.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("rope_half_kernel").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        (
            func,
            module,
            kernel,
            (
                self.s * self.h,
                Expression::from(1usize),
                Expression::from(1usize),
            ),
            (
                Expression::from(TPB),
                Expression::from(1usize),
                Expression::from(1usize),
            ),
            Expression::from(0usize),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.s * self.h * self.d
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.output_dtype.bits()).ceil_div(8)
    }

    fn output_dtype(&self) -> DType {
        self.output_dtype
    }

    fn bytes_loaded(&self) -> Expression {
        (self.s * self.h * self.d * self.dtype.bits()).ceil_div(8) + self.s * self.d * 4
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.s * self.h * self.d * 4
    }

    fn kernel_name(&self) -> &'static str {
        "RoPEHalf"
    }
}

#[derive(Debug, Clone)]
pub struct RoPEHalfCustom(pub RoPEHalfKernel);

impl CustomOp for RoPEHalfCustom {
    fn to_llir_op(&self) -> LLIROp {
        LLIROp::new::<dyn KernelOp>(Box::new(self.0.clone()) as Box<dyn KernelOp>)
    }
}

/// Half-rotation RoPE over a head group inside a projection output.
///
/// `x` is `(S, pitch)` (e.g. a fused QKV output), `offset`/`h`/`d` select the
/// head group, `cos`/`sin` are `(S, d/2)` F32. Returns contiguous `(S, h*d)`
/// in `x`'s dtype. One kernel replaces the ~10-op rope chain per projection.
pub fn apply_rope_half(
    x: GraphTensor,
    offset: usize,
    h: usize,
    d: usize,
    cos: GraphTensor,
    sin: GraphTensor,
) -> GraphTensor {
    apply_rope_half_as(x, offset, h, d, cos, sin, x.dtype)
}

/// [`apply_rope_half`] with an explicit output dtype. Rotation math remains
/// F32; only the final store is converted.
pub fn apply_rope_half_as(
    x: GraphTensor,
    offset: usize,
    h: usize,
    d: usize,
    cos: GraphTensor,
    sin: GraphTensor,
    output_dtype: DType,
) -> GraphTensor {
    assert_eq!(cos.dtype, DType::F32, "RoPE cos must be F32");
    assert_eq!(sin.dtype, DType::F32, "RoPE sin must be F32");
    let x_dims = x.dims();
    assert_eq!(x_dims.len(), 2, "RoPE x must be 2-D (S, pitch)");
    let s = x_dims[0];
    let pitch = x_dims[1].to_usize().expect("RoPE: pitch must be static");
    assert!(offset + h * d <= pitch, "RoPE head group exceeds row pitch");

    let kern = RoPEHalfKernel {
        s,
        h,
        d,
        pitch,
        offset,
        dtype: x.dtype,
        output_dtype,
    };
    let cx = unsafe { &mut *x.graph_ref };
    let id = cx.add_op(kern, &[x.id, cos.id, sin.id]);
    GraphTensor::from_id(
        id,
        ShapeTracker::new_with_element_bits((s, h * d), output_dtype.bits()),
        cx,
        output_dtype,
    )
}

// ═══════════════════════════════════════════════════════════
// Fused RoPE + KV-cache scatter
//
// The fused kernels below are selectable egglog alternatives to materialized
// RoPE followed by in-place scatter. They write rotated values straight to the
// cache slots, removing one launch and one `(s, kv_dim)` temporary when measured
// search selects them. The materialized form remains in the same e-class.
// ═══════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct RoPEScatterKernel {
    pub rope: RoPEHalfKernel,
    /// Total element count of the scatter destination (the cache pool).
    dest_size: Expression,
    /// Flattened scatter-index expression over `z`, where `z` is the element
    /// position in the rope output's contiguous (s, h·d) layout.
    idx_flat: Expression,
}

impl Default for RoPEScatterKernel {
    fn default() -> Self {
        Self {
            rope: RoPEHalfKernel::default(),
            dest_size: 1.into(),
            idx_flat: Expression::from('z'),
        }
    }
}

impl EgglogOp for RoPEScatterKernel {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelRoPEHalfScatter",
            &[
                ("s", EXPRESSION),
                ("h", EXPRESSION),
                ("d", EXPRESSION),
                ("out_width", EXPRESSION),
                ("pitch", EXPRESSION),
                ("offset", EXPRESSION),
                ("dest_shape", ELIST),
                ("index_shape", ELIST),
                ("index_strides", ELIST),
                ("input_dtype", DTYPE),
                ("output_dtype", DTYPE),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        5
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![Rule::raw(include_str!("rope/half_split_rewrite.egg"))]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        let s = extract_expr(egraph, kind_children[0], expr_cache).unwrap();
        let h = extract_expr(egraph, kind_children[1], expr_cache)
            .unwrap()
            .to_usize()
            .expect("RoPEHalf h must be static");
        let d = extract_expr(egraph, kind_children[2], expr_cache)
            .unwrap()
            .to_usize()
            .expect("RoPEHalf d must be static");
        let pitch = extract_expr(egraph, kind_children[4], expr_cache)
            .unwrap()
            .to_usize()
            .expect("RoPEHalf pitch must be static");
        let offset = extract_expr(egraph, kind_children[5], expr_cache)
            .unwrap()
            .to_usize()
            .expect("RoPEHalf offset must be static");
        let dest_shape =
            extract_expr_list(egraph, kind_children[6], list_cache, expr_cache).unwrap();
        let index_shape =
            extract_expr_list(egraph, kind_children[7], list_cache, expr_cache).unwrap();
        let index_strides =
            extract_expr_list(egraph, kind_children[8], list_cache, expr_cache).unwrap();
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                rope: RoPEHalfKernel {
                    s,
                    h,
                    d,
                    pitch,
                    offset,
                    dtype: extract_dtype(egraph, kind_children[9]),
                    output_dtype: extract_dtype(egraph, kind_children[10]),
                },
                dest_size: dest_shape.into_iter().product(),
                idx_flat: orbitkv_compiler::shape::flatten_strides(&index_shape, &index_strides),
            }) as Box<dyn KernelOp>),
            input_enodes,
        )
    }
}

impl KernelOp for RoPEScatterKernel {
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
        compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
        String,
        (Expression, Expression, Expression),
        (Expression, Expression, Expression),
        Expression,
        FxHashMap<Symbol, CudaSlice<u8>>,
    ) {
        let h = self.rope.h;
        let d = self.rope.d;
        let pitch = self.rope.pitch;
        let offset = self.rope.offset;
        let half = d / 2;
        let in_ty = crate::cuda_dtype(self.rope.dtype);
        let out_ty = crate::cuda_dtype(self.rope.output_dtype);
        let includes =
            crate::kernel::hlir::dtype_includes(&[self.rope.dtype, self.rope.output_dtype]);

        let vars: FxHashSet<Symbol> = self
            .idx_flat
            .dyn_vars()
            .into_iter()
            .chain(self.dest_size.dyn_vars())
            .collect();
        let (dyn_defines, _sorted) = crate::kernel::hlir::generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let idx_expr = self.idx_flat.to_kernel();
        let dest_n = self.dest_size.to_kernel();

        let kernel = format!(
            include_str!("rope/scatter.cu.in"),
            TPB = TPB,
            d = d,
            dest_n = dest_n,
            dyn_defines = dyn_defines,
            dyn_dims_param = dyn_dims_param,
            h = h,
            half = half,
            idx_expr = idx_expr,
            in_ty = in_ty,
            includes = includes,
            offset = offset,
            out_ty = out_ty,
            pitch = pitch,
        );

        let (module, func) = if let Some((m, f)) = compile_cache.get(&kernel) {
            (m.clone(), f.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("rope_scatter_kernel").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        (
            func,
            module,
            kernel,
            (
                self.rope.s * self.rope.h,
                Expression::from(1usize),
                Expression::from(1usize),
            ),
            (
                Expression::from(TPB),
                Expression::from(1usize),
                Expression::from(1usize),
            ),
            Expression::from(0usize),
            FxHashMap::default(),
        )
    }

    fn build_params(
        &self,
        _stream: &Arc<CudaStream>,
        _output_ptr: u64,
        input_ptrs: &[u64],
        _internal_bufs: &[cudarc::driver::CudaSlice<u8>],
        dyn_dims_ptr: u64,
    ) -> Vec<u64> {
        // rope_scatter_kernel: (dest, indexes, x, cos, sin [, dyn_dims]).
        // Writes in place through dest (input 0), not through output_ptr.
        let mut params = vec![
            input_ptrs[0],
            input_ptrs[1],
            input_ptrs[2],
            input_ptrs[3],
            input_ptrs[4],
        ];
        if dyn_dims_ptr != 0 {
            params.push(dyn_dims_ptr);
        }
        params
    }

    fn output_aliases_input(&self) -> Option<usize> {
        Some(0)
    }

    fn mutates_aliased_input(&self) -> bool {
        true
    }

    fn collect_dyn_vars_into(&self, vars: &mut FxHashSet<Symbol>) {
        self.idx_flat.collect_dyn_vars_into(vars);
        self.dest_size.collect_dyn_vars_into(vars);
        self.rope.s.collect_dyn_vars_into(vars);
    }

    fn output_size(&self) -> Expression {
        self.dest_size
    }

    fn output_bytes(&self) -> Expression {
        (self.dest_size * self.rope.output_dtype.bits()).ceil_div(8)
    }

    fn output_dtype(&self) -> DType {
        self.rope.output_dtype
    }

    fn bytes_loaded(&self) -> Expression {
        let rotated = self.rope.s * self.rope.h * self.rope.d;
        (rotated * self.rope.dtype.bits()).ceil_div(8) + rotated * 4 + self.rope.s * self.rope.d * 4
    }

    fn bytes_stored(&self) -> Expression {
        (self.rope.s * self.rope.h * self.rope.d * self.rope.output_dtype.bits()).ceil_div(8)
    }

    fn flops(&self) -> Expression {
        self.rope.s * self.rope.h * self.rope.d * 4
    }

    fn kernel_name(&self) -> &'static str {
        "RoPEScatter"
    }
}

// ═══════════════════════════════════════════════════════════
// KernelRoPE — egglog-matched fused rotary (half convention, bf16).
//
// Matches the full HLIR rotary chain the qwen/gemma models spell:
// inv-freq (iota·2 → cast → ×(1/hd) → ×ln(theta) → ×log2(e) → exp2 → recip),
// angles (cast(pos) × inv_freq → sum), sin / cos-as-sin(π/2−x), bf16 casts,
// the x0 strided view + x1 offset-slice gather, the rotation arithmetic, and
// the concat (2 clamped gathers + 2 mask iotas). The rule roots at the
// concat Add eclass — the last materialized tensor of the chain; the
// trailing transpose+merge is a view applied by consumers, so the fused
// kernel writes the same (heads, seq, hd) buffer the concat produces and
// every consumer works unchanged. One kernel replaces ~13 launches per rope
// call; the angle chain (shared by the q and k calls within a layer)
// becomes dead.
//
// The kernel mirrors the decomposed numerics exactly: angle math in F32 with
// the same op order/spellings (exp2f, 1.0f/x, sinf, sin(−x+π/2)), and a bf16
// rounding at every decomposed op boundary (cos/sin casts, each mul, the −1
// mul). The concat's ×{0,1} masks and +0 are value-preserving and elided.
// ═══════════════════════════════════════════════════════════

#[derive(Default, Debug, Clone)]
pub struct KernelRoPE {
    /// `(heads, seq, head_dim)` — seq may be dynamic.
    out_shape: Vec<Expression>,
    /// Row stride of the x input in elements (the projection width).
    width: usize,
    ln_theta: f64,
    inv_hd: f64,
}

impl KernelRoPE {
    /// Construct the shared RoPE descriptor for a full-CUDA fused consumer.
    #[doc(hidden)]
    pub fn from_parts(
        out_shape: Vec<Expression>,
        width: usize,
        ln_theta: f64,
        inv_hd: f64,
    ) -> Self {
        Self {
            out_shape,
            width,
            ln_theta,
            inv_hd,
        }
    }

    #[doc(hidden)]
    pub fn out_shape(&self) -> &[Expression] {
        &self.out_shape
    }

    #[doc(hidden)]
    pub fn width(&self) -> usize {
        self.width
    }

    #[doc(hidden)]
    pub fn ln_theta(&self) -> f64 {
        self.ln_theta
    }

    #[doc(hidden)]
    pub fn inv_hd(&self) -> f64 {
        self.inv_hd
    }
}

use orbitkv_compiler::{
    egglog_utils::{
        api::{Rule, SortDef, sort},
        base::{DTYPE, ELIST, EXPRESSION, F64, OP_KIND},
        extract_dtype, extract_expr, extract_expr_list,
    },
    op::EgglogOp,
    prelude::{ENodeId, SerializedEGraph},
};

impl EgglogOp for KernelRoPE {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelRoPE",
            &[
                ("out_shape", ELIST),
                ("out_width", EXPRESSION),
                ("head_stride", EXPRESSION),
                ("width", EXPRESSION),
                ("ln_theta", F64),
                ("inv_hd", F64),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        2
    }

    fn rewrites(&self) -> Vec<Rule> {
        // Two-stage match via an intermediate relation. A single ~45-atom
        // join blew up egglog's query planner on real graphs (4m56s on the
        // 279-node mini layer); splitting at the angle chain keeps each join
        // anchored: stage 1 is pinned by the Constant atoms (log2e, pi/2,
        // ln theta chain) and emits one `rope_angles` fact per rope site,
        // stage 2 joins the rotation/concat with ?cosb/?sinb already bound,
        // which makes every Mul atom selective.
        let angle_stage: &str = include_str!("rope/angle_witnesses.egg");
        let rotation_stage: &str = include_str!("rope/rotation_witnesses.egg");

        // Stage-2 conditions in dependency order, segmented for readability.
        let segments: Vec<&str> = vec![
            "
                    (rope_rotated ?x0out ?x1out ?x ?pos ?e_hd ?e_hd2 ?e_w ?e_seq ?ln_theta ?inv_hd)",
            "
                    ; root: the concat Add - (heads, seq, hd) contiguous
                    (= ?x0g (Op (Gather ?g2_osh ?g2_ostr ?g2_dsh ?g2_dstr)
                        (ICons ?c0idx (ICons ?x0out (INil)))))
                    (= ?x0m (Op (Mul ?m6_sh ?m6_a ?m6_b ?m6_o)
                        (ICons ?x0g (ICons ?mk0b (INil)))))
                    (= ?cat (Op (Add ?a3_sh ?a3_a ?a3_b ?a3_o)
                        (ICons ?x0m (ICons ?x1m (INil)))))
                    (= ?a3_sh (ECons ?heads (ECons ?seqd (ECons ?hdd (ENil)))))",
            "
                    ; concat half 0 pins
                    (= ?c0idx (Op (Iota
                        (MAdd (MAdd (MMin (MMod (MIter) ?e_hd) ?e_hdm1)
                                    (MMul (MMod (MDiv (MIter) ?e_hd) ?e_seq) ?e_hd2))
                              (MMul (MDiv (MIter) ?e_hs) ?e_ch))
                        ?cat_range) (INil)))
                    (= ?mk0b (Op (Cast ?k0_size (Bf16)) (ICons ?mk0 (INil))))
                    (= ?mk0 (Op (Iota (MLt (MMod (MIter) ?e_hd) ?e_hd2) ?cat_range) (INil)))",
            "
                    ; concat half 1
                    (= ?x1g (Op (Gather ?g3_osh ?g3_ostr ?g3_dsh ?g3_dstr)
                        (ICons ?c1idx (ICons ?x1out (INil)))))
                    (= ?x1m (Op (Mul ?m7_sh ?m7_a ?m7_b ?m7_o)
                        (ICons ?x1g (ICons ?mk1b (INil)))))
                    (= ?c1idx (Op (Iota
                        (MAdd (MAdd (MMax (MSub (MMod (MIter) ?e_hd) ?e_hd2) (MNum 0))
                                    (MMul (MMod (MDiv (MIter) ?e_hd) ?e_seq) ?e_hd2))
                              (MMul (MDiv (MIter) ?e_hs) ?e_ch))
                        ?cat_range) (INil)))
                    (= ?mk1b (Op (Cast ?k1_size (Bf16)) (ICons ?mk1 (INil))))
                    (= ?mk1 (Op (Iota (MGte (MMod (MIter) ?e_hd) ?e_hd2) ?cat_range) (INil)))",
            "
                    ; layout consistency
                    (= ?hdd ?e_hd)
                    (= ?seqd ?e_seq)
                    (= ?e_hd2 (MNum ?e_hd2_n))
                    (= ?e_hdm1 (MNum ?e_hdm1_n))
                    (= ?e_hdm1_n (- ?e_hd2_n 1))
                    (rope_row_dims ?e_hd ?e_hd2 ?e_seq ?e_hs ?e_ch)
                    (rope_tensor_range ?heads ?e_hs ?cat_range)",
        ];

        let concat_rule = format!(
            include_str!("rope/concat_rewrite.egg.in"),
            segments.join("\n")
        );

        // `GraphTensor::pad_with` selects between the gathered value and an
        // exact typed zero through an interleaved scatter/gather.  This avoids
        // the IEEE-invalid `0 * NaN` arithmetic mask used by the historical
        // padding spelling.  Match both safe padded halves independently so
        // the final RoPE join stays selective on large model graphs.
        //
        // The scatter/gather constraints below are the semantic proof for the
        // normalization: even slots are overwritten with zero, odd slots with
        // the gathered half, and the final gather chooses `2*z + mask(z)`.
        // Consequently the left relation contributes x0 only for i < hd/2,
        // the right relation contributes x1 only for i >= hd/2, and their Add
        // is exactly concat(x0, x1), including NaN and infinity behavior.
        let safe_concat_stage: &str = include_str!("rope/concat_witnesses.egg");
        vec![Rule::raw(format!(
            "{angle_stage}\n{rotation_stage}\n{concat_rule}\n{safe_concat_stage}"
        ))]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        let out_shape =
            extract_expr_list(egraph, kind_children[0], list_cache, expr_cache).unwrap();
        let width = extract_expr(egraph, kind_children[3], expr_cache)
            .unwrap()
            .to_usize()
            .expect("RoPE width must be static");
        let ln_theta: f64 = egraph.enodes[kind_children[4]]
            .0
            .replace('"', "")
            .parse()
            .unwrap();
        let inv_hd: f64 = egraph.enodes[kind_children[5]]
            .0
            .replace('"', "")
            .parse()
            .unwrap();
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                out_shape,
                width,
                ln_theta,
                inv_hd,
            }) as Box<dyn KernelOp>),
            input_enodes,
        )
    }
}

impl KernelOp for KernelRoPE {
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
        compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
        String,
        (Expression, Expression, Expression),
        (Expression, Expression, Expression),
        Expression,
        FxHashMap<Symbol, CudaSlice<u8>>,
    ) {
        let heads = self.out_shape[0].to_usize().expect("RoPE heads is static");
        let seq = self.out_shape[1];
        let hd = self.out_shape[2]
            .to_usize()
            .expect("RoPE head_dim is static");
        let w = self.width;
        let half = hd / 2;
        let lnt = self.ln_theta as f32;
        let inv_hd = self.inv_hd as f32;

        let vars: FxHashSet<Symbol> = seq.dyn_vars().into_iter().collect();
        let (dyn_defines, _sorted) = crate::kernel::hlir::generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let seq_expr = seq.to_kernel();

        let kernel = format!(
            include_str!("rope/indexed.cu.in"),
            dyn_defines = dyn_defines,
            dyn_dims_param = dyn_dims_param,
            half = half,
            hd = hd,
            inv_hd = inv_hd,
            lnt = lnt,
            seq_expr = seq_expr,
            w = w,
        );

        let (module, func) = if let Some((m, f)) = compile_cache.get(&kernel) {
            (m.clone(), f.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("rope_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        let tpb = hd.min(256);
        (
            func,
            module,
            kernel,
            (
                Expression::from(hd.div_ceil(tpb)),
                seq,
                Expression::from(heads),
            ),
            (
                Expression::from(tpb),
                Expression::from(1usize),
                Expression::from(1usize),
            ),
            Expression::from(0usize),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn output_bytes(&self) -> Expression {
        self.output_size() * 2
    }

    fn output_dtype(&self) -> DType {
        DType::Bf16
    }

    fn collect_dyn_vars_into(&self, vars: &mut FxHashSet<Symbol>) {
        self.out_shape[1].collect_dyn_vars_into(vars);
    }

    fn bytes_loaded(&self) -> Expression {
        self.output_size() * 2 + self.out_shape[1] * 4
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.output_size() * 16
    }

    fn kernel_name(&self) -> &'static str {
        "RoPEFused"
    }
}
