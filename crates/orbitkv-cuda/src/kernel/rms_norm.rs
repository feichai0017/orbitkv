//! Fused RMSNorm: `x * rsqrt(mean(x²) + eps) * w` in one kernel.
//!
//! Replaces the decomposed norm sandwich (cast → square-mul → mean-reduce →
//! +eps → sqrt → recip → mul → weight-mul → cast: ~6-8 graph nodes) that the
//! 16-bit pipeline spells as `norm_in_f32`. Per the dtype contract the norm
//! computes in F32: the kernel loads `dtype` rows, accumulates the mean of
//! squares in F32, and rounds once at the store — the same semantics the
//! explicit-cast spelling expresses, minus the per-op intermediate roundings
//! (the decomposed path computes entirely in F32 between the casts too).
//!
//! Layout: x `(rows, cols)` contiguous in `dtype` with dynamic `rows`;
//! w `(cols,)` F32. One block per row; F32 warp + block reduction.

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use orbitkv_compiler::{
    dtype::DType, op::CustomOp, op::LLIROp, prelude::FxHashMap, prelude::GraphTensor,
    prelude::Symbol, shape::Expression,
};

use crate::compile_module_image_for_current_device;
use crate::kernel::KernelOp;

// Match the decomposed Mul + KernelSumReduce execution geometry. The norm's
// square-and-reduce contraction is also in GenericMatmul's e-class, so all
// three implementations preserve the same explicit product rounding, Kahan
// accumulation, and reduction tree.
const TPB: usize = 256;

#[derive(Debug, Clone)]
pub struct RMSNormKernel {
    pub rows: Expression,
    pub cols: usize,
    pub eps: f32,
    pub dtype: DType,
}

impl KernelOp for RMSNormKernel {
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
        let cols = self.cols;
        let eps = self.eps;
        let ty = crate::cuda_dtype(self.dtype);
        let includes = crate::kernel::hlir::dtype_includes(&[self.dtype]);
        let kernel = format!(
            include_str!("rms_norm/rms_norm.cu.in"),
            TPB = TPB,
            cols = cols,
            eps = eps,
            includes = includes,
            ty = ty,
        );

        let (module, func) = if let Some((m, f)) = compile_cache.get(&kernel) {
            (m.clone(), f.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("rms_norm_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        (
            func,
            module,
            kernel,
            (
                self.rows,
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
        self.rows * self.cols
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn bytes_loaded(&self) -> Expression {
        // Two passes over x plus the weight row.
        (self.rows * self.cols * self.dtype.bits() * 2).ceil_div(8) + self.cols * 4
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.rows * self.cols * 4
    }

    fn kernel_name(&self) -> &'static str {
        "RMSNorm"
    }
}

#[derive(Debug, Clone)]
pub struct RMSNormCustom(pub RMSNormKernel);

impl CustomOp for RMSNormCustom {
    fn to_llir_op(&self) -> LLIROp {
        LLIROp::new::<dyn KernelOp>(Box::new(self.0.clone()) as Box<dyn KernelOp>)
    }
}

/// Fused `x * rsqrt(mean(x², last axis) + eps) * w`.
///
/// `x` is `(rows, cols)` in any float dtype (F32 accumulation inside),
/// `w` is `(cols,)` F32. Returns `(rows, cols)` in `x`'s dtype.
pub fn fused_rms_norm(x: GraphTensor, w: GraphTensor, eps: f32) -> GraphTensor {
    assert_eq!(w.dtype, DType::F32, "RMSNorm weight must be F32");
    let x_dims = x.dims();
    assert_eq!(x_dims.len(), 2, "RMSNorm x must be 2-D (rows, cols)");
    let rows = x_dims[0];
    let cols = x_dims[1].to_usize().expect("RMSNorm cols must be static");
    assert_eq!(
        w.dims()[0].to_usize().expect("RMSNorm weight dim"),
        cols,
        "RMSNorm weight length mismatch"
    );

    let kern = RMSNormKernel {
        rows,
        cols,
        eps,
        dtype: x.dtype,
    };
    let cx = unsafe { &mut *x.graph_ref };
    cx.custom_op(RMSNormCustom(kern), vec![x, w], (rows, cols), x.dtype)
}

// ═══════════════════════════════════════════════════════════
// Egglog-matched RMSNorm: unions the fused kernel into the decomposed HLIR
// chain the models spell (per the pure-HLIR rule):
//
//   Cast(F32)(x_bf16) → Mul(x,x) → Sum(last) → ×Recip(Iota(cols)) → +eps
//   → Sqrt → Recip → ×x → ×w → Cast(Bf16)
//
// The searchable rewrite is intentionally limited to 2-D `(rows, cols)`
// norms. A 3-D per-head Q/K view may be a slice of a wider projection, so
// its logical shape alone does not prove the dense row layout required by
// the kernel. Such views stay on the decomposed path until the e-graph can
// carry an explicit base-layout proof.
// ═══════════════════════════════════════════════════════════

#[derive(Default, Debug, Clone)]
pub struct KernelRMSNorm;

use orbitkv_compiler::{
    egglog_utils::{
        api::{Rule, SortDef, sort},
        base::{ELIST, F64, OP_KIND},
        extract_expr_list,
    },
    op::EgglogOp,
    prelude::{ENodeId, SerializedEGraph},
};

impl EgglogOp for KernelRMSNorm {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelRMSNorm",
            &[("out_shape", ELIST), ("eps", F64)],
        )
    }

    fn n_inputs(&self) -> usize {
        2
    }

    fn rewrites(&self) -> Vec<Rule> {
        // Two relation-staged parts (pre → late): the rinv core (anchored by
        // the rare Sqrt→Recip pair and the eps Constant) emits a fact; the
        // weight-mul tail joins with ?rin/?xf bound, so each variant's pins
        // are cheap. A monolithic join explodes on rolled bodies with
        // several distinct layer instances.
        let core = include_str!("rms_norm/rms_norm_match.egg").to_string();

        // 2-D out-shape destructure + contiguous input / broadcast weight
        // stride pins.
        let variants = [(
            "2d",
            "(= ?wg_shape (ECons ?rows (ECons ?cols2 (ENil))))
                    (= ?cols ?cols2)
                    (= ?sq_a (ECons (MMul (MIter) ?cols)
                        (ECons (MIter) (ENil))))
                    (= ?sq_b ?sq_a)
                    (= ?wg_b (ECons (MNum 0) (ECons (MIter) (ENil))))",
        )];
        let tails = variants
            .into_iter()
            .map(|(variant, shape_pins)| {
                format!(
                    include_str!("rms_norm/rms_norm_rewrite.egg.in"),
                    shape_pins = shape_pins,
                    variant = variant,
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        vec![Rule::raw(format!("{core}\n{tails}"))]
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
        let eps: f64 = egraph.enodes[kind_children[1]]
            .0
            .replace('"', "")
            .parse()
            .unwrap();
        let cols = out_shape
            .last()
            .and_then(|c| c.to_usize())
            .expect("RMSNorm cols must be static");
        let rows = out_shape[..out_shape.len() - 1]
            .iter()
            .copied()
            .product::<Expression>();
        (
            LLIROp::new::<dyn KernelOp>(Box::new(RMSNormKernel {
                rows,
                cols,
                eps: eps as f32,
                dtype: DType::Bf16,
            }) as Box<dyn KernelOp>),
            input_enodes,
        )
    }
}
