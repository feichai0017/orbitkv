//! Fused SwiGLU reading both halves of a fused gate_up GEMM row.
//!
//! `out[r, c] = silu(x[r, c]) * x[r, I + c]` for `x (rows, 2I)`. Avoids the
//! offset-slice problem (tensors don't have offsets, so `x[:, I:]` lowers to
//! a Gather materialization) by indexing the second half inside the kernel,
//! and replaces gather + swish/mul region with one launch. The explicit custom
//! operation computes SiLU in F32 and rounds once at the output store. The
//! searchable BF16 decomposition preserves its constants and every operation's
//! BF16 rounding boundary instead.

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use orbitkv_compiler::{
    dtype::DType, op::CustomOp, op::LLIROp, prelude::FxHashMap, prelude::GraphTensor,
    prelude::Symbol, shape::Expression,
};

use crate::compile_module_image_for_current_device;
use crate::kernel::KernelOp;

const TPB: usize = 256;

/// Distinct arithmetic contracts; these are not interchangeable candidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwigluArithmetic {
    /// F32 SiLU and multiplication, rounded only at the output store.
    StoreOnce,
    /// BF16 constants and rounding after each operation of the matched graph.
    Bf16Decomposed,
}

#[derive(Debug, Clone)]
pub struct SwigluKernel {
    pub rows: Expression,
    pub intermediate: usize,
    pub dtype: DType,
    pub arithmetic: SwigluArithmetic,
}

impl KernelOp for SwigluKernel {
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
        let i = self.intermediate;
        let ty = crate::cuda_dtype(self.dtype);
        let includes = crate::kernel::hlir::dtype_includes(&[self.dtype]);
        // 2D grid: x = rows (dynamic, carried by the grid expression only),
        // y = static column tiles — one block per (row, tile) so a single
        // decode row still spreads across the GPU instead of one SM.
        let col_tiles = i.div_ceil(TPB);
        let kernel = format!(
            include_str!("swiglu/swiglu.cu.in"),
            TPB = TPB,
            i = i,
            includes = includes,
            ty = ty,
            body = self.element_body(),
        );

        let (module, func) = if let Some((m, f)) = compile_cache.get(&kernel) {
            (m.clone(), f.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("swiglu_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        (
            func,
            module,
            kernel,
            (
                self.rows,
                Expression::from(col_tiles),
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
        self.rows * self.intermediate
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn bytes_loaded(&self) -> Expression {
        (self.rows * self.intermediate * 2 * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.rows * self.intermediate * 6
    }

    fn kernel_name(&self) -> &'static str {
        match self.arithmetic {
            SwigluArithmetic::StoreOnce => "Swiglu",
            SwigluArithmetic::Bf16Decomposed => "SwigluBf16Decomposed",
        }
    }
}

impl SwigluKernel {
    fn element_body(&self) -> String {
        match self.arithmetic {
            SwigluArithmetic::StoreOnce => format!(
                "float silu = g / (1.0f + expf(-g));\n    out[row * I + col] = ({ty})(silu * u);",
                ty = crate::cuda_dtype(self.dtype),
            ),
            SwigluArithmetic::Bf16Decomposed => {
                assert_eq!(self.dtype, DType::Bf16);
                include_str!("swiglu/bf16_decomposed.cu.in").to_owned()
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct SwigluCustom(pub SwigluKernel);

impl CustomOp for SwigluCustom {
    fn to_llir_op(&self) -> LLIROp {
        LLIROp::new::<dyn KernelOp>(Box::new(self.0.clone()) as Box<dyn KernelOp>)
    }
}

/// `silu(x[:, :I]) * x[:, I:]` for a fused `(rows, 2I)` gate_up projection.
///
/// SiLU and multiplication compute in F32, with one rounding at the output
/// store. A noncontiguous input view is materialized before the custom operation.
pub fn fused_swiglu(x: GraphTensor, intermediate: usize) -> GraphTensor {
    let x_dims = x.dims();
    assert_eq!(
        x_dims.len(),
        2,
        "swiglu x must be 2-D (rows, 2*intermediate)"
    );
    let rows = x_dims[0];
    assert_eq!(
        x_dims[1].to_usize().expect("swiglu cols must be static"),
        2 * intermediate,
        "swiglu expects [gate | up] halves"
    );
    let kern = SwigluKernel {
        rows,
        intermediate,
        dtype: x.dtype,
        arithmetic: SwigluArithmetic::StoreOnce,
    };
    let x = if x.shape.is_contiguous() {
        x
    } else {
        x.gather(x.graph().iota('z', x.dims()))
    };
    let cx = unsafe { &mut *x.graph_ref };
    cx.custom_op(SwigluCustom(kern), vec![x], (rows, intermediate), x.dtype)
}

// ═══════════════════════════════════════════════════════════
// Egglog-matched BF16 SwiGLU: the pure-HLIR spelling
//   gate = x[:, :I] (view), up = x[:, I:] (offset slice → Gather)
//   silu = gate · recip(1 + exp2(gate · (−1) · log2e))
//   out  = silu · up                       [bf16]
// Every operation above has BF16 output. The rule must preserve those rounding
// boundaries, and prove the exact row layout before dropping view addressing.
// ═══════════════════════════════════════════════════════════

use orbitkv_compiler::{
    egglog_utils::{
        api::{Rule, SortDef, sort},
        base::{ELIST, OP_KIND},
        extract_expr_list,
    },
    op::EgglogOp,
    prelude::{ENodeId, SerializedEGraph},
};

/// Structural SwiGLU proof shared by generated-kernel rules.
#[doc(hidden)]
pub fn swiglu_chain_atoms() -> &'static str {
    include_str!("swiglu/bf16_match.egg")
}

#[derive(Default, Debug, Clone)]
pub struct KernelSwiglu;

impl EgglogOp for KernelSwiglu {
    fn sort(&self) -> SortDef {
        sort(OP_KIND, "KernelSwiglu", &[("out_shape", ELIST)])
    }

    fn n_inputs(&self) -> usize {
        1
    }

    fn rewrites(&self) -> Vec<Rule> {
        // Static arithmetic has already been folded before this ruleset. Use
        // integer equality for that range, while symbolic rows retain MMul.
        [
            ("symbolic rows", "(= ?up_range (MMul ?rows ?i))"),
            (
                "static rows",
                "(= ?rows (MNum ?row_count))
                 (= ?up_range (MNum ?element_count))
                 (= ?element_count (* ?row_count ?width))",
            ),
        ]
        .into_iter()
        .map(|(variant, range_proof)| {
            Rule::raw(format!(
                include_str!("swiglu/swiglu_rewrite.egg.in"),
                chain = swiglu_chain_atoms(),
                range_proof = range_proof,
                variant = variant,
            ))
        })
        .collect()
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
        let kern = SwigluKernel {
            rows: out_shape[0],
            intermediate: out_shape[1].to_usize().expect("swiglu I must be static"),
            dtype: DType::Bf16,
            arithmetic: SwigluArithmetic::Bf16Decomposed,
        };
        (
            LLIROp::new::<dyn KernelOp>(Box::new(kern) as Box<dyn KernelOp>),
            input_enodes,
        )
    }
}

#[cfg(test)]
#[path = "../../tests/unit/kernel/swiglu/mod.rs"]
mod tests;
