//! Fused argmax over the last axis.
//!
//! Matches the frontend `argmax(axis)` decomposition (src/frontend/unary.rs):
//!
//! ```text
//! mx      = Max(x, axis)                       broadcast back over axis
//! ne      = Cast(F32)(LessThan(x, mx)) + Cast(F32)(LessThan(mx, x))
//! eq      = Cast(Bool)(ne * -1 + 1)
//! one_hot = Cast(Int)(eq)
//! out     = Max(one_hot * Iota(cols), axis)
//! ```
//!
//! and unions a single kernel into the final Max's eclass. Output dtype is
//! Int, exactly like the decomposed chain. Tie rule preserved: the highest
//! index among equal maxima wins (the decomposition takes a max over
//! `index * one_hot`). Used for GPU-side greedy sampling: the host reads one
//! i32 per row instead of a vocab-sized logit row.

use std::sync::Arc;

use crate::{
    compile_module_image_for_current_device, cuda_dtype,
    kernel::{KernelOp, hlir::dtype_includes},
};
use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use orbitkv_compiler::{
    egglog_utils::{
        api::{Rule, SortDef, sort},
        base::{DTYPE, ELIST, EXPRESSION, OP_KIND},
        extract_dtype, extract_expr, extract_expr_list,
    },
    op::*,
    prelude::*,
};

// Wide blocks: one block scans a whole vocab row (≈128K elements), so
// maximize per-block parallelism.
const TPB: usize = 1024;

#[derive(Default, Debug, Clone)]
pub struct KernelArgmax {
    out_shape: Vec<Expression>,
    cols: Expression,
    dtype: DType,
}

impl EgglogOp for KernelArgmax {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelArgmax",
            &[("shape", ELIST), ("cols", EXPRESSION), ("dtype", DTYPE)],
        )
    }

    fn n_inputs(&self) -> usize {
        1
    }

    fn rewrites(&self) -> Vec<Rule> {
        // The scalar constants (-1, +1) sit behind frontend-emitted identity
        // casts; the kernel_lower fold unions a KernelConstant into each
        // Cast(Constant) eclass (F32 included), which is matched directly
        // here (const_like lives in the flashinfer host-op file, which loads
        // after kernel ops).
        vec![Rule::raw(include_str!("argmax/argmax_rewrite.egg"))]
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
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                out_shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache)
                    .unwrap(),
                cols: extract_expr(egraph, kind_children[1], expr_cache).unwrap(),
                dtype: extract_dtype(egraph, kind_children[2]),
            }) as Box<dyn KernelOp>),
            input_enodes,
        )
    }
}

impl KernelOp for KernelArgmax {
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
        let vars = self
            .out_shape
            .iter()
            .flat_map(|e| e.dyn_vars())
            .chain(self.cols.dyn_vars())
            .collect::<FxHashSet<_>>();
        let ty = cuda_dtype(self.dtype);
        let includes = dtype_includes(&[self.dtype]);
        let (dyn_defines, _sorted_dims) = crate::kernel::hlir::generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let cols = self.cols.to_kernel();
        let n_rows: Expression = self
            .out_shape
            .iter()
            .copied()
            .product::<Expression>()
            .max(1);

        // Tie rule: highest index wins, matching both the decomposed chain
        // (max over index*one_hot) and the CPU sampler's `max_by` (last max).
        let kernel = format!(
            include_str!("argmax/argmax.cu.in"),
            TPB = TPB,
            cols = cols,
            dyn_defines = dyn_defines,
            dyn_dims_param = dyn_dims_param,
            includes = includes,
            ty = ty,
        );

        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("argmax_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        (
            func,
            module,
            kernel,
            (n_rows, 1.into(), 1.into()),
            (TPB.into(), 1.into(), 1.into()),
            // warp_vals[32] + warp_idxs[32], both four-byte elements. The
            // previous 64-byte launch contract under-allocated shared memory
            // by 4x and let fused greedy sampling corrupt the CUDA context.
            (2 * (TPB / 32) * std::mem::size_of::<u32>()).into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.out_shape
            .iter()
            .copied()
            .product::<Expression>()
            .max(1)
    }

    fn output_bytes(&self) -> Expression {
        self.output_size() * 4
    }

    fn output_dtype(&self) -> DType {
        DType::Int
    }

    fn bytes_loaded(&self) -> Expression {
        (self.output_size() * self.cols * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.output_size() * self.cols
    }

    fn kernel_name(&self) -> &'static str {
        "Argmax"
    }
}
