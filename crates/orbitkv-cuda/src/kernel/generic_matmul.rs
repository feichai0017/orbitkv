use std::sync::Arc;

use crate::{
    compile_module_image_for_current_device, cuda_dtype,
    kernel::{
        KernelOp,
        hlir::{dtype_includes, generate_dyn_dims_defines},
    },
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
    shape::flatten_strides,
};

const MATMUL_BACKEND_RELATION_DECLARATIONS: &str = include_str!("generic_matmul/declarations.egg");

#[derive(Default, Debug, Clone)]
pub struct GenericMatmul {
    out_shape: Vec<Expression>,
    mul_shape: Vec<Expression>,
    k: Expression,
    lhs_strides: Vec<Expression>,
    rhs_strides: Vec<Expression>,
    sum_input_strides: Vec<Expression>,
    sum_iter_stride: Expression,
    out_strides: Vec<Expression>,
    dtype: DType,
}

impl EgglogOp for GenericMatmul {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "GenericMatmul",
            &[
                ("out_shape", ELIST),
                ("mul_shape", ELIST),
                ("k", EXPRESSION),
                ("lhs_strides", ELIST),
                ("rhs_strides", ELIST),
                ("sum_input_strides", ELIST),
                ("sum_iter_stride", EXPRESSION),
                ("out_strides", ELIST),
                ("dtype", DTYPE),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        2
    }

    fn egglog_declarations(&self) -> Vec<String> {
        vec![MATMUL_BACKEND_RELATION_DECLARATIONS.to_string()]
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![
            // The exact-layout witnesses below pattern-match the canonical
            // nested stride spellings, but HLIR serialization and constant
            // folding often leave only the folded product in the e-graph
            // (e.g. z*8388608 instead of (z*4096)*2048), so a semantically
            // canonical layout fails the syntactic match. Seed the canonical
            // spellings for every GenericMatmul: the expr fold rules union a
            // seeded spelling into the actual stride's e-class exactly when
            // the two are numerically equal, which lets the witness match
            // without ever equating unequal strides.
            Rule::raw(include_str!("generic_matmul/output_layout.egg")),
            Rule::raw(include_str!("generic_matmul/contiguous_inputs.egg")),
            // A low-precision materialized Mul followed by a separate reduction is
            // not the floating-point matmul contract implemented by GenericMatmul,
            // GEMV, or cuBLASLt: it rounds every product to F16/BF16 before the
            // reduction instead of accumulating products in F32. Keep search wide
            // across the F32-accumulating backends, not across a numerically
            // different product-materialization algorithm. F32 keeps its
            // decomposed alternative because there is no low-precision product
            // rounding to eliminate.
            //
            // GraphTensor::matmul is already decomposed to Mul + Sum at this IR
            // boundary, so there is no separate provenance marker to distinguish
            // it from the same contraction authored directly. The shared
            // GenericMatmul e-class is therefore the witness that this contraction
            // has the backend matmul accumulator contract. A future first-class
            // contraction marker could make that provenance explicit.
            // For F32, GenericMatmul deliberately preserves the decomposed
            // path's explicit product rounding and Kahan accumulation.
            //
            // Both forms must be covered because cleanup can observe either the
            // original HLIR Sum or its CUDA KernelSum lowering depending on rule
            // scheduling.
            Rule::raw(include_str!("generic_matmul/strided_output.egg")),
        ]
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
                mul_shape: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache)
                    .unwrap(),
                k: extract_expr(egraph, kind_children[2], expr_cache).unwrap(),
                lhs_strides: extract_expr_list(egraph, kind_children[3], list_cache, expr_cache)
                    .unwrap(),
                rhs_strides: extract_expr_list(egraph, kind_children[4], list_cache, expr_cache)
                    .unwrap(),
                sum_input_strides: extract_expr_list(
                    egraph,
                    kind_children[5],
                    list_cache,
                    expr_cache,
                )
                .unwrap(),
                sum_iter_stride: extract_expr(egraph, kind_children[6], expr_cache).unwrap(),
                out_strides: extract_expr_list(egraph, kind_children[7], list_cache, expr_cache)
                    .unwrap(),
                dtype: extract_dtype(egraph, kind_children[8]),
            })),
            input_enodes,
        )
    }
}

impl KernelOp for GenericMatmul {
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
        let vars = self.all_dyn_vars();
        let dtype = cuda_dtype(self.dtype);
        let includes = dtype_includes(&[self.dtype]);
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };

        let n_outputs = self.output_size();
        let sum_base_idx = flatten_strides(&self.out_shape, &self.sum_input_strides).to_kernel();
        let iter_offset = self.sum_iter_stride.to_kernel_with_index("i");
        let lhs_idx =
            flatten_strides(&self.mul_shape, &self.lhs_strides).to_kernel_with_index("mul_idx");
        let rhs_idx =
            flatten_strides(&self.mul_shape, &self.rhs_strides).to_kernel_with_index("mul_idx");
        let out_idx = flatten_strides(&self.out_shape, &self.out_strides).to_kernel();
        let k = self.k.to_kernel();

        let kernel = format!(
            include_str!("generic_matmul/matmul.cu.in"),
            n_outputs = n_outputs.to_kernel(),
            dtype = dtype,
            dyn_defines = dyn_defines,
            dyn_dims_param = dyn_dims_param,
            includes = includes,
            iter_offset = iter_offset,
            k = k,
            lhs_idx = lhs_idx,
            out_idx = out_idx,
            rhs_idx = rhs_idx,
            sum_base_idx = sum_base_idx,
        );

        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("generic_matmul").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        (
            func,
            module,
            kernel,
            (n_outputs, 1.into(), 1.into()),
            (256.into(), 1.into(), 1.into()),
            32.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.out_shape
            .iter()
            .copied()
            .product::<Expression>()
            .max(Expression::from(1))
    }

    fn collect_dyn_vars_into(&self, vars: &mut FxHashSet<Symbol>) {
        for expression in self
            .out_shape
            .iter()
            .chain(&self.mul_shape)
            .chain(std::iter::once(&self.k))
            .chain(&self.lhs_strides)
            .chain(&self.rhs_strides)
            .chain(&self.sum_input_strides)
            .chain(std::iter::once(&self.sum_iter_stride))
            .chain(&self.out_strides)
        {
            expression.collect_dyn_vars_into(vars);
        }
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_loaded(&self) -> Expression {
        (self.output_size() * self.k * self.dtype.bits() * 2).ceil_div(8)
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.output_size() * self.k * 2
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn kernel_name(&self) -> &'static str {
        "GenericMatmul"
    }
}
