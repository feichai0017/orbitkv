//! Convert only the elements selected by a gather, retaining its view addressing.
//!
//! Both cast-before-gather and cast-after-gather have the same single conversion
//! per output element. Egglog offers this region alongside the original graph;
//! shared producers and the complete execution cost determine which is selected.

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use orbitkv_compiler::{
    egglog_utils::{
        api::{Rule, SortDef, sort},
        base::{DTYPE, ELIST, EXPRESSION, OP_KIND},
        extract_dtype, extract_expr, extract_expr_list,
    },
    op::{EgglogOp, LLIROp},
    prelude::{DType, ENodeId, Expression, FxHashMap, FxHashSet, SerializedEGraph, Symbol},
    shape::flatten_strides,
};

use super::{
    KernelOp,
    hlir::{dtype_includes, generate_dyn_dims_defines},
};
use crate::{compile_module_image_for_current_device, cuda_dtype};

// A pointwise copy/conversion uses one thread per addressed output element.
const THREADS: usize = 256;

#[derive(Clone, Debug, Default)]
pub struct KernelGatherCast {
    size: Expression,
    index_shape: Vec<Expression>,
    index_strides: Vec<Expression>,
    data_shape: Vec<Expression>,
    data_strides: Vec<Expression>,
    input_dtype: DType,
    output_dtype: DType,
}

impl EgglogOp for KernelGatherCast {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelGatherCast",
            &[
                ("size", EXPRESSION),
                ("index_shape", ELIST),
                ("index_strides", ELIST),
                ("data_shape", ELIST),
                ("data_strides", ELIST),
                ("input_dtype", DTYPE),
                ("output_dtype", DTYPE),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        2
    }

    fn egglog_declarations(&self) -> Vec<String> {
        vec![include_str!("gather_cast/declarations.egg").to_owned()]
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![Rule::raw(include_str!("gather_cast/rewrites.egg"))]
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
        let size = extract_expr(egraph, kind_children[0], expr_cache).unwrap();
        let mut list = |index| {
            extract_expr_list(egraph, kind_children[index], list_cache, expr_cache).unwrap()
        };
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                size,
                index_shape: list(1),
                index_strides: list(2),
                data_shape: list(3),
                data_strides: list(4),
                input_dtype: extract_dtype(egraph, kind_children[5]),
                output_dtype: extract_dtype(egraph, kind_children[6]),
            })),
            input_enodes,
        )
    }
}

impl KernelGatherCast {
    fn source(&self) -> String {
        let vars = self.all_dyn_vars();
        let (dyn_defines, _) = generate_dyn_dims_defines(&vars);
        let dyn_parameter = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        format!(
            include_str!("gather_cast/kernel.cu.in"),
            includes = dtype_includes(&[self.input_dtype, self.output_dtype]),
            dyn_defines = dyn_defines,
            dyn_parameter = dyn_parameter,
            input_dtype = cuda_dtype(self.input_dtype),
            output_dtype = cuda_dtype(self.output_dtype),
            size = self.size.to_kernel(),
            index_address = flatten_strides(&self.index_shape, &self.index_strides).to_kernel(),
            data_address = flatten_strides(&self.data_shape, &self.data_strides).to_kernel(),
        )
    }
}

impl KernelOp for KernelGatherCast {
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
        let source = self.source();
        let (module, function) = if let Some((module, function)) = compile_cache.get(&source) {
            (Arc::clone(module), function.clone())
        } else {
            let image = compile_module_image_for_current_device(stream.context(), &source).unwrap();
            let module = stream.context().load_module(image).unwrap();
            let function = module.load_function("gather_cast").unwrap();
            compile_cache.insert(source.clone(), (Arc::clone(&module), function.clone()));
            (module, function)
        };
        (
            function,
            module,
            source,
            (self.size.ceil_div(THREADS), 1.into(), 1.into()),
            (THREADS.into(), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn collect_dyn_vars_into(&self, vars: &mut FxHashSet<Symbol>) {
        self.size.collect_dyn_vars_into(vars);
        for expression in self
            .index_shape
            .iter()
            .chain(&self.index_strides)
            .chain(&self.data_shape)
            .chain(&self.data_strides)
        {
            expression.collect_dyn_vars_into(vars);
        }
    }

    fn output_size(&self) -> Expression {
        self.size
    }
    fn output_dtype(&self) -> DType {
        self.output_dtype
    }
    fn output_bytes(&self) -> Expression {
        (self.size * self.output_dtype.bits()).ceil_div(8)
    }
    fn bytes_loaded(&self) -> Expression {
        (self.size * (self.input_dtype.bits() + DType::Int.bits())).ceil_div(8)
    }
    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }
    fn kernel_name(&self) -> &'static str {
        "GatherCast"
    }
}

#[cfg(test)]
#[path = "../../tests/unit/kernel/gather_cast/mod.rs"]
mod tests;
