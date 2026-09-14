//! In-place CUDA candidate for a decayed rank-one recurrent-state update.
//!
//! The semantic graph remains ordinary OrbitKV HLIR:
//! `next = state * decay + key * delta`. This module contributes only an
//! equivalent implementation candidate; egglog performs the match and the
//! normal alias/resource validator decides whether in-place mutation is legal.

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use orbitkv_compiler::{
    dtype::DType,
    egglog_utils::{
        api::{Rule, SortDef, sort},
        base::{ELIST, OP_KIND},
        extract_expr_list,
    },
    op::{EgglogOp, LLIROp},
    prelude::{ENodeId, Expression, FxHashMap, FxHashSet, SerializedEGraph, Symbol},
    shape::flatten_strides,
};

use crate::{
    compile_module_image_for_current_device,
    kernel::{KernelOp, hlir::generate_dyn_dims_defines},
};

const THREADS: usize = 256;
const REWRITES: &str = include_str!("recurrent_state.egg");
const CUDA_SOURCE: &str = include_str!("recurrent_state.cu");

/// CUDA implementation of `state * decay + key * delta` that aliases state.
#[derive(Clone, Debug, Default)]
pub struct KernelDeltaStateUpdate {
    shape: Vec<Expression>,
    state_strides: Vec<Expression>,
    decay_strides: Vec<Expression>,
    key_strides: Vec<Expression>,
    delta_strides: Vec<Expression>,
}

impl EgglogOp for KernelDeltaStateUpdate {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelDeltaStateUpdate",
            &[
                ("shape", ELIST),
                ("state_strides", ELIST),
                ("decay_strides", ELIST),
                ("key_strides", ELIST),
                ("delta_strides", ELIST),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        4
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![Rule::raw(REWRITES)]
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
        let mut list = |index| {
            extract_expr_list(egraph, kind_children[index], list_cache, expr_cache).unwrap()
        };
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                shape: list(0),
                state_strides: list(1),
                decay_strides: list(2),
                key_strides: list(3),
                delta_strides: list(4),
            }) as Box<dyn KernelOp>),
            input_enodes,
        )
    }
}

impl KernelOp for KernelDeltaStateUpdate {
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
        let mut variables = FxHashSet::default();
        self.collect_dyn_vars_into(&mut variables);
        let (dynamic_defines, _) = generate_dyn_dims_defines(&variables);
        let dynamic_parameter = if variables.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let total: Expression = self.shape.iter().copied().product();
        let state_index = flatten_strides(&self.shape, &self.state_strides).to_kernel();
        let decay_index = flatten_strides(&self.shape, &self.decay_strides).to_kernel();
        let key_index = flatten_strides(&self.shape, &self.key_strides).to_kernel();
        let delta_index = flatten_strides(&self.shape, &self.delta_strides).to_kernel();
        let source = RecurrentKernelSource {
            dynamic_defines: &dynamic_defines,
            dynamic_parameter,
            total: &total.to_kernel(),
            state_index: &state_index,
            decay_index: &decay_index,
            key_index: &key_index,
            delta_index: &delta_index,
        }
        .render();
        let (module, function) = if let Some((module, function)) = compile_cache.get(&source) {
            (module.clone(), function.clone())
        } else {
            let image = compile_module_image_for_current_device(stream.context(), &source).unwrap();
            let module = stream.context().load_module(image).unwrap();
            let function = module.load_function("delta_state_update").unwrap();
            compile_cache.insert(source.clone(), (module.clone(), function.clone()));
            (module, function)
        };
        (
            function,
            module,
            source,
            (total.ceil_div(THREADS), 1.into(), 1.into()),
            (THREADS.into(), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.shape.iter().copied().product()
    }

    fn collect_dyn_vars_into(&self, variables: &mut FxHashSet<Symbol>) {
        for expression in self
            .shape
            .iter()
            .chain(&self.state_strides)
            .chain(&self.decay_strides)
            .chain(&self.key_strides)
            .chain(&self.delta_strides)
        {
            expression.collect_dyn_vars_into(variables);
        }
    }

    fn output_bytes(&self) -> Expression {
        self.output_size() * 4
    }

    fn bytes_loaded(&self) -> Expression {
        self.output_size() * 16
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.output_size() * 3
    }

    fn build_params(
        &self,
        _stream: &Arc<CudaStream>,
        _output_ptr: u64,
        input_ptrs: &[u64],
        _internal_bufs: &[CudaSlice<u8>],
        dyn_dims_ptr: u64,
    ) -> Vec<u64> {
        let mut parameters = input_ptrs.to_vec();
        if dyn_dims_ptr != 0 {
            parameters.push(dyn_dims_ptr);
        }
        parameters
    }

    fn output_aliases_input(&self) -> Option<usize> {
        Some(0)
    }

    fn output_dtype(&self) -> DType {
        DType::F32
    }

    fn kernel_name(&self) -> &'static str {
        "DeltaStateUpdate"
    }
}

struct RecurrentKernelSource<'a> {
    dynamic_defines: &'a str,
    dynamic_parameter: &'a str,
    total: &'a str,
    state_index: &'a str,
    decay_index: &'a str,
    key_index: &'a str,
    delta_index: &'a str,
}

impl RecurrentKernelSource<'_> {
    fn render(&self) -> String {
        let replacements = [
            ("@DYNAMIC_DEFINES@", self.dynamic_defines),
            ("@DYNAMIC_PARAMETER@", self.dynamic_parameter),
            ("@TOTAL@", self.total),
            ("@STATE_INDEX@", self.state_index),
            ("@DECAY_INDEX@", self.decay_index),
            ("@KEY_INDEX@", self.key_index),
            ("@DELTA_INDEX@", self.delta_index),
        ];
        let mut source = CUDA_SOURCE.to_owned();
        for (placeholder, value) in replacements {
            assert_eq!(
                source.matches(placeholder).count(),
                1,
                "CUDA template placeholder {placeholder} must occur exactly once",
            );
            source = source.replace(placeholder, value);
        }
        source
    }
}

#[cfg(test)]
#[path = "../../tests/unit/kernel/recurrent_state/mod.rs"]
mod tests;
