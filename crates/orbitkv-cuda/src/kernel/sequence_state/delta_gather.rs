//! Read a delta scan's initial state directly through its existing Gather.
//!
//! This candidate only removes the selected-state materialization. It keeps
//! the original packed output and explicit state commit; it neither owns the
//! arena nor mutates it. Index validity and addressing retain Gather semantics.

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use orbitkv_compiler::{
    egglog_utils::{
        api::{Rule, SortDef, sort},
        base::{ELIST, OP_KIND},
        extract_expr_list,
    },
    op::{EgglogOp, LLIROp},
    prelude::*,
    shape::flatten_strides,
};

use super::{
    compile_kernel,
    delta_registers::{register_block, register_bytes_loaded, register_grid, register_source},
    delta_scan::{GEOMETRY_FIELDS, PackedDeltaScanKernel},
};
use crate::kernel::KernelOp;

#[derive(Clone, Debug, Default)]
pub struct KernelDeltaGather {
    pub(super) scan: PackedDeltaScanKernel,
    pub(super) index_shape: Vec<Expression>,
    pub(super) index_strides: Vec<Expression>,
    pub(super) data_shape: Vec<Expression>,
    pub(super) data_strides: Vec<Expression>,
}

impl EgglogOp for KernelDeltaGather {
    fn sort(&self) -> SortDef {
        let mut fields = GEOMETRY_FIELDS.to_vec();
        fields.extend([
            ("index_shape", ELIST),
            ("index_strides", ELIST),
            ("data_shape", ELIST),
            ("data_strides", ELIST),
        ]);
        sort(OP_KIND, "KernelDeltaGather", &fields)
    }

    fn n_inputs(&self) -> usize {
        8
    }

    fn egglog_declarations(&self) -> Vec<String> {
        vec![include_str!("declarations.egg").to_owned()]
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![Rule::raw(include_str!("delta_gather.egg"))]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        inputs: Vec<&'a ENodeId>,
        lists: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expressions: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        let scan = PackedDeltaScanKernel::extract_geometry(egraph, children, expressions);
        assert!((1..=128).contains(&scan.spec.key_width));
        let mut list = |index| {
            extract_expr_list(
                egraph,
                children[GEOMETRY_FIELDS.len() + index],
                lists,
                expressions,
            )
            .unwrap()
        };
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                scan,
                index_shape: list(0),
                index_strides: list(1),
                data_shape: list(2),
                data_strides: list(3),
            })),
            inputs,
        )
    }
}

impl KernelDeltaGather {
    pub(super) fn source(&self) -> String {
        // Use the same addressing emitter as KernelGather. The linear index
        // variables are 64-bit, but dynamic Min/Max retain its existing integer
        // width semantics; this is not a new general 64-bit Gather contract.
        let index_address = flatten_strides(&self.index_shape, &self.index_strides)
            .to_kernel_with_index("state_index");
        let data_address = flatten_strides(&self.data_shape, &self.data_strides)
            .to_kernel_with_index("gather_index");
        let load = format!(
            "const long long gather_index = state_indices[{index_address}];\n\
             state[width] = state_arena[{data_address}];"
        );
        register_source(
            &self.scan,
            &self.all_dyn_vars(),
            "packed_delta_gather",
            "const float* state_arena, const int* state_indices",
            &load,
        )
    }
}

impl KernelOp for KernelDeltaGather {
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
        cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
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
        let (module, function) = compile_kernel(stream, cache, &source, "packed_delta_gather");
        (
            function,
            module,
            source,
            register_grid(&self.scan),
            register_block(),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.scan.output_size()
    }
    fn output_bytes(&self) -> Expression {
        self.scan.output_bytes()
    }
    fn bytes_loaded(&self) -> Expression {
        register_bytes_loaded(&self.scan)
            + self.scan.requests
                * self.scan.spec.value_heads
                * self.scan.spec.key_width
                * self.scan.spec.value_width
                * std::mem::size_of::<i32>()
    }
    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }
    fn flops(&self) -> Expression {
        self.scan.flops()
    }
    fn collect_dyn_vars_into(&self, vars: &mut FxHashSet<Symbol>) {
        self.scan.collect_dyn_vars_into(vars);
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
    fn kernel_name(&self) -> &'static str {
        "PackedDeltaGather"
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/kernel/sequence_state/delta_gather/mod.rs"]
mod tests;
