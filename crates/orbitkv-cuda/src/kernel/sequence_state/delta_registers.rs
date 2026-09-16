use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use orbitkv_compiler::{
    egglog_utils::{
        api::{Rule, SortDef, sort},
        base::OP_KIND,
    },
    op::{EgglogOp, LLIROp},
    prelude::*,
};

use super::{
    compile_kernel,
    delta_scan::{GEOMETRY_FIELDS, PackedDeltaScanKernel},
    render_source,
};
use crate::kernel::{KernelOp, hlir::generate_dyn_dims_defines};

#[derive(Clone, Debug, Default)]
pub struct KernelDeltaRegisters {
    scan: PackedDeltaScanKernel,
}

const REGISTER_THREADS: usize = 32;

impl EgglogOp for KernelDeltaRegisters {
    fn sort(&self) -> SortDef {
        sort(OP_KIND, "KernelDeltaRegisters", &GEOMETRY_FIELDS)
    }
    fn n_inputs(&self) -> usize {
        7
    }
    fn egglog_declarations(&self) -> Vec<String> {
        vec![include_str!("declarations.egg").to_owned()]
    }
    fn rewrites(&self) -> Vec<Rule> {
        vec![Rule::raw(include_str!("delta_registers.egg"))]
    }
    fn cleanup(&self) -> bool {
        false
    }
    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        inputs: Vec<&'a ENodeId>,
        _lists: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expressions: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        let scan = PackedDeltaScanKernel::extract_geometry(egraph, children, expressions);
        assert!((1..=128).contains(&scan.spec.key_width));
        (LLIROp::new::<dyn KernelOp>(Box::new(Self { scan })), inputs)
    }
}

impl KernelDeltaRegisters {
    #[cfg(test)]
    pub(super) fn from_scan(scan: PackedDeltaScanKernel) -> Self {
        Self { scan }
    }

    pub(super) fn source(&self) -> String {
        register_source(
            &self.scan,
            &self.all_dyn_vars(),
            "packed_delta_registers",
            "const float* initial_state",
            "state[width] = initial_state[state_index];",
        )
    }
}

/// Both candidates share the exact update/reduction algorithm. Only the
/// initial state load differs; outputs and launch geometry stay identical.
pub(super) fn register_source(
    scan: &PackedDeltaScanKernel,
    variables: &FxHashSet<Symbol>,
    entry: &str,
    state_parameters: &str,
    state_load: &str,
) -> String {
    let (defines, _) = generate_dyn_dims_defines(variables);
    render_source(
        include_str!("delta_registers.cu"),
        [
            ("@ENTRY@", entry.to_owned()),
            ("@STATE_PARAMETERS@", state_parameters.to_owned()),
            ("@STATE_LOAD@", state_load.to_owned()),
            ("@DYNAMIC_DEFINES@", defines),
            (
                "@DYNAMIC_PARAMETER@",
                if variables.is_empty() {
                    ""
                } else {
                    ", const int* dyn_dims"
                }
                .to_owned(),
            ),
            ("@TOKENS@", scan.tokens.to_kernel()),
            ("@REQUESTS@", scan.requests.to_kernel()),
            (
                "@TOKEN_VALUE_ELEMENTS@",
                (scan.tokens * scan.spec.value_heads * scan.spec.value_width).to_kernel(),
            ),
            ("@KEY_HEADS@", scan.spec.key_heads.to_string()),
            ("@VALUE_HEADS@", scan.spec.value_heads.to_string()),
            ("@KEY_WIDTH@", scan.spec.key_width.to_string()),
            ("@VALUE_WIDTH@", scan.spec.value_width.to_string()),
            (
                "@NORMALIZATION_EPSILON@",
                format!("{:e}f", scan.spec.normalization_epsilon),
            ),
            (
                "@ROUND_NORMALIZED_QK_TO_BF16@",
                scan.spec.round_normalized_qk_to_bf16.to_string(),
            ),
            (
                "@ROUND_FINAL_STATE_TO_BF16@",
                scan.spec.round_final_state_to_bf16.to_string(),
            ),
        ],
    )
}

pub(super) fn register_grid(scan: &PackedDeltaScanKernel) -> (Expression, Expression, Expression) {
    (
        scan.requests,
        scan.spec.value_heads.into(),
        Expression::from(scan.spec.value_width).ceil_div(REGISTER_THREADS),
    )
}

pub(super) fn register_block() -> (Expression, Expression, Expression) {
    (REGISTER_THREADS.into(), 1.into(), 1.into())
}

pub(super) fn register_bytes_loaded(scan: &PackedDeltaScanKernel) -> Expression {
    // Q/K normalization and the shared decay/beta scalars are read once per
    // value-column tile. State remains in registers through all query tokens.
    let tiles = scan.spec.value_width.div_ceil(REGISTER_THREADS);
    let state_elements =
        scan.requests * scan.spec.value_heads * scan.spec.key_width * scan.spec.value_width;
    let token_elements = scan.tokens
        * scan.spec.value_heads
        * (tiles * (2 * scan.spec.key_width + 2) + scan.spec.value_width);
    let indptr_elements = scan.requests * scan.spec.value_heads * tiles * 2;
    (state_elements + token_elements) * std::mem::size_of::<f32>()
        + indptr_elements * std::mem::size_of::<i32>()
}

impl KernelOp for KernelDeltaRegisters {
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
        let (module, function) = compile_kernel(stream, cache, &source, "packed_delta_registers");
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
    }
    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }
    fn flops(&self) -> Expression {
        self.scan.flops()
    }
    fn collect_dyn_vars_into(&self, vars: &mut FxHashSet<Symbol>) {
        self.scan.collect_dyn_vars_into(vars);
    }
    fn kernel_name(&self) -> &'static str {
        "PackedDeltaRegisters"
    }
}
