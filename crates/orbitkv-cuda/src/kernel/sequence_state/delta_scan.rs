use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use orbitkv_compiler::{
    dtype::DType,
    op::{CustomOp, LLIROp},
    prelude::{Expression, FxHashMap, FxHashSet, GraphTensor, Symbol},
};

use super::{THREADS, compile_kernel, contiguous, render_source};
use crate::kernel::{KernelOp, hlir::generate_dyn_dims_defines};

#[derive(Clone, Copy)]
pub struct PackedDeltaScanPlan {
    pub query: GraphTensor,
    pub key: GraphTensor,
    pub value: GraphTensor,
    pub log_decay: GraphTensor,
    pub update_gate: GraphTensor,
    pub state: GraphTensor,
    pub query_indptr: GraphTensor,
}

#[derive(Clone, Copy, Debug)]
pub struct PackedDeltaScanSpec {
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_width: usize,
    pub value_width: usize,
    pub normalization_epsilon: f32,
}

#[derive(Clone, Copy)]
pub struct PackedDeltaScanOutput {
    pub values: GraphTensor,
    pub state: GraphTensor,
}

#[derive(Clone, Debug)]
pub(super) struct PackedDeltaScanKernel {
    pub tokens: Expression,
    pub requests: Expression,
    pub spec: PackedDeltaScanSpec,
}

impl CustomOp for PackedDeltaScanKernel {
    fn to_llir_op(&self) -> LLIROp {
        LLIROp::new::<dyn KernelOp>(Box::new(self.clone()) as Box<dyn KernelOp>)
    }

    fn compiler_facts(&self, custom_op_id: usize) -> String {
        format!("(packed-delta-scan-op {custom_op_id})")
    }
}

pub fn packed_delta_scan(
    plan: PackedDeltaScanPlan,
    spec: PackedDeltaScanSpec,
) -> PackedDeltaScanOutput {
    let PackedDeltaScanPlan {
        query,
        key,
        value,
        log_decay,
        update_gate,
        state,
        query_indptr,
    } = plan;
    assert!(spec.key_heads > 0 && spec.value_heads > 0);
    assert!(spec.value_heads.is_multiple_of(spec.key_heads));
    assert!(spec.key_width > 0 && spec.value_width > 0);
    assert!(
        spec.value_heads
            .checked_mul(spec.key_width)
            .and_then(|n| n.checked_mul(spec.value_width))
            .is_some()
    );
    assert!(spec.normalization_epsilon.is_finite() && spec.normalization_epsilon >= 0.0);
    let (tokens, _, _) = query.dims3();
    let (requests, _, _, _) = state.dims4();
    assert_eq!(
        query.dims(),
        [tokens, spec.key_heads.into(), spec.key_width.into()]
    );
    assert_eq!(key.dims(), query.dims());
    assert_eq!(
        value.dims(),
        [tokens, spec.value_heads.into(), spec.value_width.into()]
    );
    assert_eq!(log_decay.dims(), [tokens, spec.value_heads.into()]);
    assert_eq!(update_gate.dims(), log_decay.dims());
    assert_eq!(
        state.dims(),
        [
            requests,
            spec.value_heads.into(),
            spec.key_width.into(),
            spec.value_width.into()
        ]
    );
    assert_eq!(query_indptr.dims(), [requests + 1]);
    assert_eq!(query_indptr.dtype, DType::Int);
    assert!(
        [query, key, value, log_decay, update_gate, state]
            .iter()
            .all(|tensor| tensor.dtype == DType::F32)
    );
    assert!(
        [key, value, log_decay, update_gate, state, query_indptr]
            .iter()
            .all(|tensor| tensor.graph_ref == query.graph_ref)
    );
    let inputs = [
        query,
        key,
        value,
        log_decay,
        update_gate,
        state,
        query_indptr,
    ]
    .map(contiguous);
    let kernel = PackedDeltaScanKernel {
        tokens,
        requests,
        spec,
    };
    let value_elements = tokens * spec.value_heads * spec.value_width;
    let size = kernel.output_size();
    let graph = query.graph();
    let packed = graph.custom_op(kernel, inputs.to_vec(), size, DType::F32);
    PackedDeltaScanOutput {
        values: packed.gather(graph.iota('z', (tokens, spec.value_heads, spec.value_width))),
        state: packed.gather(graph.iota(
            Expression::from('z') + value_elements,
            (requests, spec.value_heads, spec.key_width, spec.value_width),
        )),
    }
}

impl PackedDeltaScanKernel {
    pub(super) fn source(&self) -> String {
        let variables = self.all_dyn_vars();
        let (defines, _) = generate_dyn_dims_defines(&variables);
        let parameter = if variables.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        render_source(
            include_str!("delta_scan.cu"),
            [
                ("@DYNAMIC_DEFINES@", defines),
                ("@DYNAMIC_PARAMETER@", parameter.to_owned()),
                ("@TOKENS@", self.tokens.to_kernel()),
                ("@REQUESTS@", self.requests.to_kernel()),
                (
                    "@TOKEN_VALUE_ELEMENTS@",
                    (self.tokens * self.spec.value_heads * self.spec.value_width).to_kernel(),
                ),
                ("@KEY_HEADS@", self.spec.key_heads.to_string()),
                ("@VALUE_HEADS@", self.spec.value_heads.to_string()),
                ("@KEY_WIDTH@", self.spec.key_width.to_string()),
                ("@VALUE_WIDTH@", self.spec.value_width.to_string()),
                (
                    "@NORMALIZATION_EPSILON@",
                    format!("{:e}f", self.spec.normalization_epsilon),
                ),
            ],
        )
    }
}

impl KernelOp for PackedDeltaScanKernel {
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
        let (module, function) = compile_kernel(stream, cache, &source, "packed_delta_scan");
        (
            function,
            module,
            source,
            (self.requests, self.spec.value_heads.into(), 1.into()),
            (THREADS.into(), 1.into(), 1.into()),
            ((2 * self.spec.key_width + THREADS) * std::mem::size_of::<f32>()).into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        (self.tokens + self.requests * self.spec.key_width)
            * self.spec.value_heads
            * self.spec.value_width
    }

    fn output_bytes(&self) -> Expression {
        self.output_size() * 4
    }

    fn collect_dyn_vars_into(&self, variables: &mut FxHashSet<Symbol>) {
        self.tokens.collect_dyn_vars_into(variables);
        self.requests.collect_dyn_vars_into(variables);
    }

    fn bytes_loaded(&self) -> Expression {
        let state_width = self.spec.key_width * self.spec.value_width;
        (self.tokens
            * self.spec.value_heads
            * (state_width * 2 + self.spec.key_width * 2 + self.spec.value_width + 2)
            + self.requests * self.spec.value_heads * state_width)
            * 4
    }

    fn bytes_stored(&self) -> Expression {
        (self.tokens
            * self.spec.value_heads
            * (self.spec.key_width * self.spec.value_width * 2 + self.spec.value_width)
            + self.requests * self.spec.value_heads * self.spec.key_width * self.spec.value_width)
            * 4
    }

    fn flops(&self) -> Expression {
        self.tokens
            * self.spec.value_heads
            * (self.spec.key_width * self.spec.value_width * 7 + self.spec.key_width * 4)
    }

    fn kernel_name(&self) -> &'static str {
        "PackedDeltaScan"
    }
}
