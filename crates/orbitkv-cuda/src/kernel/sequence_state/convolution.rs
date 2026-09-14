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
pub struct PackedConvolutionPlan {
    pub input: GraphTensor,
    pub weights: GraphTensor,
    pub history: GraphTensor,
    pub query_indptr: GraphTensor,
}

#[derive(Clone, Copy, Debug)]
pub struct PackedConvolutionSpec {
    pub channels: usize,
    pub kernel_width: usize,
}

#[derive(Clone, Copy)]
pub struct PackedConvolutionOutput {
    pub values: GraphTensor,
    pub history: GraphTensor,
}

#[derive(Clone, Debug)]
pub(super) struct PackedConvolutionKernel {
    pub tokens: Expression,
    pub requests: Expression,
    pub spec: PackedConvolutionSpec,
}

impl CustomOp for PackedConvolutionKernel {
    fn to_llir_op(&self) -> LLIROp {
        LLIROp::new::<dyn KernelOp>(Box::new(self.clone()) as Box<dyn KernelOp>)
    }

    fn compiler_facts(&self, custom_op_id: usize) -> String {
        format!("(packed-convolution-op {custom_op_id})")
    }
}

pub fn packed_causal_convolution(
    plan: PackedConvolutionPlan,
    spec: PackedConvolutionSpec,
) -> PackedConvolutionOutput {
    let PackedConvolutionPlan {
        input,
        weights,
        history,
        query_indptr,
    } = plan;
    assert!(spec.channels > 0 && spec.kernel_width >= 2);
    assert!(spec.channels.checked_mul(spec.kernel_width).is_some());
    let (tokens, channels) = input.dims2();
    let (requests, history_channels, history_width) = history.dims3();
    assert_eq!(channels, Expression::from(spec.channels));
    assert_eq!(history_channels, channels);
    assert_eq!(history_width, Expression::from(spec.kernel_width - 1));
    assert_eq!(
        weights.dims(),
        [channels, Expression::from(spec.kernel_width)]
    );
    assert_eq!(query_indptr.dims(), [requests + 1]);
    assert_eq!(query_indptr.dtype, DType::Int);
    assert!(
        [input, weights, history]
            .iter()
            .all(|tensor| tensor.dtype == DType::Bf16)
    );
    assert!(
        [weights, history, query_indptr]
            .iter()
            .all(|tensor| tensor.graph_ref == input.graph_ref)
    );
    let inputs = [input, weights, history, query_indptr].map(contiguous);
    let kernel = PackedConvolutionKernel {
        tokens,
        requests,
        spec,
    };
    let value_elements = tokens * spec.channels;
    let graph = input.graph();
    let size = kernel.output_size();
    let packed = graph.custom_op(kernel, inputs.to_vec(), size, DType::Bf16);
    PackedConvolutionOutput {
        values: packed.gather(graph.iota('z', (tokens, channels))),
        history: packed.gather(graph.iota(
            Expression::from('z') + value_elements,
            (requests, channels, history_width),
        )),
    }
}

impl PackedConvolutionKernel {
    pub(super) fn source(&self) -> String {
        let variables = self.all_dyn_vars();
        let (defines, _) = generate_dyn_dims_defines(&variables);
        let parameter = if variables.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        render_source(
            include_str!("convolution.cu"),
            [
                ("@DYNAMIC_DEFINES@", defines),
                ("@DYNAMIC_PARAMETER@", parameter.to_owned()),
                ("@TOKENS@", self.tokens.to_kernel()),
                ("@REQUESTS@", self.requests.to_kernel()),
                ("@CHANNELS@", self.spec.channels.to_string()),
                ("@HISTORY_WIDTH@", (self.spec.kernel_width - 1).to_string()),
                ("@KERNEL_WIDTH@", self.spec.kernel_width.to_string()),
            ],
        )
    }
}

impl KernelOp for PackedConvolutionKernel {
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
        let (module, function) =
            compile_kernel(stream, cache, &source, "packed_causal_convolution");
        (
            function,
            module,
            source,
            (
                self.requests,
                Expression::from(self.spec.channels).ceil_div(THREADS),
                1.into(),
            ),
            (THREADS.into(), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        (self.tokens + self.requests * (self.spec.kernel_width - 1)) * self.spec.channels
    }

    fn output_bytes(&self) -> Expression {
        self.output_size() * 2
    }

    fn output_dtype(&self) -> DType {
        DType::Bf16
    }

    fn collect_dyn_vars_into(&self, variables: &mut FxHashSet<Symbol>) {
        self.tokens.collect_dyn_vars_into(variables);
        self.requests.collect_dyn_vars_into(variables);
    }

    fn bytes_loaded(&self) -> Expression {
        (self.tokens * self.spec.channels * self.spec.kernel_width * 2
            + self.requests * self.spec.channels * (self.spec.kernel_width - 1))
            * 2
    }

    fn bytes_stored(&self) -> Expression {
        (self.tokens * self.spec.channels * self.spec.kernel_width
            + self.requests * self.spec.channels * (self.spec.kernel_width - 1))
            * 2
    }

    fn flops(&self) -> Expression {
        self.tokens * self.spec.channels * (self.spec.kernel_width * 2 + 4)
    }

    fn kernel_name(&self) -> &'static str {
        "PackedCausalConvolution"
    }
}
