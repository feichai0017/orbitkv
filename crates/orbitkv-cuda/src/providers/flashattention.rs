//! FlashAttention-3 paged attention with provider-owned metadata and workspace.

pub mod jit;
mod plan;

use cudarc::driver::CudaStream;
use orbitkv_compiler::{
    dtype::DType,
    egglog_utils::{
        SerializedEGraph,
        api::{Rule, SortDef, sort},
        base::{DTYPE, EXPRESSION, F64, OP_KIND, STRING},
        extract_dtype, extract_expr,
    },
    op::{EgglogOp, LLIROp},
    prelude::{DynMap, ENodeId, Expression, FxHashMap, NodeIndex, Symbol},
};
use orbitkv_ops::ops::attention::{AttentionSpec, PagedKvLayout};
use std::sync::{Arc, Mutex};

use super::attention::{
    AttentionKernelCapability, AttentionProviderCapabilities, PageSizes, mask_from_window,
};
use super::{
    CudaGraphCaptureResource, DeviceBuffer, HostDeviceMemoryPlan, HostOp, ResourceViolation,
};
use plan::{Plan, Prepared};

const ALGORITHM: &str = "fa3-paged";
const HEAD_DIMENSIONS: &[(usize, usize)] = &[(64, 64), (128, 128), (256, 256)];
pub const CAPABILITIES: AttentionProviderCapabilities = AttentionProviderCapabilities {
    name: "flashattention",
    compute_majors: 9..=9,
    kernels: &[
        AttentionKernelCapability {
            algorithm: ALGORITHM,
            dtype: DType::F16,
            head_dimensions: HEAD_DIMENSIONS,
            layout: PagedKvLayout::TokenMajor,
            page_sizes: PageSizes::Any,
            supports_prefill: true,
        },
        AttentionKernelCapability {
            algorithm: ALGORITHM,
            dtype: DType::Bf16,
            head_dimensions: HEAD_DIMENSIONS,
            layout: PagedKvLayout::TokenMajor,
            page_sizes: PageSizes::Any,
            supports_prefill: true,
        },
    ],
};

#[derive(Default)]
pub struct FlashAttention {
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    page_size: usize,
    query_tokens: Expression,
    context_page_capacity: usize,
    requests: Expression,
    dtype: DType,
    scale: f64,
    window_left: i64,
    provider: String,
    prepared: Mutex<Option<Arc<Prepared>>>,
}

impl std::fmt::Debug for FlashAttention {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self {
            query_heads,
            kv_heads,
            head_dim,
            page_size,
            query_tokens,
            context_page_capacity,
            requests,
            dtype,
            scale,
            window_left,
            provider,
            prepared: _,
        } = self;
        f.debug_struct("FlashAttention")
            .field("algorithm", &ALGORITHM)
            .field("query_heads", query_heads)
            .field("kv_heads", kv_heads)
            .field("head_dim", head_dim)
            .field("page_size", page_size)
            .field("query_tokens", query_tokens)
            .field("context_page_capacity", context_page_capacity)
            .field("requests", requests)
            .field("dtype", dtype)
            .field("scale", scale)
            .field("window_left", window_left)
            .field("provider", provider)
            .finish()
    }
}

impl FlashAttention {
    fn config(&self) -> jit::Config {
        jit::Config {
            head_dim: self.head_dim,
            bf16: self.dtype == DType::Bf16,
            local: self.window_left >= 0,
        }
    }

    fn supports_geometry(&self) -> bool {
        let Some(mask) = mask_from_window(self.window_left) else {
            return false;
        };
        self.scale.is_finite()
            && self.scale > 0.0
            && CAPABILITIES.supports_geometry(
                ALGORITHM,
                AttentionSpec {
                    query_heads: self.query_heads,
                    kv_heads: self.kv_heads,
                    query_key_dim: self.head_dim,
                    value_dim: self.head_dim,
                    dtype: self.dtype,
                    scale: self.scale,
                    mask,
                },
                PagedKvLayout::TokenMajor,
                self.page_size,
                true,
            )
    }

    fn prepare(
        &self,
        stream: &Arc<CudaStream>,
        dimensions: &DynMap,
    ) -> anyhow::Result<Arc<Prepared>> {
        let plan = Plan::new(self, dimensions).map_err(|error| anyhow::anyhow!("{error:?}"))?;
        let mut cached = self
            .prepared
            .lock()
            .map_err(|_| anyhow::anyhow!("FlashAttention plan lock poisoned"))?;
        if let Some(prepared) = &*cached
            && prepared.plan == plan
            && prepared.stream_key == stream.cu_stream() as usize
        {
            return Ok(Arc::clone(prepared));
        }
        let library = jit::ensure_compiled(
            crate::target::CudaTarget::from_context(stream.context())?,
            self.config(),
        )?;
        let prepared = Arc::new(Prepared::new(plan, library, stream)?);
        *cached = Some(Arc::clone(&prepared));
        Ok(prepared)
    }
}

impl EgglogOp for FlashAttention {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "FlashAttention",
            &[
                ("query_heads", EXPRESSION),
                ("kv_heads", EXPRESSION),
                ("head_dim", EXPRESSION),
                ("page_size", EXPRESSION),
                ("query_tokens", EXPRESSION),
                ("context_page_capacity", EXPRESSION),
                ("requests", EXPRESSION),
                ("dtype", DTYPE),
                ("scale", F64),
                ("window_left", F64),
                ("provider", STRING),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        7
    }
    fn cleanup(&self) -> bool {
        false
    }
    fn egglog_declarations(&self) -> Vec<String> {
        AttentionProviderCapabilities::declarations()
    }
    fn rewrites(&self) -> Vec<Rule> {
        let Ok(provider) = jit::provider_identity() else {
            return vec![];
        };
        let mut rules = CAPABILITIES.eligibility_rules();
        for (name, bound) in [
            ("constant", "(= ?ctx (MNum ?capacity))"),
            ("bounded", "(= ?capacity (upper ?ctx))"),
        ] {
            rules.push(Rule::raw(
                include_str!("flashattention/paged_attention.egg")
                    .replace("@PROVIDER_IDENTITY@", &provider)
                    .replace("@CONTEXT_BOUND@", bound)
                    .replace("@BOUND_NAME@", name),
            ));
        }
        rules
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        fields: &[&'a ENodeId],
        inputs: Vec<&'a ENodeId>,
        _lists: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expressions: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        assert_eq!(fields.len(), 11, "FlashAttention schedule ABI mismatch");
        let provider = egraph.enodes[fields[10]].0.replace('"', "");
        super::provider_source::validate_provider_identity(
            &provider,
            &jit::provider_identity().unwrap(),
        )
        .unwrap();
        let mut expression =
            |index: usize| extract_expr(egraph, fields[index], expressions).unwrap();
        let query_heads = expression(0).to_usize().unwrap();
        let kv_heads = expression(1).to_usize().unwrap();
        let head_dim = expression(2).to_usize().unwrap();
        let page_size = expression(3).to_usize().unwrap();
        let query_tokens = expression(4);
        let context_page_capacity = expression(5)
            .to_usize()
            .expect("FlashAttention requires an explicit page capacity; rebuild the schedule");
        let requests = expression(6);
        let scalar = |index: usize| {
            egraph.enodes[fields[index]]
                .0
                .replace('"', "")
                .parse::<f64>()
                .unwrap()
        };
        let op = Self {
            query_heads,
            kv_heads,
            head_dim,
            page_size,
            query_tokens,
            context_page_capacity,
            requests,
            dtype: extract_dtype(egraph, fields[7]),
            scale: scalar(8),
            window_left: scalar(9) as i64,
            provider,
            prepared: Mutex::default(),
        };
        assert!(
            op.supports_geometry(),
            "unsupported FlashAttention geometry"
        );
        (
            LLIROp::new::<dyn HostOp>(Box::new(op) as Box<dyn HostOp>),
            inputs,
        )
    }
}

impl HostOp for FlashAttention {
    fn provider_dependencies(&self) -> Vec<super::registry::ProviderId> {
        vec![super::registry::ProviderId::FlashAttention]
    }

    fn prepare_compilation(
        &self,
        stream: &Arc<CudaStream>,
        dimensions: &DynMap,
    ) -> anyhow::Result<()> {
        self.prepare(stream, dimensions)?;
        Ok(())
    }
    fn execute(
        &self,
        stream: &Arc<CudaStream>,
        output: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dimensions: &DynMap,
    ) -> anyhow::Result<()> {
        self.prepare(stream, dimensions)?
            .launch(self, stream, output, inputs, buffers)
    }
    fn output_size(&self) -> Expression {
        self.query_tokens * self.query_heads * self.head_dim
    }
    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }
    fn output_dtype(&self) -> DType {
        self.dtype
    }
    fn cuda_graph_capture_arity(&self) -> Option<usize> {
        Some(7)
    }
    fn cuda_graph_capture_dyn_dims(&self) -> Vec<Symbol> {
        let mut symbols = [self.query_tokens, self.requests]
            .into_iter()
            .flat_map(|expression| expression.to_symbols())
            .collect::<Vec<_>>();
        symbols.sort();
        symbols.dedup();
        symbols
    }
    fn prepare_cuda_graph_capture(
        &self,
        stream: &Arc<CudaStream>,
        _output: NodeIndex,
        _inputs: &[NodeIndex],
        _buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dimensions: &DynMap,
    ) -> anyhow::Result<()> {
        self.prepare(stream, dimensions)?;
        Ok(())
    }
    fn cuda_graph_capture_resources(&self) -> Vec<CudaGraphCaptureResource> {
        self.prepared
            .lock()
            .unwrap()
            .as_ref()
            .map(|plan| Arc::clone(plan) as CudaGraphCaptureResource)
            .into_iter()
            .collect()
    }
    fn device_memory_plan(
        &self,
        _output: NodeIndex,
        _inputs: &[NodeIndex],
        _lengths: &FxHashMap<NodeIndex, usize>,
        dimensions: &DynMap,
    ) -> Result<HostDeviceMemoryPlan, ResourceViolation> {
        let plan = Plan::new(self, dimensions)?;
        Ok(HostDeviceMemoryPlan {
            persistent_bytes: plan.bytes,
            transient_peak_bytes: plan.bytes,
            ..Default::default()
        })
    }
    fn stats_name(&self) -> Option<&'static str> {
        Some("FlashAttention")
    }
}

#[cfg(test)]
#[path = "../../tests/unit/providers/flashattention/mod.rs"]
mod tests;
