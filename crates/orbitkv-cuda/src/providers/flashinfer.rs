pub mod find_indptrs;
pub mod jit;

use std::sync::{Arc, Mutex};

mod metadata;
mod workspace;
pub(crate) use metadata::FlashInferMetadataCache;
pub(crate) use workspace::resident_shared_device_memory_allocations;
pub use workspace::shared_device_memory_allocation;
use workspace::{FLOAT_WORKSPACE_SIZE, INT_WORKSPACE_SIZE, PlanWorkspace, with_plan_staging};

use orbitkv_compiler::{
    dtype::DType,
    egglog_utils::{
        api::{Rule, SortDef, sort},
        base::{DTYPE, EXPRESSION, F64, OP_KIND},
        extract_dtype, extract_expr,
    },
    op::{EgglogOp, LLIROp},
    prelude::{
        tracing::{Level, span},
        *,
    },
};

use super::attention::AttentionProviderCapabilities;
use jit::FlashInferDType;
use orbitkv_ops::ops::attention::{AttentionSpec, PagedKvLayout};

mod algorithm;
pub use algorithm::{CAPABILITIES, FlashInferAlgorithm};

use crate::{
    cudarc::driver::{CudaSlice, CudaStream, DevicePtr, result},
    providers::{DeviceBuffer, HostOp},
    resource::{HostDeviceMemoryPlan, ResourceViolation},
};

/// FlashInfer attention op (batch decode for f32/f16/bf16, batch prefill for
/// f16/bf16).
///
/// Lowers logical or structurally proved paged attention to an explicitly
/// selected FlashInfer kernel family. CUDA-core decode and tensor-core
/// attention compete for 16-bit decode; only tensor-core attention supports
/// packed prefill. Preparation and any auxiliary launches belong to this op.
///
/// Runtime graph inputs: Q, K_pool, V_pool, compact page indices, and optionally
/// qo_indptr + kv_indptr + kv_last_page_len for externally planned paged
/// execution. The additive causal mask is only an egglog proof anchor and is
/// not a runtime dependency.
pub struct FlashInferAttention {
    pub algorithm: FlashInferAlgorithm,
    pub num_qo_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub page_size: usize,
    /// Total query tokens across all packed requests.
    pub query_tokens: Expression,
    /// Request count is part of the compiled geometry, independent of the
    /// metadata buffers currently installed for another dynamic bucket.
    pub requests: Expression,
    /// Total pages attended over. Captured from the matched shape by the
    /// rules, so it does not depend on what the graph calls this dim.
    pub context_dim: Expression,
    pub dtype: DType,
    /// Softmax scale; 0.0 = default `1/sqrt(head_dim)`.
    pub sm_scale: f64,
    /// Sliding-window size in FlashInfer's `window_left` convention
    /// (number of previous kv positions visible); -1 = no window.
    pub window_left: i64,

    pub plan_info: Mutex<Vec<i64>>,
}

impl std::fmt::Debug for FlashInferAttention {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Exhaustive destructuring makes additions choose explicitly between
        // semantic identity and transient provider state.
        let Self {
            algorithm,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            query_tokens,
            context_dim,
            requests,
            dtype,
            sm_scale,
            window_left,
            plan_info: _,
        } = self;
        f.debug_struct("FlashInferAttention")
            .field("algorithm", algorithm)
            .field("num_qo_heads", num_qo_heads)
            .field("num_kv_heads", num_kv_heads)
            .field("head_dim", head_dim)
            .field("page_size", page_size)
            .field("query_tokens", query_tokens)
            .field("context_dim", context_dim)
            .field("requests", requests)
            .field("dtype", dtype)
            .field("sm_scale", sm_scale)
            .field("window_left", window_left)
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FlashInferPlanSpec {
    algorithm: FlashInferAlgorithm,
    total_q_tokens: usize,
    batch_size: usize,
    c: usize,
    num_qo_heads: usize,
    num_kv_heads: usize,
    page_size: usize,
    head_dim: usize,
    kv_dim: usize,
    max_kv_pages: usize,
    dtype: FlashInferDType,
    /// f32 bits of the softmax scale actually passed to the kernel.
    sm_scale_bits: u32,
    /// FlashInfer window_left; -1 = no sliding window. Part of the spec so
    /// the shared prepare cache and capture signatures distinguish kernel
    /// variants.
    window_left: i32,
    kv_indptr_host: Vec<i32>,
    qo_indptr_host: Vec<i32>,
}

impl FlashInferPlanSpec {
    fn uses_tensor_cores(&self) -> bool {
        self.algorithm == FlashInferAlgorithm::TensorCore
    }

    fn uses_capacity_plan(&self, explicit_indptr: bool) -> bool {
        !explicit_indptr && !self.uses_tensor_cores() && self.total_q_tokens == 1
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FlashInferPointers {
    q: u64,
    k_cache: u64,
    v_cache: u64,
    gather_idx: u64,
    output: u64,
    pub(crate) explicit_qo_indptr: Option<u64>,
    pub(crate) explicit_kv_indptr: Option<u64>,
    pub(crate) explicit_last_page_len: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FlashInferCaptureSignature {
    pub(crate) spec: FlashInferPlanSpec,
    pub(crate) ptrs: FlashInferPointers,
}

/// Pointer-free proof that two derived-decode islands may reuse one prepared
/// allocation. Equal shapes are not enough: the mutable metadata buffers are
/// populated from `gather_idx`, so the producer node is part of the key.
/// Explicit-indptr plans deliberately have no key because their device
/// contents can change without their node identities changing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FlashInferPrepareKey {
    spec: FlashInferPlanSpec,
    gather_idx: NodeIndex,
}

impl FlashInferPrepareKey {
    pub(crate) fn for_inputs(spec: FlashInferPlanSpec, inputs: &[NodeIndex]) -> Option<Self> {
        (inputs.len() == 4).then(|| Self {
            spec,
            gather_idx: inputs[3],
        })
    }
}

#[derive(Debug, Clone)]
pub(crate) struct FlashInferDeviceResourceSpec {
    /// `None` means explicit indptr contents cannot be proven equal without a
    /// forbidden preflight device read, so this plan must not be deduplicated.
    pub(crate) cache_key: Option<FlashInferPrepareKey>,
    spec: FlashInferPlanSpec,
    explicit_qo_indptr: bool,
    explicit_kv_indptr: bool,
    explicit_last_page_len: bool,
    enable_cuda_graph: bool,
}

impl FlashInferDeviceResourceSpec {
    pub(crate) fn prepared_device_bytes(&self) -> Result<usize, ResourceViolation> {
        prepared_device_bytes(
            &self.spec,
            self.explicit_qo_indptr,
            self.explicit_kv_indptr,
            self.explicit_last_page_len,
            self.enable_cuda_graph,
        )
    }
}

fn prepared_device_bytes(
    spec: &FlashInferPlanSpec,
    explicit_qo_indptr: bool,
    explicit_kv_indptr: bool,
    explicit_last_page_len: bool,
    enable_cuda_graph: bool,
) -> Result<usize, ResourceViolation> {
    let uses_tensor_cores = spec.uses_tensor_cores();
    let mut allocations = vec![INT_WORKSPACE_SIZE];

    if !explicit_kv_indptr {
        // Derived prefill owns [0, c]. CUDA-graph decode owns [0, c]
        // plus the one-element current-c update buffer.
        allocations.push(2usize * std::mem::size_of::<i32>());
        if enable_cuda_graph && !uses_tensor_cores && spec.total_q_tokens == 1 {
            allocations.push(std::mem::size_of::<i32>());
        }
    }
    if uses_tensor_cores && !explicit_qo_indptr {
        allocations.push(2usize * std::mem::size_of::<i32>());
    }
    if !explicit_last_page_len {
        allocations.push(
            spec.c
                .max(1)
                .checked_mul(std::mem::size_of::<i32>())
                .ok_or(ResourceViolation::ArithmeticOverflow {
                    resource: "FlashInfer indices",
                })?,
        );
    }
    if !explicit_last_page_len {
        allocations.push(
            spec.batch_size
                .max(1)
                .checked_mul(std::mem::size_of::<i32>())
                .ok_or(ResourceViolation::ArithmeticOverflow {
                    resource: "FlashInfer last-page lengths",
                })?,
        );
    }
    allocations.push(
        spec.total_q_tokens
            .checked_mul(spec.num_qo_heads)
            .and_then(|elements| elements.checked_mul(spec.head_dim))
            .and_then(|elements| elements.checked_mul(spec.dtype.size_of()))
            .map(|bytes| bytes.max(1))
            .ok_or(ResourceViolation::ArithmeticOverflow {
                resource: "FlashInfer temporary output",
            })?,
    );
    allocations
        .into_iter()
        .try_fold(0usize, |sum, bytes| sum.checked_add(bytes))
        .ok_or(ResourceViolation::ArithmeticOverflow {
            resource: "FlashInfer prepared device memory",
        })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FlashInferResolvedAttention {
    spec: FlashInferPlanSpec,
    ptrs: FlashInferPointers,
}

impl FlashInferResolvedAttention {
    pub(crate) fn has_explicit_indptr(&self) -> bool {
        self.ptrs.explicit_kv_indptr.is_some()
    }

    pub(crate) fn current_c(&self) -> usize {
        self.spec.c
    }

    pub(crate) fn graph_plan_capacity(&self, existing_capacity: Option<usize>) -> usize {
        if !self.spec.uses_capacity_plan(self.has_explicit_indptr()) {
            return self.spec.c;
        }
        if let Some(capacity) = existing_capacity
            && self.spec.c <= capacity
        {
            return capacity;
        }
        flashinfer_graph_plan_capacity(self.spec.c, self.spec.max_kv_pages)
    }

    pub(crate) fn signature_for_graph_plan(&self, plan_c: usize) -> FlashInferCaptureSignature {
        let mut spec = self.spec.clone();
        let ptrs = self.ptrs;
        if spec.uses_capacity_plan(self.has_explicit_indptr()) {
            spec.c = plan_c;
            spec.kv_indptr_host = vec![0, plan_c as i32];
        }
        FlashInferCaptureSignature { spec, ptrs }
    }
}

pub(crate) struct PreparedFlashInferAttention {
    lib: &'static jit::FlashInferLib,
    spec: FlashInferPlanSpec,
    plan_info: Vec<i64>,
    workspace: PlanWorkspace,
    _owned_kv_indptr: Option<CudaSlice<i32>>,
    owned_kv_indptr_ptr: Option<u64>,
    _owned_qo_indptr: Option<CudaSlice<i32>>,
    owned_qo_indptr_ptr: Option<u64>,
    current_c: Option<Mutex<CudaSlice<i32>>>,
    current_c_ptr: Option<u64>,
    _indices: Option<CudaSlice<i32>>,
    indices_ptr: u64,
    _last_page_len: Option<CudaSlice<i32>>,
    last_page_len_ptr: u64,
    _temp_output: CudaSlice<u8>,
    temp_output_ptr: u64,
}

impl Default for FlashInferAttention {
    fn default() -> Self {
        Self {
            algorithm: FlashInferAlgorithm::default(),
            num_qo_heads: 0,
            num_kv_heads: 0,
            head_dim: 0,
            page_size: 0,
            query_tokens: Expression::default(),
            requests: Expression::default(),
            context_dim: Expression::default(),
            dtype: DType::F32,
            sm_scale: 0.0,
            window_left: -1,
            plan_info: Mutex::new(Vec::new()),
        }
    }
}

impl FlashInferAttention {
    fn group_size(num_qo_heads: usize, num_kv_heads: usize) -> anyhow::Result<usize> {
        num_qo_heads
            .checked_div(num_kv_heads)
            .filter(|group| *group > 0 && num_qo_heads.is_multiple_of(num_kv_heads))
            .ok_or_else(|| anyhow::anyhow!("FlashInfer query/KV head geometry is invalid"))
    }

    /// Constructs a paged-attention execution node whose page geometry is
    /// supplied by an external state compiler.
    ///
    /// This is an explicit low-level FlashInfer entry point. OrbitKV uses the
    /// provider-neutral [`orbitkv_ops::ops::attention::attention`] operation
    /// instead. A direct [`Graph::custom_op`] insertion requires explicit CSR
    /// metadata and bypasses the egglog provider-identity replay validation.
    #[allow(clippy::too_many_arguments)]
    pub fn paged(
        algorithm: FlashInferAlgorithm,
        num_qo_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        page_size: usize,
        query_tokens: Expression,
        context_pages: Expression,
        requests: Expression,
        dtype: DType,
        sm_scale: f64,
        window_left: Option<usize>,
    ) -> Self {
        let op = Self {
            algorithm,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            query_tokens,
            context_dim: context_pages,
            requests,
            dtype,
            sm_scale,
            window_left: window_left.map_or(-1, |value| {
                i64::try_from(value).expect("sliding window does not fit i64")
            }),
            plan_info: Mutex::new(Vec::new()),
        };
        assert!(
            op.supports_geometry(false),
            "unsupported FlashInfer attention geometry"
        );
        op
    }

    fn supports_geometry(&self, prefill: bool) -> bool {
        let Some(mask) = super::attention::mask_from_window(self.window_left) else {
            return false;
        };
        CAPABILITIES.supports_geometry(
            self.algorithm.as_str(),
            AttentionSpec {
                query_heads: self.num_qo_heads,
                kv_heads: self.num_kv_heads,
                query_key_dim: self.head_dim,
                value_dim: self.head_dim,
                dtype: self.dtype,
                scale: self.sm_scale,
                mask,
            },
            PagedKvLayout::TokenMajor,
            self.page_size,
            prefill,
        )
    }
}

impl orbitkv_compiler::op::CustomOp for FlashInferAttention {
    fn to_llir_op(&self) -> LLIROp {
        LLIROp::new::<dyn HostOp>(Box::new(Self {
            algorithm: self.algorithm,
            num_qo_heads: self.num_qo_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            page_size: self.page_size,
            query_tokens: self.query_tokens,
            context_dim: self.context_dim,
            requests: self.requests,
            dtype: self.dtype,
            sm_scale: self.sm_scale,
            window_left: self.window_left,
            plan_info: Mutex::new(Vec::new()),
        }) as Box<dyn HostOp>)
    }

    fn compiler_facts(&self, custom_op_id: usize) -> String {
        let _ = custom_op_id;
        String::new()
    }

    fn compiler_declarations(&self) -> &'static str {
        ""
    }
}

impl EgglogOp for FlashInferAttention {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "FlashInferAttention",
            &[
                ("num_qo_heads", EXPRESSION),
                ("num_kv_heads", EXPRESSION),
                ("head_dim", EXPRESSION),
                ("page_size", EXPRESSION),
                ("query_tokens", EXPRESSION),
                ("context_dim", EXPRESSION),
                ("requests", EXPRESSION),
                ("dtype", DTYPE),
                ("sm_scale", F64),
                ("window_left", F64),
                ("algorithm", orbitkv_compiler::egglog_utils::base::STRING),
                ("provider", orbitkv_compiler::egglog_utils::base::STRING),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        // Q, K_pool, V_pool, compact gather_idx. The egglog rules still use
        // flat gather indices and masks as structural proof anchors, but
        // extract() returns only the runtime inputs FlashInfer actually uses.
        5
    }

    fn egglog_declarations(&self) -> Vec<String> {
        AttentionProviderCapabilities::declarations()
    }

    fn rewrites(&self) -> Vec<Rule> {
        let Ok(provider) = jit::provider_identity() else {
            return vec![];
        };
        let mut rules = CAPABILITIES.eligibility_rules();
        rules.push(Rule::raw(
            include_str!("flashinfer/paged_attention.egg")
                .replace("@PROVIDER_IDENTITY@", &provider),
        ));
        rules.push(Rule::raw(
            include_str!("flashinfer/flashinfer_attention.egg")
                .replace("@PROVIDER_IDENTITY@", &provider),
        ));
        rules
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a orbitkv_compiler::egglog_utils::SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        _list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        let [
            qo_heads,
            kv_heads,
            head_dim,
            page_size,
            query_tokens,
            context_dim,
            requests,
            dtype,
            scale,
            window,
            algorithm,
            provider,
        ] = kind_children
        else {
            panic!(
                "FlashInfer schedule lacks request geometry or algorithm/provider identity; rebuild the selected schedule"
            );
        };
        let recorded_provider = egraph.enodes[*provider].0.replace('"', "");
        let current_provider = jit::provider_identity().unwrap_or_else(|error| panic!("{error}"));
        crate::providers::provider_source::validate_provider_identity(
            &recorded_provider,
            &current_provider,
        )
        .unwrap_or_else(|error| panic!("{error}"));
        let num_qo_heads = extract_expr(egraph, qo_heads, expr_cache)
            .unwrap()
            .exec(&FxHashMap::default())
            .unwrap();
        let num_kv_heads = extract_expr(egraph, kv_heads, expr_cache)
            .unwrap()
            .exec(&FxHashMap::default())
            .unwrap();
        let head_dim = extract_expr(egraph, head_dim, expr_cache)
            .unwrap()
            .exec(&FxHashMap::default())
            .unwrap();
        let page_size = extract_expr(egraph, page_size, expr_cache)
            .unwrap()
            .exec(&FxHashMap::default())
            .unwrap();
        let query_tokens = extract_expr(egraph, query_tokens, expr_cache).unwrap();
        let context_dim = extract_expr(egraph, context_dim, expr_cache).unwrap();
        let requests = extract_expr(egraph, requests, expr_cache).unwrap();
        let dtype = extract_dtype(egraph, dtype);
        let sm_scale: f64 = egraph.enodes[*scale].0.replace('"', "").parse().unwrap();
        let window_left = egraph.enodes[*window]
            .0
            .replace('"', "")
            .parse::<f64>()
            .unwrap()
            .round() as i64;
        assert!(
            FlashInferDType::from_dtype(dtype).is_some(),
            "FlashInferAttention extracted with unsupported dtype {dtype:?}"
        );

        let extracted = Self {
            algorithm: FlashInferAlgorithm::from_name(
                &egraph.enodes[*algorithm].0.replace('"', ""),
            )
            .expect("unknown FlashInfer algorithm in compiled graph"),
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            query_tokens,
            context_dim,
            requests,
            dtype,
            sm_scale,
            window_left,
            plan_info: Mutex::new(Vec::new()),
        };
        assert!(
            extracted.supports_geometry(false),
            "unsupported FlashInfer schedule geometry"
        );

        let final_inputs = if input_enodes.len() == 7 {
            input_enodes
        } else {
            let flat_idx_node = input_enodes[3];
            let gather_idx = find_indptrs::try_find_compact_gather_idx(egraph, flat_idx_node)
                .expect(
                    "FlashInferAttention matched a gather without recoverable compact gather_idx",
                );
            vec![
                input_enodes[0],
                input_enodes[1],
                input_enodes[2],
                gather_idx,
            ]
        };

        let op = LLIROp::new::<dyn HostOp>(Box::new(extracted) as Box<dyn HostOp>);
        (op, final_inputs)
    }

    fn cleanup(&self) -> bool {
        false
    }
}

impl FlashInferAttention {
    pub(crate) fn accepts_graph_inputs(&self, count: usize) -> bool {
        count == 4 || count == 6 && self.page_size == 1 || count == 7
    }

    /// Resolve only dimensions and allocation sizes. No pointer is fabricated
    /// and no explicit-indptr contents are copied from the device during
    /// preflight.
    pub(crate) fn device_resource_spec(
        &self,
        inputs: &[NodeIndex],
        buffer_lengths: &FxHashMap<NodeIndex, usize>,
        dyn_map: &DynMap,
        enable_cuda_graph: bool,
    ) -> Result<FlashInferDeviceResourceSpec, ResourceViolation> {
        let total_q_tokens =
            self.query_tokens
                .exec(dyn_map)
                .ok_or(ResourceViolation::UnresolvedExpression {
                    resource: "FlashInfer total query tokens",
                })?;
        let c = self
            .context_dim
            .exec(dyn_map)
            .ok_or(ResourceViolation::UnresolvedExpression {
                resource: "FlashInfer context length",
            })?;
        if !self.accepts_graph_inputs(inputs.len()) {
            return Err(ResourceViolation::HostResourcePlanning {
                name: "FlashInferAttention input arity",
            });
        }
        let dtype = FlashInferDType::from_dtype(self.dtype).ok_or(
            ResourceViolation::HostResourcePlanning {
                name: "FlashInferAttention dtype",
            },
        )?;
        let kv_dim = self.num_kv_heads.checked_mul(self.head_dim).ok_or(
            ResourceViolation::ArithmeticOverflow {
                resource: "FlashInfer KV width",
            },
        )?;
        let kv_page_bytes = kv_dim
            .checked_mul(dtype.size_of())
            .and_then(|bytes| bytes.checked_mul(self.page_size))
            .ok_or(ResourceViolation::ArithmeticOverflow {
                resource: "FlashInfer KV page bytes",
            })?;
        let k_bytes = buffer_lengths.get(&inputs[1]).copied().ok_or(
            ResourceViolation::HostResourcePlanning {
                name: "FlashInfer K-cache length",
            },
        )?;
        let v_bytes = buffer_lengths.get(&inputs[2]).copied().ok_or(
            ResourceViolation::HostResourcePlanning {
                name: "FlashInfer V-cache length",
            },
        )?;
        let max_kv_pages = k_bytes
            .checked_div(kv_page_bytes)
            .zip(v_bytes.checked_div(kv_page_bytes))
            .map(|(k_pages, v_pages)| k_pages.min(v_pages).max(c))
            .unwrap_or(c);
        let explicit_indptr = inputs.len() >= 6;
        let explicit_last_page_len = inputs.len() == 7;
        let batch_size =
            self.requests
                .exec(dyn_map)
                .ok_or(ResourceViolation::UnresolvedExpression {
                    resource: "FlashInfer request count",
                })?;
        if total_q_tokens == 0
            || batch_size == 0
            || total_q_tokens < batch_size
            || (!explicit_indptr && batch_size != 1 && batch_size != total_q_tokens)
            || !self.supports_geometry(total_q_tokens > batch_size)
        {
            return Err(ResourceViolation::HostResourcePlanning {
                name: "FlashInfer algorithm geometry",
            });
        }
        let sm_scale = if self.sm_scale == 0.0 {
            1.0 / (self.head_dim as f32).sqrt()
        } else {
            self.sm_scale as f32
        };
        let mut spec = FlashInferPlanSpec {
            algorithm: self.algorithm,
            total_q_tokens,
            batch_size,
            c,
            num_qo_heads: self.num_qo_heads,
            num_kv_heads: self.num_kv_heads,
            page_size: self.page_size,
            head_dim: self.head_dim,
            kv_dim,
            max_kv_pages,
            dtype,
            sm_scale_bits: sm_scale.to_bits(),
            window_left: self.window_left as i32,
            kv_indptr_host: Vec::new(),
            qo_indptr_host: Vec::new(),
        };
        if enable_cuda_graph && spec.uses_capacity_plan(explicit_indptr) {
            let plan_c = flashinfer_graph_plan_capacity(c, max_kv_pages);
            spec.c = plan_c;
            spec.kv_indptr_host = vec![0, plan_c as i32];
        }
        Ok(FlashInferDeviceResourceSpec {
            cache_key: FlashInferPrepareKey::for_inputs(spec.clone(), inputs),
            spec,
            explicit_qo_indptr: explicit_indptr,
            explicit_kv_indptr: explicit_indptr,
            explicit_last_page_len,
            enable_cuda_graph,
        })
    }

    pub(crate) fn resolve_for_graph(
        &self,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &DynMap,
    ) -> anyhow::Result<FlashInferResolvedAttention> {
        let total_q_tokens = self
            .query_tokens
            .exec(dyn_map)
            .ok_or_else(|| anyhow::anyhow!("FlashInferAttention query_tokens is unresolved"))?;
        let c = self
            .context_dim
            .exec(dyn_map)
            .ok_or_else(|| anyhow::anyhow!("FlashInferAttention context_dim is unresolved"))?;
        if !self.accepts_graph_inputs(inputs.len()) {
            anyhow::bail!(
                "FlashInferAttention expects 4 inputs (derived causal decode), 6 inputs (explicit indptrs), or 7 inputs (external page plan), got {}",
                inputs.len()
            );
        }

        let get_buf = |name: &str, node: NodeIndex| -> anyhow::Result<DeviceBuffer> {
            buffers.get(&node).copied().ok_or_else(|| {
                anyhow::anyhow!("FlashInferAttention missing {name} buffer for {node:?}")
            })
        };

        let q_buf = get_buf("Q", inputs[0])?;
        let k_buf = get_buf("K_cache", inputs[1])?;
        let v_buf = get_buf("V_cache", inputs[2])?;
        let gather_idx_buf = get_buf("gather_idx", inputs[3])?;
        let out_buf = get_buf("output", self_node)?;

        let dtype = FlashInferDType::from_dtype(self.dtype).ok_or_else(|| {
            anyhow::anyhow!(
                "FlashInferAttention does not support dtype {:?}",
                self.dtype
            )
        })?;
        let kv_dim = self.num_kv_heads * self.head_dim;
        let kv_page_bytes = kv_dim * dtype.size_of() * self.page_size;
        let max_kv_pages = k_buf
            .len()
            .checked_div(kv_page_bytes)
            .zip(v_buf.len().checked_div(kv_page_bytes))
            .map(|(k_pages, v_pages)| k_pages.min(v_pages).max(c))
            .unwrap_or(c);
        let batch_size = self
            .requests
            .exec(dyn_map)
            .ok_or_else(|| anyhow::anyhow!("FlashInferAttention request count is unresolved"))?;
        anyhow::ensure!(
            total_q_tokens > 0
                && batch_size > 0
                && total_q_tokens >= batch_size
                && self.supports_geometry(total_q_tokens > batch_size),
            "FlashInfer algorithm does not support the resolved dtype/geometry/phase"
        );
        let (kv_indptr_host, explicit_qo_indptr, explicit_kv_indptr) = if inputs.len() >= 6 {
            let rows = batch_size
                .checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("FlashInferAttention CSR row count overflows"))?;
            let bytes = rows
                .checked_mul(std::mem::size_of::<i32>())
                .ok_or_else(|| anyhow::anyhow!("FlashInferAttention CSR byte size overflows"))?;
            let qo = get_buf("qo_indptr", inputs[4])?;
            let kv = get_buf("kv_indptr", inputs[5])?;
            for (name, buffer) in [("qo_indptr", qo), ("kv_indptr", kv)] {
                anyhow::ensure!(
                    buffer.len() == bytes,
                    "FlashInferAttention {name} has {} bytes; compiled geometry requires {bytes} for {batch_size} requests",
                    buffer.len()
                );
            }
            // Contents are read during preparation, where synchronization is
            // allowed. Capture signatures retain the validated pointers.
            (Vec::with_capacity(rows), Some(qo.ptr()), Some(kv.ptr()))
        } else {
            anyhow::ensure!(
                batch_size == 1 || batch_size == total_q_tokens,
                "FlashInferAttention packed multi-request prefill requires explicit CSR metadata"
            );
            (Vec::new(), None, None)
        };
        let explicit_last_page_len = if inputs.len() == 7 {
            let last = get_buf("kv_last_page_len", inputs[6])?;
            let bytes = batch_size
                .checked_mul(std::mem::size_of::<i32>())
                .ok_or_else(|| {
                    anyhow::anyhow!("FlashInferAttention last-page metadata size overflows")
                })?;
            anyhow::ensure!(
                last.len() == bytes,
                "FlashInferAttention kv_last_page_len has {} bytes; compiled geometry requires {bytes} for {batch_size} requests",
                last.len()
            );
            Some(last.ptr())
        } else {
            None
        };

        let sm_scale = if self.sm_scale == 0.0 {
            1.0 / (self.head_dim as f32).sqrt()
        } else {
            self.sm_scale as f32
        };
        Ok(FlashInferResolvedAttention {
            spec: FlashInferPlanSpec {
                algorithm: self.algorithm,
                total_q_tokens,
                batch_size,
                c,
                num_qo_heads: self.num_qo_heads,
                num_kv_heads: self.num_kv_heads,
                page_size: self.page_size,
                head_dim: self.head_dim,
                kv_dim,
                max_kv_pages,
                dtype,
                sm_scale_bits: sm_scale.to_bits(),
                window_left: self.window_left as i32,
                kv_indptr_host,
                qo_indptr_host: Vec::new(),
            },
            ptrs: FlashInferPointers {
                q: q_buf.ptr(),
                k_cache: k_buf.ptr(),
                v_cache: v_buf.ptr(),
                gather_idx: gather_idx_buf.ptr(),
                output: out_buf.ptr(),
                explicit_qo_indptr,
                explicit_kv_indptr,
                explicit_last_page_len,
            },
        })
    }

    pub(crate) fn prepare_resolved_for_graph(
        &self,
        stream: &Arc<CudaStream>,
        mut resolved: FlashInferResolvedAttention,
        enable_cuda_graph: bool,
    ) -> anyhow::Result<PreparedFlashInferAttention> {
        let group_size = Self::group_size(self.num_qo_heads, self.num_kv_heads)?;
        let lib = jit::ensure_compiled(
            crate::target::CudaTarget::from_context(stream.context())?,
            self.head_dim,
            self.window_left >= 0,
            group_size,
        )?;
        let cu_stream = stream.cu_stream() as *mut std::ffi::c_void;
        let spec = &mut resolved.spec;
        let uses_tensor_cores = spec.uses_tensor_cores();
        if uses_tensor_cores && !spec.dtype.supports_prefill() {
            anyhow::bail!(
                "FlashInfer tensor-core attention requires f16/bf16; got {:?} with s={} batch={}",
                spec.dtype,
                spec.total_q_tokens,
                spec.batch_size
            );
        }

        let mut current_c = None;
        let mut current_c_ptr = None;

        let read_device_i32s = |ptr: u64, n: usize| -> anyhow::Result<Vec<i32>> {
            let mut values = vec![0_i32; n];
            unsafe {
                result::memcpy_dtoh_async(&mut values, ptr, stream.cu_stream())?;
            }
            stream.synchronize()?;
            Ok(values)
        };

        let (owned_kv_indptr, owned_kv_indptr_ptr) = if let Some(kv_indptr_ptr) =
            resolved.ptrs.explicit_kv_indptr
        {
            let r = spec.batch_size + 1;
            spec.kv_indptr_host = read_device_i32s(kv_indptr_ptr, r)?;
            (None, None)
        } else if uses_tensor_cores {
            // Tensor-core attention: s q tokens attending causally to a
            // c-token context whose last s slots are the q tokens themselves.
            spec.kv_indptr_host = vec![0, spec.c as i32];
            let dev = stream.clone_htod(&spec.kv_indptr_host)?;
            let ptr = dev.device_ptr(stream).0;
            (Some(dev), Some(ptr))
        } else if enable_cuda_graph && spec.total_q_tokens == 1 {
            let actual_c = spec.c;
            let plan_c = flashinfer_graph_plan_capacity(actual_c, spec.max_kv_pages);
            spec.c = plan_c;
            spec.kv_indptr_host = vec![0, plan_c as i32];
            let dev = stream.clone_htod(&[0i32, actual_c as i32])?;
            let ptr = dev.device_ptr(stream).0;
            let current = stream.clone_htod(&[actual_c as i32])?;
            current_c_ptr = Some(current.device_ptr(stream).0);
            current_c = Some(Mutex::new(current));
            (Some(dev), Some(ptr))
        } else {
            if enable_cuda_graph {
                anyhow::bail!(
                    "FlashInfer graph capture for derived causal decode currently requires s=1, got s={}",
                    spec.total_q_tokens
                );
            }
            if spec.total_q_tokens != 1 {
                anyhow::bail!(
                    "FlashInfer derived causal decode without explicit indptrs requires s=1, got s={} (f32 prefill is unsupported — tensor cores are 16-bit)",
                    spec.total_q_tokens
                );
            }
            spec.kv_indptr_host = vec![0, spec.c as i32];
            let dev = stream.clone_htod(&spec.kv_indptr_host)?;
            let ptr = dev.device_ptr(stream).0;
            (Some(dev), Some(ptr))
        };

        // Tensor-core decode and prefill need query segmentation for plan/run.
        let (owned_qo_indptr, owned_qo_indptr_ptr) = if uses_tensor_cores {
            if let Some(qo_indptr_ptr) = resolved.ptrs.explicit_qo_indptr {
                let r = spec.batch_size + 1;
                spec.qo_indptr_host = read_device_i32s(qo_indptr_ptr, r)?;
                (None, None)
            } else {
                spec.qo_indptr_host = vec![0, spec.total_q_tokens as i32];
                let dev = stream.clone_htod(&spec.qo_indptr_host)?;
                let ptr = dev.device_ptr(stream).0;
                (Some(dev), Some(ptr))
            }
        } else {
            (None, None)
        };

        let total_pages = spec.kv_indptr_host.last().copied().unwrap_or_default();
        if total_pages < 0 || total_pages as usize > spec.c {
            anyhow::bail!(
                "FlashInfer describes {total_pages} KV pages, but compact page indices contain only {}",
                spec.c
            );
        }

        let (indices, indices_ptr) = if resolved.ptrs.explicit_last_page_len.is_some() {
            (None, resolved.ptrs.gather_idx)
        } else {
            let buffer = unsafe { stream.alloc::<i32>(spec.c.max(1))? };
            let ptr = buffer.device_ptr(stream).0;
            (Some(buffer), ptr)
        };
        let (last_page_len, last_page_len_ptr) =
            if let Some(ptr) = resolved.ptrs.explicit_last_page_len {
                (None, ptr)
            } else {
                let values = vec![self.page_size as i32; spec.batch_size];
                let buffer = if values.is_empty() {
                    unsafe { stream.alloc::<i32>(1)? }
                } else {
                    stream.clone_htod(&values)?
                };
                let ptr = buffer.device_ptr(stream).0;
                (Some(buffer), ptr)
            };
        let temp_output_bytes =
            (spec.total_q_tokens * spec.num_qo_heads * spec.head_dim * spec.dtype.size_of()).max(1);
        let temp_output = unsafe { stream.alloc::<u8>(temp_output_bytes)? };
        let temp_output_ptr = temp_output.device_ptr(stream).0;

        let workspace = PlanWorkspace::new(stream)?;

        let mut plan_info_buf = [0i64; 16];
        let mut plan_info_len: i32 = 0;
        let plan_ret = with_plan_staging(stream, |staging| {
            if uses_tensor_cores {
                unsafe {
                    (lib.prefill_plan)(
                        workspace.float_ptr as *mut std::ffi::c_void,
                        FLOAT_WORKSPACE_SIZE,
                        workspace.int_ptr as *mut std::ffi::c_void,
                        workspace.metadata.len(),
                        staging,
                        spec.qo_indptr_host.as_mut_ptr(),
                        spec.kv_indptr_host.as_mut_ptr(),
                        spec.total_q_tokens as i32,
                        spec.batch_size as i32,
                        spec.num_qo_heads as i32,
                        spec.num_kv_heads as i32,
                        spec.page_size as i32,
                        spec.head_dim as i32,
                        spec.dtype as i32,
                        spec.window_left,
                        cu_stream,
                        plan_info_buf.as_mut_ptr(),
                        &mut plan_info_len,
                    )
                }
            } else {
                unsafe {
                    (lib.plan)(
                        workspace.float_ptr as *mut std::ffi::c_void,
                        FLOAT_WORKSPACE_SIZE,
                        workspace.int_ptr as *mut std::ffi::c_void,
                        workspace.metadata.len(),
                        staging,
                        spec.kv_indptr_host.as_mut_ptr(),
                        spec.batch_size as i32,
                        spec.num_qo_heads as i32,
                        spec.num_kv_heads as i32,
                        spec.page_size as i32,
                        spec.head_dim as i32,
                        spec.dtype as i32,
                        enable_cuda_graph,
                        cu_stream,
                        plan_info_buf.as_mut_ptr(),
                        &mut plan_info_len,
                    )
                }
            }
        })?;
        if plan_ret != 0 {
            return Err(flashinfer_native_error(
                lib,
                if uses_tensor_cores {
                    "prefill plan"
                } else {
                    "decode plan"
                },
                plan_ret,
            ));
        }

        Ok(PreparedFlashInferAttention {
            lib,
            spec: spec.clone(),
            plan_info: plan_info_buf[..plan_info_len as usize].to_vec(),
            workspace,
            _owned_kv_indptr: owned_kv_indptr,
            owned_kv_indptr_ptr,
            _owned_qo_indptr: owned_qo_indptr,
            owned_qo_indptr_ptr,
            current_c,
            current_c_ptr,
            _indices: indices,
            indices_ptr,
            _last_page_len: last_page_len,
            last_page_len_ptr,
            _temp_output: temp_output,
            temp_output_ptr,
        })
    }
}

fn flashinfer_native_error(lib: &jit::FlashInferLib, operation: &str, code: i32) -> anyhow::Error {
    lib.error_message().map_or_else(
        || anyhow::anyhow!("FlashInfer {operation} failed with error code {code}"),
        |message| {
            anyhow::anyhow!("FlashInfer {operation} failed with error code {code}: {message}")
        },
    )
}

impl PreparedFlashInferAttention {
    pub(crate) fn plan_c(&self) -> usize {
        self.spec.c
    }

    pub(crate) fn update_current_c(
        &self,
        stream: &Arc<CudaStream>,
        c: usize,
    ) -> anyhow::Result<()> {
        if let Some(current_c) = &self.current_c {
            let mut current_c = current_c
                .lock()
                .map_err(|_| anyhow::anyhow!("FlashInfer current_c lock poisoned"))?;
            stream.memcpy_htod(&[c as i32], &mut *current_c)?;
        }
        Ok(())
    }

    /// Enqueue the attention kernels onto `stream`.
    ///
    /// `include_metadata` controls whether the kv-metadata preparation kernel
    /// (indices/indptr/valid-mask derived from `gather_idx` + `current_c`) is
    /// launched first. CUDA-graph callers that share prepared allocations set
    /// this for every user: sharing is allowed only between dependency-ordered
    /// users, so each refresh completes before that user consumes the buffers.
    pub(crate) fn enqueue(
        &self,
        stream: &Arc<CudaStream>,
        ptrs: FlashInferPointers,
        include_metadata: bool,
    ) -> anyhow::Result<()> {
        let cu_stream = stream.cu_stream() as *mut std::ffi::c_void;
        let kv_indptr_ptr = ptrs
            .explicit_kv_indptr
            .or(self.owned_kv_indptr_ptr)
            .ok_or_else(|| anyhow::anyhow!("FlashInfer decode is missing kv_indptr pointer"))?;

        let mut plan_info = self.plan_info.clone();
        if !include_metadata {
            // A preceding dependency has already populated the metadata.
        } else if let Some(current_c_ptr) = self.current_c_ptr {
            let metadata_ret = unsafe {
                (self.lib.prepare_decode_metadata)(
                    self.workspace.int_ptr as *mut std::ffi::c_void,
                    plan_info.as_mut_ptr(),
                    plan_info.len() as i32,
                    current_c_ptr as *const i32,
                    ptrs.gather_idx as *const i32,
                    self.indices_ptr as *mut i32,
                    kv_indptr_ptr as *mut i32,
                    self.spec.c as i32,
                    self.spec.kv_dim as i32,
                    cu_stream,
                )
            };
            if metadata_ret != 0 {
                return Err(flashinfer_native_error(
                    self.lib,
                    "decode metadata preparation",
                    metadata_ret,
                ));
            }
        } else if self.spec.c > 0 && self._indices.is_some() {
            unsafe {
                (self.lib.extract_slot_indices)(
                    ptrs.gather_idx as *const i32,
                    self.indices_ptr as *mut i32,
                    self.spec.c as i32,
                    self.spec.kv_dim as i32,
                    cu_stream,
                );
            }
        }

        // At total_q_tokens == 1 the (batch, heads, dim) → (heads, batch, dim)
        // output transpose is a byte-identity, so the kernel writes the real
        // output buffer directly and the transpose launch is skipped — one
        // fewer graph node per attention island per step.
        let direct_output = self.spec.total_q_tokens == 1;
        let run_output_ptr = if direct_output {
            ptrs.output
        } else {
            self.temp_output_ptr
        };

        let run_ret = if self.spec.uses_tensor_cores() {
            let qo_indptr_ptr = ptrs
                .explicit_qo_indptr
                .or(self.owned_qo_indptr_ptr)
                .ok_or_else(|| anyhow::anyhow!("FlashInfer prefill is missing qo_indptr"))?;
            unsafe {
                (self.lib.prefill_run)(
                    self.workspace.float_ptr as *mut std::ffi::c_void,
                    FLOAT_WORKSPACE_SIZE,
                    self.workspace.int_ptr as *mut std::ffi::c_void,
                    plan_info.as_mut_ptr(),
                    plan_info.len() as i32,
                    ptrs.q as *mut std::ffi::c_void,
                    ptrs.k_cache as *mut std::ffi::c_void,
                    ptrs.v_cache as *mut std::ffi::c_void,
                    qo_indptr_ptr as *mut i32,
                    kv_indptr_ptr as *mut i32,
                    self.indices_ptr as *mut i32,
                    self.last_page_len_ptr as *mut i32,
                    run_output_ptr as *mut std::ffi::c_void,
                    self.spec.total_q_tokens as i32,
                    self.spec.batch_size as i32,
                    self.spec.num_qo_heads as i32,
                    self.spec.num_kv_heads as i32,
                    self.spec.page_size as i32,
                    self.spec.head_dim as i32,
                    self.spec.dtype as i32,
                    f32::from_bits(self.spec.sm_scale_bits),
                    self.spec.window_left,
                    cu_stream,
                )
            }
        } else {
            unsafe {
                (self.lib.run)(
                    self.workspace.float_ptr as *mut std::ffi::c_void,
                    FLOAT_WORKSPACE_SIZE,
                    self.workspace.int_ptr as *mut std::ffi::c_void,
                    plan_info.as_mut_ptr(),
                    plan_info.len() as i32,
                    ptrs.q as *mut std::ffi::c_void,
                    ptrs.k_cache as *mut std::ffi::c_void,
                    ptrs.v_cache as *mut std::ffi::c_void,
                    kv_indptr_ptr as *mut i32,
                    self.indices_ptr as *mut i32,
                    self.last_page_len_ptr as *mut i32,
                    run_output_ptr as *mut std::ffi::c_void,
                    self.spec.batch_size as i32,
                    self.spec.num_qo_heads as i32,
                    self.spec.num_kv_heads as i32,
                    self.spec.page_size as i32,
                    self.spec.head_dim as i32,
                    self.spec.dtype as i32,
                    f32::from_bits(self.spec.sm_scale_bits),
                    self.spec.window_left,
                    cu_stream,
                )
            }
        };

        if run_ret != 0 {
            return Err(flashinfer_native_error(
                self.lib,
                if self.spec.uses_tensor_cores() {
                    "prefill run"
                } else {
                    "decode run"
                },
                run_ret,
            ));
        }

        if !direct_output {
            unsafe {
                (self.lib.transpose_output)(
                    self.temp_output_ptr as *const std::ffi::c_void,
                    ptrs.output as *mut std::ffi::c_void,
                    self.spec.total_q_tokens as i32,
                    self.spec.num_qo_heads as i32,
                    self.spec.head_dim as i32,
                    self.spec.dtype as i32,
                    cu_stream,
                );
            }
        }

        Ok(())
    }
}

pub(crate) fn flashinfer_graph_plan_capacity(actual_c: usize, max_kv_pages: usize) -> usize {
    let required = actual_c.max(1);
    if let Some(capacity) = std::env::var("ORBITKV_FLASHINFER_DECODE_GRAPH_CAPACITY")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
    {
        return capacity.max(required);
    }
    // Tiered capacity instead of the full KV pool: planning at the pool size
    // (e.g. 4096) makes the decode kernel split KV into a padded grid that is
    // mostly invalid blocks at short contexts (measured 6.3µs decode + 2.3µs
    // merge per layer at c≈200 planned for 4096). Plan at the next power of
    // two of the current context (min 256); when c outgrows the tier, the
    // signature changes and the island recaptures with the next tier — a few
    // recaptures per sequence instead of µs lost on every step.
    required
        .next_power_of_two()
        .max(256)
        .min(max_kv_pages.max(required))
}

impl HostOp for FlashInferAttention {
    fn provider_dependencies(&self) -> Vec<super::registry::ProviderId> {
        vec![super::registry::ProviderId::FlashInfer]
    }

    fn prepare_compilation(
        &self,
        stream: &Arc<CudaStream>,
        _dyn_map: &DynMap,
    ) -> anyhow::Result<()> {
        let group_size = Self::group_size(self.num_qo_heads, self.num_kv_heads)?;
        let _ = jit::ensure_compiled(
            crate::target::CudaTarget::from_context(stream.context())?,
            self.head_dim,
            self.window_left >= 0,
            group_size,
        )?;
        Ok(())
    }

    fn execute(
        &self,
        stream: &Arc<CudaStream>,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &DynMap,
    ) -> anyhow::Result<()> {
        let resolved = self.resolve_for_graph(self_node, inputs, buffers, dyn_map)?;
        let ptrs = resolved.ptrs;
        let prepared = self.prepare_resolved_for_graph(stream, resolved, false)?;

        let _span = span!(
            Level::TRACE,
            "FlashInferAttention",
            prepared.spec.total_q_tokens,
            prepared.spec.batch_size,
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
        )
        .entered();
        prepared.enqueue(stream, ptrs, true)
    }

    fn output_size(&self) -> Expression {
        self.query_tokens * self.num_qo_heads * self.head_dim
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn cuda_graph_capture_arity(&self) -> Option<usize> {
        // The direct OrbitKV path supplies explicit qo_indptr, kv_indptr, and
        // last-page lengths. The structural rewrite keeps the legacy
        // four-input form and is handled by the specialized capture path.
        Some(7)
    }

    fn device_memory_plan(
        &self,
        _self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffer_lengths: &FxHashMap<NodeIndex, usize>,
        dyn_map: &DynMap,
    ) -> Result<HostDeviceMemoryPlan, ResourceViolation> {
        let resource_spec = self.device_resource_spec(inputs, buffer_lengths, dyn_map, false)?;
        Ok(HostDeviceMemoryPlan {
            transient_peak_bytes: resource_spec.prepared_device_bytes()?,
            shared_allocations: vec![shared_device_memory_allocation()],
            ..Default::default()
        })
    }

    fn resource_buffer_nodes(&self, inputs: &[NodeIndex]) -> Vec<NodeIndex> {
        // device_resource_spec derives max_kv_pages from the logical lengths
        // of K and V. Q, gather indices, and explicit indptr contents do not
        // alter the allocation plan.
        inputs.get(1..3).unwrap_or_default().to_vec()
    }

    fn stats_name(&self) -> Option<&'static str> {
        Some("FlashInferAttention")
    }
}

#[cfg(test)]
#[path = "../../tests/unit/providers/flashinfer/mod.rs"]
mod tests;
