//! Configuration-driven decoder graph using OrbitKV-managed paged attention.

use std::collections::BTreeSet;

use luminal::prelude::rand::SeedableRng;
use luminal::{
    dtype::DType,
    op::Runtime,
    prelude::{Expression, Graph, GraphTensor, Symbol, sym},
    shape::ToShape,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use luminal_cuda_lite::{
    cudarc::driver::{CudaContext, CudaSlice, CudaStream},
    runtime::{CapturedCudaExecution, CudaRuntime},
};

use crate::{
    ExecutorArena, ExecutorPlan, FixedStateArenaRegistration, FixedStateDeviceArenas,
    FixedStateExecutionEvidence, FixedStateGraphBinding, FixedStateGraphResource,
    FixedStateRuntimeBinding,
    cuda::{KvCacheBinding, PagedAttentionMetadata},
};
use orbitkv::{EngineFixedStatePlan, StatePoolIdentity};

#[path = "model/runtime_input.rs"]
mod runtime_input;
use runtime_input::{validate_stateful_decode, validate_step};
#[path = "model/artifact.rs"]
mod artifact;
pub use artifact::DecoderArtifact;
use artifact::{decoder_artifact_identity, new_artifact};
#[path = "model/compiler.rs"]
mod compiler;
use compiler::{bind_fixed_state, prepare_decoder_compilation};
#[path = "model/config.rs"]
mod config;
pub use config::{
    DecoderActivation, DecoderBlockLayout, DecoderConfig, DecoderLayerKind, DecoderNormWeights,
    DecoderWeightFormat, GatedDeltaConfig,
};
#[path = "model/topology.rs"]
mod topology;
use topology::DecoderTopology;
#[path = "model/recurrent_layer.rs"]
mod recurrent_layer;
pub use recurrent_layer::{
    GatedDeltaCore, GatedDeltaCoreOutput, GatedDeltaDecodeOutput, GatedDeltaProjection,
    GatedDeltaStateBindings, GatedDeltaStateGraph,
};
#[path = "model/weights.rs"]
mod weights;
use weights::{DecoderWeightFeatures, inspect_weight_features};

#[derive(Debug, Error)]
pub enum DecoderError {
    #[error("invalid decoder config JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("decoder geometry is invalid: {0}")]
    InvalidGeometry(&'static str),
    #[error("decoder attention classes do not exactly cover the model layers")]
    UnsupportedPlan,
    #[error("decoder execution does not yet support {0}")]
    UnsupportedExecution(&'static str),
    #[error(transparent)]
    Executor(#[from] crate::ExecutorError),
    #[error(transparent)]
    Device(#[from] luminal_cuda_lite::cudarc::driver::DriverError),
    #[error(transparent)]
    Recurrent(#[from] crate::RecurrentError),
    #[error(transparent)]
    Convolution(#[from] crate::ConvolutionError),
    #[error(transparent)]
    FixedState(#[from] crate::FixedStateDeviceError),
    #[error(transparent)]
    FixedStateGraph(#[from] crate::FixedStateGraphError),
    #[error("decoder artifact is incompatible: {0}")]
    Artifact(String),
    #[error("decoder runtime input geometry exceeds its compiled capacity")]
    InputCapacity,
    #[error("CUDA graph capture requires exactly one query token per request")]
    CaptureRequiresDecode,
    #[error("no decode CUDA graph has been captured")]
    MissingDecodeCapture,
    #[error("decode step does not match the captured CUDA graph signature")]
    DecodeCaptureMismatch,
}

/// Dynamic-shape and search policy for one compiled decoder executable.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DecoderCompileConfig {
    pub maximum_query_tokens: usize,
    pub representative_prefill_tokens: usize,
    pub maximum_batch_size: usize,
    pub maximum_context_pages: usize,
    pub representative_context_pages: usize,
    pub search_graphs: usize,
    pub search_seed: u64,
}

impl DecoderCompileConfig {
    fn validate(self) -> Result<(), DecoderError> {
        let int_bytes = std::mem::size_of::<i32>();
        if self.maximum_query_tokens < 2
            || !(2..=self.maximum_query_tokens).contains(&self.representative_prefill_tokens)
            || self.maximum_batch_size == 0
            || self.maximum_batch_size > self.maximum_query_tokens
            || self.maximum_context_pages == 0
            || !(1..=self.maximum_context_pages).contains(&self.representative_context_pages)
            || self.search_graphs < 2
            || self.maximum_query_tokens > i32::MAX as usize
            || self.maximum_batch_size > i32::MAX as usize
            || self.maximum_context_pages > i32::MAX as usize
            || self.maximum_query_tokens.checked_mul(int_bytes).is_none()
            || self.maximum_context_pages.checked_mul(int_bytes).is_none()
            || self
                .maximum_batch_size
                .checked_add(1)
                .and_then(|rows| rows.checked_mul(int_bytes))
                .is_none()
        {
            return Err(DecoderError::InvalidGeometry("compile buckets"));
        }
        Ok(())
    }
}

/// Inputs for one dispatch into the precompiled decoder buckets.
#[derive(Clone, Copy)]
pub struct DecoderStep<'a> {
    pub tokens: &'a [u32],
    pub positions: &'a [u32],
    pub classes: &'a [DecoderClassStep<'a>],
}

/// Manager-authored fixed-state transition for one request in a decoder step.
#[derive(Clone, Copy)]
pub struct DecoderFixedStateStep<'a> {
    pub request_id: u64,
    pub states: &'a [EngineFixedStatePlan],
}

/// Dynamic page metadata and write destinations for one compiled attention class.
#[derive(Clone, Copy)]
pub struct DecoderClassStep<'a> {
    pub class_id: u16,
    pub write_slots: &'a [u64],
    pub attention: &'a crate::AttentionBatch,
}

/// Persistent storage identities consumed while compiling one decoder.
#[derive(Clone, Copy)]
pub struct DecoderStorage<'a> {
    pub token_arenas: &'a [ExecutorArena],
    pub fixed_state_pools: &'a [(u16, StatePoolIdentity)],
}

impl<'a> DecoderStorage<'a> {
    #[must_use]
    pub const fn new(
        token_arenas: &'a [ExecutorArena],
        fixed_state_pools: &'a [(u16, StatePoolIdentity)],
    ) -> Self {
        Self {
            token_arenas,
            fixed_state_pools,
        }
    }

    #[must_use]
    pub const fn token_only(token_arenas: &'a [ExecutorArena]) -> Self {
        Self::new(token_arenas, &[])
    }
}

type InputAllocation = (GraphTensor, u64, usize);

#[derive(Clone, Debug, Eq, PartialEq)]
struct DecodeCaptureSignature {
    query_tokens: usize,
    batch_size: usize,
    classes: Box<[DecodeClassCaptureSignature]>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct DecodeClassCaptureSignature {
    class_id: u16,
    context_pages: usize,
    query_indptr: Box<[i32]>,
    page_indptr: Box<[i32]>,
}

struct CapturedDecode {
    signature: DecodeCaptureSignature,
    execution: CapturedCudaExecution,
}

/// One compiled graph with one persistent K/V arena per attention class.
pub struct CompiledDecoder {
    graph: Graph,
    decoder: DecoderGraph,
    captured_decode: Option<CapturedDecode>,
    runtime: CudaRuntime,
    persistent_cache: Vec<CudaSlice<u8>>,
    fixed_state: Option<CompiledFixedState>,
    compile: DecoderCompileConfig,
    vocabulary_size: usize,
    page_tokens: usize,
    cache_updates_in_place: bool,
    dynamic_input_allocations: Box<[InputAllocation]>,
}

#[derive(Clone)]
pub struct DecoderInputs {
    pub token_ids: GraphTensor,
    pub positions: GraphTensor,
    pub classes: Vec<DecoderClassInputs>,
}

#[derive(Clone, Copy)]
pub struct DecoderClassInputs {
    pub class_id: u16,
    pub write_slots: GraphTensor,
    pub attention: PagedAttentionMetadata,
}

pub struct DecoderOutputs {
    pub logits: GraphTensor,
    pub sampled_tokens: GraphTensor,
    cache: Vec<DecoderCacheState>,
    fixed_states: Box<[FixedStateGraphResource]>,
}

#[derive(Clone, Copy)]
struct DecoderCacheState {
    binding: KvCacheBinding,
    key_update: GraphTensor,
    value_update: GraphTensor,
}

struct CompiledFixedState {
    arenas: FixedStateDeviceArenas,
    graph_bindings: Box<[FixedStateGraphBinding]>,
    runtime_bindings: Box<[FixedStateRuntimeBinding]>,
}

struct DecoderCompilation {
    graph: Graph,
    decoder: DecoderGraph,
    runtime: CudaRuntime,
    persistent_cache: Vec<CudaSlice<u8>>,
    fixed_state_scratch: Vec<CudaSlice<u8>>,
    options: luminal::prelude::CompileOptions,
    identity: String,
    page_tokens: usize,
}

/// Greedy token IDs returned by the default device execution path.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecoderStepOutput {
    pub token_ids: Box<[u32]>,
}

/// Decoder output paired with device-completion evidence for manager-owned
/// recurrent and convolution state.
pub struct StatefulDecoderStepOutput {
    pub token_ids: Box<[u32]>,
    pub fixed_states: Box<[FixedStateExecutionEvidence]>,
}

/// Persistent K/V update behavior selected for one dynamic-shape bucket.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CacheUpdateBucket {
    pub bucket_index: usize,
    pub tensor_count: usize,
    pub in_place_tensors: usize,
    pub copy_back_tensors: usize,
    pub copy_back_bytes: usize,
}

/// Optional diagnostic readback used for correctness comparison.
///
/// Production decode should use [`CompiledDecoder::execute`], which transfers
/// only sampled token IDs.
#[derive(Clone, Debug, PartialEq)]
pub struct DecoderDiagnosticOutput {
    pub token_ids: Box<[u32]>,
    pub logits: Box<[f32]>,
}

pub struct DecoderGraph {
    pub inputs: DecoderInputs,
    pub outputs: DecoderOutputs,
    class_dimensions: Box<[DecoderClassDimensions]>,
}

#[derive(Clone, Copy)]
struct DecoderDimensions {
    query_tokens: Expression,
    request_count: Expression,
    kv_width: usize,
}

#[derive(Clone, Copy)]
struct DecoderClassDimensions {
    class_id: u16,
    context_pages: Symbol,
    backend_base_index: u64,
    page_count: u32,
    cache_slots: usize,
}

struct DecoderLayerGraphBuilder<'a> {
    graph: &'a mut Graph,
    config: &'a DecoderConfig,
    weights: DecoderWeightFeatures,
    plan: &'a ExecutorPlan,
    inputs: &'a DecoderInputs,
    dimensions: DecoderDimensions,
    class_dimensions: &'a [DecoderClassDimensions],
}

impl DecoderGraph {
    /// Returns the persistent K/V inputs for every token-attention layer.
    ///
    /// # Errors
    ///
    /// Rejects a plan that does not assign every returned binding exactly once.
    pub fn cache_bindings(
        &self,
        plan: &ExecutorPlan,
    ) -> Result<Box<[KvCacheBinding]>, DecoderError> {
        let mut bindings = Vec::with_capacity(self.outputs.cache.len());
        for state in &self.outputs.cache {
            let matches = plan.classes.iter().filter(|class| {
                class.class_id == state.binding.class_id
                    && class.layers.contains(&state.binding.layer)
            });
            if matches.count() != 1 {
                return Err(DecoderError::UnsupportedPlan);
            }
            bindings.push(state.binding);
        }
        Ok(bindings.into_boxed_slice())
    }

    /// Builds a decoder-only transformer whose layers are assigned to the
    /// attention classes compiled into `ExecutorPlan`.
    ///
    /// # Errors
    ///
    /// Rejects duplicate or incomplete layer coverage, incompatible arena or
    /// model geometry, and paged-attention construction failures.
    fn build(
        graph: &mut Graph,
        config: &DecoderConfig,
        weights: DecoderWeightFeatures,
        plan: &ExecutorPlan,
        arenas: &[ExecutorArena],
        fixed_state_registrations: &[FixedStateArenaRegistration],
    ) -> Result<Self, DecoderError> {
        let topology = DecoderTopology::compile(config, plan)?;
        let (dimensions, class_dimensions) = validate_plan(config, plan, arenas, &topology)?;
        let inputs = decoder_inputs(graph, &class_dimensions, dimensions);
        let embedding = weight(
            graph,
            format!("{}.embed_tokens.weight", config.tensor_prefix),
            (config.vocabulary_size, config.hidden_size),
            DType::Bf16,
        );
        let hidden = token_embedding(&embedding, &inputs.token_ids, config.hidden_size)
            * config.embedding_scale;
        let mut fixed_state = topology
            .has_fixed_state()
            .then(|| {
                GatedDeltaStateGraph::new(
                    graph,
                    config,
                    plan,
                    fixed_state_registrations,
                    dimensions.request_count,
                )
            })
            .transpose()?;
        let (hidden, cache) = DecoderLayerGraphBuilder {
            graph,
            config,
            weights,
            plan,
            inputs: &inputs,
            dimensions,
            class_dimensions: &class_dimensions,
        }
        .build(&topology, &mut fixed_state, hidden)?;
        let norm = DecoderNorm::new(
            graph,
            config,
            &format!("{}.norm.weight", config.tensor_prefix),
        );
        let normalized = norm.forward(&hidden);
        let lm_head = if config.tied_embeddings {
            embedding
        } else {
            weight(
                graph,
                "lm_head.weight",
                (config.vocabulary_size, config.hidden_size),
                DType::Bf16,
            )
        };
        let logits = normalized.matmul(lm_head.t()).cast(DType::F32);
        let sampled_tokens = logits.argmax(1).output();
        let logits = logits.output();
        Ok(Self {
            inputs,
            outputs: DecoderOutputs {
                logits,
                sampled_tokens,
                cache,
                fixed_states: fixed_state.map_or_else(
                    || Vec::new().into_boxed_slice(),
                    |states| Vec::from(states.finish().resources()).into_boxed_slice(),
                ),
            },
            class_dimensions,
        })
    }
}

impl DecoderLayerGraphBuilder<'_> {
    fn build(
        &mut self,
        topology: &DecoderTopology,
        fixed_state: &mut Option<GatedDeltaStateGraph>,
        mut hidden: GraphTensor,
    ) -> Result<(GraphTensor, Vec<DecoderCacheState>), DecoderError> {
        let mut cache = Vec::with_capacity(topology.token_layers().len());
        for layer_index in 0..self.config.layers {
            let layer = u32::try_from(layer_index)
                .map_err(|_| DecoderError::InvalidGeometry("layer index"))?;
            match topology
                .layer(layer_index)
                .ok_or(DecoderError::UnsupportedPlan)?
            {
                topology::DecoderLayerState::TokenKv { class_id } => {
                    let (next, state) = self.token_layer(layer, class_id, &hidden)?;
                    hidden = next;
                    cache.push(state);
                }
                topology::DecoderLayerState::GatedDelta { .. } => {
                    let envelope = DecoderLayerEnvelope::new(self.graph, self.config, layer_index);
                    let normalized = envelope.state_input(&hidden);
                    let state_output =
                        fixed_state
                            .as_mut()
                            .ok_or(DecoderError::UnsupportedPlan)?
                            .apply_decode_core(self.graph, self.config, layer, &normalized)?;
                    hidden = envelope.finish(&hidden, state_output);
                }
            }
        }
        Ok((hidden, cache))
    }

    fn token_layer(
        &mut self,
        layer: u32,
        class_id: u16,
        hidden: &GraphTensor,
    ) -> Result<(GraphTensor, DecoderCacheState), DecoderError> {
        let class = self
            .plan
            .classes
            .get(usize::from(class_id))
            .filter(|class| class.class_id == class_id)
            .ok_or(DecoderError::UnsupportedPlan)?;
        let dimensions = self
            .class_dimensions
            .get(usize::from(class_id))
            .filter(|dimensions| dimensions.class_id == class_id)
            .ok_or(DecoderError::UnsupportedPlan)?;
        let inputs = self
            .inputs
            .classes
            .get(usize::from(class_id))
            .filter(|inputs| inputs.class_id == class_id)
            .ok_or(DecoderError::UnsupportedPlan)?;
        let cache = |graph: &mut Graph, component: &str| {
            graph
                .named_tensor(
                    format!("kv.{layer}.{component}"),
                    (dimensions.cache_slots, self.dimensions.kv_width),
                )
                .persist()
                .as_dtype(DType::Bf16)
        };
        let key = cache(self.graph, "key");
        let value = cache(self.graph, "value");
        let block = TokenAttentionLayer::new(self.graph, self.config, self.weights, layer as usize);
        let (hidden, key_update, value_update) = block.forward(
            &TokenAttentionInputs {
                hidden,
                positions: &self.inputs.positions,
                write_slots: &inputs.write_slots,
                metadata: &inputs.attention,
                k_cache: &key,
                v_cache: &value,
            },
            class,
            self.config,
            self.dimensions,
            *dimensions,
        )?;
        Ok((
            hidden,
            DecoderCacheState {
                binding: KvCacheBinding {
                    class_id,
                    layer,
                    key,
                    value,
                },
                key_update: key_update.output(),
                value_update: value_update.output(),
            },
        ))
    }
}

impl CompiledDecoder {
    /// Builds and searches one decoder on a CUDA device selected by ordinal.
    ///
    /// This is the high-level composition boundary. Callers that do not need
    /// to coordinate another CUDA subsystem should use it instead of depending
    /// directly on Luminal's stream type.
    ///
    /// # Errors
    ///
    /// Returns device initialization or decoder compilation failures.
    pub fn compile_on_device(
        config: &DecoderConfig,
        plan: &ExecutorPlan,
        storage: DecoderStorage<'_>,
        device_index: usize,
        weight_files: &[std::path::PathBuf],
        compile: DecoderCompileConfig,
    ) -> Result<Self, DecoderError> {
        Self::compile_or_load_on_device(
            config,
            plan,
            storage,
            device_index,
            weight_files,
            compile,
            None,
        )
        .map(|(decoder, _)| decoder)
    }

    /// Builds a decoder from a stored schedule or searches a new schedule and
    /// returns the exact artifact selected for this executable.
    ///
    /// A supplied artifact is strict: identity or LLIR validation failure is
    /// returned to the caller and never falls back to a new search.
    ///
    /// # Errors
    ///
    /// Returns device, model, artifact, or compilation failures.
    pub fn compile_or_load_on_device(
        config: &DecoderConfig,
        plan: &ExecutorPlan,
        storage: DecoderStorage<'_>,
        device_index: usize,
        weight_files: &[std::path::PathBuf],
        compile: DecoderCompileConfig,
        artifact: Option<&DecoderArtifact>,
    ) -> Result<(Self, DecoderArtifact), DecoderError> {
        let context = CudaContext::new(device_index)?;
        let stream = context.new_stream()?;
        Self::compile_or_load(
            config,
            plan,
            storage,
            &stream,
            weight_files,
            compile,
            artifact,
        )
    }

    /// Builds and searches one decoder graph with decode and prefill buckets.
    ///
    /// The K/V arena is registered before search, so every candidate is
    /// measured against the deployed persistent-state contract. Each selected
    /// bucket either updates the registered arena in place or
    /// pays a graph-visible device copy back to that same stable address.
    ///
    /// # Errors
    ///
    /// Rejects invalid bucket geometry or incompatible model plans. Luminal
    /// compile failures currently surface through its native panic boundary.
    pub fn compile(
        config: &DecoderConfig,
        plan: &ExecutorPlan,
        storage: DecoderStorage<'_>,
        stream: &std::sync::Arc<CudaStream>,
        weight_files: &[std::path::PathBuf],
        compile: DecoderCompileConfig,
    ) -> Result<Self, DecoderError> {
        Self::compile_or_load(config, plan, storage, stream, weight_files, compile, None)
            .map(|(decoder, _)| decoder)
    }

    /// Compiles or strictly loads one selected decoder schedule.
    ///
    /// # Errors
    ///
    /// Rejects incompatible artifacts and propagates model or device failures.
    pub fn compile_or_load(
        config: &DecoderConfig,
        plan: &ExecutorPlan,
        storage: DecoderStorage<'_>,
        stream: &std::sync::Arc<CudaStream>,
        weight_files: &[std::path::PathBuf],
        compile: DecoderCompileConfig,
        artifact: Option<&DecoderArtifact>,
    ) -> Result<(Self, DecoderArtifact), DecoderError> {
        let prepared =
            prepare_decoder_compilation(config, plan, storage, stream, weight_files, compile)?;
        let DecoderCompilation {
            mut graph,
            decoder,
            mut runtime,
            mut persistent_cache,
            fixed_state_scratch,
            options,
            identity,
            page_tokens,
        } = prepared;
        if let Some(artifact) = artifact
            && artifact.identity != identity
        {
            return Err(DecoderError::Artifact(
                "model, plan, arena, or compile identity changed".into(),
            ));
        }
        let effective_artifact = if let Some(artifact) = artifact {
            graph.prepare_selected_schedule(&options);
            graph.install_selected_schedule(artifact.schedule.clone());
            graph
                .load_selected_schedule(&mut runtime)
                .map_err(DecoderError::Artifact)?;
            artifact.clone()
        } else {
            let mut rng =
                luminal::prelude::rand::rngs::SmallRng::seed_from_u64(compile.search_seed);
            runtime = graph.compile_with_rng(runtime, options, &mut rng);
            new_artifact(
                identity,
                graph
                    .selected_schedule()
                    .cloned()
                    .ok_or_else(|| DecoderError::Artifact("selected schedule missing".into()))?,
            )
        };
        let fixed_state = bind_fixed_state(
            plan,
            storage.fixed_state_pools,
            stream,
            &decoder.outputs.fixed_states,
            &mut runtime,
            fixed_state_scratch,
        )?;
        // Explicit-CSR attention may recapture library islands as context
        // geometry changes. Keep every searched bucket, but only one
        // materialized CUDA graph at a time so graph-pool reclamation cannot
        // leave an inactive phase holding stale captured resources. A bucket
        // switch rematerializes the target without repeating graph search.
        runtime.set_max_materialized_buckets(Some(1));
        runtime.release_pooled_memory();
        let dynamic_input_allocations = capture_input_allocations(&runtime, &decoder)?;
        let cache_updates_in_place = cache_updates_in_place(&runtime, &decoder);
        for cache in &mut persistent_cache {
            stream.memset_zeros(cache)?;
        }
        Ok((
            Self {
                graph,
                decoder,
                captured_decode: None,
                runtime,
                persistent_cache,
                fixed_state,
                compile,
                vocabulary_size: config.vocabulary_size,
                page_tokens,
                cache_updates_in_place,
                dynamic_input_allocations,
            },
            effective_artifact,
        ))
    }

    /// Executes one precompiled decode or prefill bucket and reads back only
    /// the on-device greedy token IDs.
    ///
    /// # Errors
    ///
    /// Rejects inconsistent token geometry or values exceeding compile-time
    /// capacities. Executor metadata validation errors are propagated.
    pub fn execute(&mut self, step: DecoderStep<'_>) -> Result<DecoderStepOutput, DecoderError> {
        self.execute_graph(step)?;
        Ok(DecoderStepOutput {
            token_ids: self.read_sampled_tokens(step.tokens.len())?,
        })
    }

    /// Executes one decode token per request and returns authenticated
    /// completion evidence for every fixed-state transition.
    ///
    /// # Errors
    ///
    /// Rejects prefill, missing or reordered state plans, and any device or
    /// graph execution failure.
    pub fn execute_with_fixed_states(
        &mut self,
        step: DecoderStep<'_>,
        states: &[DecoderFixedStateStep<'_>],
    ) -> Result<StatefulDecoderStepOutput, DecoderError> {
        validate_stateful_decode(step, states)?;
        let initialized = self
            .fixed_state
            .as_ref()
            .ok_or(DecoderError::UnsupportedExecution(
                "fixed-state decoder step",
            ))?
            .arenas
            .prepare_batch(states.iter().map(|state| (state.request_id, state.states)))?
            .initialize()?;
        self.prepare_graph(step)?;
        let fixed_state = self
            .fixed_state
            .as_ref()
            .ok_or(DecoderError::UnsupportedExecution(
                "fixed-state decoder step",
            ))?;
        let ready =
            initialized.upload_destination_slots(&mut self.runtime, &fixed_state.graph_bindings)?;
        let pending = ready.complete_after(
            &mut self.runtime,
            &self.graph,
            &fixed_state.runtime_bindings,
        )?;
        let fixed_states = pending.wait()?;
        Ok(StatefulDecoderStepOutput {
            token_ids: self.read_sampled_tokens(step.tokens.len())?,
            fixed_states,
        })
    }

    /// Warms and captures one fixed-signature decode execution.
    ///
    /// The warmup is the execution represented by the returned token. The
    /// subsequent stream capture records the already-prepared device work
    /// without executing it a second time. Input buffers remain outside the
    /// graph and keep their compile-time addresses.
    ///
    /// # Errors
    ///
    /// Rejects prefill, invalid geometry, unstable input allocations, and CUDA
    /// graph construction failures.
    pub fn capture_decode(
        &mut self,
        step: DecoderStep<'_>,
    ) -> Result<DecoderStepOutput, DecoderError> {
        let signature = DecodeCaptureSignature::from_step(step)?;
        self.captured_decode = None;
        self.execute_graph(step)?;
        let token_ids = self.read_sampled_tokens(step.tokens.len())?;
        let execution = self.runtime.capture_execution(&self.graph.dyn_map)?;
        self.captured_decode = Some(CapturedDecode {
            signature,
            execution,
        });
        Ok(DecoderStepOutput { token_ids })
    }

    /// Captures a decode and additionally reads the warmup logits for parity
    /// diagnostics. Production serving should use [`Self::capture_decode`].
    ///
    /// # Errors
    ///
    /// Propagates capture and output validation failures.
    pub fn capture_decode_with_logits(
        &mut self,
        step: DecoderStep<'_>,
    ) -> Result<DecoderDiagnosticOutput, DecoderError> {
        let output = self.capture_decode(step)?;
        let logits = self.runtime.get_f32(self.decoder.outputs.logits);
        let expected = step
            .tokens
            .len()
            .checked_mul(self.vocabulary_size)
            .ok_or(DecoderError::InputCapacity)?;
        if logits.len() != expected || logits.iter().any(|value| !value.is_finite()) {
            return Err(DecoderError::InvalidGeometry("logits output"));
        }
        Ok(DecoderDiagnosticOutput {
            token_ids: output.token_ids,
            logits: logits.into_boxed_slice(),
        })
    }

    /// Replays the captured decode graph after updating stable-address inputs.
    ///
    /// Page identities and the last-page token count may change. Query count,
    /// batch count, context-page count, and CSR indptr contents must exactly
    /// match the capture because they affect library planning and launch
    /// geometry. A mismatch requires [`Self::capture_decode`] again.
    ///
    /// # Errors
    ///
    /// Rejects missing captures, changed capture signatures, invalid inputs,
    /// unstable allocations, and CUDA launch failures.
    pub fn replay_decode(
        &mut self,
        step: DecoderStep<'_>,
    ) -> Result<DecoderStepOutput, DecoderError> {
        if self.fixed_state.is_some() {
            return Err(DecoderError::UnsupportedExecution(
                "fixed-state CUDA graph replay",
            ));
        }
        validate_step(
            step,
            self.compile,
            &self.decoder.class_dimensions,
            self.page_tokens,
            self.vocabulary_size,
        )?;
        let signature = DecodeCaptureSignature::from_step(step)?;
        let captured = self
            .captured_decode
            .as_ref()
            .ok_or(DecoderError::MissingDecodeCapture)?;
        if captured.signature != signature {
            return Err(DecoderError::DecodeCaptureMismatch);
        }
        self.bind_step_inputs(step)?;
        self.runtime
            .prepare_captured_execution(&self.graph.dyn_map)?;
        self.captured_decode
            .as_ref()
            .ok_or(DecoderError::MissingDecodeCapture)?
            .execution
            .launch()?;
        Ok(DecoderStepOutput {
            token_ids: self.read_sampled_tokens(step.tokens.len())?,
        })
    }

    /// Replays a captured decode and additionally reads logits for parity
    /// diagnostics. Production serving should use [`Self::replay_decode`].
    ///
    /// # Errors
    ///
    /// Propagates capture-signature, launch, and output validation failures.
    pub fn replay_decode_with_logits(
        &mut self,
        step: DecoderStep<'_>,
    ) -> Result<DecoderDiagnosticOutput, DecoderError> {
        let output = self.replay_decode(step)?;
        let logits = self.runtime.get_f32(self.decoder.outputs.logits);
        let expected = step
            .tokens
            .len()
            .checked_mul(self.vocabulary_size)
            .ok_or(DecoderError::InputCapacity)?;
        if logits.len() != expected || logits.iter().any(|value| !value.is_finite()) {
            return Err(DecoderError::InvalidGeometry("logits output"));
        }
        Ok(DecoderDiagnosticOutput {
            token_ids: output.token_ids,
            logits: logits.into_boxed_slice(),
        })
    }

    /// Executes one precompiled bucket and additionally reads logits for
    /// correctness diagnostics.
    ///
    /// # Errors
    ///
    /// Propagates step validation and execution-output failures. This path is
    /// not intended for the serving hot loop because it transfers full logits.
    pub fn execute_with_logits(
        &mut self,
        step: DecoderStep<'_>,
    ) -> Result<DecoderDiagnosticOutput, DecoderError> {
        self.execute_graph(step)?;
        let token_ids = self.read_sampled_tokens(step.tokens.len())?;
        let logits = self.runtime.get_f32(self.decoder.outputs.logits);
        let expected = step
            .tokens
            .len()
            .checked_mul(self.vocabulary_size)
            .ok_or(DecoderError::InputCapacity)?;
        if logits.len() != expected || logits.iter().any(|value| !value.is_finite()) {
            return Err(DecoderError::InvalidGeometry("logits output"));
        }
        Ok(DecoderDiagnosticOutput {
            token_ids,
            logits: logits.into_boxed_slice(),
        })
    }

    fn execute_graph(&mut self, step: DecoderStep<'_>) -> Result<(), DecoderError> {
        if self.fixed_state.is_some() {
            return Err(DecoderError::UnsupportedExecution(
                "fixed-state decoder step requires state plans",
            ));
        }
        self.prepare_graph(step)?;
        self.runtime.execute(&self.graph.dyn_map);
        Ok(())
    }

    fn prepare_graph(&mut self, step: DecoderStep<'_>) -> Result<(), DecoderError> {
        validate_step(
            step,
            self.compile,
            &self.decoder.class_dimensions,
            self.page_tokens,
            self.vocabulary_size,
        )?;
        // A same-signature eager execution keeps all captured pointers and
        // library plans valid. Prefill or changed decode geometry may replan
        // resources or switch buckets, so drop the old graph before allowing
        // that mutation. Invalid steps return above without disturbing an
        // otherwise reusable capture.
        let preserves_capture = self.captured_decode.as_ref().is_some_and(|captured| {
            DecodeCaptureSignature::from_step(step)
                .is_ok_and(|signature| signature == captured.signature)
        });
        if !preserves_capture {
            self.captured_decode = None;
        }
        self.bind_step_inputs(step)?;
        Ok(())
    }

    fn bind_step_inputs(&mut self, step: DecoderStep<'_>) -> Result<(), DecoderError> {
        self.graph.set_dim('s', step.tokens.len());
        let first_class = step.classes.first().ok_or(DecoderError::InputCapacity)?;
        self.graph.set_dim(
            'b',
            first_class
                .attention
                .query_indptr
                .len()
                .checked_sub(1)
                .ok_or(DecoderError::InputCapacity)?,
        );
        self.runtime.set_data(
            self.decoder.inputs.token_ids,
            step.tokens
                .iter()
                .map(|&token| i32::try_from(token).map_err(|_| DecoderError::InputCapacity))
                .collect::<Result<Vec<_>, _>>()?,
        );
        self.runtime.set_data(
            self.decoder.inputs.positions,
            step.positions
                .iter()
                .map(|&position| i32::try_from(position).map_err(|_| DecoderError::InputCapacity))
                .collect::<Result<Vec<_>, _>>()?,
        );
        for ((class_step, class_inputs), class_dimensions) in step
            .classes
            .iter()
            .zip(&self.decoder.inputs.classes)
            .zip(&self.decoder.class_dimensions)
        {
            self.graph.set_dim(
                class_dimensions.context_pages,
                class_step.attention.page_indices.len(),
            );
            self.runtime.set_data(
                class_inputs.write_slots,
                class_step
                    .write_slots
                    .iter()
                    .map(|&slot| i32::try_from(slot).map_err(|_| DecoderError::InputCapacity))
                    .collect::<Result<Vec<_>, _>>()?,
            );
            class_inputs
                .attention
                .upload(&mut self.runtime, class_step.attention)?;
        }
        if self.dynamic_input_allocations.iter().any(
            |(input, expected_pointer, expected_capacity)| {
                self.runtime.input_allocation(*input)
                    != Some((*expected_pointer, *expected_capacity))
            },
        ) {
            return Err(DecoderError::InvalidGeometry(
                "dynamic input address changed",
            ));
        }
        Ok(())
    }

    fn read_sampled_tokens(&self, rows: usize) -> Result<Box<[u32]>, DecoderError> {
        let raw = self.runtime.get_i32(self.decoder.outputs.sampled_tokens);
        if raw.len() < rows {
            return Err(DecoderError::InvalidGeometry("sampled token output"));
        }
        raw[..rows]
            .iter()
            .map(|&token| {
                u32::try_from(token)
                    .ok()
                    .filter(|&token| {
                        token < u32::try_from(self.vocabulary_size).unwrap_or(u32::MAX)
                    })
                    .ok_or(DecoderError::InvalidGeometry("sampled token output"))
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Vec::into_boxed_slice)
    }

    /// Returns stable persistent cache bindings for token-attention layers.
    ///
    /// # Errors
    ///
    /// Rejects a plan that does not assign every returned binding exactly once.
    pub fn cache_bindings(
        &self,
        plan: &ExecutorPlan,
    ) -> Result<Box<[KvCacheBinding]>, DecoderError> {
        self.decoder.cache_bindings(plan)
    }

    #[must_use]
    pub fn compile_config(&self) -> DecoderCompileConfig {
        self.compile
    }

    #[must_use]
    pub fn persistent_cache_count(&self) -> usize {
        self.persistent_cache.len()
    }

    /// Whether every selected bucket writes directly into the persistent K/V
    /// arena. `false` means Luminal selected a materializing update followed by
    /// the registered device-to-device copy back into the same stable arena.
    #[must_use]
    pub const fn cache_updates_in_place(&self) -> bool {
        self.cache_updates_in_place
    }

    /// Reports persistent K/V mutation behavior for every compiled bucket.
    ///
    /// A non-aliasing update is registered back to the same stable arena and
    /// therefore requires a graph-visible full-tensor device copy.
    #[must_use]
    pub fn cache_update_buckets(&self) -> Box<[CacheUpdateBucket]> {
        (0..self.runtime.compiled_bucket_count())
            .map(|bucket_index| {
                let mut in_place_tensors = 0usize;
                let mut copy_back_bytes = 0usize;
                for (layer, state) in self.decoder.outputs.cache.iter().enumerate() {
                    for (component, (output, input)) in [
                        (state.key_update, state.binding.key),
                        (state.value_update, state.binding.value),
                    ]
                    .into_iter()
                    .enumerate()
                    {
                        if self
                            .runtime
                            .output_aliases_input_in_bucket(output, input, bucket_index)
                            == Some(true)
                        {
                            in_place_tensors += 1;
                        } else {
                            copy_back_bytes += self.persistent_cache[layer * 2 + component].len();
                        }
                    }
                }
                let tensor_count = self.decoder.outputs.cache.len() * 2;
                CacheUpdateBucket {
                    bucket_index,
                    tensor_count,
                    in_place_tensors,
                    copy_back_tensors: tensor_count - in_place_tensors,
                    copy_back_bytes,
                }
            })
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    /// Number of decode/prefill executables selected during the one-time
    /// bucketed compile.
    #[must_use]
    pub fn compiled_bucket_count(&self) -> usize {
        self.runtime.compiled_bucket_count()
    }

    /// Bucket used by the most recent execution.
    #[must_use]
    pub fn active_bucket_index(&self) -> usize {
        self.runtime.active_bucket_index()
    }

    /// Whether a fixed-signature decode CUDA graph is ready for replay.
    #[must_use]
    pub const fn has_captured_decode(&self) -> bool {
        self.captured_decode.is_some()
    }
}

fn register_persistent_cache(
    runtime: &mut CudaRuntime,
    decoder: &DecoderGraph,
    plan: &ExecutorPlan,
    config: &DecoderConfig,
) -> Result<Vec<CudaSlice<u8>>, DecoderError> {
    let bindings = decoder.cache_bindings(plan)?;
    decoder
        .outputs
        .cache
        .iter()
        .zip(&bindings)
        .map(|(state, binding)| {
            let cache_bytes = decoder
                .class_dimensions
                .get(usize::from(binding.class_id))
                .filter(|class| class.class_id == binding.class_id)
                .ok_or(DecoderError::UnsupportedPlan)?
                .cache_slots
                .checked_mul(config.kv_heads)
                .and_then(|elements| elements.checked_mul(config.head_dim))
                .and_then(|elements| elements.checked_mul(2))
                .ok_or(DecoderError::InvalidGeometry("cache bytes"))?;
            Ok([
                runtime.alias_state_required(state.binding.key, state.key_update, cache_bytes),
                runtime.alias_state_required(state.binding.value, state.value_update, cache_bytes),
            ])
        })
        .collect::<Result<Vec<_>, DecoderError>>()
        .map(|caches| caches.into_iter().flatten().collect())
}

fn decoder_compile_options(
    decoder: &DecoderGraph,
    compile: DecoderCompileConfig,
    stateful: bool,
) -> luminal::prelude::CompileOptions {
    let query_buckets = if stateful {
        vec![luminal::prelude::DimBucket::new(1, compile.maximum_batch_size).representative(1)]
    } else {
        vec![
            luminal::prelude::DimBucket::new(1, 1),
            luminal::prelude::DimBucket::new(2, compile.maximum_query_tokens)
                .representative(compile.representative_prefill_tokens),
        ]
    };
    let options = luminal::prelude::CompileOptions::default().dim_buckets('s', &query_buckets);
    let options = if stateful {
        options
    } else {
        options.dim_buckets(
            'b',
            &[luminal::prelude::DimBucket::new(1, compile.maximum_batch_size).representative(1)],
        )
    };
    decoder
        .class_dimensions
        .iter()
        .fold(options, |options, class| {
            options.dim_buckets(
                class.context_pages,
                &[
                    luminal::prelude::DimBucket::new(1, compile.maximum_context_pages)
                        .representative(compile.representative_context_pages),
                ],
            )
        })
        .search_graph_limit(compile.search_graphs)
}

impl DecodeCaptureSignature {
    fn from_step(step: DecoderStep<'_>) -> Result<Self, DecoderError> {
        let Some(first_class) = step.classes.first() else {
            return Err(DecoderError::CaptureRequiresDecode);
        };
        let batch_size = first_class
            .attention
            .query_indptr
            .len()
            .checked_sub(1)
            .ok_or(DecoderError::InputCapacity)?;
        let decode_indptr = first_class.attention.query_indptr.first() == Some(&0)
            && first_class.attention.query_indptr.last().copied() == i32::try_from(batch_size).ok()
            && first_class
                .attention
                .query_indptr
                .windows(2)
                .all(|row| row[1] == row[0] + 1);
        if batch_size == 0
            || step.tokens.len() != batch_size
            || step.positions.len() != batch_size
            || !decode_indptr
            || step.classes.iter().any(|class| {
                class.write_slots.len() != batch_size
                    || class.attention.query_indptr != first_class.attention.query_indptr
            })
        {
            return Err(DecoderError::CaptureRequiresDecode);
        }
        Ok(Self {
            query_tokens: step.tokens.len(),
            batch_size,
            classes: step
                .classes
                .iter()
                .map(|class| DecodeClassCaptureSignature {
                    class_id: class.class_id,
                    context_pages: class.attention.page_indices.len(),
                    query_indptr: class.attention.query_indptr.clone(),
                    page_indptr: class.attention.page_indptr.clone(),
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        })
    }
}

fn dynamic_inputs(decoder: &DecoderGraph) -> Vec<GraphTensor> {
    let mut inputs = vec![decoder.inputs.token_ids, decoder.inputs.positions];
    for class in &decoder.inputs.classes {
        inputs.extend([
            class.write_slots,
            class.attention.page_indices,
            class.attention.query_indptr,
            class.attention.page_indptr,
            class.attention.last_page_len,
        ]);
    }
    inputs.extend(
        decoder
            .outputs
            .fixed_states
            .iter()
            .map(|state| state.binding.destination_slots),
    );
    inputs
}

fn capture_input_allocations(
    runtime: &CudaRuntime,
    decoder: &DecoderGraph,
) -> Result<Box<[InputAllocation]>, DecoderError> {
    dynamic_inputs(decoder)
        .into_iter()
        .map(|input| {
            let (pointer, capacity) = runtime
                .input_allocation(input)
                .ok_or(DecoderError::InvalidGeometry("dynamic input allocation"))?;
            Ok((input, pointer, capacity))
        })
        .collect::<Result<Vec<_>, DecoderError>>()
        .map(Vec::into_boxed_slice)
}

fn cache_updates_in_place(runtime: &CudaRuntime, decoder: &DecoderGraph) -> bool {
    decoder.outputs.cache.iter().all(|state| {
        runtime.output_aliases_input_in_all_buckets(state.key_update, state.binding.key)
            && runtime.output_aliases_input_in_all_buckets(state.value_update, state.binding.value)
    })
}

fn seed_compile_inputs(
    runtime: &mut CudaRuntime,
    decoder: &DecoderGraph,
    compile: DecoderCompileConfig,
    page_tokens: usize,
    representative_query_tokens: usize,
) {
    let int_bytes = std::mem::size_of::<i32>();
    runtime.set_data_with_capacity(
        decoder.inputs.token_ids,
        vec![1_i32; representative_query_tokens],
        compile.maximum_query_tokens * int_bytes,
    );
    runtime.set_data_with_capacity(
        decoder.inputs.positions,
        (0..i32::try_from(representative_query_tokens).unwrap()).collect::<Vec<_>>(),
        compile.maximum_query_tokens * int_bytes,
    );
    for (class, dimensions) in decoder.inputs.classes.iter().zip(&decoder.class_dimensions) {
        let base_page = i32::try_from(dimensions.backend_base_index).unwrap();
        let base_slot = dimensions
            .backend_base_index
            .checked_mul(page_tokens as u64)
            .and_then(|slot| i32::try_from(slot).ok())
            .unwrap();
        runtime.set_data_with_capacity(
            class.write_slots,
            (0..i32::try_from(representative_query_tokens).unwrap())
                .map(|offset| base_slot.checked_add(offset).unwrap())
                .collect::<Vec<_>>(),
            compile.maximum_query_tokens * int_bytes,
        );
        runtime.set_data_with_capacity(
            class.attention.page_indices,
            vec![base_page; compile.representative_context_pages],
            compile.maximum_context_pages * int_bytes,
        );
        runtime.set_data_with_capacity(
            class.attention.query_indptr,
            vec![0_i32, i32::try_from(representative_query_tokens).unwrap()],
            (compile.maximum_batch_size + 1) * int_bytes,
        );
        runtime.set_data_with_capacity(
            class.attention.page_indptr,
            vec![
                0_i32,
                i32::try_from(compile.representative_context_pages).unwrap(),
            ],
            (compile.maximum_batch_size + 1) * int_bytes,
        );
        runtime.set_data_with_capacity(
            class.attention.last_page_len,
            vec![i32::try_from(page_tokens.min(representative_query_tokens)).unwrap()],
            compile.maximum_batch_size * int_bytes,
        );
    }
}

fn validate_plan(
    config: &DecoderConfig,
    plan: &ExecutorPlan,
    arenas: &[ExecutorArena],
    topology: &DecoderTopology,
) -> Result<(DecoderDimensions, Box<[DecoderClassDimensions]>), DecoderError> {
    let layer_count =
        u32::try_from(config.layers).map_err(|_| DecoderError::InvalidGeometry("layer count"))?;
    crate::validate_arenas(arenas, plan.classes.len())?;
    let component_bytes = u64::try_from(config.kv_heads * config.head_dim * 2)
        .map_err(|_| DecoderError::InvalidGeometry("component bytes per token"))?;
    let mut layers = BTreeSet::new();
    let mut class_dimensions = Vec::with_capacity(plan.classes.len());
    for class in &plan.classes {
        let arena = arenas
            .get(usize::from(class.class_id))
            .filter(|arena| arena.class_id == class.class_id)
            .ok_or(DecoderError::UnsupportedPlan)?;
        let physical_pages = arena
            .backend_base_index
            .checked_add(u64::from(arena.page_count))
            .and_then(|pages| usize::try_from(pages).ok())
            .ok_or(DecoderError::InvalidGeometry("backend page range"))?;
        let page_tokens = usize::try_from(class.page_tokens)
            .map_err(|_| DecoderError::InvalidGeometry("page tokens"))?;
        let cache_slots = physical_pages
            .checked_mul(page_tokens)
            .ok_or(DecoderError::InvalidGeometry("cache slots"))?;
        if class.page_tokens != plan.page_tokens
            || class.key_bytes_per_token_per_layer != component_bytes
            || class.value_bytes_per_token_per_layer != component_bytes
            || physical_pages == 0
            || physical_pages > i32::MAX as usize
            || cache_slots > i32::MAX as usize
            || class
                .layers
                .iter()
                .any(|&layer| layer >= layer_count || !layers.insert(layer))
        {
            return Err(DecoderError::UnsupportedPlan);
        }
        class_dimensions.push(DecoderClassDimensions {
            class_id: class.class_id,
            context_pages: sym(&format!("c_{}", class.class_id)),
            backend_base_index: arena.backend_base_index,
            page_count: arena.page_count,
            cache_slots,
        });
    }
    if layers != topology.token_layers() {
        return Err(DecoderError::UnsupportedPlan);
    }
    Ok((
        DecoderDimensions {
            query_tokens: Expression::from('s'),
            request_count: if topology.has_fixed_state() {
                Expression::from('s')
            } else {
                Expression::from('b')
            },
            kv_width: config
                .kv_heads
                .checked_mul(config.head_dim)
                .ok_or(DecoderError::InvalidGeometry("KV width"))?,
        },
        class_dimensions.into_boxed_slice(),
    ))
}

fn decoder_inputs(
    graph: &mut Graph,
    classes: &[DecoderClassDimensions],
    dimensions: DecoderDimensions,
) -> DecoderInputs {
    let token_ids = graph
        .named_tensor("tokens", dimensions.query_tokens)
        .as_dtype(DType::Int);
    let positions = graph
        .named_tensor("positions", dimensions.query_tokens)
        .as_dtype(DType::Int);
    let classes = classes
        .iter()
        .map(|class| DecoderClassInputs {
            class_id: class.class_id,
            write_slots: graph
                .named_tensor(
                    format!("kv.{}.write_slots", class.class_id),
                    dimensions.query_tokens,
                )
                .as_dtype(DType::Int),
            attention: PagedAttentionMetadata::new(
                graph,
                class.class_id,
                dimensions.request_count,
                Expression::from(class.context_pages),
            ),
        })
        .collect();
    DecoderInputs {
        token_ids,
        positions,
        classes,
    }
}

#[path = "model/block.rs"]
mod block;
use block::{DecoderLayerEnvelope, DecoderNorm, TokenAttentionInputs, TokenAttentionLayer};

fn token_embedding(table: &GraphTensor, tokens: &GraphTensor, hidden: usize) -> GraphTensor {
    let count = tokens.dims1();
    table.gather(
        (*tokens * hidden).expand_dim(1, hidden)
            + tokens.graph().arange(hidden).expand_dim(0, count),
    )
}

pub(super) fn weight(
    graph: &mut Graph,
    name: impl ToString,
    shape: impl ToShape,
    dtype: DType,
) -> GraphTensor {
    graph.named_tensor(name, shape).persist().as_dtype(dtype)
}

#[cfg(test)]
#[path = "model/tests.rs"]
mod tests;
