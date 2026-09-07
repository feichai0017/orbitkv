//! Configuration-driven decoder graph using OrbitKV-managed paged attention.

use std::collections::BTreeSet;

use luminal::prelude::rand::SeedableRng;
use luminal::{
    dtype::DType,
    graph::SelectedSchedule,
    op::Runtime,
    prelude::{Expression, Graph, GraphTensor, Symbol, sym},
    shape::ToShape,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use luminal_cuda_lite::{
    cudarc::driver::{CudaContext, CudaSlice, CudaStream},
    runtime::{CapturedCudaExecution, CudaRuntime},
};

use crate::{
    ExecutorArena, ExecutorPlan, RelocationBatch,
    cuda::{CudaRelocationError, KvCacheBinding, PagedAttentionMetadata, PendingRelocationCopy},
};

#[path = "model/runtime_input.rs"]
mod runtime_input;
use runtime_input::validate_step;
#[path = "model/config.rs"]
mod config;
pub use config::{
    DecoderActivation, DecoderAttentionKind, DecoderBlockLayout, DecoderConfig, DecoderNormWeights,
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
    #[error(transparent)]
    Executor(#[from] crate::ExecutorError),
    #[error(transparent)]
    Device(#[from] luminal_cuda_lite::cudarc::driver::DriverError),
    #[error(transparent)]
    Relocation(#[from] CudaRelocationError),
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

const DECODER_ARTIFACT_SCHEMA: u32 = 1;

/// Portable graph-selection artifact for one native decoder configuration.
///
/// It contains no weights, device pointers, or KV contents. The identity binds
/// the selected Luminal schedule to the canonical manifest, model semantics,
/// weight-family geometry, physical arena shape, and compile buckets.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecoderArtifact {
    schema: u32,
    identity: String,
    schedule: SelectedSchedule,
}

#[derive(Serialize)]
struct DecoderArtifactIdentity<'a> {
    manifest_fingerprint: &'a str,
    compiler_facts_digest: &'a str,
    page_tokens: u32,
    decoder: &'a DecoderConfig,
    weights: DecoderWeightFeatures,
    arenas: Vec<DecoderArtifactArena>,
    compile: DecoderCompileConfig,
}

#[derive(Serialize)]
struct DecoderArtifactArena {
    class_id: u16,
    backend_base_index: u64,
    page_count: u32,
}

impl DecoderArtifact {
    /// Serializes the artifact as compact JSON bytes.
    ///
    /// # Errors
    ///
    /// Returns serialization failures without emitting partial data.
    pub fn to_bytes(&self) -> Result<Vec<u8>, DecoderError> {
        serde_json::to_vec(self).map_err(DecoderError::from)
    }

    /// Parses a decoder artifact. Compatibility is checked when it is loaded
    /// against a concrete model and executor plan.
    ///
    /// # Errors
    ///
    /// Rejects malformed JSON or an unknown artifact schema.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DecoderError> {
        let artifact = serde_json::from_slice::<Self>(bytes)?;
        if artifact.schema != DECODER_ARTIFACT_SCHEMA {
            return Err(DecoderError::Artifact(format!(
                "schema {} != {DECODER_ARTIFACT_SCHEMA}",
                artifact.schema
            )));
        }
        Ok(artifact)
    }
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

/// Dynamic page metadata and write destinations for one compiled attention class.
#[derive(Clone, Copy)]
pub struct DecoderClassStep<'a> {
    pub class_id: u16,
    pub write_slots: &'a [u64],
    pub attention: &'a crate::AttentionBatch,
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
    pub cache_inputs: Vec<(GraphTensor, GraphTensor)>,
    pub cache_updates: Vec<(GraphTensor, GraphTensor)>,
}

/// Greedy token IDs returned by the default device execution path.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecoderStepOutput {
    pub token_ids: Box<[u32]>,
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

impl DecoderGraph {
    /// Returns the persistent K/V inputs indexed by compiled class and layer.
    ///
    /// # Errors
    ///
    /// Rejects a plan that does not assign every decoder layer exactly once.
    pub fn cache_bindings(
        &self,
        plan: &ExecutorPlan,
    ) -> Result<Box<[KvCacheBinding]>, DecoderError> {
        let mut bindings = Vec::with_capacity(self.outputs.cache_updates.len());
        for (layer, &(key, value)) in self.outputs.cache_inputs.iter().enumerate() {
            let layer =
                u32::try_from(layer).map_err(|_| DecoderError::InvalidGeometry("layer index"))?;
            let mut classes = plan
                .classes
                .iter()
                .filter(|class| class.layers.contains(&layer));
            let class = classes.next().ok_or(DecoderError::UnsupportedPlan)?;
            if classes.next().is_some() {
                return Err(DecoderError::UnsupportedPlan);
            }
            bindings.push(KvCacheBinding {
                class_id: class.class_id,
                layer,
                key,
                value,
            });
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
    ) -> Result<Self, DecoderError> {
        let (dimensions, class_dimensions) = validate_plan(config, plan, arenas)?;
        let inputs = decoder_inputs(graph, &class_dimensions, dimensions);
        let embedding = weight(
            graph,
            "model.embed_tokens.weight",
            (config.vocabulary_size, config.hidden_size),
            DType::Bf16,
        );
        let mut hidden = token_embedding(&embedding, &inputs.token_ids, config.hidden_size)
            * config.embedding_scale;
        let mut cache_inputs = Vec::with_capacity(config.layers);
        let mut cache_updates = Vec::with_capacity(config.layers);
        for layer in 0..config.layers {
            let layer =
                u32::try_from(layer).map_err(|_| DecoderError::InvalidGeometry("layer index"))?;
            let class = class_for_layer(plan, layer)?;
            let class_dimensions = class_dimensions
                .get(usize::from(class.class_id))
                .filter(|dimensions| dimensions.class_id == class.class_id)
                .ok_or(DecoderError::UnsupportedPlan)?;
            let class_inputs = inputs
                .classes
                .get(usize::from(class.class_id))
                .filter(|inputs| inputs.class_id == class.class_id)
                .ok_or(DecoderError::UnsupportedPlan)?;
            let k_cache = graph
                .named_tensor(
                    format!("kv.{layer}.key"),
                    (class_dimensions.cache_slots, dimensions.kv_width),
                )
                .persist()
                .as_dtype(DType::Bf16);
            let v_cache = graph
                .named_tensor(
                    format!("kv.{layer}.value"),
                    (class_dimensions.cache_slots, dimensions.kv_width),
                )
                .persist()
                .as_dtype(DType::Bf16);
            let block = DecoderLayer::new(
                graph,
                config,
                weights,
                usize::try_from(layer).map_err(|_| DecoderError::InvalidGeometry("layer index"))?,
            );
            let (next, key_update, value_update) = block.forward(
                &LayerInputs {
                    hidden: &hidden,
                    positions: &inputs.positions,
                    write_slots: &class_inputs.write_slots,
                    metadata: &class_inputs.attention,
                    k_cache: &k_cache,
                    v_cache: &v_cache,
                },
                class,
                config,
                dimensions,
                *class_dimensions,
            )?;
            hidden = next;
            cache_inputs.push((k_cache, v_cache));
            cache_updates.push((key_update.output(), value_update.output()));
        }
        let norm = DecoderNorm::new(graph, config, "model.norm.weight");
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
                cache_inputs,
                cache_updates,
            },
            class_dimensions,
        })
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
        arenas: &[ExecutorArena],
        device_index: usize,
        weight_files: &[std::path::PathBuf],
        compile: DecoderCompileConfig,
    ) -> Result<Self, DecoderError> {
        Self::compile_or_load_on_device(
            config,
            plan,
            arenas,
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
    #[allow(clippy::too_many_arguments)]
    pub fn compile_or_load_on_device(
        config: &DecoderConfig,
        plan: &ExecutorPlan,
        arenas: &[ExecutorArena],
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
            arenas,
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
        arenas: &[ExecutorArena],
        stream: &std::sync::Arc<CudaStream>,
        weight_files: &[std::path::PathBuf],
        compile: DecoderCompileConfig,
    ) -> Result<Self, DecoderError> {
        Self::compile_or_load(config, plan, arenas, stream, weight_files, compile, None)
            .map(|(decoder, _)| decoder)
    }

    /// Compiles or strictly loads one selected decoder schedule.
    ///
    /// # Errors
    ///
    /// Rejects incompatible artifacts and propagates model or device failures.
    #[allow(clippy::too_many_arguments)]
    pub fn compile_or_load(
        config: &DecoderConfig,
        plan: &ExecutorPlan,
        arenas: &[ExecutorArena],
        stream: &std::sync::Arc<CudaStream>,
        weight_files: &[std::path::PathBuf],
        compile: DecoderCompileConfig,
        artifact: Option<&DecoderArtifact>,
    ) -> Result<(Self, DecoderArtifact), DecoderError> {
        compile.validate()?;
        if weight_files.is_empty() {
            return Err(DecoderError::InvalidGeometry("weight files"));
        }
        let mut graph = Graph::default();
        let weights = inspect_weight_features(weight_files, config)?;
        let compiler_facts = plan.luminal_compiler_facts(arenas)?;
        let identity = decoder_artifact_identity(
            config,
            plan,
            arenas,
            weights,
            compile,
            compiler_facts.digest(),
        )?;
        if let Some(artifact) = artifact
            && artifact.identity != identity
        {
            return Err(DecoderError::Artifact(
                "model, plan, arena, or compile identity changed".into(),
            ));
        }
        let decoder = DecoderGraph::build(&mut graph, config, weights, plan, arenas)?;
        let page_tokens = usize::try_from(plan.page_tokens)
            .map_err(|_| DecoderError::InvalidGeometry("page tokens"))?;
        if page_tokens > i32::MAX as usize {
            return Err(DecoderError::InvalidGeometry("backend index range"));
        }
        let mut runtime = CudaRuntime::initialize(stream.clone());
        for weights_path in weight_files {
            runtime.load_safetensors(
                &graph,
                weights_path
                    .to_str()
                    .ok_or(DecoderError::InvalidGeometry("weights path"))?,
            );
        }
        let mut persistent_cache = register_persistent_cache(&mut runtime, &decoder, plan, config)?;

        graph.set_dim('s', compile.representative_prefill_tokens);
        graph.set_dim('b', 1);
        for class in &decoder.class_dimensions {
            graph.set_dim(class.context_pages, compile.representative_context_pages);
        }
        seed_compile_inputs(&mut runtime, &decoder, compile, page_tokens);
        let options = decoder_compile_options(&decoder, compile)
            .compiler_facts(compiler_facts.egglog().to_owned());
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
            DecoderArtifact {
                schema: DECODER_ARTIFACT_SCHEMA,
                identity,
                schedule: graph
                    .selected_schedule()
                    .cloned()
                    .ok_or_else(|| DecoderError::Artifact("selected schedule missing".into()))?,
            }
        };
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
        self.runtime.execute(&self.graph.dyn_map);
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

    /// Returns the persistent cache bindings for relocation execution.
    ///
    /// # Errors
    ///
    /// Rejects a plan that does not assign every decoder layer exactly once.
    pub fn cache_bindings(
        &self,
        plan: &ExecutorPlan,
    ) -> Result<Box<[KvCacheBinding]>, DecoderError> {
        self.decoder.cache_bindings(plan)
    }

    /// Enqueues manager-authored relocation on the same stream and persistent
    /// K/V arena used by model execution.
    ///
    /// # Errors
    ///
    /// Propagates binding validation, lowering, and CUDA submission failures.
    pub fn enqueue_relocation(
        &self,
        batch: &RelocationBatch,
        plan: &ExecutorPlan,
    ) -> Result<PendingRelocationCopy, DecoderError> {
        let bindings = self.cache_bindings(plan)?;
        if bindings.len().checked_mul(2) != Some(self.persistent_cache.len()) {
            return Err(DecoderError::InvalidGeometry("cache binding"));
        }
        Ok(batch.enqueue(&self.runtime, &bindings)?)
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
                for (layer, (&(key_input, value_input), &(key_output, value_output))) in self
                    .decoder
                    .outputs
                    .cache_inputs
                    .iter()
                    .zip(&self.decoder.outputs.cache_updates)
                    .enumerate()
                {
                    for (component, (output, input)) in
                        [(key_output, key_input), (value_output, value_input)]
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
                let tensor_count = self.decoder.outputs.cache_updates.len() * 2;
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

fn decoder_artifact_identity(
    config: &DecoderConfig,
    plan: &ExecutorPlan,
    arenas: &[ExecutorArena],
    weights: DecoderWeightFeatures,
    compile: DecoderCompileConfig,
    compiler_facts_digest: &str,
) -> Result<String, DecoderError> {
    let identity = DecoderArtifactIdentity {
        manifest_fingerprint: &plan.manifest_fingerprint,
        compiler_facts_digest,
        page_tokens: plan.page_tokens,
        decoder: config,
        weights,
        arenas: arenas
            .iter()
            .map(|arena| DecoderArtifactArena {
                class_id: arena.class_id,
                backend_base_index: arena.backend_base_index,
                page_count: arena.page_count,
            })
            .collect(),
        compile,
    };
    let bytes = serde_json::to_vec(&identity)?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
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
        .cache_inputs
        .iter()
        .zip(&decoder.outputs.cache_updates)
        .zip(&bindings)
        .map(
            |((&(key_input, value_input), &(key_output, value_output)), binding)| {
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
                    runtime.alias_state_required(key_input, key_output, cache_bytes),
                    runtime.alias_state_required(value_input, value_output, cache_bytes),
                ])
            },
        )
        .collect::<Result<Vec<_>, DecoderError>>()
        .map(|caches| caches.into_iter().flatten().collect())
}

fn decoder_compile_options(
    decoder: &DecoderGraph,
    compile: DecoderCompileConfig,
) -> luminal::prelude::CompileOptions {
    decoder
        .class_dimensions
        .iter()
        .fold(
            luminal::prelude::CompileOptions::default()
                .dim_buckets(
                    's',
                    &[
                        luminal::prelude::DimBucket::new(1, 1),
                        luminal::prelude::DimBucket::new(2, compile.maximum_query_tokens)
                            .representative(compile.representative_prefill_tokens),
                    ],
                )
                .dim_buckets(
                    'b',
                    &[
                        luminal::prelude::DimBucket::new(1, compile.maximum_batch_size)
                            .representative(1),
                    ],
                ),
            |options, class| {
                options.dim_buckets(
                    class.context_pages,
                    &[
                        luminal::prelude::DimBucket::new(1, compile.maximum_context_pages)
                            .representative(compile.representative_context_pages),
                    ],
                )
            },
        )
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
    decoder
        .outputs
        .cache_inputs
        .iter()
        .zip(&decoder.outputs.cache_updates)
        .all(|(&(key_input, value_input), &(key_output, value_output))| {
            runtime.output_aliases_input_in_all_buckets(key_output, key_input)
                && runtime.output_aliases_input_in_all_buckets(value_output, value_input)
        })
}

fn seed_compile_inputs(
    runtime: &mut CudaRuntime,
    decoder: &DecoderGraph,
    compile: DecoderCompileConfig,
    page_tokens: usize,
) {
    let int_bytes = std::mem::size_of::<i32>();
    runtime.set_data_with_capacity(
        decoder.inputs.token_ids,
        vec![1_i32; compile.representative_prefill_tokens],
        compile.maximum_query_tokens * int_bytes,
    );
    runtime.set_data_with_capacity(
        decoder.inputs.positions,
        (0..i32::try_from(compile.representative_prefill_tokens).unwrap()).collect::<Vec<_>>(),
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
            (0..i32::try_from(compile.representative_prefill_tokens).unwrap())
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
            vec![
                0_i32,
                i32::try_from(compile.representative_prefill_tokens).unwrap(),
            ],
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
            vec![i32::try_from(page_tokens.min(compile.representative_prefill_tokens)).unwrap()],
            compile.maximum_batch_size * int_bytes,
        );
    }
}

fn validate_plan(
    config: &DecoderConfig,
    plan: &ExecutorPlan,
    arenas: &[ExecutorArena],
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
    if layers != (0..layer_count).collect() {
        return Err(DecoderError::UnsupportedPlan);
    }
    if let Some(layer_attention) = &config.layer_attention {
        for (layer, expected) in layer_attention.iter().enumerate() {
            let class = class_for_layer(
                plan,
                u32::try_from(layer).map_err(|_| DecoderError::InvalidGeometry("layer index"))?,
            )?;
            let matches = matches!(
                (expected, class.visibility),
                (DecoderAttentionKind::Full, crate::AttentionVisibility::Full)
                    | (
                        DecoderAttentionKind::Sliding,
                        crate::AttentionVisibility::Sliding { .. }
                    )
            );
            if !matches {
                return Err(DecoderError::UnsupportedPlan);
            }
        }
    }
    Ok((
        DecoderDimensions {
            query_tokens: Expression::from('s'),
            kv_width: config
                .kv_heads
                .checked_mul(config.head_dim)
                .ok_or(DecoderError::InvalidGeometry("KV width"))?,
        },
        class_dimensions.into_boxed_slice(),
    ))
}

fn class_for_layer(
    plan: &ExecutorPlan,
    layer: u32,
) -> Result<&crate::AttentionClass, DecoderError> {
    let mut classes = plan
        .classes
        .iter()
        .filter(|class| class.layers.contains(&layer));
    let class = classes.next().ok_or(DecoderError::UnsupportedPlan)?;
    if classes.next().is_some() {
        return Err(DecoderError::UnsupportedPlan);
    }
    Ok(class)
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
                Expression::from('b'),
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
use block::{DecoderLayer, DecoderNorm, LayerInputs};

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
