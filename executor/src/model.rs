//! Configuration-driven decoder graph using OrbitKV-managed paged attention.

use std::collections::BTreeSet;

use luminal::prelude::rand::SeedableRng;
use luminal::{
    dtype::DType,
    op::Runtime,
    prelude::{Expression, F32Pow, Graph, GraphTensor},
    shape::ToShape,
};
use luminal_nn::{LayerNorm, scatter_rows};
use serde::Deserialize;
use thiserror::Error;

use luminal_cuda_lite::{
    cudarc::driver::{CudaSlice, CudaStream},
    runtime::CudaRuntime,
};

use crate::{
    ExecutorPlan, RelocationBatch,
    cuda::{
        AttentionKernel, CudaRelocationError, KvCacheBinding, PagedAttentionInputs,
        PagedAttentionMetadata, PendingRelocationCopy, paged_attention,
    },
};

#[derive(Clone, Debug, PartialEq)]
pub struct DecoderConfig {
    pub layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub vocabulary_size: usize,
    pub rope_theta: f32,
    pub rms_epsilon: f32,
    pub tied_embeddings: bool,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DecoderWeightLayout {
    pub qkv_bias: bool,
    pub qk_norm: bool,
}

#[derive(Debug, Error)]
pub enum DecoderError {
    #[error("invalid decoder config JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("decoder geometry is invalid: {0}")]
    InvalidGeometry(&'static str),
    #[error("decoder currently requires one token-KV attention class covering every layer")]
    UnsupportedPlan,
    #[error(transparent)]
    Executor(#[from] crate::ExecutorError),
    #[error(transparent)]
    Device(#[from] luminal_cuda_lite::cudarc::driver::DriverError),
    #[error(transparent)]
    Relocation(#[from] CudaRelocationError),
    #[error("decoder runtime input geometry exceeds its compiled capacity")]
    InputCapacity,
}

/// Dynamic-shape and search policy for one compiled decoder executable.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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
    pub write_slots: &'a [u64],
    pub attention: &'a crate::AttentionBatch,
}

type InputAllocation = (GraphTensor, u64, usize);

/// One compiled graph and one persistent K/V arena shared by prefill and decode.
pub struct CompiledDecoder {
    graph: Graph,
    decoder: DecoderGraph,
    runtime: CudaRuntime,
    persistent_cache: Vec<CudaSlice<u8>>,
    compile: DecoderCompileConfig,
    vocabulary_size: usize,
    class_id: u16,
    page_tokens: usize,
    physical_pages: usize,
    cache_slots: usize,
    cache_updates_in_place: bool,
    dynamic_input_allocations: Box<[InputAllocation]>,
}

#[derive(Deserialize)]
struct DecoderConfigInput {
    num_hidden_layers: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    #[serde(default)]
    head_dim: Option<usize>,
    vocab_size: usize,
    rope_theta: f32,
    rms_norm_eps: f32,
    tie_word_embeddings: bool,
}

impl DecoderConfig {
    /// Parses the common decoder geometry required by the graph builder.
    ///
    /// # Errors
    ///
    /// Rejects missing JSON fields, zero dimensions, inconsistent head
    /// geometry, unsupported head dimensions, or invalid floating constants.
    pub fn from_json(bytes: &[u8]) -> Result<Self, DecoderError> {
        let input = serde_json::from_slice::<DecoderConfigInput>(bytes)?;
        let head_dim = input.head_dim.unwrap_or_else(|| {
            input
                .hidden_size
                .checked_div(input.num_attention_heads.max(1))
                .unwrap_or_default()
        });
        if input.layers_or_width_is_zero()
            || head_dim == 0
            || input.num_attention_heads.checked_mul(head_dim) != Some(input.hidden_size)
            || !input
                .num_attention_heads
                .is_multiple_of(input.num_key_value_heads)
            || !matches!(head_dim, 64 | 128 | 256)
        {
            return Err(DecoderError::InvalidGeometry("dimensions"));
        }
        if !input.rope_theta.is_finite()
            || input.rope_theta <= 0.0
            || !input.rms_norm_eps.is_finite()
            || input.rms_norm_eps <= 0.0
        {
            return Err(DecoderError::InvalidGeometry("floating constants"));
        }
        Ok(Self {
            layers: input.num_hidden_layers,
            hidden_size: input.hidden_size,
            intermediate_size: input.intermediate_size,
            query_heads: input.num_attention_heads,
            kv_heads: input.num_key_value_heads,
            head_dim,
            vocabulary_size: input.vocab_size,
            rope_theta: input.rope_theta,
            rms_epsilon: input.rms_norm_eps,
            tied_embeddings: input.tie_word_embeddings,
        })
    }
}

impl DecoderConfigInput {
    fn layers_or_width_is_zero(&self) -> bool {
        self.num_hidden_layers == 0
            || self.hidden_size == 0
            || self.intermediate_size == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || self.vocab_size == 0
    }
}

#[derive(Clone, Copy)]
pub struct DecoderInputs {
    pub token_ids: GraphTensor,
    pub positions: GraphTensor,
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
}

#[derive(Clone, Copy)]
struct DecoderDimensions {
    query_tokens: Expression,
    context_pages: Expression,
    cache_slots: usize,
    kv_width: usize,
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

    /// Builds a decoder-only transformer whose KV state is addressed solely by
    /// one `ExecutorPlan` attention class.
    ///
    /// # Errors
    ///
    /// Rejects multi-class or incomplete layer coverage in this first native
    /// model path, and propagates paged-attention geometry failures.
    pub fn build(
        graph: &mut Graph,
        config: &DecoderConfig,
        weights: DecoderWeightLayout,
        plan: &ExecutorPlan,
        physical_pages: usize,
    ) -> Result<Self, DecoderError> {
        let dimensions = validate_plan(config, plan, physical_pages)?;
        let inputs = decoder_inputs(graph, plan.classes[0].class_id, dimensions);
        let embedding = weight(
            graph,
            "model.embed_tokens.weight",
            (config.vocabulary_size, config.hidden_size),
            DType::Bf16,
        );
        let mut hidden = token_embedding(&embedding, &inputs.token_ids, config.hidden_size);
        let mut cache_inputs = Vec::with_capacity(config.layers);
        let mut cache_updates = Vec::with_capacity(config.layers);
        for layer in 0..config.layers {
            let k_cache = graph
                .named_tensor(
                    format!("kv.{layer}.key"),
                    (dimensions.cache_slots, dimensions.kv_width),
                )
                .persist()
                .as_dtype(DType::Bf16);
            let v_cache = graph
                .named_tensor(
                    format!("kv.{layer}.value"),
                    (dimensions.cache_slots, dimensions.kv_width),
                )
                .persist()
                .as_dtype(DType::Bf16);
            let block = DecoderLayer::new(graph, config, weights, layer);
            let (next, key_update, value_update) = block.forward(
                &LayerInputs {
                    hidden: &hidden,
                    positions: &inputs.positions,
                    write_slots: &inputs.write_slots,
                    metadata: &inputs.attention,
                    k_cache: &k_cache,
                    v_cache: &v_cache,
                },
                &plan.classes[0],
                config,
                dimensions,
            )?;
            hidden = next;
            cache_inputs.push((k_cache, v_cache));
            cache_updates.push((key_update.output(), value_update.output()));
        }
        let norm = LayerNorm::new(
            config.hidden_size,
            Some("model.norm.weight"),
            None,
            false,
            config.rms_epsilon,
            graph,
        );
        let normalized = norm.forward(hidden.cast(DType::F32)).cast(DType::Bf16);
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
        })
    }
}

impl CompiledDecoder {
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
        weights: DecoderWeightLayout,
        plan: &ExecutorPlan,
        physical_pages: usize,
        stream: &std::sync::Arc<CudaStream>,
        weight_files: &[std::path::PathBuf],
        compile: DecoderCompileConfig,
    ) -> Result<Self, DecoderError> {
        compile.validate()?;
        if weight_files.is_empty() {
            return Err(DecoderError::InvalidGeometry("weight files"));
        }
        let mut graph = Graph::default();
        let decoder = DecoderGraph::build(&mut graph, config, weights, plan, physical_pages)?;
        let cache_bytes = physical_pages
            .checked_mul(
                usize::try_from(plan.page_tokens)
                    .map_err(|_| DecoderError::InvalidGeometry("page tokens"))?,
            )
            .and_then(|tokens| tokens.checked_mul(config.kv_heads))
            .and_then(|elements| elements.checked_mul(config.head_dim))
            .and_then(|elements| elements.checked_mul(2))
            .ok_or(DecoderError::InvalidGeometry("cache bytes"))?;
        let page_tokens = usize::try_from(plan.page_tokens)
            .map_err(|_| DecoderError::InvalidGeometry("page tokens"))?;
        let cache_slots = physical_pages
            .checked_mul(page_tokens)
            .ok_or(DecoderError::InvalidGeometry("cache slots"))?;
        if page_tokens > i32::MAX as usize
            || physical_pages > i32::MAX as usize
            || cache_slots > i32::MAX as usize
        {
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
        let mut persistent_cache = decoder
            .outputs
            .cache_inputs
            .iter()
            .zip(&decoder.outputs.cache_updates)
            .flat_map(|(&(key_input, value_input), &(key_output, value_output))| {
                [
                    runtime.alias_state(key_input, key_output, cache_bytes),
                    runtime.alias_state(value_input, value_output, cache_bytes),
                ]
            })
            .collect::<Vec<_>>();

        graph.set_dim('s', compile.representative_prefill_tokens);
        graph.set_dim('b', 1);
        graph.set_dim('c', compile.representative_context_pages);
        seed_compile_inputs(&mut runtime, &decoder, compile, page_tokens);
        let options = luminal::prelude::CompileOptions::default()
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
            )
            .dim_buckets(
                'c',
                &[
                    luminal::prelude::DimBucket::new(1, compile.maximum_context_pages)
                        .representative(compile.representative_context_pages),
                ],
            )
            .search_graph_limit(compile.search_graphs);
        let mut rng = luminal::prelude::rand::rngs::SmallRng::seed_from_u64(compile.search_seed);
        let runtime = graph.compile_with_rng(runtime, options, &mut rng);
        runtime.release_pooled_memory();
        let dynamic_input_allocations = capture_input_allocations(&runtime, &decoder)?;
        let cache_updates_in_place = cache_updates_in_place(&runtime, &decoder);
        for cache in &mut persistent_cache {
            stream.memset_zeros(cache)?;
        }
        Ok(Self {
            graph,
            decoder,
            runtime,
            persistent_cache,
            compile,
            vocabulary_size: config.vocabulary_size,
            class_id: plan.classes[0].class_id,
            page_tokens,
            physical_pages,
            cache_slots,
            cache_updates_in_place,
            dynamic_input_allocations,
        })
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
            self.class_id,
            self.page_tokens,
            self.physical_pages,
            self.cache_slots,
            self.vocabulary_size,
        )?;
        self.graph.set_dim('s', step.tokens.len());
        self.graph.set_dim(
            'b',
            step.attention
                .query_indptr
                .len()
                .checked_sub(1)
                .ok_or(DecoderError::InputCapacity)?,
        );
        self.graph.set_dim('c', step.attention.page_indices.len());
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
        self.runtime.set_data(
            self.decoder.inputs.write_slots,
            step.write_slots
                .iter()
                .map(|&slot| i32::try_from(slot).map_err(|_| DecoderError::InputCapacity))
                .collect::<Result<Vec<_>, _>>()?,
        );
        self.decoder
            .inputs
            .attention
            .upload(&mut self.runtime, step.attention)?;
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
        self.runtime.execute(&self.graph.dyn_map);
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
}

fn dynamic_inputs(decoder: &DecoderGraph) -> [GraphTensor; 7] {
    [
        decoder.inputs.token_ids,
        decoder.inputs.positions,
        decoder.inputs.write_slots,
        decoder.inputs.attention.page_indices,
        decoder.inputs.attention.query_indptr,
        decoder.inputs.attention.page_indptr,
        decoder.inputs.attention.last_page_len,
    ]
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
    runtime.set_data_with_capacity(
        decoder.inputs.write_slots,
        (0..i32::try_from(compile.representative_prefill_tokens).unwrap()).collect::<Vec<_>>(),
        compile.maximum_query_tokens * int_bytes,
    );
    runtime.set_data_with_capacity(
        decoder.inputs.attention.page_indices,
        vec![0_i32; compile.representative_context_pages],
        compile.maximum_context_pages * int_bytes,
    );
    runtime.set_data_with_capacity(
        decoder.inputs.attention.query_indptr,
        vec![
            0_i32,
            i32::try_from(compile.representative_prefill_tokens).unwrap(),
        ],
        (compile.maximum_batch_size + 1) * int_bytes,
    );
    runtime.set_data_with_capacity(
        decoder.inputs.attention.page_indptr,
        vec![
            0_i32,
            i32::try_from(compile.representative_context_pages).unwrap(),
        ],
        (compile.maximum_batch_size + 1) * int_bytes,
    );
    runtime.set_data_with_capacity(
        decoder.inputs.attention.last_page_len,
        vec![i32::try_from(page_tokens.min(compile.representative_prefill_tokens)).unwrap()],
        compile.maximum_batch_size * int_bytes,
    );
}

fn validate_step(
    step: DecoderStep<'_>,
    compile: DecoderCompileConfig,
    class_id: u16,
    page_tokens: usize,
    physical_pages: usize,
    cache_slots: usize,
    vocabulary_size: usize,
) -> Result<(), DecoderError> {
    let batch = step
        .attention
        .query_indptr
        .len()
        .checked_sub(1)
        .ok_or(DecoderError::InputCapacity)?;
    if step.tokens.is_empty()
        || step.attention.class_id != class_id
        || step.tokens.len() != step.positions.len()
        || step.tokens.len() != step.write_slots.len()
        || step.tokens.len() > compile.maximum_query_tokens
        || step
            .tokens
            .iter()
            .any(|&token| usize::try_from(token).map_or(true, |token| token >= vocabulary_size))
        || batch == 0
        || batch > compile.maximum_batch_size
        || step.attention.page_indices.is_empty()
        || step.attention.page_indices.len() > compile.maximum_context_pages
        || step.attention.page_indices.iter().any(|page| *page < 0)
        || step
            .attention
            .page_indices
            .iter()
            .any(|&page| usize::try_from(page).map_or(true, |page| page >= physical_pages))
        || step
            .write_slots
            .iter()
            .any(|&slot| usize::try_from(slot).map_or(true, |slot| slot >= cache_slots))
        || step.attention.query_indptr.first() != Some(&0)
        || step.attention.query_indptr.last().copied() != i32::try_from(step.tokens.len()).ok()
        || step.attention.page_indptr.first() != Some(&0)
        || step.attention.page_indptr.last().copied()
            != i32::try_from(step.attention.page_indices.len()).ok()
        || step.attention.page_indptr.len() != batch + 1
        || step.attention.last_page_len.len() != batch
        || step.attention.last_page_len.iter().any(|&tokens| {
            tokens <= 0 || usize::try_from(tokens).map_or(true, |tokens| tokens > page_tokens)
        })
        || step
            .attention
            .query_indptr
            .windows(2)
            .any(|row| row[0] >= row[1])
        || step
            .attention
            .page_indptr
            .windows(2)
            .any(|row| row[0] >= row[1])
    {
        return Err(DecoderError::InputCapacity);
    }
    Ok(())
}

fn validate_plan(
    config: &DecoderConfig,
    plan: &ExecutorPlan,
    physical_pages: usize,
) -> Result<DecoderDimensions, DecoderError> {
    let layer_count =
        u32::try_from(config.layers).map_err(|_| DecoderError::InvalidGeometry("layer count"))?;
    if plan.classes.len() != 1
        || physical_pages == 0
        || plan.classes[0].key_bytes_per_token_per_layer
            != u64::try_from(config.kv_heads * config.head_dim * 2)
                .map_err(|_| DecoderError::InvalidGeometry("key bytes per token"))?
        || plan.classes[0].value_bytes_per_token_per_layer
            != u64::try_from(config.kv_heads * config.head_dim * 2)
                .map_err(|_| DecoderError::InvalidGeometry("value bytes per token"))?
        || plan.classes[0]
            .layers
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            != (0..layer_count).collect()
    {
        return Err(DecoderError::UnsupportedPlan);
    }
    Ok(DecoderDimensions {
        query_tokens: Expression::from('s'),
        context_pages: Expression::from('c'),
        cache_slots: physical_pages
            .checked_mul(plan.classes[0].page_tokens as usize)
            .ok_or(DecoderError::InvalidGeometry("cache slots"))?,
        kv_width: config
            .kv_heads
            .checked_mul(config.head_dim)
            .ok_or(DecoderError::InvalidGeometry("KV width"))?,
    })
}

fn decoder_inputs(
    graph: &mut Graph,
    class_id: u16,
    dimensions: DecoderDimensions,
) -> DecoderInputs {
    let token_ids = graph
        .named_tensor("tokens", dimensions.query_tokens)
        .as_dtype(DType::Int);
    let positions = graph
        .named_tensor("positions", dimensions.query_tokens)
        .as_dtype(DType::Int);
    let write_slots = graph
        .named_tensor("kv.write_slots", dimensions.query_tokens)
        .as_dtype(DType::Int);
    let attention = PagedAttentionMetadata::new(
        graph,
        class_id,
        Expression::from('b'),
        dimensions.context_pages,
    );
    DecoderInputs {
        token_ids,
        positions,
        write_slots,
        attention,
    }
}

struct DecoderLayer {
    attention_norm: LayerNorm,
    feed_forward_norm: LayerNorm,
    q_weight: GraphTensor,
    k_weight: GraphTensor,
    v_weight: GraphTensor,
    o_weight: GraphTensor,
    q_bias: Option<GraphTensor>,
    k_bias: Option<GraphTensor>,
    v_bias: Option<GraphTensor>,
    q_norm: Option<GraphTensor>,
    k_norm: Option<GraphTensor>,
    gate: GraphTensor,
    up: GraphTensor,
    down: GraphTensor,
}

struct LayerInputs<'a> {
    hidden: &'a GraphTensor,
    positions: &'a GraphTensor,
    write_slots: &'a GraphTensor,
    metadata: &'a PagedAttentionMetadata,
    k_cache: &'a GraphTensor,
    v_cache: &'a GraphTensor,
}

impl DecoderLayer {
    fn new(
        graph: &mut Graph,
        config: &DecoderConfig,
        layout: DecoderWeightLayout,
        layer: usize,
    ) -> Self {
        let prefix = format!("model.layers.{layer}");
        let q_width = config.query_heads * config.head_dim;
        let kv_width = config.kv_heads * config.head_dim;
        let projection = |graph: &mut Graph, name: &str, width| {
            weight(
                graph,
                format!("{prefix}.self_attn.{name}.weight"),
                (width, config.hidden_size),
                DType::Bf16,
            )
        };
        Self {
            attention_norm: LayerNorm::new(
                config.hidden_size,
                Some(&format!("{prefix}.input_layernorm.weight")),
                None,
                false,
                config.rms_epsilon,
                graph,
            ),
            feed_forward_norm: LayerNorm::new(
                config.hidden_size,
                Some(&format!("{prefix}.post_attention_layernorm.weight")),
                None,
                false,
                config.rms_epsilon,
                graph,
            ),
            q_weight: projection(graph, "q_proj", q_width),
            k_weight: projection(graph, "k_proj", kv_width),
            v_weight: projection(graph, "v_proj", kv_width),
            o_weight: weight(
                graph,
                format!("{prefix}.self_attn.o_proj.weight"),
                (config.hidden_size, q_width),
                DType::Bf16,
            ),
            q_bias: layout.qkv_bias.then(|| {
                weight(
                    graph,
                    format!("{prefix}.self_attn.q_proj.bias"),
                    q_width,
                    DType::Bf16,
                )
            }),
            k_bias: layout.qkv_bias.then(|| {
                weight(
                    graph,
                    format!("{prefix}.self_attn.k_proj.bias"),
                    kv_width,
                    DType::Bf16,
                )
            }),
            v_bias: layout.qkv_bias.then(|| {
                weight(
                    graph,
                    format!("{prefix}.self_attn.v_proj.bias"),
                    kv_width,
                    DType::Bf16,
                )
            }),
            q_norm: layout.qk_norm.then(|| {
                weight(
                    graph,
                    format!("{prefix}.self_attn.q_norm.weight"),
                    config.head_dim,
                    DType::F32,
                )
            }),
            k_norm: layout.qk_norm.then(|| {
                weight(
                    graph,
                    format!("{prefix}.self_attn.k_norm.weight"),
                    config.head_dim,
                    DType::F32,
                )
            }),
            gate: weight(
                graph,
                format!("{prefix}.mlp.gate_proj.weight"),
                (config.intermediate_size, config.hidden_size),
                DType::Bf16,
            ),
            up: weight(
                graph,
                format!("{prefix}.mlp.up_proj.weight"),
                (config.intermediate_size, config.hidden_size),
                DType::Bf16,
            ),
            down: weight(
                graph,
                format!("{prefix}.mlp.down_proj.weight"),
                (config.hidden_size, config.intermediate_size),
                DType::Bf16,
            ),
        }
    }

    fn forward(
        &self,
        inputs: &LayerInputs<'_>,
        class: &crate::AttentionClass,
        config: &DecoderConfig,
        dimensions: DecoderDimensions,
    ) -> Result<(GraphTensor, GraphTensor, GraphTensor), DecoderError> {
        let normalized = self
            .attention_norm
            .forward((*inputs.hidden).cast(DType::F32))
            .cast(DType::Bf16);
        let project = |weight: GraphTensor, bias: Option<GraphTensor>| {
            let output = normalized.matmul(weight.t());
            bias.map_or(output, |bias| bias.expand_lhs(&output.dims()[..1]) + output)
        };
        let mut q = project(self.q_weight, self.q_bias)
            .split_dims(1, config.head_dim)
            .transpose(0, 1);
        let mut k = project(self.k_weight, self.k_bias)
            .split_dims(1, config.head_dim)
            .transpose(0, 1);
        if let Some(norm) = self.q_norm {
            q = qk_norm(&q, &norm);
        }
        if let Some(norm) = self.k_norm {
            k = qk_norm(&k, &norm);
        }
        q = rotary(&q, inputs.positions, config.rope_theta);
        k = rotary(&k, inputs.positions, config.rope_theta);
        let value = project(self.v_weight, self.v_bias);
        let key_rows = k.transpose(0, 1).merge_dims(1, 2);
        let key_update = scatter_rows(
            key_rows,
            *inputs.write_slots,
            *inputs.k_cache,
            config.kv_heads * config.head_dim,
        );
        let value_update = scatter_rows(
            value,
            *inputs.write_slots,
            *inputs.v_cache,
            config.kv_heads * config.head_dim,
        );
        let attention = paged_attention(
            PagedAttentionInputs {
                q,
                k_cache: key_update,
                v_cache: value_update,
                query_tokens: dimensions.query_tokens,
                context_pages: dimensions.context_pages,
            },
            *inputs.metadata,
            class,
            AttentionKernel {
                query_heads: config.query_heads,
                kv_heads: config.kv_heads,
                head_dim: config.head_dim,
                dtype: DType::Bf16,
                softmax_scale: 0.0,
            },
        )?;
        let attention = attention
            .transpose(0, 1)
            .merge_dims(1, 2)
            .matmul(self.o_weight.t());
        let hidden = *inputs.hidden + attention;
        let normalized = self
            .feed_forward_norm
            .forward(hidden.cast(DType::F32))
            .cast(DType::Bf16);
        let gate = normalized.matmul(self.gate.t()).cast(DType::F32);
        let up = normalized.matmul(self.up.t()).cast(DType::F32);
        let feed_forward = (gate.swish() * up).cast(DType::Bf16).matmul(self.down.t());
        Ok((hidden + feed_forward, key_update, value_update))
    }
}

fn qk_norm(input: &GraphTensor, weight: &GraphTensor) -> GraphTensor {
    let dtype = input.dtype;
    let normalized =
        (*input).cast(DType::F32).std_norm(2, 1e-6) * (*weight).expand_lhs(&input.dims()[..2]);
    normalized.cast(dtype)
}

fn rotary(input: &GraphTensor, positions: &GraphTensor, theta: f32) -> GraphTensor {
    let head_dim = input.dims()[2];
    let frequencies = input
        .graph()
        .arange_options(0, head_dim, 2)
        .cast(DType::F32)
        / head_dim;
    let inverse = theta.pow(frequencies).reciprocal();
    let angles = (*positions)
        .cast(DType::F32)
        .expand_dim(1, 1)
        .matmul(inverse.expand_dim(0, 1));
    let first = input.slice((.., .., ..head_dim / 2));
    let second = input.slice((.., .., head_dim / 2..));
    let cosine = angles
        .cos()
        .cast(input.dtype)
        .expand_dim(0, input.dims()[0]);
    let sine = angles
        .sin()
        .cast(input.dtype)
        .expand_dim(0, input.dims()[0]);
    (first * cosine - second * sine).concat_along(first * sine + second * cosine, 2)
}

fn token_embedding(table: &GraphTensor, tokens: &GraphTensor, hidden: usize) -> GraphTensor {
    let count = tokens.dims1();
    table.gather(
        (*tokens * hidden).expand_dim(1, hidden)
            + tokens.graph().arange(hidden).expand_dim(0, count),
    )
}

fn weight(
    graph: &mut Graph,
    name: impl ToString,
    shape: impl ToShape,
    dtype: DType,
) -> GraphTensor {
    graph.named_tensor(name, shape).persist().as_dtype(dtype)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compile_config_requires_distinct_decode_and_prefill_ranges() {
        let valid = DecoderCompileConfig {
            maximum_query_tokens: 32,
            representative_prefill_tokens: 8,
            maximum_batch_size: 4,
            maximum_context_pages: 64,
            representative_context_pages: 8,
            search_graphs: 2,
            search_seed: 1,
        };
        assert!(valid.validate().is_ok());
        assert!(
            DecoderCompileConfig {
                maximum_query_tokens: 1,
                ..valid
            }
            .validate()
            .is_err()
        );
        assert!(
            DecoderCompileConfig {
                search_graphs: 1,
                ..valid
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn step_validation_enforces_compiled_capacities() {
        let compile = DecoderCompileConfig {
            maximum_query_tokens: 4,
            representative_prefill_tokens: 2,
            maximum_batch_size: 1,
            maximum_context_pages: 2,
            representative_context_pages: 1,
            search_graphs: 2,
            search_seed: 1,
        };
        let attention = crate::AttentionBatch {
            class_id: 0,
            query_indptr: vec![0, 2].into_boxed_slice(),
            page_indptr: vec![0, 1].into_boxed_slice(),
            page_indices: vec![0].into_boxed_slice(),
            last_page_len: vec![2].into_boxed_slice(),
        };
        let valid = DecoderStep {
            tokens: &[1, 2],
            positions: &[0, 1],
            write_slots: &[0, 1],
            attention: &attention,
        };
        assert!(validate_step(valid, compile, 0, 16, 4, 64, 32).is_ok());
        assert!(
            validate_step(
                DecoderStep {
                    tokens: &[1, 2],
                    positions: &[0],
                    ..valid
                },
                compile,
                0,
                16,
                4,
                64,
                32,
            )
            .is_err()
        );
    }
}
