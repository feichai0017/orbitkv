//! Decoder compilation lifecycle: model/arena binding, search or strict replay,
//! and installation of the executable state. Workload fixtures live in tuning.

use std::{path::PathBuf, sync::Arc};

use orbitkv::StatePoolIdentity;
use orbitkv_compiler::{
    op::Runtime,
    prelude::{Graph, Symbol, bf16, rand::SeedableRng, tracing},
};
use orbitkv_cuda::{
    cudarc::driver::{CudaContext, CudaSlice, CudaStream},
    runtime::CudaRuntime,
};

use super::{
    CompiledDecoder, CompiledFixedState, DecoderArtifact, DecoderCompilation, DecoderCompileConfig,
    DecoderConfig, DecoderError, DecoderGraph, DecoderStorage, DecoderTuningProfile,
    artifact::{decoder_artifact_identity_with_tuning, new_artifact},
    cache_updates_in_place, capture_input_allocations, inspect_weight_features,
    representative::RepresentativeInputs,
    tuning::decoder_compile_options,
};
use crate::{ExecutorPlan, FixedStateDeviceArenas, FixedStateGraphResource};

#[allow(clippy::mutable_key_type)] // OrbitKV Symbols are stable interned dimension identities.
pub(super) fn prepare_decoder_compilation(
    config: &DecoderConfig,
    plan: &ExecutorPlan,
    storage: DecoderStorage<'_>,
    stream: &Arc<CudaStream>,
    weight_files: &[PathBuf],
    compile: DecoderCompileConfig,
    tuning: &DecoderTuningProfile,
) -> Result<DecoderCompilation, DecoderError> {
    let _stage = tracing::info_span!(target: "orbitkv::stage", "orbitkv.decoder.prepare").entered();
    compile.validate()?;
    config.require_executable()?;
    if weight_files.is_empty() {
        return Err(DecoderError::InvalidGeometry("weight files"));
    }
    let weights = tracing::info_span!(target: "orbitkv::stage", "orbitkv.weights.inspect")
        .in_scope(|| inspect_weight_features(weight_files, config))?;
    let fixed_state_registrations = plan.fixed_state_registrations(storage.fixed_state_pools)?;
    let compiler_facts = plan.compiler_facts(storage.token_arenas)?;
    let identity = decoder_artifact_identity_with_tuning(
        config,
        plan,
        storage.token_arenas,
        &fixed_state_registrations,
        weights,
        compile,
        compiler_facts.digest(),
        tuning,
    )?;
    let mut graph = Graph::default();
    let decoder =
        tracing::info_span!(target: "orbitkv::stage", "orbitkv.graph.build").in_scope(|| {
            DecoderGraph::build(
                &mut graph,
                config,
                weights,
                plan,
                storage.token_arenas,
                &fixed_state_registrations,
            )
        })?;
    let page_tokens = usize::try_from(plan.page_tokens)
        .map_err(|_| DecoderError::InvalidGeometry("page tokens"))?;
    if page_tokens > i32::MAX as usize {
        return Err(DecoderError::InvalidGeometry("backend index range"));
    }
    let mut options = decoder_compile_options(&decoder, compile, tuning, page_tokens)?;
    let mut facts = compiler_facts.egglog().to_owned();
    if tuning.enable_shared_fp8_quantization {
        facts.push('\n');
        facts.push_str(orbitkv_cuda::providers::deepgemm::SHARED_QUANTIZATION_COMPILER_FACT);
    }
    options = options.compiler_facts(facts);
    if options
        .bucket_representatives
        .as_ref()
        .unwrap()
        .iter()
        .any(|dims| {
            fixed_state_registrations.iter().any(|state| {
                dims[&orbitkv_compiler::prelude::Symbol::from('b')] > state.slot_count as usize
            })
        })
    {
        return Err(DecoderError::InvalidGeometry(
            "tuning fixed-state slot capacity",
        ));
    }
    let representative = options.bucket_representatives.as_ref().unwrap()[0].clone();
    for (&dim, &value) in &representative {
        graph.set_dim(dim, value);
    }
    let mut runtime = initialize_weight_runtime(&graph, stream, weight_files)?;
    let _bindings_stage =
        tracing::info_span!(target: "orbitkv::stage", "orbitkv.inputs.bind_and_seed").entered();
    let persistent_cache = register_persistent_cache(&mut runtime, &decoder, plan, config)?;
    let mut fixed_state_scratch = Vec::with_capacity(decoder.outputs.fixed_states.len());
    let representative_batch = representative[&Symbol::from('b')];
    for state in &decoder.outputs.fixed_states {
        state.binding.seed_destination_slots(
            &mut runtime,
            representative_batch,
            compile.maximum_batch_size,
        )?;
        fixed_state_scratch.push(
            state
                .binding
                .allocate_compile_scratch(&mut runtime, state.policy),
        );
    }
    RepresentativeInputs::new(&decoder, compile, page_tokens)
        .install(&mut runtime, &representative);
    Ok(DecoderCompilation {
        graph,
        decoder,
        runtime,
        persistent_cache,
        fixed_state_scratch,
        options,
        identity,
        page_tokens,
    })
}

fn initialize_weight_runtime(
    graph: &Graph,
    stream: &Arc<CudaStream>,
    weight_files: &[PathBuf],
) -> Result<CudaRuntime, DecoderError> {
    let mut runtime = tracing::info_span!(target: "orbitkv::stage", "cuda.initialize")
        .in_scope(|| CudaRuntime::initialize(Arc::clone(stream)));
    let _weights_stage = tracing::info_span!(target: "orbitkv::stage", "orbitkv.weights.load", files = weight_files.len()).entered();
    for weights_path in weight_files {
        runtime
            .load_safetensors(graph, weights_path)
            .map_err(|error| DecoderError::WeightLoading(format!("{error:#}")))?;
    }
    Ok(runtime)
}

pub(super) fn bind_fixed_state(
    plan: &ExecutorPlan,
    identities: &[(u16, StatePoolIdentity)],
    stream: &Arc<CudaStream>,
    resources: &[FixedStateGraphResource],
    runtime: &mut CudaRuntime,
    scratch: Vec<CudaSlice<u8>>,
) -> Result<Option<CompiledFixedState>, DecoderError> {
    if resources.is_empty() {
        return Ok(None);
    }
    let arenas = FixedStateDeviceArenas::allocate(plan, identities, Arc::clone(stream))?;
    let runtime_bindings = resources
        .iter()
        .map(|state| arenas.bind_graph_state(runtime, state.binding, state.policy))
        .collect::<Result<Vec<_>, _>>()?
        .into_boxed_slice();
    drop(scratch);
    Ok(Some(CompiledFixedState {
        arenas,
        graph_bindings: resources
            .iter()
            .map(|state| state.binding)
            .collect::<Vec<_>>()
            .into_boxed_slice(),
        runtime_bindings,
    }))
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
                .and_then(|elements| elements.checked_mul(std::mem::size_of::<bf16>()))
                .ok_or(DecoderError::InvalidGeometry("cache bytes"))?;
            Ok([
                runtime.alias_state_required(state.binding.key, state.key_update, cache_bytes),
                runtime.alias_state_required(state.binding.value, state.value_update, cache_bytes),
            ])
        })
        .collect::<Result<Vec<_>, DecoderError>>()
        .map(|caches| caches.into_iter().flatten().collect())
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
    /// Rejects invalid bucket geometry or incompatible model plans. `OrbitKV`
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
        Self::compile_or_load_with_tuning(
            config,
            plan,
            storage,
            stream,
            weight_files,
            compile,
            &DecoderTuningProfile::default(),
            artifact,
        )
    }

    /// Compiles or strictly loads a schedule for a workload tuning profile.
    ///
    /// # Errors
    /// Rejects invalid representatives, incompatible artifacts, or device failures.
    #[allow(clippy::too_many_arguments)] // Keeps existing capacity config literals compatible.
    pub fn compile_or_load_with_tuning(
        config: &DecoderConfig,
        plan: &ExecutorPlan,
        storage: DecoderStorage<'_>,
        stream: &std::sync::Arc<CudaStream>,
        weight_files: &[std::path::PathBuf],
        compile: DecoderCompileConfig,
        tuning: &DecoderTuningProfile,
        artifact: Option<&DecoderArtifact>,
    ) -> Result<(Self, DecoderArtifact), DecoderError> {
        let _stage = tracing::info_span!(target: "orbitkv::stage", "orbitkv.decoder.compile_or_load", replay = artifact.is_some()).entered();
        if let Some(artifact) = artifact {
            artifact.validate()?;
            artifact
                .cuda_modules
                .validate_for_device(stream.context())
                .map_err(DecoderError::Artifact)?;
        }
        let prepared = prepare_decoder_compilation(
            config,
            plan,
            storage,
            stream,
            weight_files,
            compile,
            tuning,
        )?;
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
            let _stage =
                tracing::info_span!(target: "orbitkv::stage", "orbitkv.schedule.replay").entered();
            graph.prepare_selected_schedule(&options);
            graph.install_selected_schedule(artifact.schedule.clone());
            runtime
                .load_selected_schedule_with_modules(&graph, &artifact.cuda_modules)
                .map_err(DecoderError::Artifact)?;
            artifact.clone()
        } else {
            let mut rng =
                orbitkv_compiler::prelude::rand::rngs::SmallRng::seed_from_u64(compile.search_seed);
            runtime = graph.compile_with_rng(runtime, options, &mut rng);
            let modules = runtime
                .capture_module_artifact(&graph)
                .map_err(DecoderError::Artifact)?;
            new_artifact(
                identity,
                graph
                    .selected_schedule()
                    .cloned()
                    .ok_or_else(|| DecoderError::Artifact("selected schedule missing".into()))?,
                modules,
            )
        };
        let _finalize_stage =
            tracing::info_span!(target: "orbitkv::stage", "orbitkv.decoder.finalize").entered();
        let fixed_state = bind_fixed_state(
            plan,
            storage.fixed_state_pools,
            stream,
            &decoder.outputs.fixed_states,
            &mut runtime,
            fixed_state_scratch,
        )?;
        // Conservative deployment policy, independent of model or bucket
        // count. Raising this requires qualifying retained provider resources
        // and their aggregate memory alongside KV and workspace budgets.
        // A switch rebuilds the target driver graph without repeating search.
        runtime.set_max_materialized_buckets(Some(super::DEFAULT_GRAPH_CACHE_CAPACITY.get()));
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

    /// Builds and searches one decoder on a CUDA device selected by ordinal.
    ///
    /// This is the high-level composition boundary. Callers that do not need
    /// to coordinate another CUDA subsystem should use it instead of depending
    /// directly on `OrbitKV`'s stream type.
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
        Self::compile_or_load_on_device_with_tuning(
            config,
            plan,
            storage,
            device_index,
            weight_files,
            compile,
            &DecoderTuningProfile::default(),
            artifact,
        )
    }

    /// Compiles or loads with an explicit, artifact-bound workload tuning profile.
    ///
    /// # Errors
    /// Returns model, device, tuning, or strict artifact compatibility failures.
    #[allow(clippy::too_many_arguments)] // Keeps existing capacity config literals compatible.
    pub fn compile_or_load_on_device_with_tuning(
        config: &DecoderConfig,
        plan: &ExecutorPlan,
        storage: DecoderStorage<'_>,
        device_index: usize,
        weight_files: &[std::path::PathBuf],
        compile: DecoderCompileConfig,
        tuning: &DecoderTuningProfile,
        artifact: Option<&DecoderArtifact>,
    ) -> Result<(Self, DecoderArtifact), DecoderError> {
        let context = CudaContext::new(device_index)?;
        let stream = context.new_stream()?;
        Self::compile_or_load_with_tuning(
            config,
            plan,
            storage,
            &stream,
            weight_files,
            compile,
            tuning,
            artifact,
        )
    }
}
