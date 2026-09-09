use std::{path::PathBuf, sync::Arc};

use luminal::{op::Runtime, prelude::Graph};
use luminal_cuda_lite::{
    cudarc::driver::{CudaSlice, CudaStream},
    runtime::CudaRuntime,
};
use orbitkv::StatePoolIdentity;

use super::{
    CompiledFixedState, DecoderCompilation, DecoderCompileConfig, DecoderConfig, DecoderError,
    DecoderGraph, DecoderStorage, decoder_artifact_identity, decoder_compile_options,
    inspect_weight_features, register_persistent_cache, seed_compile_inputs,
};
use crate::{ExecutorPlan, FixedStateDeviceArenas, FixedStateGraphResource};

pub(super) fn prepare_decoder_compilation(
    config: &DecoderConfig,
    plan: &ExecutorPlan,
    storage: DecoderStorage<'_>,
    stream: &Arc<CudaStream>,
    weight_files: &[PathBuf],
    compile: DecoderCompileConfig,
) -> Result<DecoderCompilation, DecoderError> {
    compile.validate()?;
    config.require_executable()?;
    if weight_files.is_empty() {
        return Err(DecoderError::InvalidGeometry("weight files"));
    }
    let weights = inspect_weight_features(weight_files, config)?;
    let fixed_state_registrations = plan.fixed_state_registrations(storage.fixed_state_pools)?;
    let compiler_facts = plan.luminal_compiler_facts(storage.token_arenas)?;
    let identity = decoder_artifact_identity(
        config,
        plan,
        storage.token_arenas,
        &fixed_state_registrations,
        weights,
        compile,
        compiler_facts.digest(),
    )?;
    let mut graph = Graph::default();
    let decoder = DecoderGraph::build(
        &mut graph,
        config,
        weights,
        plan,
        storage.token_arenas,
        &fixed_state_registrations,
    )?;
    let page_tokens = usize::try_from(plan.page_tokens)
        .map_err(|_| DecoderError::InvalidGeometry("page tokens"))?;
    if page_tokens > i32::MAX as usize {
        return Err(DecoderError::InvalidGeometry("backend index range"));
    }
    let mut runtime = CudaRuntime::initialize(Arc::clone(stream));
    for weights_path in weight_files {
        runtime.load_safetensors(
            &graph,
            weights_path
                .to_str()
                .ok_or(DecoderError::InvalidGeometry("weights path"))?,
        );
    }
    let persistent_cache = register_persistent_cache(&mut runtime, &decoder, plan, config)?;
    let mut fixed_state_scratch = Vec::with_capacity(decoder.outputs.fixed_states.len());
    for state in &decoder.outputs.fixed_states {
        state
            .binding
            .seed_destination_slots(&mut runtime, 1, compile.maximum_batch_size)?;
        fixed_state_scratch.push(
            state
                .binding
                .allocate_compile_scratch(&mut runtime, state.policy),
        );
    }
    let stateful = !decoder.outputs.fixed_states.is_empty();
    let representative_query_tokens = if stateful {
        1
    } else {
        compile.representative_prefill_tokens
    };
    graph.set_dim('s', representative_query_tokens);
    graph.set_dim('b', 1);
    for class in &decoder.class_dimensions {
        graph.set_dim(class.context_pages, compile.representative_context_pages);
    }
    seed_compile_inputs(
        &mut runtime,
        &decoder,
        compile,
        page_tokens,
        representative_query_tokens,
    );
    let options = decoder_compile_options(&decoder, compile, stateful)
        .compiler_facts(compiler_facts.egglog().to_owned());
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
