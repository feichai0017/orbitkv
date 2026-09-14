#![cfg(feature = "cuda")]
#![forbid(unsafe_code)]

use half::bf16;
use orbitkv::{
    AttentionStatePlanInput, AttentionStateSpec, AttentionStateStorage, CacheSharingPolicy,
    EngineAppendIntent, EngineCompletionEvidence, EnginePublicationEvidence, EngineReleaseEvidence,
    EngineReleaseOutcome, EngineRequestId, EngineRetirementEvidence, RecurrentFamily,
    RuntimeSession, StateCheckpointPool, compile_attention_state_plan, compile_plan,
    compile_runtime_manifest,
    kv_manager::{BackendArenaRegistration, CanonicalKvManager, ManagerConfig},
    plan::RetentionKind,
};
use orbitkv_compiler::{
    dtype::DType,
    op::Runtime,
    prelude::{CompileOptions, Expression, Graph, ToId},
};
use orbitkv_cuda::kernel::sequence_state::{
    PackedConvolutionPlan, PackedConvolutionSpec, PackedDeltaScanPlan, PackedDeltaScanSpec,
    packed_causal_convolution, packed_delta_scan,
};
use orbitkv_cuda::{cudarc::driver::CudaContext, runtime::CudaRuntime};
use orbitkv_executor::{
    ExecutorArena, ExecutorPlan, FixedStateDeviceArenas, FixedStateDeviceBatch,
    FixedStateExecutionEvidence, FixedStateGraphBinding, FixedStateWritePolicy, GatedDeltaGeometry,
    GatedDeltaStepInputs, PreparedBatch, RecurrentStateGraphArena, gated_delta_step,
};

const STATE_ID: u16 = 1;
const STATE_BYTES: u64 = 64;

struct TestControlPlane {
    session: RuntimeSession,
    executor_plan: ExecutorPlan,
    token_arena: ExecutorArena,
    state_identity: orbitkv::StatePoolIdentity,
}

struct RecurrentExecution {
    graph: Graph,
    runtime: CudaRuntime,
    binding: orbitkv_executor::FixedStateRuntimeBinding,
    graph_binding: FixedStateGraphBinding,
    inputs: [orbitkv_compiler::prelude::GraphTensor; 5],
    value_output: orbitkv_compiler::prelude::GraphTensor,
}

fn state_input() -> AttentionStatePlanInput {
    AttentionStatePlanInput {
        page_tokens: 16,
        states: vec![
            AttentionStateSpec {
                name: "attention".into(),
                layers: vec![1],
                storage: AttentionStateStorage::TokenKv {
                    key_bytes_per_token_per_layer: 128,
                    value_bytes_per_token_per_layer: 128,
                    retention: RetentionKind::Full,
                    window_tokens: None,
                },
            },
            AttentionStateSpec {
                name: "recurrent".into(),
                layers: vec![0],
                storage: AttentionStateStorage::Recurrent {
                    family: RecurrentFamily::Gdn,
                    state_bytes_per_layer: STATE_BYTES,
                    checkpoint_slots_per_request: 2,
                },
            },
        ],
    }
}

fn retirement_evidence(
    retirements: &[orbitkv::EngineRetirement],
) -> Box<[EngineRetirementEvidence]> {
    retirements
        .iter()
        .map(|retirement| EngineRetirementEvidence {
            page: retirement.page,
            backend_domain: retirement.backend_domain,
            acknowledged: true,
            backend_index: retirement.backend_index,
        })
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

fn complete_step(
    session: &mut RuntimeSession,
    prepared: &PreparedBatch,
    arena: ExecutorArena,
    fixed: FixedStateExecutionEvidence,
    completion_value: u64,
) {
    let evidence = prepared
        .execution_evidence_after_state_success(&[arena], &[fixed])
        .expect("merge token and fixed-state evidence");
    let ticket = session
        .submit_execution(&evidence)
        .expect("submit execution");
    let publication = session
        .complete_execution_by_batch(
            ticket.batch_id(),
            EngineCompletionEvidence {
                completion_domain: 1,
                completion_value,
                confirmed: true,
            },
        )
        .expect("complete execution");
    session
        .confirm_publication(&EnginePublicationEvidence {
            publication_id: publication.publication_id,
            mirror_cleanup_confirmed: true,
            reclamation_receipts: retirement_evidence(&publication.retirements),
        })
        .expect("confirm publication");
}

fn stable_base(batch: &FixedStateDeviceBatch) -> u64 {
    let destination = batch.destinations[0];
    destination
        .device_ptr
        .checked_sub(u64::try_from(destination.byte_offset).unwrap())
        .unwrap()
}

fn control_plane() -> TestControlPlane {
    let input = state_input();
    let manifest = compile_runtime_manifest(input.clone()).expect("compile runtime manifest");
    let manager_plan = compile_plan(
        compile_attention_state_plan(input)
            .expect("compile attention state")
            .token_manager_plan()
            .expect("token manager plan"),
    )
    .expect("compile manager plan");
    let registration = BackendArenaRegistration {
        pool_id: 41,
        class_id: 0,
        backend_domain: 7,
        page_count: 4,
        reserved: 0,
        backend_base_index: 8,
    };
    let manager = CanonicalKvManager::new(
        &manager_plan,
        ManagerConfig {
            maximum_requests: 1,
            maximum_operations: 4,
            maximum_prefixes: 1,
            maximum_reclamations: 4,
            maximum_step_tokens: 16,
        },
        &[registration],
    )
    .expect("create manager");
    let arena_stats = manager.arena_stats()[0];
    let state_pool = StateCheckpointPool::new(
        arena_stats.engine_epoch,
        arena_stats.pool_epoch + 1,
        42,
        STATE_BYTES,
        2,
    )
    .expect("create fixed-state pool");
    let state_identity = state_pool.identity();
    let session = RuntimeSession::with_fixed_states(
        manager,
        CacheSharingPolicy::RequestPrivate,
        [(STATE_ID, state_pool)],
    )
    .expect("create joint session");
    TestControlPlane {
        session,
        executor_plan: ExecutorPlan::compile(&manifest).expect("compile executor plan"),
        token_arena: ExecutorArena::bind(arena_stats, registration).expect("bind token arena"),
        state_identity,
    }
}

fn execute_step(
    control: &mut TestControlPlane,
    state_arenas: &FixedStateDeviceArenas,
    execution: &mut RecurrentExecution,
    request_id: EngineRequestId,
    target_boundary: u64,
    completion_value: u64,
) -> FixedStateDeviceBatch {
    let plan = control
        .session
        .prepare_append_batch(&[EngineAppendIntent {
            request_id,
            target_boundary,
        }])
        .expect("prepare step");
    let prepared = control
        .executor_plan
        .lower_prepared(plan, &[control.token_arena])
        .expect("lower step");
    let device = state_arenas
        .prepare_batch(prepared.fixed_state_requests())
        .expect("resolve state batch slots");
    let initialized = device.initialize().expect("initialize state transition");
    let ranges = initialized.ranges()[0].clone();
    let ready = initialized
        .upload_destination_slots(&mut execution.runtime, &[execution.graph_binding])
        .expect("upload manager destination slot");
    seed_recurrent_inputs(&mut execution.runtime, &execution.inputs);
    let evidence = ready
        .complete_after(
            &mut execution.runtime,
            &execution.graph,
            std::slice::from_ref(&execution.binding),
        )
        .expect("bind model execution receipt")
        .wait()
        .expect("wait for state event");
    complete_step(
        &mut control.session,
        &prepared,
        control.token_arena,
        evidence[0].clone(),
        completion_value,
    );
    let values = execution.runtime.get_f32(execution.value_output);
    let expected = if completion_value == 1 {
        [1.0, 0.75, 0.5, 0.25]
    } else {
        [1.5, 1.125, 0.75, 0.375]
    };
    for (actual, expected) in values.iter().zip(expected) {
        assert!((actual - expected).abs() <= 1e-6, "{actual} != {expected}");
    }
    ranges
}

fn recurrent_execution(
    control: &TestControlPlane,
    state_arenas: &FixedStateDeviceArenas,
    stream: std::sync::Arc<orbitkv_cuda::cudarc::driver::CudaStream>,
) -> RecurrentExecution {
    let registration = control
        .executor_plan
        .fixed_state_registrations(&[(STATE_ID, control.state_identity)])
        .unwrap()[0];
    let geometry = GatedDeltaGeometry {
        key_heads: 1,
        value_heads: 1,
        key_width: 4,
        value_width: 4,
        normalization_epsilon: 0.0,
    };
    let mut graph = Graph::new();
    let mut state_graph = RecurrentStateGraphArena::new(
        &mut graph,
        &control.executor_plan.fixed_states[0],
        registration,
        1.into(),
    )
    .expect("build recurrent state arena graph");
    let query = graph.named_tensor("query", (1, 1, 4)).as_dtype(DType::F32);
    let key = graph.named_tensor("key", (1, 1, 4)).as_dtype(DType::F32);
    let value = graph.named_tensor("value", (1, 1, 4)).as_dtype(DType::F32);
    let log_decay = graph.named_tensor("log_decay", (1, 1)).as_dtype(DType::F32);
    let update_gate = graph
        .named_tensor("update_gate", (1, 1))
        .as_dtype(DType::F32);
    let previous_state = state_graph.layer_state(0, geometry).unwrap();
    let recurrent = gated_delta_step(
        GatedDeltaStepInputs {
            query,
            key,
            value,
            log_decay,
            update_gate,
            previous_state,
            batch_size: 1.into(),
        },
        geometry,
    )
    .expect("build gated-delta semantics");
    state_graph
        .commit_layer(0, geometry, recurrent.next_state)
        .expect("commit recurrent state");
    let value_output = recurrent.values.output();
    let graph_binding = state_graph.finish();
    let mut runtime = CudaRuntime::initialize(stream);
    let inputs = [query, key, value, log_decay, update_gate];
    seed_recurrent_inputs(&mut runtime, &inputs);
    graph_binding
        .seed_destination_slots(&mut runtime, 1, 1)
        .expect("seed slot metadata");
    let compile_scratch = graph_binding
        .allocate_compile_scratch(&mut runtime, FixedStateWritePolicy::RequiredInPlace);
    runtime = graph.compile(runtime, CompileOptions::default().search_graph_limit(8));
    let binding = state_arenas
        .bind_graph_state(
            &mut runtime,
            graph_binding,
            FixedStateWritePolicy::RequiredInPlace,
        )
        .expect("bind OrbitKV arena to OrbitKV");
    drop(compile_scratch);
    seed_recurrent_inputs(&mut runtime, &inputs);
    RecurrentExecution {
        graph,
        runtime,
        binding,
        graph_binding,
        inputs,
        value_output,
    }
}

fn seed_recurrent_inputs(
    runtime: &mut CudaRuntime,
    inputs: &[orbitkv_compiler::prelude::GraphTensor; 5],
) {
    runtime.set_data(inputs[0], vec![1.0_f32, 0.0, 0.0, 0.0]);
    runtime.set_data(inputs[1], vec![1.0_f32, 0.0, 0.0, 0.0]);
    runtime.set_data(inputs[2], vec![4.0_f32, 3.0, 2.0, 1.0]);
    runtime.set_data(inputs[3], vec![0.0_f32]);
    runtime.set_data(inputs[4], vec![0.5_f32]);
}

#[test]
#[ignore = "requires an SM90 CUDA device"]
#[allow(clippy::too_many_lines)]
fn packed_state_kernels_match_ragged_sequence_references_on_h20() {
    const TOKENS: usize = 5;
    const REQUESTS: usize = 2;
    const CHANNELS: usize = 4;
    const HISTORY_WIDTH: usize = 2;

    let context = CudaContext::new(0).expect("CUDA context");
    context.bind_to_thread().expect("bind CUDA context");
    let stream = context.default_stream();
    let mut graph = Graph::new();
    let input = graph
        .named_tensor("packed_input", ('s', CHANNELS))
        .as_dtype(DType::Bf16);
    let weights = graph
        .named_tensor("convolution_weights", (CHANNELS, HISTORY_WIDTH + 1))
        .as_dtype(DType::Bf16);
    let history = graph
        .named_tensor("convolution_history", ('b', CHANNELS, HISTORY_WIDTH))
        .as_dtype(DType::Bf16);
    let query_indptr = graph
        .named_tensor("query_indptr", Expression::from('b') + 1)
        .as_dtype(DType::Int);
    let convolution = packed_causal_convolution(
        PackedConvolutionPlan {
            input,
            weights,
            history,
            query_indptr,
        },
        PackedConvolutionSpec {
            channels: CHANNELS,
            kernel_width: HISTORY_WIDTH + 1,
        },
    );
    let convolution_values = convolution.values.output();
    let convolution_history = convolution.history.output();
    let recurrent_state = graph.named_tensor("recurrent_state", ('b', 2, 1, 1));
    let log_decay = graph.named_tensor("log_decay", ('s', 2));
    let update_gate = graph.named_tensor("update_gate", ('s', 2));
    let convolved = convolution.values.cast(DType::F32);
    let recurrent = packed_delta_scan(
        PackedDeltaScanPlan {
            query: convolved.slice((.., ..1)).split_dims(1, 1),
            key: convolved.slice((.., 1..2)).split_dims(1, 1),
            value: convolved.slice((.., 2..)).split_dims(1, 1),
            log_decay,
            update_gate,
            state: recurrent_state,
            query_indptr,
        },
        PackedDeltaScanSpec {
            key_heads: 1,
            value_heads: 2,
            key_width: 1,
            value_width: 1,
            normalization_epsilon: 1e-6,
        },
    );
    let recurrent_values = recurrent.values.output();
    let recurrent_state_output = recurrent.state.output();

    let input_data = [
        0.5, 1.0, 1.5, 2.0, -0.5, 0.25, 0.75, 1.25, 1.0, -0.75, 0.5, 1.5, 0.25, 0.5, -1.0, 0.75,
        1.25, 0.75, 1.0, -0.5,
    ];
    let weight_data = [
        0.25, -0.5, 1.0, -0.25, 0.75, 0.5, 0.5, 0.25, -0.75, 0.1, 0.2, 0.8,
    ];
    let history_data = [
        0.1, -0.2, 0.3, 0.4, -0.5, 0.25, 0.75, -0.1, -0.25, 0.5, 0.2, -0.4, 0.6, 0.3, -0.75, 0.2,
    ];
    let indptr = vec![0_i32, 2, 5];
    let log_decay_data = vec![
        -0.1, -0.2, -0.3, -0.15, -0.25, -0.4, -0.05, -0.2, -0.35, -0.1,
    ];
    let update_gate_data = vec![0.2, 0.4, 0.6, 0.3, 0.5, 0.7, 0.8, 0.45, 0.65, 0.25];
    let state_data = vec![0.1, -0.2, 0.3, 0.4];
    let input_bf16 = as_bf16(&input_data);
    let weights_bf16 = as_bf16(&weight_data);
    let history_bf16 = as_bf16(&history_data);

    graph.set_dim('s', TOKENS);
    graph.set_dim('b', REQUESTS);
    let mut runtime = CudaRuntime::initialize(stream);
    set_packed_inputs(
        &mut runtime,
        &[
            input,
            weights,
            history,
            query_indptr,
            log_decay,
            update_gate,
            recurrent_state,
        ],
        &input_bf16,
        &weights_bf16,
        &history_bf16,
        &indptr,
        &log_decay_data,
        &update_gate_data,
        &state_data,
    );
    runtime = graph.compile(runtime, CompileOptions::default().search_graph_limit(1));
    set_packed_inputs(
        &mut runtime,
        &[
            input,
            weights,
            history,
            query_indptr,
            log_decay,
            update_gate,
            recurrent_state,
        ],
        &input_bf16,
        &weights_bf16,
        &history_bf16,
        &indptr,
        &log_decay_data,
        &update_gate_data,
        &state_data,
    );
    runtime.execute(&graph.dyn_map);

    let (expected_convolution, expected_history) =
        packed_convolution_reference(&input_bf16, &weights_bf16, &history_bf16, &[0, 2, 5]);
    assert_bf16_close(
        &runtime.get_bf16(convolution_values),
        &expected_convolution,
        2e-2,
    );
    assert_bf16_close(
        &runtime.get_bf16(convolution_history),
        &expected_history,
        0.0,
    );
    let (expected_values, expected_state) = packed_delta_reference(
        &expected_convolution,
        &log_decay_data,
        &update_gate_data,
        &state_data,
        &[0, 2, 5],
    );
    assert_f32_close(&runtime.get_f32(recurrent_values), &expected_values, 2e-5);
    assert_f32_close(
        &runtime.get_f32(recurrent_state_output),
        &expected_state,
        2e-5,
    );
}

#[allow(clippy::too_many_arguments)]
fn set_packed_inputs(
    runtime: &mut CudaRuntime,
    tensors: &[orbitkv_compiler::prelude::GraphTensor; 7],
    input: &[bf16],
    weights: &[bf16],
    history: &[bf16],
    query_indptr: &[i32],
    log_decay: &[f32],
    update_gate: &[f32],
    state: &[f32],
) {
    runtime.set_data(tensors[0], input.to_vec());
    runtime.set_data(tensors[1], weights.to_vec());
    runtime.set_data(tensors[2], history.to_vec());
    runtime.set_data(tensors[3], query_indptr.to_vec());
    runtime.set_data(tensors[4], log_decay.to_vec());
    runtime.set_data(tensors[5], update_gate.to_vec());
    runtime.set_data(tensors[6], state.to_vec());
}

fn packed_convolution_reference(
    input: &[bf16],
    weights: &[bf16],
    initial_history: &[bf16],
    query_indptr: &[usize],
) -> (Vec<bf16>, Vec<bf16>) {
    const CHANNELS: usize = 4;
    const HISTORY_WIDTH: usize = 2;
    let mut values = vec![bf16::ZERO; input.len()];
    let mut histories = initial_history.to_vec();
    for request in 0..query_indptr.len() - 1 {
        let history = &mut histories
            [request * CHANNELS * HISTORY_WIDTH..(request + 1) * CHANNELS * HISTORY_WIDTH];
        for token in query_indptr[request]..query_indptr[request + 1] {
            for channel in 0..CHANNELS {
                let history_base = channel * HISTORY_WIDTH;
                let weight_base = channel * (HISTORY_WIDTH + 1);
                let mut value = 0.0_f32;
                for offset in 0..HISTORY_WIDTH {
                    value = history[history_base + offset]
                        .to_f32()
                        .mul_add(weights[weight_base + offset].to_f32(), value);
                }
                value = input[token * CHANNELS + channel]
                    .to_f32()
                    .mul_add(weights[weight_base + HISTORY_WIDTH].to_f32(), value);
                values[token * CHANNELS + channel] = bf16::from_f32(value / (1.0 + (-value).exp()));
                history[history_base] = history[history_base + 1];
                history[history_base + 1] = input[token * CHANNELS + channel];
            }
        }
    }
    (values, histories)
}

fn packed_delta_reference(
    convolved: &[bf16],
    log_decay: &[f32],
    update_gate: &[f32],
    initial_state: &[f32],
    query_indptr: &[usize],
) -> (Vec<f32>, Vec<f32>) {
    let mut values = Vec::with_capacity(query_indptr.last().copied().unwrap_or_default() * 2);
    let mut states = Vec::with_capacity(initial_state.len());
    for request in 0..query_indptr.len() - 1 {
        let token_range = query_indptr[request]..query_indptr[request + 1];
        let mut query = Vec::with_capacity(token_range.len());
        let mut key = Vec::with_capacity(token_range.len());
        let mut value = Vec::with_capacity(token_range.len() * 2);
        let mut decay = Vec::with_capacity(token_range.len() * 2);
        let mut gate = Vec::with_capacity(token_range.len() * 2);
        for token in token_range.clone() {
            query.push(convolved[token * 4].to_f32());
            key.push(convolved[token * 4 + 1].to_f32());
            value.extend([
                convolved[token * 4 + 2].to_f32(),
                convolved[token * 4 + 3].to_f32(),
            ]);
            decay.extend_from_slice(&log_decay[token * 2..token * 2 + 2]);
            gate.extend_from_slice(&update_gate[token * 2..token * 2 + 2]);
        }
        let result = orbitkv_executor::gated_delta_reference(
            GatedDeltaGeometry {
                key_heads: 1,
                value_heads: 2,
                key_width: 1,
                value_width: 1,
                normalization_epsilon: 1e-6,
            },
            orbitkv_executor::GatedDeltaReferenceInput {
                batch_size: 1,
                sequence_tokens: token_range.len(),
                query: &query,
                key: &key,
                value: &value,
                log_decay: &decay,
                update_gate: &gate,
                initial_state: &initial_state[request * 2..request * 2 + 2],
            },
        )
        .expect("packed delta reference");
        values.extend_from_slice(&result.values);
        states.extend_from_slice(&result.state);
    }
    (values, states)
}

fn as_bf16(values: &[f32]) -> Vec<bf16> {
    values.iter().copied().map(bf16::from_f32).collect()
}

fn assert_bf16_close(actual: &[bf16], expected: &[bf16], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        let actual = actual.to_f32();
        let expected = expected.to_f32();
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} != {expected}"
        );
    }
}

fn assert_f32_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} != {expected}"
        );
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn fixed_state_arena_preserves_address_and_event_order_across_steps() {
    let mut control = control_plane();
    let context = CudaContext::new(0).expect("CUDA context");
    context.bind_to_thread().expect("bind CUDA context");
    let stream = context.default_stream();
    let state_arenas = FixedStateDeviceArenas::allocate(
        &control.executor_plan,
        &[(STATE_ID, control.state_identity)],
        stream.clone(),
    )
    .expect("allocate stable state arena");
    assert_eq!(state_arenas.arena_count(), 1);

    let mut execution = recurrent_execution(&control, &state_arenas, stream);

    let request_id = EngineRequestId(9);
    control
        .session
        .acquire_requests(&[request_id])
        .expect("acquire request");
    let first_ranges = execute_step(
        &mut control,
        &state_arenas,
        &mut execution,
        request_id,
        1,
        1,
    );
    assert!(first_ranges.sources[0].is_none());
    let arena_base = stable_base(&first_ranges);
    assert_eq!(
        execution
            .runtime
            .input_allocation(execution.graph_binding.arena_input.to_id()),
        Some((arena_base, usize::try_from(STATE_BYTES * 2).unwrap()))
    );
    let second_ranges = execute_step(
        &mut control,
        &state_arenas,
        &mut execution,
        request_id,
        2,
        2,
    );
    let source = second_ranges.sources[0].expect("published source state");
    assert_eq!(source, first_ranges.destinations[0]);
    assert_ne!(
        second_ranges.destinations[0].byte_offset,
        source.byte_offset
    );
    assert_eq!(stable_base(&second_ranges), arena_base);
    let release = control
        .session
        .prepare_release_batch(&[request_id])
        .expect("prepare request release");
    assert_eq!(
        control.session.confirm_release(&EngineReleaseEvidence {
            release_id: release.release_id,
            mirror_cleanup_confirmed: true,
            reclamation_receipts: retirement_evidence(&release.retirements),
        }),
        Ok(EngineReleaseOutcome::Completed)
    );
    let state_stats = control.session.fixed_state_stats();
    assert_eq!(state_stats[0].1.free_slots, 2);
    assert_eq!(state_stats[0].1.live_slots, 0);
    assert_eq!(control.session.stats().active_requests, 0);
    assert_eq!(control.session.stats().active_pages, 0);
}
