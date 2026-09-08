#![cfg(feature = "cuda")]
#![forbid(unsafe_code)]

use luminal::{
    dtype::DType,
    op::Runtime,
    prelude::{CompileOptions, Graph, ToId},
};
use luminal_cuda_lite::{cudarc::driver::CudaContext, runtime::CudaRuntime};
use orbitkv::{
    AttentionStatePlanInput, AttentionStateSpec, AttentionStateStorage, CacheSharingPolicy,
    EngineAppendIntent, EngineCompletionEvidence, EnginePublicationEvidence, EngineReleaseEvidence,
    EngineReleaseOutcome, EngineRequestId, EngineRetirementEvidence, RecurrentFamily,
    RuntimeSession, StateCheckpointPool, compile_attention_state_plan, compile_plan,
    compile_runtime_manifest,
    kv_manager::{BackendArenaRegistration, CanonicalKvManager, ManagerConfig},
    plan::RetentionKind,
};
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
    value_output: luminal::prelude::GraphTensor,
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
    stream: std::sync::Arc<luminal_cuda_lite::cudarc::driver::CudaStream>,
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
    let compile_scratch = graph_binding.allocate_compile_scratch(&mut runtime);
    runtime = graph.compile(runtime, CompileOptions::default().search_graph_limit(8));
    let binding = state_arenas
        .bind_graph_state(
            &mut runtime,
            graph_binding,
            FixedStateWritePolicy::RequiredInPlace,
        )
        .expect("bind OrbitKV arena to Luminal");
    drop(compile_scratch);
    seed_recurrent_inputs(&mut runtime, &inputs);
    RecurrentExecution {
        graph,
        runtime,
        binding,
        graph_binding,
        value_output,
    }
}

fn seed_recurrent_inputs(runtime: &mut CudaRuntime, inputs: &[luminal::prelude::GraphTensor; 5]) {
    runtime.set_data(inputs[0], vec![1.0_f32, 0.0, 0.0, 0.0]);
    runtime.set_data(inputs[1], vec![1.0_f32, 0.0, 0.0, 0.0]);
    runtime.set_data(inputs[2], vec![4.0_f32, 3.0, 2.0, 1.0]);
    runtime.set_data(inputs[3], vec![0.0_f32]);
    runtime.set_data(inputs[4], vec![0.5_f32]);
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
