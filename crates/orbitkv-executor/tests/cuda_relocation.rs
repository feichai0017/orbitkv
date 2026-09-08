#![cfg(feature = "cuda")]
#![forbid(unsafe_code)]

use luminal::prelude::*;
use luminal_cuda_lite::{cudarc::driver::CudaContext, runtime::CudaRuntime};
use luminal_nn::scatter_rows;
use orbitkv::{
    AttentionStatePlanInput, AttentionStateSpec, AttentionStateStorage, CacheSharingPolicy,
    EngineAppendIntent, EngineCompletionEvidence, EnginePublicationEvidence,
    EngineRelocationPublicationEvidence, EngineRequestId, EngineRetirementEvidence,
    EngineTokenDispositionBatchItem, EngineTokenDispositionUpdate, RuntimeSession,
    compile_attention_state_plan, compile_plan, compile_runtime_manifest,
    kv_manager::{
        BackendArenaRegistration, CanonicalKvManager, ManagerConfig, RelocationPolicy,
        TokenDisposition,
    },
    plan::RetentionKind,
};
use orbitkv_executor::cuda::{
    AttentionKernel, PagedAttentionInputs, PagedAttentionMetadata, paged_attention,
};
use orbitkv_executor::{ExecutorArena, ExecutorPlan, RelocationBatch, cuda::KvCacheBinding};

const PAGE_TOKENS: u64 = 16;
const PAGE_COUNT: u32 = 8;
const HEAD_DIM: usize = 64;
const TOKEN_BYTES: u64 = (HEAD_DIM * 2) as u64;
const COMPLETION_DOMAIN: u64 = 9;

struct RelocationFixture {
    session: RuntimeSession,
    executor: ExecutorPlan,
    arena: ExecutorArena,
    request_id: EngineRequestId,
}

fn fixture() -> RelocationFixture {
    let state = AttentionStateSpec {
        name: "attention".into(),
        layers: vec![0],
        storage: AttentionStateStorage::TokenKv {
            key_bytes_per_token_per_layer: TOKEN_BYTES,
            value_bytes_per_token_per_layer: TOKEN_BYTES,
            retention: RetentionKind::Full,
            window_tokens: None,
        },
    };
    let input = AttentionStatePlanInput {
        page_tokens: PAGE_TOKENS,
        states: vec![state],
    };
    let manifest = compile_runtime_manifest(input.clone()).unwrap();
    let manager_plan = compile_plan(
        compile_attention_state_plan(input)
            .unwrap()
            .token_manager_plan()
            .unwrap(),
    )
    .unwrap();
    let registration = BackendArenaRegistration {
        pool_id: 1,
        class_id: 0,
        backend_domain: 1,
        page_count: PAGE_COUNT,
        reserved: 0,
        backend_base_index: 0,
    };
    let manager = CanonicalKvManager::new(
        &manager_plan,
        ManagerConfig {
            maximum_requests: 1,
            maximum_operations: 2,
            maximum_prefixes: 1,
            maximum_reclamations: PAGE_COUNT,
            maximum_step_tokens: 64,
        },
        &[registration],
    )
    .unwrap();
    let session = RuntimeSession::new(manager, CacheSharingPolicy::RequestPrivate);
    let arena = ExecutorArena::bind(session.arena_stats()[0], registration).unwrap();
    RelocationFixture {
        session,
        executor: ExecutorPlan::compile(&manifest).unwrap(),
        arena,
        request_id: EngineRequestId(1),
    }
}

fn append_initial(fixture: &mut RelocationFixture) {
    fixture
        .session
        .acquire_requests(&[fixture.request_id])
        .unwrap();
    let append = fixture
        .session
        .prepare_append_batch(&[EngineAppendIntent {
            request_id: fixture.request_id,
            target_boundary: 48,
        }])
        .unwrap();
    let lowered = fixture
        .executor
        .lower_prepared(append, &[fixture.arena])
        .unwrap();
    let evidence = lowered
        .execution_evidence_after_success(&[fixture.arena])
        .unwrap();
    let ticket = fixture.session.submit_execution(&evidence).unwrap();
    let publication = fixture
        .session
        .complete_execution_by_batch(
            ticket.batch_id(),
            EngineCompletionEvidence {
                completion_domain: COMPLETION_DOMAIN,
                completion_value: 1,
                confirmed: true,
            },
        )
        .unwrap();
    fixture
        .session
        .confirm_publication(&EnginePublicationEvidence {
            publication_id: publication.publication_id,
            mirror_cleanup_confirmed: true,
            reclamation_receipts: Box::default(),
        })
        .unwrap();
}

fn prepare_relocation(fixture: &mut RelocationFixture) -> RelocationBatch {
    let updates = (0..48)
        .filter(|token| token % PAGE_TOKENS >= 8)
        .map(|token_id| EngineTokenDispositionUpdate {
            class_id: 0,
            token_id,
            disposition: TokenDisposition::policy_evicted(1, 1, 1),
        })
        .collect::<Vec<_>>();
    fixture
        .session
        .mark_token_dispositions_batch(&[EngineTokenDispositionBatchItem {
            request_id: fixture.request_id,
            updates: updates.into_boxed_slice(),
        }])
        .unwrap();
    let prepared = fixture
        .session
        .prepare_relocation_batch(&[orbitkv::EnginePrepareRelocationItem {
            request_id: fixture.request_id,
            class_id: 0,
            policy: RelocationPolicy::static_fragmentation(250, 8, 2, true),
        }])
        .unwrap();
    fixture
        .executor
        .lower_relocation(&prepared, &[fixture.arena])
        .unwrap()
}

fn complete_relocation(
    fixture: &mut RelocationFixture,
    batch: &RelocationBatch,
    evidence: &orbitkv::EngineRelocationExecutionEvidence,
) {
    let ticket = fixture.session.submit_relocation(evidence).unwrap();
    let publication = fixture
        .session
        .complete_relocation(
            ticket.relocation_id(),
            EngineCompletionEvidence {
                completion_domain: COMPLETION_DOMAIN,
                completion_value: 2,
                confirmed: true,
            },
        )
        .unwrap();
    assert_eq!(publication.relocation_id, batch.relocation_id());
    fixture
        .session
        .confirm_relocation_publication(&EngineRelocationPublicationEvidence {
            relocation_id: publication.relocation_id,
            mirror_cleanup_confirmed: true,
            reclamation_receipts: publication
                .retirements
                .iter()
                .map(|retirement| EngineRetirementEvidence {
                    page: retirement.page,
                    backend_domain: retirement.backend_domain,
                    acknowledged: true,
                    backend_index: retirement.backend_index,
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        })
        .unwrap();
}

fn token_rows(slots: usize, offset: f32) -> Vec<bf16> {
    (0..slots)
        .flat_map(|slot| {
            let value = bf16::from_f32(f32::from(u16::try_from(slot).unwrap()) + offset);
            std::iter::repeat_n(value, HEAD_DIM)
        })
        .collect()
}

fn verify_relocated_rows(batch: &RelocationBatch, source: &[bf16], relocated: &[u8]) {
    for movement in &batch.requests()[0].copies {
        let source_slot = usize::try_from(movement.source_slot).unwrap();
        let destination_slot = usize::try_from(movement.destination_slot).unwrap();
        let source_begin = source_slot * HEAD_DIM;
        let destination_begin = destination_slot * HEAD_DIM * 2;
        let expected = source[source_begin..source_begin + HEAD_DIM]
            .iter()
            .flat_map(|value| value.to_bits().to_ne_bytes())
            .collect::<Vec<_>>();
        assert_eq!(
            &relocated[destination_begin..destination_begin + HEAD_DIM * 2],
            expected
        );
    }
}

struct DecodeGraph {
    graph: Graph,
    q: GraphTensor,
    new_key: GraphTensor,
    new_value: GraphTensor,
    write_slot: GraphTensor,
    key_cache: GraphTensor,
    value_cache: GraphTensor,
    metadata: PagedAttentionMetadata,
    output: GraphTensor,
}

fn build_decode_graph(fixture: &RelocationFixture) -> DecodeGraph {
    let slots = usize::try_from(PAGE_COUNT).unwrap() * usize::try_from(PAGE_TOKENS).unwrap();
    let mut graph = Graph::default();
    let q = graph
        .named_tensor("q", (1, 1, HEAD_DIM))
        .as_dtype(DType::Bf16);
    let new_key = graph
        .named_tensor("new_key", (1, HEAD_DIM))
        .as_dtype(DType::Bf16);
    let new_value = graph
        .named_tensor("new_value", (1, HEAD_DIM))
        .as_dtype(DType::Bf16);
    let write_slot = graph.named_tensor("write_slot", 1).as_dtype(DType::Int);
    let key_cache = graph
        .named_tensor("key_cache", (slots, HEAD_DIM))
        .as_dtype(DType::Bf16);
    let value_cache = graph
        .named_tensor("value_cache", (slots, HEAD_DIM))
        .as_dtype(DType::Bf16);
    let key_update = scatter_rows(new_key, write_slot, key_cache, HEAD_DIM);
    let value_update = scatter_rows(new_value, write_slot, value_cache, HEAD_DIM);
    let metadata = PagedAttentionMetadata::new(&mut graph, 0, 1.into(), 2.into());
    let output = paged_attention(
        PagedAttentionInputs {
            q,
            k_cache: key_update,
            v_cache: value_update,
            query_tokens: 1.into(),
            context_pages: 2.into(),
        },
        metadata,
        &fixture.executor.classes[0],
        AttentionKernel {
            query_heads: 1,
            kv_heads: 1,
            head_dim: HEAD_DIM,
            dtype: DType::Bf16,
            softmax_scale: 0.0,
        },
    )
    .unwrap()
    .output();
    DecodeGraph {
        graph,
        q,
        new_key,
        new_value,
        write_slot,
        key_cache,
        value_cache,
        metadata,
        output,
    }
}

fn upload_decode_inputs(
    runtime: &mut CudaRuntime,
    decoder: &DecodeGraph,
    attention: &orbitkv_executor::AttentionBatch,
    write_slot: u64,
) {
    runtime.set_data(decoder.q, vec![bf16::from_f32(0.0); HEAD_DIM]);
    runtime.set_data(decoder.new_key, vec![bf16::from_f32(0.0); HEAD_DIM]);
    runtime.set_data(decoder.new_value, vec![bf16::from_f32(100.0); HEAD_DIM]);
    runtime.set_data(decoder.write_slot, vec![i32::try_from(write_slot).unwrap()]);
    decoder.metadata.upload(runtime, attention).unwrap();
}

fn run_packed_decode(
    fixture: &mut RelocationFixture,
    key_buffer: luminal_cuda_lite::cudarc::driver::CudaSlice<u8>,
    value_buffer: luminal_cuda_lite::cudarc::driver::CudaSlice<u8>,
    stream: std::sync::Arc<luminal_cuda_lite::cudarc::driver::CudaStream>,
) {
    let source = fixture
        .session
        .prepare_append_batch(&[EngineAppendIntent {
            request_id: fixture.request_id,
            target_boundary: 49,
        }])
        .unwrap();
    let view = fixture
        .session
        .prepared_execution_view(source.batch_id)
        .unwrap();
    let attention = fixture.executor.attention_batch(0, &view).unwrap();
    let prepared = fixture
        .executor
        .lower_prepared(source, &[fixture.arena])
        .unwrap();
    assert_eq!(attention.page_indices.len(), 2);
    assert_eq!(&*attention.last_page_len, &[9]);
    let mut decoder = build_decode_graph(fixture);
    let slots = usize::try_from(PAGE_COUNT).unwrap() * usize::try_from(PAGE_TOKENS).unwrap();
    let cache_bytes = slots * HEAD_DIM * 2;
    let mut runtime = CudaRuntime::initialize(stream);
    runtime.set_zeros(decoder.key_cache, cache_bytes);
    runtime.set_zeros(decoder.value_cache, cache_bytes);
    let write_slot = prepared.steps()[0].classes[0].write_slots[0];
    upload_decode_inputs(&mut runtime, &decoder, &attention, write_slot);
    let mut runtime = decoder
        .graph
        .compile(runtime, CompileOptions::default().search_graph_limit(1));
    runtime.set_buffer(decoder.key_cache, key_buffer);
    runtime.set_buffer(decoder.value_cache, value_buffer);
    upload_decode_inputs(&mut runtime, &decoder, &attention, write_slot);
    runtime.execute(&decoder.graph.dyn_map);
    let actual = runtime.get_bf16(decoder.output);
    let expected = 592.0 / 25.0;
    assert!(
        actual
            .iter()
            .all(|value| (value.to_f32() - expected).abs() <= 0.25),
        "expected {expected}, got {:?}",
        actual.first()
    );
    let evidence = prepared
        .execution_evidence_after_success(&[fixture.arena])
        .unwrap();
    let ticket = fixture.session.submit_execution(&evidence).unwrap();
    let publication = fixture
        .session
        .complete_execution_by_batch(
            ticket.batch_id(),
            EngineCompletionEvidence {
                completion_domain: COMPLETION_DOMAIN,
                completion_value: 3,
                confirmed: true,
            },
        )
        .unwrap();
    fixture
        .session
        .confirm_publication(&EnginePublicationEvidence {
            publication_id: publication.publication_id,
            mirror_cleanup_confirmed: true,
            reclamation_receipts: Box::default(),
        })
        .unwrap();
}

#[test]
#[ignore = "requires a CUDA device"]
fn manager_authored_token_moves_execute_and_publish() {
    let mut fixture = fixture();
    append_initial(&mut fixture);
    let batch = prepare_relocation(&mut fixture);
    assert_eq!(batch.requests()[0].copies.len(), 24);

    let slots = usize::try_from(PAGE_COUNT).unwrap() * usize::try_from(PAGE_TOKENS).unwrap();
    let key_data = vec![bf16::from_f32(0.0); slots * HEAD_DIM];
    let value_data = token_rows(slots, 1.0);
    let mut graph = Graph::default();
    let key = graph
        .named_tensor("key", (slots, HEAD_DIM))
        .as_dtype(DType::Bf16)
        .persist();
    let value = graph
        .named_tensor("value", (slots, HEAD_DIM))
        .as_dtype(DType::Bf16)
        .persist();
    let key_output = key.output();
    let value_output = value.output();
    let context = CudaContext::new(0).unwrap();
    let stream = context.default_stream();
    let mut runtime = CudaRuntime::initialize(stream.clone());
    runtime.set_data(key, key_data.clone());
    runtime.set_data(value, value_data.clone());
    let mut runtime = graph.compile(runtime, CompileOptions::default().search_graph_limit(1));
    runtime.set_data(key, key_data.clone());
    runtime.set_data(value, value_data.clone());
    runtime.execute(&graph.dyn_map);

    let bindings = [KvCacheBinding {
        class_id: 0,
        layer: 0,
        key,
        value,
    }];
    let mut samples = Vec::new();
    let mut evidence = None;
    for _ in 0..3 {
        let completed = batch
            .enqueue(&runtime, &bindings)
            .unwrap()
            .wait_measured()
            .unwrap();
        assert!(completed.sample.bytes > 0);
        assert!(!completed.sample.device_time.is_zero());
        samples.push(completed.sample);
        evidence = Some(completed.evidence);
    }
    let _bandwidth =
        orbitkv_executor::model::RelocationBandwidthProfile::from_samples(&samples).unwrap();
    let evidence = evidence.unwrap();
    let key_buffer = runtime.remove_buffer(key_output);
    let value_buffer = runtime.remove_buffer(value_output);
    let copied_key_bytes = stream.clone_dtoh(&key_buffer).unwrap();
    let copied_value_bytes = stream.clone_dtoh(&value_buffer).unwrap();
    verify_relocated_rows(&batch, &key_data, &copied_key_bytes);
    verify_relocated_rows(&batch, &value_data, &copied_value_bytes);
    complete_relocation(&mut fixture, &batch, &evidence);
    assert_eq!(fixture.session.stats().active_pages, 2);
    run_packed_decode(&mut fixture, key_buffer, value_buffer, stream);
    assert_eq!(fixture.session.stats().active_pages, 2);
}
