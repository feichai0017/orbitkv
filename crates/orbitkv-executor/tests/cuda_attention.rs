#![cfg(feature = "cuda")]
#![forbid(unsafe_code)]

use luminal::prelude::*;
use luminal_cuda_lite::runtime::CudaRuntime;
use orbitkv::{
    AttentionStatePlanInput, AttentionStateSpec, AttentionStateStorage, CacheSharingPolicy,
    EngineAppendIntent, EngineCompletionEvidence, EnginePublicationEvidence, EngineRequestId,
    RuntimeSession, compile_plan, compile_runtime_manifest,
    kv_manager::{BackendArenaRegistration, CanonicalKvManager, ManagerConfig},
    plan::RetentionKind,
};
use orbitkv_executor::{
    AttentionBatch, ExecutorArena, ExecutorPlan, PreparedBatch,
    cuda::{AttentionKernel, PagedAttentionInputs, PagedAttentionMetadata, paged_attention},
};

const PAGE_TOKENS: usize = 16;
const HEAD_DIM: usize = 64;
const PHYSICAL_PAGES: usize = 4;

fn upload_inputs(
    runtime: &mut CudaRuntime,
    q: &GraphTensor,
    k: &GraphTensor,
    v: &GraphTensor,
    metadata: &PagedAttentionMetadata,
    batch: &AttentionBatch,
) {
    let zero = bf16::from_f32(0.0);
    let poison = bf16::from_f32(1_000.0);
    let mut values = vec![poison; PHYSICAL_PAGES * PAGE_TOKENS * HEAD_DIM];
    let first_page = usize::try_from(batch.page_indices[0]).unwrap();
    let second_page = usize::try_from(batch.page_indices[1]).unwrap();
    for token in 0..PAGE_TOKENS {
        let value = bf16::from_f32(f32::from(u16::try_from(token + 1).unwrap()));
        let begin = (first_page * PAGE_TOKENS + token) * HEAD_DIM;
        values[begin..begin + HEAD_DIM].fill(value);
    }
    for token in 0..2 {
        let value = bf16::from_f32(f32::from(u16::try_from(PAGE_TOKENS + token + 1).unwrap()));
        let begin = (second_page * PAGE_TOKENS + token) * HEAD_DIM;
        values[begin..begin + HEAD_DIM].fill(value);
    }
    runtime.set_data(*q, vec![zero; HEAD_DIM]);
    runtime.set_data(*k, vec![zero; PHYSICAL_PAGES * PAGE_TOKENS * HEAD_DIM]);
    runtime.set_data(*v, values);
    (*metadata).upload(runtime, batch).unwrap();
}

fn prepared_full_batch() -> (
    RuntimeSession,
    ExecutorPlan,
    [ExecutorArena; 1],
    PreparedBatch,
    AttentionBatch,
) {
    let state = AttentionStateSpec {
        name: "attention".into(),
        layers: vec![0],
        storage: AttentionStateStorage::TokenKv {
            key_bytes_per_token_per_layer: 2 * HEAD_DIM as u64,
            value_bytes_per_token_per_layer: 2 * HEAD_DIM as u64,
            retention: RetentionKind::Full,
            window_tokens: None,
        },
    };
    let input = AttentionStatePlanInput {
        page_tokens: PAGE_TOKENS as u64,
        states: vec![state],
    };
    let manifest = compile_runtime_manifest(input.clone()).unwrap();
    let manager_plan = compile_plan(
        orbitkv::compile_attention_state_plan(input)
            .unwrap()
            .token_manager_plan()
            .unwrap(),
    )
    .unwrap();
    let registration = BackendArenaRegistration {
        pool_id: 1,
        class_id: 0,
        backend_domain: 1,
        page_count: 2,
        reserved: 0,
        backend_base_index: 2,
    };
    let manager = CanonicalKvManager::new(
        &manager_plan,
        ManagerConfig {
            maximum_requests: 1,
            maximum_operations: 1,
            maximum_prefixes: 1,
            maximum_reclamations: 2,
            maximum_step_tokens: 32,
        },
        &[registration],
    )
    .unwrap();
    let mut session = RuntimeSession::new(manager, CacheSharingPolicy::RequestPrivate);
    let request_id = EngineRequestId(1);
    session.acquire_requests(&[request_id]).unwrap();
    let executor_plan = ExecutorPlan::compile(&manifest).unwrap();
    let initial = session
        .prepare_append_batch(&[EngineAppendIntent {
            request_id,
            target_boundary: 17,
        }])
        .unwrap();
    let arena = ExecutorArena::bind(session.arena_stats()[0], registration).unwrap();
    let initial = executor_plan.lower_prepared(initial, &[arena]).unwrap();
    let initial_evidence = initial.execution_evidence_after_success(&[arena]).unwrap();
    let initial_ticket = session.submit_execution(&initial_evidence).unwrap();
    let initial_publication = session
        .complete_execution_by_batch(
            initial_ticket.batch_id(),
            EngineCompletionEvidence {
                completion_domain: 1,
                completion_value: 1,
                confirmed: true,
            },
        )
        .unwrap();
    session
        .confirm_publication(&EnginePublicationEvidence {
            publication_id: initial_publication.publication_id,
            mirror_cleanup_confirmed: true,
            reclamation_receipts: Box::default(),
        })
        .unwrap();

    let source = session
        .prepare_append_batch(&[EngineAppendIntent {
            request_id,
            target_boundary: 18,
        }])
        .unwrap();
    let view = session.prepared_execution_view(source.batch_id).unwrap();
    let attention = executor_plan.attention_batch(0, &view).unwrap();
    let prepared = executor_plan.lower_prepared(source, &[arena]).unwrap();
    (session, executor_plan, [arena], prepared, attention)
}

#[test]
#[ignore = "requires a CUDA device and one-time FlashInfer JIT compilation"]
fn external_block_page_plan_executes_on_cuda() {
    let (mut session, executor_plan, arenas, prepared, batch) = prepared_full_batch();
    assert_eq!(&*batch.page_indices, &[2, 3]);
    assert_eq!(&*batch.last_page_len, &[2]);
    let mut graph = Graph::default();
    let q = graph
        .named_tensor("q", (1, 1, HEAD_DIM))
        .as_dtype(DType::Bf16);
    let k = graph
        .named_tensor("k", (PHYSICAL_PAGES, PAGE_TOKENS, 1, HEAD_DIM))
        .as_dtype(DType::Bf16);
    let v = graph
        .named_tensor("v", (PHYSICAL_PAGES, PAGE_TOKENS, 1, HEAD_DIM))
        .as_dtype(DType::Bf16);
    let metadata = PagedAttentionMetadata::new(&mut graph, 0, 1.into(), 2.into());
    let output = paged_attention(
        PagedAttentionInputs {
            q,
            k_cache: k,
            v_cache: v,
            query_tokens: 1.into(),
            context_pages: 2.into(),
        },
        metadata,
        &executor_plan.classes[0],
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
    let mut runtime = CudaRuntime::new().expect("CUDA runtime");
    upload_inputs(&mut runtime, &q, &k, &v, &metadata, &batch);
    runtime = graph.compile(runtime, CompileOptions::default().search_graph_limit(1));
    upload_inputs(&mut runtime, &q, &k, &v, &metadata, &batch);
    runtime.execute(&graph.dyn_map);

    let result = runtime.get_bf16(output);
    assert_eq!(result.len(), HEAD_DIM);
    for value in result {
        assert!((value.to_f32() - 9.5).abs() <= 0.125, "{value:?}");
    }

    let evidence = prepared.execution_evidence_after_success(&arenas).unwrap();
    let ticket = session.submit_execution(&evidence).unwrap();
    let publication = session
        .complete_execution_by_batch(
            ticket.batch_id(),
            EngineCompletionEvidence {
                completion_domain: 1,
                completion_value: 2,
                confirmed: true,
            },
        )
        .unwrap();
    session
        .confirm_publication(&EnginePublicationEvidence {
            publication_id: publication.publication_id,
            mirror_cleanup_confirmed: true,
            reclamation_receipts: Box::default(),
        })
        .unwrap();
    assert_eq!(session.stats().active_pages, 2);
}
