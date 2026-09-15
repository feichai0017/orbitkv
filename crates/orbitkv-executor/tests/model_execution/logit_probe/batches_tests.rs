use super::*;

fn probe(batches: Option<serde_json::Value>) -> Probe {
    serde_json::from_value(serde_json::json!({
        "schema": 1, "model_directory": "unused", "device_index": 0,
        "page_tokens": 16, "kv_dtype_bytes": 2, "page_counts": [16],
        "compile": {
            "maximum_query_tokens": 4, "representative_prefill_tokens": 4,
            "maximum_batch_size": 2, "maximum_context_pages": 16,
            "representative_context_pages": 2, "search_graphs": 1, "search_seed": 0
        },
        "cases": [
            {"id": "a", "prompt_token_ids": [1,2,3,4], "continuation_token_ids": [5,6]},
            {"id": "b", "prompt_token_ids": [7,8], "continuation_token_ids": [9,10]}
        ],
        "batches": batches.unwrap_or(serde_json::Value::Null)
    }))
    .unwrap()
}

#[test]
fn default_schedule_preserves_logical_teacher_forced_steps() {
    let plan = probe(None).plan_batches().unwrap();
    let geometry = plan
        .iter()
        .map(|b| {
            assert_eq!(b.len(), 1);
            let q = &b[0];
            (q.case, q.start, q.end, q.output_step, q.finished)
        })
        .collect::<Vec<_>>();
    assert_eq!(
        geometry,
        [
            (0, 0, 4, Some(0), false),
            (0, 4, 5, Some(1), true),
            (1, 0, 2, Some(0), false),
            (1, 2, 3, Some(1), true),
        ]
    );
}

#[test]
fn chunking_reordering_and_mixed_prefill_decode_keep_each_history() {
    let probe = probe(Some(serde_json::json!([
        [{"case":"a","tokens":2},{"case":"b","tokens":2}],
        [{"case":"b","tokens":1},{"case":"a","tokens":2}],
        [{"case":"a","tokens":1}]
    ])));
    let plan = probe.plan_batches().unwrap();
    assert_eq!(plan[0][0].output_step, None);
    assert_eq!((plan[1][0].case, plan[1][0].output_step), (1, Some(1)));
    assert!(plan[1][0].finished);
    assert_eq!((plan[1][1].start, plan[1][1].output_step), (2, Some(0)));
    assert_eq!(probe.cases[0].inputs(plan[2][0].start, plan[2][0].end), [5]);
    assert_eq!(probe.cases[1].inputs(plan[1][0].start, plan[1][0].end), [9]);
    assert!(plan[2].iter().all(|q| q.finished));
}

#[test]
fn finished_requests_release_capacity_while_peers_remain_active() {
    let mut probe = probe(Some(serde_json::json!([
        [{"case":"a","tokens":2},{"case":"b","tokens":2}],
        [{"case":"b","tokens":1}],
        [{"case":"c","tokens":1},{"case":"a","tokens":2}],
        [{"case":"a","tokens":1}]
    ])));
    probe.cases.push(Case {
        id: "c".into(),
        prompt_token_ids: vec![11],
        continuation_token_ids: vec![12],
    });
    let plan = probe.plan_batches().unwrap();
    assert!(plan[1][0].finished);
    assert_eq!(plan[2][0].case, 2);
    assert!(plan[2][0].finished);
    assert!(!plan[2][1].finished);
}

#[test]
fn default_schedule_chunks_prompts_to_the_submission_capacity() {
    let mut probe = probe(None);
    probe.compile.maximum_query_tokens = 3;
    let plan = probe.plan_batches().unwrap();
    assert_eq!((plan[0][0].start, plan[0][0].end), (0, 3));
    assert_eq!(plan[0][0].output_step, None);
    assert_eq!((plan[1][0].start, plan[1][0].end), (3, 4));
    assert_eq!(plan[1][0].output_step, Some(0));
}

#[test]
fn invalid_submission_plans_fail_before_execution() {
    for batches in [
        serde_json::json!([]),
        serde_json::json!([[]]),
        serde_json::json!([[{"case":"missing","tokens":1}]]),
        serde_json::json!([[{"case":"a","tokens":0}]]),
        serde_json::json!([[{"case":"a","tokens":1},{"case":"a","tokens":1}]]),
        serde_json::json!([[{"case":"a","tokens":4},{"case":"b","tokens":2}]]),
        serde_json::json!([[{"case":"a","tokens":4}],[{"case":"a","tokens":2}]]),
    ] {
        assert!(probe(Some(batches)).plan_batches().is_err());
    }
    let mut too_many_live = probe(Some(serde_json::json!([
        [{"case":"a","tokens":4}], [{"case":"b","tokens":2}],
        [{"case":"a","tokens":1}], [{"case":"b","tokens":1}]
    ])));
    too_many_live.compile.maximum_batch_size = 1;
    assert!(too_many_live.plan_batches().is_err());
}
