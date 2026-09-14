use super::*;

#[test]
fn rejects_duplicate_request_ids() {
    let request = RequestIntent {
        request_id: RequestId(7),
        input_tokens: vec![1].into_boxed_slice(),
        target_boundary: 1,
        sampling: SamplingIntent::greedy(1, []),
    };
    assert_eq!(
        BatchIntent::new(vec![request.clone(), request].into_boxed_slice()),
        Err(BatchIntentError::DuplicateRequest)
    );
}

#[test]
fn accepts_prefill_and_decode_without_physical_state() {
    let batch = BatchIntent::new(
        vec![
            RequestIntent {
                request_id: RequestId(1),
                input_tokens: vec![10, 11, 12].into_boxed_slice(),
                target_boundary: 3,
                sampling: SamplingIntent::greedy(8, [2]),
            },
            RequestIntent {
                request_id: RequestId(2),
                input_tokens: vec![20].into_boxed_slice(),
                target_boundary: 41,
                sampling: SamplingIntent::greedy(1, []),
            },
        ]
        .into_boxed_slice(),
    )
    .unwrap();
    assert_eq!(batch.requests.len(), 2);
}

#[test]
fn rejects_zero_output_budget() {
    assert_eq!(
        BatchIntent::new(
            vec![RequestIntent {
                request_id: RequestId(1),
                input_tokens: vec![10].into_boxed_slice(),
                target_boundary: 1,
                sampling: SamplingIntent::greedy(0, []),
            }]
            .into_boxed_slice(),
        ),
        Err(BatchIntentError::InvalidSampling)
    );
}
