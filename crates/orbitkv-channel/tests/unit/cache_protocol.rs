use super::*;

#[test]
fn cancel_query_preserves_scope_and_rejects_malformed_frames() {
    let request = CancelQueryRequest {
        ticket: QueryTicket {
            operation_id: 7,
            revision: 2,
        },
    };
    let mut bytes = request.encode().unwrap();
    assert_eq!(CancelQueryRequest::decode(&bytes).unwrap(), request);
    for end in 0..bytes.len() {
        assert!(CancelQueryRequest::decode(&bytes[..end]).is_err());
    }
    bytes.push(0);
    assert!(CancelQueryRequest::decode(&bytes).is_err());
}

#[test]
fn request_round_trip_preserves_variable_hashes() {
    let request = QueryBundleRequest {
        ticket: QueryTicket {
            operation_id: 7,
            revision: 2,
        },
        instance_id: "model-a".to_string(),
        request_id: "request-7".to_string(),
        block_hashes: vec![vec![1; 32], vec![2; 17]],
        group_id: 3,
        wait_for_full_prefix: true,
        warmup: false,
        discover: false,
        materialize: false,
    };
    assert_eq!(
        QueryBundleRequest::decode(&request.encode().unwrap()).unwrap(),
        request
    );
    let mut previous_schema = request.encode().unwrap();
    previous_schema[4..6].copy_from_slice(&(QUERY_VERSION - 1).to_le_bytes());
    assert_eq!(
        QueryBundleRequest::decode(&previous_schema),
        Err(QueryCodecError::UnsupportedVersion(QUERY_VERSION - 1))
    );
    for command in [
        QueryCommand::Submit(request.clone()),
        QueryCommand::Poll(request.ticket),
    ] {
        let bytes = command.encode().unwrap();
        assert_eq!(QueryCommand::decode(&bytes).unwrap(), command);
        for end in 0..bytes.len() {
            assert!(QueryCommand::decode(&bytes[..end]).is_err());
        }
    }
    for ticket in [
        QueryTicket {
            operation_id: 0,
            revision: 1,
        },
        QueryTicket {
            operation_id: 1,
            revision: 0,
        },
    ] {
        assert_eq!(
            QueryCommand::Poll(ticket).encode(),
            Err(QueryCodecError::InvalidQueryTicket)
        );
    }
}

#[test]
fn response_round_trip_preserves_lease_and_positions() {
    let response = QueryBundleResponse {
        outcome: QueryOutcomeCode::Ready,
        num_hit_blocks: 2,
        lease: vec![3; 16],
        hit_positions: vec![1, 4],
    };
    assert_eq!(
        QueryBundleResponse::decode(&response.encode().unwrap()).unwrap(),
        response
    );
}

#[test]
fn decoder_rejects_trailing_bytes_and_invalid_loading_payload() {
    let mut encoded = QueryBundleResponse::loading().encode().unwrap();
    encoded.push(0);
    assert_eq!(
        QueryBundleResponse::decode(&encoded),
        Err(QueryCodecError::TrailingBytes(1))
    );

    let invalid = QueryBundleResponse {
        outcome: QueryOutcomeCode::Loading,
        num_hit_blocks: 1,
        lease: Vec::new(),
        hit_positions: Vec::new(),
    };
    assert_eq!(
        QueryBundleResponse::decode(&invalid.encode().unwrap()),
        Err(QueryCodecError::InvalidLoadingPayload)
    );
}

#[test]
fn release_request_round_trip_and_empty_rejection() {
    let request = ReleaseRequest { lease: vec![7; 16] };
    assert_eq!(
        ReleaseRequest::decode(&request.encode().unwrap()).unwrap(),
        request
    );
    assert_eq!(
        ReleaseRequest::decode(&ReleaseRequest { lease: Vec::new() }.encode().unwrap()),
        Err(QueryCodecError::EmptyLease)
    );
}

#[test]
fn publish_request_round_trip_and_shape_validation() {
    let request = PublishRequest {
        instance_id: "model-a".to_string(),
        tp_rank: 2,
        pp_rank: 1,
        device_id: 3,
        layers: vec![PublishLayer {
            layer_name: "layer.0".to_string(),
            block_ids: vec![4, 7],
            block_hashes: vec![vec![1; 32], vec![2; 32]],
        }],
    };
    assert_eq!(
        PublishRequest::decode(&request.encode().unwrap()).unwrap(),
        request
    );

    let invalid = PublishRequest {
        layers: vec![PublishLayer {
            layer_name: "bad".to_string(),
            block_ids: vec![1],
            block_hashes: Vec::new(),
        }],
        ..request
    };
    assert!(matches!(
        invalid.encode(),
        Err(QueryCodecError::PublishShapeMismatch { .. })
    ));
}

#[test]
fn restore_request_and_response_round_trip() {
    let request = RestoreRequest {
        instance_id: "model-a".to_string(),
        tp_rank: 1,
        device_id: 2,
        layer_groups: vec![vec!["layer.0".to_string()], Vec::new()],
        loads: vec![RestoreLease {
            lease: vec![9; 16],
            block_ids_by_group: vec![vec![Some(3), None], vec![None, Some(7)]],
        }],
    };
    assert_eq!(
        RestoreRequest::decode(&request.encode().unwrap()).unwrap(),
        request
    );

    let response = RestoreResponse {
        operation_id: 42,
        state: RestoreState::Failed,
        message: "cuda copy failed".to_string(),
    };
    assert_eq!(
        RestoreResponse::decode(&response.encode().unwrap()).unwrap(),
        response
    );

    let poll = RestoreCommand::Poll { operation_id: 42 };
    assert_eq!(
        RestoreCommand::decode(&poll.encode().unwrap()).unwrap(),
        poll
    );
}

#[test]
fn candidate_hints_cannot_carry_leases_or_ambiguous_positions() {
    for (lease, positions, count) in [
        (vec![1], vec![1, 2], 2),
        (vec![], vec![2, 1], 2),
        (vec![], vec![1, 1], 2),
        (vec![], vec![1, 2], 1),
    ] {
        let response = QueryBundleResponse {
            outcome: QueryOutcomeCode::Candidates,
            num_hit_blocks: count,
            lease,
            hit_positions: positions,
        };
        assert_eq!(
            QueryBundleResponse::decode(&response.encode().unwrap()),
            Err(QueryCodecError::InvalidCandidates)
        );
    }
}
