use orbitkv::{
    EngineRequestId, ExternalObjectKey, ExternalReplicaTarget, runtime_session::ExternalTransferId,
};

use super::*;
use crate::AttentionVisibility;

#[test]
fn expands_logical_pages_into_layer_component_iovecs() {
    let plan = crate::tests::support::executor_plan(
        "test",
        16,
        vec![AttentionClass {
            class_id: 0,
            name: "attention".into(),
            layers: vec![2, 5].into_boxed_slice(),
            page_tokens: 16,
            key_bytes_per_token_per_layer: 8,
            value_bytes_per_token_per_layer: 12,
            visibility: AttentionVisibility::Full,
        }],
    );
    let transfer_id = ExternalTransferId::from_parts(1, 1);
    let target = ExternalReplicaTarget {
        storage_domain: 7,
        object_index: 9,
        base_offset: 1_000,
    };
    let source = ExternalExportPlan {
        transfer_id,
        request_id: EngineRequestId(3),
        key: ExternalObjectKey {
            namespace: [1; 32],
            digest: [2; 32],
            plan_fingerprint: [3; 32],
            boundary: 16,
        },
        target,
        total_bytes: 640,
        copies: vec![ExternalExportCopy {
            transfer_id,
            copy_index: 0,
            class_id: 0,
            source_backend_domain: 3,
            source_backend_index: 12,
            destination_storage_domain: 7,
            destination_object_index: 9,
            destination_offset: 1_000,
            byte_count: 640,
            logical_ordinal: 0,
            valid_token_count: 16,
            visible_token_offset: 0,
            visible_token_count: 16,
        }]
        .into_boxed_slice(),
    };
    let lowered = plan
        .lower_external_export(
            &source,
            &[ExecutorArena {
                engine_epoch: 1,
                pool_epoch: 2,
                pool_id: 4,
                class_id: 0,
                backend_domain: 3,
                first_page_id: 1,
                page_count: 8,
                backend_base_index: 10,
            }],
        )
        .unwrap();
    assert_eq!(lowered.total_bytes, 640);
    assert_eq!(lowered.spans.len(), 4);
    assert_eq!(lowered.spans[0].source_tensor_offset, 1_536);
    assert_eq!(lowered.spans[0].destination_offset, 1_000);
    assert_eq!(lowered.spans[1].source_tensor_offset, 2_304);
    assert_eq!(lowered.spans[1].destination_offset, 1_128);
    assert_eq!(lowered.spans[2].layer, 5);
    assert_eq!(lowered.spans[3].destination_offset, 1_448);
}

#[test]
fn expands_restore_into_external_to_tensor_iovecs() {
    let plan = crate::tests::support::executor_plan(
        "test",
        16,
        vec![AttentionClass {
            class_id: 0,
            name: "attention".into(),
            layers: vec![2, 5].into_boxed_slice(),
            page_tokens: 16,
            key_bytes_per_token_per_layer: 8,
            value_bytes_per_token_per_layer: 12,
            visibility: AttentionVisibility::Full,
        }],
    );
    let transfer_id = ExternalTransferId::from_parts(1, 2);
    let source = ExternalRestorePlan {
        transfer_id,
        request_id: EngineRequestId(4),
        key: ExternalObjectKey {
            namespace: [1; 32],
            digest: [2; 32],
            plan_fingerprint: [3; 32],
            boundary: 16,
        },
        total_bytes: 640,
        copies: vec![ExternalRestoreCopy {
            transfer_id,
            copy_index: 0,
            class_id: 0,
            source_storage_domain: 7,
            source_object_index: 9,
            source_offset: 1_000,
            destination_backend_domain: 3,
            destination_backend_index: 12,
            byte_count: 640,
            logical_ordinal: 0,
            valid_token_count: 16,
            visible_token_offset: 0,
            visible_token_count: 16,
            expected_checksum: [4; 32],
        }]
        .into_boxed_slice(),
    };
    let lowered = plan
        .lower_external_restore(
            &source,
            &[ExecutorArena {
                engine_epoch: 1,
                pool_epoch: 2,
                pool_id: 4,
                class_id: 0,
                backend_domain: 3,
                first_page_id: 1,
                page_count: 8,
                backend_base_index: 10,
            }],
        )
        .unwrap();
    assert_eq!(lowered.spans.len(), 4);
    assert_eq!(lowered.spans[0].source_offset, 1_000);
    assert_eq!(lowered.spans[0].destination_tensor_offset, 1_536);
    assert_eq!(lowered.spans[1].source_offset, 1_128);
    assert_eq!(lowered.spans[1].destination_tensor_offset, 2_304);
    assert_eq!(lowered.spans[3].source_offset, 1_448);
}
