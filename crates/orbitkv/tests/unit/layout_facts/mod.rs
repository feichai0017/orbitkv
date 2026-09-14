use super::*;
use crate::{
    AttentionStatePlanInput, AttentionStateSpec, AttentionStateStorage, compile_runtime_manifest,
};

#[test]
fn facts_preserve_static_geometry() {
    let manifest = compile_runtime_manifest(AttentionStatePlanInput {
        page_tokens: 16,
        states: vec![
            AttentionStateSpec {
                name: "global".into(),
                layers: vec![0, 2],
                storage: AttentionStateStorage::TokenKv {
                    key_bytes_per_token_per_layer: 128,
                    value_bytes_per_token_per_layer: 128,
                    retention: RetentionKind::Full,
                    window_tokens: None,
                },
            },
            AttentionStateSpec {
                name: "local".into(),
                layers: vec![1, 3],
                storage: AttentionStateStorage::TokenKv {
                    key_bytes_per_token_per_layer: 128,
                    value_bytes_per_token_per_layer: 128,
                    retention: RetentionKind::Sliding,
                    window_tokens: Some(64),
                },
            },
            AttentionStateSpec {
                name: "recurrent".into(),
                layers: vec![4],
                storage: AttentionStateStorage::Recurrent {
                    family: RecurrentFamily::Gdn,
                    state_bytes_per_layer: 256,
                    checkpoint_slots_per_request: 2,
                },
            },
        ],
    })
    .unwrap();

    let facts = manifest.state_layout_facts().unwrap();
    assert_eq!(facts.manifest_fingerprint, manifest.fingerprint);
    assert_eq!(facts.page_tokens, 16);
    assert_eq!(facts.classes.len(), 3);
    assert_eq!(facts.classes[0].manager_class_id, Some(0));
    assert_eq!(facts.classes[1].manager_class_id, Some(1));
    assert_eq!(facts.classes[1].window_tokens, Some(64));
    assert_eq!(facts.classes[2].manager_class_id, None);
    assert!(matches!(
        facts.classes[2].storage,
        StateStorageFacts::RecurrentCheckpoints {
            family: RecurrentFamily::Gdn,
            ..
        }
    ));
}
