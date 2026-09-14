use super::*;
use crate::plan::TokenComponentSpec;

fn latent_full_plan() -> CompiledKvPlan {
    compile_plan(KvPlanInput {
        page_tokens: PAGE_TOKENS,
        classes: vec![KvClassSpec {
            name: "latent".into(),
            layers: vec![0],
            retention: RetentionKind::Full,
            bytes_per_token_per_layer: 1_152,
            window_tokens: None,
            storage: TokenStorageKind::LatentKv,
            components: vec![
                TokenComponentSpec {
                    name: "latent".into(),
                    bytes_per_token_per_layer: 1_024,
                },
                TokenComponentSpec {
                    name: "rope".into(),
                    bytes_per_token_per_layer: 128,
                },
            ],
        }],
    })
    .expect("latent Full plan")
}

#[test]
fn full_request_private_rejects_every_prefix_share_entrypoint() {
    let backends = [backend(0, 69, 4, 19_000)];
    let mut session = RuntimeSession::new(
        CanonicalKvManager::new(
            &latent_full_plan(),
            ManagerConfig {
                maximum_requests: 2,
                maximum_operations: 8,
                maximum_prefixes: 2,
                maximum_reclamations: 4,
                maximum_step_tokens: 64,
            },
            &backends,
        )
        .expect("latent manager"),
        CacheSharingPolicy::RequestPrivate,
    );
    let source = EngineRequestId(901);
    let target = EngineRequestId(902);
    session
        .acquire_requests(&[source, target])
        .expect("acquire private requests");
    let key = prefix_key(0xE1, 16);
    let session_epoch = session.arena_stats()[0].engine_epoch;
    let prefix_id = EnginePrefixId::from_parts(session_epoch, 1);
    let control_id = EngineControlId::from_parts(session_epoch, 1);
    let hint = EnginePrefixLookup {
        key,
        candidate: Some(prefix_id),
        resident_count: 1,
    };
    let pending_attach = EnginePendingAttachCancel {
        control_id,
        request_id: target,
        prefix_id,
        view_version: ViewVersion(1),
        boundary: 16,
        resident_count: 1,
    };
    let before = session.stats();

    assert_eq!(
        session.lookup_prefix_batch(&[key]),
        Err(RuntimeSessionError::PrefixOperationsUnsupported)
    );
    assert_eq!(
        session.publish_prefix_batch(&[(source, key)]),
        Err(RuntimeSessionError::PrefixOperationsUnsupported)
    );
    assert_eq!(
        session.prepare_prefix_attach(&[(target, hint)]),
        Err(RuntimeSessionError::PrefixOperationsUnsupported)
    );
    assert_eq!(
        session.prepare_request_fork(&[(source, target)]),
        Err(RuntimeSessionError::PrefixOperationsUnsupported)
    );
    assert_eq!(
        session.prepare_prefix_evict(&[prefix_id]),
        Err(RuntimeSessionError::PrefixOperationsUnsupported)
    );
    assert_eq!(
        session.publish_prefix_and_release_batch(&[(source, key)]),
        Err(RuntimeSessionError::PrefixOperationsUnsupported)
    );
    assert_eq!(
        session.commit_control(control_id),
        Err(RuntimeSessionError::PrefixOperationsUnsupported)
    );
    assert_eq!(
        session.abort_control(control_id),
        Err(RuntimeSessionError::PrefixOperationsUnsupported)
    );
    assert_eq!(
        session.confirm_control(&EngineControlEvidence {
            control_id,
            mirror_updates_confirmed: true,
            reclamation_receipts: Box::new([]),
        }),
        Err(RuntimeSessionError::PrefixOperationsUnsupported)
    );
    assert_eq!(
        session.quarantine_control(control_id),
        Err(RuntimeSessionError::PrefixOperationsUnsupported)
    );
    assert_eq!(
        session.cancel_pending_attach(pending_attach),
        Err(RuntimeSessionError::PrefixOperationsUnsupported)
    );
    assert_eq!(
        session.finalize_pending_attach_cancel(pending_attach),
        Err(RuntimeSessionError::PrefixOperationsUnsupported)
    );
    assert_eq!(session.stats(), before);
}
