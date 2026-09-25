use super::*;

#[cfg(feature = "mooncake")]
#[tokio::test]
async fn mooncake_initialization_failure_is_returned_to_caller() {
    let config = crate::EngineConfig {
        mooncake_nic_names: vec!["definitely-not-a-real-nic".to_string()],
        membership: Some(test_membership()),
        ..crate::EngineConfig::default()
    };

    let err = match OrbitKVEngine::new_with_config(1 << 20, false, config) {
        Ok(_) => panic!("engine startup must fail when Mooncake NIC init fails"),
        Err(err) => err.to_string(),
    };

    assert!(
        err.contains("Failed to initialise Mooncake Transfer Engine"),
        "{err}"
    );
    assert!(err.contains("definitely-not-a-real-nic"), "{err}");
}

#[cfg(not(feature = "mooncake"))]
#[tokio::test]
async fn remote_transfer_config_is_ignored_without_feature() {
    let config = crate::EngineConfig {
        mooncake_nic_names: vec!["mlx5_0".to_string()],
        membership: Some(test_membership()),
        ..crate::EngineConfig::default()
    };

    let engine = OrbitKVEngine::new_with_config(1 << 20, false, config)
        .expect("a build without Mooncake should ignore remote transfer config");

    assert!(!engine.has_remote_transport());
}

fn test_membership() -> std::sync::Arc<orbitkv_catalog::MembershipView> {
    std::sync::Arc::new(orbitkv_catalog::MembershipView::new(
        orbitkv_state::CacheOwner {
            endpoint: "127.0.0.1:50055".into(),
            incarnation: uuid::Uuid::new_v4(),
        },
        orbitkv_catalog::Placement::new(vec!["a".into()]).unwrap(),
    ))
}
