use super::*;

#[cfg(feature = "mooncake")]
#[tokio::test]
async fn mooncake_initialization_failure_is_returned_to_caller() {
    let config = storage::StorageConfig {
        mooncake_nic_names: vec!["definitely-not-a-real-nic".to_string()],
        metaserver_addr: Some("http://127.0.0.1:50056".to_string()),
        advertise_addr: Some("127.0.0.1:50055".to_string()),
        ..storage::StorageConfig::default()
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
    let config = storage::StorageConfig {
        mooncake_nic_names: vec!["mlx5_0".to_string()],
        metaserver_addr: Some("http://127.0.0.1:50056".to_string()),
        ..storage::StorageConfig::default()
    };

    let engine = OrbitKVEngine::new_with_config(1 << 20, false, config)
        .expect("a build without Mooncake should ignore remote transfer config");

    assert!(!engine.has_remote_transport());
}
