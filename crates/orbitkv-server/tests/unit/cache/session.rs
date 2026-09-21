use super::*;

#[test]
fn install_replaces_topology_and_token() {
    let registry = SessionRegistry::default();
    let first = registry.install("inst".to_string(), "ns1".to_string(), 8, 8);
    let second = registry.install("inst".to_string(), "ns2".to_string(), 4, 4);

    assert_ne!(first, second);
    assert_eq!(
        registry.topology("inst"),
        Some(SessionTopology {
            namespace: "ns2".to_string(),
            tp_size: 4,
            world_size: 4,
        })
    );
    assert!(!registry.take("inst", first));
    assert!(registry.take("inst", second));
    assert_eq!(registry.topology("inst"), None);
}
