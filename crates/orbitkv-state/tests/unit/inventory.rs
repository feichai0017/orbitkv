use super::*;

#[test]
fn exact_scope_is_sorted_deduplicated_and_distinct_from_all_or_empty() {
    let first = InventoryScope::exact(["b".into(), "a".into(), "b".into()]).unwrap();
    let second = InventoryScope::exact(["a".into(), "b".into()]).unwrap();
    let empty = InventoryScope::exact(Vec::new()).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.namespaces().unwrap(), ["a", "b"]);
    assert!(first.contains("a"));
    assert!(!first.contains("c"));
    assert_ne!(first.digest(), InventoryScope::AllNamespaces.digest());
    assert_ne!(empty.digest(), InventoryScope::AllNamespaces.digest());
    assert_eq!(empty.label(), "no_namespaces");
}

#[test]
fn exact_scope_rejects_empty_oversized_and_overcount_descriptors() {
    assert!(InventoryScope::exact([String::new()]).is_err());
    assert!(
        InventoryScope::exact(
            (0..=INVENTORY_SCOPE_MAX_NAMESPACES).map(|index| format!("namespace-{index}"))
        )
        .is_err()
    );
    assert!(InventoryScope::exact(["x".repeat(INVENTORY_OPEN_MAX_BYTES)]).is_err());
}
