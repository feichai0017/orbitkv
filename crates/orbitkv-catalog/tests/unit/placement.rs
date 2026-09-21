use super::*;
use crate::MembershipView;
use orbitkv_state::{CacheOwner, StateKey, catalog_shard};
use std::time::{Duration, Instant};
use uuid::Uuid;

#[test]
fn canonical_placement_is_stable_across_member_loss_and_restart() {
    let placement = Placement::new(vec!["b".into(), "a".into()]).unwrap();
    assert_eq!(
        placement,
        Placement::new(vec!["a".into(), "b".into()]).unwrap()
    );
    assert_ne!(
        placement.id(),
        Placement::new(vec!["a".into()]).unwrap().id()
    );
    let a = CacheOwner {
        endpoint: "127.0.0.1:50055".into(),
        incarnation: Uuid::new_v4(),
    };
    let b = CacheOwner {
        endpoint: "127.0.0.1:50056".into(),
        incarnation: Uuid::new_v4(),
    };
    let view = MembershipView::new(a.clone(), placement.clone());
    assert!(view.renew(Instant::now(), Duration::from_secs(60)));
    view.replace_members([("a".into(), a.clone()), ("b".into(), b.clone())]);
    let shard = (0..CATALOG_SHARDS)
        .find(|&shard| placement.host(shard) == Some("b"))
        .unwrap();
    assert_eq!(view.catalog_owner(shard), Some(b.clone()));
    view.replace_members([("a".into(), a.clone())]);
    assert_eq!(view.catalog_owner(shard), None);
    let replacement = CacheOwner {
        incarnation: Uuid::new_v4(),
        ..b
    };
    view.replace_members([("a".into(), a), ("b".into(), replacement.clone())]);
    assert_eq!(view.catalog_owner(shard), Some(replacement));
    assert_eq!(view.catalog_owner(CATALOG_SHARDS), None);

    for nodes in [
        vec![],
        vec!["a".into(), "a".into()],
        vec!["../a".into()],
        (0..17).map(|n| n.to_string()).collect(),
    ] {
        assert!(Placement::new(nodes).is_err());
    }
    // Length framing separates namespace bytes from the content hash.
    assert_ne!(
        catalog_shard(&StateKey::new("ab".into(), b"c".to_vec())),
        catalog_shard(&StateKey::new("a".into(), b"bc".to_vec()))
    );
}
