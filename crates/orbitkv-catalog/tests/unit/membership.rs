use super::*;

fn owner(endpoint: &str) -> CacheOwner {
    CacheOwner {
        endpoint: endpoint.into(),
        incarnation: uuid::Uuid::new_v4(),
    }
}

#[test]
fn registration_and_complete_snapshot_are_both_required() {
    let local = owner("127.0.0.1:50055");
    let peer = owner("127.0.0.1:50056");
    let view = MembershipView::new(local.clone(), Placement::new(vec!["local".into()]).unwrap());
    view.replace_members([
        ("local".into(), local.clone()),
        ("peer".into(), peer.clone()),
    ]);
    assert!(!view.permits(&peer));
    assert!(view.renew(Instant::now(), Duration::from_secs(30)));
    assert!(view.permits(&peer));
    assert!(!view.permits(&owner(&peer.endpoint)));
    view.invalidate_snapshot();
    assert!(!view.permits(&local));
    view.replace_members([("local".into(), local.clone())]);
    assert!(view.permits(&local));
    assert!(!view.permits(&peer));
}

#[test]
fn expired_or_fenced_runtime_cannot_be_revived_by_delayed_acknowledgements() {
    for cause in ["deadline", "explicit", "replacement"] {
        let local = owner("127.0.0.1:50055");
        let view =
            MembershipView::new(local.clone(), Placement::new(vec!["local".into()]).unwrap());
        view.replace_members([("local".into(), local.clone())]);
        assert!(view.renew(Instant::now(), Duration::from_secs(30)));
        match cause {
            "deadline" => {
                view.state.write().valid_until = Some(Instant::now() - Duration::from_secs(1));
            }
            "explicit" => view.fence(),
            _ => view.replace_members([("local".into(), owner(&local.endpoint))]),
        }
        assert!(!view.permits(&local));
        assert!(!view.renew(Instant::now(), Duration::from_secs(30)));
        view.replace_members([("local".into(), local.clone())]);
        assert!(!view.permits(&local));
    }
    let view = MembershipView::new(
        owner("127.0.0.1:50055"),
        Placement::new(vec!["local".into()]).unwrap(),
    );
    assert!(!view.renew(
        Instant::now() - Duration::from_secs(20),
        Duration::from_secs(30)
    ));
}
