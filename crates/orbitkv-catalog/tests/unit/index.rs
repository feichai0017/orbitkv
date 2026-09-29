use super::*;
use orbitkv_state::CacheOwner;
use std::time::{Duration, Instant};

fn owner(port: u16) -> CacheOwner {
    CacheOwner {
        endpoint: format!("127.0.0.1:{port}"),
        incarnation: Uuid::new_v4(),
    }
}

fn setup(limit: usize) -> (GlobalIndex, Arc<MembershipView>, CacheOwner) {
    let local = owner(51001);
    let remote = owner(51002);
    let membership = Arc::new(MembershipView::new(local.clone()));
    assert!(membership.renew(Instant::now(), Duration::from_secs(60)));
    membership.replace_members([("local".into(), local), ("remote".into(), remote.clone())]);
    (
        GlobalIndex::new(membership.clone(), limit),
        membership,
        remote,
    )
}

fn record(sequence: u64, medium: ReplicaMedium, present: bool) -> InventoryRecord {
    InventoryRecord {
        key: StateKey::new("model".into(), vec![1]),
        sequence,
        present,
        metadata: Some(ReplicaMetadata {
            medium,
            representation: ReplicaRepresentation::Raw,
            stored_bytes: Some(1024),
        }),
    }
}

#[test]
fn snapshot_and_publisher_readiness_preserve_both_media_and_fence_restarts() {
    let (index, membership, remote) = setup(4096);
    let dram = record(1, ReplicaMedium::Dram, true);
    let ssd = record(2, ReplicaMedium::Ssd, true);
    let key = dram.key.clone();
    index
        .apply(
            10,
            vec![
                IndexUpdate::Residency {
                    owner: remote.incarnation,
                    record: dram.clone(),
                },
                IndexUpdate::Residency {
                    owner: remote.incarnation,
                    record: ssd,
                },
                IndexUpdate::Publisher {
                    owner: remote.incarnation,
                    ready: false,
                },
            ],
        )
        .unwrap();
    assert!(
        index.lookup(std::slice::from_ref(&key))[0]
            .replicas
            .is_empty()
    );
    index.finish_snapshot(10).unwrap();
    assert!(
        index.lookup(std::slice::from_ref(&key))[0]
            .replicas
            .is_empty()
    );
    index
        .apply(
            11,
            vec![IndexUpdate::Publisher {
                owner: remote.incarnation,
                ready: true,
            }],
        )
        .unwrap();
    let candidates = index.lookup(std::slice::from_ref(&key));
    assert_eq!(candidates[0].replicas.len(), 2);
    index
        .apply(
            12,
            vec![IndexUpdate::Residency {
                owner: remote.incarnation,
                record: record(3, ReplicaMedium::Dram, true),
            }],
        )
        .unwrap();
    assert_eq!(
        index.lookup(std::slice::from_ref(&key))[0].replicas.len(),
        2
    );
    index
        .apply(
            13,
            vec![IndexUpdate::Residency {
                owner: remote.incarnation,
                record: record(4, ReplicaMedium::Dram, false),
            }],
        )
        .unwrap();
    assert_eq!(
        index.lookup(std::slice::from_ref(&key))[0].replicas[0]
            .metadata
            .medium,
        ReplicaMedium::Ssd
    );
    let replacement = owner(51002);
    membership.replace_members([
        ("local".into(), membership.owner().clone()),
        ("remote".into(), replacement),
    ]);
    assert!(
        index.lookup(std::slice::from_ref(&key))[0]
            .replicas
            .is_empty()
    );
    index
        .apply(14, vec![IndexUpdate::RemoveOwner(remote.incarnation)])
        .unwrap();
    assert_eq!(index.bytes(), 0);
}

#[test]
fn over_budget_or_conflicting_updates_withdraw_the_entire_view() {
    for limit in [300, 4096] {
        let (index, _, remote) = setup(limit);
        index.finish_snapshot(1).unwrap();
        let mut conflicting = record(1, ReplicaMedium::Dram, true);
        conflicting.metadata.as_mut().unwrap().stored_bytes = Some(2048);
        let result = index.apply(
            2,
            vec![
                IndexUpdate::Publisher {
                    owner: remote.incarnation,
                    ready: true,
                },
                IndexUpdate::Residency {
                    owner: remote.incarnation,
                    record: record(1, ReplicaMedium::Dram, true),
                },
                IndexUpdate::Residency {
                    owner: remote.incarnation,
                    record: conflicting,
                },
            ],
        );
        assert!(result.is_err());
        assert_eq!(index.revision(), None);
        assert_eq!(index.bytes(), 0);
    }
}

#[test]
fn candidate_limits_preserve_ssd_alternatives_and_global_coverage() {
    let (index, membership, remote) = setup(1 << 20);
    let mut members = vec![("local".into(), membership.owner().clone())];
    let mut updates = Vec::new();
    for i in 0..6 {
        let owner = if i == 0 {
            remote.clone()
        } else {
            owner(51003 + i)
        };
        members.push((format!("remote-{i}"), owner.clone()));
        updates.push(IndexUpdate::Publisher {
            owner: owner.incarnation,
            ready: true,
        });
        for medium in [ReplicaMedium::Dram, ReplicaMedium::Ssd] {
            updates.push(IndexUpdate::Residency {
                owner: owner.incarnation,
                record: record(1, medium, true),
            });
        }
    }
    membership.replace_members(members);
    index.apply(1, updates).unwrap();
    index.finish_snapshot(1).unwrap();
    let key = record(1, ReplicaMedium::Dram, true).key;
    let candidates = index.lookup(std::slice::from_ref(&key));
    for medium in [ReplicaMedium::Dram, ReplicaMedium::Ssd] {
        assert_eq!(
            candidates[0]
                .replicas
                .iter()
                .filter(|r| r.metadata.medium == medium)
                .count(),
            4
        );
    }
    let first = candidates[0].replicas[0].owner.incarnation;
    index
        .apply(2, vec![IndexUpdate::RemoveOwner(first)])
        .unwrap();
    assert_eq!(
        index.lookup(&[key])[0].replicas.len(),
        8,
        "bounded output must not truncate stored coverage"
    );
}
