use super::*;
use crate::cluster::tests::etcd::{Etcd, join, view, wait_for};
use crate::cluster::tests::gate::TcpGate;
use orbitkv_state::{ReplicaMetadata, ReplicaRepresentation, StateKey};
use std::time::Instant;

fn record(sequence: u64, medium: ReplicaMedium, present: bool) -> InventoryRecord {
    InventoryRecord {
        key: StateKey::new("publication-test".into(), vec![1; 32]),
        sequence,
        present,
        metadata: Some(ReplicaMetadata {
            medium,
            representation: ReplicaRepresentation::Raw,
            stored_bytes: Some(4096),
        }),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires ETCD_BIN; real publication, Watch and fencing fault gate"]
async fn publication_reconciles_lost_replies_without_resurrecting_retired_residencies() {
    let server = Etcd::start(1).await;
    let requester = view(53002);
    let (requester_cluster, index, _) =
        join(&server, "publish", "requester", requester.clone(), 60).await;
    let source = view(53001);
    let mut client = Client::connect(&server.endpoints, None).await.unwrap();
    let granted_at = Instant::now();
    let grant = client.lease_grant(60, None).await.unwrap();
    assert!(source.renew(granted_at, Duration::from_secs(60)));
    let cluster = cluster_id(grant.header()).unwrap();
    let prefix = "/orbitkv/v2/publish/";
    let member = crate::cluster::registration::register(
        &mut client,
        prefix,
        "source",
        source.owner(),
        grant.id(),
        cluster,
    )
    .await
    .unwrap();
    let mut publisher = Publisher {
        client: client.clone(),
        prefix: prefix.into(),
        member: member.clone(),
        view: source.clone(),
        cluster,
        progress: Progress::default(),
        previous: None,
    };
    wait_for(|| requester.permits(source.owner())).await;
    let dram = record(1, ReplicaMedium::Dram, true);
    let ssd = record(2, ReplicaMedium::Ssd, true);
    let key = dram.key.clone();
    let revision = publisher
        .records(&[dram.clone(), ssd.clone()], 2, false)
        .await
        .unwrap();
    wait_for(|| index.revision().is_some_and(|r| r >= revision)).await;
    assert!(
        index.lookup(std::slice::from_ref(&key))[0]
            .replicas
            .is_empty(),
        "incomplete publisher was exposed"
    );
    let revision = publisher.commit(Vec::new(), 2, true).await.unwrap();
    wait_for(|| index.revision().is_some_and(|r| r >= revision)).await;
    assert_eq!(
        index.lookup(std::slice::from_ref(&key))[0].replicas.len(),
        2
    );

    // The server committed this put, but the publisher never received its reply.
    let next = record(3, ReplicaMedium::Dram, true);
    let marker = format!("{prefix}publishers/{}", source.owner().incarnation);
    let marker_value = serde_json::to_vec(&Progress {
        operation: publisher.progress.operation + 1,
        sequence: 3,
        ready: true,
    })
    .unwrap();
    let delayed_transaction = Txn::new()
        .when([
            Compare::value(
                marker.clone(),
                CompareOp::Equal,
                publisher.previous.clone().unwrap(),
            ),
            Compare::value(
                format!("{prefix}members/source"),
                CompareOp::Equal,
                serde_json::to_vec(&member).unwrap(),
            ),
        ])
        .and_then([
            TxnOp::put(
                record_key(prefix, source.owner().incarnation, &next).unwrap(),
                orbitkv_proto::proto::engine::InventoryRecord::from(next.clone()).encode_to_vec(),
                Some(PutOptions::new().with_lease(grant.id())),
            ),
            TxnOp::put(
                marker,
                marker_value,
                Some(PutOptions::new().with_lease(grant.id())),
            ),
        ]);
    let committed = client.txn(delayed_transaction.clone()).await.unwrap();
    assert!(committed.succeeded());
    let committed_revision = committed.header().unwrap().revision();
    assert_eq!(
        publisher.records(&[next], 3, true).await.unwrap(),
        committed_revision
    );
    let revision = publisher
        .records(&[record(4, ReplicaMedium::Dram, false)], 4, true)
        .await
        .unwrap();
    wait_for(|| index.revision().is_some_and(|r| r >= revision)).await;
    let candidates = index.lookup(std::slice::from_ref(&key));
    assert_eq!(candidates[0].replicas.len(), 1);
    assert_eq!(
        candidates[0].replicas[0].metadata.medium,
        ReplicaMedium::Ssd
    );
    assert!(
        !client.txn(delayed_transaction).await.unwrap().succeeded(),
        "late retry resurrected retired DRAM"
    );

    let revision = publisher
        .records(&[record(5, ReplicaMedium::Dram, true)], 5, true)
        .await
        .unwrap();
    wait_for(|| index.revision().is_some_and(|r| r >= revision)).await;
    assert_eq!(
        index.lookup(std::slice::from_ref(&key))[0].replicas[0].sequence,
        5
    );
    client.lease_revoke(grant.id()).await.unwrap();
    wait_for(|| !requester.permits(source.owner())).await;
    assert!(
        index.lookup(std::slice::from_ref(&key))[0]
            .replicas
            .is_empty()
    );
    assert!(
        publisher
            .records(&[record(6, ReplicaMedium::Dram, true)], 6, true)
            .await
            .is_err()
    );
    assert!(!source.registration_valid());
    requester_cluster.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires ETCD_BIN; paginated metadata above the default gRPC decode limit"]
async fn large_complete_snapshot_and_owner_reconciliation_remain_bounded() {
    let server = Etcd::start(1).await;
    let source = view(54001);
    let mut client = Client::connect(&server.endpoints, None).await.unwrap();
    let sent_at = Instant::now();
    let grant = client.lease_grant(60, None).await.unwrap();
    assert!(source.renew(sent_at, Duration::from_secs(60)));
    let cluster = cluster_id(grant.header()).unwrap();
    let prefix = "/orbitkv/v2/large-snapshot/";
    crate::cluster::watch::install_format(&mut client, prefix, cluster)
        .await
        .unwrap();
    let member = crate::cluster::registration::register(
        &mut client,
        prefix,
        "source",
        source.owner(),
        grant.id(),
        cluster,
    )
    .await
    .unwrap();
    let mut publisher = Publisher {
        client,
        prefix: prefix.into(),
        member,
        view: source.clone(),
        cluster,
        progress: Progress::default(),
        previous: None,
    };
    let records: Vec<_> = (1u64..=140)
        .map(|sequence| {
            let mut value = record(sequence, ReplicaMedium::Dram, true);
            value.key = StateKey::new("n".repeat(34 * 1024), sequence.to_le_bytes().to_vec());
            value
        })
        .collect();
    for chunk in records.chunks(8) {
        publisher
            .records(chunk, chunk.last().unwrap().sequence, false)
            .await
            .unwrap();
    }
    publisher.commit(Vec::new(), 140, true).await.unwrap();
    let reader = view(54002);
    let index = Arc::new(orbitkv_catalog::GlobalIndex::new(reader.clone(), 8 << 20));
    let inventory = Arc::new(ResidencyInventory::new(4096));
    let reader_cluster = crate::cluster::Cluster::join(
        &server.endpoints,
        "large-snapshot",
        "reader",
        60,
        reader,
        inventory,
        index.clone(),
    )
    .await
    .unwrap();
    wait_for(|| index.status().available).await;
    assert!(index.bytes() > 4 << 20);
    let keys: Vec<_> = records.iter().map(|r| r.key.clone()).collect();
    assert!(
        index
            .lookup(&keys)
            .iter()
            .all(|row| row.replicas.len() == 1)
    );
    // Reconciliation must remove advertisements no longer in the real inventory.
    let revision = publisher
        .rebuild(&ResidencyInventory::new(4096))
        .await
        .unwrap();
    wait_for(|| index.revision().is_some_and(|r| r >= revision)).await;
    assert!(
        index
            .lookup(&keys)
            .iter()
            .all(|row| row.replicas.is_empty())
    );
    reader_cluster.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires ETCD_BIN; real quota exhaustion, cursor retention and recovery"]
async fn quota_exhaustion_keeps_cursor_and_recovers_without_resurrecting_a_deleted_replica() {
    use etcd_client::{AlarmAction, AlarmOptions, AlarmType, CompactionOptions};

    let server = Etcd::start_with_args(1, &["--quota-backend-bytes", "2097152"]).await;
    let requester = view(55002);
    let (requester_cluster, index, _) =
        join(&server, "quota", "requester", requester.clone(), 120).await;
    let source = view(55001);
    let mut client = Client::connect(&server.endpoints, None).await.unwrap();
    let sent_at = Instant::now();
    let grant = client.lease_grant(120, None).await.unwrap();
    assert!(source.renew(sent_at, Duration::from_secs(120)));
    let cluster = cluster_id(grant.header()).unwrap();
    let prefix = "/orbitkv/v2/quota/";
    let member = crate::cluster::registration::register(
        &mut client,
        prefix,
        "source",
        source.owner(),
        grant.id(),
        cluster,
    )
    .await
    .unwrap();
    let mut publisher = Publisher {
        client: client.clone(),
        prefix: prefix.into(),
        member,
        view: source.clone(),
        cluster,
        progress: Progress::default(),
        previous: None,
    };
    let key = record(1, ReplicaMedium::Dram, true).key;
    let revision = publisher
        .records(&[record(1, ReplicaMedium::Dram, true)], 1, true)
        .await
        .unwrap();
    wait_for(|| index.revision().is_some_and(|r| r >= revision)).await;
    assert_eq!(
        index.lookup(std::slice::from_ref(&key))[0].replicas.len(),
        1
    );
    let old_marker = publisher.previous.clone().unwrap();
    let marker = format!("{prefix}publishers/{}", source.owner().incarnation);

    let mut rejected = None;
    for i in 0..64 {
        if let Err(error) = client
            .put(format!("quota-fill/{i}"), vec![b'x'; 128 * 1024], None)
            .await
        {
            rejected = Some(error.to_string());
            break;
        }
        // Quota uses committed backend size; allow etcd's batched commit to catch up.
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let rejection = rejected.expect("2 MiB backend quota was not exhausted by bounded writes");
    assert!(
        rejection.contains("space"),
        "unexpected write error: {rejection}"
    );
    let alarms = client
        .alarm(AlarmAction::Get, AlarmType::None, None)
        .await
        .unwrap();
    assert!(
        alarms
            .alarms()
            .iter()
            .any(|alarm| alarm.alarm() == AlarmType::Nospace)
    );
    let blocked = tokio::spawn(async move {
        let result = publisher
            .records(
                &[
                    record(2, ReplicaMedium::Dram, false),
                    record(3, ReplicaMedium::Ssd, true),
                ],
                3,
                true,
            )
            .await;
        (publisher, result)
    });
    tokio::time::sleep(Duration::from_millis(250)).await;
    assert!(
        !blocked.is_finished(),
        "quota failure must not be reported as committed"
    );
    assert_eq!(
        client.get(marker.clone(), None).await.unwrap().kvs()[0].value(),
        old_marker
    );
    assert!(source.registration_valid() && requester.registration_valid());

    let deleted = client
        .delete("quota-fill/", Some(DeleteOptions::new().with_prefix()))
        .await
        .unwrap();
    client
        .compact(
            deleted.header().unwrap().revision(),
            Some(CompactionOptions::new().with_physical()),
        )
        .await
        .unwrap();
    client.defragment().await.unwrap();
    for alarm in alarms.alarms() {
        let mut options = AlarmOptions::new();
        options.with_member(alarm.member_id());
        client
            .alarm(AlarmAction::Deactivate, alarm.alarm(), Some(options))
            .await
            .unwrap();
    }
    let status = client.status().await.unwrap();
    let alarms = client
        .alarm(AlarmAction::Get, AlarmType::None, None)
        .await
        .unwrap();
    std::fs::write(
        server.directory.join("quota-after-maintenance.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "db_size": status.db_size(),
            "alarms": alarms.alarms().iter().map(|a| (a.member_id(), a.alarm() as i32)).collect::<Vec<_>>(),
        }))
        .unwrap(),
    )
    .unwrap();
    assert!(alarms.alarms().is_empty(), "quota alarm was not disarmed");
    let started = Instant::now();
    let (publisher, result) = tokio::time::timeout(Duration::from_secs(15), blocked)
        .await
        .unwrap()
        .unwrap();
    let committed = result.unwrap();
    wait_for(|| index.revision().is_some_and(|r| r >= committed)).await;
    let candidates = index.lookup(std::slice::from_ref(&key));
    assert_eq!(candidates[0].replicas.len(), 1);
    assert_eq!(
        candidates[0].replicas[0].metadata.medium,
        ReplicaMedium::Ssd
    );
    assert_eq!(candidates[0].replicas[0].sequence, 3);
    assert_eq!(publisher.progress.sequence, 3);
    assert!(source.registration_valid());
    std::fs::write(
        server.directory.join("quota-recovery.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "quota_bytes": 2097152, "rejection": rejection,
            "recovery_ms": started.elapsed().as_secs_f64() * 1000.0,
            "committed_revision": committed, "final_sequence": publisher.progress.sequence,
            "registration_valid": source.registration_valid(), "index_bytes": index.bytes(),
        }))
        .unwrap(),
    )
    .unwrap();
    client.lease_revoke(grant.id()).await.unwrap();
    requester_cluster.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires ETCD_BIN; real Watch budget exhaustion and complete rebuild after eviction"]
async fn index_budget_exhaustion_hides_partial_results_until_eviction_allows_a_complete_rebuild() {
    let server = Etcd::start(1).await;
    let reader = view(56002);
    let index = Arc::new(orbitkv_catalog::GlobalIndex::new(reader.clone(), 2048));
    let member = crate::cluster::Cluster::join(
        &server.endpoints,
        "index-budget",
        "reader",
        12,
        reader.clone(),
        Arc::new(ResidencyInventory::new(4096)),
        index.clone(),
    )
    .await
    .unwrap();
    wait_for(|| index.status().available).await;
    let source = view(56001);
    let mut client = Client::connect(&server.endpoints, None).await.unwrap();
    let sent_at = Instant::now();
    let grant = client.lease_grant(120, None).await.unwrap();
    assert!(source.renew(sent_at, Duration::from_secs(120)));
    let cluster = cluster_id(grant.header()).unwrap();
    let prefix = "/orbitkv/v2/index-budget/";
    let registered = crate::cluster::registration::register(
        &mut client,
        prefix,
        "source",
        source.owner(),
        grant.id(),
        cluster,
    )
    .await
    .unwrap();
    let mut publisher = Publisher {
        client: client.clone(),
        prefix: prefix.into(),
        member: registered,
        view: source.clone(),
        cluster,
        progress: Progress::default(),
        previous: None,
    };
    let records: Vec<_> = (1u64..=24)
        .map(|sequence| {
            let mut value = record(sequence, ReplicaMedium::Dram, true);
            value.key = StateKey::new("budget".into(), sequence.to_le_bytes().to_vec());
            value
        })
        .collect();
    publisher.records(&records, 24, true).await.unwrap();
    wait_for(|| !index.status().available).await;
    let keys: Vec<_> = records.iter().map(|r| r.key.clone()).collect();
    for _ in 0..32 {
        assert!(!index.status().available);
        assert!(
            index
                .lookup(&keys)
                .iter()
                .all(|row| row.replicas.is_empty())
        );
        assert!(index.bytes() <= 2048);
        assert!(
            reader.registration_valid(),
            "budget repair must not stop lease renewal"
        );
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    let retired: Vec<_> = records
        .iter()
        .skip(1)
        .enumerate()
        .map(|(i, record)| {
            let mut record = record.clone();
            record.present = false;
            record.sequence = 25 + i as u64;
            record
        })
        .collect();
    let revision = publisher.records(&retired, 47, true).await.unwrap();
    wait_for(|| index.status().available && index.revision().is_some_and(|r| r >= revision)).await;
    let rows = index.lookup(&keys);
    assert_eq!(rows[0].replicas.len(), 1);
    assert!(rows[1..].iter().all(|row| row.replicas.is_empty()));
    assert!(reader.registration_valid());
    std::fs::write(
        server.directory.join("index-budget-recovery.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "index_budget": 2048, "offered_records": 24, "final_residents": 1,
            "index_bytes": index.bytes(), "recovered_revision": index.revision(),
            "registration_valid": reader.registration_valid(),
        }))
        .unwrap(),
    )
    .unwrap();
    client.lease_revoke(grant.id()).await.unwrap();
    member.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires ETCD_BIN; real delayed Watch, network partition, compaction and rebuild"]
async fn delayed_and_compacted_watch_recovers_from_a_partition_without_partial_visibility() {
    let server = Etcd::start(1).await;
    let gate = TcpGate::start(&server.endpoints[0]).await;
    let reader = view(58002);
    let index = Arc::new(orbitkv_catalog::GlobalIndex::new(reader.clone(), 1 << 20));
    let reader_cluster = crate::cluster::Cluster::join(
        std::slice::from_ref(&gate.endpoint),
        "watch-fault",
        "reader",
        120,
        reader.clone(),
        Arc::new(ResidencyInventory::new(4096)),
        index.clone(),
    )
    .await
    .unwrap();
    wait_for(|| index.status().available).await;

    let source = view(58001);
    let mut client = Client::connect(&server.endpoints, None).await.unwrap();
    let sent_at = Instant::now();
    let grant = client.lease_grant(120, None).await.unwrap();
    assert!(source.renew(sent_at, Duration::from_secs(120)));
    let cluster = cluster_id(grant.header()).unwrap();
    let prefix = "/orbitkv/v2/watch-fault/";
    let registered = crate::cluster::registration::register(
        &mut client,
        prefix,
        "source",
        source.owner(),
        grant.id(),
        cluster,
    )
    .await
    .unwrap();
    let mut publisher = Publisher {
        client: client.clone(),
        prefix: prefix.into(),
        member: registered,
        view: source.clone(),
        cluster,
        progress: Progress::default(),
        previous: None,
    };
    wait_for(|| reader.permits(source.owner())).await;

    let dram = record(1, ReplicaMedium::Dram, true);
    let key = dram.key.clone();
    let revision = publisher.records(&[dram], 1, true).await.unwrap();
    wait_for(|| index.revision().is_some_and(|value| value >= revision)).await;
    assert_eq!(
        index.lookup(std::slice::from_ref(&key))[0].replicas.len(),
        1
    );

    gate.set_downstream_delay(Duration::from_millis(400));
    let delayed_at = Instant::now();
    let revision = publisher
        .records(&[record(2, ReplicaMedium::Ssd, true)], 2, true)
        .await
        .unwrap();
    wait_for(|| index.revision().is_some_and(|value| value >= revision)).await;
    let watch_delay = delayed_at.elapsed();
    assert!(
        watch_delay >= Duration::from_millis(300),
        "configured Watch delay was not observed: {watch_delay:?}"
    );
    gate.set_downstream_delay(Duration::ZERO);

    let old_revision = index.revision().unwrap();
    let partitioned_at = Instant::now();
    gate.partition().await;
    let second_key = StateKey::new("partition-final".into(), vec![2; 32]);
    let mut second = record(4, ReplicaMedium::Dram, true);
    second.key = second_key.clone();
    let committed = publisher
        .records(&[record(3, ReplicaMedium::Dram, false), second], 4, true)
        .await
        .unwrap();
    client.compact(committed, None).await.unwrap();
    tokio::time::sleep(Duration::from_millis(1500)).await;
    assert_eq!(index.revision(), Some(old_revision));
    assert!(index.status().available && reader.registration_valid());
    assert_eq!(
        index.lookup(std::slice::from_ref(&key))[0].replicas.len(),
        2
    );
    assert!(
        index.lookup(std::slice::from_ref(&second_key))[0]
            .replicas
            .is_empty()
    );

    let partition_duration = partitioned_at.elapsed();
    gate.heal(Duration::from_millis(200));
    let healed_at = Instant::now();
    let deadline = Instant::now() + Duration::from_secs(20);
    let mut unavailable_at = None;
    let keys = [key.clone(), second_key.clone()];
    loop {
        let rows = index.lookup(&keys);
        let old_complete = rows[0].replicas.len() == 2 && rows[1].replicas.is_empty();
        let hidden = rows.iter().all(|row| row.replicas.is_empty());
        let final_complete = rows[0].replicas.len() == 1
            && rows[0].replicas[0].metadata.medium == ReplicaMedium::Ssd
            && rows[1].replicas.len() == 1;
        assert!(
            old_complete || hidden || final_complete,
            "Watch exposed a partial index during repair"
        );
        if hidden {
            unavailable_at.get_or_insert_with(Instant::now);
        }
        if final_complete {
            assert!(
                index.revision().is_some_and(|value| value >= committed),
                "final candidates became visible before their revision"
            );
            break;
        }
        assert!(Instant::now() < deadline, "compacted Watch did not rebuild");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let unavailable_at = unavailable_at.expect("compaction rebuild never hid incomplete coverage");
    let heal_to_complete = healed_at.elapsed();
    let incomplete_visibility = unavailable_at.elapsed();
    assert!(reader.registration_valid() && reader.permits(source.owner()));
    let status = client.status().await.unwrap();
    std::fs::write(
        server.directory.join("watch-partition-recovery.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "configured_watch_delay_ms": 400,
            "observed_watch_delay_ms": watch_delay.as_secs_f64() * 1000.0,
            "partition_ms": partition_duration.as_secs_f64() * 1000.0,
            "heal_to_complete_ms": heal_to_complete.as_secs_f64() * 1000.0,
            "incomplete_visibility_ms": incomplete_visibility.as_secs_f64() * 1000.0,
            "old_revision": old_revision,
            "committed_revision": committed,
            "final_revision": index.revision(),
            "index_bytes": index.bytes(),
            "etcd_db_size": status.db_size(),
            "registration_valid": reader.registration_valid(),
        }))
        .unwrap(),
    )
    .unwrap();
    client.lease_revoke(grant.id()).await.unwrap();
    reader_cluster.shutdown().await;
    gate.shutdown().await;
}
