use super::*;
use crate::cluster::tests::etcd::{Etcd, join, view, wait_for};
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
