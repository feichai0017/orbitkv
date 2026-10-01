use super::*;
#[cfg(feature = "test-hooks")]
use crate::proto::engine::inventory_server::InventoryServer;
#[cfg(feature = "test-hooks")]
use orbitkv_state::{
    DiscoveryCoverage, ReplicaMedium, ReplicaMetadata, ReplicaRepresentation, StateKey,
};
#[cfg(feature = "test-hooks")]
use tonic::transport::Server;

#[cfg(feature = "test-hooks")]
use crate::cluster::tests::etcd::{Etcd, join, view, wait_for};
#[cfg(feature = "test-hooks")]
use crate::cluster::tests::gate::TcpGate;

#[cfg(feature = "test-hooks")]
fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

#[cfg(feature = "test-hooks")]
async fn serve(runtime: InventoryRuntime, port: u16) -> tokio::task::JoinHandle<()> {
    let task = tokio::spawn(async move {
        Server::builder()
            .add_service(
                InventoryServer::new(runtime.service())
                    .max_decoding_message_size(MESSAGE_BYTES)
                    .max_encoding_message_size(MESSAGE_BYTES),
            )
            .serve(([127, 0, 0, 1], port).into())
            .await
            .unwrap();
    });
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while tokio::net::TcpStream::connect(("127.0.0.1", port))
        .await
        .is_err()
    {
        assert!(std::time::Instant::now() < deadline);
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    task
}

#[cfg(feature = "test-hooks")]
fn metadata(medium: ReplicaMedium) -> ReplicaMetadata {
    ReplicaMetadata {
        medium,
        representation: ReplicaRepresentation::Raw,
        stored_bytes: Some(4096),
    }
}

fn runtime_for_protocol_test() -> (InventoryRuntime, Member, CacheOwner) {
    let own = CacheOwner {
        endpoint: "127.0.0.1:59001".into(),
        incarnation: Uuid::new_v4(),
    };
    let requester = CacheOwner {
        endpoint: "127.0.0.1:59002".into(),
        incarnation: Uuid::new_v4(),
    };
    let membership = Arc::new(MembershipView::new(own.clone()));
    assert!(membership.renew(std::time::Instant::now(), Duration::from_secs(60)));
    membership.replace_members([
        ("source".into(), own.clone()),
        ("requester".into(), requester.clone()),
    ]);
    let member = Member {
        node_id: "source".into(),
        epoch: 1,
        owner: own,
        protocol: INVENTORY_STREAM_PROTOCOL.into(),
        lease: 1,
    };
    let format = ClusterFormat {
        protocol: INVENTORY_STREAM_PROTOCOL.into(),
        cluster_uuid: Uuid::new_v4(),
        revision: 1,
        encoded: b"format".to_vec(),
    };
    let index = Arc::new(GlobalIndex::new(membership.clone(), 1 << 20));
    let runtime = InventoryRuntime::new(
        format,
        member.clone(),
        Arc::new(ResidencyInventory::new(1 << 20)),
        index,
        membership,
    );
    runtime.replace_members(
        1,
        [
            (member.node_id.clone(), member.clone()),
            (
                "requester".into(),
                Member {
                    node_id: "requester".into(),
                    epoch: 1,
                    owner: requester.clone(),
                    protocol: INVENTORY_STREAM_PROTOCOL.into(),
                    lease: 2,
                },
            ),
        ]
        .into_iter()
        .collect(),
    );
    (runtime, member, requester)
}

#[test]
fn open_and_frame_credit_reject_wrong_identity_and_future_ack() {
    let (runtime, source, requester) = runtime_for_protocol_test();
    let session = Uuid::new_v4();
    let valid = InventoryOpen {
        protocol: INVENTORY_STREAM_PROTOCOL.into(),
        cluster_uuid: runtime.shared.format.cluster_uuid.as_bytes().to_vec(),
        source_node: source.node_id.clone(),
        source_epoch: source.epoch,
        source_incarnation: source.owner.incarnation.as_bytes().to_vec(),
        requester_incarnation: requester.incarnation.as_bytes().to_vec(),
        scope_digest: all_namespaces_scope_digest().to_vec(),
        session_id: session.as_bytes().to_vec(),
        resume_sequence: None,
        installed_view_id: Vec::new(),
    };
    assert!(validate_open(&runtime, &valid).is_ok());
    let mut wrong = valid.clone();
    wrong.cluster_uuid = Uuid::new_v4().as_bytes().to_vec();
    assert!(validate_open(&runtime, &wrong).is_err());

    let (output, _) = mpsc::channel(1);
    let (control, controls) = watch::channel(ClientControl {
        consumed_frame_id: 1,
        ..ClientControl::default()
    });
    let mut sender = FrameSender::new(runtime, session, output, controls);
    assert!(sender.release_acknowledged().is_err());
    control.send_modify(|state| state.consumed_frame_id = 0);
    assert!(sender.release_acknowledged().is_ok());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg(feature = "test-hooks")]
#[ignore = "requires ETCD_BIN; real membership plus inventory gRPC snapshot/delta/reset"]
async fn stream_installs_exact_owner_views_and_recovers_overflow_without_etcd_blocks() {
    let etcd = Etcd::start(1).await;
    let source_backend_port = free_port();
    let gate = TcpGate::start(&format!("http://127.0.0.1:{source_backend_port}")).await;
    let source_port = gate.endpoint.rsplit(':').next().unwrap().parse().unwrap();
    let target_port = free_port();
    let source_view = view(source_port);
    let target_view = view(target_port);
    let (source, _, source_inventory) =
        join(&etcd, "inventory-stream", "source", source_view.clone(), 60).await;
    let (target, target_index, _) =
        join(&etcd, "inventory-stream", "target", target_view.clone(), 60).await;
    let source_runtime = source.inventory();
    let target_runtime = target.inventory();
    let source_server = serve(source_runtime.clone(), source_backend_port).await;
    let target_server = serve(target_runtime.clone(), target_port).await;
    gate.set_downstream_delay(Duration::from_millis(2));
    wait_for(|| {
        source_view.permits(target_view.owner()) && target_view.permits(source_view.owner())
    })
    .await;

    let first = StateKey::new("model".into(), vec![1; 32]);
    let second = StateKey::new("model".into(), vec![2; 32]);
    source_inventory.test_change(
        &first,
        ReplicaMedium::Dram,
        Some(metadata(ReplicaMedium::Dram)),
    );
    source_inventory.test_change(
        &first,
        ReplicaMedium::Ssd,
        Some(metadata(ReplicaMedium::Ssd)),
    );
    source_inventory.test_change(
        &second,
        ReplicaMedium::Dram,
        Some(metadata(ReplicaMedium::Dram)),
    );
    let fence = source_runtime.capture_fence().unwrap();
    target_runtime
        .await_fence(
            &fence,
            &all_namespaces_scope_digest(),
            Duration::from_secs(10),
        )
        .await
        .unwrap();
    let rows = target_index.lookup(&[first.clone(), second.clone()]);
    assert_eq!(rows[0].replicas.len(), 2);
    assert_eq!(rows[1].replicas.len(), 1);
    assert!(rows.iter().all(|row| {
        row.coverage == DiscoveryCoverage::CompleteAtWatermarks
            && row
                .replicas
                .iter()
                .all(|replica| replica.owner == *source_view.owner())
    }));

    gate.partition().await;
    for id in 0u32..400 {
        let key = StateKey::new("overflow".into(), id.to_le_bytes().to_vec());
        source_inventory.test_change(
            &key,
            ReplicaMedium::Dram,
            Some(metadata(ReplicaMedium::Dram)),
        );
        source_inventory.test_change(&key, ReplicaMedium::Dram, None);
    }
    source_inventory.test_change(&first, ReplicaMedium::Dram, None);
    source_inventory.test_change(
        &second,
        ReplicaMedium::Ssd,
        Some(metadata(ReplicaMedium::Ssd)),
    );
    let final_fence = source_runtime.capture_fence().unwrap();
    gate.heal(Duration::ZERO);
    target_runtime
        .await_fence(
            &final_fence,
            &all_namespaces_scope_digest(),
            Duration::from_secs(15),
        )
        .await
        .unwrap();
    let rows = target_index.lookup(&[first.clone(), second.clone()]);
    assert_eq!(rows[0].replicas.len(), 1);
    assert_eq!(rows[0].replicas[0].metadata.medium, ReplicaMedium::Ssd);
    assert_eq!(rows[1].replicas.len(), 2);
    assert_eq!(
        target_index
            .owner_watermark(source_view.owner().incarnation)
            .unwrap()
            .1,
        final_fence.inventory_sequence
    );
    assert!(source_inventory.status().history_gaps > 0);
    assert!(target_runtime.status().resets > 0);

    let mut client = etcd_client::Client::connect(&etcd.endpoints, None)
        .await
        .unwrap();
    let metadata = client
        .get(
            "/orbitkv/v2/inventory-stream/",
            Some(
                etcd_client::GetOptions::new()
                    .with_prefix()
                    .with_keys_only(),
            ),
        )
        .await
        .unwrap();
    assert!(metadata.kvs().iter().all(|kv| {
        let key = kv.key_str().unwrap();
        !key.contains("/blocks/") && !key.contains("/publishers/")
    }));

    source_server.abort();
    target_server.abort();
    gate.shutdown().await;
    target.shutdown().await;
    source.shutdown().await;
}

#[cfg(feature = "test-hooks")]
#[path = "inventory_capacity.rs"]
mod capacity;

#[tokio::test]
async fn fenced_source_stops_before_queueing_an_inventory_frame() {
    let (runtime, _, _) = runtime_for_protocol_test();
    let (output, mut response) = mpsc::channel(4);
    let (_control, controls) = watch::channel(ClientControl::default());
    let mut sender = FrameSender::new(runtime.clone(), Uuid::new_v4(), output, controls);
    let frame = sender
        .frame(inventory_server_frame::Body::Progress(InventoryProgress {
            through_sequence: 0,
            source_head_sequence: 0,
        }))
        .unwrap();
    runtime.shared.membership.fence();
    assert!(sender.send(frame).await.is_err());
    assert!(response.try_recv().is_err());
    assert_eq!(runtime.status().frames_sent, 0);
    assert_eq!(runtime.status().outbound_queue_bytes, 0);
}
