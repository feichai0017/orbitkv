use super::*;
use crate::endpoint::ProcessEndpoint;
use crate::proto::engine::{RegisterContextRequest, SessionRequest};
use crate::registry::CudaTensorRegistry;
use orbitkv_channel::lifecycle::LifecycleCommand;
use orbitkv_channel::{CallOptions, ChannelClient, ChannelError, QueryBundleRequest};
use orbitkv_common::hll::MultiWindowHllTracker;
use orbitkv_core::StorageConfig;
use prost::Message;
use std::time::Duration;
use tokio::sync::Notify;

pub(crate) fn test_engine() -> Arc<OrbitKVEngine> {
    Arc::new(OrbitKVEngine::new_with_config(1 << 20, false, StorageConfig::default()).unwrap())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn process_channel_lifecycle_and_cache_control_need_no_grpc() {
    let engine = test_engine();
    let lifecycle = LifecycleService::new(
        Arc::clone(&engine),
        RegistryHandle::spawn(CudaTensorRegistry::empty()),
    );
    let shutdown = Arc::new(Notify::new());
    let id = uuid::Uuid::new_v4();
    let socket = std::env::temp_dir().join(format!("orbitkv-lifecycle-{id}.sock"));
    let mut endpoint = ProcessEndpoint::start(
        format!("orbitkv/test/lifecycle/{id}"),
        42,
        socket.clone(),
        1 << 20,
        1 << 16,
        engine,
        tokio::runtime::Handle::current(),
        Arc::new(std::sync::Mutex::new(MultiWindowHllTracker::new(
            vec![("test".into(), Duration::from_secs(60))],
            4,
        ))),
        Arc::clone(&shutdown),
        lifecycle.clone(),
    )
    .unwrap();
    let (first, second) = tokio::task::spawn_blocking(move || {
        let first = ChannelClient::connect(&socket, CallOptions::default()).unwrap();
        first.lifecycle(LifecycleCommand::Health, &[]).unwrap();
        assert!(matches!(
            first.lifecycle(LifecycleCommand::Register, &[0xff]),
            Err(ChannelError::Lifecycle { code: 1, .. })
        ));
        let bad_version = RegisterContextRequest {
            instance_id: "inst".into(),
            namespace: "ns".into(),
            client_version: "old".into(),
            ..Default::default()
        };
        assert!(matches!(
            first.lifecycle(LifecycleCommand::Register, &bad_version.encode_to_vec()),
            Err(ChannelError::Lifecycle { code: 2, .. })
        ));
        first.lifecycle(LifecycleCommand::Health, &[]).unwrap();
        assert!(
            first
                .query_bundle(
                    1,
                    &orbitkv_channel::QueryCommand::Submit(QueryBundleRequest {
                        ticket: orbitkv_channel::QueryTicket {
                            operation_id: 1,
                            revision: 1
                        },
                        instance_id: "missing".into(),
                        request_id: "query".into(),
                        block_hashes: vec![],
                        wait_for_full_prefix: true,
                        warmup: false,
                        group_id: 0,
                    })
                )
                .is_err()
        );
        let session = SessionRequest {
            instance_id: "inst".into(),
            namespace: "ns".into(),
            tp_size: 1,
            world_size: 1,
        }
        .encode_to_vec();
        first
            .lifecycle(LifecycleCommand::Session, &session)
            .unwrap();
        let incompatible = SessionRequest {
            instance_id: "inst".into(),
            namespace: "different-model".into(),
            tp_size: 1,
            world_size: 1,
        }
        .encode_to_vec();
        assert!(matches!(
            first.lifecycle(LifecycleCommand::Session, &incompatible),
            Err(ChannelError::Lifecycle { code: 2, .. })
        ));
        let second = ChannelClient::connect(&socket, CallOptions::default()).unwrap();
        second
            .lifecycle(LifecycleCommand::Session, &session)
            .unwrap();
        (first, second)
    })
    .await
    .unwrap();
    // Old connection teardown must leave the replacement session intact.
    first.close();
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(lifecycle.sessions.topology("inst").is_some());
    second.close();
    tokio::time::timeout(Duration::from_secs(2), async {
        while lifecycle.sessions.topology("inst").is_some() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("disconnect must clean up the current session");
    assert!(matches!(
        second.lifecycle(LifecycleCommand::Health, &[]),
        Err(ChannelError::SessionRequiresReconnect)
    ));
    endpoint.stop();
    shutdown.notify_waiters();
}
