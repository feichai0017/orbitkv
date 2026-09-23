use super::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use orbitkv_proto::proto::engine::engine_server::{Engine, EngineServer};
use orbitkv_proto::proto::engine::{
    HealthRequest, HealthResponse, QueryBlocksForTransferRequest, QueryBlocksForTransferResponse,
    ReleaseTransferLockResponse,
};
use tokio_stream::wrappers::TcpListenerStream;
use tonic::transport::Endpoint;
use tonic::{Request, Response, Status};

/// Stub engine that only counts ReleaseTransferLock calls.
struct ReleaseCounter {
    calls: Arc<AtomicUsize>,
    failures: usize,
    source: Option<Arc<crate::storage::transfer_lock::TransferLockManager>>,
}

#[tonic::async_trait]
impl Engine for ReleaseCounter {
    async fn release_transfer_lock(
        &self,
        request: Request<ReleaseTransferLockRequest>,
    ) -> Result<Response<ReleaseTransferLockResponse>, Status> {
        let released = self.source.as_ref().map_or(0, |source| {
            source.release(&request.get_ref().transfer_session_id)
        });
        if self.calls.fetch_add(1, Ordering::SeqCst) < self.failures {
            return Err(Status::unavailable("release acknowledgement lost"));
        }
        Ok(Response::new(ReleaseTransferLockResponse {
            status: Some(orbitkv_proto::proto::engine::ResponseStatus {
                ok: true,
                message: String::new(),
            }),
            released_blocks: released as u64,
        }))
    }

    async fn query_blocks_for_transfer(
        &self,
        _request: Request<QueryBlocksForTransferRequest>,
    ) -> Result<Response<QueryBlocksForTransferResponse>, Status> {
        Err(Status::unimplemented("stub"))
    }
    async fn health(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        Err(Status::unimplemented("stub"))
    }
}

/// Serve a ReleaseCounter on an ephemeral loopback port; return a
/// connected client and the shared counter.
async fn start_counter_server(failures: usize) -> (EngineClient<Channel>, Arc<AtomicUsize>) {
    let counter = Arc::new(AtomicUsize::new(0));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback");
    let addr = listener.local_addr().expect("local addr");
    let service = EngineServer::new(ReleaseCounter {
        calls: Arc::clone(&counter),
        failures,
        source: None,
    });
    tokio::spawn(
        tonic::transport::Server::builder()
            .add_service(service)
            .serve_with_incoming(TcpListenerStream::new(listener)),
    );
    let channel = Endpoint::from_shared(format!("http://{addr}"))
        .expect("endpoint")
        .connect_lazy();
    (EngineClient::new(channel), counter)
}

async fn wait_for_count(counter: &AtomicUsize, expected: usize) {
    tokio::time::timeout(Duration::from_secs(5), async {
        while counter.load(Ordering::SeqCst) < expected {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("release RPC should arrive");
}

fn guard(client: &EngineClient<Channel>, session: &str) -> TransferLockGuard {
    let budget = Arc::new(TransferCompletions::default());
    let mut guard = budget.reserve(client.clone(), "remote").unwrap();
    guard.authorize(session.into());
    guard
}

#[tokio::test(flavor = "multi_thread")]
async fn releases_exactly_once_on_every_exit_path() {
    let (client, counter) = start_counter_server(0).await;

    // Explicit release on the coded path.
    drop(guard(&client, "explicit"));
    wait_for_count(&counter, 1).await;

    // Drop without release (future cancelled) still releases.
    drop(guard(&client, "dropped"));
    wait_for_count(&counter, 2).await;

    // Panic unwinding through the guard still releases.
    let g = guard(&client, "panicked");
    let task = tokio::spawn(async move {
        let _g = g;
        panic!("simulated fetch panic");
    });
    assert!(task.await.is_err());
    wait_for_count(&counter, 3).await;

    // No double release: settle window after all three paths.
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(counter.load(Ordering::SeqCst), 3);

    // Empty session (holder returned none) never sends an RPC.
    drop(guard(&client, ""));
    drop(guard(&client, ""));
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(counter.load(Ordering::SeqCst), 3);
}

#[tokio::test(flavor = "multi_thread")]
async fn cancellation_keeps_buffers_and_source_pin_until_blocking_transfer_finishes() {
    let (client, counter) = start_counter_server(0).await;
    let buffer = Arc::new(());
    let observed = Arc::downgrade(&buffer);
    let g = guard(&client, "in-flight");
    let (started, ready) = tokio::sync::oneshot::channel();
    let (finish, finished) = std::sync::mpsc::channel();
    let task = tokio::spawn(async move {
        g.run_with_buffers(buffer, move || {
            started.send(()).unwrap();
            finished.recv_timeout(Duration::from_secs(5)).unwrap();
        })
        .await
        .unwrap()
    });
    ready.await.unwrap();
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    assert!(observed.upgrade().is_some());
    assert_eq!(counter.load(Ordering::SeqCst), 0);
    finish.send(()).unwrap();
    wait_for_count(&counter, 1).await;
    assert!(observed.upgrade().is_none());
}

#[tokio::test(flavor = "multi_thread")]
async fn completion_survives_prolonged_lost_acknowledgements() {
    let (client, counter) = start_counter_server(5).await;
    let budget = Arc::new(TransferCompletions::default());
    let mut g = budget.reserve(client, "remote").unwrap();
    g.authorize("completed".into());
    drop(g);
    wait_for_count(&counter, 4).await;
    assert_eq!(budget.slots.available_permits(), MAX_COMPLETIONS - 1);
    tokio::time::timeout(Duration::from_secs(10), async {
        while budget.slots.available_permits() != MAX_COMPLETIONS {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    assert_eq!(counter.load(Ordering::SeqCst), 6);
    assert!(budget.peers.lock().is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn admission_is_bounded_and_an_unreachable_peer_does_not_block_other_peers() {
    let (offline, _) = start_counter_server(usize::MAX).await;
    let (healthy, calls) = start_counter_server(0).await;
    let budget = Arc::new(TransferCompletions::default());
    for i in 0..MAX_PEER_COMPLETIONS {
        let mut g = budget.reserve(offline.clone(), "offline").unwrap();
        g.authorize(format!("session-{i}"));
        drop(g);
    }
    assert!(budget.reserve(offline.clone(), "offline").is_none());
    let mut g = budget.reserve(healthy.clone(), "healthy").unwrap();
    g.authorize("healthy-session".into());
    drop(g);
    wait_for_count(&calls, 1).await;
    tokio::time::timeout(Duration::from_secs(2), async {
        while budget.peers.lock().contains_key("healthy") {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .unwrap();
    let mut held = Vec::new();
    for peer in 1..MAX_COMPLETIONS / MAX_PEER_COMPLETIONS {
        for _ in 0..MAX_PEER_COMPLETIONS {
            held.push(
                budget
                    .reserve(offline.clone(), &format!("peer-{peer}"))
                    .unwrap(),
            );
        }
    }
    assert_eq!(budget.slots.available_permits(), 0);
    assert!(budget.reserve(healthy.clone(), "new-peer").is_none());
    drop(held);
    assert!(budget.reserve(healthy, "new-peer").is_some());
}

#[tokio::test(flavor = "multi_thread")]
async fn source_pins_reconcile_after_connection_loss_and_duplicate_replies() {
    use crate::block::{SealedBlock, StateKey};
    use crate::storage::transfer_lock::TransferLockManager;
    let source = Arc::new(TransferLockManager::new(Duration::ZERO, 1));
    let block = Arc::new(SealedBlock::from_slots(Vec::new()));
    let memory = Arc::downgrade(&block);
    let session = source
        .lock_blocks(
            "old-runtime",
            vec![(StateKey::new("ns".into(), vec![1]), block)],
        )
        .unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    let channel = Endpoint::from_shared(format!("http://{addr}"))
        .unwrap()
        .connect_lazy();
    let client = EngineClient::new(channel);
    let budget = Arc::new(TransferCompletions::default());
    let mut g = budget.reserve(client.clone(), &addr.to_string()).unwrap();
    g.authorize(session);
    drop(g);
    // Longer than the old three-attempt retry window; the control endpoint is down.
    tokio::time::sleep(Duration::from_secs(2)).await;
    assert_eq!(source.expire(), 1);
    assert!(memory.upgrade().is_some());
    assert_eq!(budget.slots.available_permits(), MAX_COMPLETIONS - 1);
    // A new requester incarnation does not release the old source reservation.
    let new_block = Arc::new(SealedBlock::from_slots(Vec::new()));
    let new_memory = Arc::downgrade(&new_block);
    let new_session = source
        .lock_blocks(
            "new-runtime",
            vec![(StateKey::new("ns".into(), vec![2]), new_block)],
        )
        .unwrap();
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    let counter = Arc::new(AtomicUsize::new(0));
    let server = tokio::spawn(
        tonic::transport::Server::builder()
            .add_service(EngineServer::new(ReleaseCounter {
                calls: counter.clone(),
                failures: 1,
                source: Some(source.clone()),
            }))
            .serve_with_incoming(TcpListenerStream::new(listener)),
    );
    tokio::time::timeout(Duration::from_secs(15), async {
        while budget.slots.available_permits() != MAX_COMPLETIONS {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    assert!(
        counter.load(Ordering::SeqCst) >= 2,
        "first reply was lost after release"
    );
    assert!(memory.upgrade().is_none());
    assert!(
        new_memory.upgrade().is_some(),
        "late old completion must not release a new session"
    );
    assert_eq!(source.release(&new_session), 1);
    assert!(new_memory.upgrade().is_none());
    server.abort();
}
