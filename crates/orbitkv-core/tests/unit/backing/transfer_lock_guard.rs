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
struct ReleaseCounter(Arc<AtomicUsize>);

#[tonic::async_trait]
impl Engine for ReleaseCounter {
    async fn release_transfer_lock(
        &self,
        _request: Request<ReleaseTransferLockRequest>,
    ) -> Result<Response<ReleaseTransferLockResponse>, Status> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(Response::new(ReleaseTransferLockResponse {
            status: None,
            released_blocks: 0,
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
async fn start_counter_server() -> (EngineClient<Channel>, Arc<AtomicUsize>) {
    let counter = Arc::new(AtomicUsize::new(0));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback");
    let addr = listener.local_addr().expect("local addr");
    let service = EngineServer::new(ReleaseCounter(Arc::clone(&counter)));
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
    TransferLockGuard::new(client.clone(), session.to_string(), "remote", "req")
}

#[tokio::test(flavor = "multi_thread")]
async fn releases_exactly_once_on_every_exit_path() {
    let (client, counter) = start_counter_server().await;

    // Explicit release on the coded path.
    guard(&client, "explicit").release();
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
    guard(&client, "").release();
    drop(guard(&client, ""));
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(counter.load(Ordering::SeqCst), 3);
}

#[tokio::test(flavor = "multi_thread")]
async fn cancellation_keeps_buffers_and_source_pin_until_blocking_transfer_finishes() {
    let (client, counter) = start_counter_server().await;
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
