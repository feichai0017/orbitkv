use super::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use orbitkv_proto::proto::engine::engine_server::{Engine, EngineServer};
use orbitkv_proto::proto::engine::{
    HealthRequest, HealthResponse, OpenTransferWindowResponse, ReleaseTransferLockResponse,
    ResponseStatus,
};
use tokio::sync::Notify;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{Request, Response};

use crate::block::{SealedBlock, StateKey};
use crate::storage::transfer_lock::{TransferLockManager, TransferTicket as NativeTicket};

struct Source {
    locks: Arc<TransferLockManager>,
    opens: AtomicUsize,
    releases: AtomicUsize,
    released_blocks: AtomicUsize,
    failures: AtomicUsize,
    lose_authorization: AtomicBool,
    lose_setup: AtomicBool,
    delay: Mutex<Option<Arc<Notify>>>,
    started: Notify,
    finished: Notify,
    late_rejected: AtomicBool,
}

impl Default for Source {
    fn default() -> Self {
        Self {
            locks: Arc::new(TransferLockManager::new(Duration::ZERO, 1)),
            opens: AtomicUsize::new(0),
            releases: AtomicUsize::new(0),
            released_blocks: AtomicUsize::new(0),
            failures: AtomicUsize::new(0),
            lose_authorization: AtomicBool::new(false),
            lose_setup: AtomicBool::new(false),
            delay: Mutex::new(None),
            started: Notify::new(),
            finished: Notify::new(),
            late_rejected: AtomicBool::new(false),
        }
    }
}

fn native(ticket: TransferTicket) -> NativeTicket {
    NativeTicket {
        window: ticket.window_id.parse().unwrap(),
        slot: ticket.slot as usize,
        generation: ticket.generation,
    }
}

fn ok() -> ResponseStatus {
    ResponseStatus {
        ok: true,
        message: String::new(),
    }
}

struct Service(Arc<Source>);

#[tonic::async_trait]
impl Engine for Service {
    async fn open_transfer_window(
        &self,
        request: Request<OpenTransferWindowRequest>,
    ) -> Result<Response<OpenTransferWindowResponse>, Status> {
        let source = &self.0;
        source.opens.fetch_add(1, Ordering::SeqCst);
        let window = source
            .locks
            .open(request.into_inner().requester_incarnation.parse().unwrap())
            .unwrap();
        if source.lose_setup.load(Ordering::SeqCst) {
            return Err(Status::unavailable("lost setup reply"));
        }
        Ok(Response::new(OpenTransferWindowResponse {
            window_id: window.to_string(),
        }))
    }

    async fn query_blocks_for_transfer(
        &self,
        request: Request<QueryBlocksForTransferRequest>,
    ) -> Result<Response<QueryBlocksForTransferResponse>, Status> {
        let source = self.0.clone();
        // Processing may already be queued even when tonic drops the response
        // future. Keep it alive to exercise actual release-before-authorize.
        tokio::spawn(async move {
            let delay = source.delay.lock().clone();
            source.started.notify_one();
            if let Some(delay) = delay {
                delay.notified().await;
            }
            let ticket = native(request.into_inner().ticket.unwrap());
            let result = source.locks.lock_blocks(
                ticket,
                vec![(
                    StateKey::new("ns".into(), vec![1]),
                    Arc::new(SealedBlock::from_slots(Vec::new())),
                )],
            );
            source
                .late_rejected
                .store(result.is_err(), Ordering::SeqCst);
            source.finished.notify_one();
            result.map_err(|error| match error {
                crate::storage::transfer_lock::TransferLockError::UnknownWindow => {
                    Status::not_found("evicted window")
                }
                _ => Status::failed_precondition("closed ticket"),
            })?;
            if source.lose_authorization.load(Ordering::SeqCst) {
                return Err(Status::unavailable(
                    "lost authorization reply after pinning",
                ));
            }
            Ok(Response::new(QueryBlocksForTransferResponse {
                status: Some(ok()),
                ..Default::default()
            }))
        })
        .await
        .unwrap()
    }

    async fn release_transfer_lock(
        &self,
        request: Request<ReleaseTransferLockRequest>,
    ) -> Result<Response<ReleaseTransferLockResponse>, Status> {
        let source = &self.0;
        let released = source
            .locks
            .release(native(request.into_inner().ticket.unwrap()))
            .unwrap();
        source.released_blocks.fetch_add(released, Ordering::SeqCst);
        if source.releases.fetch_add(1, Ordering::SeqCst) < source.failures.load(Ordering::SeqCst) {
            return Err(Status::unavailable("release acknowledgement lost"));
        }
        Ok(Response::new(ReleaseTransferLockResponse {
            status: Some(ok()),
            released_blocks: released as u64,
        }))
    }

    async fn health(&self, _: Request<HealthRequest>) -> Result<Response<HealthResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }
}

struct Fixture {
    source: Arc<Source>,
    owner: CacheOwner,
    server: tokio::task::JoinHandle<Result<(), tonic::transport::Error>>,
}

impl Fixture {
    async fn new() -> Self {
        let source = Arc::new(Source::default());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let owner = CacheOwner {
            endpoint: listener.local_addr().unwrap().to_string(),
            incarnation: Uuid::new_v4(),
        };
        let server = tokio::spawn(
            tonic::transport::Server::builder()
                .add_service(EngineServer::new(Service(source.clone())))
                .serve_with_incoming(TcpListenerStream::new(listener)),
        );
        Self {
            source,
            owner,
            server,
        }
    }

    fn segment(&self) -> FetchSegment {
        FetchSegment {
            owner: self.owner.clone(),
            records: vec![orbitkv_state::InventoryRecord {
                key: StateKey::new("ns".into(), vec![1]),
                sequence: 1,
                present: true,
            }],
        }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.server.abort();
    }
}

async fn settled(budget: &TransferCompletions) {
    tokio::time::timeout(Duration::from_secs(15), async {
        while budget.slots.available_permits() != MAX_COMPLETIONS {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn lost_authorization_reply_releases_pins_and_reuses_window_and_slot() {
    let fixture = Fixture::new().await;
    let budget = TransferCompletions::default();
    fixture
        .source
        .lose_authorization
        .store(true, Ordering::SeqCst);
    assert!(
        budget
            .authorize(&fixture.segment(), Uuid::new_v4())
            .await
            .is_err()
    );
    settled(&budget).await;
    assert_eq!(fixture.source.released_blocks.load(Ordering::SeqCst), 1);
    let first_peer = budget.peer(&fixture.owner).unwrap();
    assert_eq!(first_peer.state.lock().slots[0].generation, 1);
    fixture
        .source
        .lose_authorization
        .store(false, Ordering::SeqCst);
    let (guard, _) = budget
        .authorize(&fixture.segment(), Uuid::new_v4())
        .await
        .unwrap();
    assert_eq!(guard.completion.as_ref().unwrap().generation, 2);
    assert_eq!(
        fixture.source.opens.load(Ordering::SeqCst),
        1,
        "setup is reused"
    );
    drop(guard);
    settled(&budget).await;
    assert_eq!(fixture.source.released_blocks.load(Ordering::SeqCst), 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn cancellation_closes_ticket_before_delayed_authorization_and_slot_reuse() {
    let fixture = Fixture::new().await;
    let delay = Arc::new(Notify::new());
    *fixture.source.delay.lock() = Some(delay.clone());
    let budget = Arc::new(TransferCompletions::default());
    let task = {
        let budget = budget.clone();
        let segment = fixture.segment();
        tokio::spawn(async move { budget.authorize(&segment, Uuid::new_v4()).await.is_ok() })
    };
    fixture.source.started.notified().await;
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    settled(&budget).await;
    assert_eq!(fixture.source.releases.load(Ordering::SeqCst), 1);
    assert_eq!(fixture.source.released_blocks.load(Ordering::SeqCst), 0);
    // Reuse before the first queued request executes.
    *fixture.source.delay.lock() = None;
    let (new_read, _) = budget
        .authorize(&fixture.segment(), Uuid::new_v4())
        .await
        .unwrap();
    fixture.source.finished.notified().await;
    delay.notify_one();
    tokio::time::timeout(Duration::from_secs(2), fixture.source.finished.notified())
        .await
        .unwrap();
    assert!(fixture.source.late_rejected.load(Ordering::SeqCst));
    assert_eq!(
        fixture.source.locks.expire(),
        1,
        "only the new READ owns a pin"
    );
    drop(new_read);
    settled(&budget).await;
    assert_eq!(fixture.source.released_blocks.load(Ordering::SeqCst), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn lost_setup_reply_and_idle_eviction_recover_without_payload_leaks() {
    let fixture = Fixture::new().await;
    let budget = TransferCompletions::default();
    fixture.source.lose_setup.store(true, Ordering::SeqCst);
    assert!(
        budget
            .authorize(&fixture.segment(), Uuid::new_v4())
            .await
            .is_err()
    );
    settled(&budget).await;
    assert_eq!(fixture.source.releases.load(Ordering::SeqCst), 0);
    assert_eq!(fixture.source.locks.expire(), 0);
    fixture.source.lose_setup.store(false, Ordering::SeqCst);
    drop(
        budget
            .authorize(&fixture.segment(), Uuid::new_v4())
            .await
            .unwrap(),
    );
    settled(&budget).await;
    for _ in 0..1024 {
        fixture.source.locks.open(Uuid::new_v4()).unwrap();
    }
    assert!(
        matches!(budget.authorize(&fixture.segment(), Uuid::new_v4()).await, Err(error) if error.code() == tonic::Code::NotFound)
    );
    settled(&budget).await;
    drop(
        budget
            .authorize(&fixture.segment(), Uuid::new_v4())
            .await
            .unwrap(),
    );
    settled(&budget).await;
    assert_eq!(fixture.source.opens.load(Ordering::SeqCst), 3);
    assert_eq!(fixture.source.released_blocks.load(Ordering::SeqCst), 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn cancellation_keeps_buffers_and_source_pin_until_blocking_transfer_finishes() {
    let fixture = Fixture::new().await;
    let budget = TransferCompletions::default();
    let (guard, _) = budget
        .authorize(&fixture.segment(), Uuid::new_v4())
        .await
        .unwrap();
    let buffer = Arc::new(());
    let observed = Arc::downgrade(&buffer);
    let (started, ready) = tokio::sync::oneshot::channel();
    let (finish, finished) = std::sync::mpsc::channel();
    let task = tokio::spawn(async move {
        guard
            .run_with_buffers(buffer, move || {
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
    assert_eq!(fixture.source.releases.load(Ordering::SeqCst), 0);
    assert_eq!(fixture.source.locks.expire(), 1);
    finish.send(()).unwrap();
    settled(&budget).await;
    assert!(observed.upgrade().is_none());
    assert_eq!(fixture.source.released_blocks.load(Ordering::SeqCst), 1);
    // Unwind before READ also closes its single-use ticket.
    let (guard, _) = budget
        .authorize(&fixture.segment(), Uuid::new_v4())
        .await
        .unwrap();
    assert!(
        tokio::spawn(async move {
            let _held = guard;
            panic!("fetch failed");
        })
        .await
        .is_err()
    );
    settled(&budget).await;
    assert_eq!(fixture.source.released_blocks.load(Ordering::SeqCst), 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn completion_survives_prolonged_lost_acknowledgements() {
    let fixture = Fixture::new().await;
    fixture.source.failures.store(5, Ordering::SeqCst);
    let budget = TransferCompletions::default();
    drop(
        budget
            .authorize(&fixture.segment(), Uuid::new_v4())
            .await
            .unwrap(),
    );
    tokio::time::timeout(Duration::from_secs(5), async {
        while fixture.source.releases.load(Ordering::SeqCst) < 4 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    assert_eq!(budget.slots.available_permits(), MAX_COMPLETIONS - 1);
    settled(&budget).await;
    assert_eq!(fixture.source.releases.load(Ordering::SeqCst), 6);
    assert_eq!(fixture.source.released_blocks.load(Ordering::SeqCst), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn admission_is_bounded_and_unreachable_peer_does_not_block_other_peers() {
    let offline = Fixture::new().await;
    let healthy = Fixture::new().await;
    offline.source.failures.store(usize::MAX, Ordering::SeqCst);
    let budget = TransferCompletions::default();
    for _ in 0..TRANSFER_WINDOW_SLOTS {
        drop(
            budget
                .reserve(&offline.owner, Uuid::new_v4())
                .await
                .unwrap(),
        );
    }
    assert!(
        budget
            .reserve(&offline.owner, Uuid::new_v4())
            .await
            .is_err()
    );
    drop(
        budget
            .authorize(&healthy.segment(), Uuid::new_v4())
            .await
            .unwrap(),
    );
    tokio::time::timeout(Duration::from_secs(2), async {
        while healthy.source.released_blocks.load(Ordering::SeqCst) == 0 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    let mut held = Vec::new();
    for _ in 1..MAX_COMPLETIONS / TRANSFER_WINDOW_SLOTS {
        let owner = CacheOwner {
            incarnation: Uuid::new_v4(),
            ..offline.owner.clone()
        };
        for _ in 0..TRANSFER_WINDOW_SLOTS {
            held.push(budget.reserve(&owner, Uuid::new_v4()).await.unwrap());
        }
    }
    assert_eq!(budget.slots.available_permits(), 0);
    assert!(
        budget
            .reserve(&healthy.owner, Uuid::new_v4())
            .await
            .is_err()
    );
    offline.source.failures.store(0, Ordering::SeqCst);
    drop(held);
    settled(&budget).await;
    drop(
        budget
            .reserve(&healthy.owner, Uuid::new_v4())
            .await
            .unwrap(),
    );
    settled(&budget).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn idle_peer_cache_is_bounded_and_never_replaces_busy_peer_state() {
    let fixture = Fixture::new().await;
    let budget = TransferCompletions::default();
    let held = budget
        .reserve(&fixture.owner, Uuid::new_v4())
        .await
        .unwrap();
    let peer = budget.peer(&fixture.owner).unwrap();
    for _ in 0..CACHED_PEERS * 2 {
        budget
            .peer(&CacheOwner {
                incarnation: Uuid::new_v4(),
                ..fixture.owner.clone()
            })
            .unwrap();
    }
    assert_eq!(budget.peers.lock().len(), CACHED_PEERS);
    assert!(Arc::ptr_eq(&peer, &budget.peer(&fixture.owner).unwrap()));
    drop(held);
    settled(&budget).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn connection_outage_reconciles_old_requester_without_releasing_new_runtime() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    let owner = CacheOwner {
        endpoint: addr.to_string(),
        incarnation: Uuid::new_v4(),
    };
    let source = Arc::new(Source::default());
    let budget = TransferCompletions::default();
    let peer = budget.peer(&owner).unwrap();
    let requester = Uuid::new_v4();
    peer.state
        .lock()
        .window
        .set(source.locks.open(requester).unwrap().to_string())
        .unwrap();
    let guard = budget.reserve(&owner, requester).await.unwrap();
    let ticket = native(guard.completion.as_ref().unwrap().ticket().unwrap());
    let old = Arc::new(SealedBlock::from_slots(Vec::new()));
    let old_memory = Arc::downgrade(&old);
    source
        .locks
        .lock_blocks(ticket, vec![(StateKey::new("ns".into(), vec![1]), old)])
        .unwrap();
    drop(guard);
    tokio::time::sleep(Duration::from_secs(2)).await;
    assert_eq!(source.locks.expire(), 1);
    assert!(old_memory.upgrade().is_some());
    assert_eq!(budget.slots.available_permits(), MAX_COMPLETIONS - 1);
    let new_ticket = NativeTicket {
        window: source.locks.open(Uuid::new_v4()).unwrap(),
        slot: 0,
        generation: 1,
    };
    let new = Arc::new(SealedBlock::from_slots(Vec::new()));
    let new_memory = Arc::downgrade(&new);
    source
        .locks
        .lock_blocks(new_ticket, vec![(StateKey::new("ns".into(), vec![2]), new)])
        .unwrap();
    source.failures.store(1, Ordering::SeqCst);
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    let server = tokio::spawn(
        tonic::transport::Server::builder()
            .add_service(EngineServer::new(Service(source.clone())))
            .serve_with_incoming(TcpListenerStream::new(listener)),
    );
    settled(&budget).await;
    assert_eq!(source.releases.load(Ordering::SeqCst), 2);
    assert!(old_memory.upgrade().is_none());
    assert!(
        new_memory.upgrade().is_some(),
        "an old-runtime completion cannot close a new window"
    );
    source.locks.release(new_ticket).unwrap();
    assert!(new_memory.upgrade().is_none());
    server.abort();
}
