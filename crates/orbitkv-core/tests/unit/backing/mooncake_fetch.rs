use super::*;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

fn test_allocate_fn(calls: Arc<AtomicUsize>) -> AllocateFn {
    let allocator = Arc::new(crate::pinned_pool::PinnedAllocator::new_global(
        32 * 1024 * 1024,
        1,
        false,
        false,
        None,
    ));
    Arc::new(move |size, _numa| {
        calls.fetch_add(1, Ordering::Relaxed);
        allocator.allocate(NonZeroU64::new(size)?, NumaNode::UNKNOWN)
    })
}

fn remaining(bytes: u64) -> HashMap<NumaNode, u64> {
    HashMap::from([(NumaNode(0), bytes)])
}

#[test]
fn chunked_slabs_bump_within_chunk_then_refill() {
    let calls = Arc::new(AtomicUsize::new(0));
    let allocate_fn = test_allocate_fn(Arc::clone(&calls));
    let mut slabs = ChunkedSlabs::new(&allocate_fn, 1024, remaining(1536));

    let (p1, a1) = slabs.alloc_segment(NumaNode(0), 512, "K").expect("first");
    let (p2, _a2) = slabs.alloc_segment(NumaNode(0), 512, "V").expect("second");
    assert_eq!(p2.as_ptr() as usize - p1.as_ptr() as usize, 512);
    assert_eq!(calls.load(Ordering::Relaxed), 1);

    // Third segment exceeds the current chunk: a fresh chunk is allocated
    // while earlier segments stay valid through their own chunk Arc.
    let (_p3, a3) = slabs.alloc_segment(NumaNode(0), 512, "K").expect("third");
    assert_eq!(calls.load(Ordering::Relaxed), 2);
    assert_eq!(slabs.chunk_count, 2);
    assert!(!Arc::ptr_eq(&a1, &a3));
}

#[test]
fn chunked_slabs_oversized_segment_gets_dedicated_chunk() {
    let calls = Arc::new(AtomicUsize::new(0));
    let allocate_fn = test_allocate_fn(Arc::clone(&calls));
    let mut slabs = ChunkedSlabs::new(&allocate_fn, 1024, remaining(4096));

    slabs
        .alloc_segment(NumaNode(0), 4096, "K")
        .expect("oversized segment");
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(slabs.chunk_count, 1);
}

#[test]
fn chunked_slabs_allocation_failure_is_an_error() {
    let allocate_fn: AllocateFn = Arc::new(|_, _| None);
    let mut slabs = ChunkedSlabs::new(&allocate_fn, 1024, remaining(512));

    let err = match slabs.alloc_segment(NumaNode(0), 512, "K") {
        Ok(_) => panic!("allocation should fail"),
        Err(err) => err,
    };
    assert!(err.contains("failed to allocate fetch chunk"));
}

#[test]
fn chunked_slabs_chunk_clamped_to_batch_remaining() {
    // A small fetch must not request the whole chunk_bytes cap — that
    // fails outright on pools smaller than the cap (jz p2p IT regression).
    let sizes = Arc::new(Mutex::new(Vec::new()));
    let recorded = Arc::clone(&sizes);
    let inner = test_allocate_fn(Arc::new(AtomicUsize::new(0)));
    let allocate_fn: AllocateFn = Arc::new(move |size, numa| {
        recorded.lock().unwrap().push(size);
        inner(size, numa)
    });
    let mut slabs = ChunkedSlabs::new(&allocate_fn, 256 << 20, remaining(4096));

    slabs.alloc_segment(NumaNode(0), 1024, "K").expect("first");
    slabs.alloc_segment(NumaNode(0), 3072, "V").expect("second");

    // One chunk sized to the batch total, not to the 256 MiB cap.
    assert_eq!(*sizes.lock().unwrap(), vec![4096]);
    assert_eq!(slabs.chunk_count, 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn cancellation_during_authorization_still_releases_the_returned_session() {
    use orbitkv_proto::proto::engine::{
        self as wire,
        engine_server::{Engine, EngineServer},
    };
    use tokio::sync::Notify;
    use tokio_stream::wrappers::TcpListenerStream;
    use tonic::{Request, Response, Status};

    struct DelayedSource {
        started: Arc<Notify>,
        reply: Arc<Notify>,
        released: Arc<Notify>,
    }
    #[tonic::async_trait]
    impl Engine for DelayedSource {
        async fn query_blocks_for_transfer(
            &self,
            _: Request<QueryBlocksForTransferRequest>,
        ) -> Result<Response<QueryBlocksForTransferResponse>, Status> {
            self.started.notify_one();
            self.reply.notified().await;
            Ok(Response::new(QueryBlocksForTransferResponse {
                status: Some(wire::ResponseStatus {
                    ok: true,
                    message: String::new(),
                }),
                transfer_session_id: "late-session".into(),
                ..Default::default()
            }))
        }
        async fn release_transfer_lock(
            &self,
            request: Request<wire::ReleaseTransferLockRequest>,
        ) -> Result<Response<wire::ReleaseTransferLockResponse>, Status> {
            assert_eq!(request.get_ref().transfer_session_id, "late-session");
            self.released.notify_one();
            Ok(Response::new(wire::ReleaseTransferLockResponse {
                status: Some(wire::ResponseStatus {
                    ok: true,
                    message: String::new(),
                }),
                released_blocks: 1,
            }))
        }
        async fn health(
            &self,
            _: Request<wire::HealthRequest>,
        ) -> Result<Response<wire::HealthResponse>, Status> {
            Err(Status::unimplemented("unused"))
        }
    }
    let started = Arc::new(Notify::new());
    let reply = Arc::new(Notify::new());
    let released = Arc::new(Notify::new());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(
        tonic::transport::Server::builder()
            .add_service(EngineServer::new(DelayedSource {
                started: started.clone(),
                reply: reply.clone(),
                released: released.clone(),
            }))
            .serve_with_incoming(TcpListenerStream::new(listener)),
    );
    let task = tokio::spawn(async move {
        let channels = parking_lot::Mutex::new(LinkedHashMap::new());
        let completions = Arc::new(TransferCompletions::default());
        let segment = FetchSegment {
            owner: orbitkv_state::CacheOwner {
                endpoint: addr.to_string(),
                incarnation: uuid::Uuid::new_v4(),
            },
            records: vec![orbitkv_state::InventoryRecord {
                key: StateKey::new("ns".into(), vec![1]),
                sequence: 1,
                present: true,
            }],
        };
        let _ = query_remote_blocks(&channels, &completions, &segment, "requester").await;
    });
    tokio::time::timeout(Duration::from_secs(2), started.notified())
        .await
        .unwrap();
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    reply.notify_one();
    tokio::time::timeout(Duration::from_secs(2), released.notified())
        .await
        .unwrap();
    server.abort();
}
