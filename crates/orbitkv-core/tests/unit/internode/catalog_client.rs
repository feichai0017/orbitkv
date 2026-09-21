use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicUsize};

use orbitkv_catalog::{BlockHashStore, CatalogService};
use orbitkv_proto::proto::engine::{
    self as wire,
    catalog_server::{Catalog as CatalogRpc, CatalogServer},
    sync_inventory_request::Operation,
};
use parking_lot::RwLock;
use tokio::sync::oneshot;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{Response, async_trait, transport::Server};

use super::*;
use crate::block::SealedBlock;

#[derive(Clone)]
struct Catalog {
    store: Arc<RwLock<Arc<BlockHashStore>>>,
    membership: Arc<RwLock<Option<Arc<MembershipView>>>>,
    other_shards: [Arc<BlockHashStore>; CATALOG_SHARDS],
    offline: Arc<AtomicBool>,
    lose_reply: Arc<AtomicUsize>,
    begins: Arc<AtomicUsize>,
    locates: Arc<AtomicUsize>,
    pause_page: Arc<AtomicBool>,
    page_received: Arc<Notify>,
    release_page: Arc<Notify>,
}

impl Catalog {
    fn new() -> Self {
        Self {
            membership: Arc::new(RwLock::new(None)),
            other_shards: std::array::from_fn(|_| Arc::new(BlockHashStore::new())),
            store: Arc::new(RwLock::new(Arc::new(BlockHashStore::with_config(
                orbitkv_catalog::store::StoreConfig {
                    node_stale_after: Duration::from_secs(1),
                    ..Default::default()
                },
            )))),
            offline: Arc::new(AtomicBool::new(false)),
            lose_reply: Arc::new(AtomicUsize::new(0)),
            begins: Arc::new(AtomicUsize::new(0)),
            locates: Arc::new(AtomicUsize::new(0)),
            pause_page: Arc::new(AtomicBool::new(false)),
            page_received: Arc::new(Notify::new()),
            release_page: Arc::new(Notify::new()),
        }
    }

    fn service(&self) -> Result<CatalogService, Status> {
        if self.offline.load(Ordering::Acquire) {
            return Err(Status::unavailable("injected outage"));
        }
        let mut stores = self.other_shards.clone();
        stores[0] = self.store.read().clone();
        Ok(CatalogService::new(
            stores,
            self.membership.read().as_ref().unwrap().clone(),
        ))
    }

    fn visible(&self, key: u32) -> bool {
        !self.store.read().locate_blocks("ns", &[hash(key)], "")[0]
            .replicas
            .is_empty()
    }
}

#[async_trait]
impl CatalogRpc for Catalog {
    async fn heartbeat_node(
        &self,
        request: Request<wire::HeartbeatNodeRequest>,
    ) -> Result<Response<wire::HeartbeatNodeResponse>, Status> {
        self.service()?.heartbeat_node(request).await
    }
    async fn unregister_node(
        &self,
        request: Request<wire::UnregisterNodeRequest>,
    ) -> Result<Response<wire::UnregisterNodeResponse>, Status> {
        self.service()?.unregister_node(request).await
    }
    async fn locate_blocks(
        &self,
        request: Request<wire::LocateBlocksRequest>,
    ) -> Result<Response<wire::LocateBlocksResponse>, Status> {
        self.locates.fetch_add(1, Ordering::AcqRel);
        self.service()?.locate_blocks(request).await
    }
    async fn sync_inventory(
        &self,
        request: Request<wire::SyncInventoryRequest>,
    ) -> Result<Response<wire::SyncInventoryResponse>, Status> {
        let kind = match request.get_ref().operation.as_ref() {
            Some(Operation::Begin(_)) => {
                self.begins.fetch_add(1, Ordering::AcqRel);
                0
            }
            Some(Operation::Page(_)) => 1,
            Some(Operation::Delta(_)) => 2,
            Some(Operation::Commit(_)) => 3,
            None => 0,
        };
        let result = self.service()?.sync_inventory(request).await;
        if kind == 1 && self.pause_page.swap(false, Ordering::AcqRel) {
            self.page_received.notify_one();
            self.release_page.notified().await;
        }
        if kind != 0
            && self
                .lose_reply
                .compare_exchange(kind, 0, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
        {
            return Err(Status::unavailable("reply lost after applying inventory"));
        }
        result
    }
}

struct TestServer {
    addr: SocketAddr,
    stop: oneshot::Sender<()>,
    task: tokio::task::JoinHandle<()>,
}

impl TestServer {
    async fn start(catalog: Catalog, addr: SocketAddr) -> Self {
        let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
        let addr = listener.local_addr().unwrap();
        let membership = test_view(addr, "catalog");
        *catalog.membership.write() = Some(membership.clone());
        let (stop, stopped) = oneshot::channel();
        let task = tokio::spawn(async move {
            Server::builder()
                .add_service(CatalogServer::new(catalog))
                .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async {
                    let _ = stopped.await;
                })
                .await
                .unwrap();
        });
        Self { addr, stop, task }
    }

    async fn stop(self) {
        let _ = self.stop.send(());
        tokio::time::timeout(Duration::from_secs(5), self.task)
            .await
            .unwrap()
            .unwrap();
    }
}

fn cache(journal: usize) -> Arc<ReadCache> {
    Arc::new(ReadCache::new(
        64 * 1024 * 1024,
        false,
        None,
        Some(journal * CATALOG_SHARDS),
    ))
}

fn insert(cache: &ReadCache, key: u32) {
    cache.insert_retained_for_test(
        StateKey::new("ns".into(), hash(key)),
        Arc::new(SealedBlock::from_slots(Vec::new())),
    );
}

fn hash(n: u32) -> Vec<u8> {
    for salt in 0_u32.. {
        let value = [n.to_be_bytes(), salt.to_be_bytes()].concat();
        if catalog_shard(&StateKey::new("ns".into(), value.clone())) == 0 {
            return value;
        }
    }
    unreachable!()
}

fn test_view(addr: SocketAddr, role: &str) -> Arc<MembershipView> {
    let owners = [
        (
            "catalog".to_string(),
            CacheOwner {
                endpoint: addr.to_string(),
                incarnation: Uuid::from_u128(1),
            },
        ),
        (
            "owner".to_string(),
            CacheOwner {
                endpoint: "127.0.0.1:58001".into(),
                incarnation: Uuid::from_u128(2),
            },
        ),
        (
            "requester".to_string(),
            CacheOwner {
                endpoint: "127.0.0.1:58002".into(),
                incarnation: Uuid::from_u128(3),
            },
        ),
    ];
    let owner = owners
        .iter()
        .find(|(name, _)| name == role)
        .unwrap()
        .1
        .clone();
    let view = Arc::new(MembershipView::new(
        owner,
        orbitkv_catalog::Placement::new(vec!["catalog".into()]).unwrap(),
    ));
    assert!(view.renew(std::time::Instant::now(), Duration::from_secs(3600)));
    view.replace_members(owners);
    view
}

fn client(server: &TestServer, cache: &Arc<ReadCache>) -> CatalogClient {
    CatalogClient::new(test_view(server.addr, "owner"), Arc::downgrade(cache)).unwrap()
}

async fn until(mut predicate: impl FnMut() -> bool) {
    tokio::time::timeout(Duration::from_secs(10), async {
        while !predicate() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("inventory did not converge");
}

#[tokio::test]
async fn real_grpc_restart_recovers_idle_inventory_without_new_writes() {
    let source = cache(16 * 1024);
    for key in 0..1500 {
        insert(&source, key);
    }
    let first = Catalog::new();
    let server = TestServer::start(first.clone(), "127.0.0.1:0".parse().unwrap()).await;
    let client = client(&server, &source);
    client
        .flush_with_timeout(Duration::from_secs(10))
        .await
        .unwrap();
    assert_eq!(first.store.read().owner_count(), 1500);
    assert!(first.visible(0) && first.visible(1499));
    let addr = server.addr;
    server.stop().await;
    let restarted = Catalog::new();
    let server = TestServer::start(restarted.clone(), addr).await;
    assert_ne!(
        first.store.read().catalog_epoch(),
        restarted.store.read().catalog_epoch()
    );
    until(|| restarted.visible(0) && restarted.visible(1499)).await;
    assert_eq!(restarted.store.read().owner_count(), 1500);
    client.shutdown().await;
    assert_eq!(restarted.store.read().owner_count(), 0);
    server.stop().await;
}

#[tokio::test]
async fn outage_journal_overflow_and_lost_replies_recover_without_silent_success() {
    let catalog = Catalog::new();
    let server = TestServer::start(catalog.clone(), "127.0.0.1:0".parse().unwrap()).await;
    let source = cache(300);
    insert(&source, 0);
    let client = client(&server, &source);
    client
        .flush_with_timeout(Duration::from_secs(5))
        .await
        .unwrap();
    let initial_begins = catalog.begins.load(Ordering::Acquire);
    catalog.offline.store(true, Ordering::Release);
    source.clear_for_test();
    for key in 1..100 {
        insert(&source, key);
    }
    assert!(
        client
            .flush_with_timeout(Duration::from_millis(100))
            .await
            .is_err()
    );
    catalog.offline.store(false, Ordering::Release);
    client
        .flush_with_timeout(Duration::from_secs(10))
        .await
        .unwrap();
    assert!(catalog.visible(99));
    assert!(!catalog.visible(0));
    assert!(catalog.begins.load(Ordering::Acquire) > initial_begins);
    let repaired_begins = catalog.begins.load(Ordering::Acquire);
    catalog.lose_reply.store(2, Ordering::Release);
    insert(&source, 100);
    client
        .flush_with_timeout(Duration::from_secs(5))
        .await
        .unwrap();
    assert!(catalog.visible(100));
    // The heartbeat ACK resumes an applied live delta without another snapshot.
    assert_eq!(catalog.begins.load(Ordering::Acquire), repaired_begins);
    client.shutdown().await;
    server.stop().await;
}

#[tokio::test]
async fn snapshot_overflow_and_ambiguous_page_restart_with_a_new_generation() {
    for lost_reply in [false, true] {
        let source = cache(300);
        for key in 0..1500 {
            insert(&source, key);
        }
        let catalog = Catalog::new();
        catalog.pause_page.store(true, Ordering::Release);
        if lost_reply {
            catalog.lose_reply.store(1, Ordering::Release);
        }
        let server = TestServer::start(catalog.clone(), "127.0.0.1:0".parse().unwrap()).await;
        let client = client(&server, &source);
        tokio::time::timeout(Duration::from_secs(5), catalog.page_received.notified())
            .await
            .unwrap();
        assert!(!catalog.visible(0));
        // Evict a page already accepted by the directory, then overflow its cut.
        source.clear_for_test();
        for key in 1500..1600 {
            insert(&source, key);
        }
        catalog.release_page.notify_one();
        client
            .flush_with_timeout(Duration::from_secs(10))
            .await
            .unwrap();
        assert!(catalog.begins.load(Ordering::Acquire) >= 2);
        assert_eq!(catalog.store.read().owner_count(), 100);
        assert!(!catalog.visible(0) && catalog.visible(1599));
        client.shutdown().await;
        server.stop().await;
    }
}

#[cfg(feature = "mooncake")]
#[tokio::test]
async fn discovery_coalesces_bounds_batches_and_reuses_only_positive_versioned_evidence() {
    let catalog = Catalog::new();
    let server = TestServer::start(catalog.clone(), "127.0.0.1:0".parse().unwrap()).await;
    let source = cache(64 * 1024);
    for key in 0..300 {
        insert(&source, key);
    }
    let owner = client(&server, &source);
    owner
        .flush_with_timeout(Duration::from_secs(5))
        .await
        .unwrap();
    let destination = cache(4096);
    let requester = Arc::new(
        CatalogClient::new(
            test_view(server.addr, "requester"),
            Arc::downgrade(&destination),
        )
        .unwrap(),
    );
    let hashes: Vec<_> = (0_u32..300).map(hash).collect();
    let queries = (0..16).map(|_| requester.locate_blocks("ns", &hashes));
    let responses = futures::future::join_all(queries).await;
    for response in &responses {
        let rows = response.as_ref().unwrap();
        assert_eq!(rows.len(), 300);
        assert!(
            rows.iter()
                .all(|r| r.replicas.len() == 1 && r.replicas[0].owner.incarnation == owner.node_id)
        );
    }
    assert_eq!(catalog.locates.load(Ordering::Acquire), 3);
    catalog.offline.store(true, Ordering::Release);
    assert_eq!(
        requester.locate_blocks("ns", &hashes).await.unwrap().len(),
        300
    );
    assert_eq!(catalog.locates.load(Ordering::Acquire), 3);
    let mut extended = hashes.clone();
    extended.push(hash(999));
    let partial = requester.locate_blocks("ns", &extended).await.unwrap();
    assert!(partial[..300].iter().all(|row| !row.replicas.is_empty()));
    assert!(partial[300].replicas.is_empty());
    assert_eq!(catalog.locates.load(Ordering::Acquire), 4);
    catalog.offline.store(false, Ordering::Release);
    let old = responses[0].as_ref().unwrap()[0].clone();
    source.clear_for_test();
    insert(&source, 0);
    owner
        .flush_with_timeout(Duration::from_secs(5))
        .await
        .unwrap();
    requester.reject_candidate(&old.key, &old.replicas[0]);
    let current = requester.locate_blocks("ns", &hashes[..1]).await.unwrap();
    assert!(current[0].replicas[0].sequence > old.replicas[0].sequence);
    requester.reject_candidate(&old.key, &old.replicas[0]);
    assert_eq!(
        requester.locate_blocks("ns", &hashes[..1]).await.unwrap(),
        current
    );
    assert_eq!(catalog.locates.load(Ordering::Acquire), 5);
    let missing = vec![hash(999)];
    assert!(
        requester.locate_blocks("ns", &missing).await.unwrap()[0]
            .replicas
            .is_empty()
    );
    insert(&source, 999);
    owner
        .flush_with_timeout(Duration::from_secs(5))
        .await
        .unwrap();
    assert_eq!(
        requester.locate_blocks("ns", &missing).await.unwrap()[0]
            .replicas
            .len(),
        1
    );
    assert_eq!(catalog.locates.load(Ordering::Acquire), 7);
    // The requester never discovers itself, and namespaces never share evidence.
    assert!(
        owner.locate_blocks("ns", &hashes[..1]).await.unwrap()[0]
            .replicas
            .is_empty()
    );
    assert!(
        requester
            .locate_blocks("different-model", &hashes[..1])
            .await
            .unwrap()[0]
            .replicas
            .is_empty()
    );
    requester.shutdown().await;
    owner.shutdown().await;
    server.stop().await;
}

#[tokio::test]
async fn two_embedded_catalogs_partition_inventory_and_repair_restarted_member() {
    let catalogs = [Catalog::new(), Catalog::new()];
    let server_a = TestServer::start(catalogs[0].clone(), "127.0.0.1:0".parse().unwrap()).await;
    let server_b = TestServer::start(catalogs[1].clone(), "127.0.0.1:0".parse().unwrap()).await;
    let placement = orbitkv_catalog::Placement::new(vec!["a".into(), "b".into()]).unwrap();
    let mut members = vec![
        (
            "a".into(),
            CacheOwner {
                endpoint: server_a.addr.to_string(),
                incarnation: Uuid::new_v4(),
            },
        ),
        (
            "b".into(),
            CacheOwner {
                endpoint: server_b.addr.to_string(),
                incarnation: Uuid::new_v4(),
            },
        ),
        (
            "source".into(),
            CacheOwner {
                endpoint: "127.0.0.1:59001".into(),
                incarnation: Uuid::new_v4(),
            },
        ),
    ];
    let views: Vec<_> = members
        .iter()
        .map(|(_, owner)| {
            let view = Arc::new(MembershipView::new(owner.clone(), placement.clone()));
            assert!(view.renew(std::time::Instant::now(), Duration::from_secs(300)));
            view.replace_members(members.clone());
            view
        })
        .collect();
    for (catalog, view) in catalogs.iter().zip(&views) {
        *catalog.membership.write() = Some(view.clone());
    }
    let source = cache(4096);
    let keys: Vec<_> = (0_u32..256)
        .map(|n| StateKey::new("ns".into(), n.to_be_bytes().to_vec()))
        .collect();
    for key in &keys {
        source.insert_retained_for_test(key.clone(), Arc::new(SealedBlock::from_slots(Vec::new())));
    }
    let client = CatalogClient::new(views[2].clone(), Arc::downgrade(&source)).unwrap();
    client
        .flush_with_timeout(Duration::from_secs(10))
        .await
        .unwrap();
    #[cfg(feature = "mooncake")]
    {
        let empty = cache(4096);
        let query = CatalogClient::new(views[0].clone(), Arc::downgrade(&empty)).unwrap();
        let hashes: Vec<_> = keys.iter().map(|key| key.hash.clone()).collect();
        let rows = query.locate_blocks("ns", &hashes).await.unwrap();
        assert_eq!(rows.len(), keys.len());
        assert!(
            rows.iter()
                .zip(&keys)
                .all(|(row, key)| row.key == *key && row.replicas.len() == 1)
        );
        let calls: usize = catalogs
            .iter()
            .map(|catalog| catalog.locates.load(Ordering::Acquire))
            .sum();
        assert_eq!(calls, CATALOG_SHARDS);
        assert_eq!(query.locate_blocks("ns", &hashes).await.unwrap(), rows);
        assert_eq!(
            catalogs
                .iter()
                .map(|catalog| catalog.locates.load(Ordering::Acquire))
                .sum::<usize>(),
            calls
        );
        query.shutdown().await;
    }
    let stores = |catalog: &Catalog| {
        let mut stores = catalog.other_shards.clone();
        stores[0] = catalog.store.read().clone();
        stores
    };
    for key in &keys {
        let shard = catalog_shard(key);
        for (index, node) in ["a", "b"].iter().enumerate() {
            let rows = stores(&catalogs[index])[shard].locate_blocks(
                "ns",
                std::slice::from_ref(&key.hash),
                "",
            );
            assert_eq!(
                rows[0].replicas.len(),
                usize::from(placement.host(shard) == Some(node))
            );
        }
    }
    // A shard request to a valid but unassigned Manager must be rejected.
    let shard_b = (0..CATALOG_SHARDS)
        .find(|&shard| placement.host(shard) == Some("b"))
        .unwrap();
    let mut wrong = connect(views[0].owner()).unwrap();
    let request = HeartbeatNodeRequest {
        route: Some(route(&views[0], shard_b, views[0].owner())),
        node: views[2].owner().endpoint.clone(),
        node_id: views[2].owner().incarnation.to_string(),
    };
    assert_eq!(
        wrong.heartbeat_node(request).await.unwrap_err().code(),
        tonic::Code::FailedPrecondition
    );

    // Liveness loss does not move B's shards to A or report flush success.
    views[2].replace_members([members[0].clone(), members[2].clone()]);
    assert!(
        client
            .flush_with_timeout(Duration::from_millis(100))
            .await
            .is_err()
    );
    let restart = Catalog::new();
    let addr = server_b.addr;
    server_b.stop().await;
    let server_b = TestServer::start(restart.clone(), addr).await;
    members[1].1.incarnation = Uuid::new_v4();
    let replacement = Arc::new(MembershipView::new(members[1].1.clone(), placement));
    assert!(replacement.renew(std::time::Instant::now(), Duration::from_secs(300)));
    replacement.replace_members(members.clone());
    *restart.membership.write() = Some(replacement);
    for view in &views {
        view.replace_members(members.clone());
    }
    client
        .flush_with_timeout(Duration::from_secs(10))
        .await
        .unwrap();
    for key in &keys {
        let shard = catalog_shard(key);
        if views[2].placement().host(shard) == Some("b") {
            assert_eq!(
                stores(&restart)[shard].locate_blocks("ns", std::slice::from_ref(&key.hash), "")[0]
                    .replicas
                    .len(),
                1
            );
        }
    }
    client.shutdown().await;
    server_a.stop().await;
    server_b.stop().await;
}
