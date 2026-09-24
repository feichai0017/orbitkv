//! P2P Mooncake remote fetch integration test.
//!
//! Verifies the end-to-end flow:
//! Engine A saves blocks → Catalog discovers them → Engine B fetches via Mooncake READ
//! → data integrity verified.
//!
//! Run with: `cargo test -p orbitkv-server --test p2p_mooncake -- --ignored`

use std::ffi::c_void;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use cudarc::driver::CudaContext;
use cudarc::driver::sys;
use orbitkv_catalog::{BlockHashStore, CatalogService, MembershipView, Placement};
use orbitkv_core::*;
use orbitkv_proto::proto::engine::{
    OpenTransferWindowRequest, QueryBlocksForTransferRequest, ReleaseTransferLockRequest,
    TransferTicket, catalog_server::CatalogServer, engine_client::EngineClient,
};
use orbitkv_server::proto::engine::engine_server::EngineServer;
use orbitkv_state::group_hash;
use orbitkv_state::{BlockCandidates, CATALOG_SHARDS, StateKey, catalog_shard};
use tonic::transport::Server;

// ── GPU buffer (from crates/orbitkv-core/tests/common/gpu_buffer.rs) ──────────────

struct GpuBuffer {
    ptr: sys::CUdeviceptr,
    len: usize,
}

impl GpuBuffer {
    fn alloc(len: usize) -> Self {
        assert!(len > 0);
        let mut ptr: sys::CUdeviceptr = 0;
        check_cuda(
            unsafe { sys::cuMemAlloc_v2(&raw mut ptr, len) },
            "cuMemAlloc_v2",
        );
        Self { ptr, len }
    }

    fn as_u64(&self) -> u64 {
        self.ptr
    }

    fn copy_from_host(&self, data: &[u8]) {
        assert_eq!(data.len(), self.len);
        check_cuda(
            unsafe { sys::cuMemcpyHtoD_v2(self.ptr, data.as_ptr() as *const c_void, self.len) },
            "cuMemcpyHtoD_v2",
        );
    }

    fn copy_to_host(&self) -> Vec<u8> {
        let mut output = vec![0u8; self.len];
        check_cuda(
            unsafe { sys::cuMemcpyDtoH_v2(output.as_mut_ptr() as *mut c_void, self.ptr, self.len) },
            "cuMemcpyDtoH_v2",
        );
        output
    }

    fn zero(&self) {
        check_cuda(
            unsafe { sys::cuMemsetD8_v2(self.ptr, 0, self.len) },
            "cuMemsetD8_v2",
        );
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        if self.ptr != 0 {
            check_cuda(unsafe { sys::cuMemFree_v2(self.ptr) }, "cuMemFree_v2");
            self.ptr = 0;
        }
    }
}

fn check_cuda(result: sys::CUresult, op: &str) {
    assert!(
        result == sys::CUresult::CUDA_SUCCESS,
        "{op} failed with {result:?}"
    );
}

// ── Helpers (from crates/orbitkv-core/tests/common/helpers.rs) ────────────────────

fn fill_test_pattern(host_data: &mut [u8], block_size: usize) {
    for (i, block) in host_data.chunks_exact_mut(block_size).enumerate() {
        let fill = ((i % 251) + 1) as u8;
        block.fill(fill);
    }
}

fn make_block_ids(num_blocks: usize) -> Vec<usize> {
    (0..num_blocks).collect()
}

fn make_block_hashes(num_blocks: usize, salt: u8) -> Vec<Vec<u8>> {
    (0..num_blocks)
        .map(|idx| {
            let mut hash = Vec::with_capacity(5);
            hash.push(salt);
            hash.extend_from_slice(&(idx as u32).to_le_bytes());
            hash
        })
        .collect()
}

// ── Infrastructure ──────────────────────────────────────────────────────────

fn get_free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .expect("bind ephemeral port")
        .local_addr()
        .expect("local_addr")
        .port()
}

async fn wait_for_grpc_ready(port: u16) {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if tokio::net::TcpStream::connect(format!("127.0.0.1:{port}"))
            .await
            .is_ok()
        {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for gRPC on port {port}"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

async fn spawn_engine_server(
    engine: Arc<OrbitKVEngine>,
    port: u16,
    view: Arc<MembershipView>,
) -> [Arc<BlockHashStore>; CATALOG_SHARDS] {
    let stores = std::array::from_fn(|_| Arc::new(BlockHashStore::new()));
    let catalog = CatalogService::new(stores.clone(), view);
    let service = P2pTransferService::new(engine);
    let addr: SocketAddr = ([127, 0, 0, 1], port).into();
    tokio::spawn(async move {
        Server::builder()
            .add_service(EngineServer::new(service))
            .add_service(CatalogServer::new(catalog))
            .serve(addr)
            .await
            .expect("peer services");
    });
    wait_for_grpc_ready(port).await;
    stores
}

fn locate(
    stores: &[Arc<BlockHashStore>; CATALOG_SHARDS],
    namespace: &str,
    hashes: &[Vec<u8>],
    exclude: &str,
) -> Vec<BlockCandidates> {
    hashes
        .iter()
        .map(|hash| {
            let shard = catalog_shard(&StateKey::new(namespace.into(), hash.clone()));
            stores[shard]
                .locate_blocks(namespace, std::slice::from_ref(hash), exclude)
                .remove(0)
        })
        .collect()
}

async fn wait_for_cache(
    engine: &OrbitKVEngine,
    instance_id: &str,
    block_hashes: &[Vec<u8>],
    expected_hit: usize,
    timeout: Duration,
) {
    let deadline = Instant::now() + timeout;
    loop {
        let status = engine
            .count_prefix_hit_blocks_with_prefetch(
                instance_id,
                "wait-for-cache",
                block_hashes,
                orbitkv_core::QueryMode::Demand,
            )
            .await
            .expect("count_prefix_hit_blocks_with_prefetch");
        let hit = {
            let QueryResult { blocks, .. } = status;
            blocks.len()
        };
        if hit >= expected_hit {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for {expected_hit} cached blocks (got {hit})"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

async fn wait_for_catalog_registration(
    store: &[Arc<BlockHashStore>; CATALOG_SHARDS],
    namespace: &str,
    hashes: &[Vec<u8>],
    expected: usize,
    timeout: Duration,
) {
    let deadline = Instant::now() + timeout;
    loop {
        let found = locate(store, namespace, hashes, "");
        let count = found
            .iter()
            .take_while(|row| !row.replicas.is_empty())
            .count();
        if count >= expected {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for Catalog registration ({} / {})",
            count,
            expected
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

async fn wait_for_catalog_ownership(
    store: &[Arc<BlockHashStore>; CATALOG_SHARDS],
    namespace: &str,
    hashes: &[Vec<u8>],
    node: &str,
    expected: usize,
    timeout: Duration,
) {
    let deadline = Instant::now() + timeout;
    loop {
        let found = locate(store, namespace, hashes, "");
        let owned = found
            .iter()
            .filter(|entry| entry.replicas.iter().any(|r| r.owner.endpoint == node))
            .count();
        if owned >= expected {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for Catalog ownership by {node} ({owned} / {expected})"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn ib_device() -> String {
    std::env::var("ORBITKV_IB_DEVICE").unwrap_or_else(|_| "mlx5_1".into())
}

fn mooncake_nics() -> Vec<String> {
    if std::env::var_os("MC_FORCE_TCP").is_some() {
        Vec::new()
    } else {
        vec![ib_device()]
    }
}

// ── Test ────────────────────────────────────────────────────────────────────

const NUM_BLOCKS: usize = 4;
const BLOCK_SIZE: usize = 1024;
const TOTAL_SIZE: usize = NUM_BLOCKS * BLOCK_SIZE;
const NAMESPACE: &str = "test-p2p";
const LAYER: &str = "layer_0";
const DEVICE_ID: i32 = 0;

#[tokio::test]
#[ignore = "requires CUDA and Mooncake; set MC_FORCE_TCP=1 for the same-host TCP gate"]
async fn p2p_mooncake_remote_fetch_roundtrip() {
    orbitkv_common::logging::init_stdout_colored("debug");
    let _cuda_ctx = CudaContext::new(0).expect("CUDA init");

    // Allocate ephemeral ports
    let port_a = get_free_port();

    // ── 2. Create Engine A (source of blocks) ──
    let membership_a = Arc::new(MembershipView::new(
        orbitkv_state::CacheOwner {
            endpoint: format!("127.0.0.1:{port_a}"),
            incarnation: uuid::Uuid::new_v4(),
        },
        Placement::new(vec!["a".into()]).unwrap(),
    ));
    membership_a.replace_members([("a".into(), membership_a.owner().clone())]);
    assert!(membership_a.renew(Instant::now(), Duration::from_secs(300)));
    let config_a = StorageConfig {
        membership: Some(membership_a.clone()),
        mooncake_nic_names: mooncake_nics(),
        transfer_budget_bytes: Some(TOTAL_SIZE),
        transfer_lock_timeout: Duration::ZERO,
        ..StorageConfig::default()
    };
    let engine_a = Arc::new(
        OrbitKVEngine::new_with_config(16 << 20, false, config_a).expect("engine A should start"),
    );

    // ── 3. Start Engine A gRPC server ──
    let stores = spawn_engine_server(Arc::clone(&engine_a), port_a, membership_a.clone()).await;

    // ── 4. Save blocks on Engine A ──
    let gpu_a = GpuBuffer::alloc(TOTAL_SIZE);
    let mut host_data = vec![0u8; TOTAL_SIZE];
    fill_test_pattern(&mut host_data, BLOCK_SIZE);
    gpu_a.copy_from_host(&host_data);

    engine_a
        .register_context_layer_batch(
            "inst-a",
            NAMESPACE,
            DEVICE_ID,
            0, // tp_rank
            0, // pp_rank
            1, // tp_size
            1, // world_size
            &[LAYER.to_string()],
            &[gpu_a.as_u64()],
            &[TOTAL_SIZE],
            &[NUM_BLOCKS],
            &[BLOCK_SIZE],
            &[0], // kv_strides
            &[1], // segments
            TransferMode::Direct,
            false,
        )
        .expect("register layer on engine A");

    let cache_namespace = engine_a
        .instance_namespace("inst-a")
        .expect("sealed namespace");
    let block_ids = make_block_ids(NUM_BLOCKS);
    let block_hashes = make_block_hashes(NUM_BLOCKS, 42);
    let stored_hashes: Vec<_> = block_hashes
        .iter()
        .map(|hash| group_hash(hash, 0))
        .collect();

    engine_a
        .batch_save_kv_blocks_from_ipc(
            "inst-a",
            0,
            0,
            DEVICE_ID,
            vec![LayerSave {
                layer_name: LAYER.to_string(),
                block_ids: block_ids.clone(),
                block_hashes: block_hashes.clone(),
            }],
        )
        .await
        .expect("save blocks on engine A");

    // ── 5. Wait for Engine A cache ──
    wait_for_cache(
        &engine_a,
        "inst-a",
        &block_hashes,
        NUM_BLOCKS,
        Duration::from_secs(5),
    )
    .await;

    // ── 6. Wait for acknowledged owner inventory ──
    engine_a
        .flush_saves_and_inventory()
        .await
        .expect("publish inventory");
    wait_for_catalog_registration(
        &stores,
        &cache_namespace,
        &stored_hashes,
        NUM_BLOCKS,
        Duration::from_secs(10),
    )
    .await;

    // Source authorization fences both restarts and individual residency episodes.
    let evidence = locate(&stores, &cache_namespace, &stored_hashes, "requester");
    let mut peer = EngineClient::connect(format!("http://127.0.0.1:{port_a}"))
        .await
        .unwrap();
    let window = peer
        .open_transfer_window(OpenTransferWindowRequest {
            owner_incarnation: evidence[0].replicas[0].owner.incarnation.to_string(),
            requester_incarnation: uuid::Uuid::new_v4().to_string(),
        })
        .await
        .unwrap()
        .into_inner()
        .window_id;
    let ticket = TransferTicket {
        window_id: window,
        slot: 0,
        generation: 1,
    };
    let authorization = QueryBlocksForTransferRequest {
        namespace: cache_namespace.clone(),
        block_hashes: stored_hashes.clone(),
        ticket: Some(ticket.clone()),
        owner_incarnation: evidence[0].replicas[0].owner.incarnation.to_string(),
        residency_sequences: evidence.iter().map(|r| r.replicas[0].sequence).collect(),
    };
    assert!(membership_a.renew(Instant::now(), Duration::from_secs(300)));
    let mut stale_runtime = authorization.clone();
    stale_runtime.owner_incarnation = uuid::Uuid::new_v4().to_string();
    let mut stale_residency = authorization.clone();
    stale_residency.residency_sequences[0] += 1;
    for stale in [stale_runtime, stale_residency] {
        assert_eq!(
            peer.query_blocks_for_transfer(stale)
                .await
                .unwrap_err()
                .code(),
            tonic::Code::FailedPrecondition
        );
    }
    let granted = peer
        .query_blocks_for_transfer(authorization.clone())
        .await
        .unwrap()
        .into_inner();
    assert_eq!(granted.blocks.len(), NUM_BLOCKS);
    assert_eq!(engine_a.expire_transfer_locks(), 1);
    assert_eq!(engine_a.expire_transfer_locks(), 0);
    assert_eq!(
        peer.query_blocks_for_transfer(authorization.clone())
            .await
            .unwrap_err()
            .code(),
        tonic::Code::FailedPrecondition
    );
    let mut concurrent = authorization.clone();
    concurrent.ticket = Some(TransferTicket {
        slot: 1,
        ..ticket.clone()
    });
    assert_eq!(
        peer.query_blocks_for_transfer(concurrent)
            .await
            .unwrap_err()
            .code(),
        tonic::Code::ResourceExhausted
    );
    peer.release_transfer_lock(ReleaseTransferLockRequest {
        ticket: Some(ticket.clone()),
    })
    .await
    .unwrap();
    // Release-before-authorize must fence a queued RPC without ever pinning data.
    let cancelled = TransferTicket {
        generation: 2,
        ..ticket.clone()
    };
    peer.release_transfer_lock(ReleaseTransferLockRequest {
        ticket: Some(cancelled.clone()),
    })
    .await
    .unwrap();
    let mut queued = authorization.clone();
    queued.ticket = Some(cancelled);
    assert_eq!(
        peer.query_blocks_for_transfer(queued)
            .await
            .unwrap_err()
            .code(),
        tonic::Code::FailedPrecondition
    );
    for invalid in [
        None,
        Some(TransferTicket {
            slot: 64,
            ..ticket.clone()
        }),
        Some(TransferTicket {
            generation: 0,
            ..ticket.clone()
        }),
        Some(TransferTicket {
            window_id: "bad-id".into(),
            ..ticket.clone()
        }),
    ] {
        let mut query = authorization.clone();
        query.ticket = invalid.clone();
        assert_eq!(
            peer.query_blocks_for_transfer(query)
                .await
                .unwrap_err()
                .code(),
            tonic::Code::InvalidArgument
        );
        assert_eq!(
            peer.release_transfer_lock(ReleaseTransferLockRequest { ticket: invalid })
                .await
                .unwrap_err()
                .code(),
            tonic::Code::InvalidArgument
        );
    }

    // ── 7. Create Engine B (fetcher) ──
    let port_b = get_free_port();
    let membership_b = Arc::new(MembershipView::new(
        orbitkv_state::CacheOwner {
            endpoint: format!("127.0.0.1:{port_b}"),
            incarnation: uuid::Uuid::new_v4(),
        },
        Placement::new(vec!["a".into()]).unwrap(),
    ));
    let members = [
        ("a".into(), membership_a.owner().clone()),
        ("b".into(), membership_b.owner().clone()),
    ];
    membership_a.replace_members(members.clone());
    membership_b.replace_members(members);
    assert!(membership_b.renew(Instant::now(), Duration::from_secs(300)));
    let config_b = StorageConfig {
        membership: Some(membership_b.clone()),
        mooncake_nic_names: mooncake_nics(),
        ..StorageConfig::default()
    };
    let engine_b =
        OrbitKVEngine::new_with_config(16 << 20, false, config_b).expect("engine B should start");

    let gpu_b = GpuBuffer::alloc(TOTAL_SIZE);
    gpu_b.zero();

    engine_b
        .register_context_layer_batch(
            "inst-b",
            NAMESPACE,
            DEVICE_ID,
            0, // tp_rank
            0, // pp_rank
            1, // tp_size
            1, // world_size
            &[LAYER.to_string()],
            &[gpu_b.as_u64()],
            &[TOTAL_SIZE],
            &[NUM_BLOCKS],
            &[BLOCK_SIZE],
            &[0],
            &[1],
            TransferMode::Direct,
            false,
        )
        .expect("register layer on engine B");

    // ── 8. Start the remote query before the producer registers the blocks ──
    let delayed_hashes = make_block_hashes(NUM_BLOCKS, 43);
    let stored_delayed: Vec<_> = delayed_hashes
        .iter()
        .map(|hash| group_hash(hash, 0))
        .collect();
    let mut waiting = Box::pin(engine_b.count_prefix_hit_blocks_with_prefetch(
        "inst-b",
        "req-wait-for-producer",
        &delayed_hashes,
        orbitkv_core::QueryMode::WaitForFullPrefix,
    ));
    assert!(
        tokio::time::timeout(Duration::from_millis(20), &mut waiting)
            .await
            .is_err()
    );

    engine_a
        .batch_save_kv_blocks_from_ipc(
            "inst-a",
            0,
            0,
            DEVICE_ID,
            vec![LayerSave {
                layer_name: LAYER.to_string(),
                block_ids: block_ids.clone(),
                block_hashes: delayed_hashes.clone(),
            }],
        )
        .await
        .expect("save delayed blocks on engine A");

    wait_for_catalog_registration(
        &stores,
        &cache_namespace,
        &stored_delayed,
        NUM_BLOCKS,
        Duration::from_secs(10),
    )
    .await;

    // ── 9. Engine B observes the producer and fetches via Mooncake READ ──
    let result = tokio::time::timeout(Duration::from_secs(30), waiting)
        .await
        .expect("remote fetch timeout")
        .expect("remote fetch");
    assert_eq!(result.blocks.len(), NUM_BLOCKS);
    let lease = engine_b
        .create_query_lease("inst-b", result.blocks)
        .expect("lease");

    // ── 9b. Verify Engine B re-registered fetched blocks to Catalog ──
    // Mooncake-fetched blocks are now resident on B, so B must advertise them so
    // other nodes can discover and fetch from B (not just from A).
    engine_b
        .flush_saves_and_inventory()
        .await
        .expect("restored inventory");
    wait_for_catalog_ownership(
        &stores,
        &cache_namespace,
        &stored_delayed,
        &format!("127.0.0.1:{port_b}"),
        NUM_BLOCKS,
        Duration::from_secs(10),
    )
    .await;

    // Eviction removes A's evidence while the copied replica on B remains usable.
    assert!(engine_a.cleanup_memory_cache().evicted_blocks > 0);
    engine_a
        .flush_saves_and_inventory()
        .await
        .expect("evicted inventory");
    let remaining = locate(&stores, &cache_namespace, &stored_delayed, "");
    assert_eq!(remaining.len(), NUM_BLOCKS);
    for entry in remaining {
        assert_eq!(
            entry
                .replicas
                .into_iter()
                .map(|r| r.owner.endpoint)
                .collect::<Vec<_>>(),
            vec![format!("127.0.0.1:{port_b}")]
        );
    }
    assert!(
        locate(&stores, NAMESPACE, &delayed_hashes, "")
            .iter()
            .all(|row| row.replicas.is_empty())
    );

    assert_eq!(
        peer.query_blocks_for_transfer(authorization.clone())
            .await
            .unwrap_err()
            .code(),
        tonic::Code::FailedPrecondition
    );
    engine_a
        .batch_save_kv_blocks_from_ipc(
            "inst-a",
            0,
            0,
            DEVICE_ID,
            vec![LayerSave {
                layer_name: LAYER.into(),
                block_ids: block_ids.clone(),
                block_hashes: block_hashes.clone(),
            }],
        )
        .await
        .unwrap();
    engine_a.flush_saves_and_inventory().await.unwrap();
    assert_eq!(
        peer.query_blocks_for_transfer(authorization)
            .await
            .unwrap_err()
            .code(),
        tonic::Code::FailedPrecondition
    );

    // ── 10. Load from Engine B cache → GPU ──
    let fresh = locate(&stores, &cache_namespace, &stored_hashes, "requester");
    let fresh_authorization = QueryBlocksForTransferRequest {
        namespace: cache_namespace.clone(),
        block_hashes: stored_hashes.clone(),
        ticket: Some(TransferTicket {
            generation: 3,
            ..ticket.clone()
        }),
        owner_incarnation: membership_a.owner().incarnation.to_string(),
        residency_sequences: fresh
            .iter()
            .map(|row| {
                row.replicas
                    .iter()
                    .find(|replica| replica.owner == *membership_a.owner())
                    .unwrap()
                    .sequence
            })
            .collect(),
    };
    peer.query_blocks_for_transfer(fresh_authorization.clone())
        .await
        .unwrap()
        .into_inner();
    membership_a.fence();
    membership_b.fence();
    assert_eq!(
        peer.query_blocks_for_transfer(fresh_authorization)
            .await
            .unwrap_err()
            .code(),
        tonic::Code::FailedPrecondition
    );
    peer.release_transfer_lock(ReleaseTransferLockRequest {
        ticket: Some(TransferTicket {
            generation: 3,
            ..ticket
        }),
    })
    .await
    .unwrap();
    let resident = engine_b
        .count_prefix_hit_blocks_with_prefetch(
            "inst-b",
            "fenced-local-hit",
            &delayed_hashes,
            orbitkv_core::QueryMode::Demand,
        )
        .await
        .unwrap();
    assert_eq!(
        resident.blocks.len(),
        NUM_BLOCKS,
        "local hits survive membership loss"
    );
    let remote = engine_b
        .count_prefix_hit_blocks_with_prefetch(
            "inst-b",
            "fenced-remote-miss",
            &block_hashes,
            orbitkv_core::QueryMode::Demand,
        )
        .await
        .unwrap();
    assert!(
        remote.blocks.is_empty(),
        "fenced requester must not fetch remote-only blocks"
    );

    let receiver = engine_b
        .restore(
            "inst-b",
            0,
            DEVICE_ID,
            &[vec![LAYER]],
            &[(lease, vec![block_ids.iter().copied().map(Some).collect()])],
        )
        .expect("batch_load on engine B");

    tokio::time::timeout(Duration::from_secs(5), receiver)
        .await
        .expect("restore timeout")
        .expect("restore worker disappeared")
        .result
        .expect("restore failed");

    // ── 11. Verify data integrity ──
    let loaded = gpu_b.copy_to_host();
    assert_eq!(
        loaded, host_data,
        "GPU data mismatch: remote-fetched blocks differ from original"
    );
}

#[tokio::test]
#[ignore = "requires CUDA, nvCOMP 5.3 and Mooncake; set MC_FORCE_TCP=1 for same-host TCP"]
async fn encoded_peer_payloads_restore_the_same_gpu_image() {
    use orbitkv_state::{AttentionRole, Scalar16, StorageFormat};
    let _cuda = CudaContext::new(0).unwrap();
    for codec in [
        StorageCodec::Ans,
        StorageCodec::Fp8,
        StorageCodec::TurboQuant4,
        StorageCodec::TurboQuant3,
    ] {
        let port_a = get_free_port();
        let port_b = get_free_port();
        let make_view = |port| {
            Arc::new(MembershipView::new(
                orbitkv_state::CacheOwner {
                    endpoint: format!("127.0.0.1:{port}"),
                    incarnation: uuid::Uuid::new_v4(),
                },
                Placement::new(vec!["a".into()]).unwrap(),
            ))
        };
        let view_a = make_view(port_a);
        let view_b = make_view(port_b);
        for view in [&view_a, &view_b] {
            view.replace_members([
                ("a".into(), view_a.owner().clone()),
                ("b".into(), view_b.owner().clone()),
            ]);
            assert!(view.renew(Instant::now(), Duration::from_secs(120)));
        }
        let make_engine = |view| {
            Arc::new(
                OrbitKVEngine::new_with_config(
                    16 << 20,
                    false,
                    StorageConfig {
                        codec,
                        membership: Some(view),
                        mooncake_nic_names: mooncake_nics(),
                        ..Default::default()
                    },
                )
                .unwrap(),
            )
        };
        let source = make_engine(view_a.clone());
        let target = make_engine(view_b.clone());
        let stores = spawn_engine_server(source.clone(), port_a, view_a).await;
        spawn_engine_server(target.clone(), port_b, view_b).await;
        let gpu_source = GpuBuffer::alloc(8192);
        let gpu_target = GpuBuffer::alloc(8192);
        gpu_source.copy_from_host(&[0xa0, 0x3f].repeat(4096));
        gpu_target.zero();
        for (engine, gpu, id) in [
            (&source, &gpu_source, "source"),
            (&target, &gpu_target, "target"),
        ] {
            engine
                .register_context_layer_batch_strided(
                    id,
                    "encoded-peers",
                    0,
                    0,
                    0,
                    1,
                    1,
                    &["layer".into()],
                    &[gpu.as_u64()],
                    &[8192],
                    &[1],
                    &[8192],
                    &[0],
                    &[1],
                    None,
                    None,
                    Some(&[StorageFormat::Attention {
                        scalar: Scalar16::Bf16,
                        role: AttentionRole::Key,
                        head_dim: 128,
                        layer_index: 2,
                        layer_count: 8,
                    }]),
                    TransferMode::Direct,
                    false,
                )
                .unwrap();
        }
        let hashes = vec![b"encoded-page".to_vec()];
        source
            .batch_save_kv_blocks_from_ipc(
                "source",
                0,
                0,
                0,
                vec![LayerSave {
                    layer_name: "layer".into(),
                    block_ids: vec![0],
                    block_hashes: hashes.clone(),
                }],
            )
            .await
            .unwrap();
        wait_for_cache(&source, "source", &hashes, 1, Duration::from_secs(10)).await;
        source.flush_saves_and_inventory().await.unwrap();
        let namespace = source.instance_namespace("source").unwrap();
        wait_for_catalog_registration(
            &stores,
            &namespace,
            &[group_hash(&hashes[0], 0)],
            1,
            Duration::from_secs(10),
        )
        .await;
        let mut images = Vec::new();
        for (engine, gpu, id) in [
            (&source, &gpu_source, "source"),
            (&target, &gpu_target, "target"),
        ] {
            let result = engine
                .count_prefix_hit_blocks_with_prefetch(id, "codec", &hashes, QueryMode::Demand)
                .await
                .unwrap();
            assert_eq!(result.blocks.len(), 1);
            assert!(
                result.blocks[0].memory_footprint() < 8192,
                "{codec:?} did not use encoded residency"
            );
            let lease = engine.create_query_lease(id, result.blocks).unwrap();
            gpu.zero();
            engine
                .restore(id, 0, 0, &[vec!["layer"]], &[(lease, vec![vec![Some(0)]])])
                .unwrap()
                .await
                .unwrap()
                .result
                .unwrap();
            images.push(gpu.copy_to_host());
        }
        assert_eq!(
            images[0], images[1],
            "peer codec metadata/payload changed for {codec:?}"
        );
        source.unregister_instance_and_wait("source").await.unwrap();
        target.unregister_instance_and_wait("target").await.unwrap();
    }
}
