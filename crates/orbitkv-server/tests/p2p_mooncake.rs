//! P2P Mooncake remote fetch integration test.
//!
//! Verifies the end-to-end flow:
//! Engine A saves blocks → MetaServer discovers them → Engine B fetches via Mooncake READ
//! → data integrity verified.
//!
//! Run with: `cargo test -p orbitkv-server --test p2p_mooncake -- --ignored`

use std::ffi::c_void;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use cudarc::driver::CudaContext;
use cudarc::driver::sys;
use orbitkv_core::sync_state::{LOAD_STATE_ERROR, LOAD_STATE_SUCCESS};
use orbitkv_core::*;
use orbitkv_metaserver::{BlockHashStore, GrpcMetaService};
use orbitkv_proto::proto::engine::{
    QueryBlocksForTransferRequest, ReleaseTransferLockRequest, engine_client::EngineClient,
    meta_server_server::MetaServerServer,
};
use orbitkv_server::proto::engine::engine_server::EngineServer;
use orbitkv_state::group_hash;
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

async fn spawn_metaserver(port: u16) -> Arc<BlockHashStore> {
    let store = Arc::new(BlockHashStore::new());
    let service = GrpcMetaService::new(Arc::clone(&store));
    let addr: SocketAddr = ([127, 0, 0, 1], port).into();
    tokio::spawn(async move {
        Server::builder()
            .add_service(MetaServerServer::new(service))
            .serve(addr)
            .await
            .expect("MetaServer gRPC serve");
    });
    wait_for_grpc_ready(port).await;
    store
}

async fn spawn_engine_server(engine: Arc<OrbitKVEngine>, port: u16) {
    let service = P2pTransferService::new(engine);
    let addr: SocketAddr = ([127, 0, 0, 1], port).into();
    tokio::spawn(async move {
        Server::builder()
            .add_service(EngineServer::new(service))
            .serve(addr)
            .await
            .expect("Engine gRPC serve");
    });
    wait_for_grpc_ready(port).await;
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
                false,
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

async fn wait_for_metaserver_registration(
    store: &BlockHashStore,
    namespace: &str,
    hashes: &[Vec<u8>],
    expected: usize,
    timeout: Duration,
) {
    let deadline = Instant::now() + timeout;
    loop {
        let found = store.locate_blocks(namespace, hashes, "");
        let count = found
            .iter()
            .take_while(|row| !row.replicas.is_empty())
            .count();
        if count >= expected {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for MetaServer registration ({} / {})",
            count,
            expected
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

async fn wait_for_metaserver_ownership(
    store: &BlockHashStore,
    namespace: &str,
    hashes: &[Vec<u8>],
    node: &str,
    expected: usize,
    timeout: Duration,
) {
    let deadline = Instant::now() + timeout;
    loop {
        let found = store.locate_blocks(namespace, hashes, "");
        let owned = found
            .iter()
            .filter(|entry| entry.replicas.iter().any(|r| r.owner.endpoint == node))
            .count();
        if owned >= expected {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for MetaServer ownership by {node} ({owned} / {expected})"
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
    let meta_port = get_free_port();
    let port_a = get_free_port();

    // ── 1. Start MetaServer ──
    let meta_store = spawn_metaserver(meta_port).await;

    // ── 2. Create Engine A (source of blocks) ──
    let config_a = StorageConfig {
        metaserver_addr: Some(format!("http://127.0.0.1:{meta_port}")),
        advertise_addr: Some(format!("127.0.0.1:{port_a}")),
        mooncake_nic_names: mooncake_nics(),
        ..StorageConfig::default()
    };
    let engine_a = Arc::new(
        OrbitKVEngine::new_with_config(16 << 20, false, config_a).expect("engine A should start"),
    );

    // ── 3. Start Engine A gRPC server ──
    spawn_engine_server(Arc::clone(&engine_a), port_a).await;

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
    wait_for_metaserver_registration(
        &meta_store,
        &cache_namespace,
        &stored_hashes,
        NUM_BLOCKS,
        Duration::from_secs(10),
    )
    .await;

    // Source authorization fences both restarts and individual residency episodes.
    let evidence = meta_store.locate_blocks(&cache_namespace, &stored_hashes, "requester");
    let mut peer = EngineClient::connect(format!("http://127.0.0.1:{port_a}"))
        .await
        .unwrap();
    let authorization = QueryBlocksForTransferRequest {
        namespace: cache_namespace.clone(),
        block_hashes: stored_hashes.clone(),
        requester_id: "test-requester".into(),
        owner_incarnation: evidence[0].replicas[0].owner.incarnation.to_string(),
        residency_sequences: evidence.iter().map(|r| r.replicas[0].sequence).collect(),
    };
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
    peer.release_transfer_lock(ReleaseTransferLockRequest {
        transfer_session_id: granted.transfer_session_id,
    })
    .await
    .unwrap();

    // ── 7. Create Engine B (fetcher) ──
    let port_b = get_free_port();
    let config_b = StorageConfig {
        metaserver_addr: Some(format!("http://127.0.0.1:{meta_port}")),
        advertise_addr: Some(format!("127.0.0.1:{port_b}")),
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
        true,
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

    wait_for_metaserver_registration(
        &meta_store,
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

    // ── 9b. Verify Engine B re-registered fetched blocks to MetaServer ──
    // Mooncake-fetched blocks are now resident on B, so B must advertise them so
    // other nodes can discover and fetch from B (not just from A).
    engine_b
        .flush_saves_and_inventory()
        .await
        .expect("restored inventory");
    wait_for_metaserver_ownership(
        &meta_store,
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
    let remaining = meta_store.locate_blocks(&cache_namespace, &stored_delayed, "");
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
        meta_store
            .locate_blocks(NAMESPACE, &delayed_hashes, "")
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
    let load_state = LoadState::new().expect("create LoadState");
    let shm_name = load_state.shm_name().to_string();

    engine_b
        .batch_load_kv_blocks_multi_layer(
            "inst-b",
            0,
            DEVICE_ID,
            &shm_name,
            &[vec![LAYER]],
            &[(lease, vec![block_ids.iter().copied().map(Some).collect()])],
        )
        .expect("batch_load on engine B");

    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let state = load_state.get();
        if state == LOAD_STATE_SUCCESS {
            break;
        }
        assert!(state != LOAD_STATE_ERROR, "load reported ERROR");
        assert!(Instant::now() < deadline, "timed out waiting for load");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    // ── 11. Verify data integrity ──
    let loaded = gpu_b.copy_to_host();
    assert_eq!(
        loaded, host_data,
        "GPU data mismatch: remote-fetched blocks differ from original"
    );
}
