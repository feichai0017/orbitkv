//! Embeddable P2P transfer gRPC service.
//!
//! A node that serves cross-node fetches exposes `QueryBlocksForTransfer`
//! (authorize and pin blocks, returning Mooncake addresses) and
//! `ReleaseTransferLock`. The Cache Manager serves only these peer RPCs plus
//! `Health`; inference processes use the node-local UDS/iceoryx2 endpoint.
//!
//! The embedder remains responsible for periodic GC of expired transfer locks
//! (`OrbitKVEngine::gc_expired_transfer_locks`), mirroring orbitkv-server's
//! background GC task — a crashed peer must not pin blocks forever.

use std::sync::Arc;

use log::{debug, info};
use tonic::{Request, Response, Status, async_trait};

use orbitkv_proto::proto::engine::engine_server::{Engine, EngineServer};
use orbitkv_proto::proto::engine::{
    HealthRequest, HealthResponse, QueryBlocksForTransferRequest, QueryBlocksForTransferResponse,
    ReleaseTransferLockRequest, ReleaseTransferLockResponse, ResponseStatus, TransferBlockInfo,
    TransferSlotInfo,
};

use crate::{LayerBlock, OrbitKVEngine};

/// Match orbitkv-server's cap: a `QueryBlocksForTransfer` response carries
/// per-slot descriptors for every requested block, which overflows tonic's
/// default 4 MiB limit on large batches.
const MAX_GRPC_MESSAGE_SIZE: usize = 64 * 1024 * 1024;

/// Peer-only gRPC service for fetching blocks from this Cache Manager or
/// an embedded `OrbitKVEngine`.
pub struct P2pTransferService {
    engine: Arc<OrbitKVEngine>,
}

impl P2pTransferService {
    pub fn new(engine: Arc<OrbitKVEngine>) -> Self {
        Self { engine }
    }

    /// Serve on `addr` until `shutdown` resolves. Must run inside a tokio
    /// runtime. The address must be the engine's routable advertise address —
    /// peers discover it through the MetaServer and dial it for handshakes.
    pub async fn serve(
        engine: Arc<OrbitKVEngine>,
        addr: std::net::SocketAddr,
        shutdown: impl std::future::Future<Output = ()> + Send,
    ) -> Result<(), tonic::transport::Error> {
        info!("P2P transfer service listening on {addr}");
        let service = EngineServer::new(Self::new(engine))
            .max_decoding_message_size(MAX_GRPC_MESSAGE_SIZE)
            .max_encoding_message_size(MAX_GRPC_MESSAGE_SIZE);
        tonic::transport::Server::builder()
            .add_service(service)
            .serve_with_shutdown(addr, shutdown)
            .await
    }

    /// [`Self::serve`] over a caller-bound listener stream. Binding first lets
    /// the embedder fail loud on a taken port before reporting itself ready.
    pub async fn serve_with_incoming<I, IO, IE>(
        engine: Arc<OrbitKVEngine>,
        incoming: I,
        shutdown: impl std::future::Future<Output = ()> + Send,
    ) -> Result<(), tonic::transport::Error>
    where
        I: futures::Stream<Item = Result<IO, IE>>,
        IO: tonic::transport::server::Connected
            + tokio::io::AsyncRead
            + tokio::io::AsyncWrite
            + Send
            + Unpin
            + 'static,
        IE: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        let service = EngineServer::new(Self::new(engine))
            .max_decoding_message_size(MAX_GRPC_MESSAGE_SIZE)
            .max_encoding_message_size(MAX_GRPC_MESSAGE_SIZE);
        tonic::transport::Server::builder()
            .add_service(service)
            .serve_with_incoming_shutdown(incoming, shutdown)
            .await
    }

    fn ok_status() -> ResponseStatus {
        ResponseStatus {
            ok: true,
            message: String::new(),
        }
    }

    fn build_transfer_slot_info(
        raw_block: &crate::RawBlock,
        numa_node: orbitkv_common::NumaNode,
    ) -> TransferSlotInfo {
        let layer_block = LayerBlock::new(raw_block);
        TransferSlotInfo {
            k_ptr: layer_block.k_ptr() as u64,
            k_size: layer_block.k_size() as u64,
            v_ptr: layer_block.v_ptr().map(|p| p as u64).unwrap_or(0),
            v_size: layer_block.v_size().unwrap_or(0) as u64,
            numa_node: numa_node.0,
        }
    }
}

#[async_trait]
impl Engine for P2pTransferService {
    async fn query_blocks_for_transfer(
        &self,
        request: Request<QueryBlocksForTransferRequest>,
    ) -> Result<Response<QueryBlocksForTransferResponse>, Status> {
        let req = request.into_inner();

        if !self.engine.has_remote_transport() {
            return Err(Status::failed_precondition(
                "Mooncake transfer engine is not configured",
            ));
        }

        let (session_id, found_blocks) = self.engine.query_blocks_for_transfer(
            &req.namespace,
            &req.block_hashes,
            &req.requester_id,
        );

        let blocks: Vec<TransferBlockInfo> = found_blocks
            .iter()
            .map(|(key, block)| {
                let slots: Vec<TransferSlotInfo> = block
                    .slots()
                    .iter()
                    .zip(block.slot_numas())
                    .map(|(raw, &numa)| Self::build_transfer_slot_info(raw, numa))
                    .collect();
                TransferBlockInfo {
                    block_hash: key.hash.clone(),
                    slots,
                }
            })
            .collect();

        debug!(
            "P2P query_blocks_for_transfer: requester={} requested={} found={} session={}",
            req.requester_id,
            req.block_hashes.len(),
            blocks.len(),
            session_id,
        );

        Ok(Response::new(QueryBlocksForTransferResponse {
            status: Some(Self::ok_status()),
            blocks,
            transfer_session_id: session_id,
            lock_timeout_secs: self.engine.transfer_lock_timeout().as_secs() as u32,
            transfer_endpoint: self
                .engine
                .transfer_endpoint()
                .ok_or_else(|| {
                    Status::failed_precondition("Mooncake transfer engine is not configured")
                })?
                .to_string(),
        }))
    }

    async fn release_transfer_lock(
        &self,
        request: Request<ReleaseTransferLockRequest>,
    ) -> Result<Response<ReleaseTransferLockResponse>, Status> {
        let req = request.into_inner();
        let released = self.engine.release_transfer_lock(&req.transfer_session_id);
        debug!(
            "P2P release_transfer_lock: session={} released={released}",
            req.transfer_session_id
        );
        Ok(Response::new(ReleaseTransferLockResponse {
            status: Some(Self::ok_status()),
            released_blocks: released as u64,
        }))
    }

    async fn health(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        Ok(Response::new(HealthResponse {
            status: Some(Self::ok_status()),
        }))
    }
}
