//! Embeddable P2P transfer gRPC service.
//!
//! A node opens reusable transfer windows, authorizes single-use tickets with
//! `QueryBlocksForTransfer`, and closes them with `ReleaseTransferLock`.
//! The Cache Manager serves only these peer RPCs plus
//! `Health`; inference processes use the node-local UDS/iceoryx2 endpoint.
//!
//! Source pins are budgeted and retained after timeout. An overdue session
//! requires terminal completion or transport revocation before memory reuse.

use std::sync::Arc;

use log::{debug, info};
use tonic::{Request, Response, Status, async_trait};

use orbitkv_proto::proto::engine::engine_server::{Engine, EngineServer};
use orbitkv_proto::proto::engine::{
    HealthRequest, HealthResponse, OpenTransferWindowRequest, OpenTransferWindowResponse,
    QueryBlocksForTransferRequest, QueryBlocksForTransferResponse, ReleaseTransferLockRequest,
    ReleaseTransferLockResponse, ResponseStatus, TransferBlockInfo, TransferSlotInfo,
    TransferTicket as WireTicket,
};

use crate::storage::TransferAuthorizationError;
use crate::storage::transfer_lock::{TRANSFER_WINDOW_SLOTS, TransferLockError, TransferTicket};
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
    /// peers discover it through the Catalog and dial it for handshakes.
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

    fn parse_ticket(ticket: Option<WireTicket>) -> Result<TransferTicket, Status> {
        let ticket = ticket.ok_or_else(|| Status::invalid_argument("missing transfer ticket"))?;
        let window = ticket
            .window_id
            .parse::<uuid::Uuid>()
            .map_err(|_| Status::invalid_argument("invalid transfer window"))?;
        if window.is_nil()
            || ticket.slot as usize >= TRANSFER_WINDOW_SLOTS
            || ticket.generation == 0
        {
            return Err(Status::invalid_argument("invalid transfer ticket"));
        }
        Ok(TransferTicket {
            window,
            slot: ticket.slot as usize,
            generation: ticket.generation,
        })
    }

    fn authorization_error(error: TransferAuthorizationError) -> Status {
        match error {
            TransferAuthorizationError::StaleReplica => {
                Status::failed_precondition("stale owner or residency candidate")
            }
            TransferAuthorizationError::Lock(TransferLockError::UnknownWindow) => {
                Status::not_found("unknown transfer window")
            }
            TransferAuthorizationError::Lock(TransferLockError::StaleTicket) => {
                Status::failed_precondition("closed or occupied transfer ticket")
            }
            TransferAuthorizationError::Lock(TransferLockError::BudgetExhausted) => {
                Status::resource_exhausted("source transfer reservation budget exhausted")
            }
        }
    }

    fn ok_status() -> ResponseStatus {
        ResponseStatus {
            ok: true,
            message: String::new(),
        }
    }

    fn build_transfer_slot_info(
        raw_block: &crate::RawBlock,
        numa_node: crate::NumaNode,
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
    async fn open_transfer_window(
        &self,
        request: Request<OpenTransferWindowRequest>,
    ) -> Result<Response<OpenTransferWindowResponse>, Status> {
        if !self.engine.has_remote_transport() {
            return Err(Status::failed_precondition(
                "Mooncake transfer engine is not configured",
            ));
        }
        let req = request.into_inner();
        let owner = req
            .owner_incarnation
            .parse()
            .map_err(|_| Status::invalid_argument("invalid owner incarnation"))?;
        let requester = req
            .requester_incarnation
            .parse::<uuid::Uuid>()
            .map_err(|_| Status::invalid_argument("invalid requester incarnation"))?;
        if requester.is_nil() {
            return Err(Status::invalid_argument("nil requester incarnation"));
        }
        self.engine
            .storage
            .validate_transfer_owner(owner)
            .map_err(Self::authorization_error)?;
        let window = self
            .engine
            .storage
            .transfer_lock
            .open(requester)
            .ok_or_else(|| Status::resource_exhausted("source transfer window budget exhausted"))?;
        Ok(Response::new(OpenTransferWindowResponse {
            window_id: window.to_string(),
        }))
    }

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

        orbitkv_state::validate_discovery_query(&req.namespace, &req.block_hashes)
            .map_err(Status::invalid_argument)?;
        if req.residency_sequences.len() != req.block_hashes.len()
            || req.residency_sequences.contains(&0)
        {
            return Err(Status::invalid_argument("missing residency sequences"));
        }
        let owner = req
            .owner_incarnation
            .parse()
            .map_err(|_| Status::invalid_argument("invalid owner incarnation"))?;
        let records: Vec<_> = req
            .block_hashes
            .iter()
            .zip(&req.residency_sequences)
            .map(|(hash, &sequence)| orbitkv_state::InventoryRecord {
                key: orbitkv_state::StateKey::new(req.namespace.clone(), hash.clone()),
                sequence,
                present: true,
            })
            .collect();
        let ticket = Self::parse_ticket(req.ticket)?;
        let found_blocks = self
            .engine
            .storage
            .authorize_transfer(owner, ticket, &records)
            .map_err(Self::authorization_error)?;

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
            "P2P query_blocks_for_transfer: requested={} found={} ticket={:?}",
            req.block_hashes.len(),
            blocks.len(),
            ticket,
        );

        Ok(Response::new(QueryBlocksForTransferResponse {
            status: Some(Self::ok_status()),
            blocks,
            lock_timeout_secs: self.engine.storage.transfer_lock.lock_timeout().as_secs() as u32,
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
        let ticket = Self::parse_ticket(request.into_inner().ticket)?;
        let released = self
            .engine
            .storage
            .transfer_lock
            .release(ticket)
            .map_err(|error| Self::authorization_error(TransferAuthorizationError::Lock(error)))?;
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
