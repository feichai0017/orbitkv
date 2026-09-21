use std::sync::Arc;
use std::time::Instant;

use tonic::{Request, Response, Status, async_trait};
use uuid::Uuid;

use crate::metric::record_rpc_result;
use crate::proto::engine::meta_server_server::MetaServer;
use crate::proto::engine::{
    HeartbeatNodeRequest, HeartbeatNodeResponse, LocateBlocksRequest, LocateBlocksResponse,
    SyncInventoryRequest, SyncInventoryResponse, UnregisterNodeRequest, UnregisterNodeResponse,
};
use crate::store::{BlockHashStore, StoreError};
#[derive(Clone)]
pub struct GrpcMetaService {
    store: Arc<BlockHashStore>,
}

impl GrpcMetaService {
    pub fn new(store: Arc<BlockHashStore>) -> Self {
        Self { store }
    }
}

fn parse_uuid(value: &str) -> Result<Uuid, Status> {
    Uuid::parse_str(value).map_err(|e| Status::invalid_argument(format!("invalid UUID: {e}")))
}

fn store_error(err: StoreError) -> Status {
    match err {
        StoreError::UnknownNode => Status::failed_precondition("unknown node"),
        StoreError::StaleSession => Status::failed_precondition("stale node session"),
        StoreError::CatalogRestarted => Status::failed_precondition("catalog epoch changed"),
        StoreError::OutOfOrder => {
            Status::failed_precondition("inventory sequence or generation mismatch")
        }
        StoreError::InvalidInventory => Status::invalid_argument("invalid inventory batch"),
        StoreError::Capacity => Status::resource_exhausted("owner inventory byte budget exceeded"),
    }
}

#[async_trait]
impl MetaServer for GrpcMetaService {
    async fn heartbeat_node(
        &self,
        request: Request<HeartbeatNodeRequest>,
    ) -> Result<Response<HeartbeatNodeResponse>, Status> {
        let start = Instant::now();
        let req = request.into_inner();
        let result = async {
            let node_id = parse_uuid(&req.node_id)?;
            let store = Arc::clone(&self.store);
            // Session takeover may clean up a large previous inventory.
            let progress =
                tokio::task::spawn_blocking(move || store.heartbeat_node(&req.node, node_id))
                    .await
                    .map_err(|e| Status::internal(e.to_string()))?
                    .map_err(store_error)?;
            Ok(Response::new(HeartbeatNodeResponse {
                stale_after_secs: self.store.config().node_stale_after.as_secs(),
                catalog_epoch: self.store.catalog_epoch().to_string(),
                progress: Some(progress.into()),
            }))
        }
        .await;
        record_rpc_result("heartbeat_node", &result, start);
        result
    }

    async fn unregister_node(
        &self,
        request: Request<UnregisterNodeRequest>,
    ) -> Result<Response<UnregisterNodeResponse>, Status> {
        let start = Instant::now();
        let req = request.into_inner();
        let result = async {
            let node_id = parse_uuid(&req.node_id)?;
            let store = Arc::clone(&self.store);
            let removed =
                tokio::task::spawn_blocking(move || store.unregister_node(&req.node, node_id))
                    .await
                    .map_err(|e| Status::internal(e.to_string()))?
                    .map_err(store_error)?;
            Ok(Response::new(UnregisterNodeResponse {
                removed_owners: removed as u64,
            }))
        }
        .await;
        record_rpc_result("unregister_node", &result, start);
        result
    }

    async fn sync_inventory(
        &self,
        request: Request<SyncInventoryRequest>,
    ) -> Result<Response<SyncInventoryResponse>, Status> {
        let start = Instant::now();
        let req = request.into_inner();
        let result = async {
            let node_id = parse_uuid(&req.node_id)?;
            let epoch = parse_uuid(&req.catalog_epoch)?;
            let operation = req
                .operation
                .ok_or_else(|| Status::invalid_argument("missing inventory operation"))?;
            let store = Arc::clone(&self.store);
            let (progress, reclaimable) = tokio::task::spawn_blocking(move || {
                store.sync_inventory(&req.node, node_id, epoch, req.generation, operation.into())
            })
            .await
            .map_err(|e| Status::internal(e.to_string()))?
            .map_err(store_error)?;
            Ok(Response::new(SyncInventoryResponse {
                progress: Some(progress.into()),
                reclaimable: reclaimable.into_iter().map(Into::into).collect(),
            }))
        }
        .await;
        record_rpc_result("sync_inventory", &result, start);
        result
    }

    async fn locate_blocks(
        &self,
        request: Request<LocateBlocksRequest>,
    ) -> Result<Response<LocateBlocksResponse>, Status> {
        let start = Instant::now();
        let req = request.into_inner();
        let result = async {
            orbitkv_state::validate_discovery_query(&req.namespace, &req.block_hashes)
                .map_err(Status::invalid_argument)?;
            if req.exclude_node.len() > orbitkv_state::DISCOVERY_MAX_ENDPOINT_BYTES {
                return Err(Status::invalid_argument("requester endpoint too long"));
            }
            let store = Arc::clone(&self.store);
            let blocks = tokio::task::spawn_blocking(move || {
                store.locate_blocks(&req.namespace, &req.block_hashes, &req.exclude_node)
            })
            .await
            .map_err(|e| Status::internal(e.to_string()))?;
            Ok(Response::new(LocateBlocksResponse {
                blocks: blocks.into_iter().map(Into::into).collect(),
            }))
        }
        .await;
        record_rpc_result("locate_blocks", &result, start);
        result
    }
}
