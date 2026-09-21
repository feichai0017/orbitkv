use std::sync::Arc;
use std::time::Instant;

use tonic::{Request, Response, Status, async_trait};
use uuid::Uuid;

use crate::MembershipView;
use crate::metric::record_rpc_result;
use crate::store::{BlockHashStore, StoreError};
use orbitkv_proto::proto::engine::CatalogRoute;
use orbitkv_proto::proto::engine::catalog_server::Catalog;
use orbitkv_proto::proto::engine::{
    HeartbeatNodeRequest, HeartbeatNodeResponse, LocateBlocksRequest, LocateBlocksResponse,
    SyncInventoryRequest, SyncInventoryResponse, UnregisterNodeRequest, UnregisterNodeResponse,
};
use orbitkv_state::{CATALOG_SHARDS, CacheOwner, StateKey, catalog_shard};
#[derive(Clone)]
pub struct CatalogService {
    stores: [Arc<BlockHashStore>; CATALOG_SHARDS],
    membership: Arc<MembershipView>,
    jobs: Arc<tokio::sync::Semaphore>,
}

impl CatalogService {
    pub fn new(
        stores: [Arc<BlockHashStore>; CATALOG_SHARDS],
        membership: Arc<MembershipView>,
    ) -> Self {
        Self {
            stores,
            membership,
            jobs: Arc::new(tokio::sync::Semaphore::new(16)),
        }
    }

    fn store(&self, route: Option<&CatalogRoute>) -> Result<Arc<BlockHashStore>, Status> {
        let route = route.ok_or_else(|| Status::invalid_argument("missing catalog route"))?;
        let owner = self.membership.owner();
        if route.placement_id != self.membership.placement_id()
            || route.incarnation != owner.incarnation.to_string()
            || !self.membership.permits(owner)
            || self.membership.catalog_owner(route.shard as usize).as_ref() != Some(owner)
        {
            return Err(Status::failed_precondition(
                "catalog placement or incarnation is unavailable",
            ));
        }
        self.stores
            .get(route.shard as usize)
            .cloned()
            .ok_or_else(|| Status::invalid_argument("invalid catalog shard"))
    }

    fn publisher(&self, endpoint: &str, incarnation: &str) -> Result<Uuid, Status> {
        let owner = CacheOwner {
            endpoint: endpoint.into(),
            incarnation: parse_uuid(incarnation)?,
        };
        if !self.membership.permits(&owner) {
            return Err(Status::failed_precondition(
                "publisher is not a current member",
            ));
        }
        Ok(owner.incarnation)
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
        StoreError::Capacity => Status::resource_exhausted("catalog shard byte budget exceeded"),
    }
}

#[async_trait]
impl Catalog for CatalogService {
    async fn heartbeat_node(
        &self,
        request: Request<HeartbeatNodeRequest>,
    ) -> Result<Response<HeartbeatNodeResponse>, Status> {
        let job = self
            .jobs
            .clone()
            .try_acquire_owned()
            .map_err(|_| Status::resource_exhausted("catalog operation budget exhausted"))?;
        let start = Instant::now();
        let req = request.into_inner();
        let result = async {
            let node_id = self.publisher(&req.node, &req.node_id)?;
            let store = self.store(req.route.as_ref())?;
            let config = store.config();
            let epoch = store.catalog_epoch();
            // Session takeover may clean up a large previous inventory.
            let progress = tokio::task::spawn_blocking(move || {
                let _job = job;
                store.heartbeat_node(&req.node, node_id)
            })
            .await
            .map_err(|e| Status::internal(e.to_string()))?
            .map_err(store_error)?;
            Ok(Response::new(HeartbeatNodeResponse {
                stale_after_secs: config.node_stale_after.as_secs(),
                catalog_epoch: epoch.to_string(),
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
        let job = self
            .jobs
            .clone()
            .try_acquire_owned()
            .map_err(|_| Status::resource_exhausted("catalog operation budget exhausted"))?;
        let start = Instant::now();
        let req = request.into_inner();
        let result = async {
            let node_id = self.publisher(&req.node, &req.node_id)?;
            let store = self.store(req.route.as_ref())?;
            let removed = tokio::task::spawn_blocking(move || {
                let _job = job;
                store.unregister_node(&req.node, node_id)
            })
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
        let job = self
            .jobs
            .clone()
            .try_acquire_owned()
            .map_err(|_| Status::resource_exhausted("catalog operation budget exhausted"))?;
        let start = Instant::now();
        let req = request.into_inner();
        let result = async {
            let node_id = self.publisher(&req.node, &req.node_id)?;
            let epoch = parse_uuid(&req.catalog_epoch)?;
            let shard = req
                .route
                .as_ref()
                .ok_or_else(|| Status::invalid_argument("missing route"))?
                .shard as usize;
            let operation = req
                .operation
                .ok_or_else(|| Status::invalid_argument("missing inventory operation"))?;
            let operation: orbitkv_state::InventoryOperation = operation.into();
            if let orbitkv_state::InventoryOperation::Snapshot { records, .. }
            | orbitkv_state::InventoryOperation::Delta { records, .. } = &operation
                && records
                    .iter()
                    .any(|record| catalog_shard(&record.key) != shard)
            {
                return Err(Status::invalid_argument(
                    "inventory record belongs to another shard",
                ));
            }
            let store = self.store(req.route.as_ref())?;
            let (progress, reclaimable) = tokio::task::spawn_blocking(move || {
                let _job = job;
                store.sync_inventory(&req.node, node_id, epoch, req.generation, operation)
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
        let job = self
            .jobs
            .clone()
            .try_acquire_owned()
            .map_err(|_| Status::resource_exhausted("catalog operation budget exhausted"))?;
        let start = Instant::now();
        let req = request.into_inner();
        let result = async {
            orbitkv_state::validate_discovery_query(&req.namespace, &req.block_hashes)
                .map_err(Status::invalid_argument)?;
            if req.exclude_node.len() > orbitkv_state::DISCOVERY_MAX_ENDPOINT_BYTES {
                return Err(Status::invalid_argument("requester endpoint too long"));
            }
            let store = self.store(req.route.as_ref())?;
            let shard = req.route.as_ref().expect("validated route").shard as usize;
            if req.block_hashes.iter().any(|hash| {
                catalog_shard(&StateKey::new(req.namespace.clone(), hash.clone())) != shard
            }) {
                return Err(Status::invalid_argument(
                    "query contains a key from another shard",
                ));
            }
            let mut blocks = tokio::task::spawn_blocking(move || {
                let _job = job;
                store.locate_blocks(&req.namespace, &req.block_hashes, &req.exclude_node)
            })
            .await
            .map_err(|e| Status::internal(e.to_string()))?;
            for row in &mut blocks {
                row.replicas
                    .retain(|replica| self.membership.permits(&replica.owner));
            }
            Ok(Response::new(LocateBlocksResponse {
                blocks: blocks.into_iter().map(Into::into).collect(),
            }))
        }
        .await;
        record_rpc_result("locate_blocks", &result, start);
        result
    }
}
