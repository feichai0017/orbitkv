use std::sync::Arc;
use std::time::Instant;

use tonic::{Request, Response, Status, async_trait};
use uuid::Uuid;

use crate::metric::record_rpc_result;
use crate::proto::engine::meta_server_server::MetaServer;
use crate::proto::engine::{
    FetchSegment, HeartbeatNodeRequest, HeartbeatNodeResponse, QueryPrefixBlocksRequest,
    QueryPrefixBlocksResponse, SyncInventoryRequest, SyncInventoryResponse, UnregisterNodeRequest,
    UnregisterNodeResponse,
};
use crate::store::{BlockHashStore, PrefixEntry, StoreError};
fn plan_fetch_segments(
    entries: &[PrefixEntry],
    exclude_node: &str,
) -> Result<Vec<FetchSegment>, &'static str> {
    let mut segments = Vec::new();
    let mut offset = 0usize;

    while offset < entries.len() {
        let mut best: Option<(&str, usize)> = None;
        for candidate in &entries[offset].nodes {
            let candidate = candidate.as_ref();
            if candidate == exclude_node {
                continue;
            }

            let end = entries[offset..]
                .iter()
                .take_while(|entry| entry.nodes.iter().any(|node| node.as_ref() == candidate))
                .count()
                + offset;

            if best.is_none_or(|(best_node, best_end)| {
                end > best_end || (end == best_end && candidate < best_node)
            }) {
                best = Some((candidate, end));
            }
        }

        let Some((node, end)) = best else {
            break;
        };
        let block_count =
            u32::try_from(end - offset).map_err(|_| "fetch segment block count exceeds uint32")?;
        segments.push(FetchSegment {
            node: node.to_string(),
            block_count,
        });
        offset = end;
    }

    Ok(segments)
}

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

    async fn query_prefix_blocks(
        &self,
        request: Request<QueryPrefixBlocksRequest>,
    ) -> Result<Response<QueryPrefixBlocksResponse>, Status> {
        let start = Instant::now();
        let req = request.into_inner();
        let result = async {
            if req.block_hashes.is_empty() {
                return Err(Status::invalid_argument("block_hashes cannot be empty"));
            }
            let store = Arc::clone(&self.store);
            let existing = tokio::task::spawn_blocking(move || {
                store.query_prefix(&req.namespace, &req.block_hashes)
            })
            .await
            .map_err(|e| Status::internal(e.to_string()))?;
            let segments = plan_fetch_segments(&existing, &req.exclude_node)
                .map_err(Status::invalid_argument)?;
            Ok(Response::new(QueryPrefixBlocksResponse { segments }))
        }
        .await;
        record_rpc_result("query_prefix_blocks", &result, start);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn prefix_entry(hash: u8, nodes: &[&str]) -> PrefixEntry {
        PrefixEntry {
            block_hash: vec![hash],
            nodes: nodes.iter().map(|node| Arc::<str>::from(*node)).collect(),
        }
    }

    fn planned(entries: &[PrefixEntry], exclude_node: &str) -> Vec<(String, u32)> {
        plan_fetch_segments(entries, exclude_node)
            .expect("small test plan should fit uint32")
            .into_iter()
            .map(|segment| (segment.node, segment.block_count))
            .collect()
    }

    #[test]
    fn planner_combines_fragmented_remote_prefix() {
        let entries = vec![
            prefix_entry(1, &["node-a"]),
            prefix_entry(2, &["node-a"]),
            prefix_entry(3, &["node-b"]),
            prefix_entry(4, &["node-b"]),
        ];

        assert_eq!(
            planned(&entries, "requester"),
            vec![("node-a".into(), 2), ("node-b".into(), 2)]
        );
    }

    #[test]
    fn planner_chooses_farthest_owner_and_stable_tie_break() {
        let entries = vec![
            prefix_entry(1, &["node-c", "node-b", "node-a"]),
            prefix_entry(2, &["node-c", "node-b", "node-a"]),
            prefix_entry(3, &["node-c"]),
        ];

        assert_eq!(planned(&entries, "requester"), vec![("node-c".into(), 3)]);
        assert_eq!(
            planned(&entries[..2], "requester"),
            vec![("node-a".into(), 2)]
        );
    }

    #[test]
    fn planner_excludes_requester_and_stops_at_remote_gap() {
        let entries = vec![
            prefix_entry(1, &["requester", "node-a"]),
            prefix_entry(2, &["requester"]),
            prefix_entry(3, &["node-b"]),
        ];

        assert_eq!(planned(&entries, "requester"), vec![("node-a".into(), 1)]);
    }

    #[test]
    fn planner_keeps_single_owner_prefix_in_one_segment() {
        let entries = vec![
            prefix_entry(1, &["node-a"]),
            prefix_entry(2, &["node-a"]),
            prefix_entry(3, &["node-a"]),
        ];

        assert_eq!(planned(&entries, "requester"), vec![("node-a".into(), 3)]);
    }
}
