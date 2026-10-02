use etcd_client::{Client, Compare, CompareOp, PutOptions, Txn, TxnOp};
use orbitkv_state::CacheOwner;

use super::format::ClusterFormat;
use super::{BootstrapError, Member, cluster_id, rpc};

pub(super) async fn register(
    client: &mut Client,
    prefix: &str,
    node: &str,
    owner: &CacheOwner,
    lease: i64,
    expected_cluster: u64,
    format: &ClusterFormat,
) -> Result<Member, BootstrapError> {
    let key = format!("{prefix}members/{node}");
    let epoch_key = format!("{prefix}epochs/{node}");
    let format_key = format!("{prefix}format");
    for _ in 0..8 {
        let response = rpc(client.get(epoch_key.clone(), None)).await?;
        if cluster_id(response.header()).map_err(BootstrapError::Rejected)? != expected_cluster {
            return Err(BootstrapError::Rejected(
                "coordinator cluster changed during registration".into(),
            ));
        }
        let (previous, revision) = match response.kvs().first() {
            Some(kv) => (
                kv.value_str()
                    .map_err(|error| BootstrapError::Rejected(error.to_string()))?
                    .parse::<u64>()
                    .map_err(|_| BootstrapError::Rejected("invalid persisted node epoch".into()))?,
                kv.mod_revision(),
            ),
            None => (0, 0),
        };
        let epoch = previous
            .checked_add(1)
            .ok_or_else(|| BootstrapError::Rejected("node epoch exhausted".into()))?;
        let member = Member {
            node_id: node.into(),
            epoch,
            owner: owner.clone(),
            protocol: orbitkv_state::INVENTORY_STREAM_PROTOCOL.into(),
            lease,
        };
        let bytes = serde_json::to_vec(&member)
            .map_err(|error| BootstrapError::Rejected(error.to_string()))?;
        let transaction = Txn::new()
            .when([
                Compare::mod_revision(format_key.clone(), CompareOp::Equal, format.revision),
                Compare::value(format_key.clone(), CompareOp::Equal, format.encoded.clone()),
                Compare::version(key.clone(), CompareOp::Equal, 0),
                Compare::mod_revision(epoch_key.clone(), CompareOp::Equal, revision),
            ])
            .and_then([
                TxnOp::put(epoch_key.clone(), epoch.to_string(), None),
                TxnOp::put(
                    key.clone(),
                    bytes,
                    Some(PutOptions::new().with_lease(lease)),
                ),
            ]);
        let response = rpc(client.txn(transaction)).await?;
        if cluster_id(response.header()).map_err(BootstrapError::Rejected)? != expected_cluster {
            return Err(BootstrapError::Rejected(
                "coordinator cluster changed during registration".into(),
            ));
        }
        if response.succeeded() {
            return Ok(member);
        }
        let current = rpc(client.get(key.clone(), None)).await?;
        if cluster_id(current.header()).map_err(BootstrapError::Rejected)? != expected_cluster {
            return Err(BootstrapError::Rejected(
                "coordinator cluster changed during registration reconciliation".into(),
            ));
        }
        if let Some(kv) = current.kvs().first() {
            let mut existing: Member = serde_json::from_slice(kv.value())
                .map_err(|error| BootstrapError::Rejected(error.to_string()))?;
            existing.lease = kv.lease();
            if existing.node_id == node
                && existing.owner == *owner
                && existing.lease == lease
                && existing.epoch > 0
            {
                return Ok(existing);
            }
            return Err(BootstrapError::Rejected(format!(
                "node {node} already has a live registration"
            )));
        }
    }
    Err(BootstrapError::Rejected(
        "registration contention exceeded retry budget".into(),
    ))
}
