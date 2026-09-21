use etcd_client::{Client, Compare, CompareOp, PutOptions, Txn, TxnOp};
use orbitkv_state::CacheOwner;

use super::{Member, cluster_id, rpc};

pub(super) async fn register(
    client: &mut Client,
    prefix: &str,
    node: &str,
    owner: &CacheOwner,
    lease: i64,
    expected_cluster: u64,
) -> Result<Member, String> {
    let key = format!("{prefix}members/{node}");
    let epoch_key = format!("{prefix}epochs/{node}");
    for _ in 0..8 {
        let response = rpc(client.get(epoch_key.clone(), None)).await?;
        if cluster_id(response.header())? != expected_cluster {
            return Err("coordinator cluster changed during registration".into());
        }
        let (previous, revision) = match response.kvs().first() {
            Some(kv) => (
                kv.value_str()
                    .map_err(|error| error.to_string())?
                    .parse::<u64>()
                    .map_err(|_| "invalid persisted node epoch")?,
                kv.mod_revision(),
            ),
            None => (0, 0),
        };
        let epoch = previous.checked_add(1).ok_or("node epoch exhausted")?;
        let member = Member {
            node_id: node.into(),
            epoch,
            owner: owner.clone(),
            lease,
        };
        let bytes = serde_json::to_vec(&member).map_err(|error| error.to_string())?;
        let transaction = Txn::new()
            .when([
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
        if cluster_id(response.header())? != expected_cluster {
            return Err("coordinator cluster changed during registration".into());
        }
        if response.succeeded() {
            return Ok(member);
        }
        let current = rpc(client.get(key.clone(), None)).await?;
        if !current.kvs().is_empty() {
            return Err(format!("node {node} already has a live registration"));
        }
    }
    Err("registration contention exceeded retry budget".into())
}
