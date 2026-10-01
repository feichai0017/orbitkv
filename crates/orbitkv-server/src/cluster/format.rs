use etcd_client::{Client, Compare, CompareOp, Txn, TxnOp};
use orbitkv_state::INVENTORY_STREAM_PROTOCOL;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::{BootstrapError, cluster_id, rpc};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ClusterFormat {
    pub protocol: String,
    pub cluster_uuid: Uuid,
    #[serde(skip)]
    pub revision: i64,
    #[serde(skip)]
    pub encoded: Vec<u8>,
}

impl ClusterFormat {
    fn new() -> Self {
        Self {
            protocol: INVENTORY_STREAM_PROTOCOL.into(),
            cluster_uuid: Uuid::new_v4(),
            revision: 0,
            encoded: Vec::new(),
        }
    }

    pub(super) fn decode(value: &[u8]) -> Result<Self, String> {
        let format: Self = serde_json::from_slice(value).map_err(|error| error.to_string())?;
        if format.protocol != INVENTORY_STREAM_PROTOCOL || format.cluster_uuid.is_nil() {
            return Err("cluster inventory protocol identity differs".into());
        }
        Ok(format)
    }

    pub(super) fn same_identity(&self, other: &Self) -> bool {
        self.protocol == other.protocol && self.cluster_uuid == other.cluster_uuid
    }
}

pub(super) async fn install(
    client: &mut Client,
    prefix: &str,
    expected_cluster: u64,
) -> Result<ClusterFormat, BootstrapError> {
    let key = format!("{prefix}format");
    let candidate = ClusterFormat::new();
    let bytes = serde_json::to_vec(&candidate)
        .map_err(|error| BootstrapError::Rejected(error.to_string()))?;
    let response = rpc(client.txn(
        Txn::new()
            .when([Compare::version(key.clone(), CompareOp::Equal, 0)])
            .and_then([TxnOp::put(key.clone(), bytes, None)]),
    ))
    .await?;
    if cluster_id(response.header()).map_err(BootstrapError::Rejected)? != expected_cluster {
        return Err(BootstrapError::Rejected(
            "coordinator changed during format registration".into(),
        ));
    }
    let response = rpc(client.get(key, None)).await?;
    if cluster_id(response.header()).map_err(BootstrapError::Rejected)? != expected_cluster
        || response.kvs().len() != 1
        || response.kvs()[0].lease() != 0
    {
        return Err(BootstrapError::Rejected(
            "cluster metadata format is missing or leased".into(),
        ));
    }
    let kv = &response.kvs()[0];
    let mut format = ClusterFormat::decode(kv.value()).map_err(BootstrapError::Rejected)?;
    format.revision = kv.mod_revision();
    format.encoded = kv.value().to_vec();
    Ok(format)
}
