use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use etcd_client::{Client, Compare, CompareOp, DeleteOptions, PutOptions, Txn, TxnOp};
use orbitkv_catalog::MembershipView;
use orbitkv_core::{InventoryDelta, PublishedInventory, ResidencyInventory};
use orbitkv_state::{InventoryRecord, ReplicaMedium};
use prost::Message;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{Member, cluster_id, rpc};

pub(super) const MAX_RECORD_BYTES: usize = 128 * 1024;
const MAX_BATCH_RECORDS: usize = 48;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Progress {
    pub operation: u64,
    pub sequence: u64,
    pub ready: bool,
}

struct Publisher {
    client: Client,
    prefix: String,
    member: Member,
    view: Arc<MembershipView>,
    cluster: u64,
    progress: Progress,
    previous: Option<Vec<u8>>,
}

pub(super) async fn run(
    client: Client,
    prefix: String,
    member: Member,
    cluster: u64,
    view: Arc<MembershipView>,
    inventory: Arc<ResidencyInventory>,
) {
    let mut publisher = Publisher {
        client,
        prefix,
        member,
        view,
        cluster,
        progress: Progress::default(),
        previous: None,
    };
    let changed = inventory.changed();
    let mut was_idle = true;
    while publisher.view.registration_valid() {
        inventory.acknowledge(PublishedInventory::default());
        match publisher.rebuild(&inventory).await {
            Ok(revision) => inventory.acknowledge(PublishedInventory {
                sequence: publisher.progress.sequence,
                revision,
                ready: true,
            }),
            Err(error) => {
                log::warn!("Inventory reconciliation failed: {error}");
                tokio::time::sleep(Duration::from_secs(1)).await;
                continue;
            }
        }
        loop {
            let notified = changed.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            let through = inventory.sequence();
            if !publisher.view.registration_valid() {
                return;
            }
            if publisher.progress.sequence == through {
                was_idle = true;
                tokio::select! { _ = notified => {}, _ = tokio::time::sleep(Duration::from_secs(1)) => {} }
                continue;
            }
            let coalescing_wait = if was_idle {
                inventory.wait_to_publish(publisher.progress.sequence).await
            } else {
                Duration::ZERO
            };
            let through = inventory.sequence();
            let delta = match inventory.coalesced_changes(publisher.progress.sequence, through) {
                Ok(delta) => delta,
                Err(error) => {
                    log::warn!(
                        "Inventory publication history unavailable; rebuilding complete snapshot: {error:?}"
                    );
                    break;
                }
            };
            let input_records = delta.input_records;
            let input_bytes = delta.input_bytes;
            let output_records = delta.records.len();
            let sequence = delta.through;
            match publisher.interval(delta, true).await {
                Ok(published) => {
                    inventory.record_delta_publication(
                        input_records,
                        input_bytes,
                        output_records,
                        published.transactions,
                        published.encoded_bytes,
                        coalescing_wait,
                    );
                    inventory.acknowledge(PublishedInventory {
                        sequence,
                        revision: published.revision,
                        ready: true,
                    });
                    was_idle = false;
                }
                Err(error) => {
                    log::warn!("Inventory publication stopped: {error}");
                    inventory.acknowledge(PublishedInventory::default());
                    return;
                }
            }
        }
    }
}

impl Publisher {
    async fn rebuild(&mut self, inventory: &ResidencyInventory) -> Result<i64, String> {
        let block_prefix = format!("{}blocks/{}/", self.prefix, self.member.owner.incarnation);
        self.commit(
            vec![TxnOp::delete(
                block_prefix,
                Some(DeleteOptions::new().with_prefix()),
            )],
            0,
            false,
        )
        .await?;
        let start = inventory.sequence();
        let mut cursor = None;
        loop {
            let records = inventory
                .page(cursor.as_ref())
                .map_err(|error| format!("inventory snapshot: {error:?}"))?;
            if records.is_empty() {
                break;
            }
            for batch in records.chunks(MAX_BATCH_RECORDS) {
                self.records(batch, start, false).await?;
            }
            let last = records.last().ok_or("empty inventory page")?;
            cursor = Some((
                last.key.clone(),
                last.metadata.ok_or("missing residency metadata")?.medium,
            ));
            tokio::task::yield_now().await;
        }
        let end = inventory.sequence();
        let mut after = start;
        while after < end {
            let delta = inventory
                .coalesced_changes(after, end)
                .map_err(|_| "inventory changed beyond retained snapshot history")?;
            if delta.records.is_empty() {
                return Err("incomplete inventory replay".into());
            }
            after = delta.through;
            self.interval(delta, false).await?;
        }
        self.commit(Vec::new(), end, true).await
    }

    async fn interval(
        &mut self,
        delta: InventoryDelta,
        ready: bool,
    ) -> Result<PublishedInterval, String> {
        if delta.records.is_empty() {
            return self
                .commit(Vec::new(), delta.through, ready)
                .await
                .map(|revision| PublishedInterval {
                    revision,
                    transactions: 1,
                    encoded_bytes: 0,
                });
        }
        let previous_sequence = self.progress.sequence;
        let previous_ready = self.progress.ready;
        let transactions = delta.records.len().div_ceil(MAX_BATCH_RECORDS);
        let mut revision = 0;
        let mut encoded_bytes = 0;
        for (position, batch) in delta.records.chunks(MAX_BATCH_RECORDS).enumerate() {
            let final_batch = position + 1 == transactions;
            let sequence = if final_batch {
                delta.through
            } else {
                previous_sequence
            };
            let batch_ready = if final_batch { ready } else { previous_ready };
            revision = self.records(batch, sequence, batch_ready).await?;
            encoded_bytes += self.encoded_bytes(batch)?;
        }
        Ok(PublishedInterval {
            revision,
            transactions,
            encoded_bytes,
        })
    }

    async fn records(
        &mut self,
        records: &[InventoryRecord],
        sequence: u64,
        ready: bool,
    ) -> Result<i64, String> {
        let mut changes = BTreeMap::new();
        let mut bytes = 0usize;
        for record in records {
            let key = record_key(&self.prefix, self.member.owner.incarnation, record)?;
            let operation = if record.present {
                let encoded = orbitkv_proto::proto::engine::InventoryRecord::from(record.clone())
                    .encode_to_vec();
                if encoded.len() > MAX_RECORD_BYTES {
                    return Err("residency record too large".into());
                }
                bytes += key.len() + encoded.len();
                TxnOp::put(
                    key.clone(),
                    encoded,
                    Some(PutOptions::new().with_lease(self.member.lease)),
                )
            } else {
                bytes += key.len();
                TxnOp::delete(key.clone(), None)
            };
            changes.insert(key, operation);
        }
        if bytes > orbitkv_state::INVENTORY_BATCH_BYTES + MAX_BATCH_RECORDS * 128 {
            return Err("inventory transaction exceeds byte budget".into());
        }
        self.commit(changes.into_values().collect(), sequence, ready)
            .await
    }

    fn encoded_bytes(&self, records: &[InventoryRecord]) -> Result<usize, String> {
        records.iter().try_fold(0usize, |bytes, record| {
            let key = record_key(&self.prefix, self.member.owner.incarnation, record)?;
            let value = if record.present {
                orbitkv_proto::proto::engine::InventoryRecord::from(record.clone()).encoded_len()
            } else {
                0
            };
            bytes
                .checked_add(key.len() + value)
                .ok_or_else(|| "inventory transaction byte count overflow".into())
        })
    }

    async fn commit(
        &mut self,
        mut operations: Vec<TxnOp>,
        sequence: u64,
        ready: bool,
    ) -> Result<i64, String> {
        let progress = Progress {
            operation: self
                .progress
                .operation
                .checked_add(1)
                .ok_or("publisher operation exhausted")?,
            sequence,
            ready,
        };
        let marker = format!(
            "{}publishers/{}",
            self.prefix, self.member.owner.incarnation
        );
        let value = serde_json::to_vec(&progress).map_err(|error| error.to_string())?;
        operations.push(TxnOp::put(
            marker.clone(),
            value.clone(),
            Some(PutOptions::new().with_lease(self.member.lease)),
        ));
        let member_key = format!("{}members/{}", self.prefix, self.member.node_id);
        let member_value = serde_json::to_vec(&self.member).map_err(|error| error.to_string())?;
        let cursor = match &self.previous {
            Some(previous) => Compare::value(marker.clone(), CompareOp::Equal, previous.clone()),
            None => Compare::version(marker.clone(), CompareOp::Equal, 0),
        };
        let transaction = Txn::new()
            .when([
                Compare::value(member_key.clone(), CompareOp::Equal, member_value),
                Compare::lease(member_key, CompareOp::Equal, self.member.lease),
                cursor,
            ])
            .and_then(operations);
        let mut delay = Duration::from_millis(100);
        loop {
            if !self.view.registration_valid() {
                return Err("publisher registration expired".into());
            }
            match rpc(self.client.txn(transaction.clone())).await {
                Ok(response) => {
                    if cluster_id(response.header())? != self.cluster {
                        self.view.fence();
                        return Err("coordinator changed during publication".into());
                    }
                    let revision = if response.succeeded() {
                        response
                            .header()
                            .ok_or("missing commit revision")?
                            .revision()
                    } else {
                        let response = match rpc(self.client.get(marker.clone(), None)).await {
                            Ok(response) => response,
                            Err(error) => {
                                log::warn!("Publisher reconciliation will retry: {error}");
                                tokio::time::sleep(delay).await;
                                continue;
                            }
                        };
                        if cluster_id(response.header())? != self.cluster {
                            self.view.fence();
                            return Err("coordinator changed during reconciliation".into());
                        }
                        let committed = response
                            .kvs()
                            .first()
                            .filter(|kv| kv.value() == value && kv.lease() == self.member.lease);
                        let Some(committed) = committed else {
                            self.view.fence();
                            return Err("publisher or member fencing rejected transaction".into());
                        };
                        committed.mod_revision()
                    };
                    self.previous = Some(value);
                    self.progress = progress;
                    return Ok(revision);
                }
                Err(error) => log::warn!(
                    "Inventory transaction will retry without advancing its cursor: {error}"
                ),
            }
            tokio::time::sleep(delay).await;
            delay = (delay * 2).min(Duration::from_secs(3));
        }
    }
}

struct PublishedInterval {
    revision: i64,
    transactions: usize,
    encoded_bytes: usize,
}

pub(super) fn record_key(
    prefix: &str,
    owner: uuid::Uuid,
    record: &InventoryRecord,
) -> Result<String, String> {
    let metadata = record.metadata.ok_or("missing inventory medium")?;
    let medium = match metadata.medium {
        ReplicaMedium::Dram => "dram",
        ReplicaMedium::Ssd => "ssd",
        _ => return Err("unsupported inventory medium".into()),
    };
    let mut digest = Sha256::new();
    digest.update(b"orbitkv/state-location/v2\0");
    digest.update((record.key.namespace.len() as u64).to_le_bytes());
    digest.update(record.key.namespace.as_bytes());
    digest.update(&record.key.hash);
    Ok(format!(
        "{prefix}blocks/{owner}/{:x}/{medium}",
        digest.finalize()
    ))
}

#[cfg(test)]
#[path = "../../tests/unit/cluster/publish.rs"]
mod tests;
