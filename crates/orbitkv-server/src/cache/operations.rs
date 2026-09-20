use std::sync::{Arc, Mutex};

use orbitkv_common::hll::MultiWindowHllTracker;
use orbitkv_core::QueryLeaseId;
use orbitkv_core::{EngineError, LayerSave, OrbitKVEngine, PrefetchStatus};
use thiserror::Error;

#[derive(Clone, Debug)]
pub(crate) struct QueryInput {
    pub instance_id: String,
    pub block_hashes: Vec<Vec<u8>>,
    pub request_id: String,
    pub wait_for_full_prefix: bool,
    pub group_id: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum QueryOutcome {
    Loading,
    Ready {
        num_hit_blocks: u64,
        lease: Vec<u8>,
        hit_positions: Vec<u32>,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct PublishLayerInput {
    pub layer_name: String,
    pub block_ids: Vec<u32>,
    pub block_hashes: Vec<Vec<u8>>,
}

#[derive(Clone, Debug)]
pub(crate) struct PublishInput {
    pub instance_id: String,
    pub tp_rank: u32,
    pub pp_rank: u32,
    pub device_id: i32,
    pub layers: Vec<PublishLayerInput>,
}

#[derive(Clone, Debug)]
pub(crate) struct RestoreLeaseInput {
    pub lease: Vec<u8>,
    pub block_ids_by_group: Vec<Vec<Option<u32>>>,
}

#[derive(Clone, Debug)]
pub(crate) struct RestoreInput {
    pub instance_id: String,
    pub tp_rank: u32,
    pub device_id: i32,
    pub layer_groups: Vec<Vec<String>>,
    pub loads: Vec<RestoreLeaseInput>,
}

pub(crate) fn execute_restore(
    engine: &OrbitKVEngine,
    input: RestoreInput,
) -> Result<tokio::sync::oneshot::Receiver<Result<(), EngineError>>, EngineError> {
    if input.device_id < 0 {
        return Err(EngineError::InvalidArgument(format!(
            "device_id {} must be >= 0",
            input.device_id
        )));
    }
    let tp_rank = usize::try_from(input.tp_rank).map_err(|_| {
        EngineError::InvalidArgument(format!("tp_rank {} does not fit usize", input.tp_rank))
    })?;
    let loads = input
        .loads
        .into_iter()
        .map(|load| {
            let lease =
                QueryLeaseId::from_bytes(&load.lease).map_err(EngineError::InvalidArgument)?;
            let groups = load
                .block_ids_by_group
                .into_iter()
                .map(|targets| {
                    targets
                        .into_iter()
                        .map(|target| target.map(|id| id as usize))
                        .collect()
                })
                .collect();
            Ok((lease, groups))
        })
        .collect::<Result<Vec<_>, EngineError>>()?;
    let layer_groups = input
        .layer_groups
        .iter()
        .map(|group| group.iter().map(String::as_str).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    engine.batch_load_kv_blocks_multi_layer_inproc(
        &input.instance_id,
        tp_rank,
        input.device_id,
        &layer_groups,
        &loads,
    )
}

pub(crate) async fn execute_publish(
    engine: &OrbitKVEngine,
    input: PublishInput,
) -> Result<(), EngineError> {
    if input.device_id < 0 {
        return Err(EngineError::InvalidArgument(format!(
            "device_id {} must be >= 0",
            input.device_id
        )));
    }
    let tp_rank = usize::try_from(input.tp_rank).map_err(|_| {
        EngineError::InvalidArgument(format!("tp_rank {} does not fit usize", input.tp_rank))
    })?;
    let pp_rank = usize::try_from(input.pp_rank).map_err(|_| {
        EngineError::InvalidArgument(format!("pp_rank {} does not fit usize", input.pp_rank))
    })?;
    let mut saves = Vec::with_capacity(input.layers.len());
    for layer in input.layers {
        if layer.block_ids.len() != layer.block_hashes.len() {
            return Err(EngineError::InvalidArgument(format!(
                "block_ids length {} does not match block_hashes {} for layer {}",
                layer.block_ids.len(),
                layer.block_hashes.len(),
                layer.layer_name
            )));
        }
        saves.push(LayerSave {
            layer_name: layer.layer_name,
            block_ids: layer.block_ids.into_iter().map(|id| id as usize).collect(),
            block_hashes: layer.block_hashes,
        });
    }
    engine
        .batch_save_kv_blocks_from_ipc(&input.instance_id, tp_rank, pp_rank, input.device_id, saves)
        .await
}

#[derive(Debug, Error)]
pub(crate) enum ReleaseError {
    #[error("{0}")]
    InvalidLease(String),
    #[error("query lease is unknown or expired")]
    UnknownOrExpired,
}

pub(crate) fn execute_release(engine: &OrbitKVEngine, lease: &[u8]) -> Result<(), ReleaseError> {
    let lease = QueryLeaseId::from_bytes(lease).map_err(ReleaseError::InvalidLease)?;
    if engine.release_query_lease(&lease) {
        Ok(())
    } else {
        Err(ReleaseError::UnknownOrExpired)
    }
}

pub(crate) async fn execute_query(
    engine: &OrbitKVEngine,
    hll_tracker: &Arc<Mutex<MultiWindowHllTracker>>,
    input: QueryInput,
) -> Result<QueryOutcome, EngineError> {
    if input.request_id.is_empty() {
        return Err(EngineError::InvalidArgument(
            "request_id must not be empty".to_string(),
        ));
    }

    if input.group_id > 0 && input.wait_for_full_prefix {
        let status = engine
            .query_group_membership_with_fetch(
                &input.instance_id,
                &input.request_id,
                input.group_id,
                &input.block_hashes,
            )
            .await?;
        return match status {
            PrefetchStatus::Ready { blocks, .. } => {
                let complete = blocks.len() == input.block_hashes.len();
                let hit_positions: Vec<u32> = (0..blocks.len() as u32).collect();
                let lease = if complete && !blocks.is_empty() {
                    engine
                        .create_query_lease(&input.instance_id, blocks)?
                        .to_bytes()
                        .to_vec()
                } else {
                    Vec::new()
                };
                Ok(QueryOutcome::Ready {
                    num_hit_blocks: hit_positions.len() as u64,
                    lease,
                    hit_positions,
                })
            }
            PrefetchStatus::Loading => Ok(QueryOutcome::Loading),
        };
    }

    if input.group_id > 0 {
        let hits = engine.query_group_membership(
            &input.instance_id,
            input.group_id,
            &input.block_hashes,
        )?;
        let mut hit_positions = Vec::new();
        let mut blocks = Vec::new();
        for (position, block) in hits.into_iter().enumerate() {
            if let Some(block) = block {
                hit_positions.push(position as u32);
                blocks.push(block);
            }
        }
        let lease = if blocks.is_empty() {
            Vec::new()
        } else {
            engine
                .create_query_lease(&input.instance_id, blocks)?
                .to_bytes()
                .to_vec()
        };
        return Ok(QueryOutcome::Ready {
            num_hit_blocks: hit_positions.len() as u64,
            lease,
            hit_positions,
        });
    }

    let status = engine
        .count_prefix_hit_blocks_with_prefetch(
            &input.instance_id,
            &input.request_id,
            &input.block_hashes,
            input.wait_for_full_prefix,
        )
        .await?;
    match status {
        PrefetchStatus::Ready { blocks, missing } => {
            let hit = blocks.len();
            let miss_count = missing.min(input.block_hashes.len());
            let miss_start = input.block_hashes.len() - miss_count;
            debug_assert_eq!(hit + miss_count, input.block_hashes.len());
            if let Ok(namespace) = engine.instance_namespace(&input.instance_id)
                && let Ok(mut tracker) = hll_tracker.lock()
            {
                tracker.record_namespaced_misses(
                    &namespace,
                    input.block_hashes.len() as u64,
                    &input.block_hashes[miss_start..],
                );
            }
            let lease = if hit == 0 {
                Vec::new()
            } else {
                engine
                    .create_query_lease(&input.instance_id, blocks)?
                    .to_bytes()
                    .to_vec()
            };
            Ok(QueryOutcome::Ready {
                num_hit_blocks: hit as u64,
                lease,
                hit_positions: Vec::new(),
            })
        }
        PrefetchStatus::Loading => Ok(QueryOutcome::Loading),
    }
}
