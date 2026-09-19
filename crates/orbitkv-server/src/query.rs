use std::sync::{Arc, Mutex};

use orbitkv_common::hll::MultiWindowHllTracker;
use orbitkv_core::{EngineError, OrbitKVEngine, PrefetchStatus};

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
