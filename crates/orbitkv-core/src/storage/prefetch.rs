//! Fetch backing blocks in the caller-owned query future.

use std::collections::HashMap;
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};
use tokio::sync::OnceCell;

use log::warn;
use parking_lot::Mutex;

use crate::QueryMode;
#[cfg(feature = "mooncake")]
use crate::backing::MooncakeFetchStore;
use crate::backing::{PrefetchResult, SsdBackingStore};
use crate::block::{QueryResult, RestoreSource, SealedBlock, StateKey};
use crate::metrics::core_metrics;

use super::read_cache::ReadCache;
use super::tier_attribution::{
    AttributionSource, TierAttribution, record_cache_tier_block_requests,
};

const REMOTE_WAIT_POLL_INTERVAL: Duration = Duration::from_millis(10);
const REMOTE_WAIT_TIMEOUT: Duration = Duration::from_secs(30);

#[cfg(feature = "mooncake")]
#[derive(Clone)]
pub(super) struct RemoteFetch(Arc<MooncakeFetchStore>);
#[cfg(not(feature = "mooncake"))]
#[derive(Clone)]
pub(super) struct RemoteFetch;

#[cfg(feature = "mooncake")]
impl RemoteFetch {
    pub(super) fn new(store: Arc<MooncakeFetchStore>) -> Self {
        Self(store)
    }

    async fn try_fetch_prefix(
        &self,
        req_id: &str,
        namespace: &str,
        remaining_hashes: &[Vec<u8>],
        require_full_prefix: bool,
    ) -> Option<(usize, PrefetchResult)> {
        let plan = self.0.query_plan(namespace, remaining_hashes).await?;
        let found = plan.block_count();
        if require_full_prefix && found != remaining_hashes.len() {
            return None;
        }
        let blocks = self
            .0
            .fetch_plan(&plan, req_id, namespace, remaining_hashes)
            .await;
        if require_full_prefix && blocks.len() != found {
            // Complete this query with the partial result; do not retry a
            // stale advertisement throughout the producer-wait deadline.
            warn!(
                "Mooncake fetch returned fewer blocks than planned: req_id={} returned={} planned={}",
                req_id,
                blocks.len(),
                found
            );
        }
        Some((found, blocks))
    }
}

#[cfg(not(feature = "mooncake"))]
impl RemoteFetch {
    async fn try_fetch_prefix(
        &self,
        _req_id: &str,
        _namespace: &str,
        _remaining_hashes: &[Vec<u8>],
        _require_full_prefix: bool,
    ) -> Option<(usize, PrefetchResult)> {
        None
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum PrefetchSource {
    Ssd,
    Remote,
}

impl PrefetchSource {
    const fn as_attribution(self) -> AttributionSource {
        match self {
            Self::Ssd => AttributionSource::Ssd,
            Self::Remote => AttributionSource::Remote,
        }
    }
}

#[derive(Clone)]
struct PrefetchTaskResult {
    source: Option<PrefetchSource>,
    cache_inserts: PrefetchResult,
    ready_blocks: Vec<Arc<SealedBlock>>,
    missing: usize,
}

struct PrefetchTaskDeps {
    remote_fetch: Option<RemoteFetch>,
    ssd_store: Option<Arc<SsdBackingStore>>,
}

struct PrefetchTaskInput {
    req_id: String,
    namespace: String,
    remaining_keys: Vec<StateKey>,
    prefix_blocks: Vec<Arc<SealedBlock>>,
    total: usize,
    wait_for_full_prefix: bool,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct FetchKey {
    keys: Vec<StateKey>,
    hit: usize,
    wait_for_full_prefix: bool,
}
type SharedRead = OnceCell<PrefetchTaskResult>;

pub(super) struct PrefetchScheduler {
    reads: Mutex<HashMap<FetchKey, Weak<SharedRead>>>,
    ssd_store: Option<Arc<SsdBackingStore>>,
    remote_fetch: Option<RemoteFetch>,
    codec_budget: usize,
}

impl PrefetchScheduler {
    pub(super) fn new(
        ssd_store: Option<Arc<SsdBackingStore>>,
        remote_fetch: Option<RemoteFetch>,
        codec_budget: usize,
    ) -> Self {
        Self {
            codec_budget,
            reads: Mutex::new(HashMap::new()),
            ssd_store,
            remote_fetch,
        }
    }

    pub(super) async fn check_and_prefetch(
        &self,
        read_cache: &ReadCache,
        req_id: &str,
        namespace: &str,
        hashes: &[Vec<u8>],
        mode: QueryMode,
    ) -> QueryResult {
        let warming = matches!(mode, QueryMode::Warmup | QueryMode::Prepare);
        let wait_for_full_prefix = mode == QueryMode::WaitForFullPrefix;
        let keys: Vec<StateKey> = hashes
            .iter()
            .map(|hash| StateKey::new(namespace.to_string(), hash.clone()))
            .collect();
        let (hit, prefix_blocks) = read_cache.get_prefix_blocks(&keys, warming);
        if hit == keys.len() || (self.remote_fetch.is_none() && self.ssd_store.is_none()) {
            if !warming && let Some(ssd) = &self.ssd_store {
                ssd.ingest_batch(keys.iter().zip(&prefix_blocks), true);
            }
            record_tier_attribution(keys.len(), hit, 0, None);
            return QueryResult {
                blocks: prefix_blocks
                    .into_iter()
                    .map(RestoreSource::Memory)
                    .collect(),
                missing: keys.len() - hit,
            };
        }

        // Demand leases can own disk extents without materializing host data.
        // Speculative preparation continues to fill DRAM before GPU pages exist.
        if !warming
            && let Some(ssd) = &self.ssd_store
            && let Some(disk) = ssd.pin_prefix(&keys[hit..], self.codec_budget)
            && !disk.is_empty()
            && (!wait_for_full_prefix || hit + disk.len() == keys.len())
        {
            let count = hit + disk.len();
            ssd.ingest_batch(keys.iter().zip(&prefix_blocks), true);
            record_tier_attribution(keys.len(), hit, disk.len(), Some(AttributionSource::Ssd));
            return QueryResult {
                blocks: prefix_blocks
                    .into_iter()
                    .map(RestoreSource::Memory)
                    .chain(disk.into_iter().map(RestoreSource::Ssd))
                    .collect(),
                missing: keys.len() - count,
            };
        }

        let key = FetchKey {
            keys: keys.clone(),
            hit,
            wait_for_full_prefix,
        };
        let read = {
            let mut reads = self.reads.lock();
            if let Some(read) = reads.get(&key).and_then(Weak::upgrade) {
                core_metrics().query_coalesced_reads.add(1, &[]);
                read
            } else {
                reads.retain(|_, read| read.strong_count() > 0);
                let read = Arc::new(SharedRead::new());
                reads.insert(key, Arc::downgrade(&read));
                read
            }
        };
        let result = read
            .get_or_init(|| async {
                let mut result = run_prefetch_task(
                    PrefetchTaskDeps {
                        remote_fetch: self.remote_fetch.clone(),
                        ssd_store: self.ssd_store.clone(),
                    },
                    PrefetchTaskInput {
                        req_id: req_id.to_string(),
                        namespace: namespace.to_string(),
                        remaining_keys: keys[hit..].to_vec(),
                        prefix_blocks,
                        total: keys.len(),
                        wait_for_full_prefix,
                    },
                )
                .await;
                let inserts = std::mem::take(&mut result.cache_inserts);
                if warming {
                    for (_, block) in &inserts {
                        block.mark_warmed();
                    }
                }
                if warming || result.source == Some(PrefetchSource::Remote) {
                    read_cache.batch_insert_reclaimable(inserts);
                } else {
                    read_cache.batch_insert(inserts);
                }
                result
            })
            .await;
        if !warming {
            read_cache.retain_demand(&keys, &result.ready_blocks);
            if let Some(ssd) = &self.ssd_store {
                ssd.ingest_batch(keys.iter().zip(&result.ready_blocks), true);
            }
        }
        record_tier_attribution(
            keys.len(),
            hit,
            result.ready_blocks.len() - hit,
            result.source.map(PrefetchSource::as_attribution),
        );
        QueryResult {
            blocks: result
                .ready_blocks
                .iter()
                .cloned()
                .map(RestoreSource::Memory)
                .collect(),
            missing: result.missing,
        }
    }
}

/// Attribute each query once, including any backing fetch.
fn record_tier_attribution(
    total: usize,
    hit: usize,
    loading: usize,
    loading_source: Option<AttributionSource>,
) {
    if total == 0 {
        return;
    }
    let attribution = TierAttribution::classify(total, hit, loading, loading_source);
    record_cache_tier_block_requests(total, attribution);
}

fn build_ready_result(
    prefix_blocks: Vec<Arc<SealedBlock>>,
    total: usize,
    source: Option<PrefetchSource>,

    requested_keys: &[StateKey],
    cache_inserts: PrefetchResult,
) -> PrefetchTaskResult {
    let mut ready_blocks = prefix_blocks;
    let inserts_by_key: HashMap<_, _> = cache_inserts
        .iter()
        .map(|(key, block)| (key, block))
        .collect();
    ready_blocks.extend(
        requested_keys
            .iter()
            .map_while(|key| inserts_by_key.get(key).map(|block| Arc::clone(*block))),
    );
    let missing = total.saturating_sub(ready_blocks.len());
    PrefetchTaskResult {
        source,
        cache_inserts,
        ready_blocks,
        missing,
    }
}

async fn run_prefetch_task(deps: PrefetchTaskDeps, input: PrefetchTaskInput) -> PrefetchTaskResult {
    let PrefetchTaskInput {
        req_id,
        namespace,
        remaining_keys,
        prefix_blocks,
        total,
        wait_for_full_prefix,
    } = input;
    let remaining_hashes: Vec<Vec<u8>> = remaining_keys.iter().map(|k| k.hash.clone()).collect();

    if let Some(remote) = deps.remote_fetch.as_ref()
        && let Some((found, blocks)) = remote
            .try_fetch_prefix(&req_id, &namespace, &remaining_hashes, wait_for_full_prefix)
            .await
    {
        return build_ready_result(
            prefix_blocks,
            total,
            Some(PrefetchSource::Remote),
            &remaining_keys[..found],
            blocks,
        );
    }

    if let Some(ssd) = deps.ssd_store.as_ref() {
        let found = ssd.prefix_len(&remaining_keys);
        if found > 0 && (!wait_for_full_prefix || found == remaining_keys.len()) {
            let (found, blocks) = ssd.prefetch_prefix(remaining_keys[..found].to_vec()).await;
            if found > 0 && (!wait_for_full_prefix || found == remaining_keys.len()) {
                return build_ready_result(
                    prefix_blocks,
                    total,
                    Some(PrefetchSource::Ssd),
                    &remaining_keys[..found],
                    blocks,
                );
            }
        }
    }

    if wait_for_full_prefix && let Some(remote) = deps.remote_fetch {
        let started_at = Instant::now();
        while started_at.elapsed() < REMOTE_WAIT_TIMEOUT {
            tokio::time::sleep(REMOTE_WAIT_POLL_INTERVAL).await;
            if let Some((found, blocks)) = remote
                .try_fetch_prefix(&req_id, &namespace, &remaining_hashes, true)
                .await
            {
                return build_ready_result(
                    prefix_blocks,
                    total,
                    Some(PrefetchSource::Remote),
                    &remaining_keys[..found],
                    blocks,
                );
            }
        }
        warn!(
            "Timed out waiting for remote prefix: req_id={} timeout_secs={}",
            req_id,
            REMOTE_WAIT_TIMEOUT.as_secs()
        );
    }

    build_ready_result(prefix_blocks, total, None, &[], Vec::new())
}

#[cfg(test)]
#[path = "../../tests/unit/storage/prefetch.rs"]
mod tests;
