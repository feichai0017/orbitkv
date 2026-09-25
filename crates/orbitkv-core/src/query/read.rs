//! Fetch backing blocks in the caller-owned query future.

use std::collections::HashMap;
use std::sync::{Arc, Weak};
#[cfg(feature = "mooncake")]
use std::time::{Duration, Instant};
use tokio::sync::OnceCell;

#[cfg(feature = "mooncake")]
use log::warn;
use parking_lot::Mutex;

use crate::QueryMode;
#[cfg(feature = "mooncake")]
use crate::backing::MooncakeFetchStore;
use crate::backing::{PrefetchResult, SsdBackingStore};
use crate::block::{QueryResult, RestoreSource, SealedBlock, StateKey};
use crate::metrics::core_metrics;

use super::tier_attribution::{
    AttributionSource, TierAttribution, record_cache_tier_block_requests,
};
#[cfg(feature = "mooncake")]
use crate::planning::peer::FetchPlan;
use crate::planning::read::ReadPlan;
use crate::storage::read_cache::ReadCache;

#[cfg(feature = "mooncake")]
const REMOTE_WAIT_POLL_INTERVAL: Duration = Duration::from_millis(10);
#[cfg(feature = "mooncake")]
const REMOTE_WAIT_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Clone)]
struct MaterializedRead {
    source: Option<AttributionSource>,
    cache_inserts: PrefetchResult,
    ready_blocks: Vec<Arc<SealedBlock>>,
    missing: usize,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct ReadKey {
    keys: Vec<StateKey>,
    hit: usize,
    wait_for_full_prefix: bool,
    allow_ssd_prefetch: bool,
}
type SharedRead = OnceCell<MaterializedRead>;

pub(crate) struct ReadCoordinator {
    reads: Mutex<HashMap<ReadKey, Weak<SharedRead>>>,
    ssd_store: Option<Arc<SsdBackingStore>>,
    #[cfg(feature = "mooncake")]
    remote_fetch: Option<Arc<MooncakeFetchStore>>,
    codec_budget: usize,
}

impl ReadCoordinator {
    pub(crate) fn new(
        ssd_store: Option<Arc<SsdBackingStore>>,
        #[cfg(feature = "mooncake")] remote_fetch: Option<Arc<MooncakeFetchStore>>,
        codec_budget: usize,
    ) -> Self {
        Self {
            codec_budget,
            reads: Mutex::new(HashMap::new()),
            ssd_store,
            #[cfg(feature = "mooncake")]
            remote_fetch,
        }
    }

    pub(crate) async fn read_prefix(
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
        #[cfg(feature = "mooncake")]
        let has_remote = self.remote_fetch.is_some();
        #[cfg(not(feature = "mooncake"))]
        let has_remote = false;
        if hit == keys.len() || (!has_remote && self.ssd_store.is_none()) {
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

        let mut plan = ReadPlan::new(&keys[hit..], mode, self.ssd_store.as_ref());
        if let Some(ssd) = &self.ssd_store
            && let Some(route) = plan.deferred_ssd(ssd, self.codec_budget)
        {
            let path = route.path;
            if let Some(leases) = route.acquire(self.codec_budget) {
                let count = hit + leases.len();
                ssd.ingest_batch(keys.iter().zip(&prefix_blocks), true);
                record_tier_attribution(
                    keys.len(),
                    hit,
                    leases.len(),
                    Some(AttributionSource::Ssd),
                );
                return QueryResult {
                    blocks: prefix_blocks
                        .into_iter()
                        .map(RestoreSource::Memory)
                        .chain(
                            leases
                                .into_iter()
                                .map(|lease| RestoreSource::Ssd { lease, path }),
                        )
                        .collect(),
                    missing: keys.len() - count,
                };
            }
        }

        let allow_ssd_prefetch = warming
            || self
                .ssd_store
                .as_ref()
                .is_none_or(|ssd| ssd.read_path.is_none());
        let key = ReadKey {
            keys: keys.clone(),
            hit,
            wait_for_full_prefix,
            allow_ssd_prefetch,
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
                let (source, blocks) = self
                    .materialize(&mut plan, req_id, allow_ssd_prefetch)
                    .await;
                let mut result =
                    build_ready_result(prefix_blocks, keys.len(), source, &keys[hit..], blocks);
                let inserts = std::mem::take(&mut result.cache_inserts);
                if warming {
                    for (_, block) in &inserts {
                        block.mark_warmed();
                    }
                }
                if warming || result.source == Some(AttributionSource::Remote) {
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
            result.source,
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

    async fn materialize(
        &self,
        plan: &mut ReadPlan,
        req_id: &str,
        allow_ssd_prefetch: bool,
    ) -> (Option<AttributionSource>, PrefetchResult) {
        #[cfg(feature = "mooncake")]
        if let Some(remote) = &self.remote_fetch {
            remote.discover(&mut plan.rows).await;
            if let Some(route) = FetchPlan::new(&mut plan.rows, plan.required) {
                return (
                    Some(AttributionSource::Remote),
                    remote.fetch_plan(route, req_id).await,
                );
            }
        }
        #[cfg(not(feature = "mooncake"))]
        let _ = req_id;

        if allow_ssd_prefetch
            && let Some(ssd) = &self.ssd_store
            && let Some(route) = plan.ssd(crate::SsdReadPath::Uring, self.codec_budget)
            && let Some(leases) = route.acquire(self.codec_budget)
        {
            return (
                Some(AttributionSource::Ssd),
                ssd.read_host_batch(leases).await.unwrap_or_default(),
            );
        }

        #[cfg(feature = "mooncake")]
        if plan.wait_for_full_prefix
            && let Some(remote) = &self.remote_fetch
        {
            let started_at = Instant::now();
            while started_at.elapsed() < REMOTE_WAIT_TIMEOUT {
                tokio::time::sleep(REMOTE_WAIT_POLL_INTERVAL).await;
                remote.discover(&mut plan.rows).await;
                if let Some(route) = FetchPlan::new(&mut plan.rows, plan.required) {
                    // A submitted payload failure completes this query; only
                    // missing advertisements participate in producer waiting.
                    return (
                        Some(AttributionSource::Remote),
                        remote.fetch_plan(route, req_id).await,
                    );
                }
            }
            warn!(
                "Timed out waiting for remote prefix: req_id={req_id} timeout_secs={}",
                REMOTE_WAIT_TIMEOUT.as_secs()
            );
        }
        (None, Vec::new())
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
    source: Option<AttributionSource>,
    requested_keys: &[StateKey],
    cache_inserts: PrefetchResult,
) -> MaterializedRead {
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
    MaterializedRead {
        source,
        cache_inserts,
        ready_blocks,
        missing,
    }
}

#[cfg(test)]
#[path = "../../tests/unit/query/read.rs"]
mod tests;
