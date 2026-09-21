//! Fetch backing blocks in the caller-owned query future.

use std::collections::HashMap;
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};
use tokio::sync::OnceCell;

use log::warn;
use parking_lot::Mutex;

#[cfg(feature = "mooncake")]
use crate::backing::MooncakeFetchStore;
use crate::backing::{PrefetchResult, SsdBackingStore};
use crate::block::{QueryResult, SealedBlock, StateKey};
use crate::internode::MetaServerClient;
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
    metaserver_client: Option<Arc<MetaServerClient>>,
}

impl PrefetchScheduler {
    pub(super) fn new(
        ssd_store: Option<Arc<SsdBackingStore>>,
        remote_fetch: Option<RemoteFetch>,
        metaserver_client: Option<Arc<MetaServerClient>>,
    ) -> Self {
        Self {
            reads: Mutex::new(HashMap::new()),
            ssd_store,
            remote_fetch,
            metaserver_client,
        }
    }

    pub(super) async fn check_and_prefetch(
        &self,
        read_cache: &ReadCache,
        req_id: &str,
        namespace: &str,
        hashes: &[Vec<u8>],
        wait_for_full_prefix: bool,
    ) -> QueryResult {
        let keys: Vec<StateKey> = hashes
            .iter()
            .map(|hash| StateKey::new(namespace.to_string(), hash.clone()))
            .collect();
        let (hit, prefix_blocks) = read_cache.get_prefix_blocks(&keys);
        if hit == keys.len() || (self.remote_fetch.is_none() && self.ssd_store.is_none()) {
            record_tier_attribution(keys.len(), hit, 0, None);
            return QueryResult {
                blocks: prefix_blocks,
                missing: keys.len() - hit,
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
                let registration = if result.source == Some(PrefetchSource::Remote) {
                    let resident_keys = read_cache.batch_insert_resident_keys(inserts);
                    remote_registration_from_resident_keys(result.source, &resident_keys)
                } else {
                    read_cache.batch_insert(inserts);
                    None
                };
                if let Some(client) = &self.metaserver_client
                    && let Some((namespace, hashes)) = registration
                {
                    client.try_register_namespace(namespace, hashes);
                }
                result
            })
            .await;
        record_tier_attribution(
            keys.len(),
            hit,
            result.ready_blocks.len() - hit,
            result.source.map(PrefetchSource::as_attribution),
        );
        QueryResult {
            blocks: result.ready_blocks.clone(),
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

fn remote_registration_from_resident_keys(
    source: Option<PrefetchSource>,
    resident_keys: &[StateKey],
) -> Option<(String, Vec<Vec<u8>>)> {
    if source != Some(PrefetchSource::Remote) || resident_keys.is_empty() {
        return None;
    }

    let namespace = resident_keys[0].namespace.clone();
    let hashes = resident_keys.iter().map(|key| key.hash.clone()).collect();
    Some((namespace, hashes))
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
mod tests {
    use super::*;

    fn key(n: u8) -> StateKey {
        StateKey::new("ns".to_string(), vec![n])
    }

    fn block() -> Arc<SealedBlock> {
        Arc::new(SealedBlock::from_slots(Vec::new()))
    }

    #[test]
    fn ready_result_rebuilds_prefix_in_requested_key_order() {
        let local = block();
        let k1 = key(1);
        let k2 = key(2);
        let k3 = key(3);
        let b1 = block();
        let b2 = block();
        let b3 = block();

        let result = build_ready_result(
            vec![Arc::clone(&local)],
            4,
            Some(PrefetchSource::Ssd),
            &[k1.clone(), k2.clone(), k3.clone()],
            vec![
                (k2, Arc::clone(&b2)),
                (k1, Arc::clone(&b1)),
                (k3, Arc::clone(&b3)),
            ],
        );

        assert_eq!(result.ready_blocks.len(), 4);
        assert!(Arc::ptr_eq(&result.ready_blocks[0], &local));
        assert!(Arc::ptr_eq(&result.ready_blocks[1], &b1));
        assert!(Arc::ptr_eq(&result.ready_blocks[2], &b2));
        assert!(Arc::ptr_eq(&result.ready_blocks[3], &b3));
        assert_eq!(result.missing, 0);
        assert_eq!(result.cache_inserts.len(), 3);
    }

    #[test]
    fn ready_result_stops_at_first_missing_prefetch_key() {
        let k1 = key(1);
        let k2 = key(2);
        let k3 = key(3);
        let b1 = block();
        let b3 = block();

        let result = build_ready_result(
            Vec::new(),
            3,
            Some(PrefetchSource::Ssd),
            &[k1.clone(), k2, k3.clone()],
            vec![(k3, b3), (k1, Arc::clone(&b1))],
        );

        assert_eq!(result.ready_blocks.len(), 1);
        assert!(Arc::ptr_eq(&result.ready_blocks[0], &b1));
        assert_eq!(result.missing, 2);
        assert_eq!(result.cache_inserts.len(), 2);
    }

    #[test]
    fn remote_registration_uses_only_resident_keys() {
        let k1 = key(1);
        let k3 = key(3);

        let (namespace, hashes) =
            remote_registration_from_resident_keys(Some(PrefetchSource::Remote), &[k1, k3])
                .expect("remote resident keys should register");

        assert_eq!(namespace, "ns");
        assert_eq!(hashes, vec![vec![1], vec![3]]);
    }

    #[test]
    fn remote_registration_skips_ssd_and_empty_resident_keys() {
        let k1 = key(1);

        assert!(remote_registration_from_resident_keys(Some(PrefetchSource::Ssd), &[k1]).is_none());
        assert!(
            remote_registration_from_resident_keys(Some(PrefetchSource::Remote), &[]).is_none()
        );
        assert!(remote_registration_from_resident_keys(None, &[]).is_none());
    }
}
