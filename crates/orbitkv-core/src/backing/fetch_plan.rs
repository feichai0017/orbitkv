use orbitkv_state::{
    BlockCandidates, CacheOwner, DISCOVERY_MAX_BYTES, DISCOVERY_MAX_KEYS, InventoryRecord,
};

use super::PrefetchResult;

const MAX_STALE_RETRIES: usize = 2;

pub(super) struct FetchSegment {
    pub(super) owner: CacheOwner,
    pub(super) records: Vec<InventoryRecord>,
}

pub(crate) struct FetchPlan {
    rows: Vec<BlockCandidates>,
}

impl FetchPlan {
    pub(super) fn new(mut rows: Vec<BlockCandidates>) -> Option<Self> {
        let prefix = rows
            .iter()
            .take_while(|row| !row.replicas.is_empty())
            .count();
        rows.truncate(prefix);
        (!rows.is_empty()).then_some(Self { rows })
    }

    pub(crate) fn block_count(&self) -> usize {
        self.rows.len()
    }

    pub(super) fn matches(&self, namespace: &str, hashes: &[Vec<u8>]) -> bool {
        self.rows.len() <= hashes.len()
            && self
                .rows
                .iter()
                .zip(hashes)
                .all(|(row, hash)| row.key.namespace == namespace && row.key.hash == *hash)
    }
}

fn next_segment(rows: &[BlockCandidates], start: usize) -> Option<FetchSegment> {
    let row = rows.get(start)?;
    let mut best: Option<FetchSegment> = None;
    for candidate in &row.replicas {
        let mut bytes = row.key.namespace.len();
        let records: Vec<_> = rows[start..]
            .iter()
            .take(DISCOVERY_MAX_KEYS)
            .map_while(|row| {
                let replica = row.replicas.iter().find(|r| r.owner == candidate.owner)?;
                bytes = bytes.saturating_add(row.key.hash.len());
                (bytes <= DISCOVERY_MAX_BYTES).then(|| InventoryRecord {
                    key: row.key.clone(),
                    sequence: replica.sequence,
                    present: true,
                })
            })
            .collect();
        if records.is_empty() {
            continue;
        }
        if best.as_ref().is_none_or(|best| {
            records.len() > best.records.len()
                || (records.len() == best.records.len() && candidate.owner < best.owner)
        }) {
            best = Some(FetchSegment {
                owner: candidate.owner.clone(),
                records,
            });
        }
    }
    best
}

pub(super) enum SegmentOutcome {
    Fetched(PrefetchResult),
    /// Rejected during authorization, before any payload transfer was submitted.
    Rejected,
    Failed,
}

#[tonic::async_trait]
pub(super) trait SegmentFetcher {
    async fn fetch_segment(&self, segment: &FetchSegment) -> SegmentOutcome;
}

pub(super) async fn execute_fetch_plan<F: SegmentFetcher>(
    fetcher: &F,
    plan: &FetchPlan,
) -> (PrefetchResult, usize, usize) {
    let mut rows = plan.rows.clone();
    let mut fetched = Vec::with_capacity(rows.len());
    let mut attempts = 0;
    let mut completed = 0;
    let mut rejected = 0;
    while let Some(segment) = next_segment(&rows, fetched.len()) {
        attempts += 1;
        match fetcher.fetch_segment(&segment).await {
            SegmentOutcome::Rejected => {
                // Only discard this batch's evidence. A later key may still be valid.
                for row in &mut rows[fetched.len()..][..segment.records.len()] {
                    row.replicas.retain(|r| r.owner != segment.owner);
                }
                rejected += 1;
                if rejected > MAX_STALE_RETRIES {
                    break;
                }
            }
            SegmentOutcome::Failed => break,
            SegmentOutcome::Fetched(returned) => {
                let contiguous = returned
                    .iter()
                    .zip(&segment.records)
                    .take_while(|((key, _), record)| *key == record.key)
                    .count();
                let returned_count = returned.len();
                fetched.extend(returned.into_iter().take(contiguous));
                if contiguous != segment.records.len() || returned_count != segment.records.len() {
                    break;
                }
                completed += 1;
            }
        }
    }
    (fetched, attempts, completed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::SealedBlock;
    use orbitkv_state::{ReplicaLocation, StateKey};
    use std::{
        collections::VecDeque,
        sync::{Arc, Mutex},
    };

    fn row(hash: u8, owners: &[&str]) -> BlockCandidates {
        BlockCandidates {
            key: StateKey::new("ns".into(), vec![hash]),
            replicas: owners
                .iter()
                .map(|owner| ReplicaLocation {
                    owner: CacheOwner {
                        endpoint: (*owner).into(),
                        incarnation: uuid::Uuid::from_u128(1),
                    },
                    sequence: u64::from(hash),
                })
                .collect(),
        }
    }

    struct Fetcher {
        responses: Mutex<VecDeque<SegmentOutcome>>,
        calls: Mutex<Vec<String>>,
    }
    #[tonic::async_trait]
    impl SegmentFetcher for Fetcher {
        async fn fetch_segment(&self, segment: &FetchSegment) -> SegmentOutcome {
            self.calls
                .lock()
                .unwrap()
                .push(segment.owner.endpoint.clone());
            self.responses
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or_else(|| {
                    SegmentOutcome::Fetched(
                        segment
                            .records
                            .iter()
                            .map(|r| (r.key.clone(), Arc::new(SealedBlock::from_slots(Vec::new()))))
                            .collect(),
                    )
                })
        }
    }

    #[test]
    fn planner_selects_longest_cover_then_stable_owner_and_stops_at_gap() {
        let rows = vec![
            row(1, &["c", "b", "a"]),
            row(2, &["c", "b"]),
            row(3, &["d"]),
            row(4, &[]),
            row(5, &["a"]),
        ];
        let plan = FetchPlan::new(rows).unwrap();
        assert_eq!(plan.block_count(), 3);
        let first = next_segment(&plan.rows, 0).unwrap();
        assert_eq!(first.owner.endpoint, "b");
        assert_eq!(
            first.records.iter().map(|r| r.sequence).collect::<Vec<_>>(),
            [1, 2]
        );
        assert_eq!(next_segment(&plan.rows, 2).unwrap().owner.endpoint, "d");
        let mut rows = vec![row(1, &["a"]); DISCOVERY_MAX_KEYS + 1];
        assert_eq!(
            next_segment(&rows, 0).unwrap().records.len(),
            DISCOVERY_MAX_KEYS
        );
        rows[0].key.hash = vec![1; DISCOVERY_MAX_BYTES - 2];
        assert_eq!(next_segment(&rows, 0).unwrap().records.len(), 1);
    }

    #[tokio::test]
    async fn stale_candidate_uses_alternative_without_skipping_prefix_or_retrying_payload_failure()
    {
        for (response, expected) in [(SegmentOutcome::Rejected, 2), (SegmentOutcome::Failed, 0)] {
            let fetcher = Fetcher {
                responses: Mutex::new(VecDeque::from([response])),
                calls: Mutex::new(Vec::new()),
            };
            let plan = FetchPlan::new(vec![row(1, &["a", "b"]), row(2, &["a", "b"])]).unwrap();
            let (fetched, _, _) = execute_fetch_plan(&fetcher, &plan).await;
            assert_eq!(fetched.len(), expected);
            assert_eq!(
                fetcher.calls.lock().unwrap().len(),
                if expected == 2 { 2 } else { 1 }
            );
        }
        let fetcher = Fetcher {
            responses: Mutex::new(VecDeque::from([
                SegmentOutcome::Rejected,
                SegmentOutcome::Rejected,
                SegmentOutcome::Rejected,
            ])),
            calls: Mutex::new(Vec::new()),
        };
        let plan = FetchPlan::new(vec![row(1, &["a", "b", "c", "d"])]).unwrap();
        assert!(execute_fetch_plan(&fetcher, &plan).await.0.is_empty());
        assert_eq!(fetcher.calls.lock().unwrap().len(), 3);
    }

    #[tokio::test]
    async fn malformed_or_short_segment_never_skips_a_gap() {
        let fetcher = Fetcher {
            responses: Mutex::new(VecDeque::from([SegmentOutcome::Fetched(vec![
                (
                    StateKey::new("ns".into(), vec![1]),
                    Arc::new(SealedBlock::from_slots(Vec::new())),
                ),
                (
                    StateKey::new("wrong-model".into(), vec![2]),
                    Arc::new(SealedBlock::from_slots(Vec::new())),
                ),
            ])])),
            calls: Mutex::new(Vec::new()),
        };
        let plan = FetchPlan::new(vec![row(1, &["a"]), row(2, &["a"]), row(3, &["b"])]).unwrap();
        assert_eq!(execute_fetch_plan(&fetcher, &plan).await.0.len(), 1);
        assert_eq!(*fetcher.calls.lock().unwrap(), ["a"]);
    }
}
