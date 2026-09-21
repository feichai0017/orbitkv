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
    async fn fetch_segment(&self, segment: &FetchSegment, req_id: &str) -> SegmentOutcome;
}

pub(super) async fn execute_fetch_plan<F: SegmentFetcher>(
    fetcher: &F,
    plan: &FetchPlan,
    req_id: &str,
) -> (PrefetchResult, usize, usize) {
    let mut rows = plan.rows.clone();
    let mut fetched = Vec::with_capacity(rows.len());
    let mut attempts = 0;
    let mut completed = 0;
    let mut rejected = 0;
    while let Some(segment) = next_segment(&rows, fetched.len()) {
        attempts += 1;
        match fetcher.fetch_segment(&segment, req_id).await {
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
#[path = "../../tests/unit/backing/fetch_plan.rs"]
mod tests;
