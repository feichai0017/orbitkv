use crate::planning::peer::{FetchPlan, FetchSegment};

use super::PrefetchResult;

const MAX_STALE_RETRIES: usize = 2;

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
    mut plan: FetchPlan<'_>,
    req_id: &str,
) -> (PrefetchResult, usize, usize) {
    let mut fetched = Vec::with_capacity(plan.block_count());
    let mut attempts = 0;
    let mut completed = 0;
    let mut rejected = 0;
    while let Some(segment) = plan.next_segment(fetched.len()) {
        attempts += 1;
        match fetcher.fetch_segment(&segment, req_id).await {
            SegmentOutcome::Rejected => {
                // Only discard this batch's evidence. A later key may still be valid.
                plan.reject(fetched.len(), &segment);
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
