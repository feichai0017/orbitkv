use crate::planning::peer::{FetchPlan, FetchSegment};

use crate::storage::MaterializedBlocks;

const MAX_AUTHORIZATION_RETRIES: usize = 2;

pub(super) enum SegmentOutcome {
    Fetched(MaterializedBlocks),
    /// Rejected during authorization, before any payload transfer was submitted.
    /// Stale evidence and transient source admission both permit another owner.
    Rejected,
    Failed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FetchStatus {
    Complete,
    AuthorizationExhausted,
    PayloadFailed,
}

pub(crate) struct FetchResult {
    pub(crate) blocks: MaterializedBlocks,
    pub(crate) status: FetchStatus,
    pub(crate) attempts: usize,
    pub(crate) completed_segments: usize,
}

impl FetchResult {
    pub(crate) fn can_replan(&self) -> bool {
        self.status == FetchStatus::AuthorizationExhausted && self.blocks.is_empty()
    }
}

#[tonic::async_trait]
pub(super) trait SegmentFetcher {
    async fn fetch_segment(&self, segment: &FetchSegment, req_id: &str) -> SegmentOutcome;
}

pub(super) async fn execute_fetch_plan<F: SegmentFetcher>(
    fetcher: &F,
    mut plan: FetchPlan<'_>,
    req_id: &str,
) -> FetchResult {
    let mut fetched = Vec::with_capacity(plan.block_count());
    let mut attempts = 0;
    let mut completed = 0;
    let mut rejected = 0;
    let mut status = FetchStatus::AuthorizationExhausted;
    while let Some(segment) = plan.next_segment(fetched.len()) {
        attempts += 1;
        match fetcher.fetch_segment(&segment, req_id).await {
            SegmentOutcome::Rejected => {
                // Only discard this batch's evidence. A later key may still be valid.
                plan.reject(fetched.len(), &segment);
                rejected += 1;
                if rejected > MAX_AUTHORIZATION_RETRIES {
                    break;
                }
            }
            SegmentOutcome::Failed => {
                status = FetchStatus::PayloadFailed;
                break;
            }
            SegmentOutcome::Fetched(returned) => {
                let contiguous = returned
                    .iter()
                    .zip(&segment.records)
                    .take_while(|((key, _), record)| *key == record.key)
                    .count();
                let returned_count = returned.len();
                fetched.extend(returned.into_iter().take(contiguous));
                if contiguous != segment.records.len() || returned_count != segment.records.len() {
                    status = FetchStatus::PayloadFailed;
                    break;
                }
                completed += 1;
            }
        }
    }
    if fetched.len() == plan.block_count() {
        status = FetchStatus::Complete;
    }
    FetchResult {
        blocks: fetched,
        status,
        attempts,
        completed_segments: completed,
    }
}

#[cfg(test)]
#[path = "../../tests/unit/peer/execute.rs"]
mod tests;
