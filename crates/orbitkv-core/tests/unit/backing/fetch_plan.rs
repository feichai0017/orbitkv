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
    async fn fetch_segment(&self, segment: &FetchSegment, _req_id: &str) -> SegmentOutcome {
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
async fn stale_candidate_uses_alternative_without_skipping_prefix_or_retrying_payload_failure() {
    for (response, expected) in [(SegmentOutcome::Rejected, 2), (SegmentOutcome::Failed, 0)] {
        let fetcher = Fetcher {
            responses: Mutex::new(VecDeque::from([response])),
            calls: Mutex::new(Vec::new()),
        };
        let plan = FetchPlan::new(vec![row(1, &["a", "b"]), row(2, &["a", "b"])]).unwrap();
        let (fetched, _, _) = execute_fetch_plan(&fetcher, &plan, "test-request").await;
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
    assert!(
        execute_fetch_plan(&fetcher, &plan, "test-request")
            .await
            .0
            .is_empty()
    );
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
    assert_eq!(
        execute_fetch_plan(&fetcher, &plan, "test-request")
            .await
            .0
            .len(),
        1
    );
    assert_eq!(*fetcher.calls.lock().unwrap(), ["a"]);
}
