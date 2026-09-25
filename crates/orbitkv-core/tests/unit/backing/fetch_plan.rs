use super::*;
use crate::block::SealedBlock;
use crate::planning::replica::ReplicaSet;
use orbitkv_state::{CacheOwner, ReplicaLocation, StateKey};
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

fn row(hash: u8, owners: &[&str]) -> ReplicaSet {
    let mut row = ReplicaSet::new(StateKey::new("ns".into(), vec![hash]));
    row.set_peer_dram(
        owners
            .iter()
            .map(|owner| ReplicaLocation {
                owner: CacheOwner {
                    endpoint: (*owner).into(),
                    incarnation: uuid::Uuid::from_u128(1),
                },
                sequence: u64::from(hash),
            })
            .collect(),
    );
    row
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

#[tokio::test]
async fn stale_candidate_uses_alternative_without_skipping_prefix_or_retrying_payload_failure() {
    for (response, expected) in [(SegmentOutcome::Rejected, 2), (SegmentOutcome::Failed, 0)] {
        let fetcher = Fetcher {
            responses: Mutex::new(VecDeque::from([response])),
            calls: Mutex::new(Vec::new()),
        };
        let mut rows = vec![row(1, &["a", "b"]), row(2, &["a", "b"])];
        let plan = FetchPlan::new(&mut rows, 1).unwrap();
        let (fetched, _, _) = execute_fetch_plan(&fetcher, plan, "test-request").await;
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
    let mut rows = vec![row(1, &["a", "b", "c", "d"])];
    let plan = FetchPlan::new(&mut rows, 1).unwrap();
    assert!(
        execute_fetch_plan(&fetcher, plan, "test-request")
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
    let mut rows = vec![row(1, &["a"]), row(2, &["a"]), row(3, &["b"])];
    let plan = FetchPlan::new(&mut rows, 1).unwrap();
    assert_eq!(
        execute_fetch_plan(&fetcher, plan, "test-request")
            .await
            .0
            .len(),
        1
    );
    assert_eq!(*fetcher.calls.lock().unwrap(), ["a"]);
}
