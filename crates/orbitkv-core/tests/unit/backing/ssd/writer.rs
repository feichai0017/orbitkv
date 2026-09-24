use super::*;

#[test]
fn ordered_batch_observations_retain_failures_and_finish_at_their_own_boundary() {
    let mut batches = VecDeque::from([
        WriteBatchObservation {
            remaining: 3,
            outcome: Outcome::Cancelled, // A weak queued source already expired.
            observation: Observation::disabled(),
        },
        WriteBatchObservation {
            remaining: 2,
            outcome: Outcome::Completed,
            observation: Observation::disabled(),
        },
    ]);
    complete_write_batch(&mut batches, true, 4096);
    assert_eq!(batches.len(), 2);
    assert_eq!(batches[0].outcome, Outcome::Cancelled);
    complete_write_batch(&mut batches, false, 4096);
    assert_eq!(batches[0].outcome, Outcome::Failed);
    complete_write_batch(&mut batches, false, 0);
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].remaining, 2);
    assert_eq!(batches[0].outcome, Outcome::Completed);

    complete_write_batch(&mut batches, false, 0);
    assert_eq!(batches[0].outcome, Outcome::Cancelled);
    complete_write_batch(&mut batches, true, 4096);
    assert!(batches.is_empty());
}
