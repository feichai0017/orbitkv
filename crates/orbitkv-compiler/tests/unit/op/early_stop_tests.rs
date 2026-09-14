use super::early_stop_exceeded;
use std::time::Duration;

#[test]
fn test_early_stop_exceeded() {
    let best = Duration::from_millis(5);
    // 2x cutoff: 10ms mean is at the boundary, not over it.
    assert!(!early_stop_exceeded(Duration::from_millis(10), best, 2.0));
    assert!(early_stop_exceeded(Duration::from_millis(11), best, 2.0));
    // A candidate faster than best never stops early.
    assert!(!early_stop_exceeded(Duration::from_millis(4), best, 2.0));
    // Factor 1.0 stops anything slower than best.
    assert!(early_stop_exceeded(Duration::from_millis(6), best, 1.0));
}
