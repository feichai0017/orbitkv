use super::*;

#[test]
fn shadow_keeps_unknown_alternatives_unknown_and_never_uses_tier_rank() {
    let fast = Estimate {
        count: 10,
        seconds: 0.01,
        absolute_error: 0.0,
        updated: Instant::now(),
    };
    let slow = Estimate {
        seconds: 0.02,
        ..fast
    };
    assert_eq!(recommendation(&[Some(fast), None], 0), "unknown");
    assert_eq!(recommendation(&[None, Some(slow)], 1), "unknown");
    assert_eq!(recommendation(&[Some(fast)], 0), "unknown");
    assert_eq!(recommendation(&[Some(slow), Some(fast)], 0), "different");
    assert_eq!(recommendation(&[Some(fast), Some(slow)], 0), "agree");
}
