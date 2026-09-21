use super::*;

#[test]
fn cache_residence_duration_boundaries_cover_one_second_to_one_day() {
    assert_eq!(
        cache_residence_duration_seconds_boundaries(),
        vec![
            1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1_800.0, 3_600.0, 7_200.0, 21_600.0,
            43_200.0, 86_400.0,
        ]
    );
}

#[test]
fn cache_residence_reason_attributes_are_fixed_and_low_cardinality() {
    assert_eq!(
        &*CACHE_RESIDENCE_REASON_PRESSURE,
        &[KeyValue::new("reason", "pressure")]
    );
    assert_eq!(
        &*CACHE_RESIDENCE_REASON_CLEANUP,
        &[KeyValue::new("reason", "cleanup")]
    );
}

#[cfg(feature = "mooncake")]
#[test]
fn remote_fetch_plan_segment_boundaries_cover_failures_and_fragmented_plans() {
    assert_eq!(
        remote_fetch_plan_segment_boundaries(),
        vec![
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 16.0, 32.0, 64.0, 128.0,
        ]
    );
}
