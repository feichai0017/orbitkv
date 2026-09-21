use super::*;

#[test]
fn partial_backing_decisions_keep_selected_tier_and_residual_contract() {
    let cases = [
        // Done branch: local RAM prefix plus residual miss when no backing
        // tier is selected for this decision.
        (TierAttribution::classify(7, 3, 0, None), (3, 0, 0, 4, 7)),
        // Mooncake found only part of the non-RAM prefix.
        (
            TierAttribution::classify(5, 1, 3, Some(AttributionSource::Remote)),
            (1, 3, 0, 1, 5),
        ),
        // SSD prefetch accepted only part of the non-RAM prefix, for
        // example after backpressure trimming.
        (
            TierAttribution::classify(6, 1, 2, Some(AttributionSource::Ssd)),
            (1, 0, 2, 3, 6),
        ),
    ];

    for (attribution, expected) in cases {
        assert_eq!(
            (
                attribution.ram,
                attribution.remote,
                attribution.ssd,
                attribution.miss,
                attribution.sum()
            ),
            expected
        );
    }
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "loading > 0 without a backing source")]
fn loading_without_source_panics_in_debug() {
    let _ = TierAttribution::classify(4, 1, 1, None);
}

#[test]
#[should_panic(expected = "hit + loading must not exceed total")]
fn overcounting_panics_in_debug() {
    let _ = TierAttribution::classify(3, 2, 2, Some(AttributionSource::Ssd));
}
