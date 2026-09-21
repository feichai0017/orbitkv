use super::*;
use std::collections::HashSet;

#[test]
fn hll_empty_cardinality_is_zero() {
    let hll = HyperLogLog::new(14);
    assert_eq!(hll.cardinality(), 0.0);
}

#[test]
fn hll_single_insert() {
    let mut hll = HyperLogLog::new(14);
    hll.insert(&sha256_like(42));
    assert!(hll.cardinality() >= 0.5); // Should be ~1
}

#[test]
fn hll_accuracy_1000_distinct() {
    let mut hll = HyperLogLog::new(14);
    for i in 0u32..1000 {
        hll.insert(&sha256_like(i));
    }
    let est = hll.cardinality();
    assert!((900.0..1100.0).contains(&est), "expected ~1000, got {est}");
}

#[test]
fn hll_accuracy_10000_distinct() {
    let mut hll = HyperLogLog::new(14);
    for i in 0u32..10_000 {
        hll.insert(&sha256_like(i));
    }
    let est = hll.cardinality();
    assert!(
        (9000.0..11000.0).contains(&est),
        "expected ~10000, got {est}"
    );
}

#[test]
fn hll_duplicates_dont_increase_cardinality() {
    let mut hll = HyperLogLog::new(14);
    let hash = sha256_like(42);
    for _ in 0..1000 {
        hll.insert(&hash);
    }
    assert!(
        hll.cardinality() < 5.0,
        "cardinality should be ~1 for repeated inserts, got {}",
        hll.cardinality()
    );
}

#[test]
fn hll_merge() {
    let mut a = HyperLogLog::new(10);
    let mut b = HyperLogLog::new(10);

    for i in 0u32..500 {
        a.insert(&sha256_like(i));
    }
    for i in 500u32..1000 {
        b.insert(&sha256_like(i));
    }

    let card_a = a.cardinality();
    let card_b = b.cardinality();
    a.merge(&b);
    let card_merged = a.cardinality();

    assert!(card_merged > card_a);
    assert!(card_merged > card_b);
    assert!(
        (800.0..1200.0).contains(&card_merged),
        "expected ~1000, got {card_merged}"
    );
}

#[test]
fn hll_clear() {
    let mut hll = HyperLogLog::new(14);
    for i in 0u32..100 {
        hll.insert(&sha256_like(i));
    }
    assert!(hll.cardinality() > 50.0);
    hll.clear();
    assert_eq!(hll.cardinality(), 0.0);
}

#[test]
#[should_panic(expected = "HLL bucket_bits must be in")]
fn hll_bucket_bits_too_low() {
    HyperLogLog::new(MIN_BUCKET_BITS - 1);
}

#[test]
#[should_panic(expected = "HLL bucket_bits must be in")]
fn hll_bucket_bits_too_high() {
    HyperLogLog::new(MAX_BUCKET_BITS + 1);
}

// ---- Bit-level helper tests ----

fn make_hash(first3: [u8; 3]) -> [u8; 32] {
    let mut h = [0u8; 32];
    h[0] = first3[0];
    h[1] = first3[1];
    h[2] = first3[2];
    h
}

#[test]
fn bucket_index_basic() {
    assert_eq!(bucket_index(&make_hash([0xAB, 0, 0]), 4), 0b1010);
    assert_eq!(bucket_index(&make_hash([0xAB, 0xCD, 0]), 8), 0xAB);
    assert_eq!(bucket_index(&make_hash([0xAB, 0xCD, 0]), 12), 0xABC);
    assert_eq!(
        bucket_index(&make_hash([0xAB, 0xCD, 0xEF]), 14),
        0b10_1010_1111_0011
    );
}

fn lz(hash: &[u8], bits: u8) -> u8 {
    count_leading_zeros(hash, bits, (1u32 << (32 - bits)) - 1)
}

#[test]
fn count_leading_zeros_all_ones() {
    assert_eq!(lz(&[0xFF; 32], 4), 0);
}

#[test]
fn count_leading_zeros_all_zero() {
    // 32-14=18 from head, then 28 zero bytes → 18 + 224 = 242
    assert_eq!(lz(&[0u8; 32], 14), 242);
}

#[test]
fn count_leading_zeros_hit_in_head() {
    let mut h = [0u8; 32];
    h[1] = 0x01; // head=0x00010000, masked lz=15 - 14 = 1
    assert_eq!(lz(&h, 14), 1);
}

#[test]
fn count_leading_zeros_hit_in_tail() {
    let mut h = [0u8; 32];
    h[5] = 0x80; // 18 from head + (5-4)*8 + 0 = 26
    assert_eq!(lz(&h, 14), 26);
}

#[test]
fn count_leading_zeros_byte_aligned() {
    let mut h = [0u8; 32];
    h[1] = 0x01; // head masked lz=15 - 8 = 7
    assert_eq!(lz(&h, 8), 7);
}

// ---- HllTracker tests ----

#[test]
fn tracker_empty_metric() {
    let mut tracker = HllTracker::new(Duration::from_secs(3600), Duration::from_secs(86400), 14);
    let m = tracker.metric();
    assert_eq!(m.cardinality, 0.0);
    assert_eq!(m.total_requests, 0);
    assert_eq!(m.estimated_hit_rate, 0.0);
    assert_eq!(m.window_slot_count, 0);
}

#[test]
fn tracker_records_and_reports() {
    let mut tracker = HllTracker::new(Duration::from_secs(3600), Duration::from_secs(86400), 14);

    // Insert 100 distinct hashes, each 10 times
    for i in 0u32..100 {
        let hash = sha256_like(i);
        for _ in 0..10 {
            tracker.record(&hash);
        }
    }

    let m = tracker.metric();
    assert_eq!(m.total_requests, 1000);
    assert!(
        m.estimated_hit_rate > 0.80,
        "expected high hit rate, got {}",
        m.estimated_hit_rate
    );
    assert!(
        (80.0..120.0).contains(&m.cardinality),
        "expected ~100 cardinality, got {}",
        m.cardinality
    );
    assert_eq!(m.window_slot_count, 1);
}

#[test]
fn tracker_all_unique_low_hit_rate() {
    let mut tracker = HllTracker::new(Duration::from_secs(3600), Duration::from_secs(86400), 14);

    for i in 0u32..1000 {
        tracker.record(&sha256_like(i));
    }

    let m = tracker.metric();
    assert_eq!(m.total_requests, 1000);
    assert!(
        m.estimated_hit_rate < 0.1,
        "expected low hit rate for all unique, got {}",
        m.estimated_hit_rate
    );
}

#[test]
fn tracker_bucket_rotation() {
    let mut tracker = HllTracker::new(Duration::from_millis(1), Duration::from_secs(86400), 10);

    tracker.record(&sha256_like(0));
    std::thread::sleep(Duration::from_millis(5));
    tracker.record(&sha256_like(1));

    let m = tracker.metric();
    assert_eq!(m.total_requests, 2);
    assert!(
        m.window_slot_count >= 2,
        "expected >= 2 slots, got {}",
        m.window_slot_count
    );
}

#[test]
fn tracker_bucket_expiry() {
    let mut tracker = HllTracker::new(Duration::from_millis(1), Duration::from_millis(10), 10);

    for i in 0u32..10 {
        tracker.record(&sha256_like(i));
    }

    std::thread::sleep(Duration::from_millis(20));
    tracker.record(&sha256_like(100));

    let m = tracker.metric();
    assert_eq!(m.total_requests, 1);
    assert_eq!(m.window_slot_count, 1);
}

#[test]
fn tracker_hit_rate_50_percent() {
    let mut tracker = HllTracker::new(Duration::from_secs(3600), Duration::from_secs(86400), 14);

    // 10000 distinct hashes, each inserted twice → total 20000, cardinality ~10000
    // Expected hit rate ≈ (20000 - 10000) / 20000 = 0.50
    for i in 0u32..10_000 {
        let hash = sha256_like(i);
        tracker.record(&hash);
        tracker.record(&hash);
    }

    let m = tracker.metric();
    println!(
        "tracker_hit_rate_50_percent: cardinality={:.2}, total={}, hit_rate={:.4}",
        m.cardinality, m.total_requests, m.estimated_hit_rate
    );
    assert_eq!(m.total_requests, 20_000);
    assert!(
        (0.49..0.51).contains(&m.estimated_hit_rate),
        "expected ~0.50 hit rate, got {:.4}",
        m.estimated_hit_rate
    );
}

#[test]
fn tracker_hit_rate_66_percent() {
    let mut tracker = HllTracker::new(Duration::from_secs(3600), Duration::from_secs(86400), 14);

    // 10000 distinct hashes, each inserted 3 times → total 30000, cardinality ~10000
    // Expected hit rate ≈ (30000 - 10000) / 30000 = 0.6667
    for i in 0u32..10_000 {
        let hash = sha256_like(i);
        tracker.record(&hash);
        tracker.record(&hash);
        tracker.record(&hash);
    }

    let m = tracker.metric();
    assert_eq!(m.total_requests, 30_000);
    assert!(
        (0.656..0.676).contains(&m.estimated_hit_rate),
        "expected ~0.6667 hit rate, got {:.4}",
        m.estimated_hit_rate
    );
}

#[test]
fn hll_distinct_count_scaling() {
    for &n in &[100u32, 1_000, 5_000] {
        let mut hll = HyperLogLog::new(14);
        let mut seen = HashSet::new();
        for i in 0..n {
            let hash = sha256_like(i);
            hll.insert(&hash);
            seen.insert(hash);
        }
        let est = hll.cardinality();
        let actual = seen.len() as f64;
        let error = (est - actual).abs() / actual;
        assert!(
            error < 0.10,
            "n={n}: estimated={est:.0}, actual={actual:.0}, error={error:.4}"
        );
    }
}

// ---- MultiWindowHllTracker tests ----

#[test]
fn multi_window_records_into_each_window() {
    let mut tracker = MultiWindowHllTracker::new(
        vec![
            ("15m".into(), Duration::from_secs(15 * 60)),
            ("1h".into(), Duration::from_secs(3600)),
            ("1d".into(), Duration::from_secs(86400)),
        ],
        14,
    );

    for i in 0u32..1000 {
        let hash = sha256_like(i);
        tracker.record_hashes(&[hash.to_vec()]);
        tracker.record_hashes(&[hash.to_vec()]);
    }

    let metrics = tracker.metrics();
    assert_eq!(metrics.len(), 3);
    assert_eq!(metrics[0].0, "15m");
    assert_eq!(metrics[1].0, "1h");
    assert_eq!(metrics[2].0, "1d");
    for (label, m) in metrics {
        assert_eq!(m.total_requests, 2000, "{label}: total");
        assert!(
            (900.0..1100.0).contains(&m.cardinality),
            "{label}: cardinality {} not ~1000",
            m.cardinality
        );
    }
}

#[test]
fn namespaced_misses_keep_all_observations_in_denominator() {
    let mut tracker =
        MultiWindowHllTracker::new(vec![("15m".into(), Duration::from_secs(15 * 60))], 14);
    let miss = vec![sha256_like(1).to_vec(), sha256_like(2).to_vec()];
    tracker.record_namespaced_misses("model", 5, &miss);

    let metric = tracker.metrics().remove(0).1;
    assert_eq!(metric.total_requests, 5);
    assert!(metric.cardinality > 1.0 && metric.cardinality < 3.5);
    assert!(metric.estimated_hit_rate > 0.3);
}

#[test]
fn namespaced_misses_allow_all_hit_observation_without_hll_insert() {
    let mut tracker =
        MultiWindowHllTracker::new(vec![("15m".into(), Duration::from_secs(15 * 60))], 14);
    tracker.record_namespaced_misses("model", 4, &[]);

    let metric = tracker.metrics().remove(0).1;
    assert_eq!(metric.total_requests, 4);
    assert_eq!(metric.cardinality, 0.0);
    assert_eq!(metric.estimated_hit_rate, 1.0);
}

#[test]
fn namespaced_misses_separate_equal_raw_hashes() {
    let mut tracker =
        MultiWindowHllTracker::new(vec![("15m".into(), Duration::from_secs(15 * 60))], 16);
    let raw_hash = vec![42; 32];
    tracker.record_namespaced_misses("model-a", 1, std::slice::from_ref(&raw_hash));
    tracker.record_namespaced_misses("model-b", 1, std::slice::from_ref(&raw_hash));

    let metric = tracker.metrics().remove(0).1;
    assert_eq!(metric.total_requests, 2);
    assert!((1.5..2.5).contains(&metric.cardinality));
}

#[test]
fn derive_slot_clamps() {
    assert_eq!(
        derive_slot_duration(Duration::from_secs(15 * 60)),
        Duration::from_secs(60)
    );
    assert_eq!(
        derive_slot_duration(Duration::from_secs(3600)),
        Duration::from_secs(150)
    );
    assert_eq!(
        derive_slot_duration(Duration::from_secs(86400)),
        Duration::from_secs(3600)
    );
    assert_eq!(
        derive_slot_duration(Duration::from_secs(7 * 86400)),
        Duration::from_secs(3600)
    );
}

#[test]
#[should_panic(expected = "duplicate window duration")]
fn multi_window_rejects_duplicate_durations() {
    MultiWindowHllTracker::new(
        vec![
            ("1h".into(), Duration::from_secs(3600)),
            ("60m".into(), Duration::from_secs(3600)),
        ],
        14,
    );
}

#[test]
#[should_panic(expected = "at least one window")]
fn multi_window_rejects_empty() {
    MultiWindowHllTracker::new(vec![], 14);
}

#[test]
#[should_panic(expected = "at least 1 minute")]
fn multi_window_rejects_sub_minute_window() {
    MultiWindowHllTracker::new(vec![("30s".into(), Duration::from_secs(30))], 14);
}

/// Generate a pseudo-SHA256 hash from an integer for testing.
fn sha256_like(n: u32) -> [u8; 32] {
    let mut hash = [0u8; 32];
    let m0 = splitmix64(n as u64);
    hash[..8].copy_from_slice(&m0.to_le_bytes());
    let m1 = splitmix64(m0);
    hash[8..16].copy_from_slice(&m1.to_le_bytes());
    let m2 = splitmix64(m1);
    hash[16..24].copy_from_slice(&m2.to_le_bytes());
    let m3 = splitmix64(m2);
    hash[24..32].copy_from_slice(&m3.to_le_bytes());
    hash
}
