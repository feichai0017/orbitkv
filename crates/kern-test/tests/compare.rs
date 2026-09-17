//! The comparison primitives over enumerated inputs: every pair of
//! interesting values per dtype, changed blocks against a byte-level
//! reference, and logits rows with a known argmax, KL and top set.

use kern_manifest::types::DType;
use kern_manifest::values::{from_f64, to_f64};
use kern_test::compare::{changed_blocks, coalesce, compare, logit_stats, outside, Cmp, BLOCK, TOP};
use kern_test::workload::Rng;

const FLOATS: [DType; 4] = [DType::Bf16, DType::F16, DType::F32, DType::Fp8E4m3];

/// The values worth pairing: both zeros, neighbours across a binade, the
/// two non-finite kinds.
fn interesting(dt: DType) -> Vec<Vec<u8>> {
    let base = [0.0, -0.0, 1.0, -1.0, 2.0, 0.5, 1.5, 3.0, 1e-3, 100.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
    let w = dt.bytes() as usize;
    let mut v: Vec<Vec<u8>> = base.iter().map(|x| from_f64(dt, &[*x])).collect();
    // the next representable value after 1.0, whatever the dtype's mantissa
    let mut one = from_f64(dt, &[1.0]);
    one[0] = one[0].wrapping_add(1);
    v.push(one);
    v.retain(|b| b.len() == w);
    v.dedup();
    v
}

fn cat(elems: &[&[u8]]) -> Vec<u8> {
    elems.concat()
}

#[test]
fn a_comparison_is_symmetric_and_counts_each_kind_of_difference_once() {
    for dt in FLOATS {
        let vs = interesting(dt);
        for a in &vs {
            for b in &vs {
                let (ab, ba) = (compare(dt, a, b), compare(dt, b, a));
                let (fa, fb) = (to_f64(dt, a)[0], to_f64(dt, b)[0]);
                assert_eq!(
                    (ab.n_diff, ab.nan_only_one_side, ab.signed_zero, ab.max_ulp, ab.max_abs),
                    (ba.n_diff, ba.nan_only_one_side, ba.signed_zero, ba.max_ulp, ba.max_abs),
                    "{dt:?} {fa} vs {fb}"
                );
                assert_eq!(ab.n, 1);
                assert_eq!(ab.n_diff, (a != b) as usize, "{dt:?} {fa} vs {fb}");
                assert_eq!(
                    ab.nan_only_one_side,
                    (a != b && fa.is_nan() != fb.is_nan()) as usize,
                    "{dt:?} {fa} vs {fb}"
                );
                assert_eq!(ab.signed_zero, (a != b && fa == fb) as usize, "{dt:?} {fa} vs {fb}");
                // ulps and abs are measured only on the differing finite-or-inf pairs
                let measured = a != b && fa.is_nan() == fb.is_nan() && !fa.is_nan() && fa != fb;
                assert_eq!(ab.max_ulp.is_some(), measured, "{dt:?} {fa} vs {fb}: {ab:?}");
                assert_eq!(ab.max_abs > 0.0, measured, "{dt:?} {fa} vs {fb}: {ab:?}");
                assert_eq!((ab.identical(), ab.value_identical()), (a == b, a == b || fa == fb), "{dt:?} {fa} vs {fb}");
            }
        }
    }
}

#[test]
fn a_tensor_comparison_is_the_sum_of_its_elements() {
    for dt in FLOATS {
        let vs = interesting(dt);
        let refs: Vec<&[u8]> = vs.iter().map(Vec::as_slice).collect();
        let a = cat(&refs);
        let mut rot = refs.clone();
        rot.rotate_left(3);
        let b = cat(&rot);
        let whole = compare(dt, &a, &b);
        let parts: Vec<Cmp> = refs.iter().zip(&rot).map(|(x, y)| compare(dt, x, y)).collect();
        let sum = |f: fn(&Cmp) -> usize| parts.iter().map(f).sum::<usize>();
        assert_eq!(
            (whole.n, whole.n_diff, whole.nan_only_one_side, whole.signed_zero),
            (refs.len(), sum(|c| c.n_diff), sum(|c| c.nan_only_one_side), sum(|c| c.signed_zero)),
            "{dt:?}"
        );
        assert_eq!(whole.max_ulp, parts.iter().filter_map(|c| c.max_ulp).max(), "{dt:?}");
        assert_eq!(whole.max_abs, parts.iter().map(|c| c.max_abs).fold(0.0, f64::max), "{dt:?}");
    }
}

#[test]
fn severity_orders_identical_below_signed_zeros_below_real_differences() {
    let dt = DType::F32;
    let (z, nz, one, two) = (from_f64(dt, &[0.0]), from_f64(dt, &[-0.0]), from_f64(dt, &[1.0]), from_f64(dt, &[2.0]));
    let same = compare(dt, &one, &one);
    let zeros = compare(dt, &z, &nz);
    let close = compare(dt, &cat(&[&one, &one]), &cat(&[&one, &two]));
    let far = compare(dt, &cat(&[&one, &one]), &cat(&[&two, &two]));
    assert!(same.severity() < zeros.severity() || same.severity() == zeros.severity());
    assert!(zeros.severity() < close.severity() && close.severity() < far.severity());
    // an integer dtype has differing elements but no ulps
    let ints = compare(DType::I32, &7i32.to_le_bytes(), &9i32.to_le_bytes());
    assert_eq!((ints.n_diff, ints.max_ulp, ints.max_abs), (1, None, 2.0));
}

fn row(dt: DType, v: &[f64]) -> Vec<u8> {
    from_f64(dt, v)
}

#[test]
fn logit_stats_know_the_argmax_margin_kl_and_top_overlap() {
    let dt = DType::F32;
    let a = [0.5, 2.0, -1.0, 1.75];
    let ra = row(dt, &a);
    let same = logit_stats(dt, &ra, &ra);
    assert_eq!((same.argmax_a, same.argmax_b, same.flip(), same.rank_in_b), (1, 1, false, 1));
    assert_eq!((same.kl, same.margin_a, same.top), (0.0, 0.25, 4));
    // shifting every logit by a constant leaves argmax, KL and the top set alone
    let shifted = logit_stats(dt, &ra, &row(dt, &a.map(|x| x + 3.0)));
    assert_eq!((shifted.flip(), shifted.top, shifted.cmp.n_diff), (false, 4, 4));
    assert!(shifted.kl.abs() < 1e-9, "{}", shifted.kl);
    // a near tie broken the other way: a flip, A's token now second in B,
    // almost no mass moved
    let t = [0.5, 2.0, -1.0, 1.99];
    let tie = logit_stats(dt, &row(dt, &t), &row(dt, &[0.5, 1.99, -1.0, 2.0]));
    assert_eq!((tie.flip(), tie.argmax_b, tie.rank_in_b, tie.top), (true, 3, 2, 4));
    assert!(tie.kl > 0.0 && tie.kl < 1e-3, "{}", tie.kl);
    // a confident token displaced: the same flip, far more mass moved
    let wide = logit_stats(dt, &row(dt, &[0.0, 5.0, 0.0, 1.0]), &row(dt, &[0.0, 1.0, 0.0, 5.0]));
    assert_eq!((wide.flip(), wide.argmax_b, wide.rank_in_b), (true, 3, 2));
    assert!(wide.kl > 1.0, "{}", wide.kl);
    // the top set is A's TOP most likely tokens found among B's
    let n = TOP + 5;
    let va: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let mut vb = va.clone();
    vb.swap(n - 1, 0); // A's best becomes B's worst, A's worst B's best
    let moved = logit_stats(dt, &row(dt, &va), &row(dt, &vb));
    assert_eq!((moved.top, moved.rank_in_b, moved.argmax_b), (TOP - 1, n, 0));
    // a NaN on one side is an unbounded move
    let nan = logit_stats(dt, &ra, &row(dt, &[0.5, f64::NAN, -1.0, 1.75]));
    assert!(nan.kl.is_infinite() && nan.cmp.nan_only_one_side == 1);
}

#[test]
fn changed_blocks_name_every_block_that_differs_and_only_those() {
    let mut rng = Rng(3);
    for n in [0usize, 1, BLOCK - 1, BLOCK, BLOCK + 1, 10 * BLOCK + 7] {
        for _ in 0..20 {
            let pre: Vec<u8> = (0..n).map(|_| rng.below(256) as u8).collect();
            let mut post = pre.clone();
            for _ in 0..rng.below(4) {
                if n > 0 {
                    let at = rng.below(n as u64) as usize;
                    post[at] ^= 1 + rng.below(255) as u8;
                }
            }
            let ranges = changed_blocks(&pre, &post);
            let block = |lo: usize| lo..(lo + BLOCK).min(n);
            // every differing byte is covered; a range is whole blocks that all
            // differ, clipped at the end; no two ranges touch
            for i in (0..n).filter(|&i| pre[i] != post[i]) {
                assert!(ranges.iter().any(|r| r.contains(&i)), "{n}: byte {i} not covered by {ranges:?}");
            }
            for r in &ranges {
                assert!(r.start % BLOCK == 0 && (r.end % BLOCK == 0 || r.end == n) && r.start < r.end, "{n}: {r:?}");
                for lo in (r.start..r.end).step_by(BLOCK) {
                    assert_ne!(pre[block(lo)], post[block(lo)], "{n}: {r:?} has a clean block at {lo}");
                }
            }
            for w in ranges.windows(2) {
                assert!(w[0].end < w[1].start, "{n}: {:?} touches {:?}", w[0], w[1]);
            }
        }
    }
}

#[test]
fn merged_comparisons_add_counts_and_keep_the_worst_distance() {
    let (x, y, z) = (1.0f32.to_le_bytes(), 1.5f32.to_le_bytes(), 3.0f32.to_le_bytes());
    let a = compare(DType::F32, &[x, x].concat(), &[x, y].concat());
    let b = compare(DType::F32, &[x, x, x].concat(), &[z, x, x].concat());
    let m = a.clone().merge(b.clone());
    assert_eq!((m.n, m.n_diff, m.max_abs), (5, 2, 2.0));
    assert_eq!(m.max_ulp, a.max_ulp.max(b.max_ulp));
    assert_eq!(Cmp::default().merge(a.clone()), a);
}

#[test]
fn bytes_outside_a_write_set_are_those_no_range_covers() {
    let a = [64..128, 256..320];
    assert_eq!(outside(&[64..128, 256..320], &a), 0);
    assert_eq!(outside(&[0..64, 64..128, 300..400], &a), 64 + 80);
    assert_eq!(outside(&[], &a), 0);
    assert_eq!(outside(&[0..64, 128..192], &[]), 128);
}

#[test]
fn coalescing_closes_gaps_below_the_limit_only() {
    assert_eq!(coalesce(vec![0..64, 128..192, 4096..4160, 9000..9064], 100), [0..192, 4096..4160, 9000..9064]);
    assert_eq!(coalesce(vec![0..64, 128..192], 0), [0..64, 128..192]);
    assert_eq!(coalesce(vec![0..64, 64..128, 256..320], 0), [0..128, 256..320]);
}
