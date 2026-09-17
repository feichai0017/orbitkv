//! The runtime's comparison kernels against kern-test's host definitions:
//! every dtype over random bytes (NaNs, infinities and signed zeros come
//! with them), every operand kind, block-granular change detection on a
//! length that is not a multiple of the block and on an operand off the
//! 16-byte alignment, and logits rows from one column to a real
//! vocabulary with a tie across the whole row, a NaN on either side and a
//! signed zero. Counts and maxima must match exactly; the floating-point
//! sums to rounding. The device counts reach kern-test's types through
//! the same `cmp_of` / `logit_of` the real side uses.

use kern_manifest::types::DType;
use kern_manifest::values::from_f64;
use kern_manifest::Verified;
use kern_run::test::{cmp_of, logit_of};
use kern_runtime::{At, Capacity, HostWeights, Runtime};
use kern_test::compare::{changed_blocks, compare, logit_stats, TOP};
use kern_test::workload::Rng;

const BYTES: usize = 4 * 129_280 * 5;

/// Two byte arrays, declared as the operands of a program that is never
/// run (a manifest has to use every buffer).
fn runtime() -> Runtime {
    let cols = BYTES / 2 / 5;
    let manifest = serde_json::json!({
        "schema_version": 5, "model": "device-compare-test", "vars": {}, "states": {},
        "buffers": {
            "a": {"kind": "input", "dtype": "bf16", "shape": [5, cols]},
            "b": {"kind": "input", "dtype": "bf16", "shape": [5, cols]},
            "out": {"kind": "output", "dtype": "bf16", "shape": [5, 5]}
        },
        "modules": {},
        "ops": {"gemm": {
            "params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32", "i32"],
            "impl": {"launches": [{"entry": "extern:cublaslt_bf16_tn"}]}
        }},
        "programs": {"never": {"calls": [{"op": "gemm", "args": [
            {"buf": "a"}, {"buf": "b"}, {"buf": "out"}, {"i32": 5}, {"i32": 5}, {"i32": cols}
        ]}]}}
    });
    let verified = Verified::from_json(&manifest.to_string()).unwrap();
    let dir = std::env::temp_dir().join(format!("kern-device-compare-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let scope = HostWeights::new();
    let gpu = std::env::var("KERN_TEST_GPU").ok().and_then(|g| g.parse().ok()).unwrap_or(0);
    Runtime::load_with_host_weights(&verified, &dir, gpu, Some(Capacity { tokens: Some(1), seqs: 1 }), None, &scope)
        .unwrap()
}

/// `a` random; `b` mostly `a`, with bytes flipped, elements replaced and
/// zeros of the other sign sprinkled in.
fn pair(rng: &mut Rng, dt: DType, n: usize) -> (Vec<u8>, Vec<u8>) {
    let w = dt.bytes() as usize;
    let a: Vec<u8> = (0..n * w).map(|_| rng.below(256) as u8).collect();
    let mut b = a.clone();
    for i in 0..n {
        match rng.below(10) {
            0 => b[i * w] ^= 1 << rng.below(8),
            1 => (0..w).for_each(|k| b[i * w + k] = rng.below(256) as u8),
            2 => {
                let z = from_f64(dt, &[if rng.below(2) == 0 { 0.0 } else { -0.0 }]);
                b[i * w..(i + 1) * w].copy_from_slice(&z);
            }
            _ => {}
        }
    }
    (a, b)
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn device_compare_agrees_with_the_host_definition_on_every_dtype_and_operand() {
    let mut rt = runtime();
    let mut rng = Rng(0xc0ffee);
    for dt in [
        DType::Bf16,
        DType::F16,
        DType::F32,
        DType::Fp8E4m3,
        DType::Fp8E8m0,
        DType::I8,
        DType::U8,
        DType::I32,
        DType::U32,
        DType::I64,
        DType::U64,
    ] {
        let n = 100_003;
        let (a, b) = pair(&mut rng, dt, n);
        rt.write_buffer("a", &a).unwrap();
        rt.write_buffer("b", &b).unwrap();
        let want = compare(dt, &a, &b);
        let len = a.len();
        let got = cmp_of(rt.compare(dt, At::Buffer("a", 0..len), At::Buffer("b", 0..len)).unwrap());
        assert_eq!(got, want, "{dt:?} buffer vs buffer");
        let s = rt.save_buffer("a", 0..len).unwrap();
        let got = cmp_of(rt.compare(dt, At::Scratch(&s, 0..len), At::Buffer("b", 0..len)).unwrap());
        assert_eq!(got, want, "{dt:?} scratch vs buffer");
        let same = cmp_of(rt.compare(dt, At::Scratch(&s, 0..len), At::Buffer("a", 0..len)).unwrap());
        assert_eq!(same, compare(dt, &a, &a), "{dt:?} against itself");
    }
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn device_changed_blocks_agree_with_the_host_definition() {
    let mut rt = runtime();
    let mut rng = Rng(7);
    for n in [1usize, 63, 64, 65, 4096, 1_000_001] {
        let pre: Vec<u8> = (0..n).map(|_| rng.below(256) as u8).collect();
        let mut post = pre.clone();
        for _ in 0..(n / 200).max(1) {
            let at = rng.below(n as u64) as usize;
            post[at] ^= 0xff;
        }
        if n > 64 {
            post[n - 1] ^= 1; // the clipped tail block
        }
        rt.write_buffer("a", &pre).unwrap();
        rt.write_buffer("b", &post).unwrap();
        let want = changed_blocks(&pre, &post);
        let got = rt.changed(At::Buffer("a", 0..n), At::Buffer("b", 0..n)).unwrap();
        assert_eq!(got, want, "{n} bytes");
        let none = rt.changed(At::Buffer("a", 0..n), At::Buffer("a", 0..n)).unwrap();
        assert!(none.is_empty(), "{n} bytes against itself: {none:?}");
        // the same bytes three past a 16-byte boundary: the byte path
        let mut off = vec![0u8; 3];
        off.extend_from_slice(&pre);
        rt.write_buffer("a", &off).unwrap();
        let s = rt.save_buffer("a", 0..n + 3).unwrap();
        let got = rt.changed(At::Scratch(&s, 3..n + 3), At::Buffer("b", 0..n)).unwrap();
        assert_eq!(got, want, "{n} bytes, misaligned");
    }
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn device_logit_rows_agree_with_the_host_definition() {
    let mut rt = runtime();
    let mut rng = Rng(42);
    let rows = 5;
    for dt in [DType::F32, DType::Bf16] {
        for cols in [1usize, 7, 1000, 4096, 129_280] {
            let gauss = |rng: &mut Rng| (0..12).map(|_| rng.below(1000) as f64 / 1000.0).sum::<f64>() - 6.0;
            let mut va: Vec<f64> = (0..rows * cols).map(|_| gauss(&mut rng) * 4.0).collect();
            // B: A plus a small drift; row 1 a confident swap, row 2 a NaN
            // in B, row 3 one value everywhere on both sides (the argmax
            // and the top set are ties broken by index), row 4 a NaN in A;
            // row 0 opens with a signed zero on each side.
            let mut vb: Vec<f64> = va.iter().map(|x| x + gauss(&mut rng) * 0.05).collect();
            va[0] = 0.0;
            vb[0] = -0.0;
            if cols > 1 {
                vb.swap(cols, cols + 1);
            }
            vb[2 * cols + 3.min(cols - 1)] = f64::NAN;
            for j in 0..cols {
                va[3 * cols + j] = 1.5;
                vb[3 * cols + j] = 1.5;
            }
            va[4 * cols + 3.min(cols - 1)] = f64::NAN;
            let (a, b) = (from_f64(dt, &va), from_f64(dt, &vb));
            rt.write_buffer("a", &a).unwrap();
            rt.write_buffer("b", &b).unwrap();
            let got = rt.logits(dt, cols, TOP, At::Buffer("a", 0..a.len()), At::Buffer("b", 0..b.len())).unwrap();
            assert_eq!(got.len(), rows, "{dt:?} {cols}");
            let w = dt.bytes() as usize;
            for (r, l) in got.into_iter().enumerate() {
                let (lo, hi) = (r * cols * w, (r + 1) * cols * w);
                let want = logit_stats(dt, &a[lo..hi], &b[lo..hi]);
                let g = logit_of(l);
                assert_eq!(
                    (g.cmp.clone(), g.argmax_a, g.argmax_b, g.rank_in_b, g.top),
                    (want.cmp.clone(), want.argmax_a, want.argmax_b, want.rank_in_b, want.top),
                    "{dt:?} {cols} row {r}"
                );
                // a NaN argmax has a NaN margin on both sides
                assert!(
                    g.margin_a == want.margin_a || (g.margin_a.is_nan() && want.margin_a.is_nan()),
                    "{dt:?} {cols} row {r}: margin {} vs {}",
                    g.margin_a,
                    want.margin_a
                );
                if want.kl.is_finite() {
                    assert!(
                        (g.kl - want.kl).abs() <= 1e-9 * want.kl.abs().max(1.0),
                        "{dt:?} {cols} row {r}: {} vs {}",
                        g.kl,
                        want.kl
                    );
                } else {
                    assert!(g.kl.is_infinite(), "{dt:?} {cols} row {r}: {}", g.kl);
                }
            }
        }
    }
}
