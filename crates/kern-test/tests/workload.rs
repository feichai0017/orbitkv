//! The workload is a function of `(seed, manifest, options)` and stays
//! inside the manifest's bounds, over many seeds and every way of fixing
//! the prefill length.

mod common;

use common::{options, Fixture, CAPACITY, VOCAB};
use kern_manifest::types::Provision;
use kern_manifest::Protocol;
use kern_test::workload::{sample, Workload};
use kern_test::Options;

fn draw(o: &Options) -> anyhow::Result<Workload> {
    let m = Fixture::default().manifest();
    let pr = Protocol::check(&m).unwrap();
    sample(o, &m, &pr, Provision { tokens: CAPACITY, seq_slots: 0 }, 4)
}

fn bounded(w: &Workload, o: &Options) {
    let steps = o.decode_steps.max(1) as usize;
    assert!(w.prefill.len() + w.decode.len() <= CAPACITY as usize, "{w:?}");
    assert!((steps.div_ceil(2)..=steps).contains(&w.decode.len()), "{w:?} for {steps} steps");
    assert!(w.prefill.iter().chain(&w.decode).all(|&t| (0..VOCAB as i64).contains(&t)), "{w:?}");
    assert!((1..=8).contains(&w.chunk), "{w:?}");
    assert_eq!(w.vocab, VOCAB as u64);
}

#[test]
fn the_same_seed_draws_the_same_workload_and_different_seeds_differ() {
    let mut o = options();
    o.prefill = 0;
    o.chunk = 0;
    let mut distinct = std::collections::BTreeSet::new();
    for seed in 0..64 {
        o.seed = seed;
        let w = draw(&o).unwrap();
        assert_eq!(draw(&o).unwrap(), w);
        bounded(&w, &o);
        assert!(["uniform", "boundary"].contains(&w.how), "{w:?}");
        distinct.insert(format!("{w:?}"));
    }
    assert!(distinct.len() > 32, "{} distinct workloads over 64 seeds", distinct.len());
}

#[test]
fn a_given_prefill_and_chunk_are_taken_as_given_within_capacity() {
    for (prefill, steps, chunk) in [(5u64, 4u64, 3u64), (1, 1, 1), (8, 0, 8), (12, 4, 2), (40, 2, 5)] {
        let mut o = options();
        (o.prefill, o.decode_steps, o.chunk) = (prefill, steps, chunk);
        for seed in 0..16 {
            o.seed = seed;
            let w = draw(&o).unwrap();
            bounded(&w, &o);
            let room = CAPACITY - w.decode.len() as u64;
            assert_eq!((w.prefill.len() as u64, w.chunk as u64, w.how), (prefill.min(room), chunk, "given"), "{w:?}");
        }
    }
}

#[test]
fn a_prompt_is_the_prefill_verbatim_or_refused_when_it_does_not_fit() {
    let mut o = options();
    o.prompt = Some(vec![3, 1, 4, 1, 5, 9, 2, 6]);
    for seed in 0..16 {
        o.seed = seed;
        let w = draw(&o).unwrap();
        bounded(&w, &o);
        assert_eq!((w.prefill.as_slice(), w.how), ([3, 1, 4, 1, 5, 9, 2, 6].as_slice(), "prompt"));
    }
    o.prompt = Some((0..16).collect());
    let err = format!("{:#}", draw(&o).unwrap_err());
    assert!(err.contains("prompt is 16 tokens") && err.contains("capacity 16"), "{err}");
}
