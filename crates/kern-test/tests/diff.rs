//! The static diff as a pure function: `align` over every pair of short
//! call lists from a small alphabet, and the op/program facts `diff` states
//! about the fixture family.

mod common;

use std::collections::BTreeMap;

use common::Fixture;
use kern_manifest::types::{Arg, Call};
use kern_test::diff::{align, diff, op_changes, Span};

/// The alphabet: two unchanged ops and one changed op, each over two
/// argument buffers. Any call list is a word over these six letters.
const OPS: [&str; 3] = ["p", "q", "x"];
const BUFS: [&str; 2] = ["u", "v"];

fn call(letter: usize) -> Call {
    Call {
        label: None,
        op: OPS[letter / BUFS.len()].to_string(),
        args: vec![Arg::Buf { buf: BUFS[letter % BUFS.len()].to_string(), offset: 0 }],
    }
}

/// Every word of length at most `n` over the alphabet.
fn words(n: usize) -> Vec<Vec<Call>> {
    let k = OPS.len() * BUFS.len();
    (0..=n)
        .flat_map(|len| {
            (0..k.pow(len as u32)).map(move |mut i| {
                (0..len)
                    .map(|_| {
                        let c = call(i % k);
                        i /= k;
                        c
                    })
                    .collect()
            })
        })
        .collect()
}

fn changed() -> BTreeMap<String, &'static str> {
    BTreeMap::from([("x".to_string(), "impl")])
}

fn same(a: &Call, b: &Call) -> bool {
    a.op == b.op && serde_json::json!(a.args) == serde_json::json!(b.args)
}

/// The gaps between spans, as index pairs `(i, j)` matched one to one.
fn gaps(spans: &[Span], n: usize, m: usize) -> Vec<(usize, usize)> {
    let (mut i, mut j) = (0, 0);
    let mut out = Vec::new();
    for s in spans {
        assert!(!s.a.is_empty() || !s.b.is_empty(), "an empty span in {spans:?}");
        assert_eq!(s.a.start - i, s.b.start - j, "unequal gap before {s:?} in {spans:?}");
        out.extend((i..s.a.start).zip(j..s.b.start));
        (i, j) = (s.a.end, s.b.end);
    }
    assert_eq!(n - i, m - j, "unequal tail after {spans:?}");
    out.extend((i..n).zip(j..m));
    out
}

#[test]
fn spans_are_ordered_and_the_gaps_between_them_match_call_for_call() {
    let ws = words(3);
    let ch = changed();
    for a in &ws {
        for b in &ws {
            let spans = align(a, b, &ch);
            for w in spans.windows(2) {
                assert!(w[0].a.end <= w[1].a.start && w[0].b.end <= w[1].b.start, "{spans:?}");
            }
            for (i, j) in gaps(&spans, a.len(), b.len()) {
                assert!(same(&a[i], &b[j]), "gap call {i}/{j} differs for {spans:?}");
                assert!(!ch.contains_key(&a[i].op), "a changed op is shared at {i}/{j} in {spans:?}");
            }
        }
    }
}

#[test]
fn identical_lists_span_exactly_their_changed_calls() {
    let ch = changed();
    for a in words(4) {
        let spans = align(&a, &a, &ch);
        let in_span = |i: usize| spans.iter().any(|s| s.a.contains(&i));
        for (i, c) in a.iter().enumerate() {
            assert_eq!(in_span(i), ch.contains_key(&c.op), "call {i} of {spans:?}");
        }
        // a changed call sits in a span of its own row on both sides
        assert!(spans.iter().all(|s| s.a == s.b), "{spans:?}");
        assert_eq!(align(&a, &a, &BTreeMap::new()), []);
    }
}

#[test]
fn a_span_is_as_small_as_the_lcs_allows() {
    let ch = changed();
    let ws = words(3);
    for a in &ws {
        for b in &ws {
            let spans = align(a, b, &ch);
            let shared = gaps(&spans, a.len(), b.len()).len();
            // shared calls are an LCS of the unchanged calls: at least as
            // many as the greedy count of equal unchanged letters
            let unchanged = |w: &[Call]| w.iter().filter(|c| !ch.contains_key(&c.op)).count();
            assert!(shared <= unchanged(a).min(unchanged(b)), "{spans:?}");
            let sym = |w: &[Call]| w.iter().map(|c| format!("{}{:?}", c.op, c.args)).collect::<Vec<_>>();
            assert!(shared >= lcs(&sym(a), &sym(b), &ch), "{spans:?}");
        }
    }
}

/// A reference LCS over unchanged letters, written the slow way.
fn lcs(a: &[String], b: &[String], ch: &BTreeMap<String, &str>) -> usize {
    let mut t = vec![vec![0usize; b.len() + 1]; a.len() + 1];
    for i in 0..a.len() {
        for j in 0..b.len() {
            let x = ch.keys().any(|k| a[i].starts_with(k.as_str()));
            t[i + 1][j + 1] = if a[i] == b[j] && !x { t[i][j] + 1 } else { t[i][j + 1].max(t[i + 1][j]) };
        }
    }
    t[a.len()][b.len()]
}

#[test]
fn a_label_names_both_ranges() {
    assert_eq!(Span { a: 1..2, b: 1..3 }.label(), "A[1..2) B[1..3)");
    assert_eq!(Span { a: 0..0, b: 4..5 }.label(), "A[0..0) B[4..5)");
}

#[test]
fn op_changes_tell_interface_from_impl_from_presence() {
    let base = Fixture::default().manifest();
    let imp = Fixture::default().scale("scale_same").manifest();
    let added = Fixture::default().alt(1, "scale_wrong").manifest();
    assert_eq!(op_changes(&base, &imp), BTreeMap::from([("scale".to_string(), "impl")]));
    assert_eq!(op_changes(&base, &added), BTreeMap::from([("scale_alt".to_string(), "added")]));
    assert_eq!(op_changes(&added, &base), BTreeMap::from([("scale_alt".to_string(), "removed")]));
    assert_eq!(op_changes(&base, &base), BTreeMap::new());
}

#[test]
fn diff_spans_every_program_that_calls_a_changed_op_and_no_other() {
    let base = Fixture::default().manifest();
    let d = diff(&base, &Fixture::default().scale("scale_same").manifest());
    assert_eq!(d.spans.keys().cloned().collect::<Vec<_>>(), ["decode", "prefill"]);
    assert_eq!(
        d.programs.iter().map(|p| (p.program.as_str(), p.spans, p.calls, p.shared)).collect::<Vec<_>>(),
        [("decode", 2, 6, 4), ("prefill", 2, 6, 4)]
    );
    // the head is the last call of both programs: never spanned by a scale change
    assert!(d.spans.values().flatten().all(|s| s.a.end < 6), "{:?}", d.spans);
    let probe = diff(&Fixture::default().probe().manifest(), &Fixture::default().probe().mix("mix_leak").manifest());
    assert_eq!(probe.spans.keys().cloned().collect::<Vec<_>>(), ["decode", "prefill", "probe"]);
}

#[test]
fn a_peer_buffer_stands_for_the_exported_buffer_it_holds_addresses_of() {
    use kern_test::diff::{access, frontier_inputs};
    let m = Fixture::default().ranks(2).peer().manifest();
    let acc = access(&m, "gather", 1..2);
    // the kernel given every rank's `hidden` reads and writes `hidden`;
    // the address array itself is nothing a span consumes or produces
    assert_eq!(acc.reads.iter().cloned().collect::<Vec<_>>(), ["hidden"]);
    assert_eq!(acc.writes.iter().cloned().collect::<Vec<_>>(), ["act", "hidden"]);
    assert_eq!(frontier_inputs(&m, "gather", 1..2).into_iter().collect::<Vec<_>>(), ["hidden"]);
}

#[test]
fn what_a_once_program_writes_is_a_load_time_constant() {
    use kern_manifest::Protocol;
    use kern_test::diff::constants;
    let m = Fixture::default().once("fill_table").manifest();
    let once = Protocol::check(&m).unwrap().once;
    assert_eq!(once, ["prep"]);
    assert_eq!(constants(&m, &once).into_iter().collect::<Vec<_>>(), ["table"]);
    assert!(constants(&m, &[]).is_empty());
}
