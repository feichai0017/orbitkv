//! Every verdict row through the fake side: one fixture per row, and the
//! facts the report states on the way there.

mod common;

use common::{options, test, verdict, Fixture};

fn run(b: Fixture) -> (kern_test::Report, Vec<String>) {
    test(&Fixture::default(), &b, &options()).expect("harness runs")
}

fn line<'a>(lines: &'a [String], key: &str) -> &'a str {
    lines
        .iter()
        .find(|l| l.starts_with(key))
        .map(String::as_str)
        .unwrap_or_else(|| panic!("no `{key}` line in {lines:#?}"))
}

#[test]
fn a_no_op_swap_is_bit_identical_at_every_span() {
    let (r, lines) = run(Fixture::default().scale("scale_same"));
    assert_eq!(verdict(&r), (0, "bit-identical at every span".into()));
    let local = r.summary.local.as_ref().unwrap();
    // two layers × (2 prefill chunks + 4 decode steps) spans compared, all identical
    assert_eq!((local.compared, local.bit_identical, local.findings.len()), (12, 12, 0));
    assert_eq!(r.summary.tap.as_ref().unwrap().spans, 4);
    assert!(line(&lines, "diff").contains("scale impl extern:scale → extern:scale_same"), "{lines:#?}");
}

#[test]
fn signed_zeros_are_value_identical() {
    let (r, _) = run(Fixture::default().scale("scale_negzero"));
    assert_eq!(verdict(&r), (0, "value-identical at every span (only signed zeros differ)".into()));
    let local = r.summary.local.as_ref().unwrap();
    assert!(local.value_identical > 0 && local.bit_identical + local.value_identical == local.compared);
}

#[test]
fn a_difference_the_head_never_reads_leaves_the_logits_bit_identical() {
    let (r, lines) = run(Fixture::default().scale("scale_dim3"));
    assert_eq!(verdict(&r), (0, "spans differ, but the end-to-end logits are bit-identical on all 9 rows".into()));
    assert!(line(&lines, "local").contains("output next_token bit-identical"), "{lines:#?}");
    // the state carries the difference, and the report says so without judging it
    assert!(line(&lines, "local").contains("state kv") && line(&lines, "local").contains("bytes differ"), "{lines:#?}");
}

#[test]
fn a_rounding_change_passes_on_logit_evidence() {
    let (r, _) = run(Fixture::default().scale("scale_round"));
    let (code, head) = verdict(&r);
    assert_eq!((code, head.as_str()), (0, "logit evidence"), "{}", r.summary.verdict.summary);
    let lg = r.summary.logits.as_ref().unwrap();
    assert!(lg.differ > 0 && lg.kl_max <= 0.01 && lg.flips == lg.within, "{lg:?}");
}

#[test]
fn a_reference_that_is_not_deterministic_judges_b_against_its_own_band() {
    // A is ±3% on every `scale` call, the sign flipping on each repeat of
    // an input; B is the exact op.
    // The KL limit is below A's own band, so logit evidence cannot decide.
    let mut o = options();
    o.logit_kl = 1e-4;
    let (r, lines) = test(&Fixture::default().scale("scale_noisy"), &Fixture::default(), &o).unwrap();
    assert_eq!(verdict(&r), (0, "differences at every span lie within A's own noise floor".into()), "{lines:#?}");
    assert!(line(&lines, "noise").contains("A is not deterministic"), "{lines:#?}");
    let n = r.summary.noise.as_ref().unwrap();
    assert!(n.compared == 4 && n.clean < n.compared, "{n:?}");
}

#[test]
fn a_wide_argmax_flip_fails_when_a_reproduces_itself_end_to_end() {
    // A is noisy at its spans (±3% on `scale`) yet lands on the same
    // distribution every time; B swaps the head's argmax. The flip is B's.
    let (r, lines) =
        test(&Fixture::default().scale("scale_noisy"), &Fixture::default().head("head_swap"), &options()).unwrap();
    let v = &r.summary.verdict;
    assert!(v.code == 1 && v.summary.starts_with("B changes the argmax end-to-end at prefill chunk 0"), "{lines:#?}");
    let n = r.summary.noise.as_ref().unwrap();
    let f = n.floor.as_ref().unwrap();
    assert!(n.clean < n.compared && f.kl_max <= 0.01 && f.flips == 0, "{n:?}");
}

#[test]
fn a_wide_argmax_flip_fails() {
    let (r, lines) = run(Fixture::default().head("head_swap"));
    let v = &r.summary.verdict;
    assert!(v.code == 1 && v.summary.starts_with("B changes the argmax end-to-end at prefill chunk 0"), "{lines:#?}");
    let lg = r.summary.logits.as_ref().unwrap();
    assert!(lg.flips > lg.within, "{lg:?}");
    assert!(lines.iter().any(|l| l.starts_with("logits    ✗ flip")), "{lines:#?}");
}

#[test]
fn a_value_outside_the_declared_domain_fails() {
    let (r, lines) = run(Fixture::default().head("head_bad"));
    assert_eq!(verdict(&r), (1, "B writes a value outside a declared domain end to end".into()));
    assert!(line(&lines, "local     ✗ domain").contains("B next_token[0] = 99 outside domain"), "{lines:#?}");
}

#[test]
fn a_changed_program_the_driver_cannot_stage_is_inconclusive() {
    let (r, lines) =
        test(&Fixture::default().probe(), &Fixture::default().probe().scale("scale_same"), &options()).unwrap();
    assert_eq!(verdict(&r), (2, "a changed program was not tapped — the workload driver can't stage it".into()));
    assert_eq!(r.summary.local.as_ref().unwrap().undriven, ["probe"]);
    assert!(line(&lines, "local     ✗ probe").contains("changed but not tapped"), "{lines:#?}");
}

#[test]
fn logits_moving_past_the_limit_without_a_flip_are_inconclusive() {
    let (r, _) = run(Fixture::default().scale("scale_drift"));
    let v = &r.summary.verdict;
    assert!(v.code == 2 && v.summary.starts_with("spans differ; end-to-end KL up to"), "{}", v.summary);
    let lg = r.summary.logits.as_ref().unwrap();
    assert!(lg.kl_max > 0.01 && lg.flips == lg.within, "{lg:?}");
}

#[test]
fn without_a_logits_buffer_there_is_no_oracle() {
    let (r, lines) = test(
        &Fixture::default().logits("scores"),
        &Fixture::default().logits("scores").scale("scale_round"),
        &options(),
    )
    .unwrap();
    assert_eq!(
        verdict(&r),
        (2, "spans differ beyond bit/value identity and no driven program writes logits — no oracle".into())
    );
    assert!(line(&lines, "logits").contains("none: no driven program writes a `logits*` buffer"), "{lines:#?}");
}

#[test]
fn a_change_at_one_layer_is_named_at_that_layer_only() {
    let (r, lines) = run(Fixture::default().alt(1, "scale_wrong"));
    assert!(line(&lines, "diff").contains("scale_alt added extern:scale_wrong"), "{lines:#?}");
    // one span per program: layer 1's scale call (embed, scale0, mix0 before it)
    assert_eq!(r.summary.diff.spans["prefill"].len(), 1);
    assert_eq!(
        (r.summary.diff.spans["prefill"][0].a.clone(), r.summary.diff.spans["prefill"][0].b.clone()),
        (3..4, 3..4)
    );
    let local = r.summary.local.as_ref().unwrap();
    assert!(local.findings.iter().all(|f| f.span.starts_with("chunk") || f.span.starts_with("step")));
    assert!(local.findings.iter().all(|f| f.span.contains("A[3..4) B[3..4)")), "{:#?}", local.findings);
    assert_eq!(local.compared, local.findings.len() + local.omitted);
}

#[test]
fn nan_on_one_side_is_counted_and_never_passes_as_logit_evidence() {
    let (r, lines) = run(Fixture::default().scale("scale_nan"));
    let local = r.summary.local.as_ref().unwrap();
    assert!(local.findings.iter().all(|f| f.cmp.as_ref().unwrap().nan_only_one_side == 1), "{:#?}", local.findings);
    assert!(line(&lines, "local     ✗").contains("· 1 nan"), "{lines:#?}");
    // a NaN row has no finite move to measure against the limit: it is an
    // infinite one, not a zero one
    let lg = r.summary.logits.as_ref().unwrap();
    assert!(lg.kl_max.is_infinite(), "{lg:?}");
    assert_eq!(verdict(&r).0, 1, "{}", r.summary.verdict.summary);
    assert!(r.summary.verdict.summary.contains("KL inf above the limit"), "{}", r.summary.verdict.summary);
}

#[test]
fn a_write_outside_the_reference_write_set_is_reported() {
    let (r, lines) = run(Fixture::default().mix("mix_leak"));
    let local = r.summary.local.as_ref().unwrap();
    let leak = local.findings.iter().find(|f| f.buffer == "state kv").expect("a state finding");
    assert!(leak.what.contains("outside A's write-set"), "{leak:?}");
    assert!(lines.iter().any(|l| l.contains("wrote") && l.contains("outside A's write-set")), "{lines:#?}");
}

#[test]
fn perf_reports_every_driven_program_and_never_the_verdict() {
    let mut o = options();
    o.perf = true;
    o.graph_step = true;
    o.sweep = true;
    let (r, lines) = test(&Fixture::default(), &Fixture::default().scale("scale_same"), &o).unwrap();
    assert_eq!(verdict(&r).0, 0);
    let perf = r.summary.perf.as_ref().unwrap();
    let programs: Vec<&str> = perf.steps.iter().map(|s| s.program.as_str()).collect();
    assert_eq!(programs, ["decode", "prefill"]);
    assert_eq!(perf.steps[0].spans, 2);
    assert!(perf.steps[0].graph_ms.is_some() && perf.steps[1].graph_ms.is_none());
    assert_eq!(perf.sweep.iter().map(|p| p.rows).collect::<Vec<_>>(), [1, 3, 8]);
    assert_eq!(perf.roofline.len(), 2);
    assert!(lines.iter().any(|l| l.starts_with("sweep     prefill")), "{lines:#?}");
}

#[test]
fn identical_manifests_have_nothing_to_test() {
    let (ma, mb) = (Fixture::default().manifest(), Fixture::default().manifest());
    let d = kern_test::diff::diff(&ma, &mb);
    assert!(d.spans.is_empty() && d.ops.is_empty() && d.programs.is_empty());
    assert_eq!(d.lines(), ["diff      no interface or implementation differs"]);
}

#[test]
fn the_json_summary_names_every_section_and_the_archive_every_finding() {
    let (r, _) = run(Fixture::default().scale("scale_round"));
    let v = serde_json::to_value(&r.summary).unwrap();
    for k in ["a", "b", "diff", "tap", "local", "logits", "noise", "verdict"] {
        assert!(v.get(k).is_some(), "no `{k}` in {v}");
    }
    assert!(v.get("perf").is_none());
    assert_eq!(v["diff"]["programs"][0]["spans"], 2);
    assert!(r.detail["local"].as_array().unwrap().len() >= r.summary.local.as_ref().unwrap().findings.len());
}

#[test]
fn a_side_of_several_ranks_is_compared_rank_by_rank() {
    let (r, lines) =
        test(&Fixture::default().ranks(2), &Fixture::default().ranks(2).scale("scale_rank1_wrong"), &options())
            .unwrap();
    let tap = r.summary.tap.as_ref().unwrap();
    assert_eq!((tap.ranks, tap.spans), (2, 4));
    let local = r.summary.local.as_ref().unwrap();
    // every rank's span is compared against its own rank of A; only rank 1 differs
    assert_eq!((local.compared, local.bit_identical), (24, 12));
    assert!(local.findings.iter().all(|f| f.span.starts_with("rank 1 ")), "{:#?}", local.findings);
    assert!(local.outputs.iter().all(|f| f.buffer.starts_with("rank ")), "{:#?}", local.outputs);
    let lg = r.summary.logits.as_ref().unwrap();
    assert!(lg.kl_at.starts_with("rank 1 ") && lg.flipped.iter().all(|f| f.row.starts_with("rank 1 ")), "{lg:#?}");
    assert!(verdict(&r).0 != 0 && line(&lines, "tap").contains("2 ranks"), "{lines:#?}");
}

#[test]
fn a_and_b_must_run_as_the_same_number_of_ranks() {
    let err = test(&Fixture::default().ranks(2), &Fixture::default(), &options()).unwrap_err();
    assert!(format!("{err:#}").contains("A recorded 2 ranks; B runs as 1"), "{err:#}");
}

#[test]
fn a_buffer_declared_differently_on_the_two_sides_is_not_compared() {
    let (r, lines) = run(Fixture::default().scale("scale_same").wide_act());
    assert!(
        line(&lines, "diff").contains("buffer act changed") || lines.iter().any(|l| l.contains("act changed")),
        "{lines:#?}"
    );
    let local = r.summary.local.as_ref().unwrap();
    assert_eq!((local.compared, local.one_sided.clone()), (0, vec!["act".to_string()]));
    assert!(line(&lines, "local     declared differently").contains("act"), "{lines:#?}");
    // nothing compared is not everything identical: the oracle decides
    assert_eq!(
        verdict(&r),
        (0, "spans differ, but the end-to-end logits are bit-identical on all 9 rows".into()),
        "{lines:#?}"
    );
}

#[test]
fn nothing_compared_is_not_everything_identical() {
    // B declares the only compared buffer differently and drifts: no span
    // is comparable, and the verdict comes from the oracle, not from 0/0.
    let (r, lines) = run(Fixture::default().scale("scale_drift").wide_act());
    assert_eq!(r.summary.local.as_ref().unwrap().compared, 0);
    assert_eq!(verdict(&r).0, 2, "{lines:#?}");
    assert!(
        r.summary.verdict.summary.starts_with("spans differ; end-to-end KL up to"),
        "{}",
        r.summary.verdict.summary
    );
}

#[test]
fn a_logits_buffer_is_found_by_the_last_segment_of_its_name() {
    let (r, _) = test(
        &Fixture::default().logits("target_head.logits"),
        &Fixture::default().logits("target_head.logits").scale("scale_round"),
        &options(),
    )
    .unwrap();
    assert_eq!(verdict(&r).1, "logit evidence", "{}", r.summary.verdict.summary);
    assert_eq!(r.summary.logits.as_ref().unwrap().rows, 9);
}

#[test]
fn a_changed_once_program_is_each_sides_own_setup_not_an_untapped_program() {
    let (r, lines) = test(
        &Fixture::default().once("fill_table"),
        &Fixture::default().once("fill_table_v2").scale("scale_same"),
        &options(),
    )
    .unwrap();
    assert!(r.summary.diff.spans.contains_key("prep"), "{:?}", r.summary.diff.spans.keys());
    assert_eq!(verdict(&r), (0, "bit-identical at every span".into()), "{lines:#?}");
    assert!(r.summary.local.as_ref().unwrap().undriven.is_empty());
}

#[test]
fn a_workspace_written_in_places_is_kept_and_compared_as_its_write_set() {
    let (r, lines) =
        test(&Fixture::default().slab(), &Fixture::default().slab().scale("scale_same"), &options()).unwrap();
    assert_eq!(verdict(&r), (0, "bit-identical at every span".into()), "{lines:#?}");
    let local = r.summary.local.as_ref().unwrap();
    // act and slab at every span; the slab compared on the 64 bytes its layer wrote
    assert_eq!((local.compared, local.bit_identical), (24, 24));
    // whole copies of the 4 KB slab before and after each of the 12 spans would be
    // 96 KB; as deltas it is one whole slab, one block per write, and the acts
    let tap = r.summary.tap.as_ref().unwrap();
    assert!(
        tap.snapshot_bytes >= common::SLAB + 12 * 64 && tap.snapshot_bytes < 2 * common::SLAB,
        "{}",
        tap.snapshot_bytes
    );
}

#[test]
fn a_write_outside_the_reference_write_set_of_a_buffer_is_reported() {
    let (r, lines) =
        test(&Fixture::default().slab(), &Fixture::default().slab().scale("scale_slab_leak"), &options()).unwrap();
    let local = r.summary.local.as_ref().unwrap();
    let leak = local.findings.iter().find(|f| f.buffer == "slab").expect("a slab finding");
    assert!(leak.what.contains("wrote 64 B outside A's write-set"), "{leak:?}");
    assert!(leak.cmp.as_ref().unwrap().identical(), "{leak:?}");
    // the block A wrote agrees; only the leak differs, and the logits never see it
    assert_eq!(
        verdict(&r),
        (0, "spans differ, but the end-to-end logits are bit-identical on all 9 rows".into()),
        "{lines:#?}"
    );
}
