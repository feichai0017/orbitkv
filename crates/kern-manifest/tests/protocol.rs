//! The serving protocol as a driver reads it: what a manifest's `fill`,
//! `batch` and `once` declarations project to, and what they refuse.

use std::collections::BTreeMap;

use kern_manifest::protocol::{Axis, Bound, LineTable, PageTable, ProtocolErrors, Rows};
use kern_manifest::types::{Batch, Buffer, Dim, Fill, Op};
use kern_manifest::{verify, Manifest, Protocol};
/// A single-rank contract: a prefill chunk that only fills state, a
/// bs=1 decode and a batched one, a page table and a line table. The
/// fixture verifies, so every buffer and state is consumed by a call.
fn plain() -> Manifest {
    Manifest::from_json(include_str!("fixtures/plain.json")).unwrap()
}

/// The same plus a speculative round: 4 rows per sequence, an anchor,
/// per-row tokens and a count.
fn speculative() -> Manifest {
    let mut m = plain();
    let buf = |s: &str| serde_json::from_str::<Buffer>(s).unwrap();
    m.buffers
        .insert("anchor_token".into(), buf(r#"{"kind": "input", "dtype": "i64", "shape": ["seqs"], "fill": "token"}"#));
    m.buffers.insert(
        "verify_tokens".into(),
        buf(r#"{"kind": "output", "dtype": "i64", "shape": ["seqs", 4], "fill": "tokens"}"#),
    );
    m.buffers.insert("nacc".into(), buf(r#"{"kind": "output", "dtype": "i32", "shape": ["seqs"], "fill": "count"}"#));
    m.ops.insert("read".into(), op(&["in buffer<i64>"]));
    m.ops.insert("accept".into(), op(&["out buffer<i64>", "out buffer<i32>"]));
    m.programs.insert(
        "round".into(),
        serde_json::from_str(
            r#"{"batch": {"groups": 2, "rows": 4}, "calls": [
            {"op": "read", "args": [{"buf": "anchor_token"}]},
            {"op": "accept", "args": [{"buf": "verify_tokens"}, {"buf": "nacc"}]}]}"#,
        )
        .unwrap(),
    );
    m
}

/// A tray manifest: the tokens, the line table and the output over
/// the tray's rows, the rest this rank's own.
fn tray() -> Manifest {
    Manifest::from_json(include_str!("fixtures/tray.json")).unwrap()
}

/// The plain contract plus a run: a `span` var, the `span_at` word
/// and a decode step over 4 sequences, one of which may feed a run.
fn spanned() -> Manifest {
    let mut m = plain();
    m.vars.insert("span".into(), serde_json::from_str(r#"{"max": 6}"#).unwrap());
    m.buffers.insert(
        "span_at".into(),
        serde_json::from_str(r#"{"kind": "input", "dtype": "i32", "shape": [1], "fill": "span_at"}"#).unwrap(),
    );
    m.ops.insert("read_span".into(), op(&["in buffer<i32>"]));
    let mut span = m.programs["decode"].clone();
    span.batch = Some(Batch { groups: 4, rows: Dim::Const(1), span: Some("span".into()) });
    span.calls.push(serde_json::from_str(r#"{"op": "read_span", "args": [{"buf": "span_at"}]}"#).unwrap());
    m.programs.insert("decode_span".into(), span);
    m
}

/// An op with an extern launch, for a fixture that only needs the ABI.
fn op(params: &[&str]) -> Op {
    let params: Vec<String> = params.iter().map(|p| format!("\"{p}\"")).collect();
    serde_json::from_str(&format!(
        r#"{{"params": [{}], "impl": {{"launches": [{{"entry": "extern:x"}}]}}}}"#,
        params.join(", ")
    ))
    .unwrap()
}

/// The protocol of a fixture, which has to verify first: a fixture that
/// does not is a broken fixture, not a protocol finding.
fn protocol(m: &Manifest) -> Result<Protocol, ProtocolErrors> {
    Protocol::check(&verify(m.clone()).unwrap_or_else(|e| panic!("fixture does not verify: {e}")))
}

fn rejects(m: &Manifest, what: &str) {
    let Err(e) = protocol(m) else { panic!("accepted, expected `{what}`") };
    assert!(e.iter().any(|x| x.contains(what)), "no `{what}` in {e:#?}");
}

#[test]
fn plain_contract() {
    let p = protocol(&plain()).unwrap();
    assert_eq!((p.rows.var.as_str(), p.rows.max, p.groups.var.as_str(), p.groups.max), ("tokens", 8, "seqs", 4));
    assert_eq!(p.tray, None);
    assert_eq!(
        (p.token_rows().name.as_str(), p.slots().name.as_str(), p.seq_lens().name.as_str()),
        ("token_ids", "slot_mapping", "seq_lens")
    );
    assert_eq!(p.any(Fill::CuSeqlens).map(|f| f.axis), Some(Axis::Fixed(5)));
    assert_eq!(p.page_tables, vec![PageTable { name: "block_table".into(), width: 3 }]);
    assert_eq!(p.line_tables, vec![LineTable { name: "line_index".into(), lines: 3, width: 1, axis: Axis::Groups }]);
    let names: Vec<(&str, u64, Rows, bool)> =
        p.forwards.iter().map(|f| (f.name.as_str(), f.groups, f.rows, f.emits.is_some())).collect();
    assert!(p.forwards.iter().all(|f| !f.span));
    assert_eq!(
        names,
        [
            ("decode", 1, Rows::Const(1), true),
            ("decode_batch", 4, Rows::Const(1), true),
            ("prefill", 1, Rows::Var, false)
        ]
    );
    // The tightest bound wins; a prefill chunk is the var-rows call.
    assert_eq!(p.forward(1, Rows::Const(1)).map(|f| f.name.as_str()), Some("decode"));
    assert_eq!(p.forward(3, Rows::Const(1)).map(|f| f.name.as_str()), Some("decode_batch"));
    assert_eq!(p.forward(5, Rows::Const(1)), None);
    assert_eq!(p.chunk().map(|f| f.name.as_str()), Some("prefill"));
    assert_eq!((p.row_shapes(), p.max_groups(Rows::Const(1))), (vec![1], 4));
    assert_eq!(p.vars(3, 2, 6), BTreeMap::from([("tokens".into(), 6), ("seqs".into(), 3)]));
}

#[test]
fn speculative_contract() {
    let p = protocol(&speculative()).unwrap();
    let round = p.forwards.iter().find(|f| f.name == "round").unwrap();
    assert_eq!((round.groups, round.rows), (2, Rows::Const(4)));
    assert_eq!(round.emits.map(|i| (p.fills[i].name.as_str(), p.fills[i].width)), Some(("verify_tokens", 4)));
    assert_eq!(round.count.map(|i| p.fills[i].name.as_str()), Some("nacc"));
    assert_eq!(p.filled(Fill::Token, Axis::Groups).map(|f| f.name.as_str()), Some("anchor_token"));
    assert_eq!((p.row_shapes(), p.max_groups(Rows::Const(4))), (vec![1, 4], 2));
    // The plain decode in the same manifest hands back one per sequence.
    let decode = p.forwards.iter().find(|f| f.name == "decode").unwrap();
    assert_eq!((decode.emits.map(|i| p.fills[i].width), decode.count), (Some(1), None));
}

#[test]
fn span_contract() {
    let p = protocol(&spanned()).unwrap();
    assert_eq!(p.span, Some(Bound { var: "span".into(), max: 6 }));
    assert_eq!(p.any(Fill::SpanAt).map(|f| (f.name.as_str(), f.axis)), Some(("span_at", Axis::Fixed(1))));
    // A call with a run goes through the span program, one without
    // through the plain ones; the span program is no plain shape.
    assert_eq!(p.spanned(3).map(|f| f.name.as_str()), Some("decode_span"));
    assert_eq!(p.forward(3, Rows::Const(1)).map(|f| f.name.as_str()), Some("decode_batch"));
    assert_eq!((p.spanned(5), p.row_shapes(), p.max_groups(Rows::Const(1))), (None, vec![1], 4));
    assert_eq!(protocol(&plain()).unwrap().span, None);
}

#[test]
fn span_rules() {
    let mut m = spanned();
    m.programs.get_mut("decode_span").unwrap().batch.as_mut().unwrap().rows = Dim::Const(4);
    rejects(&m, "a span rides a call of one row per sequence, not `4`");
    let mut m = spanned();
    m.programs.get_mut("decode_span").unwrap().batch.as_mut().unwrap().span = Some("tokens".into());
    m.vars.remove("span");
    rejects(&m, "batch.span is `tokens`, which sizes the call itself");
    let mut m = spanned();
    m.vars.get_mut("span").unwrap().max = 9;
    rejects(&m, "a run of 9 rows (`span`) exceeds the 8 rows `tokens` allows");
    let mut m = spanned();
    m.buffers.get_mut("span_at").unwrap().fill = None;
    rejects(&m, "no input has fill `span_at`");
    let mut m = spanned();
    m.buffers.get_mut("span_at").unwrap().shape = vec![Dim::Const(2)];
    rejects(&m, "expected i32 [1]");
    let mut m = spanned();
    m.vars.insert("other".into(), serde_json::from_str(r#"{"max": 2}"#).unwrap());
    m.programs.get_mut("decode").unwrap().batch.as_mut().unwrap().span = Some("other".into());
    rejects(&m, "one var sizes every run");
}

#[test]
fn tray_contract() {
    let p = protocol(&tray()).unwrap();
    assert_eq!(p.tray, Some(Bound { var: "rows".into(), max: 32 }));
    assert_eq!(p.token_rows().axis, Axis::Tray);
    assert_eq!(p.line_tables[0].axis, Axis::Tray);
    assert_eq!(p.any(Fill::Error).map(|f| f.name.as_str()), Some("tp_err"));
    assert_eq!(p.once, vec!["tp_init".to_string()]);
    assert_eq!(p.vars(2, 1, 8), BTreeMap::from([("tokens".into(), 2), ("seqs".into(), 2), ("rows".into(), 8)]));
}

#[test]
fn encodes_by_dtype() {
    let p = protocol(&plain()).unwrap();
    assert_eq!(p.seq_lens().encode(&[3, -1]), vec![3, 0, 0, 0, 255, 255, 255, 255]);
    assert_eq!(p.slots().encode(&[2]), 2i64.to_le_bytes());
    assert_eq!(p.seq_lens().decode(&[7, 0, 0, 0]), vec![7]);
}

#[test]
fn missing_pieces_are_all_named() {
    let mut m = plain();
    m.buffers.get_mut("slot_mapping").unwrap().fill = None;
    m.buffers.get_mut("seq_lens").unwrap().fill = None;
    let Err(e) = protocol(&m) else { panic!() };
    assert_eq!(e.len(), 2);
    rejects(&m, "no input has fill `slot`");
    rejects(&m, "no input has fill `seq_len`");
}

#[test]
fn shape_rules() {
    let mut m = plain();
    m.buffers.get_mut("token_ids").unwrap().shape = vec![Dim::Var("seqs".into())];
    rejects(&m, "no input has fill `token` over the rows");
    let mut m = plain();
    m.buffers.get_mut("cu_seqlens_q").unwrap().shape = vec![Dim::Const(4)];
    rejects(&m, "expected [n] with n >= groups + 1");
    let mut m = plain();
    m.buffers.get_mut("block_table").unwrap().shape = vec![Dim::Var("seqs".into())];
    rejects(&m, "page table `block_table` is shaped");
    let mut m = plain();
    m.buffers.get_mut("line_index").unwrap().shape = vec![Dim::Const(3), Dim::Var("tokens".into())];
    rejects(&m, "column per row");
    let mut m = plain();
    m.buffers.get_mut("seq_lens").unwrap().shape = vec![Dim::Var("tokens".into())];
    rejects(&m, "both over var `tokens`");
}

#[test]
fn batch_rules() {
    let mut m = plain();
    m.programs.get_mut("decode_batch").unwrap().batch = Some(Batch { groups: 5, rows: Dim::Const(1), span: None });
    rejects(&m, "5 groups exceed the 4");
    let mut m = plain();
    m.programs.get_mut("decode_batch").unwrap().batch = Some(Batch { groups: 4, rows: Dim::Const(3), span: None });
    rejects(&m, "4 sequences of 3 rows exceed the 8");
    let mut m = plain();
    m.programs.get_mut("prefill").unwrap().batch =
        Some(Batch { groups: 2, rows: Dim::Var("tokens".into()), span: None });
    rejects(&m, "one sequence, not 2 groups");
    let mut m = plain();
    m.programs.get_mut("prefill").unwrap().batch = Some(Batch { groups: 1, rows: Dim::Var("seqs".into()), span: None });
    rejects(&m, "the rows of a call go in `tokens`");
    let mut m = plain();
    m.programs.get_mut("decode_batch").unwrap().batch = Some(Batch { groups: 1, rows: Dim::Const(1), span: None });
    rejects(&m, "accept the same call shape");
    let mut m = plain();
    for p in m.programs.values_mut() {
        p.batch = None;
    }
    rejects(&m, "no program declares a `batch`");
    let mut m = plain();
    m.buffers.remove("next_token");
    m.ops.remove("head");
    let write = m.programs["prefill"].calls[0].clone();
    for p in ["decode", "decode_batch"] {
        m.programs.get_mut(p).unwrap().calls = vec![write.clone()];
    }
    rejects(&m, "no call hands a token back");
}

#[test]
fn what_a_forward_hands_back_is_dataflow() {
    // A round handing back 4 per sequence must be a 4-row call.
    let mut m = speculative();
    m.programs.get_mut("round").unwrap().batch = Some(Batch { groups: 2, rows: Dim::Const(3), span: None });
    rejects(&m, "hands back `verify_tokens` of 4 per sequence, but a call has 3 rows");
    // A count needs several tokens per sequence to count.
    let mut m = speculative();
    m.programs.get_mut("decode").unwrap().calls.push(
        serde_json::from_str(r#"{"op": "accept", "args": [{"buf": "verify_tokens"}, {"buf": "nacc"}]}"#).unwrap(),
    );
    rejects(&m, "writes 2 `tokens` outputs");
    let mut m = speculative();
    m.programs.get_mut("decode").unwrap().calls =
        vec![serde_json::from_str(r#"{"op": "accept", "args": [{"buf": "next_token"}, {"buf": "nacc"}]}"#).unwrap()];
    rejects(&m, "no `tokens` output of several per sequence to count");
}
