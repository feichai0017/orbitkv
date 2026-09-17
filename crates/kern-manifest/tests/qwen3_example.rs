//! The mined Qwen3-4B decode manifest must parse and verify clean: real
//! vLLM cubin ABIs for the flat kernels, wiring cross-checked against a
//! captured decode forward. Regenerate with
//! `python3 tools/gen_qwen3_decode.py` (needs the capture dump).

use kern_manifest::protocol::Rows;
use kern_manifest::{verify, Manifest, Protocol};

const QWEN3: &str = include_str!("../../../examples/qwen3-4b.json");

#[test]
fn qwen3_decode_mined_verifies() {
    let m = verify(Manifest::from_json(QWEN3).expect("parse"))
        .unwrap_or_else(|errs| panic!("mined manifest failed verification:\n{}", errs.join("\n")));
    // The serving protocol is in the manifest: a prefill chunk, a bs=1
    // decode and a batched one, every fill on its buffer.
    let p = Protocol::check(&m).unwrap_or_else(|e| panic!("{e}"));
    let shapes: Vec<(&str, u64)> = p.forwards.iter().map(|f| (f.name.as_str(), f.groups)).collect();
    assert_eq!(shapes, [("decode", 1), ("decode_batch", 256), ("prefill", 1)]);
    assert_eq!(p.fills.len(), 6);
    // decode: embed + l0 norm + 36 layers x 12 + lm_head + sample; attention's
    // reduce launch and argmax's two stages live inside their op impls.
    assert_eq!(m.programs["decode"].calls.len(), 2 + 36 * 12 + 2);
    // prefill: same forward minus the final_norm/lm_head/sample tail — the
    // last prompt token goes through `decode` instead.
    assert_eq!(m.programs["prefill"].calls.len(), 2 + 36 * 12 - 1);
    // The runtime's entire knowledge of the KV cache: a byte count.
    assert_eq!(m.states["kv"].bytes_per_token, 36 * 2 * 1024 * 2);

    let again = Manifest::from_json(&m.to_json()).expect("reparse");
    assert_eq!(m.to_json(), again.to_json());

    // Domains: the structural inputs carry priors the runtime enforces and
    // `kern test` synthesizes from; activations deliberately carry none.
    for name in ["token_ids", "slot_mapping", "block_table", "cu_seqlens_q", "next_token"] {
        assert!(m.buffers[name].domain.is_some(), "{name} has no domain");
    }
    assert!(m.buffers["residual"].domain.is_none());
}

/// The A/B fixture for `kern test`: identical to qwen3-4b.json except
/// `silu_mul`'s *impl* is the mined vLLM cubin (6-param launch ABI) instead
/// of the HF hub package (3-param ABI). The interface — `(out, in)`, every
/// ABI constant folded into the impl — and every call are untouched: a pure
/// impl swap.
const QWEN3_SILU_MINED: &str = include_str!("../../../examples/qwen3-4b-silu-mined.json");

#[test]
fn qwen3_silu_mined_fixture_verifies() {
    let m = verify(Manifest::from_json(QWEN3_SILU_MINED).expect("parse"))
        .unwrap_or_else(|errs| panic!("silu-mined fixture failed verification:\n{}", errs.join("\n")));
    let a = Manifest::from_json(QWEN3).unwrap();
    let (oa, ob) = (&a.ops["silu_mul"], &m.ops["silu_mul"]);
    assert_eq!(oa.params, ob.params, "interface must be unchanged");
    assert_eq!(oa.params.len(), 2);
    let (la, lb) = (&oa.imp.launches[0], &ob.imp.launches[0]);
    assert_eq!(la.params_of(oa).len(), 3);
    assert!(a.modules[la.module().unwrap()].source.starts_with("hf:"));
    assert_eq!(lb.params_of(ob).len(), 6);
    assert!(!m.modules[lb.module().unwrap()].source.starts_with("hf:"));
    for p in ["decode", "prefill"] {
        assert_eq!(
            serde_json::to_string(&a.programs[p]).unwrap(),
            serde_json::to_string(&m.programs[p]).unwrap(),
            "{p}: calls must be identical"
        );
    }
}

const QWEN3_DSPARK: &str = include_str!("../../../examples/qwen3-4b-dspark.json");

#[test]
fn qwen3_dspark_mined_verifies() {
    let m = verify(Manifest::from_json(QWEN3_DSPARK).expect("parse"))
        .unwrap_or_else(|errs| panic!("dspark manifest failed verification:\n{}", errs.join("\n")));
    // A round is one program: splice_draft, the draft (embed + l0 norm +
    // 5 layers x 12 incl. final norm + lm_head + 7 unrolled markov steps x
    // (embed, bias-accumulate, argmax)), splice_verify, the verify pass
    // (prefill body + 5 fc taps + 7-row lm_head + argmax), the precompute
    // (hidden_norm + fused KV GEMM + 5 x (k_norm, rope, cache)) and the
    // prefix-match count.
    let draft = 2 + 5 * 12 + 1 + 7 * 3;
    let verify_pass = 2 + 36 * 12 + 5 + 2;
    let precompute = 2 + 5 * 3;
    assert_eq!(m.programs["round"].calls.len(), 1 + draft + 1 + verify_pass + precompute + 1);
    // The spec-ready prefill carries the 5 fc taps, then the last row's
    // head (last_row, lm_head, argmax) and the chunk's precompute; decode
    // stays clean.
    assert_eq!(m.programs["prefill"].calls.len(), 2 + 36 * 12 + 5 + 3 + precompute);
    assert_eq!(m.programs["decode"].calls.len(), 2 + 36 * 12 + 2);
    // What the serving loop reads off it: a 7-row round counted by its
    // accept count, a prefill that hands back the first token.
    let p = Protocol::check(&m).unwrap_or_else(|e| panic!("{e}"));
    let round = p.forward(256, Rows::Const(7)).expect("a 7-row forward");
    assert_eq!((round.name.as_str(), round.count.is_some()), ("round", true));
    assert!(p.chunk().expect("a chunk forward").emits.is_some());
    assert_eq!(m.states["draft_kv"].bytes_per_token, 5 * 2 * 1024 * 2);
    // Both unified instances are ABI-identical; the manifest must pin each
    // to its own module or resolution would be ambiguous.
    let pinned = |op: &str| {
        let l = &m.ops[op].imp.launches[0];
        m.modules[l.module().unwrap()].sha256.clone()
    };
    assert_ne!(pinned("attn_prefill"), pinned("attn_draft"));
}

/// The toy manifest the schema page opens with: every top-level section,
/// one op, one call. Kept valid by this test.
const MINIMAL: &str = include_str!("../../../examples/minimal.json");

#[test]
fn minimal_example_verifies() {
    let m = verify(Manifest::from_json(MINIMAL).expect("parse"))
        .unwrap_or_else(|errs| panic!("examples/minimal.json failed verification:\n{}", errs.join("\n")));
    assert_eq!(m.programs["step"].calls.len(), 1);
    assert_eq!(m.ops["scale"].imp.launches[0].module(), Some("toy"));
}
