use std::collections::BTreeMap;

use kern_manifest::types::{State, Var};
use serde::Serialize;

use crate::model::{LayerKind, Qwen38Contract};

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ProgramCall {
    pub label: String,
    pub op: String,
}

impl ProgramCall {
    fn new(label: impl Into<String>, op: impl Into<String>) -> Self {
        Self { label: label.into(), op: op.into() }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ProgramSkeleton {
    pub name: String,
    pub calls: Vec<ProgramCall>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Qwen38ProgramSkeletons {
    pub programs: BTreeMap<String, ProgramSkeleton>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Qwen38Declarations {
    pub variables: BTreeMap<String, Var>,
    pub states: BTreeMap<String, State>,
}

impl Qwen38Declarations {
    pub fn lower() -> Self {
        Self {
            variables: BTreeMap::from([("tokens".into(), Var { max: 8_192 }), ("seqs".into(), Var { max: 128 })]),
            states: BTreeMap::from([
                (
                    "gdn".into(),
                    State {
                        bytes_per_token: 0,
                        bytes: 0,
                        bytes_per_seq: Qwen38Contract::oracle_gdn_bytes_per_sequence(),
                    },
                ),
                (
                    "kv".into(),
                    State { bytes_per_token: Qwen38Contract::oracle_kv_bytes_per_token(), bytes: 0, bytes_per_seq: 0 },
                ),
            ]),
        }
    }
}

impl Qwen38ProgramSkeletons {
    /// Lower the pinned Qwen3.8 topology into the three target program
    /// skeletons. Buffer arguments and physical kernel modules remain the
    /// next lowering stage; labels and implementation families are complete.
    pub fn lower() -> Self {
        let programs = [
            ("prefill", lower_program(ProgramKind::Prefill)),
            ("decode", lower_program(ProgramKind::Decode)),
            ("decode_batch", lower_program(ProgramKind::DecodeBatch)),
        ]
        .into_iter()
        .map(|(name, calls)| (name.into(), ProgramSkeleton { name: name.into(), calls }))
        .collect();
        Self { programs }
    }
}

#[derive(Clone, Copy)]
enum ProgramKind {
    Prefill,
    Decode,
    DecodeBatch,
}

fn lower_program(kind: ProgramKind) -> Vec<ProgramCall> {
    let mut calls = vec![
        ProgramCall::new("embed", "embedding"),
        ProgramCall::new("rope_cos", "embedding"),
        ProgramCall::new("rope_sin", "embedding"),
        ProgramCall::new("l0.input_norm", "gemma_norm"),
    ];
    for layer in 0..Qwen38Contract::LAYERS {
        let final_layer = layer + 1 == Qwen38Contract::LAYERS;
        match (kind, Qwen38Contract::layer_kind(layer)) {
            (ProgramKind::Prefill, LayerKind::GatedDelta) => prefill_gdn(&mut calls, layer),
            (ProgramKind::Prefill, LayerKind::FullAttention) => prefill_attention(&mut calls, layer),
            (ProgramKind::Decode, LayerKind::GatedDelta) => decode_gdn(&mut calls, layer, false),
            (ProgramKind::Decode, LayerKind::FullAttention) => decode_attention(&mut calls, layer, final_layer, false),
            (ProgramKind::DecodeBatch, LayerKind::GatedDelta) => decode_gdn(&mut calls, layer, true),
            (ProgramKind::DecodeBatch, LayerKind::FullAttention) => {
                decode_attention(&mut calls, layer, final_layer, true);
            }
        }
    }
    match kind {
        ProgramKind::Prefill => {
            calls.push(ProgramCall::new("last_row", "last_row"));
            calls.push(ProgramCall::new("lm_head", "gemm"));
            calls.push(ProgramCall::new("sample", "argmax_row"));
        }
        ProgramKind::Decode => {
            calls.push(ProgramCall::new("lm_head", "gemm"));
            calls.push(ProgramCall::new("sample", "argmax_row"));
        }
        ProgramKind::DecodeBatch => {
            calls.push(ProgramCall::new("lm_head", "gemm16_lm_head"));
            calls.push(ProgramCall::new("sample", "argmax"));
        }
    }
    calls
}

fn prefill_gdn(calls: &mut Vec<ProgramCall>, layer: u16) {
    layer_calls(
        calls,
        layer,
        &[
            ("in_proj_qkvz", "gemm"),
            ("in_proj_ba", "gemm"),
            ("conv", "conv_fwd"),
            ("post_conv", "post_conv"),
            ("cumsum", "cumsum"),
            ("kkt", "kkt"),
            ("solve_tril", "solve_tril"),
            ("recompute_wu", "recompute"),
            ("h0_gather", "line_gather"),
            ("chunk_h", "chunk_h"),
            ("ht_scatter", "line_scatter"),
            ("chunk_o", "chunk_o"),
            ("z_copy", "copy_rows"),
            ("gated_norm", "gated_norm"),
            ("out_proj", "gemm"),
        ],
    );
    common_mlp(calls, layer, false);
}

fn prefill_attention(calls: &mut Vec<ProgramCall>, layer: u16) {
    layer_calls(
        calls,
        layer,
        &[
            ("qkv_proj", "gemm"),
            ("attn_prep", "attn_prep"),
            ("attn", "attn_prefill"),
            ("gate", "sigmoid_mul"),
            ("o_proj", "gemm"),
        ],
    );
    common_mlp(calls, layer, false);
}

fn decode_gdn(calls: &mut Vec<ProgramCall>, layer: u16, batch: bool) {
    if batch {
        layer_calls(
            calls,
            layer,
            &[
                ("in_proj_qkvz_ba", "gemm16_in_proj"),
                ("gdn_conv", "gdn_conv"),
                ("gdn_step", "gdn_step"),
                ("out_proj", "gemm16_out"),
            ],
        );
    } else {
        layer_calls(
            calls,
            layer,
            &[
                ("in_proj_qkvz", "gemm"),
                ("in_proj_ba", "gemm"),
                ("gdn_conv", "gdn_conv"),
                ("gdn_step", "gdn_step"),
                ("out_proj", "gemm"),
            ],
        );
    }
    common_mlp(calls, layer, batch);
}

fn decode_attention(calls: &mut Vec<ProgramCall>, layer: u16, _final_layer: bool, batch: bool) {
    layer_calls(
        calls,
        layer,
        &[
            ("qkv_proj", if batch { "gemm16_qkv" } else { "gemm" }),
            ("attn_prep", "attn_prep"),
            ("attn", if batch { "attn_batch" } else { "attn" }),
            ("gate", "sigmoid_mul"),
            ("o_proj", if batch { "gemm16_o" } else { "gemm" }),
        ],
    );
    common_mlp(calls, layer, batch);
}

fn common_mlp(calls: &mut Vec<ProgramCall>, layer: u16, batch: bool) {
    let final_layer = layer + 1 == Qwen38Contract::LAYERS;
    let norm_label = if final_layer { "final_norm" } else { "next_input_norm" };
    if batch {
        layer_calls(
            calls,
            layer,
            &[
                ("post_attn_norm", "gemma_fused_norm"),
                ("gate_up_silu", "gemm16_gate_up_silu"),
                ("down_proj", "gemm"),
                (norm_label, "gemma_fused_norm"),
            ],
        );
    } else {
        layer_calls(
            calls,
            layer,
            &[
                ("post_attn_norm", "gemma_fused_norm"),
                ("gate_up", "gemm"),
                ("silu_mul", "silu_mul"),
                ("down_proj", "gemm"),
                (norm_label, "gemma_fused_norm"),
            ],
        );
    }
}

fn layer_calls(calls: &mut Vec<ProgramCall>, layer: u16, entries: &[(&str, &str)]) {
    calls.extend(entries.iter().map(|(label, op)| ProgramCall::new(format!("l{layer}.{label}"), *op)));
}
