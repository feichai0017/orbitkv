use orbitkv::RecurrentFamily;

use super::*;
use crate::{AttentionClass, FixedStateClass};

fn config() -> DecoderConfig {
    DecoderConfig {
        layers: 4,
        hidden_size: 128,
        intermediate_size: 256,
        query_heads: 2,
        kv_heads: 1,
        head_dim: 64,
        vocabulary_size: 320,
        tensor_prefix: "model".into(),
        rope_theta: 10_000.0,
        rotary_dimensions: 64,
        rms_epsilon: 1e-6,
        tied_embeddings: true,
        embedding_scale: 1.0,
        activation: super::super::DecoderActivation::Silu,
        block_layout: super::super::DecoderBlockLayout::PreNorm,
        norm_weights: super::super::DecoderNormWeights::Direct,
        local_rope_theta: None,
        attention_softmax_scale: 64_f64.sqrt().recip(),
        attention_output_gate: false,
        layer_kinds: Some(
            vec![
                DecoderLayerKind::Linear,
                DecoderLayerKind::Linear,
                DecoderLayerKind::Linear,
                DecoderLayerKind::Full,
            ]
            .into_boxed_slice(),
        ),
        gated_delta: Some(GatedDeltaConfig {
            key_heads: 2,
            value_heads: 2,
            key_width: 4,
            value_width: 4,
            convolution_kernel_width: 3,
        }),
        weight_format: super::super::DecoderWeightFormat::Float,
    }
}

fn plan() -> ExecutorPlan {
    crate::tests::support::executor_plan_with_fixed_states(
        "stateful",
        16,
        vec![AttentionClass {
            class_id: 0,
            name: "full".into(),
            layers: vec![3].into_boxed_slice(),
            page_tokens: 16,
            key_bytes_per_token_per_layer: 128,
            value_bytes_per_token_per_layer: 128,
            visibility: AttentionVisibility::Full,
        }],
        vec![
            FixedStateClass {
                state_id: 1,
                name: "recurrent".into(),
                layers: vec![0, 1, 2].into_boxed_slice(),
                storage: FixedStateStorage::Recurrent {
                    family: RecurrentFamily::Gdn,
                    bytes_per_layer: 128,
                    slots_per_request: 2,
                    bytes_per_request: 768,
                },
            },
            FixedStateClass {
                state_id: 2,
                name: "convolution".into(),
                layers: vec![0, 1, 2].into_boxed_slice(),
                storage: FixedStateStorage::Convolution {
                    bytes_per_layer: 96,
                    kernel_width: 3,
                    slots_per_request: 2,
                    bytes_per_request: 576,
                },
            },
        ],
    )
}

#[test]
fn compiles_joint_token_and_fixed_state_ownership() {
    let topology = DecoderTopology::compile(&config(), &plan()).unwrap();
    assert_eq!(
        topology.layer(0),
        Some(DecoderLayerState::GatedDelta {
            recurrent_state_id: 1,
            convolution_state_id: 2,
            geometry: config().gated_delta.unwrap(),
        })
    );
    assert_eq!(
        topology.layer(3),
        Some(DecoderLayerState::TokenKv { class_id: 0 })
    );
    assert_eq!(topology.token_layers(), BTreeSet::from([3]));
}

#[test]
fn rejects_missing_or_mismatched_fixed_state_classes() {
    let mut missing = plan();
    missing.fixed_states = missing.fixed_states[..1].into();
    assert!(DecoderTopology::compile(&config(), &missing).is_err());

    let mut mismatched = plan();
    let FixedStateStorage::Recurrent {
        ref mut bytes_per_layer,
        ..
    } = mismatched.fixed_states[0].storage
    else {
        unreachable!();
    };
    *bytes_per_layer -= 4;
    assert!(DecoderTopology::compile(&config(), &mismatched).is_err());
}
