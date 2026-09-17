use crate::ir::{
    EffectKind, FullAttentionGeometry, GatedDeltaGeometry, ImplementationCandidate, Operation, ProjectionRole,
    StateEffect, StateId, StateRegion, StateScope, StateSpec, Task, TaskGraph, TaskId, ValueId,
};

const RECURRENT_STATE: StateId = StateId(0);
const CONVOLUTION_STATE: StateId = StateId(1);
const TOKEN_KV: StateId = StateId(2);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LayerKind {
    GatedDelta,
    FullAttention,
}

/// Exact M0 contract for the official `Qwen3.8-27B-FP8` checkpoint.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen38Contract;

impl Qwen38Contract {
    pub const MODEL: &'static str = "Qwen3.8-27B-FP8";
    pub const REPOSITORY: &'static str = "Qwen/Qwen3.8-27B-FP8";
    pub const REVISION: &'static str = "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a";
    pub const LAYERS: u16 = 64;
    pub const GATED_DELTA_LAYERS: u16 = 48;
    pub const ATTENTION_LAYERS: u16 = 16;
    pub const HIDDEN_SIZE: u32 = 5_120;
    pub const INTERMEDIATE_SIZE: u32 = 17_408;
    pub const FP8_SCALE_ROWS: u16 = 128;
    pub const FP8_SCALE_COLUMNS: u16 = 128;
    pub const MTP_LAYERS: u8 = 1;
    /// Padding in the imported physical GDN line after conv and recurrent state.
    pub const ORACLE_GDN_PADDING_BYTES_PER_LAYER: u64 = 4_096;

    pub const GATED_DELTA: GatedDeltaGeometry =
        GatedDeltaGeometry { key_heads: 16, value_heads: 48, key_width: 128, value_width: 128, convolution_width: 4 };

    pub const FULL_ATTENTION: FullAttentionGeometry =
        FullAttentionGeometry { query_heads: 24, kv_heads: 4, head_width: 256 };

    pub const fn layer_kind(layer: u16) -> LayerKind {
        if layer % 4 == 3 { LayerKind::FullAttention } else { LayerKind::GatedDelta }
    }

    pub const fn recurrent_bytes_per_layer() -> u64 {
        Self::GATED_DELTA.value_heads as u64
            * Self::GATED_DELTA.key_width as u64
            * Self::GATED_DELTA.value_width as u64
            * 4
    }

    pub const fn convolution_bytes_per_layer() -> u64 {
        let channels = Self::GATED_DELTA.key_heads as u64 * Self::GATED_DELTA.key_width as u64 * 2
            + Self::GATED_DELTA.value_heads as u64 * Self::GATED_DELTA.value_width as u64;
        channels * (Self::GATED_DELTA.convolution_width as u64 - 1) * 2
    }

    pub const fn kv_bytes_per_token_per_layer() -> u64 {
        2 * Self::FULL_ATTENTION.kv_heads as u64 * Self::FULL_ATTENTION.head_width as u64 * 2
    }

    pub const fn oracle_gdn_bytes_per_sequence() -> u64 {
        (Self::recurrent_bytes_per_layer()
            + Self::convolution_bytes_per_layer()
            + Self::ORACLE_GDN_PADDING_BYTES_PER_LAYER)
            * Self::GATED_DELTA_LAYERS as u64
    }

    pub const fn oracle_kv_bytes_per_token() -> u64 {
        Self::kv_bytes_per_token_per_layer() * Self::ATTENTION_LAYERS as u64
    }

    /// Construct a target-decode graph at block granularity.
    pub fn target_decode_graph() -> TaskGraph {
        let states = vec![
            StateSpec {
                id: RECURRENT_STATE,
                name: "gdn.recurrent".into(),
                scope: StateScope::PerSequence,
                bytes: Self::recurrent_bytes_per_layer() * Self::GATED_DELTA_LAYERS as u64,
            },
            StateSpec {
                id: CONVOLUTION_STATE,
                name: "gdn.convolution".into(),
                scope: StateScope::PerSequence,
                bytes: Self::convolution_bytes_per_layer() * Self::GATED_DELTA_LAYERS as u64,
            },
            StateSpec {
                id: TOKEN_KV,
                name: "attention.kv".into(),
                scope: StateScope::PerToken,
                bytes: Self::oracle_kv_bytes_per_token(),
            },
        ];

        let mut tasks = Vec::with_capacity(Self::LAYERS as usize + 3);
        let mut hidden = ValueId(0);
        let mut next_value = 1;

        push_task(
            &mut tasks,
            Operation::Projection { role: ProjectionRole::Embedding },
            vec![hidden],
            ValueId(next_value),
            Vec::new(),
            vec![ImplementationCandidate::ProviderGraph],
        );
        hidden = ValueId(next_value);
        next_value += 1;

        for layer in 0..Self::LAYERS {
            let (operation, effects, candidates) = match Self::layer_kind(layer) {
                LayerKind::GatedDelta => (
                    Operation::GatedDeltaBlock { layer, geometry: Self::GATED_DELTA },
                    vec![
                        effect(RECURRENT_STATE, layer, EffectKind::Read),
                        effect(RECURRENT_STATE, layer, EffectKind::TentativeWrite { version: 0 }),
                        effect(RECURRENT_STATE, layer, EffectKind::Commit { version: 0 }),
                        effect(CONVOLUTION_STATE, layer, EffectKind::Read),
                        effect(CONVOLUTION_STATE, layer, EffectKind::TentativeWrite { version: 0 }),
                        effect(CONVOLUTION_STATE, layer, EffectKind::Commit { version: 0 }),
                    ],
                    vec![
                        ImplementationCandidate::ProviderGraph,
                        ImplementationCandidate::GeneratedStateful,
                        ImplementationCandidate::PersistentIsland,
                    ],
                ),
                LayerKind::FullAttention => (
                    Operation::FullAttentionBlock { layer, geometry: Self::FULL_ATTENTION },
                    vec![effect(TOKEN_KV, layer, EffectKind::Append), effect(TOKEN_KV, layer, EffectKind::Read)],
                    vec![ImplementationCandidate::ProviderGraph],
                ),
            };
            push_task(&mut tasks, operation, vec![hidden], ValueId(next_value), effects, candidates);
            hidden = ValueId(next_value);
            next_value += 1;
        }

        push_task(
            &mut tasks,
            Operation::FinalNorm,
            vec![hidden],
            ValueId(next_value),
            Vec::new(),
            vec![ImplementationCandidate::ProviderGraph],
        );
        hidden = ValueId(next_value);
        next_value += 1;
        push_task(
            &mut tasks,
            Operation::Projection { role: ProjectionRole::LanguageModelHead },
            vec![hidden],
            ValueId(next_value),
            Vec::new(),
            vec![ImplementationCandidate::ProviderGraph],
        );

        TaskGraph {
            model: Self::MODEL.into(),
            inputs: vec![ValueId(0)],
            states,
            tasks,
            outputs: vec![ValueId(next_value)],
        }
    }
}

fn effect(state: StateId, layer: u16, kind: EffectKind) -> StateEffect {
    StateEffect { state, region: StateRegion::Layer(layer), kind }
}

fn push_task(
    tasks: &mut Vec<Task>,
    operation: Operation,
    inputs: Vec<ValueId>,
    output: ValueId,
    state_effects: Vec<StateEffect>,
    candidates: Vec<ImplementationCandidate>,
) {
    let id = TaskId(tasks.len());
    let dependencies = id.0.checked_sub(1).map(TaskId).into_iter().collect();
    tasks.push(Task { id, operation, inputs, outputs: vec![output], dependencies, state_effects, candidates });
}
