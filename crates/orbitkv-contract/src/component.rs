use serde::{Deserialize, Serialize};

/// One independently materialized component of restorable model state.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateComponent {
    AttentionKv,
    MlaKv,
    RecurrentCheckpoint,
    ConvolutionState,
    SlidingWindowKv,
    DraftKv,
    IndexerState,
    /// Framework or model-specific state that has not yet received a portable
    /// OrbitKV component kind. The name is part of the state identity.
    Opaque(String),
}
