use serde::{Deserialize, Serialize};

use super::StateEffect;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct TaskId(pub usize);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct ValueId(pub usize);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct GatedDeltaGeometry {
    pub key_heads: u16,
    pub value_heads: u16,
    pub key_width: u16,
    pub value_width: u16,
    pub convolution_width: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FullAttentionGeometry {
    pub query_heads: u16,
    pub kv_heads: u16,
    pub head_width: u16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProjectionRole {
    Embedding,
    LanguageModelHead,
}

/// Semantic operations are deliberately coarser than eager tensor operators.
/// Each block may lower to provider calls, a CUDA graph, or a generated island.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Operation {
    Projection { role: ProjectionRole },
    GatedDeltaBlock { layer: u16, geometry: GatedDeltaGeometry },
    FullAttentionBlock { layer: u16, geometry: FullAttentionGeometry },
    FinalNorm,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImplementationCandidate {
    ProviderGraph,
    GeneratedStateful,
    PersistentIsland,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Task {
    pub id: TaskId,
    pub operation: Operation,
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
    pub dependencies: Vec<TaskId>,
    pub state_effects: Vec<StateEffect>,
    pub candidates: Vec<ImplementationCandidate>,
}
