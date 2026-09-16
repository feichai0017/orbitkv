use serde::{Deserialize, Serialize};

/// Stable identifier for persistent model state.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct StateId(pub usize);

/// The allocation/lifetime class exposed to planning.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateScope {
    PerToken,
    PerSequence,
    Immutable,
}

/// Persistent state declaration before lowering to opaque `kern` bytes.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct StateSpec {
    pub id: StateId,
    pub name: String,
    pub scope: StateScope,
    /// Bytes for one token or sequence slot, or total bytes for immutable state.
    pub bytes: u64,
}

/// A statically known subregion. More precise regions may be added without
/// changing the state-effect vocabulary.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateRegion {
    Whole,
    Layer(u16),
}

/// Stateful actions whose ordering cannot be reconstructed from pointers.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EffectKind {
    Read,
    Append,
    Lookup,
    TentativeWrite { version: u8 },
    Commit { version: u8 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct StateEffect {
    pub state: StateId,
    pub region: StateRegion,
    pub kind: EffectKind,
}
