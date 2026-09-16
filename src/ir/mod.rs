//! A small, static, effect-aware task graph for hybrid decoders.

mod effect;
mod graph;
mod task;

pub use effect::{EffectKind, StateEffect, StateId, StateRegion, StateScope, StateSpec};
pub use graph::{GraphError, TaskGraph};
pub use task::{
    FullAttentionGeometry, GatedDeltaGeometry, ImplementationCandidate, Operation, ProjectionRole,
    Task, TaskId, ValueId,
};
