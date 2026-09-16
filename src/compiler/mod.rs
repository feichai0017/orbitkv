//! Deterministic, bounded execution-island formation.

mod partition;

pub use partition::{CompileError, ExecutionIsland, ExecutionPlan, IslandKind, partition_baseline};
