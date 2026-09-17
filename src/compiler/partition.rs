use std::fmt;

use serde::{Deserialize, Serialize};

use crate::ir::{GraphError, Operation, TaskGraph, TaskId};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IslandKind {
    Provider,
    StatefulGatedDelta { first_layer: u16, layers: u8 },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExecutionIsland {
    pub kind: IslandKind,
    pub tasks: Vec<TaskId>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExecutionPlan {
    pub model: String,
    pub islands: Vec<ExecutionIsland>,
}

#[derive(Debug)]
pub enum CompileError {
    InvalidGraph(GraphError),
    NonContiguousGatedDeltaRun { first_layer: u16, layers: usize },
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGraph(error) => write!(f, "invalid task graph: {error}"),
            Self::NonContiguousGatedDeltaRun { first_layer, layers } => {
                write!(f, "GDN run at layer {first_layer} has {layers} layers; baseline islands require 1..=3")
            }
        }
    }
}

impl std::error::Error for CompileError {}

/// Form a deterministic baseline plan. Provider boundaries stay singleton,
/// while up to three adjacent GDN blocks become one stateful island.
pub fn partition_baseline(graph: &TaskGraph) -> Result<ExecutionPlan, CompileError> {
    graph.validate().map_err(CompileError::InvalidGraph)?;

    let mut islands = Vec::new();
    let mut position = 0;
    while position < graph.tasks.len() {
        let task = &graph.tasks[position];
        let Operation::GatedDeltaBlock { layer: first_layer, .. } = task.operation else {
            islands.push(ExecutionIsland { kind: IslandKind::Provider, tasks: vec![task.id] });
            position += 1;
            continue;
        };

        let mut run = Vec::with_capacity(3);
        let mut expected_layer = first_layer;
        while position < graph.tasks.len() && run.len() < 3 {
            match graph.tasks[position].operation {
                Operation::GatedDeltaBlock { layer, .. } if layer == expected_layer => {
                    run.push(graph.tasks[position].id);
                    expected_layer += 1;
                    position += 1;
                }
                _ => break,
            }
        }
        if run.is_empty() || run.len() > 3 {
            return Err(CompileError::NonContiguousGatedDeltaRun { first_layer, layers: run.len() });
        }
        islands.push(ExecutionIsland {
            kind: IslandKind::StatefulGatedDelta { first_layer, layers: run.len() as u8 },
            tasks: run,
        });
    }

    Ok(ExecutionPlan { model: graph.model.clone(), islands })
}
