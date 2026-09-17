use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::{Deserialize, Serialize};

use super::{EffectKind, StateId, StateScope, StateSpec, Task, TaskId, ValueId};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TaskGraph {
    pub model: String,
    pub inputs: Vec<ValueId>,
    pub states: Vec<StateSpec>,
    pub tasks: Vec<Task>,
    pub outputs: Vec<ValueId>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GraphError(pub String);

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for GraphError {}

impl TaskGraph {
    /// Validate ordering, SSA values, persistent-state legality, and candidates.
    pub fn validate(&self) -> Result<(), GraphError> {
        let mut states = BTreeMap::new();
        for state in &self.states {
            if state.bytes == 0 {
                return Err(GraphError(format!("state `{}` has zero bytes", state.name)));
            }
            if states.insert(state.id, state).is_some() {
                return Err(GraphError(format!("duplicate state id {:?}", state.id)));
            }
        }

        let mut values: BTreeSet<ValueId> = self.inputs.iter().copied().collect();
        if values.len() != self.inputs.len() {
            return Err(GraphError("duplicate graph input".into()));
        }
        let mut tasks = BTreeSet::new();

        for (position, task) in self.tasks.iter().enumerate() {
            if task.id != TaskId(position) || !tasks.insert(task.id) {
                return Err(GraphError(format!("task {:?} is not in canonical topological order", task.id)));
            }
            if task.candidates.is_empty() {
                return Err(GraphError(format!("task {:?} has no implementation candidate", task.id)));
            }
            for dependency in &task.dependencies {
                if !tasks.contains(dependency) || *dependency == task.id {
                    return Err(GraphError(format!("task {:?} has non-prior dependency {:?}", task.id, dependency)));
                }
            }
            for input in &task.inputs {
                if !values.contains(input) {
                    return Err(GraphError(format!("task {:?} reads unavailable value {:?}", task.id, input)));
                }
            }
            for output in &task.outputs {
                if !values.insert(*output) {
                    return Err(GraphError(format!("value {:?} has multiple producers", output)));
                }
            }
            let mut tentative_writes = BTreeSet::new();
            for effect in &task.state_effects {
                let state = states.get(&effect.state).ok_or_else(|| {
                    GraphError(format!("task {:?} references unknown state {:?}", task.id, effect.state))
                })?;
                match (state.scope, effect.kind) {
                    (StateScope::Immutable, EffectKind::Read | EffectKind::Lookup)
                    | (StateScope::PerToken, EffectKind::Read | EffectKind::Append | EffectKind::Lookup)
                    | (
                        StateScope::PerSequence,
                        EffectKind::Read | EffectKind::TentativeWrite { .. } | EffectKind::Commit { .. },
                    ) => {}
                    _ => {
                        return Err(GraphError(format!(
                            "task {:?} applies {:?} to {:?} state `{}`",
                            task.id, effect.kind, state.scope, state.name
                        )));
                    }
                }
                match effect.kind {
                    EffectKind::TentativeWrite { version } => {
                        tentative_writes.insert((effect.state, effect.region, version));
                    }
                    EffectKind::Commit { version }
                        if !tentative_writes.contains(&(effect.state, effect.region, version)) =>
                    {
                        return Err(GraphError(format!(
                            "task {:?} commits state `{}` version {version} without a prior tentative write",
                            task.id, state.name
                        )));
                    }
                    _ => {}
                }
            }
        }

        for output in &self.outputs {
            if !values.contains(output) {
                return Err(GraphError(format!("graph output {:?} is unavailable", output)));
            }
        }
        Ok(())
    }

    pub fn state(&self, id: StateId) -> Option<&StateSpec> {
        self.states.iter().find(|state| state.id == id)
    }
}
