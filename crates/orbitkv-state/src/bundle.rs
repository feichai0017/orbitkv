use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::{StateComponent, StateDescriptor};

/// One component participating in an atomic recovery boundary.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BundleComponent {
    pub descriptor: StateDescriptor,
    pub available: bool,
}

/// Components required to claim that a logical boundary is restorable.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecoveryContract {
    pub required: BTreeSet<StateComponent>,
}

impl RecoveryContract {
    pub fn all(required: impl IntoIterator<Item = StateComponent>) -> Self {
        Self {
            required: required.into_iter().collect(),
        }
    }
}

/// Atomic set of state needed to resume execution at `boundary`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateBundle {
    pub boundary: u64,
    pub components: Vec<BundleComponent>,
    pub recovery: RecoveryContract,
}

impl StateBundle {
    /// Checks component presence only. A planner must also verify token coverage,
    /// model/format compatibility, and framework-specific recovery semantics.
    pub fn has_required_components(&self) -> bool {
        let available = self
            .components
            .iter()
            .filter(|component| component.available)
            .map(|component| &component.descriptor.component)
            .collect::<BTreeSet<_>>();
        self.recovery
            .required
            .iter()
            .all(|required| available.contains(required))
    }
}

#[cfg(test)]
#[path = "../tests/unit/bundle.rs"]
mod tests;
