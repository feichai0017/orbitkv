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
mod tests {
    use super::*;
    use crate::{StateDType, StateFormat, StateLayout, TokenRange};

    fn descriptor(component: StateComponent) -> StateDescriptor {
        StateDescriptor {
            content: [1; 32],
            span: TokenRange::new(0, 16).unwrap(),
            component,
            format: StateFormat {
                model: [2; 32],
                implementation: [3; 32],
                dtype: StateDType::Bf16,
                layout: StateLayout::MhaPageFirst,
                block_tokens: 16,
                tensor_parallel_size: 1,
                pipeline_parallel_size: 1,
                context_parallel_size: 1,
            },
        }
    }

    #[test]
    fn bundle_tracks_presence_of_every_declared_component() {
        let attention = StateComponent::AttentionKv;
        let recurrent = StateComponent::RecurrentCheckpoint;
        let mut bundle = StateBundle {
            boundary: 16,
            components: vec![
                BundleComponent {
                    descriptor: descriptor(attention.clone()),
                    available: true,
                },
                BundleComponent {
                    descriptor: descriptor(recurrent.clone()),
                    available: false,
                },
            ],
            recovery: RecoveryContract::all([attention, recurrent]),
        };

        assert!(!bundle.has_required_components());
        bundle.components[1].available = true;
        assert!(bundle.has_required_components());
    }
}
