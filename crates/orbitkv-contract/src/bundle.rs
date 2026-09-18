use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::{StateComponent, StateKey};

/// One component participating in an atomic recovery boundary.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BundleComponent {
    pub key: StateKey,
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
    pub fn is_restorable(&self) -> bool {
        let available = self
            .components
            .iter()
            .filter(|component| component.available)
            .map(|component| &component.key.component)
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

    fn key(component: StateComponent) -> StateKey {
        StateKey {
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
    fn bundle_requires_every_declared_component() {
        let attention = StateComponent::AttentionKv;
        let recurrent = StateComponent::RecurrentCheckpoint;
        let mut bundle = StateBundle {
            boundary: 16,
            components: vec![
                BundleComponent {
                    key: key(attention.clone()),
                    available: true,
                },
                BundleComponent {
                    key: key(recurrent.clone()),
                    available: false,
                },
            ],
            recovery: RecoveryContract::all([attention, recurrent]),
        };

        assert!(!bundle.is_restorable());
        bundle.components[1].available = true;
        assert!(bundle.is_restorable());
    }
}
