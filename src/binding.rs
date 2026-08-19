use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct StateBindingComponent {
    pub state_class: String,
    pub token_start: u64,
    pub token_end_exclusive: u64,
    pub physical_tokens: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct StateBindingIntent {
    pub schema: &'static str,
    pub plan_fingerprint: String,
    pub binding_id: u64,
    pub request_id: String,
    pub components: Vec<StateBindingComponent>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PhysicalStateBindingComponentReceipt {
    pub state_class: String,
    pub token_start: u64,
    pub token_end_exclusive: u64,
    pub physical_tokens: u64,
    pub physical_binding_id: String,
    pub payload_ready: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PhysicalStateBindingReceipt {
    pub schema: String,
    pub plan_fingerprint: String,
    pub binding_id: u64,
    pub backend_transaction_id: String,
    pub components: Vec<PhysicalStateBindingComponentReceipt>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct BindingCoordinatorStats {
    pub pending_bindings: u64,
    pub committed_bindings: u64,
    pub aborted_bindings: u64,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum BindingError {
    #[error("binding request id must not be empty")]
    EmptyRequestId,
    #[error("binding components must not be empty")]
    EmptyComponents,
    #[error("binding component {0:?} is invalid")]
    InvalidComponent(String),
    #[error("binding component {0:?} is duplicated")]
    DuplicateComponent(String),
    #[error("request {request:?} already has pending binding {binding_id}")]
    PendingRequestBinding { request: String, binding_id: u64 },
    #[error("binding generation exhausted")]
    GenerationExhausted,
    #[error("unknown binding transaction {0}")]
    UnknownBinding(u64),
    #[error("physical binding receipt does not match transaction {0}")]
    MismatchedReceipt(u64),
    #[error("physical binding receipt {0} does not prove payload readiness")]
    PayloadNotReady(u64),
    #[error("physical binding receipt contains an empty backend identity")]
    EmptyBackendIdentity,
}

#[derive(Clone, Debug)]
pub struct BindingCoordinator {
    plan_fingerprint: String,
    next_binding_id: u64,
    pending: BTreeMap<u64, StateBindingIntent>,
    pending_requests: BTreeMap<String, u64>,
    committed_bindings: u64,
    aborted_bindings: u64,
}

impl BindingCoordinator {
    #[must_use]
    pub fn new(plan_fingerprint: impl Into<String>) -> Self {
        Self {
            plan_fingerprint: plan_fingerprint.into(),
            next_binding_id: 1,
            pending: BTreeMap::new(),
            pending_requests: BTreeMap::new(),
            committed_bindings: 0,
            aborted_bindings: 0,
        }
    }

    /// Creates an invisible binding intent after validating all component
    /// ranges and physical token counts.
    ///
    /// # Errors
    ///
    /// Returns an error without creating an intent if the request or component
    /// set is invalid.
    pub fn prepare(
        &mut self,
        request_id: impl Into<String>,
        mut components: Vec<StateBindingComponent>,
    ) -> Result<StateBindingIntent, BindingError> {
        let request_id = request_id.into();
        if request_id.is_empty() {
            return Err(BindingError::EmptyRequestId);
        }
        if components.is_empty() {
            return Err(BindingError::EmptyComponents);
        }
        if let Some(binding_id) = self.pending_requests.get(&request_id) {
            return Err(BindingError::PendingRequestBinding {
                request: request_id,
                binding_id: *binding_id,
            });
        }
        components.sort();
        let mut names = BTreeSet::new();
        for component in &components {
            if component.state_class.is_empty()
                || component.token_start >= component.token_end_exclusive
                || component.physical_tokens
                    != component.token_end_exclusive - component.token_start
            {
                return Err(BindingError::InvalidComponent(
                    component.state_class.clone(),
                ));
            }
            if !names.insert(component.state_class.clone()) {
                return Err(BindingError::DuplicateComponent(
                    component.state_class.clone(),
                ));
            }
        }
        let binding_id = self.next_binding_id;
        self.next_binding_id = self
            .next_binding_id
            .checked_add(1)
            .ok_or(BindingError::GenerationExhausted)?;
        let intent = StateBindingIntent {
            schema: "orbitkv.state-binding-intent.v1",
            plan_fingerprint: self.plan_fingerprint.clone(),
            binding_id,
            request_id: request_id.clone(),
            components,
        };
        self.pending_requests.insert(request_id, binding_id);
        self.pending.insert(binding_id, intent.clone());
        Ok(intent)
    }

    /// Commits one physical receipt after validating the complete component
    /// batch. No partial component is committed.
    ///
    /// # Errors
    ///
    /// Returns an error while leaving the intent pending for retry or abort.
    pub fn commit(&mut self, receipt: &PhysicalStateBindingReceipt) -> Result<(), BindingError> {
        self.validate_commit(receipt)?;
        let binding_id = receipt.binding_id;
        let intent = self
            .pending
            .get(&binding_id)
            .ok_or(BindingError::UnknownBinding(binding_id))?;
        let request_id = intent.request_id.clone();
        self.pending.remove(&binding_id);
        self.pending_requests.remove(&request_id);
        self.committed_bindings = self.committed_bindings.saturating_add(1);
        Ok(())
    }

    /// Validates a physical receipt without changing coordinator state.
    ///
    /// This preflight lets a higher-level owner atomically validate both the
    /// binding transaction and its logical Prefix object before committing
    /// either metadata domain.
    ///
    /// # Errors
    ///
    /// Returns the same validation errors as [`Self::commit`].
    pub fn validate_commit(
        &self,
        receipt: &PhysicalStateBindingReceipt,
    ) -> Result<(), BindingError> {
        let binding_id = receipt.binding_id;
        let intent = self
            .pending
            .get(&binding_id)
            .ok_or(BindingError::UnknownBinding(binding_id))?;
        if receipt.schema != "orbitkv.physical-state-binding-receipt.v1"
            || receipt.plan_fingerprint != self.plan_fingerprint
            || receipt.backend_transaction_id.is_empty()
        {
            return if receipt.backend_transaction_id.is_empty() {
                Err(BindingError::EmptyBackendIdentity)
            } else {
                Err(BindingError::MismatchedReceipt(binding_id))
            };
        }
        let mut components = BTreeMap::new();
        for component in &receipt.components {
            if !component.payload_ready {
                return Err(BindingError::PayloadNotReady(binding_id));
            }
            if component.physical_binding_id.is_empty() {
                return Err(BindingError::EmptyBackendIdentity);
            }
            if components
                .insert(component.state_class.clone(), component)
                .is_some()
            {
                return Err(BindingError::MismatchedReceipt(binding_id));
            }
        }
        if components.len() != intent.components.len() {
            return Err(BindingError::MismatchedReceipt(binding_id));
        }
        for expected in &intent.components {
            let Some(actual) = components.get(&expected.state_class) else {
                return Err(BindingError::MismatchedReceipt(binding_id));
            };
            if actual.token_start != expected.token_start
                || actual.token_end_exclusive != expected.token_end_exclusive
                || actual.physical_tokens != expected.physical_tokens
            {
                return Err(BindingError::MismatchedReceipt(binding_id));
            }
        }
        Ok(())
    }

    /// Aborts an invisible intent.
    ///
    /// # Errors
    ///
    /// Returns an error when the transaction does not exist.
    pub fn abort(&mut self, binding_id: u64) -> Result<(), BindingError> {
        let intent = self
            .pending
            .remove(&binding_id)
            .ok_or(BindingError::UnknownBinding(binding_id))?;
        self.pending_requests.remove(&intent.request_id);
        self.aborted_bindings = self.aborted_bindings.saturating_add(1);
        Ok(())
    }

    #[must_use]
    pub fn stats(&self) -> BindingCoordinatorStats {
        BindingCoordinatorStats {
            pending_bindings: self.pending.len() as u64,
            committed_bindings: self.committed_bindings,
            aborted_bindings: self.aborted_bindings,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn components() -> Vec<StateBindingComponent> {
        vec![
            StateBindingComponent {
                state_class: "full-kv".into(),
                token_start: 0,
                token_end_exclusive: 1024,
                physical_tokens: 1024,
            },
            StateBindingComponent {
                state_class: "swa-kv".into(),
                token_start: 896,
                token_end_exclusive: 1024,
                physical_tokens: 128,
            },
        ]
    }

    fn receipt(intent: &StateBindingIntent) -> PhysicalStateBindingReceipt {
        PhysicalStateBindingReceipt {
            schema: "orbitkv.physical-state-binding-receipt.v1".into(),
            plan_fingerprint: intent.plan_fingerprint.clone(),
            binding_id: intent.binding_id,
            backend_transaction_id: "backend-1".into(),
            components: intent
                .components
                .iter()
                .map(|component| PhysicalStateBindingComponentReceipt {
                    state_class: component.state_class.clone(),
                    token_start: component.token_start,
                    token_end_exclusive: component.token_end_exclusive,
                    physical_tokens: component.physical_tokens,
                    physical_binding_id: format!("{}-binding", component.state_class),
                    payload_ready: true,
                })
                .collect(),
        }
    }

    #[test]
    fn coordinator_commits_complete_ready_receipt() {
        let mut coordinator = BindingCoordinator::new("sha256:plan");
        let intent = coordinator.prepare("r0", components()).unwrap();
        coordinator.commit(&receipt(&intent)).unwrap();
        assert_eq!(
            coordinator.stats(),
            BindingCoordinatorStats {
                pending_bindings: 0,
                committed_bindings: 1,
                aborted_bindings: 0,
            }
        );
    }

    #[test]
    fn coordinator_keeps_partial_or_unready_receipt_pending() {
        let mut coordinator = BindingCoordinator::new("sha256:plan");
        let intent = coordinator.prepare("r0", components()).unwrap();
        let mut partial = receipt(&intent);
        partial.components.pop();
        assert_eq!(
            coordinator.commit(&partial),
            Err(BindingError::MismatchedReceipt(intent.binding_id))
        );
        let mut unready = receipt(&intent);
        unready.components[0].payload_ready = false;
        assert_eq!(
            coordinator.commit(&unready),
            Err(BindingError::PayloadNotReady(intent.binding_id))
        );
        assert_eq!(coordinator.stats().pending_bindings, 1);
        coordinator.abort(intent.binding_id).unwrap();
        assert_eq!(coordinator.stats().aborted_bindings, 1);
    }
}
