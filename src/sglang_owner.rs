use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    BindingCoordinator, BindingCoordinatorStats, BindingError, CompiledKvPlan,
    PhysicalStateBindingReceipt, PlanError, RetentionKind, StateBindingComponent,
    StateBindingIntent,
};

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum CacheKind {
    Chunk,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum OwnerCommand {
    PlanReclamation {
        request_id: String,
        observed_evicted_seqlen: u64,
        semantic_frontier: u64,
        execution_epoch: u64,
        cache_kind: CacheKind,
    },
    CommitReclamation {
        certificate_id: u64,
    },
    CommitReclamations {
        certificate_ids: Vec<u64>,
    },
    ReleaseRequest {
        request_id: String,
    },
    PrepareBinding {
        request_id: String,
        prefix_tokens: u64,
    },
    CommitBinding {
        receipt: PhysicalStateBindingReceipt,
    },
    AbortBinding {
        binding_id: u64,
    },
    Stats,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SglangSemanticProof {
    SlidingWindow {
        semantic_frontier: u64,
        window_tokens: u64,
        maximum_reclaimable_end: u64,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SglangExecutionProof {
    NonOverlapSchedulerBarrier { execution_epoch: u64 },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SglangRetirementCertificate {
    pub schema: &'static str,
    pub plan_fingerprint: String,
    pub certificate_id: u64,
    pub request_id: String,
    pub class_name: String,
    pub page_tokens: u64,
    pub token_start: u64,
    pub token_end_exclusive: u64,
    pub semantic_proof: SglangSemanticProof,
    pub execution_proof: SglangExecutionProof,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct OwnerStats {
    pub plan_fingerprint: String,
    pub tracked_requests: u64,
    pub pending_certificates: u64,
    pub committed_reclamations: u64,
    pub committed_tokens: u64,
    pub binding: BindingCoordinatorStats,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum OwnerResponse {
    Reclamation {
        certificate: Option<SglangRetirementCertificate>,
    },
    Committed {
        certificate_ids: Vec<u64>,
    },
    Released {
        request_id: String,
    },
    BindingPrepared {
        intent: StateBindingIntent,
    },
    BindingCommitted {
        binding_id: u64,
    },
    BindingAborted {
        binding_id: u64,
    },
    Stats {
        stats: OwnerStats,
    },
    Error {
        code: &'static str,
        message: String,
    },
}

#[derive(Clone, Debug, Default)]
struct RequestState {
    committed_frontier: u64,
    pending_certificate: Option<u64>,
}

#[derive(Clone, Debug)]
struct PendingState {
    request_id: String,
    token_start: u64,
    token_end_exclusive: u64,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum OwnerError {
    #[error("SGLang owner requires exactly one sliding class, found {0}")]
    SlidingClassCount(usize),
    #[error("SGLang owner does not support retention class {0:?}")]
    UnsupportedRetention(String),
    #[error("compiled sliding class {0:?} is missing its window")]
    MissingWindow(String),
    #[error(
        "request {request:?} reported evicted frontier {observed}, expected committed frontier {expected}"
    )]
    FrontierMismatch {
        request: String,
        observed: u64,
        expected: u64,
    },
    #[error("request {request:?} already has pending certificate {certificate_id}")]
    PendingCertificate {
        request: String,
        certificate_id: u64,
    },
    #[error("unknown retirement certificate {0}")]
    UnknownCertificate(u64),
    #[error("retirement certificate references unknown request {0:?}")]
    UnknownTrackedRequest(String),
    #[error("cannot release request {request:?} with pending certificate {certificate_id}")]
    ReleaseWithPending {
        request: String,
        certificate_id: u64,
    },
    #[error("certificate generation exhausted")]
    CertificateGenerationExhausted,
    #[error("integer overflow while calculating {0}")]
    ArithmeticOverflow(&'static str),
    #[error(transparent)]
    Binding(#[from] BindingError),
    #[error(transparent)]
    Plan(#[from] PlanError),
}

pub struct SglangOwner {
    fingerprint: String,
    page_tokens: u64,
    class_name: String,
    window_tokens: u64,
    full_class_name: Option<String>,
    binding: BindingCoordinator,
    requests: BTreeMap<String, RequestState>,
    pending: BTreeMap<u64, PendingState>,
    next_certificate_id: u64,
    committed_reclamations: u64,
    committed_tokens: u64,
}

impl SglangOwner {
    /// Creates the strict first `SGLang` owning adapter.
    ///
    /// The adapter currently accepts one bounded sliding class because
    /// `SGLang`'s hybrid allocator exposes one shared SWA physical pool.
    ///
    /// # Errors
    ///
    /// Returns an error if the plan cannot be lowered to that contract.
    pub fn new(plan: &CompiledKvPlan) -> Result<Self, OwnerError> {
        if let Some(class) = plan
            .classes
            .iter()
            .find(|class| class.spec.retention == RetentionKind::Chunked)
        {
            return Err(OwnerError::UnsupportedRetention(class.spec.name.clone()));
        }
        let sliding = plan
            .classes
            .iter()
            .filter(|class| class.spec.retention == RetentionKind::Sliding)
            .collect::<Vec<_>>();
        if sliding.len() != 1 {
            return Err(OwnerError::SlidingClassCount(sliding.len()));
        }
        let class = sliding[0];
        let window_tokens = class
            .spec
            .window_tokens
            .ok_or_else(|| OwnerError::MissingWindow(class.spec.name.clone()))?;
        Ok(Self {
            fingerprint: plan.fingerprint(),
            page_tokens: plan.page_tokens,
            class_name: class.spec.name.clone(),
            window_tokens,
            full_class_name: plan
                .classes
                .iter()
                .find(|class| class.spec.retention == RetentionKind::Full)
                .map(|class| class.spec.name.clone()),
            binding: BindingCoordinator::new(plan.fingerprint()),
            requests: BTreeMap::new(),
            pending: BTreeMap::new(),
            next_certificate_id: 1,
            committed_reclamations: 0,
            committed_tokens: 0,
        })
    }

    pub fn execute(&mut self, command: OwnerCommand) -> OwnerResponse {
        match self.try_execute(command) {
            Ok(response) => response,
            Err(error) => OwnerResponse::Error {
                code: error.code(),
                message: error.to_string(),
            },
        }
    }

    fn try_execute(&mut self, command: OwnerCommand) -> Result<OwnerResponse, OwnerError> {
        match command {
            OwnerCommand::PlanReclamation {
                request_id,
                observed_evicted_seqlen,
                semantic_frontier,
                execution_epoch,
                cache_kind: CacheKind::Chunk,
            } => self.plan_reclamation(
                request_id,
                observed_evicted_seqlen,
                semantic_frontier,
                execution_epoch,
            ),
            OwnerCommand::CommitReclamation { certificate_id } => {
                self.commit_reclamations(&[certificate_id])
            }
            OwnerCommand::CommitReclamations { certificate_ids } => {
                self.commit_reclamations(&certificate_ids)
            }
            OwnerCommand::ReleaseRequest { request_id } => self.release_request(request_id),
            OwnerCommand::PrepareBinding {
                request_id,
                prefix_tokens,
            } => self.prepare_binding(request_id, prefix_tokens),
            OwnerCommand::CommitBinding { receipt } => {
                let binding_id = receipt.binding_id;
                self.binding.commit(&receipt)?;
                Ok(OwnerResponse::BindingCommitted { binding_id })
            }
            OwnerCommand::AbortBinding { binding_id } => {
                self.binding.abort(binding_id)?;
                Ok(OwnerResponse::BindingAborted { binding_id })
            }
            OwnerCommand::Stats => Ok(OwnerResponse::Stats {
                stats: self.stats(),
            }),
        }
    }

    fn plan_reclamation(
        &mut self,
        request_id: String,
        observed_evicted_seqlen: u64,
        semantic_frontier: u64,
        execution_epoch: u64,
    ) -> Result<OwnerResponse, OwnerError> {
        let state = self
            .requests
            .entry(request_id.clone())
            .or_insert_with(|| RequestState {
                committed_frontier: observed_evicted_seqlen,
                pending_certificate: None,
            });
        if observed_evicted_seqlen != state.committed_frontier {
            return Err(OwnerError::FrontierMismatch {
                request: request_id,
                observed: observed_evicted_seqlen,
                expected: state.committed_frontier,
            });
        }
        if let Some(certificate_id) = state.pending_certificate {
            return Err(OwnerError::PendingCertificate {
                request: request_id,
                certificate_id,
            });
        }

        let maximum_reclaimable_end = semantic_frontier.saturating_sub(self.window_tokens);
        let target = (maximum_reclaimable_end / self.page_tokens) * self.page_tokens;
        if target <= observed_evicted_seqlen {
            return Ok(OwnerResponse::Reclamation { certificate: None });
        }

        let certificate_id = self.next_certificate_id;
        self.next_certificate_id = self
            .next_certificate_id
            .checked_add(1)
            .ok_or(OwnerError::CertificateGenerationExhausted)?;
        let certificate = SglangRetirementCertificate {
            schema: "orbitkv.sglang-retirement-certificate.v1",
            plan_fingerprint: self.fingerprint.clone(),
            certificate_id,
            request_id: request_id.clone(),
            class_name: self.class_name.clone(),
            page_tokens: self.page_tokens,
            token_start: observed_evicted_seqlen,
            token_end_exclusive: target,
            semantic_proof: SglangSemanticProof::SlidingWindow {
                semantic_frontier,
                window_tokens: self.window_tokens,
                maximum_reclaimable_end,
            },
            execution_proof: SglangExecutionProof::NonOverlapSchedulerBarrier { execution_epoch },
        };
        state.pending_certificate = Some(certificate_id);
        self.pending.insert(
            certificate_id,
            PendingState {
                request_id,
                token_start: observed_evicted_seqlen,
                token_end_exclusive: target,
            },
        );
        Ok(OwnerResponse::Reclamation {
            certificate: Some(certificate),
        })
    }

    fn commit_reclamations(
        &mut self,
        certificate_ids: &[u64],
    ) -> Result<OwnerResponse, OwnerError> {
        let unique = certificate_ids.iter().copied().collect::<BTreeSet<_>>();
        if unique.len() != certificate_ids.len() {
            let duplicate = certificate_ids
                .iter()
                .copied()
                .find(|id| certificate_ids.iter().filter(|other| *other == id).count() > 1)
                .unwrap_or(0);
            return Err(OwnerError::UnknownCertificate(duplicate));
        }

        let mut pending_states = Vec::with_capacity(certificate_ids.len());
        let mut committed_tokens = 0_u64;
        for &certificate_id in certificate_ids {
            let pending = self
                .pending
                .get(&certificate_id)
                .cloned()
                .ok_or(OwnerError::UnknownCertificate(certificate_id))?;
            if !self.requests.contains_key(&pending.request_id) {
                return Err(OwnerError::UnknownTrackedRequest(pending.request_id));
            }
            committed_tokens = committed_tokens
                .checked_add(pending.token_end_exclusive - pending.token_start)
                .ok_or(OwnerError::ArithmeticOverflow("committed token count"))?;
            pending_states.push((certificate_id, pending));
        }
        let next_reclamations = self
            .committed_reclamations
            .checked_add(certificate_ids.len() as u64)
            .ok_or(OwnerError::ArithmeticOverflow(
                "committed reclamation count",
            ))?;
        let next_tokens = self
            .committed_tokens
            .checked_add(committed_tokens)
            .ok_or(OwnerError::ArithmeticOverflow("committed token count"))?;

        for (certificate_id, pending) in pending_states {
            let request = self
                .requests
                .get_mut(&pending.request_id)
                .ok_or_else(|| OwnerError::UnknownTrackedRequest(pending.request_id.clone()))?;
            request.committed_frontier = pending.token_end_exclusive;
            request.pending_certificate = None;
            self.pending.remove(&certificate_id);
        }
        self.committed_reclamations = next_reclamations;
        self.committed_tokens = next_tokens;
        Ok(OwnerResponse::Committed {
            certificate_ids: certificate_ids.to_vec(),
        })
    }

    fn release_request(&mut self, request_id: String) -> Result<OwnerResponse, OwnerError> {
        if let Some(state) = self.requests.get(&request_id)
            && let Some(certificate_id) = state.pending_certificate
        {
            return Err(OwnerError::ReleaseWithPending {
                request: request_id,
                certificate_id,
            });
        }
        self.requests.remove(&request_id);
        Ok(OwnerResponse::Released { request_id })
    }

    fn prepare_binding(
        &mut self,
        request_id: String,
        prefix_tokens: u64,
    ) -> Result<OwnerResponse, OwnerError> {
        if prefix_tokens == 0 {
            return Err(OwnerError::ArithmeticOverflow("binding prefix tokens"));
        }
        let local_start = prefix_tokens.saturating_sub(self.window_tokens);
        let local_start = local_start / self.page_tokens * self.page_tokens;
        let mut components = Vec::new();
        if let Some(full_class_name) = &self.full_class_name {
            components.push(StateBindingComponent {
                state_class: full_class_name.clone(),
                token_start: 0,
                token_end_exclusive: prefix_tokens,
                physical_tokens: prefix_tokens,
            });
        }
        components.push(StateBindingComponent {
            state_class: self.class_name.clone(),
            token_start: local_start,
            token_end_exclusive: prefix_tokens,
            physical_tokens: prefix_tokens - local_start,
        });
        let intent = self.binding.prepare(request_id, components)?;
        Ok(OwnerResponse::BindingPrepared { intent })
    }

    #[must_use]
    pub fn stats(&self) -> OwnerStats {
        OwnerStats {
            plan_fingerprint: self.fingerprint.clone(),
            tracked_requests: self.requests.len() as u64,
            pending_certificates: self.pending.len() as u64,
            committed_reclamations: self.committed_reclamations,
            committed_tokens: self.committed_tokens,
            binding: self.binding.stats(),
        }
    }
}

impl OwnerError {
    const fn code(&self) -> &'static str {
        match self {
            Self::SlidingClassCount(_) => "sliding_class_count",
            Self::UnsupportedRetention(_) => "unsupported_retention",
            Self::MissingWindow(_) => "missing_window",
            Self::FrontierMismatch { .. } => "frontier_mismatch",
            Self::PendingCertificate { .. } => "pending_certificate",
            Self::UnknownCertificate(_) => "unknown_certificate",
            Self::UnknownTrackedRequest(_) => "unknown_tracked_request",
            Self::ReleaseWithPending { .. } => "release_with_pending",
            Self::CertificateGenerationExhausted => "certificate_generation_exhausted",
            Self::ArithmeticOverflow(_) => "arithmetic_overflow",
            Self::Binding(_) => "binding_error",
            Self::Plan(_) => "invalid_plan",
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{KvClassSpec, KvPlanInput, compile_plan};

    use super::*;

    fn binding_receipt(intent: &StateBindingIntent) -> PhysicalStateBindingReceipt {
        PhysicalStateBindingReceipt {
            schema: "orbitkv.physical-state-binding-receipt.v1".into(),
            plan_fingerprint: intent.plan_fingerprint.clone(),
            binding_id: intent.binding_id,
            backend_transaction_id: format!("test:{}", intent.binding_id),
            components: intent
                .components
                .iter()
                .map(|component| crate::PhysicalStateBindingComponentReceipt {
                    state_class: component.state_class.clone(),
                    token_start: component.token_start,
                    token_end_exclusive: component.token_end_exclusive,
                    physical_tokens: component.physical_tokens,
                    physical_binding_id: format!("{}:physical", component.state_class),
                    payload_ready: true,
                })
                .collect(),
        }
    }

    fn owner() -> SglangOwner {
        let plan = compile_plan(KvPlanInput {
            page_tokens: 16,
            classes: vec![
                KvClassSpec {
                    name: "full".into(),
                    layers: vec![0],
                    retention: RetentionKind::Full,
                    bytes_per_token_per_layer: 128,
                    window_tokens: None,
                },
                KvClassSpec {
                    name: "swa".into(),
                    layers: vec![1],
                    retention: RetentionKind::Sliding,
                    bytes_per_token_per_layer: 128,
                    window_tokens: Some(32),
                },
            ],
        })
        .unwrap();
        SglangOwner::new(&plan).unwrap()
    }

    fn plan(owner: &mut SglangOwner, observed: u64, frontier: u64) -> OwnerResponse {
        owner.execute(OwnerCommand::PlanReclamation {
            request_id: "r0".into(),
            observed_evicted_seqlen: observed,
            semantic_frontier: frontier,
            execution_epoch: frontier,
            cache_kind: CacheKind::Chunk,
        })
    }

    #[test]
    fn certificate_requires_commit_before_frontier_advances() {
        let mut owner = owner();
        let OwnerResponse::Reclamation {
            certificate: Some(certificate),
        } = plan(&mut owner, 0, 64)
        else {
            panic!("expected a certificate");
        };
        assert_eq!(certificate.token_end_exclusive, 32);
        assert!(matches!(
            plan(&mut owner, 32, 80),
            OwnerResponse::Error {
                code: "frontier_mismatch",
                ..
            }
        ));
        assert!(matches!(
            owner.execute(OwnerCommand::CommitReclamation {
                certificate_id: certificate.certificate_id,
            }),
            OwnerResponse::Committed { .. }
        ));
        assert!(matches!(
            plan(&mut owner, 32, 80),
            OwnerResponse::Reclamation {
                certificate: Some(_)
            }
        ));
    }

    #[test]
    fn reclamation_is_page_aligned_and_window_safe() {
        let mut owner = owner();
        let OwnerResponse::Reclamation {
            certificate: Some(certificate),
        } = plan(&mut owner, 0, 79)
        else {
            panic!("expected a certificate");
        };
        assert_eq!(certificate.token_start, 0);
        assert_eq!(certificate.token_end_exclusive, 32);
        assert_eq!(certificate.token_end_exclusive % 16, 0);
        assert!(certificate.token_end_exclusive <= 79 - 32);
    }

    #[test]
    fn batch_commit_preflight_is_atomic() {
        let mut owner = owner();
        let OwnerResponse::Reclamation {
            certificate: Some(certificate),
        } = plan(&mut owner, 0, 64)
        else {
            panic!("expected a certificate");
        };
        assert!(matches!(
            owner.execute(OwnerCommand::CommitReclamations {
                certificate_ids: vec![certificate.certificate_id, 999],
            }),
            OwnerResponse::Error {
                code: "unknown_certificate",
                ..
            }
        ));
        assert_eq!(owner.stats().pending_certificates, 1);
        assert_eq!(owner.stats().committed_reclamations, 0);
        assert!(matches!(
            owner.execute(OwnerCommand::CommitReclamations {
                certificate_ids: vec![certificate.certificate_id],
            }),
            OwnerResponse::Committed { .. }
        ));
        assert_eq!(owner.stats().pending_certificates, 0);
        assert_eq!(owner.stats().committed_reclamations, 1);
    }

    #[test]
    fn hybrid_binding_components_are_derived_from_retention_plan() {
        let plan = compile_plan(KvPlanInput {
            page_tokens: 16,
            classes: vec![
                KvClassSpec {
                    name: "full".into(),
                    layers: vec![0],
                    retention: RetentionKind::Full,
                    bytes_per_token_per_layer: 128,
                    window_tokens: None,
                },
                KvClassSpec {
                    name: "swa".into(),
                    layers: vec![1],
                    retention: RetentionKind::Sliding,
                    bytes_per_token_per_layer: 128,
                    window_tokens: Some(128),
                },
            ],
        })
        .unwrap();
        let mut owner = SglangOwner::new(&plan).unwrap();
        let OwnerResponse::BindingPrepared { intent } =
            owner.execute(OwnerCommand::PrepareBinding {
                request_id: "r0".into(),
                prefix_tokens: 1024,
            })
        else {
            panic!("expected binding intent");
        };
        assert_eq!(
            intent.components,
            vec![
                StateBindingComponent {
                    state_class: "full".into(),
                    token_start: 0,
                    token_end_exclusive: 1024,
                    physical_tokens: 1024,
                },
                StateBindingComponent {
                    state_class: "swa".into(),
                    token_start: 896,
                    token_end_exclusive: 1024,
                    physical_tokens: 128,
                },
            ]
        );
        let receipt = binding_receipt(&intent);
        assert_eq!(
            owner.execute(OwnerCommand::CommitBinding { receipt }),
            OwnerResponse::BindingCommitted {
                binding_id: intent.binding_id
            }
        );
        assert_eq!(owner.stats().binding.committed_bindings, 1);
    }

    #[test]
    fn failed_binding_receipt_remains_abortable() {
        let mut owner = SglangOwner::new(
            &compile_plan(KvPlanInput {
                page_tokens: 16,
                classes: vec![KvClassSpec {
                    name: "swa".into(),
                    layers: vec![0],
                    retention: RetentionKind::Sliding,
                    bytes_per_token_per_layer: 128,
                    window_tokens: Some(128),
                }],
            })
            .unwrap(),
        )
        .unwrap();
        let OwnerResponse::BindingPrepared { intent } =
            owner.execute(OwnerCommand::PrepareBinding {
                request_id: "r0".into(),
                prefix_tokens: 1024,
            })
        else {
            panic!("expected binding intent");
        };
        assert_eq!(
            intent.components,
            vec![StateBindingComponent {
                state_class: "swa".into(),
                token_start: 896,
                token_end_exclusive: 1024,
                physical_tokens: 128,
            }]
        );
        let mut receipt = binding_receipt(&intent);
        receipt.components[0].payload_ready = false;
        assert!(matches!(
            owner.execute(OwnerCommand::CommitBinding { receipt }),
            OwnerResponse::Error {
                code: "binding_error",
                ..
            }
        ));
        assert_eq!(owner.stats().binding.pending_bindings, 1);
        assert_eq!(
            owner.execute(OwnerCommand::AbortBinding {
                binding_id: intent.binding_id
            }),
            OwnerResponse::BindingAborted {
                binding_id: intent.binding_id
            }
        );
    }
}
