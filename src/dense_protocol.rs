use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    DenseBindingIntent, DenseKvRuntime, DensePhysicalBindingReceipt,
    DensePhysicalReclamationReceipt, DenseRetirementCertificate, DenseRuntimeError,
    DenseRuntimeStats, DenseView, DenseViewBlock, RequestLease, RuntimeStatePlan,
    RuntimeStatePlanError,
};

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum DenseRuntimeCommand {
    AcquireRequest,
    PrepareBinding {
        request: RequestLease,
        boundary: u64,
    },
    PrepareHydration {
        request: RequestLease,
        boundary: u64,
    },
    CommitBinding {
        receipt: DensePhysicalBindingReceipt,
    },
    AbortBinding {
        binding_id: u64,
    },
    AdvanceSemanticFrontier {
        request: RequestLease,
        boundary: u64,
    },
    AdvanceResidentFrontier {
        request: RequestLease,
        boundary: u64,
    },
    SubmitView {
        request: RequestLease,
    },
    CompleteSubmission {
        submission_id: u64,
    },
    ReleaseRequest {
        request: RequestLease,
    },
    CommitReclamation {
        receipt: DensePhysicalReclamationReceipt,
    },
    RecycleRequest {
        request: RequestLease,
    },
    Stats,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum DenseRuntimeResponse {
    RequestAcquired {
        request: RequestLease,
    },
    BindingPrepared {
        intent: DenseBindingIntent,
    },
    BindingCommitted {
        binding_id: u64,
        blocks: Vec<DenseViewBlock>,
    },
    BindingAborted {
        binding_id: u64,
    },
    SemanticFrontierAdvanced {
        request: RequestLease,
        boundary: u64,
        certificates: Vec<DenseRetirementCertificate>,
    },
    ViewSubmitted {
        view: DenseView,
    },
    SubmissionCompleted {
        submission_id: u64,
        certificates: Vec<DenseRetirementCertificate>,
    },
    RequestReleased {
        request: RequestLease,
        certificates: Vec<DenseRetirementCertificate>,
    },
    ReclamationCommitted {
        certificate_id: u64,
    },
    RequestRecycled {
        request: RequestLease,
    },
    Stats {
        stats: DenseRuntimeStats,
    },
    Error {
        code: &'static str,
        message: String,
    },
}

#[derive(Debug, Error)]
pub enum DenseRuntimeServiceError {
    #[error("runtime StatePlan does not contain a dense runtime artifact")]
    MissingDenseRuntime,
    #[error(transparent)]
    StatePlan(#[from] RuntimeStatePlanError),
    #[error(transparent)]
    Runtime(#[from] DenseRuntimeError),
}

#[derive(Clone, Debug)]
pub struct DenseRuntimeService {
    runtime: DenseKvRuntime,
}

impl DenseRuntimeService {
    /// Creates one service from the sole validated runtime artifact.
    ///
    /// # Errors
    ///
    /// Returns an error when the `StatePlan` or dense artifact is invalid.
    pub fn from_state_plan(
        state_plan: &RuntimeStatePlan,
    ) -> Result<Self, DenseRuntimeServiceError> {
        state_plan.validate()?;
        let artifact = state_plan
            .dense_runtime
            .clone()
            .ok_or(DenseRuntimeServiceError::MissingDenseRuntime)?;
        Ok(Self {
            runtime: DenseKvRuntime::new(artifact)?,
        })
    }

    pub fn execute(&mut self, command: DenseRuntimeCommand) -> DenseRuntimeResponse {
        match self.try_execute(command) {
            Ok(response) => response,
            Err(error) => DenseRuntimeResponse::Error {
                code: error.code(),
                message: error.to_string(),
            },
        }
    }

    fn try_execute(
        &mut self,
        command: DenseRuntimeCommand,
    ) -> Result<DenseRuntimeResponse, DenseRuntimeServiceError> {
        match command {
            DenseRuntimeCommand::AcquireRequest => {
                let request = self.runtime.acquire_request()?;
                Ok(DenseRuntimeResponse::RequestAcquired { request })
            }
            DenseRuntimeCommand::PrepareBinding { request, boundary } => {
                let intent = self.runtime.prepare_binding_to(request, boundary)?;
                Ok(DenseRuntimeResponse::BindingPrepared { intent })
            }
            DenseRuntimeCommand::PrepareHydration { request, boundary } => {
                let intent = self.runtime.prepare_hydration_to(request, boundary)?;
                Ok(DenseRuntimeResponse::BindingPrepared { intent })
            }
            DenseRuntimeCommand::CommitBinding { receipt } => {
                let binding_id = receipt.binding_id;
                let blocks = self.runtime.commit_binding(&receipt)?;
                Ok(DenseRuntimeResponse::BindingCommitted { binding_id, blocks })
            }
            DenseRuntimeCommand::AbortBinding { binding_id } => {
                self.runtime.abort_binding(binding_id)?;
                Ok(DenseRuntimeResponse::BindingAborted { binding_id })
            }
            DenseRuntimeCommand::AdvanceSemanticFrontier { request, boundary } => {
                let certificates = self.runtime.advance_semantic_frontier(request, boundary)?;
                Ok(DenseRuntimeResponse::SemanticFrontierAdvanced {
                    request,
                    boundary,
                    certificates,
                })
            }
            DenseRuntimeCommand::AdvanceResidentFrontier { request, boundary } => {
                let certificates = self.runtime.advance_resident_frontier(request, boundary)?;
                Ok(DenseRuntimeResponse::SemanticFrontierAdvanced {
                    request,
                    boundary,
                    certificates,
                })
            }
            DenseRuntimeCommand::SubmitView { request } => {
                let view = self.runtime.submit_view(request)?;
                Ok(DenseRuntimeResponse::ViewSubmitted { view })
            }
            DenseRuntimeCommand::CompleteSubmission { submission_id } => {
                let certificates = self.runtime.complete_submission(submission_id)?;
                Ok(DenseRuntimeResponse::SubmissionCompleted {
                    submission_id,
                    certificates,
                })
            }
            DenseRuntimeCommand::ReleaseRequest { request } => {
                let certificates = self.runtime.release_request(request)?;
                Ok(DenseRuntimeResponse::RequestReleased {
                    request,
                    certificates,
                })
            }
            DenseRuntimeCommand::CommitReclamation { receipt } => {
                let certificate_id = receipt.certificate_id;
                self.runtime.commit_reclamation(&receipt)?;
                Ok(DenseRuntimeResponse::ReclamationCommitted { certificate_id })
            }
            DenseRuntimeCommand::RecycleRequest { request } => {
                self.runtime.recycle_request(request)?;
                Ok(DenseRuntimeResponse::RequestRecycled { request })
            }
            DenseRuntimeCommand::Stats => Ok(DenseRuntimeResponse::Stats {
                stats: self.runtime.stats(),
            }),
        }
    }
}

impl DenseRuntimeServiceError {
    const fn code(&self) -> &'static str {
        match self {
            Self::MissingDenseRuntime => "missing_dense_runtime",
            Self::StatePlan(_) => "invalid_runtime_state_plan",
            Self::Runtime(_) => "dense_runtime_error",
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        DenseBackendHandle, DensePhysicalBindingBlockReceipt, DensePhysicalBindingReceipt,
        DenseRuntimeArtifact, KvClassSpec, KvPlanInput, KvPlanSource, RetentionKind,
        RuntimeCapsuleContract, RuntimeExecutionContract, RuntimeExecutionMode,
        RuntimeOwnerTransport, RuntimeStatePlanOptions, compile_runtime_state_plan,
    };

    use super::*;

    fn state_plan() -> RuntimeStatePlan {
        let source = KvPlanSource::Legacy(KvPlanInput {
            page_tokens: 4,
            classes: vec![
                KvClassSpec {
                    name: "full".into(),
                    layers: vec![1],
                    retention: RetentionKind::Full,
                    bytes_per_token_per_layer: 128,
                    window_tokens: None,
                },
                KvClassSpec {
                    name: "swa".into(),
                    layers: vec![0],
                    retention: RetentionKind::Sliding,
                    bytes_per_token_per_layer: 128,
                    window_tokens: Some(9),
                },
            ],
        });
        let plan = source.clone().compile().unwrap();
        compile_runtime_state_plan(
            source,
            RuntimeStatePlanOptions {
                eviction_interval_tokens: 4,
                physical_plan: None,
                uniform_state_plan: None,
                dense_runtime: Some(DenseRuntimeArtifact::compile(&plan, 1, 4, 16).unwrap()),
                execution: RuntimeExecutionContract {
                    mode: RuntimeExecutionMode::Owner,
                    owner_transport: Some(RuntimeOwnerTransport::Sidecar),
                    uniform_state_plan_mode: None,
                    frontier: None,
                },
                capsule: RuntimeCapsuleContract {
                    enabled: false,
                    chunk_tokens: 4,
                    maximum_payload_bytes: 1 << 20,
                },
                prefix: None,
            },
        )
        .unwrap()
    }

    #[test]
    fn service_executes_transactional_binding_and_reclamation() {
        let mut service = DenseRuntimeService::from_state_plan(&state_plan()).unwrap();
        let DenseRuntimeResponse::RequestAcquired { request } =
            service.execute(DenseRuntimeCommand::AcquireRequest)
        else {
            panic!("request acquisition failed");
        };
        let DenseRuntimeResponse::BindingPrepared { intent } =
            service.execute(DenseRuntimeCommand::PrepareBinding {
                request,
                boundary: 4,
            })
        else {
            panic!("binding preparation failed");
        };
        let receipt = DensePhysicalBindingReceipt {
            schema: "orbitkv.dense-physical-binding-receipt.v1".into(),
            artifact_fingerprint: state_plan().dense_runtime.unwrap().artifact_fingerprint,
            binding_id: intent.binding_id,
            backend_transaction_id: "sglang:test".into(),
            blocks: intent
                .pending_blocks
                .iter()
                .map(|block| DensePhysicalBindingBlockReceipt {
                    logical: block.logical,
                    physical: block.physical,
                    backend: DenseBackendHandle {
                        domain: block.logical.class_id,
                        index: block.physical.slot,
                    },
                    payload_ready: true,
                })
                .collect(),
        };
        assert!(matches!(
            service.execute(DenseRuntimeCommand::CommitBinding { receipt }),
            DenseRuntimeResponse::BindingCommitted { .. }
        ));
        let DenseRuntimeResponse::RequestReleased { certificates, .. } =
            service.execute(DenseRuntimeCommand::ReleaseRequest { request })
        else {
            panic!("request release failed");
        };
        for certificate in certificates {
            assert!(matches!(
                service.execute(DenseRuntimeCommand::CommitReclamation {
                    receipt: DensePhysicalReclamationReceipt {
                        schema: "orbitkv.dense-physical-reclamation-receipt.v1".into(),
                        artifact_fingerprint: certificate.artifact_fingerprint.clone(),
                        certificate_id: certificate.certificate_id,
                        physical: certificate.physical,
                        backend: certificate.backend,
                    },
                }),
                DenseRuntimeResponse::ReclamationCommitted { .. }
            ));
        }
        assert!(matches!(
            service.execute(DenseRuntimeCommand::RecycleRequest { request }),
            DenseRuntimeResponse::RequestRecycled { .. }
        ));
    }
}
