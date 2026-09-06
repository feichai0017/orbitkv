use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

use crate::kv_manager::{
    ArenaStats, BackendUnobservedReceipt, BatchCompletionReceipt, CanonicalKvManager,
    ClassLowering, CopyIntent, DetachedBinding, KvManagerError, ManagerStats, PrepareBatchItem,
    PreparedStep, ReclamationCertificate, ReleaseBatchItem, RequestLease, RequestView,
    SubmittedStep, TailAction, ViewVersion, WriteIntent,
};

mod control;
mod evidence;
mod execution_view;
mod external_tier;
mod prefix_release;
mod relocation;

pub use control::{
    EngineControlEvidence, EngineControlOutcome, EngineControlPlan, EngineMaterializationPlan,
    EngineMaterializedRequest, EnginePendingAttachCancel, EnginePendingAttachCancelDisposition,
    EnginePendingAttachCancelOutcome, EnginePrefixEvictionPlan, EnginePrefixLookup,
    EnginePublishedPrefix, EngineRetirement, EngineRetirementEvidence,
};
use control::{PendingControl, SessionPrefix};
use evidence::{
    engine_retirements, flatten_evidence, validate_abort_evidence, validate_reclamation_evidence,
};
pub use execution_view::{EnginePreparedBatchView, EnginePreparedRequestView};
pub use external_tier::{
    ExternalExportAbortEvidence, ExternalExportCopy, ExternalExportPlan, ExternalExportReceipt,
    ExternalObjectKey, ExternalReplica, ExternalReplicaDeletionEvidence, ExternalReplicaPage,
    ExternalReplicaTarget, ExternalRestoreAbortEvidence, ExternalRestoreCopy, ExternalRestorePlan,
    ExternalRestoreReceipt, ExternalRestoreTicket, ExternalTierError, ExternalTierStats,
    ExternalTransferCompletion, ExternalTransferId,
};
use external_tier::{PendingExternalExport, PendingExternalRestore};
pub use prefix_release::{EnginePrefixPublishReleasePlan, EnginePublishedPrefixRelease};
use relocation::PendingRelocation;
pub use relocation::{
    EnginePrepareRelocationItem, EnginePreparedRelocation, EngineRelocationAbortEvidence,
    EngineRelocationCopyEvidence, EngineRelocationExecutionEvidence, EngineRelocationPlan,
    EngineRelocationPublication, EngineRelocationPublicationEvidence,
    EngineRelocationRequestEvidence, EngineRelocationRequestPublication, EngineRelocationTicket,
    EngineTokenDispositionBatchItem, EngineTokenDispositionUpdate, EngineTokenView,
    EngineTokenViewQuery,
};

/// Controls whether requests may share cache state through Prefix operations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum CacheSharingPolicy {
    RequestPrivate = 1,
    SharedPrefix = 2,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[repr(transparent)]
pub struct EngineRequestId(pub u64);

macro_rules! session_id {
    ($name:ident) => {
        #[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
        pub struct $name {
            session_epoch: u64,
            sequence: u64,
        }

        impl $name {
            /// Reconstructs an opaque session identity from serialized parts.
            #[must_use]
            pub const fn from_parts(session_epoch: u64, sequence: u64) -> Self {
                Self {
                    session_epoch,
                    sequence,
                }
            }

            /// Returns the runtime-session epoch that minted this identity.
            #[must_use]
            pub const fn session_epoch(self) -> u64 {
                self.session_epoch
            }

            /// Returns the sequence number within the owning runtime session.
            #[must_use]
            pub const fn sequence(self) -> u64 {
                self.sequence
            }
        }
    };
}

session_id!(EngineBatchId);
session_id!(EnginePublicationId);
session_id!(EngineReleaseId);
session_id!(EnginePrefixId);
session_id!(EngineControlId);
session_id!(EngineRelocationId);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct EngineRequestView {
    pub request_id: EngineRequestId,
    pub view_version: ViewVersion,
    pub boundary: u64,
    pub resident_count: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct EngineAppendIntent {
    pub request_id: EngineRequestId,
    pub target_boundary: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct EngineStepPlan {
    pub request_id: EngineRequestId,
    pub base_view_version: ViewVersion,
    pub target_view_version: ViewVersion,
    pub previous_boundary: u64,
    pub target_boundary: u64,
    pub class_lowerings: Box<[ClassLowering]>,
    pub tail_actions: Box<[TailAction]>,
    pub copy_intents: Box<[CopyIntent]>,
    pub write_intents: Box<[WriteIntent]>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct EngineBatchPlan {
    pub batch_id: EngineBatchId,
    pub steps: Box<[EngineStepPlan]>,
}

/// Engine-observed backend binding facts for one manager-selected page.
///
/// The canonical manager step is deliberately absent and is injected from
/// private prepared-batch state when the evidence is submitted.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct EngineBindEvidence {
    pub page: crate::kv_manager::PageLease,
    pub backend_domain: u16,
    pub mapped: bool,
    pub writable: bool,
    pub backend_index: u64,
}

/// Engine-observed backend copy facts for one manager-selected copy intent.
///
/// The canonical manager step is deliberately absent and is injected from
/// private prepared-batch state when the evidence is submitted.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct EngineCopyEvidence {
    pub class_id: u16,
    pub backend_domain: u16,
    pub token_count: u32,
    pub source_token_offset: u32,
    pub destination_token_offset: u32,
    pub observed: bool,
    pub copied: bool,
    pub ordered_before_writes: bool,
    pub source: crate::kv_manager::PageLease,
    pub destination: crate::kv_manager::PageLease,
    pub source_backend_index: u64,
    pub destination_backend_index: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct EngineStepExecutionEvidence {
    pub request_id: EngineRequestId,
    pub bind_receipts: Box<[EngineBindEvidence]>,
    pub copy_receipts: Box<[EngineCopyEvidence]>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ExecutionEvidence {
    pub batch_id: EngineBatchId,
    pub steps: Box<[EngineStepExecutionEvidence]>,
}

/// Per-request proof used to safely abandon a prepared append.
///
/// This is currently a raw engine assertion. The caller is responsible for
/// establishing that no backend work observed the corresponding step; a
/// future authenticated evidence envelope may strengthen that trust boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct EngineStepAbortEvidence {
    pub request_id: EngineRequestId,
    pub backend_unobserved: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct EngineBatchTicket {
    batch_id: EngineBatchId,
}

impl EngineBatchTicket {
    #[must_use]
    pub const fn batch_id(&self) -> EngineBatchId {
        self.batch_id
    }
}

/// Provisional, engine-supplied completion assertion.
///
/// The session validates shape, monotonicity, and batch ownership through the
/// canonical manager, but these raw fields are not authenticated or bound to
/// a GPU fence/evidence envelope. The embedding engine must establish their
/// provenance before calling [`RuntimeSession::complete_execution_by_batch`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct EngineCompletionEvidence {
    pub completion_domain: u64,
    pub completion_value: u64,
    pub confirmed: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct EngineStepPublication {
    pub request_id: EngineRequestId,
    pub view_version: ViewVersion,
    pub boundary: u64,
    pub resident_count: u32,
    pub detached: Box<[DetachedBinding]>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct EngineBatchPublication {
    pub publication_id: EnginePublicationId,
    pub batch_id: EngineBatchId,
    pub steps: Box<[EngineStepPublication]>,
    pub retirements: Box<[EngineRetirement]>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct EnginePublicationEvidence {
    pub publication_id: EnginePublicationId,
    /// Provisional engine assertion; structural validation is not proof of
    /// GPU-side mirror cleanup.
    pub mirror_cleanup_confirmed: bool,
    pub reclamation_receipts: Box<[EngineRetirementEvidence]>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct EngineReleasedRequest {
    pub request_id: EngineRequestId,
    pub detached: Box<[DetachedBinding]>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct EngineReleasePlan {
    pub release_id: EngineReleaseId,
    pub releases: Box<[EngineReleasedRequest]>,
    pub retirements: Box<[EngineRetirement]>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct EngineReleaseEvidence {
    pub release_id: EngineReleaseId,
    /// Provisional engine assertion; structural validation is not proof of
    /// GPU-side mirror cleanup. Must be false on a post-ACK ID-only retry.
    pub mirror_cleanup_confirmed: bool,
    /// Exact receipts are consumed once. They must be empty on an ID-only retry
    /// after ACK succeeded but request recycling failed.
    pub reclamation_receipts: Box<[EngineRetirementEvidence]>,
}

/// The committed phase reached by a release confirmation attempt.
///
/// Before the commit point, validation and ACK failures are ordinary errors.
/// `RecyclePending` means the ACK was consumed and only an ID-only retry is
/// valid; malformed retries are rejected without retrying the recycle.
/// `Completed` means the request identities were also recycled. An unexpected
/// post-ACK recycle error poisons the session.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[must_use = "release confirmation can leave request recycling pending"]
pub enum EngineReleaseOutcome {
    Completed,
    RecyclePending,
}

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum RuntimeSessionError {
    #[error(transparent)]
    Manager(#[from] KvManagerError),
    #[error("engine batch must contain at least one item")]
    EmptyBatch,
    #[error("engine batch contains duplicate request id {0:?}")]
    DuplicateRequest(EngineRequestId),
    #[error("engine request id {0:?} is already acquired")]
    RequestAlreadyAcquired(EngineRequestId),
    #[error("unknown engine request id {0:?}")]
    UnknownRequest(EngineRequestId),
    #[error("engine request id {request_id:?} is not ready: {state}")]
    RequestNotReady {
        request_id: EngineRequestId,
        state: &'static str,
    },
    #[error("unknown engine batch id {0:?}")]
    UnknownBatch(EngineBatchId),
    #[error("engine batch id {0:?} belongs to a different runtime session")]
    ForeignBatch(EngineBatchId),
    #[error("stale engine batch id {0:?}")]
    StaleBatch(EngineBatchId),
    #[error("engine batch {0:?} is not prepared")]
    BatchNotPrepared(EngineBatchId),
    #[error("engine batch {0:?} is not submitted")]
    BatchNotSubmitted(EngineBatchId),
    #[error("evidence cardinality mismatch for {field}: expected {expected}, got {actual}")]
    EvidenceCardinality {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("evidence request mismatch at step {index}: expected {expected:?}, got {actual:?}")]
    EvidenceRequest {
        index: usize,
        expected: EngineRequestId,
        actual: EngineRequestId,
    },
    #[error(
        "engine token-view boundary mismatch for {request_id:?}: expected {expected}, got {actual}"
    )]
    TokenViewBoundary {
        request_id: EngineRequestId,
        expected: u64,
        actual: u64,
    },
    #[error("flattened evidence cardinality exceeds the manager batch limit")]
    EvidenceTooLarge,
    #[error("unknown publication id {0:?}")]
    UnknownPublication(EnginePublicationId),
    #[error("publication id {0:?} belongs to a different runtime session")]
    ForeignPublication(EnginePublicationId),
    #[error("stale publication id {0:?}")]
    StalePublication(EnginePublicationId),
    #[error("unknown release id {0:?}")]
    UnknownRelease(EngineReleaseId),
    #[error("release id {0:?} belongs to a different runtime session")]
    ForeignRelease(EngineReleaseId),
    #[error("stale release id {0:?}")]
    StaleRelease(EngineReleaseId),
    #[error("unknown engine prefix id {0:?}")]
    UnknownPrefix(EnginePrefixId),
    #[error("engine prefix id {0:?} belongs to a different runtime session")]
    ForeignPrefix(EnginePrefixId),
    #[error("stale engine prefix id {0:?}")]
    StalePrefix(EnginePrefixId),
    #[error("engine prefix id {prefix_id:?} is not ready: {state}")]
    PrefixNotReady {
        prefix_id: EnginePrefixId,
        state: &'static str,
    },
    #[error("engine prefix batch contains duplicate prefix id {0:?}")]
    DuplicatePrefix(EnginePrefixId),
    #[error("unknown engine control id {0:?}")]
    UnknownControl(EngineControlId),
    #[error("engine control id {0:?} belongs to a different runtime session")]
    ForeignControl(EngineControlId),
    #[error("stale engine control id {0:?}")]
    StaleControl(EngineControlId),
    #[error("unknown engine relocation id {0:?}")]
    UnknownRelocation(EngineRelocationId),
    #[error("engine relocation id {0:?} belongs to a different runtime session")]
    ForeignRelocation(EngineRelocationId),
    #[error("stale engine relocation id {0:?}")]
    StaleRelocation(EngineRelocationId),
    #[error("engine relocation {0:?} is not prepared")]
    RelocationNotPrepared(EngineRelocationId),
    #[error("engine relocation {0:?} is not submitted")]
    RelocationNotSubmitted(EngineRelocationId),
    #[error("engine relocation {0:?} has no publication pending")]
    RelocationPublicationNotPending(EngineRelocationId),
    #[error("engine control {0:?} has already committed")]
    ControlAlreadyCommitted(EngineControlId),
    #[error("engine control {0:?} has not committed")]
    ControlNotCommitted(EngineControlId),
    #[error("engine control {0:?} cannot cancel a committed request")]
    ControlNotCancelable(EngineControlId),
    #[error("engine control {0:?} has no canceled request pending finalization")]
    CanceledRequestNotPending(EngineControlId),
    #[error("engine control {0:?} pending attach cancel expectation differs")]
    PendingAttachCancelMismatch(EngineControlId),
    #[error("mirror updates are not confirmed")]
    MirrorUpdatesNotConfirmed,
    #[error("mirror cleanup is not confirmed")]
    MirrorCleanupNotConfirmed,
    #[error("reclamation receipts do not exactly match the pending certificates")]
    ReclamationReceiptMismatch,
    #[error("release recycle retry must contain only the release id")]
    ReleaseRetryNotIdOnly,
    #[error("cache-sharing policy does not support Prefix/share operations")]
    PrefixOperationsUnsupported,
    #[error("{0} identity space is exhausted")]
    IdentityExhausted(&'static str),
    #[error("runtime session is poisoned: {0}")]
    SessionPoisoned(&'static str),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RequestPhase {
    Ready,
    Prepared(EngineBatchId),
    Submitted(EngineBatchId),
    PublicationPending(EnginePublicationId),
    ReleasePending(EngineReleaseId),
    ControlSource(EngineControlId),
    ControlTarget(EngineControlId),
    RelocationPrepared(EngineRelocationId),
    RelocationSubmitted(EngineRelocationId),
    RelocationPublicationPending(EngineRelocationId),
    ExternalExportPending(ExternalTransferId),
    Quarantined,
}

impl RequestPhase {
    const fn name(self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::Prepared(_) => "append prepared",
            Self::Submitted(_) => "execution submitted",
            Self::PublicationPending(_) => "publication confirmation pending",
            Self::ReleasePending(_) => "release confirmation pending",
            Self::ControlSource(_) => "control source reserved",
            Self::ControlTarget(_) => "control materialization pending",
            Self::RelocationPrepared(_) => "relocation prepared",
            Self::RelocationSubmitted(_) => "relocation submitted",
            Self::RelocationPublicationPending(_) => "relocation publication pending",
            Self::ExternalExportPending(_) => "external export pending",
            Self::Quarantined => "quarantined",
        }
    }
}

#[derive(Clone, Debug)]
struct SessionRequest {
    view: RequestView,
    phase: RequestPhase,
}

#[derive(Clone, Debug)]
struct PreparedBatch {
    requests: Box<[EngineRequestId]>,
    steps: Box<[PreparedStep]>,
}

#[derive(Clone, Debug)]
struct SubmittedBatch {
    requests: Box<[EngineRequestId]>,
    steps: Box<[SubmittedStep]>,
}

#[derive(Clone, Debug)]
enum PendingBatch {
    Prepared(PreparedBatch),
    Submitted(SubmittedBatch),
}

#[derive(Clone, Debug)]
struct PendingPublication {
    requests: Box<[EngineRequestId]>,
    retirements: Box<[ReclamationCertificate]>,
}

#[derive(Clone, Debug)]
struct PendingRelease {
    requests: Box<[EngineRequestId]>,
    leases: Box<[RequestLease]>,
    phase: PendingReleasePhase,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PendingCanceledRequest {
    outcome: EnginePendingAttachCancelOutcome,
    lease: RequestLease,
    finalized: bool,
}

#[derive(Clone, Debug)]
enum PendingReleasePhase {
    AwaitingAcknowledgement {
        retirements: Box<[ReclamationCertificate]>,
    },
    RecyclePending,
}

#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimeSessionTestFault {
    CompletionCardinality,
    AttachSecondOutput,
    ForkSecondOutput,
    PrefixPublishReleaseOutput,
    ControlCommitManagerOnce,
    PrefixRecycleFatalOnce,
    ReleaseRecycleOnce,
    ReleaseRecycleFatalOnce,
    TokenViewOrdering,
}

#[derive(Debug)]
pub struct RuntimeSession {
    manager: CanonicalKvManager,
    cache_sharing_policy: CacheSharingPolicy,
    session_epoch: u64,
    next_batch_sequence: u64,
    next_publication_sequence: u64,
    next_release_sequence: u64,
    next_prefix_sequence: u64,
    next_control_sequence: u64,
    next_relocation_sequence: u64,
    next_external_sequence: u64,
    poisoned: Option<&'static str>,
    #[cfg(any(test, feature = "test-support"))]
    test_fault: Option<RuntimeSessionTestFault>,
    requests: BTreeMap<EngineRequestId, SessionRequest>,
    batches: BTreeMap<EngineBatchId, PendingBatch>,
    publications: BTreeMap<EnginePublicationId, PendingPublication>,
    releases: BTreeMap<EngineReleaseId, PendingRelease>,
    canceled_requests: BTreeMap<EngineControlId, PendingCanceledRequest>,
    prefixes: BTreeMap<EnginePrefixId, SessionPrefix>,
    prefix_leases: BTreeMap<crate::kv_manager::PrefixLease, EnginePrefixId>,
    prefix_index: BTreeMap<crate::kv_manager::PrefixSemanticKey, EnginePrefixId>,
    controls: BTreeMap<EngineControlId, PendingControl>,
    relocations: BTreeMap<EngineRelocationId, PendingRelocation>,
    external_exports: BTreeMap<ExternalTransferId, PendingExternalExport>,
    external_restores: BTreeMap<ExternalTransferId, PendingExternalRestore>,
    external_replicas: BTreeMap<ExternalObjectKey, ExternalReplica>,
    maximum_controls: usize,
    maximum_external_operations: usize,
    maximum_external_replicas: usize,
}

impl RuntimeSession {
    /// Creates a session that exclusively owns the canonical manager.
    ///
    /// # Panics
    ///
    /// Panics only if a constructed canonical manager reports no runtime
    /// classes, which violates its constructor contract.
    #[must_use]
    pub fn new(manager: CanonicalKvManager, cache_sharing_policy: CacheSharingPolicy) -> Self {
        let maximum_controls = manager.operation_capacity();
        let session_epoch = manager
            .arena_stats()
            .first()
            .expect("canonical manager has at least one runtime class")
            .engine_epoch;
        Self {
            manager,
            cache_sharing_policy,
            session_epoch,
            next_batch_sequence: 1,
            next_publication_sequence: 1,
            next_release_sequence: 1,
            next_prefix_sequence: 1,
            next_control_sequence: 1,
            next_relocation_sequence: 1,
            next_external_sequence: 1,
            poisoned: None,
            #[cfg(any(test, feature = "test-support"))]
            test_fault: None,
            requests: BTreeMap::new(),
            batches: BTreeMap::new(),
            publications: BTreeMap::new(),
            releases: BTreeMap::new(),
            canceled_requests: BTreeMap::new(),
            prefixes: BTreeMap::new(),
            prefix_leases: BTreeMap::new(),
            prefix_index: BTreeMap::new(),
            controls: BTreeMap::new(),
            relocations: BTreeMap::new(),
            external_exports: BTreeMap::new(),
            external_restores: BTreeMap::new(),
            external_replicas: BTreeMap::new(),
            maximum_controls,
            maximum_external_operations: maximum_controls,
            maximum_external_replicas: maximum_controls,
        }
    }

    fn ensure_prefix_operations_supported(&self) -> Result<(), RuntimeSessionError> {
        if self.cache_sharing_policy == CacheSharingPolicy::RequestPrivate {
            return Err(RuntimeSessionError::PrefixOperationsUnsupported);
        }
        Ok(())
    }

    /// Acquires an ordered batch of fresh engine request ids atomically.
    ///
    /// # Errors
    ///
    /// Rejects empty, duplicate, already-acquired, or manager-capacity failures
    /// without installing a partial engine-to-manager mapping.
    pub fn acquire_requests(
        &mut self,
        request_ids: &[EngineRequestId],
    ) -> Result<Box<[EngineRequestView]>, RuntimeSessionError> {
        self.ensure_healthy()?;
        self.preflight_new_request_ids(request_ids)?;
        let views = self.manager.acquire_requests_batch(request_ids.len())?;
        if views.len() != request_ids.len() {
            return Err(self.poison("acquire result cardinality"));
        }
        Ok(request_ids
            .iter()
            .copied()
            .zip(views.iter().copied())
            .map(|(request_id, view)| {
                let old = self.requests.insert(
                    request_id,
                    SessionRequest {
                        view,
                        phase: RequestPhase::Ready,
                    },
                );
                debug_assert!(old.is_none());
                engine_view(request_id, view)
            })
            .collect::<Vec<_>>()
            .into_boxed_slice())
    }

    /// Prepares one ordered append batch using session-owned snapshot heads.
    ///
    /// # Errors
    ///
    /// Rejects empty, duplicate, unknown, or non-ready requests and propagates
    /// canonical prepare failures without changing the request mapping.
    ///
    /// # Panics
    ///
    /// Panics only if the private request map changes after preflight, which
    /// indicates an internal session invariant violation.
    pub fn prepare_append_batch(
        &mut self,
        intents: &[EngineAppendIntent],
    ) -> Result<EngineBatchPlan, RuntimeSessionError> {
        self.ensure_healthy()?;
        let records = self.preflight_append_intents(intents)?;
        let batch_id = self.allocate_batch_id()?;
        let items = intents
            .iter()
            .zip(&records)
            .map(|(intent, record)| PrepareBatchItem {
                request: record.view.request,
                expected_head: record.view.snapshot,
                target_boundary: intent.target_boundary,
            })
            .collect::<Vec<_>>();
        let prepared = self.manager.prepare_batch(&items)?;
        if prepared.len() != intents.len() {
            return Err(self.poison("prepare result cardinality"));
        }
        if prepared.iter().zip(&records).any(|(step, record)| {
            step.request != record.view.request
                || step.base_snapshot != record.view.snapshot
                || step.base_view_version != record.view.view_version
                || step.previous_boundary != record.view.boundary
        }) {
            return Err(self.poison("prepare result request ordering"));
        }
        let request_ids = intents
            .iter()
            .map(|intent| intent.request_id)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        for request_id in &request_ids {
            self.requests
                .get_mut(request_id)
                .expect("append preflight retained request")
                .phase = RequestPhase::Prepared(batch_id);
        }
        let plan = EngineBatchPlan {
            batch_id,
            steps: request_ids
                .iter()
                .copied()
                .zip(prepared.iter().cloned())
                .map(|(request_id, prepared)| engine_step_plan(request_id, &prepared))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        };
        self.batches.insert(
            batch_id,
            PendingBatch::Prepared(PreparedBatch {
                requests: request_ids,
                steps: prepared,
            }),
        );
        Ok(plan)
    }

    /// Safely abandons a prepared batch using exact ordered per-step proof.
    ///
    /// Each evidence row is bound to the stored request order; canonical step
    /// leases are derived privately. A false proof or malformed batch is
    /// rejected atomically and remains retryable.
    ///
    /// # Errors
    ///
    /// Rejects foreign, stale, submitted, reordered, incomplete, or unproven
    /// batches without changing manager or session state.
    pub fn abort_prepared_execution(
        &mut self,
        batch_id: EngineBatchId,
        evidence: &[EngineStepAbortEvidence],
    ) -> Result<(), RuntimeSessionError> {
        self.ensure_healthy()?;
        let batch = self.prepared_batch(batch_id)?.clone();
        validate_abort_evidence(&batch, evidence)?;
        self.preflight_request_phases(&batch.requests, RequestPhase::Prepared(batch_id))?;
        let receipts = batch
            .steps
            .iter()
            .zip(evidence)
            .map(|(prepared, evidence)| BackendUnobservedReceipt {
                step: prepared.step,
                backend_unobserved: u32::from(evidence.backend_unobserved),
                reserved: 0,
            })
            .collect::<Vec<_>>();
        self.manager.abort_steps_batch(&receipts)?;
        self.finish_batch(batch_id, &batch.requests, RequestPhase::Ready);
        Ok(())
    }

    /// Fail-stops a prepared batch whose backend observation is ambiguous.
    ///
    /// # Errors
    ///
    /// Rejects foreign, stale, or non-prepared batches atomically.
    pub fn quarantine_prepared_execution(
        &mut self,
        batch_id: EngineBatchId,
    ) -> Result<(), RuntimeSessionError> {
        self.ensure_healthy()?;
        let batch = self.prepared_batch(batch_id)?.clone();
        self.preflight_request_phases(&batch.requests, RequestPhase::Prepared(batch_id))?;
        let steps = batch
            .steps
            .iter()
            .map(|prepared| prepared.step)
            .collect::<Vec<_>>();
        self.manager.quarantine_steps_batch(&steps)?;
        self.finish_batch(batch_id, &batch.requests, RequestPhase::Quarantined);
        Ok(())
    }

    /// Validates and flattens grouped per-step backend evidence, then submits.
    ///
    /// # Errors
    ///
    /// Rejects unknown, stale, already-submitted, reordered, or malformed
    /// evidence. Canonical semantic receipt mismatch remains fail-stop and
    /// quarantines all requests in the batch.
    ///
    /// # Panics
    ///
    /// Panics only if the private request map loses a prepared request, which
    /// indicates an internal session invariant violation.
    pub fn submit_execution(
        &mut self,
        evidence: &ExecutionEvidence,
    ) -> Result<EngineBatchTicket, RuntimeSessionError> {
        self.ensure_healthy()?;
        let batch = self.prepared_batch(evidence.batch_id)?.clone();
        self.preflight_request_phases(&batch.requests, RequestPhase::Prepared(evidence.batch_id))?;
        let (items, binds, copies) = flatten_evidence(&batch, evidence)?;
        let submitted = match self.manager.submit_batch(&items, &binds, &copies) {
            Ok(submitted) => submitted,
            Err(error @ KvManagerError::BatchQuarantined(_)) => {
                self.finish_batch(
                    evidence.batch_id,
                    &batch.requests,
                    RequestPhase::Quarantined,
                );
                return Err(error.into());
            }
            Err(error) => return Err(error.into()),
        };
        if submitted.len() != batch.requests.len() {
            return Err(self.poison("submit result cardinality"));
        }
        if submitted
            .iter()
            .zip(&batch.steps)
            .any(|(submitted, prepared)| {
                submitted.request != prepared.request
                    || submitted.target_snapshot != prepared.target_snapshot
            })
        {
            return Err(self.poison("submit result request ordering"));
        }
        for request_id in &batch.requests {
            self.requests
                .get_mut(request_id)
                .expect("prepared batch retained request")
                .phase = RequestPhase::Submitted(evidence.batch_id);
        }
        self.batches.insert(
            evidence.batch_id,
            PendingBatch::Submitted(SubmittedBatch {
                requests: batch.requests.clone(),
                steps: submitted,
            }),
        );
        Ok(EngineBatchTicket {
            batch_id: evidence.batch_id,
        })
    }

    /// Fail-stops a submitted batch whose execution outcome is ambiguous.
    ///
    /// # Errors
    ///
    /// Rejects foreign, stale, or non-submitted batches atomically.
    pub fn quarantine_submitted_execution(
        &mut self,
        batch_id: EngineBatchId,
    ) -> Result<(), RuntimeSessionError> {
        self.ensure_healthy()?;
        let batch = self.submitted_batch(batch_id)?.clone();
        self.preflight_request_phases(&batch.requests, RequestPhase::Submitted(batch_id))?;
        let submissions = batch
            .steps
            .iter()
            .map(|submitted| submitted.submission)
            .collect::<Vec<_>>();
        self.manager.quarantine_submissions_batch(&submissions)?;
        self.finish_batch(batch_id, &batch.requests, RequestPhase::Quarantined);
        Ok(())
    }

    /// Publishes one submitted batch by its session-owned batch identity.
    ///
    /// Requests remain gated until the returned publication is confirmed,
    /// including when it contains no reclamation certificates.
    ///
    /// # Errors
    ///
    /// Rejects unknown, stale, unsubmitted, unconfirmed, or non-advancing
    /// evidence and propagates canonical completion failures atomically.
    ///
    /// # Panics
    ///
    /// Panics only if the private request map loses a submitted request, which
    /// indicates an internal session invariant violation.
    pub fn complete_execution_by_batch(
        &mut self,
        batch_id: EngineBatchId,
        evidence: EngineCompletionEvidence,
    ) -> Result<EngineBatchPublication, RuntimeSessionError> {
        self.complete_execution_impl(batch_id, evidence)
    }

    fn complete_execution_impl(
        &mut self,
        batch_id: EngineBatchId,
        evidence: EngineCompletionEvidence,
    ) -> Result<EngineBatchPublication, RuntimeSessionError> {
        self.ensure_healthy()?;
        let batch = self.submitted_batch(batch_id)?.clone();
        self.preflight_request_phases(&batch.requests, RequestPhase::Submitted(batch_id))?;
        let publication_id = self.allocate_publication_id()?;
        let submissions = batch
            .steps
            .iter()
            .map(|step| step.submission)
            .collect::<Vec<_>>();
        let Some(engine_epoch) = submissions
            .first()
            .map(|submission| submission.engine_epoch)
        else {
            return Err(self.poison("empty submitted batch"));
        };
        let completed = self.manager.complete_batch(
            BatchCompletionReceipt {
                engine_epoch,
                completion_domain: evidence.completion_domain,
                completion_value: evidence.completion_value,
                confirmed: u32::from(evidence.confirmed),
                reserved: 0,
            },
            &submissions,
        )?;
        #[cfg(any(test, feature = "test-support"))]
        let completed = self.apply_completion_test_fault(completed);
        if completed.completions.len() != batch.requests.len() {
            return Err(self.poison("completion result cardinality"));
        }
        if completed
            .completions
            .iter()
            .zip(&batch.steps)
            .any(|(completion, submitted)| {
                completion.submission != submitted.submission
                    || completion.request != submitted.request
                    || completion.publication.snapshot != submitted.target_snapshot
            })
        {
            return Err(self.poison("completion result request ordering"));
        }
        let publications = batch
            .requests
            .iter()
            .copied()
            .zip(completed.completions.iter())
            .map(|(request_id, completion)| EngineStepPublication {
                request_id,
                view_version: completion.publication.view_version,
                boundary: completion.publication.boundary,
                resident_count: completion.publication.resident_count,
                detached: completion.detached.clone(),
            })
            .collect::<Vec<_>>();
        for (request_id, completion) in batch
            .requests
            .iter()
            .copied()
            .zip(completed.completions.iter())
        {
            let record = self
                .requests
                .get_mut(&request_id)
                .expect("submitted batch retained request");
            record.view = RequestView {
                request: completion.request,
                snapshot: completion.publication.snapshot,
                view_version: completion.publication.view_version,
                boundary: completion.publication.boundary,
                resident_count: completion.publication.resident_count,
            };
            record.phase = RequestPhase::PublicationPending(publication_id);
        }
        self.batches.remove(&batch_id);
        let engine_retirements = engine_retirements(&completed.retirements);
        self.publications.insert(
            publication_id,
            PendingPublication {
                requests: batch.requests,
                retirements: completed.retirements,
            },
        );
        Ok(EngineBatchPublication {
            publication_id,
            batch_id,
            steps: publications.into_boxed_slice(),
            retirements: engine_retirements,
        })
    }

    /// Closes publication mirror effects and ACKs its exact retirements.
    ///
    /// # Errors
    ///
    /// Rejects unknown or stale publication ids, missing mirror confirmation,
    /// or receipts that are not the exact ordered image of the certificates.
    ///
    /// # Panics
    ///
    /// Panics only if the private request map changes after preflight, which
    /// indicates an internal session invariant violation.
    pub fn confirm_publication(
        &mut self,
        evidence: &EnginePublicationEvidence,
    ) -> Result<(), RuntimeSessionError> {
        self.ensure_healthy()?;
        self.ensure_publication_epoch(evidence.publication_id)?;
        let pending = self
            .publications
            .get(&evidence.publication_id)
            .cloned()
            .ok_or_else(|| self.publication_id_error(evidence.publication_id))?;
        let receipts = validate_reclamation_evidence(
            evidence.mirror_cleanup_confirmed,
            &pending.retirements,
            &evidence.reclamation_receipts,
        )?;
        self.preflight_request_phases(
            &pending.requests,
            RequestPhase::PublicationPending(evidence.publication_id),
        )?;
        if !pending.retirements.is_empty() {
            self.manager.acknowledge_reclamations_batch(&receipts)?;
        }
        for request_id in &pending.requests {
            self.requests
                .get_mut(request_id)
                .expect("publication preflight retained request")
                .phase = RequestPhase::Ready;
        }
        self.publications.remove(&evidence.publication_id);
        Ok(())
    }

    /// Semantically releases an ordered request batch and returns mirror work.
    ///
    /// # Errors
    ///
    /// Rejects empty, duplicate, unknown, or non-ready requests. In particular,
    /// it cannot absorb an unconfirmed append publication.
    ///
    /// # Panics
    ///
    /// Panics only if the private request map changes after preflight, which
    /// indicates an internal session invariant violation.
    pub fn prepare_release_batch(
        &mut self,
        request_ids: &[EngineRequestId],
    ) -> Result<EngineReleasePlan, RuntimeSessionError> {
        self.ensure_healthy()?;
        let records = self.preflight_ready_requests(request_ids)?;
        let release_id = self.allocate_release_id()?;
        let items = records
            .iter()
            .map(|record| ReleaseBatchItem {
                request: record.view.request,
                expected_head: record.view.snapshot,
            })
            .collect::<Vec<_>>();
        let manager_output = self.manager.release_batch(&items)?;
        if manager_output.releases.len() != request_ids.len() {
            return Err(self.poison("release result cardinality"));
        }
        if manager_output
            .releases
            .iter()
            .zip(&records)
            .any(|(release, record)| {
                release.request != record.view.request
                    || release.detached_snapshot != record.view.snapshot
            })
        {
            return Err(self.poison("release result request ordering"));
        }
        let leases = records
            .iter()
            .map(|record| record.view.request)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let engine_releases = request_ids
            .iter()
            .copied()
            .zip(manager_output.releases.iter())
            .map(|(request_id, release)| EngineReleasedRequest {
                request_id,
                detached: release.detached.clone(),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let pending_requests = request_ids.to_vec().into_boxed_slice();
        for request_id in &pending_requests {
            self.requests
                .get_mut(request_id)
                .expect("release preflight retained request")
                .phase = RequestPhase::ReleasePending(release_id);
        }
        let engine_retirements = engine_retirements(&manager_output.retirements);
        self.releases.insert(
            release_id,
            PendingRelease {
                requests: pending_requests,
                leases,
                phase: PendingReleasePhase::AwaitingAcknowledgement {
                    retirements: manager_output.retirements,
                },
            },
        );
        Ok(EngineReleasePlan {
            release_id,
            releases: engine_releases,
            retirements: engine_retirements,
        })
    }

    /// Confirms mirror cleanup, ACKs exact retirements, and recycles requests.
    ///
    /// Once ACK succeeds, `RequestNotRecyclable` returns
    /// [`EngineReleaseOutcome::RecyclePending`], and the caller must retry using
    /// only the release id (false mirror confirmation and no reclamation
    /// receipts). Any other post-ACK recycle error poisons the session.
    ///
    /// # Errors
    ///
    /// Rejects unknown or stale release ids, missing mirror confirmation, or
    /// receipts that are not the exact ordered image of the certificates.
    /// Canonical ACK failures are preserved. Post-ACK `RequestNotRecyclable` is
    /// represented by [`EngineReleaseOutcome::RecyclePending`]; any other
    /// recycle failure poisons the session.
    ///
    /// # Panics
    ///
    /// Panics only if private pending-release state changes after preflight,
    /// which indicates an internal session invariant violation.
    pub fn confirm_release(
        &mut self,
        evidence: &EngineReleaseEvidence,
    ) -> Result<EngineReleaseOutcome, RuntimeSessionError> {
        self.ensure_healthy()?;
        self.ensure_release_epoch(evidence.release_id)?;
        let pending = self
            .releases
            .get(&evidence.release_id)
            .cloned()
            .ok_or_else(|| self.release_id_error(evidence.release_id))?;
        let receipts = match &pending.phase {
            PendingReleasePhase::AwaitingAcknowledgement { retirements } => {
                validate_reclamation_evidence(
                    evidence.mirror_cleanup_confirmed,
                    retirements,
                    &evidence.reclamation_receipts,
                )?
            }
            PendingReleasePhase::RecyclePending => {
                if evidence.mirror_cleanup_confirmed || !evidence.reclamation_receipts.is_empty() {
                    return Err(RuntimeSessionError::ReleaseRetryNotIdOnly);
                }
                Box::new([])
            }
        };
        self.preflight_request_phases(
            &pending.requests,
            RequestPhase::ReleasePending(evidence.release_id),
        )?;
        if let PendingReleasePhase::AwaitingAcknowledgement { retirements } = &pending.phase {
            if !retirements.is_empty() {
                self.manager.acknowledge_reclamations_batch(&receipts)?;
            }
            self.releases
                .get_mut(&evidence.release_id)
                .expect("release remains pending after ACK")
                .phase = PendingReleasePhase::RecyclePending;
        }
        #[cfg(any(test, feature = "test-support"))]
        if self.test_fault == Some(RuntimeSessionTestFault::ReleaseRecycleOnce) {
            self.test_fault = None;
            return Ok(EngineReleaseOutcome::RecyclePending);
        }
        #[cfg(any(test, feature = "test-support"))]
        if self.test_fault == Some(RuntimeSessionTestFault::ReleaseRecycleFatalOnce) {
            self.test_fault = None;
            return Err(self.poison("unexpected release recycle failure after acknowledgement"));
        }
        match self.manager.recycle_requests_batch(&pending.leases) {
            Ok(()) => {}
            Err(KvManagerError::RequestNotRecyclable) => {
                return Ok(EngineReleaseOutcome::RecyclePending);
            }
            Err(_) => {
                return Err(self.poison("unexpected release recycle failure after acknowledgement"));
            }
        }
        for request_id in &pending.requests {
            self.requests
                .remove(request_id)
                .expect("release preflight retained request");
        }
        self.releases.remove(&evidence.release_id);
        Ok(EngineReleaseOutcome::Completed)
    }

    #[must_use]
    pub fn stats(&self) -> ManagerStats {
        self.manager.stats()
    }

    #[must_use]
    pub fn arena_stats(&self) -> Box<[ArenaStats]> {
        self.manager.arena_stats()
    }

    #[cfg(any(test, feature = "test-support"))]
    #[doc(hidden)]
    pub fn inject_test_fault(&mut self, fault: RuntimeSessionTestFault) {
        self.test_fault = Some(fault);
    }

    #[cfg(any(test, feature = "test-support"))]
    fn apply_completion_test_fault(
        &mut self,
        mut completed: crate::kv_manager::CompletionBatch,
    ) -> crate::kv_manager::CompletionBatch {
        if self.test_fault == Some(RuntimeSessionTestFault::CompletionCardinality) {
            self.test_fault = None;
            completed.completions = Vec::new().into_boxed_slice();
        }
        completed
    }

    fn ensure_healthy(&self) -> Result<(), RuntimeSessionError> {
        match self.poisoned {
            Some(reason) => Err(RuntimeSessionError::SessionPoisoned(reason)),
            None => Ok(()),
        }
    }

    fn poison(&mut self, reason: &'static str) -> RuntimeSessionError {
        let reason = *self.poisoned.get_or_insert(reason);
        RuntimeSessionError::SessionPoisoned(reason)
    }

    fn allocate_batch_id(&mut self) -> Result<EngineBatchId, RuntimeSessionError> {
        let sequence = allocate_sequence(&mut self.next_batch_sequence, "batch")?;
        Ok(EngineBatchId {
            session_epoch: self.session_epoch,
            sequence,
        })
    }

    fn allocate_publication_id(&mut self) -> Result<EnginePublicationId, RuntimeSessionError> {
        let sequence = allocate_sequence(&mut self.next_publication_sequence, "publication")?;
        Ok(EnginePublicationId {
            session_epoch: self.session_epoch,
            sequence,
        })
    }

    fn allocate_release_id(&mut self) -> Result<EngineReleaseId, RuntimeSessionError> {
        let sequence = allocate_sequence(&mut self.next_release_sequence, "release")?;
        Ok(EngineReleaseId {
            session_epoch: self.session_epoch,
            sequence,
        })
    }

    fn allocate_control_id(&mut self) -> Result<EngineControlId, RuntimeSessionError> {
        let sequence = self
            .next_control_sequence
            .checked_add(1)
            .ok_or(RuntimeSessionError::IdentityExhausted("control"))?;
        while self
            .controls
            .len()
            .checked_add(self.canceled_requests.len())
            .is_some_and(|count| count >= self.maximum_controls)
        {
            let Some(oldest) = self
                .canceled_requests
                .iter()
                .find_map(|(control_id, pending)| pending.finalized.then_some(*control_id))
            else {
                break;
            };
            self.canceled_requests.remove(&oldest);
        }
        if self
            .controls
            .len()
            .checked_add(self.canceled_requests.len())
            .is_none_or(|count| count >= self.maximum_controls)
        {
            return Err(KvManagerError::ArenaExhausted("control").into());
        }
        let current = self.next_control_sequence;
        self.next_control_sequence = sequence;
        Ok(EngineControlId {
            session_epoch: self.session_epoch,
            sequence: current,
        })
    }

    fn finish_batch(
        &mut self,
        batch_id: EngineBatchId,
        request_ids: &[EngineRequestId],
        phase: RequestPhase,
    ) {
        for request_id in request_ids {
            self.requests
                .get_mut(request_id)
                .expect("batch preflight retained request")
                .phase = phase;
        }
        self.batches
            .remove(&batch_id)
            .expect("batch preflight retained operation");
    }

    fn ensure_batch_epoch(&self, batch_id: EngineBatchId) -> Result<(), RuntimeSessionError> {
        if batch_id.session_epoch != self.session_epoch {
            return Err(RuntimeSessionError::ForeignBatch(batch_id));
        }
        Ok(())
    }

    fn ensure_publication_epoch(
        &self,
        publication_id: EnginePublicationId,
    ) -> Result<(), RuntimeSessionError> {
        if publication_id.session_epoch != self.session_epoch {
            return Err(RuntimeSessionError::ForeignPublication(publication_id));
        }
        Ok(())
    }

    fn ensure_release_epoch(&self, release_id: EngineReleaseId) -> Result<(), RuntimeSessionError> {
        if release_id.session_epoch != self.session_epoch {
            return Err(RuntimeSessionError::ForeignRelease(release_id));
        }
        Ok(())
    }

    fn preflight_new_request_ids(
        &self,
        request_ids: &[EngineRequestId],
    ) -> Result<(), RuntimeSessionError> {
        if request_ids.is_empty() {
            return Err(RuntimeSessionError::EmptyBatch);
        }
        let mut seen = BTreeSet::new();
        for &request_id in request_ids {
            if !seen.insert(request_id) {
                return Err(RuntimeSessionError::DuplicateRequest(request_id));
            }
            if self.requests.contains_key(&request_id) {
                return Err(RuntimeSessionError::RequestAlreadyAcquired(request_id));
            }
        }
        Ok(())
    }

    fn preflight_append_intents(
        &self,
        intents: &[EngineAppendIntent],
    ) -> Result<Vec<SessionRequest>, RuntimeSessionError> {
        if intents.is_empty() {
            return Err(RuntimeSessionError::EmptyBatch);
        }
        let mut seen = BTreeSet::new();
        intents
            .iter()
            .map(|intent| {
                if !seen.insert(intent.request_id) {
                    return Err(RuntimeSessionError::DuplicateRequest(intent.request_id));
                }
                self.ready_request(intent.request_id).cloned()
            })
            .collect()
    }

    fn preflight_ready_requests(
        &self,
        request_ids: &[EngineRequestId],
    ) -> Result<Vec<SessionRequest>, RuntimeSessionError> {
        if request_ids.is_empty() {
            return Err(RuntimeSessionError::EmptyBatch);
        }
        let mut seen = BTreeSet::new();
        request_ids
            .iter()
            .copied()
            .map(|request_id| {
                if !seen.insert(request_id) {
                    return Err(RuntimeSessionError::DuplicateRequest(request_id));
                }
                self.ready_request(request_id).cloned()
            })
            .collect()
    }

    fn ready_request(
        &self,
        request_id: EngineRequestId,
    ) -> Result<&SessionRequest, RuntimeSessionError> {
        let record = self
            .requests
            .get(&request_id)
            .ok_or(RuntimeSessionError::UnknownRequest(request_id))?;
        if record.phase != RequestPhase::Ready {
            return Err(RuntimeSessionError::RequestNotReady {
                request_id,
                state: record.phase.name(),
            });
        }
        Ok(record)
    }

    fn prepared_batch(
        &self,
        batch_id: EngineBatchId,
    ) -> Result<&PreparedBatch, RuntimeSessionError> {
        self.ensure_batch_epoch(batch_id)?;
        match self.batches.get(&batch_id) {
            Some(PendingBatch::Prepared(batch)) => Ok(batch),
            Some(PendingBatch::Submitted(_)) => {
                Err(RuntimeSessionError::BatchNotPrepared(batch_id))
            }
            None => Err(self.batch_id_error(batch_id)),
        }
    }

    fn submitted_batch(
        &self,
        batch_id: EngineBatchId,
    ) -> Result<&SubmittedBatch, RuntimeSessionError> {
        self.ensure_batch_epoch(batch_id)?;
        match self.batches.get(&batch_id) {
            Some(PendingBatch::Submitted(batch)) => Ok(batch),
            Some(PendingBatch::Prepared(_)) => {
                Err(RuntimeSessionError::BatchNotSubmitted(batch_id))
            }
            None => Err(self.batch_id_error(batch_id)),
        }
    }

    fn preflight_request_phases(
        &mut self,
        request_ids: &[EngineRequestId],
        expected: RequestPhase,
    ) -> Result<(), RuntimeSessionError> {
        for &request_id in request_ids {
            let Some(record) = self.requests.get(&request_id) else {
                return Err(self.poison("pending operation lost request"));
            };
            if record.phase != expected {
                return Err(self.poison("pending request phase changed"));
            }
        }
        Ok(())
    }

    fn batch_id_error(&self, batch_id: EngineBatchId) -> RuntimeSessionError {
        if batch_id.session_epoch != self.session_epoch {
            return RuntimeSessionError::ForeignBatch(batch_id);
        }
        if was_issued(batch_id.sequence, self.next_batch_sequence) {
            RuntimeSessionError::StaleBatch(batch_id)
        } else {
            RuntimeSessionError::UnknownBatch(batch_id)
        }
    }

    fn publication_id_error(&self, publication_id: EnginePublicationId) -> RuntimeSessionError {
        if publication_id.session_epoch != self.session_epoch {
            return RuntimeSessionError::ForeignPublication(publication_id);
        }
        if was_issued(publication_id.sequence, self.next_publication_sequence) {
            RuntimeSessionError::StalePublication(publication_id)
        } else {
            RuntimeSessionError::UnknownPublication(publication_id)
        }
    }

    fn release_id_error(&self, release_id: EngineReleaseId) -> RuntimeSessionError {
        if release_id.session_epoch != self.session_epoch {
            return RuntimeSessionError::ForeignRelease(release_id);
        }
        if was_issued(release_id.sequence, self.next_release_sequence) {
            RuntimeSessionError::StaleRelease(release_id)
        } else {
            RuntimeSessionError::UnknownRelease(release_id)
        }
    }
}

fn engine_view(request_id: EngineRequestId, view: RequestView) -> EngineRequestView {
    EngineRequestView {
        request_id,
        view_version: view.view_version,
        boundary: view.boundary,
        resident_count: view.resident_count,
    }
}

fn engine_step_plan(request_id: EngineRequestId, prepared: &PreparedStep) -> EngineStepPlan {
    EngineStepPlan {
        request_id,
        base_view_version: prepared.base_view_version,
        target_view_version: prepared.target_view_version,
        previous_boundary: prepared.previous_boundary,
        target_boundary: prepared.target_boundary,
        class_lowerings: prepared.class_lowerings.clone(),
        tail_actions: prepared.tail_actions.clone(),
        copy_intents: prepared.copy_intents.clone(),
        write_intents: prepared.write_intents.clone(),
    }
}

fn allocate_sequence(next: &mut u64, kind: &'static str) -> Result<u64, RuntimeSessionError> {
    let current = *next;
    *next = current
        .checked_add(1)
        .ok_or(RuntimeSessionError::IdentityExhausted(kind))?;
    Ok(current)
}

const fn was_issued(sequence: u64, next_sequence: u64) -> bool {
    sequence != 0 && sequence < next_sequence
}

#[cfg(test)]
#[path = "runtime_session/tests/mod.rs"]
mod tests;
