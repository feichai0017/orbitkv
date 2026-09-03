#![forbid(unsafe_code)]

pub mod attention_state;
pub mod hf_config;
pub mod kv_manager;
pub mod plan;
pub mod retention;
pub mod runtime_manifest;
pub mod runtime_session;
mod runtime_target;
pub mod state_checkpoint;
pub use attention_state::{
    AttentionStateBackend, AttentionStateError, AttentionStatePlanInput, AttentionStateSpec,
    AttentionStateStorage, CompiledAttentionState, CompiledAttentionStatePlan, RecurrentFamily,
    StateComponentGeometry, compile_attention_state_manager_plan, compile_attention_state_plan,
};
pub use hf_config::{
    HfConfigError, HfLayerInference, HfManagerPlanError, HfRetentionCompilation,
    HfRetentionOptions, HfStatePlanError, compile_hf_attention_state_input,
    compile_hf_attention_state_plan, compile_hf_config, compile_hf_token_manager_plan,
};
pub use plan::{
    CompiledKvClass, CompiledKvPlan, KvClassSpec, KvPlanInput, PlanError, TokenComponentSpec,
    TokenStorageKind, compile_plan, compile_retention_program,
};
pub use retention::{IntExpr, KvHeadRange, Predicate, RetentionProgramInput, RetentionStateDecl};
pub use runtime_manifest::{
    HfRuntimeManifestError, RUNTIME_MANIFEST_MAX_BYTES, RUNTIME_MANIFEST_SCHEMA,
    RUNTIME_MANIFEST_VERSION, RuntimeCapability, RuntimeManifest, RuntimeManifestError,
    RuntimeManifestSource, RuntimeTokenManagerPlan, compile_hf_runtime_manifest,
    compile_retention_runtime_manifest, compile_runtime_manifest,
    compile_runtime_manifest_from_plan,
};
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub use runtime_session::RuntimeSessionTestFault;
pub use runtime_session::{
    CacheSharingPolicy, EngineAppendIntent, EngineBatchId, EngineBatchPlan, EngineBatchPublication,
    EngineBatchTicket, EngineBindEvidence, EngineCompletionEvidence, EngineControlEvidence,
    EngineControlId, EngineControlOutcome, EngineControlPlan, EngineCopyEvidence,
    EngineMaterializationPlan, EngineMaterializedRequest, EnginePendingAttachCancel,
    EnginePendingAttachCancelDisposition, EnginePendingAttachCancelOutcome,
    EnginePrefixEvictionPlan, EnginePrefixId, EnginePrefixLookup, EnginePrefixPublishReleasePlan,
    EnginePrepareRelocationItem, EnginePreparedRelocation, EnginePublicationEvidence,
    EnginePublicationId, EnginePublishedPrefix, EnginePublishedPrefixRelease,
    EngineReleaseEvidence, EngineReleaseId, EngineReleaseOutcome, EngineReleasePlan,
    EngineReleasedRequest, EngineRelocationAbortEvidence, EngineRelocationCopyEvidence,
    EngineRelocationExecutionEvidence, EngineRelocationId, EngineRelocationPlan,
    EngineRelocationPublication, EngineRelocationPublicationEvidence,
    EngineRelocationRequestEvidence, EngineRelocationRequestPublication, EngineRelocationTicket,
    EngineRequestId, EngineRequestView, EngineRetirement, EngineRetirementEvidence,
    EngineStepAbortEvidence, EngineStepExecutionEvidence, EngineStepPlan, EngineStepPublication,
    EngineTokenDispositionBatchItem, EngineTokenDispositionUpdate, EngineTokenView,
    EngineTokenViewQuery, ExecutionEvidence, RuntimeSession, RuntimeSessionError,
};
pub use runtime_target::{
    EXECUTION_SIGNATURE_SCHEMA, EXECUTION_SIGNATURE_VERSION, ExecutionAddressProgram,
    ExecutionBlockDomain, ExecutionFixedBackend, ExecutionFixedState, ExecutionKvHeadRange,
    ExecutionRetirementProgram, ExecutionSignature, ExecutionStateComponent, ExecutionTokenBackend,
    ExecutionTokenClass, ExecutionTokenState, ExecutionTopology, RUNTIME_BINDING_SCHEMA,
    RUNTIME_BINDING_VERSION, RuntimeAdmissionProfile, RuntimeBinding, RuntimeTargetIdentity,
    TargetAdmissionError, admit_runtime_manifest, derive_execution_signature,
};
pub use state_checkpoint::{
    StateCheckpointError, StateCheckpointPool, StateCompletionReceipt, StateCopyIntent,
    StateCopyReceipt, StatePoolIdentity, StatePoolStats, StatePublication,
    StateRetirementCertificate, StateRetirementLease, StateSlotLease, StateTransitionLease,
};
