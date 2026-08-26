#![forbid(unsafe_code)]

pub mod attention_state;
pub mod hf_config;
pub mod kv_manager;
pub mod plan;
pub mod retention;
pub mod runtime_manifest;
pub mod state_checkpoint;
pub use attention_state::{
    AttentionStateBackend, AttentionStateError, AttentionStatePlanInput, AttentionStateSpec,
    AttentionStateStorage, CompiledAttentionState, CompiledAttentionStatePlan, RecurrentFamily,
    StateComponentGeometry, compile_attention_state_manager_plan, compile_attention_state_plan,
};
pub use hf_config::{
    HfConfigError, HfLayerInference, HfManagerPlanError, HfRetentionCompilation,
    HfRetentionOptions, HfStatePlanError, compile_hf_attention_state_input,
    compile_hf_attention_state_plan, compile_hf_config, compile_hf_manager_plan,
    compile_hf_token_manager_plan,
};
pub use plan::{
    CompiledKvClass, CompiledKvPlan, KvClassSpec, KvPlanInput, PlanError, TokenComponentSpec,
    TokenStorageKind, compile_plan,
};
pub use runtime_manifest::{
    HfRuntimeManifestError, RUNTIME_MANIFEST_MAX_BYTES, RUNTIME_MANIFEST_SCHEMA,
    RUNTIME_MANIFEST_VERSION, RuntimeCapability, RuntimeManifest, RuntimeManifestError,
    RuntimeTokenManagerPlan, compile_hf_runtime_manifest, compile_runtime_manifest,
    compile_runtime_manifest_from_plan,
};
pub use state_checkpoint::{
    StateCheckpointError, StateCheckpointPool, StateCompletionReceipt, StateCopyIntent,
    StateCopyReceipt, StatePoolIdentity, StatePoolStats, StatePublication,
    StateRetirementCertificate, StateRetirementLease, StateSlotLease, StateTransitionLease,
};
