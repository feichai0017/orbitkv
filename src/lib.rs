#![forbid(unsafe_code)]

pub mod attention_state;
pub mod hf_config;
pub mod kv_manager;
pub mod plan;
pub mod retention;
pub mod state_checkpoint;
pub use attention_state::{
    AttentionStateBackend, AttentionStateError, AttentionStatePlanInput, AttentionStateSpec,
    AttentionStateStorage, CompiledAttentionState, CompiledAttentionStatePlan, RecurrentFamily,
    StateComponentGeometry, compile_attention_state_manager_plan, compile_attention_state_plan,
};
pub use hf_config::{
    HfConfigError, HfLayerInference, HfManagerPlanError, HfRetentionCompilation,
    HfRetentionOptions, compile_hf_config, compile_hf_manager_plan,
};
pub use plan::{
    CompiledKvClass, CompiledKvPlan, KvClassSpec, KvPlanInput, PlanError, TokenComponentSpec,
    TokenStorageKind, compile_plan,
};
pub use state_checkpoint::{
    StateCheckpointError, StateCheckpointPool, StateCompletionReceipt, StateCopyIntent,
    StateCopyReceipt, StatePublication, StateRetirementCertificate, StateRetirementLease,
    StateSlotLease, StateTransitionLease,
};
