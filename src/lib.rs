#![forbid(unsafe_code)]

pub mod binding;
pub mod capsule;
pub mod hf_config;
pub mod holt_store;
pub mod manager;
pub mod optimizer;
pub mod plan;
pub mod prefix;
pub mod retention;
pub mod runtime;
pub mod sglang_owner;
pub mod state_plan;
pub mod trace;

pub use binding::{
    BindingCoordinator, BindingCoordinatorStats, BindingError,
    PhysicalStateBindingComponentReceipt, PhysicalStateBindingReceipt, StateBindingComponent,
    StateBindingIntent,
};
pub use capsule::{
    CapsuleComponent, CapsuleComponentSpec, CapsuleError, CapsuleIdentity, CapsuleManifest,
    ContentDigest, PrefixChunk, PrefixPath, build_capsule_components,
};
pub use hf_config::{
    HfConfigError, HfLayerInference, HfRetentionCompilation, HfRetentionOptions, HfSglangLowering,
    HfStatePlan, HfStatePlanError, HfStatePlanOptions, SglangUniformSwaContract,
    SglangUniformSwaOptions, UniformSwaCudaGraphMode, compile_hf_config, compile_hf_state_plan,
};
pub use holt_store::{
    CapsulePublish, HoltCapsuleError, HoltCapsuleStore, RestoredCapsule, RestoredCapsuleState,
    RestoredPrefixCapsule,
};
pub use manager::{
    BindingIntent, BlockHandle, BlockKey, BlockManagerConfig, ClassPoolConfig, ExecutionProof,
    KvBlockManager, KvView, ManagerError, ManagerStats, PhysicalBindingBlockReceipt,
    PhysicalBindingReceipt, PhysicalReclamationReceipt, RetirementCertificate, SemanticProof,
    ViewBlock,
};
pub use optimizer::{
    OptimizerError, PhysicalPlanObjective, SglangPhysicalCandidate, SglangPhysicalContract,
    SglangPhysicalCost, SglangPhysicalOptimizationInput, SglangPhysicalPlan,
    optimize_sglang_physical_plan,
};
pub use plan::{
    AddressProgram, ApplicabilityClass, ApplicabilityClassGeometry, ApplicabilityReport,
    BackendDecision, BackendRequirements, CellVersion, ClassCapacity, ClassLayoutProgram,
    CompiledKvClass, CompiledKvPlan, KvClassSpec, KvPlanInput, KvPlanSource, LayoutProgram,
    LifetimeNormalizationReport, LifetimeNormalizedClass, LogicalCellId, PhysicalBackend,
    PlanError, RetentionKind, RetirementProgram, SglangBoundedClassPolicy, SglangPolicy,
    TemporalAddress, choose_physical_backend, compile_plan, compile_retention_program,
};
pub use prefix::{
    PersistentPrefixComponent, PrefixAvailability, PrefixComponentCompleteness,
    PrefixComponentSnapshot, PrefixComponentSpec, PrefixDeviceState, PrefixError, PrefixLease,
    PrefixLeaseId, PrefixObjectId, PrefixObjectSnapshot, PrefixRuntime, PrefixRuntimeStats,
    PrefixTokenRange,
};
pub use retention::{
    AtomicRetention, InferredRegion, InferredRetention, IntExpr, KvHeadRange, Predicate,
    RetentionAnalysis, RetentionError, RetentionProgramInput, RetentionStateDecl, analyze_state,
};
pub use runtime::{
    KvRuntimeSimulator, LogicalBlock, ResidentTemporalBlock, RuntimeError, Submission,
};
pub use sglang_owner::{
    CacheKind, OwnerCommand, OwnerError, OwnerResponse, OwnerStats, SglangExecutionProof,
    SglangOwner, SglangRetirementCertificate, SglangSemanticProof,
};
pub use state_plan::{
    RuntimeCapsuleContract, RuntimeExecutionContract, RuntimeExecutionMode, RuntimeOwnerTransport,
    RuntimePrefixContract, RuntimePrefixMode, RuntimeStatePlan, RuntimeStatePlanError,
    RuntimeStatePlanOptions, RuntimeUniformStatePlanMode, compile_runtime_state_plan,
};
pub use trace::{SglangTraceEvent, TraceError, TraceSummary, summarize_sglang_trace};
