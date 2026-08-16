#![forbid(unsafe_code)]

pub mod manager;
pub mod plan;
pub mod runtime;
pub mod sglang_owner;
pub mod trace;

pub use manager::{
    BlockHandle, BlockKey, BlockManagerConfig, ClassPoolConfig, ExecutionProof, KvBlockManager,
    KvView, ManagerError, ManagerStats, RetirementCertificate, SemanticProof, ViewBlock,
};
pub use plan::{
    AddressProgram, BackendDecision, BackendRequirements, ClassCapacity, ClassLayoutProgram,
    CompiledKvClass, CompiledKvPlan, KvClassSpec, KvPlanInput, LayoutProgram, PhysicalBackend,
    PlanError, RetentionKind, RetirementProgram, SglangBoundedClassPolicy, SglangPolicy,
    choose_physical_backend, compile_plan,
};
pub use runtime::{KvRuntimeSimulator, LogicalBlock, RuntimeError, Submission};
pub use sglang_owner::{
    CacheKind, OwnerCommand, OwnerError, OwnerResponse, OwnerStats, SglangExecutionProof,
    SglangOwner, SglangRetirementCertificate, SglangSemanticProof,
};
pub use trace::{SglangTraceEvent, TraceError, TraceSummary, summarize_sglang_trace};
