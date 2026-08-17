#![forbid(unsafe_code)]

pub mod manager;
pub mod plan;
pub mod runtime;
pub mod sglang_owner;
pub mod trace;

pub use manager::{
    BlockHandle, BlockKey, BlockManagerConfig, ClassPoolConfig, ExecutionProof, KvBlockManager,
    KvView, ManagerError, ManagerStats, PhysicalReclamationReceipt, RetirementCertificate,
    SemanticProof, ViewBlock,
};
pub use plan::{
    AddressProgram, BackendDecision, BackendRequirements, CellVersion, ClassCapacity,
    ClassLayoutProgram, CompiledKvClass, CompiledKvPlan, KvClassSpec, KvPlanInput, LayoutProgram,
    LogicalCellId, PhysicalBackend, PlanError, RetentionKind, RetirementProgram,
    SglangBoundedClassPolicy, SglangPolicy, TemporalAddress, choose_physical_backend, compile_plan,
};
pub use runtime::{
    KvRuntimeSimulator, LogicalBlock, ResidentTemporalBlock, RuntimeError, Submission,
};
pub use sglang_owner::{
    CacheKind, OwnerCommand, OwnerError, OwnerResponse, OwnerStats, SglangExecutionProof,
    SglangOwner, SglangRetirementCertificate, SglangSemanticProof,
};
pub use trace::{SglangTraceEvent, TraceError, TraceSummary, summarize_sglang_trace};
