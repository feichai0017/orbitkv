use crate::command::{CommandError, CommandScope, Read, Write};
use crate::gemm::provider::{CublasLtBf16DensePlan, LoomBf16DensePlan};
use crate::memory::DeviceRegionLaunchError;
use cuda_core::{DriverError, LaunchContractError};
use cudarc::cublaslt::result;
use half::bf16;
use loom_infer::Bf16DenseGemmSpec;
use thiserror::Error;

/// Stable identity for a GEMM implementation provider.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GemmProviderId {
    CublasLt,
    Loom,
}

impl GemmProviderId {
    pub const fn name(self) -> &'static str {
        match self {
            Self::CublasLt => "cuBLASLt",
            Self::Loom => "Loom",
        }
    }
}

/// Version reported by the selected GEMM provider.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GemmProviderVersion {
    CublasLt(usize),
    Loom(&'static str),
}

/// Frozen algorithm used by a dense BF16 GEMM plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Bf16DenseGemmAlgorithm {
    CublasLtHeuristic,
    LoomSm90SimtGemvM1N16K64,
}

/// Stable information about one immutable dense BF16 GEMM plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Bf16DenseGemmPlanInfo {
    provider: GemmProviderId,
    algorithm: Bf16DenseGemmAlgorithm,
    workspace_required_bytes: usize,
    tensor_alignment_bytes: u64,
    workspace_alignment_bytes: u64,
}

impl Bf16DenseGemmPlanInfo {
    pub const fn provider(self) -> GemmProviderId {
        self.provider
    }

    pub const fn algorithm(self) -> Bf16DenseGemmAlgorithm {
        self.algorithm
    }

    pub const fn workspace_required_bytes(self) -> usize {
        self.workspace_required_bytes
    }

    pub const fn tensor_alignment_bytes(self) -> u64 {
        self.tensor_alignment_bytes
    }

    pub const fn workspace_alignment_bytes(self) -> u64 {
        self.workspace_alignment_bytes
    }
}

/// One immutable dense BF16 GEMM plan.
#[derive(Clone)]
pub struct Bf16DenseGemmPlan {
    inner: Bf16DenseGemmPlanInner,
}

#[derive(Clone)]
enum Bf16DenseGemmPlanInner {
    CublasLt(CublasLtBf16DensePlan),
    Loom(LoomBf16DensePlan),
}

impl Bf16DenseGemmPlan {
    pub(crate) const fn from_cublaslt(plan: CublasLtBf16DensePlan) -> Self {
        Self {
            inner: Bf16DenseGemmPlanInner::CublasLt(plan),
        }
    }

    pub(crate) const fn from_loom(plan: LoomBf16DensePlan) -> Self {
        Self {
            inner: Bf16DenseGemmPlanInner::Loom(plan),
        }
    }

    pub fn spec(&self) -> Bf16DenseGemmSpec {
        match &self.inner {
            Bf16DenseGemmPlanInner::CublasLt(plan) => plan.spec(),
            Bf16DenseGemmPlanInner::Loom(plan) => plan.spec(),
        }
    }

    pub fn plan_info(&self) -> Bf16DenseGemmPlanInfo {
        match &self.inner {
            Bf16DenseGemmPlanInner::CublasLt(plan) => Bf16DenseGemmPlanInfo {
                provider: GemmProviderId::CublasLt,
                algorithm: Bf16DenseGemmAlgorithm::CublasLtHeuristic,
                workspace_required_bytes: plan.workspace_required_bytes(),
                tensor_alignment_bytes: plan.tensor_alignment_bytes(),
                workspace_alignment_bytes: plan.workspace_alignment_bytes(),
            },
            Bf16DenseGemmPlanInner::Loom(plan) => Bf16DenseGemmPlanInfo {
                provider: GemmProviderId::Loom,
                algorithm: Bf16DenseGemmAlgorithm::LoomSm90SimtGemvM1N16K64,
                workspace_required_bytes: plan.workspace_required_bytes(),
                tensor_alignment_bytes: plan.tensor_alignment_bytes(),
                workspace_alignment_bytes: plan.workspace_alignment_bytes(),
            },
        }
    }

    pub fn workspace_required_bytes(&self) -> usize {
        self.plan_info().workspace_required_bytes()
    }

    pub fn tensor_alignment_bytes(&self) -> u64 {
        self.plan_info().tensor_alignment_bytes()
    }

    pub fn workspace_alignment_bytes(&self) -> u64 {
        self.plan_info().workspace_alignment_bytes()
    }

    /// Returns a provider estimate when the selected algorithm reports one.
    pub fn estimated_waves_count(&self) -> Option<f32> {
        match &self.inner {
            Bf16DenseGemmPlanInner::CublasLt(plan) => Some(plan.estimated_waves_count()),
            Bf16DenseGemmPlanInner::Loom(_) => None,
        }
    }

    /// Enqueues the frozen algorithm on the scope's caller-owned stream.
    pub fn enqueue_into(
        &self,
        scope: &mut CommandScope<'_>,
        operands: Bf16DenseGemmOperands,
    ) -> Result<(), Bf16DenseGemmEnqueueError> {
        match &self.inner {
            Bf16DenseGemmPlanInner::CublasLt(plan) => plan.enqueue_into(scope, operands),
            Bf16DenseGemmPlanInner::Loom(plan) => plan.enqueue_into(scope, operands),
        }
    }
}

/// Checked resource handles for one dense BF16 GEMM command.
#[derive(Clone, Copy, Debug)]
pub struct Bf16DenseGemmOperands {
    activation: Read<bf16>,
    weight: Read<bf16>,
    output: Write<bf16>,
    workspace: Write<u8>,
}

impl Bf16DenseGemmOperands {
    pub const fn new(
        activation: Read<bf16>,
        weight: Read<bf16>,
        output: Write<bf16>,
        workspace: Write<u8>,
    ) -> Self {
        Self {
            activation,
            weight,
            output,
            workspace,
        }
    }

    pub(crate) const fn activation(self) -> Read<bf16> {
        self.activation
    }

    pub(crate) const fn weight(self) -> Read<bf16> {
        self.weight
    }

    pub(crate) const fn output(self) -> Write<bf16> {
        self.output
    }

    pub(crate) const fn workspace(self) -> Write<u8> {
        self.workspace
    }
}

#[derive(Debug, Error)]
pub enum Bf16DenseGemmPlanError {
    #[error("cuBLASLt provider is poisoned by cleanup status {status}")]
    ProviderPoisoned { status: i32 },
    #[error("GEMM dimension {name}={value} does not fit the cuBLASLt ABI")]
    DimensionOutOfRange { name: &'static str, value: usize },
    #[error("selected GEMM needs {required} workspace bytes, above the {limit}-byte plan limit")]
    WorkspaceRequirementExceedsLimit { required: usize, limit: usize },
    #[error("Loom SM90 GEMV requires M=1, got {m}")]
    LoomMNotOne { m: usize },
    #[error("Loom SM90 GEMV requires N to be a multiple of 16, got {n}")]
    LoomNNotMultipleOf16 { n: usize },
    #[error("Loom SM90 GEMV requires K to be a multiple of 64, got {k}")]
    LoomKNotMultipleOf64 { k: usize },
    #[error("Loom SM90 GEMV requires device 'NVIDIA H20', got '{device}'")]
    LoomUnsupportedDevice { device: String },
    #[error("Loom SM90 GEMV requires compute capability 9.0, got {major}.{minor}")]
    LoomUnsupportedComputeCapability { major: i32, minor: i32 },
    #[error("Loom SM90 GEMV requires an sm_90a artifact, got {actual:?}")]
    LoomUnsupportedArtifactTarget { actual: Option<String> },
    #[error("Loom SM90 GEMV grid needs {blocks} blocks, exceeding the CUDA x-grid range")]
    LoomGridDimensionOutOfRange { blocks: usize },
    #[error("Loom provider initialization lock is poisoned")]
    LoomProviderLockPoisoned,
    #[error(transparent)]
    EmbeddedArtifact(#[from] cuda_core::EmbeddedModuleError),
    #[error(transparent)]
    EmbeddedModule(#[from] cuda_host::EmbeddedModuleError),
    #[error(transparent)]
    Launch(#[from] LaunchContractError),
    #[error(transparent)]
    Driver(#[from] DriverError),
    #[error(transparent)]
    CublasLt(#[from] result::CublasError),
}

#[derive(Debug, Error)]
pub enum Bf16DenseGemmEnqueueError {
    #[error(
        "GEMM plan belongs to CUDA device {plan_device}, but the stream belongs to device {stream_device}"
    )]
    ContextMismatch {
        plan_device: usize,
        stream_device: usize,
    },
    #[error("cuBLASLt provider is poisoned by cleanup status {status}")]
    ProviderPoisoned { status: i32 },
    #[error("{operand} length mismatch: expected {expected}, got {actual}")]
    LengthMismatch {
        operand: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("GEMM workspace is too small: need {required} bytes, got {actual}")]
    WorkspaceTooSmall { required: usize, actual: usize },
    #[error("{operand} must be {alignment}-byte aligned, got {address:#x}")]
    MisalignedBuffer {
        operand: &'static str,
        address: u64,
        alignment: u64,
    },
    #[error(transparent)]
    KernelLaunch(#[from] DeviceRegionLaunchError),
    #[error(transparent)]
    Command(#[from] CommandError),
    #[error(transparent)]
    Driver(#[from] DriverError),
    #[error(transparent)]
    CublasLt(#[from] result::CublasError),
}
