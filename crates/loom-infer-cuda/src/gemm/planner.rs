use crate::gemm::plan::{
    Bf16DenseGemmPlan, Bf16DenseGemmPlanError, GemmProviderId, GemmProviderVersion,
};
use crate::gemm::provider::{CublasLtProvider, LoomProvider};
use cuda_core::CudaContext;
use loom_infer::Bf16DenseGemmSpec;
use std::sync::Arc;

/// Provider selection for dense BF16 GEMM planning.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Bf16DenseGemmSelection {
    CublasLt,
    Loom,
}

/// Creates immutable GEMM plans for one CUDA context.
#[derive(Clone)]
pub struct GemmPlanner {
    cublaslt: CublasLtProvider,
    loom: LoomProvider,
}

impl GemmPlanner {
    pub fn load(context: &Arc<CudaContext>) -> Result<Self, Bf16DenseGemmPlanError> {
        Ok(Self {
            cublaslt: CublasLtProvider::load(context)?,
            loom: LoomProvider::new(context),
        })
    }

    /// Selects and freezes one provider algorithm before command submission.
    pub fn plan_bf16_dense(
        &self,
        spec: Bf16DenseGemmSpec,
        selection: Bf16DenseGemmSelection,
    ) -> Result<Bf16DenseGemmPlan, Bf16DenseGemmPlanError> {
        match selection {
            Bf16DenseGemmSelection::CublasLt => self
                .cublaslt
                .plan_bf16_dense(spec)
                .map(Bf16DenseGemmPlan::from_cublaslt),
            Bf16DenseGemmSelection::Loom => self
                .loom
                .plan_bf16_dense(spec)
                .map(Bf16DenseGemmPlan::from_loom),
        }
    }

    pub fn provider_version(&self, provider: GemmProviderId) -> GemmProviderVersion {
        match provider {
            GemmProviderId::CublasLt => {
                GemmProviderVersion::CublasLt(self.cublaslt.library_version())
            }
            GemmProviderId::Loom => GemmProviderVersion::Loom(env!("CARGO_PKG_VERSION")),
        }
    }

    pub fn workspace_limit_bytes(&self, selection: Bf16DenseGemmSelection) -> usize {
        match selection {
            Bf16DenseGemmSelection::CublasLt => self.cublaslt.workspace_limit_bytes(),
            Bf16DenseGemmSelection::Loom => self.loom.workspace_limit_bytes(),
        }
    }
}
