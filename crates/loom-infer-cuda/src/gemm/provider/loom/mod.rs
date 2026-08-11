//! Loom cuda-oxide provider for dense BF16 GEMM.

mod sm90;

use crate::gemm::plan::Bf16DenseGemmPlanError;
use cuda_core::CudaContext;
use cuda_host::embedded::artifact_bundles_from_current_exe;
use loom_infer::Bf16DenseGemmSpec;
use sm90::Sm90Provider;
use std::sync::{Arc, Mutex};

pub(crate) use sm90::LoomBf16DensePlan;

const H20_DEVICE_NAME: &str = "NVIDIA H20";
const H20_COMPUTE_CAPABILITY: (i32, i32) = (9, 0);
const SM90A_ARTIFACT_TARGET: &str = "sm_90a";

/// Lazily loads the experimental Loom artifact only for explicit Loom plans.
#[derive(Clone)]
pub(crate) struct LoomProvider {
    context: Arc<CudaContext>,
    sm90: Arc<Mutex<Option<Sm90Provider>>>,
}

impl LoomProvider {
    pub(crate) fn new(context: &Arc<CudaContext>) -> Self {
        Self {
            context: context.clone(),
            sm90: Arc::new(Mutex::new(None)),
        }
    }

    pub(crate) fn plan_bf16_dense(
        &self,
        spec: Bf16DenseGemmSpec,
    ) -> Result<LoomBf16DensePlan, Bf16DenseGemmPlanError> {
        sm90::validate_spec(spec)?;

        let provider = {
            let mut slot = self
                .sm90
                .lock()
                .map_err(|_| Bf16DenseGemmPlanError::LoomProviderLockPoisoned)?;
            if slot.is_none() {
                validate_h20_target(&self.context)?;
                *slot = Some(Sm90Provider::load(&self.context)?);
            }
            slot.as_ref()
                .expect("Loom SM90 provider was initialized above")
                .clone()
        };
        provider.plan_bf16_dense(spec)
    }

    pub(crate) const fn workspace_limit_bytes(&self) -> usize {
        0
    }
}

fn validate_h20_target(context: &CudaContext) -> Result<(), Bf16DenseGemmPlanError> {
    let device = context.device_name()?;
    if device != H20_DEVICE_NAME {
        return Err(Bf16DenseGemmPlanError::LoomUnsupportedDevice { device });
    }

    let (major, minor) = context.compute_capability()?;
    if (major, minor) != H20_COMPUTE_CAPABILITY {
        return Err(Bf16DenseGemmPlanError::LoomUnsupportedComputeCapability { major, minor });
    }

    let bundles = artifact_bundles_from_current_exe()?;
    let actual = bundles
        .iter()
        .find(|bundle| bundle.name == env!("CARGO_PKG_NAME"))
        .map(|bundle| bundle.target.clone());
    if actual.as_deref() != Some(SM90A_ARTIFACT_TARGET) {
        return Err(Bf16DenseGemmPlanError::LoomUnsupportedArtifactTarget { actual });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn artifact_target_gate_is_exact() {
        assert_eq!(SM90A_ARTIFACT_TARGET, "sm_90a");
        assert_ne!(SM90A_ARTIFACT_TARGET, "sm_90");
    }

    #[test]
    fn h20_identity_is_exact() {
        assert_eq!(H20_DEVICE_NAME, "NVIDIA H20");
        assert_eq!(H20_COMPUTE_CAPABILITY, (9, 0));
    }
}
