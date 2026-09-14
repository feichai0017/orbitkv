//! Execution and tuning provenance for a selected CUDA program.
//!
//! Module images retain their own target/NVRTC signature. This record also
//! covers the libraries and device on which the complete program was selected.
//! It excludes live pointers, UUIDs, free memory and mutable clock readings.

use std::collections::BTreeMap;

use cudarc::driver::{CudaContext, sys};
use serde::{Deserialize, Serialize};

use crate::{
    compilation::{cuda_nvrtc_compile_options, loaded_nvrtc_version},
    providers::{
        build::native_compiler,
        provider_source::compiler_environment_digest,
        registry::{ProviderId, ProviderOrigin, provider_lock, toolkit_library_version},
    },
    target::CudaTarget,
};

mod compatibility;
pub use compatibility::{EnvironmentChange, EnvironmentMismatch, EnvironmentRecovery};

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct DeviceTuningIdentity {
    name: String,
    multiprocessors: i32,
    total_memory_bytes: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct NvrtcIdentity {
    version: u32,
    options: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum ProviderIdentity {
    CudaToolkit { version: usize },
    NativeSource { digest: String },
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct NativeCompilerIdentity {
    executable_digest: String,
    environment_digest: String,
}

/// Provenance of the selected program, independent of model and state ownership.
///
/// Selected HostOps declare dependencies; unselected libraries and native source
/// checkouts are not probed. The provider lock additionally identifies the
/// inventory against which selection ran. Exact version matching is conservative,
/// not a claim that every differing runtime is binary-incompatible.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CudaExecutionEnvironment {
    target: CudaTarget,
    device: DeviceTuningIdentity,
    driver_api_version: i32,
    nvrtc: NvrtcIdentity,
    provider_lock_digest: String,
    providers: BTreeMap<ProviderId, ProviderIdentity>,
    native_compiler: Option<NativeCompilerIdentity>,
    cublaslt_autotune: Option<bool>,
}

impl CudaExecutionEnvironment {
    /// Capture facts from the actual execution context and selected providers.
    pub fn capture(
        context: &CudaContext,
        dependencies: impl IntoIterator<Item = ProviderId>,
    ) -> anyhow::Result<Self> {
        let target = CudaTarget::from_context(context)?;
        let mut driver_api_version = 0;
        unsafe { sys::cuDriverGetVersion(&mut driver_api_version) }.result()?;
        anyhow::ensure!(
            driver_api_version > 0,
            "CUDA driver API version is unavailable"
        );
        let nvrtc_version = loaded_nvrtc_version()
            .ok_or_else(|| anyhow::anyhow!("loaded NVRTC version is unavailable"))?;
        let mut providers = BTreeMap::new();
        for provider in dependencies {
            if providers.contains_key(&provider) {
                continue;
            }
            let identity = match provider.origin() {
                ProviderOrigin::CudaToolkit { .. } => ProviderIdentity::CudaToolkit {
                    version: toolkit_library_version(provider).map_err(anyhow::Error::msg)?,
                },
                ProviderOrigin::Git(source) => ProviderIdentity::NativeSource {
                    digest: source
                        .resolve_identity()
                        .map_err(anyhow::Error::msg)?
                        .digest,
                },
            };
            providers.insert(provider, identity);
        }
        let native_compiler = providers
            .values()
            .any(|provider| matches!(provider, ProviderIdentity::NativeSource { .. }))
            .then(|| {
                native_compiler().map(|compiler| NativeCompilerIdentity {
                    executable_digest: compiler.digest,
                    environment_digest: compiler_environment_digest(),
                })
            })
            .transpose()?;
        let cublaslt_autotune = providers
            .contains_key(&ProviderId::CublasLt)
            .then(crate::providers::cublaslt::autotune_enabled);
        Ok(Self {
            target,
            device: DeviceTuningIdentity {
                name: context.name()?,
                multiprocessors: context
                    .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?,
                total_memory_bytes: context.total_mem()?,
            },
            driver_api_version,
            nvrtc: NvrtcIdentity {
                version: nvrtc_version,
                options: cuda_nvrtc_compile_options(&target.architecture()),
            },
            provider_lock_digest: provider_lock().digest(),
            providers,
            native_compiler,
            cublaslt_autotune,
        })
    }

    /// Validate before loading weights, preparing providers or installing graphs.
    /// Strict replay rejects both rebuild and retuning requirements.
    pub fn validate_for_device(&self, context: &CudaContext) -> Result<(), String> {
        let current = Self::capture(context, self.providers.keys().copied())
            .map_err(|error| format!("cannot identify CUDA execution environment: {error:#}"))?;
        self.validate_against(&current)
            .map_err(|error| error.to_string())
    }
}

#[cfg(test)]
#[path = "../tests/unit/environment/mod.rs"]
mod tests;
