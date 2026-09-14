//! Locked source provenance for the adapters compiled into this backend.

use std::{collections::BTreeMap, fmt, path::PathBuf, str::FromStr, sync::OnceLock};

use serde::{Deserialize, Serialize};

use super::provider_source::{ProviderSource, content_digest};

/// An integrated adapter. Algorithm and shape eligibility remain in its rules.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderId {
    CublasLt,
    DeepGemm,
    FlashInfer,
    FlashAttention,
}

impl ProviderId {
    pub const ALL: &'static [Self] = &[
        Self::CublasLt,
        Self::DeepGemm,
        Self::FlashInfer,
        Self::FlashAttention,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::CublasLt => "cublaslt",
            Self::DeepGemm => "deepgemm",
            Self::FlashInfer => "flashinfer",
            Self::FlashAttention => "flashattention",
        }
    }

    pub fn origin(self) -> &'static ProviderOrigin {
        &provider_lock().providers[&self]
    }

    pub fn source(self) -> Result<&'static ProviderSource, String> {
        match self.origin() {
            ProviderOrigin::Git(source) => Ok(source),
            ProviderOrigin::CudaToolkit { library } => Err(format!(
                "{self} uses the CUDA Toolkit's {library} library; it has no source checkout"
            )),
        }
    }

    /// Fetch explicitly before compilation. Model execution never calls this.
    pub fn prefetch(self) -> Result<PathBuf, String> {
        self.source()?.prefetch()
    }
}

impl fmt::Display for ProviderId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for ProviderId {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .iter()
            .copied()
            .find(|provider| provider.as_str() == value)
            .ok_or_else(|| format!("provider {value:?} is not integrated"))
    }
}

/// Inspect locked provenance separately from the currently resolved inputs.
pub fn inspect(provider: ProviderId) -> Result<serde_json::Value, String> {
    match provider.origin() {
        ProviderOrigin::Git(source) => {
            let resolved = source.resolve_identity()?;
            Ok(serde_json::json!({
                "provider": provider,
                "origin": source,
                "resolved_root": resolved.root,
                "source_digest": resolved.digest,
            }))
        }
        ProviderOrigin::CudaToolkit { library } => {
            // Query the same dynamically loaded library used by the adapter.
            // This inspection requires cuBLASLt to be installed; `list` does not.
            let version = toolkit_library_version(provider)?;
            Ok(serde_json::json!({
                "provider": provider,
                "library": library,
                "runtime_version": version,
            }))
        }
    }
}

pub(crate) fn toolkit_library_version(provider: ProviderId) -> Result<usize, String> {
    let version = match provider {
        ProviderId::CublasLt => unsafe { cudarc::cublaslt::sys::cublasLtGetVersion() },
        _ => return Err(format!("{provider} is not a CUDA Toolkit library")),
    };
    if version == 0 {
        Err(format!("{provider} runtime version is unavailable"))
    } else {
        Ok(version)
    }
}

/// Toolkit libraries are discovered with CUDA; native JIT providers are pinned.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ProviderOrigin {
    CudaToolkit { library: String },
    Git(ProviderSource),
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderLock {
    pub providers: BTreeMap<ProviderId, ProviderOrigin>,
}

impl ProviderLock {
    fn parse(source: &str) -> Result<Self, String> {
        let lock: Self = serde_json::from_str(source).map_err(|error| error.to_string())?;
        if lock.providers.len() != ProviderId::ALL.len()
            || ProviderId::ALL
                .iter()
                .any(|id| !lock.providers.contains_key(id))
        {
            return Err("provider lock must describe every integrated adapter".into());
        }
        for (&id, origin) in &lock.providers {
            match origin {
                ProviderOrigin::Git(source) => {
                    if id == ProviderId::CublasLt || source.cache_name != id.as_str() {
                        return Err(format!("provider {id} has an inconsistent source identity"));
                    }
                    source.validate_lock()?;
                }
                ProviderOrigin::CudaToolkit { library }
                    if id == ProviderId::CublasLt && library == "cublasLt" => {}
                ProviderOrigin::CudaToolkit { .. } => {
                    return Err(format!("provider {id} needs a pinned native source"));
                }
            }
        }
        Ok(lock)
    }

    /// Records the lock independently from resolved, possibly edited headers.
    pub fn digest(&self) -> String {
        content_digest(&[&serde_json::to_vec(self).expect("provider lock is serializable")])
    }
}

pub fn provider_lock() -> &'static ProviderLock {
    static LOCK: OnceLock<ProviderLock> = OnceLock::new();
    LOCK.get_or_init(|| {
        ProviderLock::parse(include_str!("../../providers.lock.json"))
            .expect("embedded provider lock must pass validation")
    })
}

#[cfg(test)]
#[path = "../../tests/unit/providers/registry.rs"]
mod tests;
