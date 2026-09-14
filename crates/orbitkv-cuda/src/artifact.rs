//! CUDA module images for selected schedules, independent of a model or frontend.

use std::{
    cell::RefCell,
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use base64::{Engine, engine::general_purpose::STANDARD as BASE64};
use cudarc::driver::CudaContext;
use orbitkv_compiler::{op::IntoEgglogOp, prelude::Graph};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{cuda_nvrtc_compile_options, loaded_nvrtc_version, runtime::CudaRuntimeImpl};

thread_local! {
    static MODULE_ARTIFACT_SESSION: RefCell<Option<Arc<Mutex<ModuleArtifact>>>> =
        const { RefCell::new(None) };
}

// Version 3 adds image integrity and deterministic serialization. Version 2
// backend blobs must be recaptured; schedule-only artifacts are unaffected.
const MODULE_ARTIFACT_VERSION: u32 = 3;

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ModuleArtifactSignature {
    target_arch: String,
    nvrtc_options: Vec<String>,
    nvrtc_version: Option<u32>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SerializedModuleImage {
    sha256: String,
    data: String,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SerializedModuleArtifact {
    version: u32,
    signature: ModuleArtifactSignature,
    images: BTreeMap<String, SerializedModuleImage>,
}

/// Device-targeted module images, keyed by SHA-256 of generated CUDA source.
///
/// Contains neither weights nor live CUDA resources. Deserialization verifies
/// payload integrity; loading also requires matching architecture, NVRTC
/// version and compiler options. External provider libraries remain separate.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(
    try_from = "SerializedModuleArtifact",
    into = "SerializedModuleArtifact"
)]
pub struct CudaModuleArtifact {
    signature: ModuleArtifactSignature,
    images: BTreeMap<String, Vec<u8>>,
}

impl CudaModuleArtifact {
    /// Number of distinct generated modules in this artifact.
    pub fn image_count(&self) -> usize {
        self.images.len()
    }

    /// Rejects artifacts compiled for a different target or NVRTC configuration.
    pub fn validate_for_device(&self, ctx: &Arc<CudaContext>) -> Result<(), String> {
        self.validate_signature(&module_artifact_signature(ctx)?)
    }

    fn validate_signature(&self, signature: &ModuleArtifactSignature) -> Result<(), String> {
        if &self.signature != signature {
            return Err(format!(
                "CUDA module artifact target/compiler mismatch: saved {:?}, current {:?}",
                self.signature, signature
            ));
        }
        Ok(())
    }
}

impl TryFrom<SerializedModuleArtifact> for CudaModuleArtifact {
    type Error = String;

    fn try_from(serialized: SerializedModuleArtifact) -> Result<Self, Self::Error> {
        if serialized.version != MODULE_ARTIFACT_VERSION {
            return Err(format!(
                "unsupported CUDA module artifact version {}, expected {MODULE_ARTIFACT_VERSION}",
                serialized.version
            ));
        }
        let mut images = BTreeMap::new();
        for (key, image) in serialized.images {
            if key.len() != Sha256::output_size() * 2
                || !key
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            {
                return Err(format!("invalid CUDA source digest: {key}"));
            }
            let bytes = BASE64
                .decode(image.data)
                .map_err(|error| error.to_string())?;
            if bytes.is_empty() || image.sha256 != digest(&bytes) {
                return Err(format!("CUDA module image integrity check failed: {key}"));
            }
            images.insert(key, bytes);
        }
        Ok(Self {
            signature: serialized.signature,
            images,
        })
    }
}

impl From<CudaModuleArtifact> for SerializedModuleArtifact {
    fn from(artifact: CudaModuleArtifact) -> Self {
        Self {
            version: MODULE_ARTIFACT_VERSION,
            signature: artifact.signature,
            images: artifact
                .images
                .into_iter()
                .map(|(key, bytes)| {
                    (
                        key,
                        SerializedModuleImage {
                            sha256: digest(&bytes),
                            data: BASE64.encode(bytes),
                        },
                    )
                })
                .collect(),
        }
    }
}

pub(crate) struct ModuleArtifact {
    data: CudaModuleArtifact,
    loading: bool,
    capturing: bool,
}

pub(crate) enum ModuleImageLookup {
    Compile,
    Hit(Vec<u8>),
    Missing { available: usize },
}

pub(crate) fn module_artifact_session(
    ctx: &Arc<CudaContext>,
    data: Option<&str>,
) -> Result<Arc<Mutex<ModuleArtifact>>, String> {
    let signature = module_artifact_signature(ctx)?;
    let loading = data.is_some();
    let data = if let Some(data) = data {
        let artifact: CudaModuleArtifact = serde_json::from_str(data).map_err(|e| e.to_string())?;
        artifact.validate_signature(&signature)?;
        artifact
    } else {
        CudaModuleArtifact {
            signature,
            images: BTreeMap::new(),
        }
    };
    Ok(Arc::new(Mutex::new(ModuleArtifact {
        data,
        loading,
        capturing: false,
    })))
}

pub(crate) fn current_module_artifact_session() -> Option<Arc<Mutex<ModuleArtifact>>> {
    MODULE_ARTIFACT_SESSION.with(|current| current.borrow().clone())
}

pub(crate) struct ModuleArtifactGuard(Option<Option<Arc<Mutex<ModuleArtifact>>>>);

impl ModuleArtifactGuard {
    // A runtime without an attached artifact preserves an ambient capture
    // (e.g. a dynamic backend constructing the runtime).
    pub(crate) fn enter(session: Option<Arc<Mutex<ModuleArtifact>>>) -> Self {
        Self(
            session.map(|session| {
                MODULE_ARTIFACT_SESSION.with(|current| current.replace(Some(session)))
            }),
        )
    }
}

impl Drop for ModuleArtifactGuard {
    fn drop(&mut self) {
        if let Some(previous) = self.0.take() {
            MODULE_ARTIFACT_SESSION.with(|current| current.replace(previous));
        }
    }
}

pub(crate) fn with_module_artifact_session<T>(
    session: Arc<Mutex<ModuleArtifact>>,
    run: impl FnOnce() -> T,
) -> T {
    let _guard = ModuleArtifactGuard::enter(Some(session));
    run()
}

pub(crate) fn serialize_module_artifact(session: &Mutex<ModuleArtifact>) -> String {
    serde_json::to_string(&session.lock().unwrap().data).unwrap()
}

/// Recompile only the selected schedule; rejected search candidates do not
/// inflate the saved artifact or require retaining all search images in RAM.
pub(crate) fn capture_selected_schedule<O: IntoEgglogOp + 'static>(
    graph: &Graph,
    runtime: &mut CudaRuntimeImpl<O>,
    session: &Mutex<ModuleArtifact>,
) -> Result<(), String> {
    {
        let mut artifact = session.lock().unwrap();
        if artifact.loading {
            return Ok(());
        }
        artifact.data.images.clear();
        artifact.capturing = true;
    }
    runtime.clear_kernel_cache();
    let result = graph.load_selected_schedule(runtime);
    let mut artifact = session.lock().unwrap();
    artifact.capturing = false;
    artifact.loading = result.is_ok();
    result
}

impl<O: IntoEgglogOp + 'static> CudaRuntimeImpl<O> {
    /// Captures generated modules for an installed selected schedule. This
    /// rebuilds executable resources, so call before binding live serving state.
    /// Subsequent runtime compilation is strict: missing images never invoke NVRTC.
    pub fn capture_module_artifact(&mut self, graph: &Graph) -> Result<CudaModuleArtifact, String> {
        let _stage =
            tracing::info_span!(target: "orbitkv::stage", "cuda.module_artifact.capture").entered();
        let session = module_artifact_session(self.module_artifact_context(), None)?;
        self.with_selected_module_session(session.clone(), |runtime| {
            capture_selected_schedule(graph, runtime, &session)
        })?;
        Ok(session.lock().unwrap().data.clone())
    }

    /// Loads the selected schedule using validated module images, retaining
    /// strict image lookup for later bucket materialization and execution.
    pub fn load_selected_schedule_with_modules(
        &mut self,
        graph: &Graph,
        artifact: &CudaModuleArtifact,
    ) -> Result<(), String> {
        artifact.validate_for_device(self.module_artifact_context())?;
        let session = Arc::new(Mutex::new(ModuleArtifact {
            data: artifact.clone(),
            loading: true,
            capturing: false,
        }));
        self.with_selected_module_session(session, |runtime| graph.load_selected_schedule(runtime))
    }

    fn with_selected_module_session(
        &mut self,
        session: Arc<Mutex<ModuleArtifact>>,
        run: impl FnOnce(&mut Self) -> Result<(), String>,
    ) -> Result<(), String> {
        let previous = self.module_artifact.replace(session.clone());
        // Existing CUDA functions must not conceal a missing artifact image.
        self.clear_kernel_cache();
        let result = with_module_artifact_session(session, || run(self));
        if result.is_err() {
            self.module_artifact = previous;
            self.clear_kernel_cache();
        }
        result
    }
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub(crate) fn module_key(source: &str) -> String {
    digest(source.as_bytes())
}

pub(crate) fn lookup_module_image(key: &str) -> ModuleImageLookup {
    MODULE_ARTIFACT_SESSION.with(|current| {
        let session = current.borrow();
        let Some(session) = session.as_ref() else {
            return ModuleImageLookup::Compile;
        };
        let artifact = session.lock().unwrap();
        match artifact.data.images.get(key) {
            Some(image) => ModuleImageLookup::Hit(image.clone()),
            None if artifact.loading => ModuleImageLookup::Missing {
                available: artifact.data.images.len(),
            },
            None => ModuleImageLookup::Compile,
        }
    })
}

pub(crate) fn record_module_image(key: &str, image: &[u8]) {
    MODULE_ARTIFACT_SESSION.with(|current| {
        if let Some(session) = current.borrow().as_ref() {
            let mut artifact = session.lock().unwrap();
            if artifact.capturing {
                artifact.data.images.insert(key.to_owned(), image.to_vec());
            }
        }
    });
}

/// Process-wide provider caches must participate in selected-module capture
/// and strict completeness checks even when their CUDA function is already live.
/// Capture recompiles a reused module only if this session lacks its image.
pub(crate) fn observe_cached_module(
    ctx: &Arc<CudaContext>,
    source: &str,
) -> Result<(), crate::CudaModuleImageCompileError> {
    let needed = MODULE_ARTIFACT_SESSION.with(|current| {
        let session = current.borrow();
        let Some(session) = session.as_ref() else {
            return false;
        };
        let artifact = session.lock().unwrap();
        (artifact.capturing || artifact.loading)
            && !artifact.data.images.contains_key(&module_key(source))
    });
    if needed {
        crate::compile_module_image_for_current_device(ctx, source)?;
    }
    Ok(())
}

fn module_artifact_signature(ctx: &Arc<CudaContext>) -> Result<ModuleArtifactSignature, String> {
    let (major, minor) = ctx
        .compute_capability()
        .map_err(|error| error.to_string())?;
    let target_arch = format!("sm_{major}{minor}");
    Ok(ModuleArtifactSignature {
        nvrtc_options: cuda_nvrtc_compile_options(&target_arch),
        target_arch,
        nvrtc_version: loaded_nvrtc_version(),
    })
}

#[cfg(test)]
#[path = "../tests/unit/artifact/mod.rs"]
mod tests;
