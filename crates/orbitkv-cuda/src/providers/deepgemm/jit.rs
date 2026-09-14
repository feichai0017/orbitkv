//! JIT compilation and dynamic loading of DeepGEMM kernels.
//!
//! DeepGEMM's public API is PyTorch-bound, but its generated kernel has a
//! stable raw-pointer launch contract. We instantiate that kernel behind a C
//! ABI, exactly like the FlashInfer integration does for its C++ templates.
//! Provider sources are resolved from `ORBITKV_DEEPGEMM_DIR` or fetched at a
//! pinned revision into OrbitKV's provider cache. They are not vendored as a
//! recursive source submodule.

use crate::{
    providers::{build::NativeBuild, registry::ProviderId},
    target::CudaTarget,
};

use std::{
    ffi::{CStr, c_char, c_void},
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
};

use crate::providers::provider_source::{ResolvedProviderSource, content_digest};

/// Number of top heuristic tiles exposed as distinct e-graph alternatives.
/// This bounds search breadth; it is not a claim that four tiles are optimal.
pub(super) const SEARCH_VARIANTS: usize = 4;
const WRAPPER: &str = include_str!("wrapper.cu");
use super::contract;
pub(super) use super::tiling::Config;

pub(super) type RunFn = unsafe extern "C" fn(
    input: *const c_void,
    weight: *const c_void,
    weight_scale: *const c_void,
    quantized: *mut c_void,
    activation_scale: *mut c_void,
    output: *mut c_void,
    rows: i32,
    stream: *mut c_void,
) -> i32;

type RunPrequantizedFn = unsafe extern "C" fn(
    weight: *const c_void,
    weight_scale: *const c_void,
    quantized: *const c_void,
    activation_scale: *const c_void,
    output: *mut c_void,
    rows: i32,
    stream: *mut c_void,
) -> i32;

type LastErrorFn = unsafe extern "C" fn() -> *const c_char;

pub(super) struct Library {
    _library: libloading::Library,
    pub(super) run: RunFn,
    pub(super) run_prequantized: RunPrequantizedFn,
    last_error: LastErrorFn,
}

impl Library {
    pub(super) fn last_error(&self) -> String {
        let pointer = unsafe { (self.last_error)() };
        if pointer.is_null() {
            return "unknown error".to_owned();
        }
        unsafe { CStr::from_ptr(pointer) }
            .to_string_lossy()
            .into_owned()
    }
}

impl Config {
    fn source(self) -> String {
        let replacements = [
            (
                "@PROVIDER_REVISION@",
                ProviderId::DeepGemm
                    .source()
                    .expect("locked Git provider")
                    .revision
                    .clone(),
            ),
            ("@N@", self.output_features.to_string()),
            ("@K@", self.input_features.to_string()),
            ("@BLOCK_M@", self.block_m.to_string()),
            ("@BLOCK_N@", self.block_n.to_string()),
            ("@BLOCK_K@", self.block_k.to_string()),
            ("@CLUSTER_M@", self.cluster_m.to_string()),
            ("@CLUSTER_N@", self.cluster_n.to_string()),
            (
                "@CLUSTER_SIZE@",
                (self.cluster_m * self.cluster_n).to_string(),
            ),
            (
                "@MULTICAST_ON_A@",
                if self.cluster_n > 1 { "true" } else { "false" }.to_owned(),
            ),
            ("@SWIZZLE_D@", self.swizzle_d.to_string()),
            ("@STAGES@", self.stages.to_string()),
            ("@SMEM_BYTES@", self.smem_bytes.to_string()),
            ("@MATH_THREADS@", self.math_threads.to_string()),
            ("@NUM_SMS@", self.num_sms.to_string()),
        ];
        let mut source = WRAPPER.replace("@QUANTIZER@", contract::quantizer_source());
        for (placeholder, value) in replacements {
            source = source.replace(placeholder, &value);
        }
        assert!(!source.contains('@'));
        source
    }
}

pub(super) fn ensure_compiled(
    target: CudaTarget,
    config: Config,
) -> anyhow::Result<&'static Library> {
    static LIBRARIES: OnceLock<
        Mutex<std::collections::HashMap<(CudaTarget, Config), &'static Library>>,
    > = OnceLock::new();
    let libraries = LIBRARIES.get_or_init(Default::default);
    let mut libraries = libraries.lock().unwrap();
    if let Some(library) = libraries.get(&(target, config)) {
        return Ok(library);
    }
    let path = compile_or_cache(target, config)?;
    let library = Box::leak(Box::new(unsafe { Library::load(&path)? }));
    libraries.insert((target, config), library);
    Ok(library)
}

impl Library {
    unsafe fn load(path: &Path) -> anyhow::Result<Self> {
        let library = unsafe { libloading::Library::new(path)? };
        let run = unsafe { *library.get::<RunFn>(b"orbitkv_deepgemm_run\0")? };
        let run_prequantized =
            unsafe { *library.get::<RunPrequantizedFn>(b"orbitkv_deepgemm_run_prequantized\0")? };
        let last_error = unsafe { *library.get::<LastErrorFn>(b"orbitkv_deepgemm_last_error\0")? };
        Ok(Self {
            _library: library,
            run,
            run_prequantized,
            last_error,
        })
    }
}

fn compile_or_cache(target: CudaTarget, config: Config) -> anyhow::Result<PathBuf> {
    let architecture = target.hopper_architecture()?;
    let provider = resolved_source().map_err(anyhow::Error::msg)?;
    let source = config.source();
    let mut arguments = [
        "-shared",
        "-std=c++20",
        "-O3",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--diag-suppress=39,161,174,177,186,940",
        "--ptxas-options=--register-usage-level=10",
        "-Xcompiler=-fPIC,-O3,-fconcepts,-Wno-deprecated-declarations,-Wno-abi",
    ]
    .map(str::to_owned)
    .to_vec();
    // WGMMA requires the architecture-specific virtual ISA too. NVCC's
    // `-arch=sm_90a` shorthand also emits a generic sm_90 target, on which
    // DeepGEMM's unguarded WGMMA instructions cannot be assembled.
    arguments.extend([
        "-gencode".to_owned(),
        format!("arch=compute_90a,code={architecture}"),
    ]);
    for include in ["deep_gemm/include", "third-party/cutlass/include"] {
        arguments.push(format!("-I{}", provider.root.join(include).display()));
    }
    NativeBuild {
        provider: ProviderId::DeepGemm,
        sources: provider,
        source: &source,
        arguments,
        link_arguments: vec!["-lcuda".into(), "-lcudart".into()],
    }
    .compile()
}

fn resolved_source() -> Result<&'static ResolvedProviderSource, String> {
    static SOURCE: OnceLock<Result<ResolvedProviderSource, String>> = OnceLock::new();
    SOURCE
        .get_or_init(|| ProviderId::DeepGemm.source()?.resolve_identity())
        .as_ref()
        .map_err(Clone::clone)
}

/// Actual provider/dependency contents, fixed on first use in this process.
pub(super) fn provider_identity() -> Result<String, String> {
    resolved_source().map(|source| {
        format!(
            "deepgemm@sha256:{}",
            content_digest(&[
                source.digest.as_bytes(),
                WRAPPER.as_bytes(),
                contract::quantizer_source().as_bytes(),
                contract::PACKED_ACTIVATION_ABI.as_bytes(),
                // A selected schedule stores a variant index. Its mapping to
                // a tile depends on this policy as well as the CUDA template.
                include_str!("tiling.rs").as_bytes(),
            ])
        )
    })
}

#[cfg(test)]
#[path = "../../../tests/unit/providers/deepgemm/jit/mod.rs"]
mod tests;
