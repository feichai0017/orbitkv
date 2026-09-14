//! JIT compilation and dynamic loading of DeepGEMM kernels.
//!
//! DeepGEMM's public API is PyTorch-bound, but its generated kernel has a
//! stable raw-pointer launch contract. We instantiate that kernel behind a C
//! ABI, exactly like the FlashInfer integration does for its C++ templates.
//! Provider sources are resolved from `ORBITKV_DEEPGEMM_DIR` or fetched at a
//! pinned revision into OrbitKV's provider cache. They are not vendored as a
//! recursive source submodule.

use std::{
    ffi::{CStr, c_char, c_void},
    path::{Path, PathBuf},
    process::Command,
    sync::{Mutex, OnceLock},
};

use crate::host::provider_source::{
    GitDependency, ProviderSource, ResolvedProviderSource, compilation_key, content_digest,
    resolve_compiler,
};

/// Number of top heuristic tiles exposed as distinct e-graph alternatives.
/// This bounds search breadth; it is not a claim that four tiles are optimal.
pub(super) const SEARCH_VARIANTS: usize = 4;
pub(super) const DEEPGEMM_REVISION: &str = "559d79fb6994a58b8a15b4b93bf13ccc16edf247";
const DEEPGEMM_GIT_URL: &str = "https://github.com/deepseek-ai/DeepGEMM.git";
const CUTLASS_GIT_URL: &str = "https://github.com/NVIDIA/cutlass.git";
const CUTLASS_REVISION: &str = "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8";
const WRAPPER: &str = include_str!("wrapper.cu");
use super::contract;
pub(super) use super::tiling::Config;

fn provider_source() -> ProviderSource {
    static DEPENDENCIES: [GitDependency; 1] = [GitDependency {
        relative_path: "third-party/cutlass",
        url: CUTLASS_GIT_URL,
        revision: CUTLASS_REVISION,
    }];
    ProviderSource {
        name: "DeepGEMM",
        environment_variable: "ORBITKV_DEEPGEMM_DIR",
        cache_name: "deepgemm",
        url: DEEPGEMM_GIT_URL,
        revision: DEEPGEMM_REVISION,
        markers: &[
            "deep_gemm/include/deep_gemm",
            "third-party/cutlass/include/cute",
        ],
        source_paths: &["deep_gemm/include", "third-party/cutlass/include"],
        dependencies: &DEPENDENCIES,
    }
}

pub fn prefetch() -> Result<PathBuf, String> {
    provider_source().prefetch()
}

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

unsafe impl Send for Library {}
unsafe impl Sync for Library {}

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
            ("@PROVIDER_REVISION@", DEEPGEMM_REVISION.to_owned()),
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

pub(super) fn ensure_compiled(config: Config) -> anyhow::Result<&'static Library> {
    static LIBRARIES: OnceLock<Mutex<std::collections::HashMap<Config, &'static Library>>> =
        OnceLock::new();
    let libraries = LIBRARIES.get_or_init(Default::default);
    let mut libraries = libraries.lock().unwrap();
    if let Some(library) = libraries.get(&config) {
        return Ok(library);
    }
    let path = compile_or_cache(config)?;
    let library = Box::leak(Box::new(unsafe { Library::load(&path)? }));
    libraries.insert(config, library);
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

fn compile_or_cache(config: Config) -> anyhow::Result<PathBuf> {
    let stage = tracing::info_span!(target: "orbitkv::stage", "cuda.provider.jit", provider = "deepgemm", cache_hit = tracing::field::Empty);
    let _entered = stage.enter();
    let source = config.source();
    let provider = resolved_source().map_err(anyhow::Error::msg)?;
    let root = &provider.root;
    let cuda = std::env::var_os("CUDA_HOME")
        .or_else(|| std::env::var_os("CUDA_PATH"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/usr/local/cuda"));
    let compiler = resolve_compiler(&cuda.join("bin/nvcc")).map_err(anyhow::Error::msg)?;
    let mut arguments = [
        "-shared",
        "-std=c++20",
        "-gencode",
        "arch=compute_90a,code=sm_90a",
        "-O3",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--diag-suppress=39,161,174,177,186,940",
        "--ptxas-options=--register-usage-level=10",
        "-Xcompiler=-fPIC,-O3,-fconcepts,-Wno-deprecated-declarations,-Wno-abi",
    ]
    .map(str::to_owned)
    .to_vec();
    arguments.extend([
        format!("-I{}", root.join("deep_gemm/include").display()),
        format!("-I{}", root.join("third-party/cutlass/include").display()),
    ]);
    let link_arguments = ["-lcuda", "-lcudart"];
    let mut key_arguments = arguments.clone();
    key_arguments.push("<wrapper-source>".to_owned());
    key_arguments.extend(link_arguments.map(str::to_owned));
    let key = compilation_key(&provider.digest, &compiler.digest, &source, &key_arguments);
    let cache = std::env::var_os("ORBITKV_DEEPGEMM_CACHE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").unwrap_or_else(|| "/tmp".into()))
                .join(".cache/orbitkv/deepgemm")
        });
    std::fs::create_dir_all(&cache)?;
    let stem = format!("sm90_1d2d_{key}");
    let source_path = cache.join(format!("{stem}.cu"));
    let library_path = cache.join(format!("{stem}.so"));
    if library_path.exists() {
        stage.record("cache_hit", true);
        return Ok(library_path);
    }
    stage.record("cache_hit", false);
    std::fs::write(&source_path, source)?;

    let temporary = library_path.with_extension(format!("{}.tmp.so", std::process::id()));
    let output = Command::new(&compiler.executable)
        .args(&arguments)
        .arg(&source_path)
        .args(link_arguments)
        .arg("-o")
        .arg(&temporary)
        .output()?;
    if !output.status.success() {
        let _ = std::fs::remove_file(&temporary);
        anyhow::bail!(
            "DeepGEMM nvcc compilation failed:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }
    std::fs::rename(&temporary, &library_path)?;
    Ok(library_path)
}

fn resolved_source() -> Result<&'static ResolvedProviderSource, String> {
    static SOURCE: OnceLock<Result<ResolvedProviderSource, String>> = OnceLock::new();
    SOURCE
        .get_or_init(|| provider_source().resolve_identity(&[]))
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
#[path = "../../../tests/unit/host/deepgemm/jit/mod.rs"]
mod tests;
