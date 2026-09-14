//! Pinned FlashAttention-3/CUTLASS source and native C ABI compilation.

use std::{
    collections::HashMap,
    ffi::{CStr, c_char, c_void},
    path::{Path, PathBuf},
    process::Command,
    sync::{Mutex, OnceLock},
};

use crate::host::provider_source::{
    GitDependency, ProviderSource, ResolvedProviderSource, compilation_key, content_digest,
    resolve_compiler,
};

const REVISION: &str = "8d3a3b80d4758ebde5a867c50d24d4351443cf2b";
const CUTLASS_REVISION: &str = "7127592069c2fe01b041e174ba4345ef9b279671";
const WRAPPER: &str = include_str!("wrapper.cu");
const HEADER: &str = include_str!("wrapper.h");

fn provider_source() -> ProviderSource {
    static DEPENDENCIES: [GitDependency; 1] = [GitDependency {
        relative_path: "csrc/cutlass",
        url: "https://github.com/NVIDIA/cutlass.git",
        revision: CUTLASS_REVISION,
    }];
    ProviderSource {
        name: "FlashAttention",
        environment_variable: "ORBITKV_FLASHATTENTION_DIR",
        cache_name: "flashattention",
        url: "https://github.com/Dao-AILab/flash-attention.git",
        revision: REVISION,
        markers: &[
            "hopper/flash_fwd_launch_template.h",
            "csrc/cutlass/include/cute",
        ],
        source_paths: &[
            "hopper",
            "csrc/cutlass/include",
            "csrc/cutlass/tools/util/include",
        ],
        dependencies: &DEPENDENCIES,
    }
}

pub fn prefetch() -> Result<PathBuf, String> {
    provider_source().prefetch()
}

fn resolved_source() -> Result<&'static ResolvedProviderSource, String> {
    static SOURCE: OnceLock<Result<ResolvedProviderSource, String>> = OnceLock::new();
    SOURCE
        .get_or_init(|| provider_source().resolve_identity(&[]))
        .as_ref()
        .map_err(Clone::clone)
}

pub(super) fn provider_identity() -> Result<String, String> {
    resolved_source().map(|source| {
        format!(
            "flashattention@sha256:{}",
            content_digest(&[
                source.digest.as_bytes(),
                WRAPPER.as_bytes(),
                HEADER.as_bytes(),
            ])
        )
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct Config {
    pub head_dim: usize,
    pub bf16: bool,
    pub local: bool,
}

/// Matches wrapper.h; tensor pointers and strides are supplied by the provider.
#[repr(C)]
pub(super) struct Launch {
    pub query: *const c_void,
    pub key: *const c_void,
    pub value: *const c_void,
    pub page_indices: *const i32,
    pub query_indptr: *const i32,
    pub page_indptr: *const i32,
    pub last_page_len: *const i32,
    pub output: *mut c_void,
    pub page_table: *mut i32,
    pub kv_lengths: *mut i32,
    pub split_counts: *mut i32,
    pub query_tiles: *mut i32,
    pub batch_order: *mut i32,
    pub head_swizzle: *mut i32,
    pub tile_counter: *mut i32,
    pub lse: *mut f32,
    pub query_tokens: i32,
    pub requests: i32,
    pub query_heads: i32,
    pub kv_heads: i32,
    pub page_size: i32,
    pub context_pages: i32,
    pub cache_pages: i32,
    pub num_sm: i32,
    pub scale: f32,
    pub window_left: i32,
}

type RunFn = unsafe extern "C" fn(*const Launch, *mut c_void) -> i32;
type LastErrorFn = unsafe extern "C" fn() -> *const c_char;

pub(super) struct Library {
    _library: libloading::Library,
    pub run: RunFn,
    last_error: LastErrorFn,
}

impl Library {
    pub fn error(&self) -> String {
        let pointer = unsafe { (self.last_error)() };
        if pointer.is_null() {
            return "unknown FlashAttention error".to_owned();
        }
        unsafe { CStr::from_ptr(pointer) }
            .to_string_lossy()
            .into_owned()
    }

    unsafe fn load(path: &Path) -> anyhow::Result<Self> {
        let library = unsafe { libloading::Library::new(path)? };
        Ok(Self {
            run: unsafe { *library.get::<RunFn>(b"orbitkv_flashattention_run\0")? },
            last_error: unsafe {
                *library.get::<LastErrorFn>(b"orbitkv_flashattention_last_error\0")?
            },
            _library: library,
        })
    }
}

pub(super) fn ensure_compiled(config: Config) -> anyhow::Result<&'static Library> {
    static LIBRARIES: OnceLock<Mutex<HashMap<Config, &'static Library>>> = OnceLock::new();
    let mut libraries = LIBRARIES.get_or_init(Default::default).lock().unwrap();
    if let Some(library) = libraries.get(&config) {
        return Ok(library);
    }
    let path = compile_or_cache(config)?;
    let library = Box::leak(Box::new(unsafe { Library::load(&path)? }));
    libraries.insert(config, library);
    Ok(library)
}

fn compile_or_cache(config: Config) -> anyhow::Result<PathBuf> {
    let stage = tracing::info_span!(target: "orbitkv::stage", "cuda.provider.jit", provider = "flashattention", cache_hit = tracing::field::Empty);
    let _entered = stage.enter();
    let provider = resolved_source().map_err(anyhow::Error::msg)?;
    let source = WRAPPER.replace("#include \"wrapper.h\"", HEADER);
    let cuda = std::env::var_os("CUDA_HOME")
        .or_else(|| std::env::var_os("CUDA_PATH"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/usr/local/cuda"));
    let compiler = resolve_compiler(&cuda.join("bin/nvcc")).map_err(anyhow::Error::msg)?;
    let mut arguments = [
        "-shared",
        "-std=c++17",
        "-gencode",
        "arch=compute_90a,code=sm_90a",
        "-O3",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-Xcompiler=-fPIC",
    ]
    .map(str::to_owned)
    .to_vec();
    arguments.extend([
        format!("-DORBITKV_HEAD_DIM={}", config.head_dim),
        format!("-DORBITKV_BF16={}", u8::from(config.bf16)),
        format!("-DORBITKV_LOCAL={}", u8::from(config.local)),
    ]);
    for include in [
        "hopper",
        "csrc/cutlass/include",
        "csrc/cutlass/tools/util/include",
    ] {
        arguments.push(format!("-I{}", provider.root.join(include).display()));
    }
    let links = ["-lcuda", "-lcudart"];
    let mut key_arguments = arguments.clone();
    key_arguments.extend(links.map(str::to_owned));
    let key = compilation_key(&provider.digest, &compiler.digest, &source, &key_arguments);
    let cache = std::env::var_os("ORBITKV_FLASHATTENTION_CACHE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").unwrap_or_else(|| "/tmp".into()))
                .join(".cache/orbitkv/flashattention")
        });
    std::fs::create_dir_all(&cache)?;
    let path = cache.join(format!("{key}.so"));
    if path.is_file() {
        stage.record("cache_hit", true);
        return Ok(path);
    }
    stage.record("cache_hit", false);
    let source_path = cache.join(format!("{key}.cu"));
    std::fs::write(&source_path, source)?;
    let temporary = path.with_extension(format!("{}.tmp.so", std::process::id()));
    let output = Command::new(&compiler.executable)
        .args(&arguments)
        .arg(&source_path)
        .args(links)
        .arg("-o")
        .arg(&temporary)
        .output()?;
    if !output.status.success() {
        let _ = std::fs::remove_file(&temporary);
        anyhow::bail!(
            "FlashAttention nvcc compilation failed:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    std::fs::rename(temporary, &path)?;
    Ok(path)
}
