//! JIT compilation and dynamic loading of FlashInfer kernels.
//!
//! Everything runs at compile / profiling time — there is no `build.rs`.
//! `wrapper.cu` and `wrapper.h` are embedded via `include_str!()` and
//! extracted to the cache directory on first use. The FlashInfer + CUTLASS
//! header trees are located by probing `ORBITKV_FLASHINFER_DIR`, a small set
//! of default paths, or an explicitly prefetched pinned source cache. Model
//! compilation never fetches sources. `nvcc` is invoked with the model's
//! actual `HEAD_DIM` and the resulting `.so` is `dlopen`'d.
//!
//! `ensure_compiled` is called from `HostOp::prepare_compilation`, i.e. during
//! OrbitKV's candidate compile / validation phase, not from timed execution.
//! After the first call the process-wide registry makes subsequent lookups free.

use std::{
    ffi::{CStr, c_char, c_void},
    path::{Path, PathBuf},
    process::Command,
    sync::OnceLock,
};

use crate::host::provider_source::{
    GitDependency, ProviderSource, ResolvedProviderSource, compilation_key, content_digest,
    resolve_compiler,
};

// ── Function pointer types matching wrapper.h ──

/// dtype codes shared with wrapper.cu's C API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum FlashInferDType {
    F32 = 0,
    F16 = 1,
    Bf16 = 2,
}

impl FlashInferDType {
    pub fn from_dtype(dtype: orbitkv_compiler::dtype::DType) -> Option<Self> {
        match dtype {
            orbitkv_compiler::dtype::DType::F32 => Some(Self::F32),
            orbitkv_compiler::dtype::DType::F16 => Some(Self::F16),
            orbitkv_compiler::dtype::DType::Bf16 => Some(Self::Bf16),
            _ => None,
        }
    }

    pub fn size_of(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
        }
    }

    /// Prefill needs tensor cores, which only operate on 16-bit inputs.
    pub fn supports_prefill(self) -> bool {
        matches!(self, Self::F16 | Self::Bf16)
    }
}

pub type PlanFn = unsafe extern "C" fn(
    float_workspace: *mut c_void,
    float_ws_size: usize,
    int_workspace: *mut c_void,
    int_ws_size: usize,
    page_locked_int_workspace: *mut c_void,
    indptr_h: *mut i32,
    batch_size: i32,
    num_qo_heads: i32,
    num_kv_heads: i32,
    page_size: i32,
    head_dim: i32,
    dtype: i32,
    enable_cuda_graph: bool,
    stream: *mut c_void,
    plan_info_out: *mut i64,
    plan_info_len_out: *mut i32,
) -> i32;

pub type RunFn = unsafe extern "C" fn(
    float_workspace: *mut c_void,
    float_ws_size: usize,
    int_workspace: *mut c_void,
    plan_info_vec: *mut i64,
    plan_info_len: i32,
    q: *mut c_void,
    k_cache: *mut c_void,
    v_cache: *mut c_void,
    kv_indptr: *mut i32,
    kv_indices: *mut i32,
    kv_last_page_len: *mut i32,
    output: *mut c_void,
    batch_size: i32,
    num_qo_heads: i32,
    num_kv_heads: i32,
    page_size: i32,
    head_dim: i32,
    dtype: i32,
    sm_scale: f32,
    window_left: i32,
    stream: *mut c_void,
) -> i32;

pub type ExtractFn = unsafe extern "C" fn(
    slot_idx: *const i32,
    out: *mut i32,
    c: i32,
    kv_dim: i32,
    stream: *mut c_void,
);

pub type PrepareDecodeMetadataFn = unsafe extern "C" fn(
    int_workspace: *mut c_void,
    plan_info_vec: *mut i64,
    plan_info_len: i32,
    current_c: *const i32,
    slot_idx: *const i32,
    kv_indices: *mut i32,
    kv_indptr: *mut i32,
    capacity_c: i32,
    kv_dim: i32,
    stream: *mut c_void,
) -> i32;

pub type LastErrorFn = unsafe extern "C" fn() -> *const c_char;

pub type TransposeOutputFn = unsafe extern "C" fn(
    src: *const c_void,
    dst: *mut c_void,
    batch: i32,
    heads: i32,
    dim: i32,
    dtype: i32,
    stream: *mut c_void,
);

pub type PrefillPlanFn = unsafe extern "C" fn(
    float_workspace: *mut c_void,
    float_ws_size: usize,
    int_workspace: *mut c_void,
    int_ws_size: usize,
    page_locked_int_workspace: *mut c_void,
    qo_indptr_h: *mut i32,
    kv_indptr_h: *mut i32,
    total_num_rows: i32,
    batch_size: i32,
    num_qo_heads: i32,
    num_kv_heads: i32,
    page_size: i32,
    head_dim: i32,
    dtype: i32,
    window_left: i32,
    stream: *mut c_void,
    plan_info_out: *mut i64,
    plan_info_len_out: *mut i32,
) -> i32;

pub type PrefillRunFn = unsafe extern "C" fn(
    float_workspace: *mut c_void,
    float_ws_size: usize,
    int_workspace: *mut c_void,
    plan_info_vec: *mut i64,
    plan_info_len: i32,
    q: *mut c_void,
    k_cache: *mut c_void,
    v_cache: *mut c_void,
    qo_indptr: *mut i32,
    kv_indptr: *mut i32,
    kv_indices: *mut i32,
    kv_last_page_len: *mut i32,
    output: *mut c_void,
    total_num_rows: i32,
    batch_size: i32,
    num_qo_heads: i32,
    num_kv_heads: i32,
    page_size: i32,
    head_dim: i32,
    dtype: i32,
    sm_scale: f32,
    window_left: i32,
    stream: *mut c_void,
) -> i32;

// ── Embedded CUDA sources ──

const WRAPPER_CU: &str = include_str!("wrapper.cu");
const WRAPPER_H: &str = include_str!("wrapper.h");

// ── Loaded library handle ──

pub struct FlashInferLib {
    // Keep the handle alive so the dlopen'd .so remains mapped.
    _lib: libloading::Library,
    pub plan: PlanFn,
    pub run: RunFn,
    pub extract_slot_indices: ExtractFn,
    pub prepare_decode_metadata: PrepareDecodeMetadataFn,
    pub transpose_output: TransposeOutputFn,
    pub prefill_plan: PrefillPlanFn,
    pub prefill_run: PrefillRunFn,
    last_error: LastErrorFn,
}

// SAFETY: The library handle and function pointers are valid for the lifetime
// of the process. All functions are called with proper CUDA stream serialization.
unsafe impl Send for FlashInferLib {}
unsafe impl Sync for FlashInferLib {}

type FlashInferJitKey = (usize, bool, usize);
type FlashInferRegistry =
    std::sync::Mutex<std::collections::HashMap<FlashInferJitKey, &'static FlashInferLib>>;

/// One compiled wrapper per head-dimension, sliding-window, and grouped-query
/// geometry. Libraries are leaked because each `.so` remains mapped for the
/// process lifetime anyway.
static FLASHINFER_LIBS: OnceLock<FlashInferRegistry> = OnceLock::new();

/// Ensure the FlashInfer library is compiled and loaded for the given
/// HEAD_DIM and sliding-window variant. Thread-safe.
pub fn ensure_compiled(
    head_dim: usize,
    use_swa: bool,
    group_size: usize,
) -> &'static FlashInferLib {
    let libs = FLASHINFER_LIBS.get_or_init(Default::default);
    let mut libs = libs.lock().unwrap();
    if let Some(lib) = libs.get(&(head_dim, use_swa, group_size)) {
        return lib;
    }
    assert!(
        super::CAPABILITIES
            .kernels
            .iter()
            .any(|kernel| kernel.head_dimensions.contains(&(head_dim, head_dim))),
        "FlashInfer: HEAD_DIM={} has no declared native instantiation",
        head_dim
    );
    assert!(
        group_size > 0,
        "FlashInfer: GQA group size must be positive"
    );
    let so_path = compile_or_cache(head_dim, use_swa, group_size);
    let lib: &'static FlashInferLib = Box::leak(Box::new(unsafe {
        FlashInferLib::load(&so_path)
            .unwrap_or_else(|e| panic!("Failed to load FlashInfer library: {e}"))
    }));
    libs.insert((head_dim, use_swa, group_size), lib);
    lib
}

impl FlashInferLib {
    /// Load a compiled FlashInfer .so and resolve function pointers.
    ///
    /// # Safety
    /// The .so must be a valid FlashInfer wrapper compiled from wrapper.cu.
    unsafe fn load(path: &Path) -> Result<Self, libloading::Error> {
        let lib = unsafe { libloading::Library::new(path)? };
        let plan: PlanFn = unsafe { *lib.get::<PlanFn>(b"flashinfer_batch_decode_plan\0")? };
        let run: RunFn = unsafe { *lib.get::<RunFn>(b"flashinfer_batch_decode_run\0")? };
        let extract_slot_indices: ExtractFn =
            unsafe { *lib.get::<ExtractFn>(b"flashinfer_extract_slot_indices\0")? };
        let prepare_decode_metadata: PrepareDecodeMetadataFn = unsafe {
            *lib.get::<PrepareDecodeMetadataFn>(b"flashinfer_prepare_decode_metadata\0")?
        };
        let transpose_output: TransposeOutputFn =
            unsafe { *lib.get::<TransposeOutputFn>(b"flashinfer_transpose_output\0")? };
        let prefill_plan: PrefillPlanFn =
            unsafe { *lib.get::<PrefillPlanFn>(b"flashinfer_batch_prefill_plan\0")? };
        let prefill_run: PrefillRunFn =
            unsafe { *lib.get::<PrefillRunFn>(b"flashinfer_batch_prefill_run\0")? };
        let last_error: LastErrorFn =
            unsafe { *lib.get::<LastErrorFn>(b"flashinfer_last_error_message\0")? };
        Ok(Self {
            _lib: lib,
            plan,
            run,
            extract_slot_indices,
            prepare_decode_metadata,
            transpose_output,
            prefill_plan,
            prefill_run,
            last_error,
        })
    }

    pub fn error_message(&self) -> Option<String> {
        let pointer = unsafe { (self.last_error)() };
        if pointer.is_null() {
            return None;
        }
        let message = unsafe { CStr::from_ptr(pointer) }
            .to_string_lossy()
            .into_owned();
        (!message.is_empty()).then_some(message)
    }
}

/// Backend-supplied FlashInfer source compiled through Lite's shared header
/// discovery, cache, and nvcc plumbing. Superset backends own optimized
/// kernels while reusing the generic toolchain integration here.
pub struct FlashInferJitSource {
    pub label: &'static str,
    pub so_stem: &'static str,
    pub cu_name: &'static str,
    pub header_name: &'static str,
    pub cu_source: &'static str,
    pub header_source: &'static str,
    pub arch: &'static str,
    pub cutlass_tools_include: bool,
    pub extra_flags: &'static [&'static str],
    pub progress_note: &'static str,
}

/// Compile and cache backend-owned FlashInfer CUDA source without copying
/// Lite's source discovery or nvcc invocation code into the backend crate.
pub fn compile_flashinfer_source(
    head_dim: usize,
    use_swa: bool,
    source: &FlashInferJitSource,
) -> PathBuf {
    let cache_dir = cache_directory();
    std::fs::create_dir_all(&cache_dir).expect("Failed to create FlashInfer cache directory");
    let wrapper_hash =
        content_digest(&[source.cu_source.as_bytes(), source.header_source.as_bytes()]);
    let source_dir = cache_dir.join("sources").join(&wrapper_hash);
    std::fs::create_dir_all(&source_dir).expect("Failed to create FlashInfer source directory");
    let cu = source_dir.join(source.cu_name);
    write_if_changed(&cu, source.cu_source.as_bytes());
    write_if_changed(
        &source_dir.join(source.header_name),
        source.header_source.as_bytes(),
    );
    compile_or_cache_common(
        source.label,
        source.so_stem,
        head_dim,
        use_swa,
        source.arch,
        &wrapper_hash,
        &cu,
        &source_dir,
        source.cutlass_tools_include,
        source.extra_flags,
        source.progress_note,
        None,
    )
}

/// Shared nvcc compile-or-cache for embedded FlashInfer wrappers.
#[allow(clippy::too_many_arguments)] // private, two call sites, flags > struct
fn compile_or_cache_common(
    label: &str,
    so_stem: &str,
    head_dim: usize,
    use_swa: bool,
    arch: &str,
    wrapper_hash: &str,
    cu_path: &Path,
    header_dir: &Path,
    cutlass_tools_include: bool,
    extra_flags: &[&str],
    progress_note: &str,
    group_size: Option<usize>,
) -> PathBuf {
    let stage = tracing::info_span!(target: "orbitkv::stage", "cuda.provider.jit", provider = "flashinfer", cache_hit = tracing::field::Empty);
    let _entered = stage.enter();
    let cache_dir = cache_directory();
    let provider = resolved_source().unwrap_or_else(|error| panic!("{error}"));
    let flashinfer_include = provider.root.join("include");
    let cutlass_include = provider.root.join("3rdparty/cutlass/include");
    let compiler = resolve_compiler(Path::new("nvcc")).unwrap_or_else(|error| panic!("{error}"));

    let mut args = vec![
        "-shared".to_string(),
        format!("-DORBITKV_HEAD_DIM={head_dim}"),
        format!("-DORBITKV_USE_SWA={}", use_swa as u8),
        cu_path.to_str().unwrap().to_string(),
        "-I".to_string(),
        flashinfer_include.to_str().unwrap().to_string(),
        "-I".to_string(),
        cutlass_include.to_str().unwrap().to_string(),
    ];
    if let Some(group_size) = group_size {
        args.push(format!("-DORBITKV_GQA_GROUP_SIZE={group_size}"));
    }
    if cutlass_tools_include {
        // cutlass_utils.cuh also pulls in cutlass/util/* from the tools tree.
        let tools = cutlass_include.parent().unwrap().join("tools/util/include");
        args.push("-I".to_string());
        args.push(tools.to_str().unwrap().to_string());
    }
    args.extend([
        "-I".to_string(),
        header_dir.to_str().unwrap().to_string(),
        "-std=c++17".to_string(),
        format!("-arch={arch}"),
        "-O3".to_string(),
        "--expt-relaxed-constexpr".to_string(),
        "-w".to_string(),
    ]);
    args.extend(extra_flags.iter().map(|f| f.to_string()));
    args.extend(["--compiler-options".to_string(), "-fPIC".to_string()]);

    let key = compilation_key(&provider.digest, &compiler.digest, wrapper_hash, &args);
    let so_path = cache_dir.join(format!("{so_stem}_{key}.so"));
    if so_path.exists() {
        stage.record("cache_hit", true);
        eprintln!(
            "{label}: using cached library for HEAD_DIM={head_dim} ({})",
            so_path.display()
        );
        return so_path;
    }
    stage.record("cache_hit", false);
    eprintln!(
        "{label}: JIT compiling for HEAD_DIM={head_dim}, swa={}, arch={arch}{progress_note} ...",
        use_swa as u8
    );
    let start = std::time::Instant::now();
    let temporary = so_path.with_extension(format!("{}.tmp.so", std::process::id()));

    let output = Command::new(&compiler.executable)
        .args(&args)
        .arg("-o")
        .arg(&temporary)
        .output()
        .expect("Failed to run nvcc. Is the CUDA toolkit installed?");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let _ = std::fs::remove_file(&temporary);
        panic!(
            "{label} JIT compilation failed (HEAD_DIM={head_dim}, arch={arch}):\nstdout: {stdout}\nstderr: {stderr}"
        );
    }
    std::fs::rename(&temporary, &so_path).expect("Failed to install compiled FlashInfer library");

    eprintln!(
        "{label}: compiled in {:.1}s → {}",
        start.elapsed().as_secs_f64(),
        so_path.display()
    );
    so_path
}

/// Compile wrapper.cu for the given HEAD_DIM/variant, or return cached .so path.
fn compile_or_cache(head_dim: usize, use_swa: bool, group_size: usize) -> PathBuf {
    let cache_dir = cache_directory();
    std::fs::create_dir_all(&cache_dir).expect("Failed to create FlashInfer cache directory");
    // Extract bundled wrapper sources to the cache so nvcc can compile them.
    let (wrapper_cu_path, wrapper_h_dir) = extract_wrapper_sources(&cache_dir);
    compile_or_cache_common(
        "FlashInfer",
        "libflashinfer",
        head_dim,
        use_swa,
        &detect_cuda_arch(),
        &wrapper_source_hash(),
        &wrapper_cu_path,
        &wrapper_h_dir,
        /*cutlass_tools_include=*/ false,
        &["-rdc=true"],
        "",
        Some(group_size),
    )
}

/// Returns ~/.cache/orbitkv/flashinfer/
fn cache_directory() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(home)
        .join(".cache")
        .join("orbitkv")
        .join("flashinfer")
}

/// Drop the embedded wrapper.cu/wrapper.h into the cache dir so nvcc has files
/// on disk to compile. Returns (wrapper.cu path, directory containing wrapper.h).
fn extract_wrapper_sources(cache_dir: &Path) -> (PathBuf, PathBuf) {
    let source_dir = cache_dir.join("sources").join(wrapper_source_hash());
    std::fs::create_dir_all(&source_dir).expect("Failed to create FlashInfer source directory");
    let cu = source_dir.join("wrapper.cu");
    let h = source_dir.join("wrapper.h");
    write_if_changed(&cu, WRAPPER_CU.as_bytes());
    write_if_changed(&h, WRAPPER_H.as_bytes());
    (cu, source_dir)
}

fn write_if_changed(path: &Path, contents: &[u8]) {
    if let Ok(existing) = std::fs::read(path)
        && existing == contents
    {
        return;
    }
    std::fs::write(path, contents).unwrap_or_else(|e| {
        panic!(
            "FlashInfer: failed to write wrapper source to {}: {e}",
            path.display()
        )
    });
}

fn wrapper_source_hash() -> String {
    content_digest(&[WRAPPER_CU.as_bytes(), WRAPPER_H.as_bytes()])
}

// ── Pinned FlashInfer source ──
//
// These revisions select the explicitly prefetched sources. Compiled-library
// keys use the actual headers, including local edits, rather than trusting the
// revision constants. Re-check wrapper.cu when changing the upstream API.

const FLASHINFER_GIT_URL: &str = "https://github.com/flashinfer-ai/flashinfer.git";
const CUTLASS_GIT_URL: &str = "https://github.com/NVIDIA/cutlass.git";
const FLASHINFER_GIT_REV: &str = "f1e6fdcb8f65104047697f022b5d055ef022d763";
const CUTLASS_GIT_REV: &str = "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8";

fn provider_source() -> ProviderSource {
    static DEPENDENCIES: [GitDependency; 1] = [GitDependency {
        relative_path: "3rdparty/cutlass",
        url: CUTLASS_GIT_URL,
        revision: CUTLASS_GIT_REV,
    }];
    ProviderSource {
        name: "FlashInfer",
        environment_variable: "ORBITKV_FLASHINFER_DIR",
        cache_name: "flashinfer",
        url: FLASHINFER_GIT_URL,
        revision: FLASHINFER_GIT_REV,
        markers: &["include", "3rdparty/cutlass/include"],
        source_paths: &[
            "include",
            "3rdparty/cutlass/include",
            "3rdparty/cutlass/tools/util/include",
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
        .get_or_init(|| {
            let home = std::env::var("HOME").unwrap_or_default();
            let defaults = [
                PathBuf::from(&home).join("orbitkv_cuda/crates/orbitkv_cuda/flashinfer"),
                PathBuf::from(&home).join("orbitkv_cuda/flashinfer"),
                PathBuf::from("/opt/orbitkv_cuda/crates/orbitkv_cuda/flashinfer"),
            ];
            provider_source().resolve_identity(&defaults)
        })
        .as_ref()
        .map_err(Clone::clone)
}

/// Actual provider/dependency contents, fixed on first use in this process.
pub(super) fn provider_identity() -> Result<String, String> {
    resolved_source().map(|source| {
        format!(
            "flashinfer@sha256:{}",
            content_digest(&[
                source.digest.as_bytes(),
                WRAPPER_CU.as_bytes(),
                WRAPPER_H.as_bytes()
            ])
        )
    })
}

/// Detect CUDA arch via env override → nvidia-smi → default sm_80.
fn detect_cuda_arch() -> String {
    if let Ok(arch) = std::env::var("FLASHINFER_CUDA_ARCH") {
        return arch;
    }

    if let Ok(output) = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        && output.status.success()
    {
        let cap = String::from_utf8_lossy(&output.stdout);
        let cap = cap.trim().lines().next().unwrap_or("8.0");
        let sm = cap.replace('.', "");
        if !sm.is_empty() {
            return format!("sm_{}", sm);
        }
    }

    "sm_80".to_string()
}
