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

use crate::{
    providers::{build::NativeBuild, registry::ProviderId},
    target::CudaTarget,
};

use std::{
    ffi::{CStr, c_char, c_void},
    path::{Path, PathBuf},
    sync::OnceLock,
};

use crate::providers::provider_source::{ResolvedProviderSource, content_digest};

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

type FlashInferJitKey = (CudaTarget, usize, bool, usize);
type FlashInferRegistry =
    std::sync::Mutex<std::collections::HashMap<FlashInferJitKey, &'static FlashInferLib>>;

/// One compiled wrapper per head-dimension, sliding-window, and grouped-query
/// geometry. Libraries are leaked because each `.so` remains mapped for the
/// process lifetime anyway.
static FLASHINFER_LIBS: OnceLock<FlashInferRegistry> = OnceLock::new();

/// Ensure the FlashInfer library is compiled and loaded for the given
/// HEAD_DIM and sliding-window variant. Thread-safe.
pub(crate) fn ensure_compiled(
    target: CudaTarget,
    head_dim: usize,
    use_swa: bool,
    group_size: usize,
) -> anyhow::Result<&'static FlashInferLib> {
    let libs = FLASHINFER_LIBS.get_or_init(Default::default);
    let mut libs = libs.lock().unwrap();
    if let Some(lib) = libs.get(&(target, head_dim, use_swa, group_size)) {
        return Ok(lib);
    }
    anyhow::ensure!(
        super::CAPABILITIES
            .kernels
            .iter()
            .any(|kernel| kernel.head_dimensions.contains(&(head_dim, head_dim))),
        "FlashInfer: HEAD_DIM={} has no declared native instantiation",
        head_dim
    );
    anyhow::ensure!(
        group_size > 0,
        "FlashInfer: GQA group size must be positive"
    );
    anyhow::ensure!(
        super::CAPABILITIES.supports_target(target.major),
        "FlashInfer does not support execution target {}",
        target.architecture()
    );
    let so_path = compile_or_cache(target, head_dim, use_swa, group_size)?;
    let lib: &'static FlashInferLib =
        Box::leak(Box::new(unsafe { FlashInferLib::load(&so_path)? }));
    libs.insert((target, head_dim, use_swa, group_size), lib);
    Ok(lib)
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

fn compile_or_cache(
    target: CudaTarget,
    head_dim: usize,
    use_swa: bool,
    group_size: usize,
) -> anyhow::Result<PathBuf> {
    let provider = resolved_source().map_err(anyhow::Error::msg)?;
    let source = WRAPPER_CU.replace("#include \"wrapper.h\"", WRAPPER_H);
    let mut arguments = [
        "-shared",
        "-std=c++17",
        "-O3",
        "--expt-relaxed-constexpr",
        "-w",
        "-rdc=true",
        "-Xcompiler=-fPIC",
    ]
    .map(str::to_owned)
    .to_vec();
    arguments.extend([
        format!("-arch={}", target.architecture()),
        format!("-DORBITKV_HEAD_DIM={head_dim}"),
        format!("-DORBITKV_USE_SWA={}", u8::from(use_swa)),
        format!("-DORBITKV_GQA_GROUP_SIZE={group_size}"),
    ]);
    for include in ["include", "3rdparty/cutlass/include"] {
        arguments.push(format!("-I{}", provider.root.join(include).display()));
    }
    NativeBuild {
        provider: ProviderId::FlashInfer,
        sources: provider,
        source: &source,
        arguments,
        link_arguments: Vec::new(),
    }
    .compile()
}

fn resolved_source() -> Result<&'static ResolvedProviderSource, String> {
    static SOURCE: OnceLock<Result<ResolvedProviderSource, String>> = OnceLock::new();
    SOURCE
        .get_or_init(|| ProviderId::FlashInfer.source()?.resolve_identity())
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
