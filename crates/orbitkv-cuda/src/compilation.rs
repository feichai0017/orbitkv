//! NVRTC compilation, module images, and compilation resource limits.

use std::{
    cell::Cell,
    ffi::{CStr, CString},
    os::raw::c_int,
    path::{Path, PathBuf},
    sync::Arc,
};

use crate::artifact::{ModuleImageLookup, lookup_module_image, module_key, record_module_image};
use cudarc::{
    driver::{CudaContext, DriverError, sys as driver_sys},
    nvrtc::{
        Ptx,
        result::{self as nvrtc_result, NvrtcError},
        sys as nvrtc_sys,
    },
};
use orbitkv_compiler::dtype::DType;

thread_local! {
    /// Compilation is synchronous, so a thread-local budget lets each runtime
    /// control its own NVRTC safety limit without leaking policy across
    /// concurrent searches. Direct kernel compilation keeps the safe default.
    static KERNEL_SOURCE_LIMIT_BYTES: Cell<Option<usize>> =
        const { Cell::new(Some(crate::resource::DEFAULT_MAX_KERNEL_SOURCE_BYTES)) };
}

struct KernelSourceLimitGuard {
    previous: Option<usize>,
}

impl Drop for KernelSourceLimitGuard {
    fn drop(&mut self) {
        KERNEL_SOURCE_LIMIT_BYTES.with(|limit| limit.set(self.previous));
    }
}

pub(crate) fn with_kernel_source_limit<T>(limit: Option<usize>, compile: impl FnOnce() -> T) -> T {
    let previous = KERNEL_SOURCE_LIMIT_BYTES.with(|current| current.replace(limit));
    let _guard = KernelSourceLimitGuard { previous };
    compile()
}

fn kernel_source_limit() -> Option<usize> {
    KERNEL_SOURCE_LIMIT_BYTES.with(Cell::get)
}

#[cfg(test)]
#[path = "../tests/unit/kernel_source_limit_tests.rs"]
mod kernel_source_limit_tests;

/// Map a OrbitKV dtype to the CUDA C++ storage type used by generated kernels.
///
/// All generated kernels share this storage-type contract.
#[doc(hidden)]
pub fn cuda_dtype(dtype: DType) -> &'static str {
    match dtype {
        DType::F64 => "double",
        DType::F32 => "float",
        DType::F16 => "half",
        DType::Bf16 => "__nv_bfloat16",
        DType::TF32 => "float", // TF32 uses float storage, tensor cores handle the format
        DType::Int => "int",
        DType::I64 => "long long",
        DType::I16 => "short",
        DType::U16 => "unsigned short",
        DType::I8 => "signed char",
        DType::U8 => "unsigned char",
        DType::Bool => "unsigned char",
        DType::F8E4M3 => "__nv_fp8_e4m3",
        DType::F8E5M2 => "__nv_fp8_e5m2",
        DType::F8UE8M0 => "__nv_fp8_e8m0",
        DType::F6E2M3 => "__nv_fp6_e2m3",
        DType::F6E3M2 => "__nv_fp6_e3m2",
        DType::F4E2M1 => "__nv_fp4_e2m1",
        DType::I4 | DType::U4 => "unsigned char", // Sub-byte, packed storage
    }
}

const CUDA_NVRTC_INCLUDE_PATHS: [&str; 2] = ["/usr/local/cuda/include", "/usr/include"];

#[doc(hidden)]
#[derive(Debug)]
pub enum CudaModuleImageCompileFailure {
    ComputeCapability(DriverError),
    Nvrtc {
        stage: &'static str,
        error: NvrtcError,
    },
    NoModuleImageProduced,
    ArtifactMiss {
        key: String,
        available: usize,
    },
}

#[doc(hidden)]
#[derive(Debug)]
pub struct CudaModuleImageCompileError {
    pub target_arch: Option<String>,
    pub driver_version: Option<i32>,
    pub runtime_version: Option<i32>,
    pub nvrtc_options: Vec<String>,
    pub nvrtc_log: Option<String>,
    pub failure: CudaModuleImageCompileFailure,
}

impl std::fmt::Display for CudaModuleImageCompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "failed to compile CUDA module image")?;
        if let Some(target_arch) = &self.target_arch {
            write!(f, " for {target_arch}")?;
        }
        match &self.failure {
            CudaModuleImageCompileFailure::ComputeCapability(error) => {
                write!(f, ": failed to query compute capability: {error}")?;
            }
            CudaModuleImageCompileFailure::Nvrtc { stage, error } => {
                write!(f, ": NVRTC {stage} failed: {error}")?;
            }
            CudaModuleImageCompileFailure::NoModuleImageProduced => {
                write!(f, ": NVRTC produced no CUBIN for the selected target")?;
            }
            CudaModuleImageCompileFailure::ArtifactMiss { key, available } => {
                write!(
                    f,
                    ": CUBIN {key} is missing from the artifact ({available} saved)"
                )?;
            }
        }
        if let Some(version) = self.driver_version {
            write!(f, " | driver {}", format_cuda_version(version))?;
        }
        if let Some(version) = self.runtime_version {
            write!(f, " | runtime {}", format_cuda_version(version))?;
        }
        if !self.nvrtc_options.is_empty() {
            write!(f, " | options {:?}", self.nvrtc_options)?;
        }
        if let Some(log) = &self.nvrtc_log {
            write!(f, " | log: {log}")?;
        }
        Ok(())
    }
}

impl std::error::Error for CudaModuleImageCompileError {}

fn format_cuda_version(version: i32) -> String {
    format!("{}.{}", version / 1000, (version % 1000) / 10)
}

fn cuda_nvrtc_include_paths() -> &'static [String] {
    static INCLUDE_PATHS: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    INCLUDE_PATHS.get_or_init(|| {
        let mut include_paths = Vec::new();
        for env_var in ["CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"] {
            if let Ok(root) = std::env::var(env_var) {
                let path = format!("{root}/include");
                if Path::new(&path).exists() && !include_paths.contains(&path) {
                    include_paths.push(path);
                }
            }
        }
        for path in CUDA_NVRTC_INCLUDE_PATHS {
            let path = path.to_string();
            if Path::new(&path).exists() && !include_paths.contains(&path) {
                include_paths.push(path);
            }
        }

        // NVRTC parses the CUDA headers itself, so using headers from a newer
        // toolkit than the dynamically loaded compiler can make otherwise-valid
        // kernels fail before code generation (FP8 headers are particularly
        // sensitive to this). Prefer the include tree whose CUDA_VERSION matches
        // the loaded NVRTC, while preserving the configured order as a tie-break.
        // The loaded compiler and process environment are stable after startup;
        // cache the filesystem scan instead of rereading cuda.h for every kernel.
        if let Some(nvrtc_version) = loaded_nvrtc_version() {
            include_paths.sort_by_key(|path| {
                cuda_header_version(path)
                    .map(|header_version| header_version.abs_diff(nvrtc_version))
                    .unwrap_or(u32::MAX)
            });
        }
        include_paths
    })
}

type NvrtcVersionFn = unsafe extern "C" fn(*mut c_int, *mut c_int) -> nvrtc_sys::nvrtcResult;

pub(crate) fn loaded_nvrtc_version() -> Option<u32> {
    static VERSION: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *VERSION.get_or_init(|| query_nvrtc_version(nvrtc_library_candidates()))
}

fn query_nvrtc_version(candidates: impl IntoIterator<Item = PathBuf>) -> Option<u32> {
    for candidate in candidates {
        // Do not call cudarc's generated nvrtcVersion wrapper here. Under its
        // fallback dynamic loader, the wrapper panics when the library or
        // symbol is absent. Opening the library and resolving the symbol
        // directly keeps version detection optional without manipulating the
        // process-global panic hook.
        let Ok(library) = (unsafe { libloading::Library::new(candidate) }) else {
            continue;
        };
        let Ok(version_fn) = (unsafe { library.get::<NvrtcVersionFn>(b"nvrtcVersion\0") }) else {
            continue;
        };
        let (mut major, mut minor) = (0, 0);
        if unsafe { version_fn(&mut major, &mut minor) }
            .result()
            .is_ok()
            && let Some(version) = encode_nvrtc_version(major, minor)
        {
            return Some(version);
        }
    }
    None
}

fn encode_nvrtc_version(major: c_int, minor: c_int) -> Option<u32> {
    let major = u32::try_from(major).ok()?;
    let minor = u32::try_from(minor).ok()?;
    major.checked_mul(1000)?.checked_add(minor.checked_mul(10)?)
}

fn nvrtc_library_candidates() -> Vec<PathBuf> {
    use std::env::consts::{DLL_PREFIX, DLL_SUFFIX};

    let pointer_width = if cfg!(target_pointer_width = "32") {
        "32"
    } else {
        "64"
    };
    let major = driver_sys::CUDA_VERSION / 1000;
    let minor = (driver_sys::CUDA_VERSION % 1000) / 10;

    // Keep this sequence identical to cudarc 0.19's dynamic NVRTC loader.
    // Version probing must resolve the same library that compile_ptx will use;
    // searching extra toolkit versions or CUDA_HOME paths can silently pair a
    // compiler with the wrong headers and kernel-parameter ABI limit.
    [
        format!("{DLL_PREFIX}nvrtc{DLL_SUFFIX}"),
        format!("{DLL_PREFIX}nvrtc{pointer_width}{DLL_SUFFIX}"),
        format!("{DLL_PREFIX}nvrtc{pointer_width}_{major}{DLL_SUFFIX}"),
        format!("{DLL_PREFIX}nvrtc{pointer_width}_{major}{minor}{DLL_SUFFIX}"),
        format!("{DLL_PREFIX}nvrtc{pointer_width}_{major}{minor}_0{DLL_SUFFIX}"),
        format!("{DLL_PREFIX}nvrtc{pointer_width}_{major}0_{minor}{DLL_SUFFIX}"),
        format!("{DLL_PREFIX}nvrtc{pointer_width}_10{DLL_SUFFIX}"),
        format!("{DLL_PREFIX}nvrtc{pointer_width}_11{DLL_SUFFIX}"),
        format!("{DLL_PREFIX}nvrtc{pointer_width}_12{DLL_SUFFIX}"),
        format!("{DLL_PREFIX}nvrtc{pointer_width}_{major}0_0{DLL_SUFFIX}"),
        format!("{DLL_PREFIX}nvrtc{pointer_width}_9{DLL_SUFFIX}"),
        format!("{DLL_PREFIX}nvrtc{DLL_SUFFIX}.{major}"),
        format!("{DLL_PREFIX}nvrtc{DLL_SUFFIX}.12"),
        format!("{DLL_PREFIX}nvrtc{DLL_SUFFIX}.11"),
        format!("{DLL_PREFIX}nvrtc{DLL_SUFFIX}.10"),
        format!("{DLL_PREFIX}nvrtc{DLL_SUFFIX}.9"),
        format!("{DLL_PREFIX}nvrtc{DLL_SUFFIX}.1"),
    ]
    .into_iter()
    .map(PathBuf::from)
    .collect()
}

fn cuda_header_version(include_path: &str) -> Option<u32> {
    let header = std::fs::read_to_string(Path::new(include_path).join("cuda.h")).ok()?;
    parse_cuda_header_version(&header)
}

fn parse_cuda_header_version(header: &str) -> Option<u32> {
    header.lines().find_map(|line| {
        let mut fields = line.split_whitespace();
        match (fields.next(), fields.next(), fields.next()) {
            (Some("#define"), Some("CUDA_VERSION"), Some(version)) => version.parse().ok(),
            _ => None,
        }
    })
}

#[cfg(test)]
#[path = "../tests/unit/nvrtc_header_tests.rs"]
mod nvrtc_header_tests;

fn cuda_driver_diagnostics() -> (Option<i32>, Option<i32>) {
    let mut driver_version = 0;
    let driver_version = unsafe { driver_sys::cuDriverGetVersion(&mut driver_version as *mut _) }
        .result()
        .ok()
        .map(|_| driver_version);

    // Avoid touching cudarc's runtime loader here. On some environments it eagerly
    // resolves newer libcudart symbols that may not exist in the installed runtime.
    (driver_version, None)
}

pub(crate) fn cuda_nvrtc_compile_options(target_arch: &str) -> Vec<String> {
    let mut options = cuda_nvrtc_include_paths()
        .iter()
        .map(|path| format!("--include-path={path}"))
        .collect::<Vec<_>>();
    options.push(format!("--gpu-architecture={target_arch}"));
    options
}

fn build_module_image_compile_error(
    target_arch: Option<String>,
    driver_version: Option<i32>,
    runtime_version: Option<i32>,
    nvrtc_options: &[String],
    nvrtc_log: Option<String>,
    failure: CudaModuleImageCompileFailure,
) -> CudaModuleImageCompileError {
    CudaModuleImageCompileError {
        target_arch,
        driver_version,
        runtime_version,
        nvrtc_options: nvrtc_options.to_vec(),
        nvrtc_log,
        failure,
    }
}

fn read_nvrtc_log(program: nvrtc_sys::nvrtcProgram) -> Option<String> {
    let raw = unsafe { nvrtc_result::get_program_log(program).ok()? };
    if raw.is_empty() {
        return None;
    }
    let log = unsafe { CStr::from_ptr(raw.as_ptr()) }
        .to_string_lossy()
        .trim_end_matches('\0')
        .trim()
        .to_string();
    if log.is_empty() { None } else { Some(log) }
}

#[allow(clippy::slow_vector_initialization)]
fn get_cubin(program: nvrtc_sys::nvrtcProgram) -> Result<Vec<u8>, NvrtcError> {
    let mut cubin_size = 0usize;
    unsafe { nvrtc_sys::nvrtcGetCUBINSize(program, &mut cubin_size as *mut _) }.result()?;
    if cubin_size == 0 {
        return Ok(Vec::new());
    }

    let mut cubin = Vec::with_capacity(cubin_size);
    cubin.resize(cubin_size, 0u8);
    unsafe { nvrtc_sys::nvrtcGetCUBIN(program, cubin.as_mut_ptr() as *mut _) }.result()?;
    Ok(cubin)
}

#[doc(hidden)]
pub fn compile_module_image_for_current_device<S: AsRef<str>>(
    ctx: &Arc<CudaContext>,
    src: S,
) -> Result<Ptx, CudaModuleImageCompileError> {
    let (driver_version, runtime_version) = cuda_driver_diagnostics();
    let (major, minor) = ctx.compute_capability().map_err(|error| {
        build_module_image_compile_error(
            None,
            driver_version,
            runtime_version,
            &[],
            None,
            CudaModuleImageCompileFailure::ComputeCapability(error),
        )
    })?;
    let target_arch = format!("sm_{major}{minor}");
    let nvrtc_options = cuda_nvrtc_compile_options(&target_arch);
    let source = src.as_ref();
    let module_key = module_key(source);
    match lookup_module_image(&module_key) {
        ModuleImageLookup::Hit(image) => {
            let _stage = tracing::info_span!(target: "orbitkv::stage", "cuda.module_image.hit", source_digest = %module_key, image_bytes = image.len()).entered();
            return Ok(Ptx::from_binary(image));
        }
        ModuleImageLookup::Missing { available } => {
            return Err(build_module_image_compile_error(
                Some(target_arch),
                driver_version,
                runtime_version,
                &nvrtc_options,
                None,
                CudaModuleImageCompileFailure::ArtifactMiss {
                    key: module_key,
                    available,
                },
            ));
        }
        ModuleImageLookup::Compile => {}
    }

    // NVRTC compile time grows super-linearly with source size. The active
    // runtime installs its configured budget around compilation; direct calls
    // use the default safety budget.
    let src_len = src.as_ref().len();
    if let Some(limit) = kernel_source_limit()
        && src_len > limit
    {
        panic!("kernel source too large for nvrtc ({src_len} bytes > {limit})");
    }
    if src_len > 128 * 1024 {
        eprintln!("nvrtc: compiling a large kernel ({src_len} bytes)");
    }

    let _stage = tracing::info_span!(target: "orbitkv::stage", "cuda.nvrtc.compile", source_bytes = src_len, architecture = %target_arch).entered();
    let source = CString::new(src.as_ref().as_bytes())
        .expect("CUDA source code cannot contain null terminators");
    let program = nvrtc_result::create_program(&source, None).map_err(|error| {
        build_module_image_compile_error(
            Some(target_arch.clone()),
            driver_version,
            runtime_version,
            &nvrtc_options,
            None,
            CudaModuleImageCompileFailure::Nvrtc {
                stage: "create_program",
                error,
            },
        )
    })?;

    if let Err(error) = unsafe { nvrtc_result::compile_program(program, &nvrtc_options) } {
        let nvrtc_log = read_nvrtc_log(program);
        let _ = unsafe { nvrtc_result::destroy_program(program) };
        return Err(build_module_image_compile_error(
            Some(target_arch),
            driver_version,
            runtime_version,
            &nvrtc_options,
            nvrtc_log,
            CudaModuleImageCompileFailure::Nvrtc {
                stage: "compile_program",
                error,
            },
        ));
    }

    let nvrtc_log = read_nvrtc_log(program);
    let cubin = match get_cubin(program) {
        Ok(cubin) => cubin,
        Err(error) => {
            let _ = unsafe { nvrtc_result::destroy_program(program) };
            return Err(build_module_image_compile_error(
                Some(target_arch),
                driver_version,
                runtime_version,
                &nvrtc_options,
                nvrtc_log,
                CudaModuleImageCompileFailure::Nvrtc {
                    stage: "get_cubin",
                    error,
                },
            ));
        }
    };

    if let Err(error) = unsafe { nvrtc_result::destroy_program(program) } {
        return Err(build_module_image_compile_error(
            Some(target_arch),
            driver_version,
            runtime_version,
            &nvrtc_options,
            nvrtc_log,
            CudaModuleImageCompileFailure::Nvrtc {
                stage: "destroy_program",
                error,
            },
        ));
    }

    if cubin.is_empty() {
        return Err(build_module_image_compile_error(
            Some(target_arch),
            driver_version,
            runtime_version,
            &nvrtc_options,
            nvrtc_log,
            CudaModuleImageCompileFailure::NoModuleImageProduced,
        ));
    }

    record_module_image(&module_key, &cubin);
    Ok(Ptx::from_binary(cubin))
}
