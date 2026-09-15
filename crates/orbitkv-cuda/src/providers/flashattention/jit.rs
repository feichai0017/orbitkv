//! Pinned FlashAttention-3/CUTLASS source and native C ABI compilation.

use crate::{
    providers::{build::NativeBuild, registry::ProviderId},
    target::CudaTarget,
};

use std::{
    collections::HashMap,
    ffi::{CStr, c_char, c_void},
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
};

use crate::providers::provider_source::{ResolvedProviderSource, content_digest};

const WRAPPER: &str = include_str!("wrapper.cu");
const HEADER: &str = include_str!("wrapper.h");

fn resolved_source() -> Result<&'static ResolvedProviderSource, String> {
    static SOURCE: OnceLock<Result<ResolvedProviderSource, String>> = OnceLock::new();
    SOURCE
        .get_or_init(|| ProviderId::FlashAttention.source()?.resolve_identity())
        .as_ref()
        .map_err(Clone::clone)
}

pub(super) fn provider_identity() -> Result<String, String> {
    resolved_source().map(|source| {
        format!(
            "flashattention@sha256:{}",
            content_digest(&[
                source.digest.as_bytes(),
                include_str!("plan.rs").as_bytes(),
                include_str!("paged_attention.egg").as_bytes(),
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
        let _stage = tracing::info_span!(target: "orbitkv::stage", "cuda.provider.load", provider = "flashattention", path = %path.display()).entered();
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

pub(super) fn ensure_compiled(
    target: CudaTarget,
    config: Config,
) -> anyhow::Result<&'static Library> {
    static LIBRARIES: OnceLock<Mutex<HashMap<(CudaTarget, Config), &'static Library>>> =
        OnceLock::new();
    let mut libraries = LIBRARIES.get_or_init(Default::default).lock().unwrap();
    if let Some(library) = libraries.get(&(target, config)) {
        return Ok(library);
    }
    let path = compile_or_cache(target, config)?;
    let library = Box::leak(Box::new(unsafe { Library::load(&path)? }));
    libraries.insert((target, config), library);
    Ok(library)
}

fn compile_or_cache(target: CudaTarget, config: Config) -> anyhow::Result<PathBuf> {
    let architecture = target.hopper_architecture()?;
    let provider = resolved_source().map_err(anyhow::Error::msg)?;
    let source = WRAPPER.replace("#include \"wrapper.h\"", HEADER);
    let mut arguments = [
        "-shared",
        "-std=c++17",
        "-O3",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-Xcompiler=-fPIC",
    ]
    .map(str::to_owned)
    .to_vec();
    arguments.extend([
        format!("-arch={architecture}"),
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
    NativeBuild {
        provider: ProviderId::FlashAttention,
        sources: provider,
        source: &source,
        arguments,
        link_arguments: vec!["-lcuda".into(), "-lcudart".into()],
    }
    .compile()
}
