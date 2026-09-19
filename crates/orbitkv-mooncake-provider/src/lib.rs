//! Runtime loader for the pinned upstream Mooncake Transfer Engine.
//!
//! The provider builds Mooncake from the pinned submodule. At runtime it loads
//! the shared libraries next to the final executable/Python extension, from an
//! explicit `ORBITKV_MOONCAKE_LIB_DIR`, or from Cargo's build output. This
//! keeps Mooncake's C++ dependency graph out of every Rust final link.

#![allow(
    clippy::missing_safety_doc,
    reason = "crate-private C ABI shims share the safety contract documented by orbitkv-transfer"
)]

use std::env;
use std::ffi::{CStr, c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use libloading::os::unix::{Library, RTLD_GLOBAL, RTLD_NOW};

pub const SOURCE_REVISION: &str = "ffe013517eaafa8f33e5e0ee034fd6b8f5561e92";
#[cfg(feature = "cuda")]
const WORKSPACE_RUNTIME_LIB_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../.orbitkv/mooncake/cuda/lib"
);
#[cfg(not(feature = "cuda"))]
const WORKSPACE_RUNTIME_LIB_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../.orbitkv/mooncake/cpu/lib"
);

pub type NativeEngine = *mut c_void;
pub type SegmentId = i32;
pub type BatchId = u64;

#[repr(C)]
pub struct TransferRequest {
    pub opcode: i32,
    pub source: *mut c_void,
    pub target_id: SegmentId,
    pub target_offset: u64,
    pub length: u64,
}

#[repr(C)]
pub struct TransferStatus {
    pub status: i32,
    pub transferred_bytes: u64,
}

#[repr(C)]
pub struct Notify {
    pub name: *mut c_char,
    pub message: *mut c_char,
}

type Create =
    unsafe extern "C" fn(*const c_char, *const c_char, *const c_char, u64, c_int) -> NativeEngine;
type Destroy = unsafe extern "C" fn(NativeEngine);
type LocalEndpoint = unsafe extern "C" fn(NativeEngine, *mut c_char, usize) -> c_int;
type OpenSegment = unsafe extern "C" fn(NativeEngine, *const c_char) -> SegmentId;
type CloseSegment = unsafe extern "C" fn(NativeEngine, SegmentId) -> c_int;
type RegisterMemory =
    unsafe extern "C" fn(NativeEngine, *mut c_void, usize, *const c_char, c_int) -> c_int;
type UnregisterMemory = unsafe extern "C" fn(NativeEngine, *mut c_void) -> c_int;
type AllocateBatch = unsafe extern "C" fn(NativeEngine, usize) -> BatchId;
type Submit = unsafe extern "C" fn(NativeEngine, BatchId, *mut TransferRequest, usize) -> c_int;
type SubmitWithNotify =
    unsafe extern "C" fn(NativeEngine, BatchId, *mut TransferRequest, usize, Notify) -> c_int;
type TransferStatusFn =
    unsafe extern "C" fn(NativeEngine, BatchId, usize, *mut TransferStatus) -> c_int;
type FreeBatch = unsafe extern "C" fn(NativeEngine, BatchId) -> c_int;
type TakeNotifies = unsafe extern "C" fn(NativeEngine, *mut c_int) -> *mut Notify;
type FreeNotifies = unsafe extern "C" fn(*mut Notify, c_int) -> c_int;
type SendNotify = unsafe extern "C" fn(NativeEngine, u64, Notify) -> c_int;

struct Api {
    _dependencies: Vec<Library>,
    _library: Library,
    create: Create,
    destroy: Destroy,
    local_endpoint: LocalEndpoint,
    open_segment: OpenSegment,
    close_segment: CloseSegment,
    register_memory: RegisterMemory,
    unregister_memory: UnregisterMemory,
    allocate_batch: AllocateBatch,
    submit: Submit,
    submit_with_notify: SubmitWithNotify,
    transfer_status: TransferStatusFn,
    free_batch: FreeBatch,
    take_notifies: TakeNotifies,
    free_notifies: FreeNotifies,
    send_notify: SendNotify,
}

static API: LazyLock<Result<Api, String>> = LazyLock::new(load_api);

pub fn ensure_loaded() -> Result<(), String> {
    API.as_ref().map(|_| ()).map_err(Clone::clone)
}

fn api() -> &'static Api {
    API.as_ref()
        .unwrap_or_else(|error| panic!("Mooncake runtime unavailable: {error}"))
}

fn load_api() -> Result<Api, String> {
    let mut attempted = Vec::new();
    for directory in candidate_directories() {
        match unsafe { load_api_from(&directory) } {
            Ok(api) => return Ok(api),
            Err(error) => attempted.push(format!("{} ({error})", directory.display())),
        }
    }
    Err(format!(
        "could not load libtransfer_engine.so; searched {}",
        attempted.join(", "),
    ))
}

fn candidate_directories() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(path) = env::var_os("ORBITKV_MOONCAKE_LIB_DIR") {
        candidates.push(PathBuf::from(path));
    }
    if let Some(path) = module_directory() {
        candidates.push(path);
    }
    if let Ok(executable) = env::current_exe()
        && let Some(parent) = executable.parent()
    {
        candidates.push(parent.to_path_buf());
    }
    candidates.push(PathBuf::from(WORKSPACE_RUNTIME_LIB_DIR));
    candidates.dedup();
    candidates
}

#[inline(never)]
fn module_directory() -> Option<PathBuf> {
    let mut info = libc::Dl_info {
        dli_fname: std::ptr::null(),
        dli_fbase: std::ptr::null_mut(),
        dli_sname: std::ptr::null(),
        dli_saddr: std::ptr::null_mut(),
    };
    let found = unsafe { libc::dladdr(module_directory as *const () as *const c_void, &mut info) };
    if found == 0 || info.dli_fname.is_null() {
        return None;
    }
    let path = unsafe { CStr::from_ptr(info.dli_fname) };
    Path::new(path.to_str().ok()?)
        .parent()
        .map(Path::to_path_buf)
}

unsafe fn load_api_from(directory: &Path) -> Result<Api, String> {
    let flags = RTLD_NOW | RTLD_GLOBAL;
    let asio = unsafe { Library::open(Some(directory.join("libasio.so")), flags) }
        .map_err(|error| error.to_string())?;
    let common = unsafe { Library::open(Some(directory.join("libmooncake_common.so")), flags) }
        .map_err(|error| error.to_string())?;
    let library = unsafe { Library::open(Some(directory.join("libtransfer_engine.so")), flags) }
        .map_err(|error| error.to_string())?;

    Ok(Api {
        create: unsafe { symbol(&library, b"createTransferEngine\0")? },
        destroy: unsafe { symbol(&library, b"destroyTransferEngine\0")? },
        local_endpoint: unsafe { symbol(&library, b"getLocalIpAndPort\0")? },
        open_segment: unsafe { symbol(&library, b"openSegment\0")? },
        close_segment: unsafe { symbol(&library, b"closeSegment\0")? },
        register_memory: unsafe { symbol(&library, b"registerLocalMemory\0")? },
        unregister_memory: unsafe { symbol(&library, b"unregisterLocalMemory\0")? },
        allocate_batch: unsafe { symbol(&library, b"allocateBatchID\0")? },
        submit: unsafe { symbol(&library, b"submitTransfer\0")? },
        submit_with_notify: unsafe { symbol(&library, b"submitTransferWithNotify\0")? },
        transfer_status: unsafe { symbol(&library, b"getTransferStatus\0")? },
        free_batch: unsafe { symbol(&library, b"freeBatchID\0")? },
        take_notifies: unsafe { symbol(&library, b"getNotifsFromEngine\0")? },
        free_notifies: unsafe { symbol(&library, b"freeNotifsMsgBuf\0")? },
        send_notify: unsafe { symbol(&library, b"genNotifyInEngine\0")? },
        _dependencies: vec![asio, common],
        _library: library,
    })
}

unsafe fn symbol<T: Copy>(library: &Library, name: &[u8]) -> Result<T, String> {
    unsafe { library.get::<T>(name) }
        .map(|symbol| *symbol)
        .map_err(|error| format!("missing Mooncake symbol: {error}"))
}

pub unsafe fn create(
    metadata: *const c_char,
    local_server_name: *const c_char,
    bind_host: *const c_char,
    rpc_port: u64,
) -> NativeEngine {
    unsafe { (api().create)(metadata, local_server_name, bind_host, rpc_port, 1) }
}

pub unsafe fn destroy(engine: NativeEngine) {
    unsafe { (api().destroy)(engine) };
}

pub unsafe fn local_ip_and_port(engine: NativeEngine, output: *mut c_char, len: usize) -> c_int {
    unsafe { (api().local_endpoint)(engine, output, len) }
}

pub unsafe fn open_segment(engine: NativeEngine, name: *const c_char) -> SegmentId {
    unsafe { (api().open_segment)(engine, name) }
}

pub unsafe fn close_segment(engine: NativeEngine, segment: SegmentId) -> c_int {
    unsafe { (api().close_segment)(engine, segment) }
}

pub unsafe fn register_memory(
    engine: NativeEngine,
    address: *mut c_void,
    length: usize,
    location: *const c_char,
) -> c_int {
    unsafe { (api().register_memory)(engine, address, length, location, 1) }
}

pub unsafe fn unregister_memory(engine: NativeEngine, address: *mut c_void) -> c_int {
    unsafe { (api().unregister_memory)(engine, address) }
}

pub unsafe fn allocate_batch(engine: NativeEngine, batch_size: usize) -> BatchId {
    unsafe { (api().allocate_batch)(engine, batch_size) }
}

pub unsafe fn submit(
    engine: NativeEngine,
    batch: BatchId,
    requests: &mut [TransferRequest],
) -> c_int {
    unsafe { (api().submit)(engine, batch, requests.as_mut_ptr(), requests.len()) }
}

pub unsafe fn submit_with_notify(
    engine: NativeEngine,
    batch: BatchId,
    requests: &mut [TransferRequest],
    notify: Notify,
) -> c_int {
    unsafe {
        (api().submit_with_notify)(engine, batch, requests.as_mut_ptr(), requests.len(), notify)
    }
}

pub unsafe fn transfer_status(
    engine: NativeEngine,
    batch: BatchId,
    task: usize,
    status: &mut TransferStatus,
) -> c_int {
    unsafe { (api().transfer_status)(engine, batch, task, status) }
}

pub unsafe fn free_batch(engine: NativeEngine, batch: BatchId) -> c_int {
    unsafe { (api().free_batch)(engine, batch) }
}

pub unsafe fn take_notifies(engine: NativeEngine, count: &mut c_int) -> *mut Notify {
    unsafe { (api().take_notifies)(engine, count) }
}

pub unsafe fn free_notifies(messages: *mut Notify, count: c_int) -> c_int {
    unsafe { (api().free_notifies)(messages, count) }
}

pub unsafe fn notify(engine: NativeEngine, target: SegmentId, notify: Notify) -> c_int {
    unsafe { (api().send_notify)(engine, target as u64, notify) }
}
