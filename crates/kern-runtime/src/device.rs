//! Device allocations: pool allocations, virtual-memory mappings that
//! other ranks can import, the chunk arenas behind the pooled states, and
//! the pinned host blocks and timing events the streams work with.

use std::os::raw::c_void;
use std::sync::Arc;

use cudarc::driver::{sys, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceSlice, SyncOnDrop};

use crate::error::{bail, cuda_check, Error, Result};
use crate::host_weights::HostWeight;

/// What a peer maps to reach one of this rank's allocations. A rank in
/// another process — on this tray or across the NVL72 fabric — gets a
/// fabric handle; a rank in this process gets the allocation handle
/// itself, which `cuMemMap` takes directly, so a device without fabric
/// support (an HGX B300, a tray without an IMEX channel) still shares
/// every buffer with the ranks it loads beside. Which one a device hands
/// out is decided at allocation ([`alloc_vmm`]). `bytes` is the mapped
/// size (the requested size rounded up to the allocation granularity),
/// which the importer must map in full.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PeerHandle {
    Fabric {
        fabric: [u8; 64],
        bytes: u64,
    },
    /// Meaningful in the exporting process only.
    Local {
        handle: sys::CUmemGenericAllocationHandle,
        bytes: u64,
    },
}

impl PeerHandle {
    pub const BYTES: usize = 72;

    pub(crate) fn bytes(&self) -> u64 {
        match self {
            Self::Fabric { bytes, .. } | Self::Local { bytes, .. } => *bytes,
        }
    }

    /// Wire form of a fabric handle, what a caller carries to another
    /// process: 64 bytes of handle, then the mapped size, little-endian.
    /// `None` for a local handle, which no other process can map.
    pub fn to_bytes(&self) -> Option<[u8; Self::BYTES]> {
        let Self::Fabric { fabric, bytes } = self else { return None };
        let mut out = [0u8; Self::BYTES];
        out[..64].copy_from_slice(fabric);
        out[64..].copy_from_slice(&bytes.to_le_bytes());
        Some(out)
    }

    pub fn from_bytes(b: &[u8]) -> Option<PeerHandle> {
        if b.len() != Self::BYTES {
            return None;
        }
        Some(Self::Fabric { fabric: b[..64].try_into().ok()?, bytes: u64::from_le_bytes(b[64..].try_into().ok()?) })
    }
}

impl std::fmt::Debug for PeerHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fabric { fabric, bytes } => write!(
                f,
                "PeerHandle::Fabric({bytes} bytes, {:02x}{:02x}{:02x}{:02x}…)",
                fabric[0], fabric[1], fabric[2], fabric[3]
            ),
            Self::Local { handle, bytes } => write!(f, "PeerHandle::Local({bytes} bytes, handle {handle:#x})"),
        }
    }
}

/// One device allocation. `ptr`/`bytes` are what the compiled programs
/// bake in; the backing decides whether a peer can map it.
pub(crate) struct DeviceBuf {
    pub(crate) ptr: u64,
    /// Bytes declared (a pooled state: the initial layout's).
    pub(crate) bytes: u64,
    /// Bytes addressable: `bytes`, or a pooled state's whole reservation.
    span: u64,
    stream: Arc<CudaStream>,
    backing: Backing,
}

enum Backing {
    /// Immutable portable host weight, shared across local runtimes.
    Host(Arc<HostWeight>),
    /// `cuMemAlloc` through cudarc: local only.
    Pool(#[allow(dead_code)] CudaSlice<u8>),
    /// `cuMemCreate` + reserve + map: every rank can map it, or a peer's
    /// allocation mapped into this address space.
    Vmm(Vmm),
    /// A pooled state's arena, owned by the remap thread's [`Mapper`].
    Reserved,
}

/// A physical allocation mapped at a reserved address; unmapped, freed and
/// released in that order on drop. A mapping of a peer's local handle does
/// not own it: the peer's own `Vmm` releases it.
struct Vmm {
    handle: sys::CUmemGenericAllocationHandle,
    va: sys::CUdeviceptr,
    size: usize,
    /// Created with a fabric handle, so it exports as one.
    fabric: bool,
    owns_handle: bool,
}

impl Drop for Vmm {
    fn drop(&mut self) {
        unsafe {
            sys::cuMemUnmap(self.va, self.size);
            sys::cuMemAddressFree(self.va, self.size);
            if self.owns_handle {
                sys::cuMemRelease(self.handle);
            }
        }
    }
}

/// Pool allocation, zeroed on the stream.
pub(crate) fn alloc(stream: &Arc<CudaStream>, bytes: u64) -> Result<DeviceBuf> {
    let slice = stream.alloc_zeros::<u8>(bytes.max(1) as usize)?;
    let ptr = {
        let (p, _sync) = slice.device_ptr(stream);
        p
    };
    Ok(DeviceBuf { ptr, bytes, span: bytes, stream: stream.clone(), backing: Backing::Pool(slice) })
}

/// A pool allocation nobody has written: for scratch that is filled
/// before it is read, so the zeroing pass is not paid.
pub(crate) fn alloc_uninit(stream: &Arc<CudaStream>, bytes: u64) -> Result<DeviceBuf> {
    let slice = unsafe { stream.alloc::<u8>(bytes.max(1) as usize) }?;
    let ptr = {
        let (p, _sync) = slice.device_ptr(stream);
        p
    };
    Ok(DeviceBuf { ptr, bytes, span: bytes, stream: stream.clone(), backing: Backing::Pool(slice) })
}

/// Stable alias of an immutable host weight in this stream's CUDA context.
pub(crate) fn alloc_host(stream: &Arc<CudaStream>, weight: Arc<HostWeight>) -> Result<DeviceBuf> {
    stream.context().bind_to_thread()?;
    let ptr = weight.device_pointer()?;
    let bytes = weight.bytes;
    Ok(DeviceBuf { ptr, bytes, span: bytes, stream: stream.clone(), backing: Backing::Host(weight) })
}

fn fabric_handle_type() -> sys::CUmemAllocationHandleType {
    sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_FABRIC
}

fn none_handle_type() -> sys::CUmemAllocationHandleType {
    sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE
}

/// Whether the device hands out fabric handles (`cuMemCreate` with
/// `CU_MEM_HANDLE_TYPE_FABRIC`): the attribute says so, and
/// `KERN_NO_FABRIC` is unset. The variable is a gate's way of running the
/// local-handle path on a tray that has fabric support.
fn fabric_supported(dev: i32) -> Result<bool> {
    if std::env::var_os("KERN_NO_FABRIC").is_some() {
        return Ok(false);
    }
    let mut v: i32 = 0;
    cuda_check(
        unsafe {
            sys::cuDeviceGetAttribute(
                &mut v,
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
                dev,
            )
        },
        "cuDeviceGetAttribute(HANDLE_TYPE_FABRIC_SUPPORTED)",
    )?;
    Ok(v != 0)
}

fn alloc_prop(dev: i32, handle_type: sys::CUmemAllocationHandleType) -> sys::CUmemAllocationProp {
    // Zeroed, then the fields that matter: the struct grows across CUDA
    // versions and cudarc's bindings follow the one it was built against.
    let mut prop: sys::CUmemAllocationProp = unsafe { std::mem::zeroed() };
    prop.type_ = sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.requestedHandleTypes = handle_type;
    prop.location.type_ = sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = dev;
    prop
}

fn granularity(prop: &sys::CUmemAllocationProp) -> Result<usize> {
    let mut g: usize = 0;
    cuda_check(
        unsafe {
            sys::cuMemGetAllocationGranularity(
                &mut g,
                prop,
                sys::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_MINIMUM,
            )
        },
        "cuMemGetAllocationGranularity",
    )?;
    Ok(g.max(1))
}

/// Reserve `size` bytes of address space, map `handle` there and grant
/// this device read/write access. On failure nothing leaks: the handle is
/// the caller's to release.
fn map_handle(
    dev: i32,
    handle: sys::CUmemGenericAllocationHandle,
    size: usize,
    align: usize,
) -> Result<sys::CUdeviceptr> {
    let mut va: sys::CUdeviceptr = 0;
    cuda_check(unsafe { sys::cuMemAddressReserve(&mut va, size, align, 0, 0) }, "cuMemAddressReserve")?;
    if let Err(e) = cuda_check(unsafe { sys::cuMemMap(va, size, 0, handle, 0) }, "cuMemMap") {
        unsafe { sys::cuMemAddressFree(va, size) };
        return Err(e);
    }
    let mut access: sys::CUmemAccessDesc = unsafe { std::mem::zeroed() };
    access.location.type_ = sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = dev;
    access.flags = sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if let Err(e) = cuda_check(unsafe { sys::cuMemSetAccess(va, size, &access, 1) }, "cuMemSetAccess") {
        unsafe {
            sys::cuMemUnmap(va, size);
            sys::cuMemAddressFree(va, size);
        }
        return Err(e);
    }
    Ok(va)
}

/// Virtual-memory allocation on `dev`, zeroed on the stream. Every rank
/// can map it: with a fabric handle when the device offers one, through
/// the allocation handle itself otherwise (see [`PeerHandle`]).
pub(crate) fn alloc_vmm(stream: &Arc<CudaStream>, dev: i32, bytes: u64, what: &str) -> Result<DeviceBuf> {
    let fabric = fabric_supported(dev)?;
    let prop = alloc_prop(dev, if fabric { fabric_handle_type() } else { none_handle_type() });
    let g = granularity(&prop)?;
    let size = (bytes.max(1) as usize).div_ceil(g) * g;
    let mut handle: sys::CUmemGenericAllocationHandle = 0;
    let created = unsafe { sys::cuMemCreate(&mut handle, size, &prop, 0) };
    let fabric = match (created, fabric) {
        (sys::CUresult::CUDA_SUCCESS, fabric) => fabric,
        // The attribute says fabric, the driver says no (no IMEX channel,
        // say): the ranks in this process map the handle itself.
        (r, true) => {
            tracing::warn!("{what}: cuMemCreate with a fabric handle failed ({r:?}); allocating without one");
            let prop = alloc_prop(dev, none_handle_type());
            cuda_check(unsafe { sys::cuMemCreate(&mut handle, size, &prop, 0) }, "cuMemCreate")?;
            false
        }
        (r, false) => return Err(Error::Cuda(format!("{what}: cuMemCreate({size} bytes): {r:?}"))),
    };
    let va = match map_handle(dev, handle, size, g) {
        Ok(va) => va,
        Err(e) => {
            unsafe { sys::cuMemRelease(handle) };
            return Err(e);
        }
    };
    let vmm = Vmm { handle, va, size, fabric, owns_handle: true };
    cuda_check(unsafe { sys::cuMemsetD8Async(va, 0, size, stream.cu_stream()) }, "cuMemsetD8Async")?;
    Ok(DeviceBuf { ptr: va, bytes, span: bytes, stream: stream.clone(), backing: Backing::Vmm(vmm) })
}

/// Map a peer's allocation into this device's address space. The mapping
/// is a [`DeviceBuf`] so it lives exactly as long as the pointers derived
/// from it.
pub(crate) fn import(stream: &Arc<CudaStream>, dev: i32, h: &PeerHandle, what: &str) -> Result<DeviceBuf> {
    let t0 = std::time::Instant::now();
    let (handle, owns_handle) = match h {
        PeerHandle::Fabric { fabric, .. } => {
            let mut fh = sys::CUmemFabricHandle { data: *fabric };
            let mut handle: sys::CUmemGenericAllocationHandle = 0;
            cuda_check(
                unsafe {
                    sys::cuMemImportFromShareableHandle(
                        &mut handle,
                        &mut fh as *mut _ as *mut c_void,
                        fabric_handle_type(),
                    )
                },
                &format!("{what}: cuMemImportFromShareableHandle"),
            )?;
            (handle, true)
        }
        PeerHandle::Local { handle, .. } => (*handle, false),
    };
    let release = |handle| {
        if owns_handle {
            unsafe { sys::cuMemRelease(handle) };
        }
    };
    let g = granularity(&alloc_prop(dev, none_handle_type()))?;
    let size = h.bytes() as usize;
    if size == 0 || !size.is_multiple_of(g) {
        release(handle);
        bail!(Cuda, "{what}: peer handle maps {size} bytes, not a multiple of the {g}-byte granularity");
    }
    let imported = t0.elapsed();
    let va = match map_handle(dev, handle, size, g) {
        Ok(va) => va,
        Err(e) => {
            release(handle);
            return Err(Error::Cuda(format!("{what}: {e}")));
        }
    };
    tracing::debug!("{what}: {} GiB imported in {imported:?}, mapped in {:?}", size >> 30, t0.elapsed() - imported);
    let vmm = Vmm { handle, va, size, fabric: false, owns_handle };
    Ok(DeviceBuf { ptr: va, bytes: h.bytes(), span: h.bytes(), stream: stream.clone(), backing: Backing::Vmm(vmm) })
}

impl DeviceBuf {
    pub(crate) fn host_weight(&self) -> Option<&Arc<HostWeight>> {
        match &self.backing {
            Backing::Host(weight) => Some(weight),
            _ => None,
        }
    }

    /// The same allocation, its views ordered on `stream` from now on.
    pub(crate) fn on(mut self, stream: &Arc<CudaStream>) -> DeviceBuf {
        self.stream = stream.clone();
        self
    }

    /// A pooled state: `bytes` of its initial layout at `ptr`, `span`
    /// bytes reserved there; the arena behind it is the [`Mapper`]'s.
    pub(crate) fn reserved(stream: &Arc<CudaStream>, ptr: u64, bytes: u64, span: u64) -> DeviceBuf {
        DeviceBuf { ptr, bytes, span, stream: stream.clone(), backing: Backing::Reserved }
    }

    /// What a peer maps to reach this allocation: a fabric handle when it
    /// was created with one, the allocation handle otherwise; nothing for
    /// pool memory, host memory and a pooled state's arena.
    pub(crate) fn export(&self) -> Result<Option<PeerHandle>> {
        let Backing::Vmm(v) = &self.backing else { return Ok(None) };
        let bytes = v.size as u64;
        if !v.fabric {
            return Ok(Some(PeerHandle::Local { handle: v.handle, bytes }));
        }
        let mut fh = sys::CUmemFabricHandle { data: [0; 64] };
        cuda_check(
            unsafe {
                sys::cuMemExportToShareableHandle(&mut fh as *mut _ as *mut c_void, v.handle, fabric_handle_type(), 0)
            },
            "cuMemExportToShareableHandle",
        )?;
        Ok(Some(PeerHandle::Fabric { fabric: fh.data, bytes }))
    }

    /// Bytes addressable from `ptr`: a pooled state's whole reservation,
    /// so a pointer or tensormap taken at load stays valid for every page
    /// and slot a remap makes later.
    pub(crate) fn span(&self) -> u64 {
        self.span
    }

    /// A byte range of the allocation, for the cudarc copy/memset entry
    /// points.
    pub(crate) fn view(&self, range: std::ops::Range<usize>) -> Result<BufView> {
        if range.start > range.end || range.end as u64 > self.span {
            bail!(Api, "byte range [{}, {}) outside the {}-byte allocation", range.start, range.end, self.span);
        }
        Ok(BufView { ptr: self.ptr + range.start as u64, len: range.end - range.start, stream: self.stream.clone() })
    }
}

/// Page-locked host memory: the host tier's block, allocated once.
pub(crate) struct Pinned {
    ptr: *mut c_void,
    bytes: u64,
}

// Only ever touched through copies enqueued from the runtime's thread.
unsafe impl Send for Pinned {}
unsafe impl Sync for Pinned {}

impl Pinned {
    /// `bytes` of pinned memory on the NUMA node `dev` hangs off (a GB300
    /// tray has two Grace CPUs, two GPUs each: local DRAM copies at ~180
    /// GiB/s per direction, the other socket's at ~115); anywhere when the
    /// node cannot be told.
    pub(crate) fn alloc(bytes: u64, dev: i32) -> Result<Pinned> {
        let node = numa_node(dev);
        if let Some(n) = node {
            mempolicy(libc::MPOL_BIND, Some(n));
        }
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let r = cuda_check(unsafe { sys::cuMemHostAlloc(&mut ptr, bytes.max(1) as usize, 0) }, "cuMemHostAlloc");
        if node.is_some() {
            mempolicy(libc::MPOL_DEFAULT, None);
        }
        r?;
        Ok(Pinned { ptr, bytes })
    }

    pub(crate) fn ptr(&self) -> u64 {
        self.ptr as u64
    }

    pub(crate) fn bytes(&self) -> u64 {
        self.bytes
    }
}

impl Drop for Pinned {
    fn drop(&mut self) {
        unsafe { sys::cuMemFreeHost(self.ptr) };
    }
}

/// The NUMA node of `dev`'s PCI slot, as sysfs reports it.
fn numa_node(dev: i32) -> Option<u32> {
    let mut id = [0u8; 32];
    let r = unsafe { sys::cuDeviceGetPCIBusId(id.as_mut_ptr() as *mut std::ffi::c_char, id.len() as i32, dev) };
    if r != sys::CUresult::CUDA_SUCCESS {
        return None;
    }
    let id = std::ffi::CStr::from_bytes_until_nul(&id).ok()?.to_str().ok()?.to_ascii_lowercase();
    let node = std::fs::read_to_string(format!("/sys/bus/pci/devices/{id}/numa_node")).ok()?;
    node.trim().parse::<i32>().ok().filter(|&n| n >= 0).map(|n| n as u32)
}

/// This thread's memory policy: bound to one node, or the default.
fn mempolicy(mode: i32, node: Option<u32>) {
    let mask: [u64; 16] = node.map_or([0; 16], |n| {
        let mut m = [0u64; 16];
        m[(n / 64) as usize] |= 1 << (n % 64);
        m
    });
    let (ptr, bits) = match node {
        Some(_) => (mask.as_ptr(), mask.len() * 64),
        None => (std::ptr::null(), 0),
    };
    // Best effort: a refused policy only costs bandwidth.
    unsafe { libc::syscall(libc::SYS_set_mempolicy, mode, ptr, bits) };
}

/// `rows` rows of `width` bytes between device address `dev.0` (rows
/// `dev.1` apart) and host address `host.0` (rows `host.1` apart), on
/// `stream`, towards the host or the device: one copy-engine transfer
/// either way.
/// Which side of the bus an address is on, for the driver's 2D copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Space {
    Host,
    Device,
}

impl Space {
    fn memory_type(self) -> sys::CUmemorytype {
        match self {
            Self::Host => sys::CUmemorytype::CU_MEMORYTYPE_HOST,
            Self::Device => sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
        }
    }
}

/// `rows` rows of `width` bytes, `src.1` apart at `src.0`, landing `dst.1`
/// apart at `dst.0`; each side says which space it is in.
pub(crate) fn copy_2d(
    stream: sys::CUstream,
    dst: (u64, u64, Space),
    src: (u64, u64, Space),
    width: u64,
    rows: u64,
) -> Result<()> {
    let host = |ptr: u64, space: Space| if space == Space::Host { ptr as *mut c_void } else { std::ptr::null_mut() };
    let device = |ptr: u64, space: Space| if space == Space::Device { ptr } else { 0 };
    let c = sys::CUDA_MEMCPY2D {
        srcXInBytes: 0,
        srcY: 0,
        srcMemoryType: src.2.memory_type(),
        srcHost: host(src.0, src.2),
        srcDevice: device(src.0, src.2),
        srcArray: std::ptr::null_mut(),
        srcPitch: src.1 as usize,
        dstXInBytes: 0,
        dstY: 0,
        dstMemoryType: dst.2.memory_type(),
        dstHost: host(dst.0, dst.2),
        dstDevice: device(dst.0, dst.2),
        dstArray: std::ptr::null_mut(),
        dstPitch: dst.1 as usize,
        WidthInBytes: width as usize,
        Height: rows as usize,
    };
    cuda_check(unsafe { sys::cuMemcpy2DAsync_v2(&c, stream) }, "cuMemcpy2DAsync")
}

/// One contiguous device-to-device copy on the stream.
pub(crate) fn copy_1d(stream: sys::CUstream, dst: u64, src: u64, bytes: u64) -> Result<()> {
    cuda_check(unsafe { sys::cuMemcpyAsync(dst, src, bytes as usize, stream) }, "cuMemcpyAsync")
}

/// Another process's allocation mapped into this context: what a weight
/// cache's bucket becomes once [`crate::Runtime::map`] imports its handle.
/// Unmapped and released on drop; the owner's allocation outlives it.
pub struct Mapped(DeviceBuf);

impl Mapped {
    pub(crate) fn new(buf: DeviceBuf) -> Self {
        Self(buf)
    }

    pub fn ptr(&self) -> u64 {
        self.0.ptr
    }

    pub fn bytes(&self) -> u64 {
        self.0.bytes
    }
}

/// The device's UUID the way the driver's tools print it
/// (`GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`), which a weight cache
/// daemon keys its per-GPU endpoint by.
pub fn device_uuid(gpu: usize) -> Result<String> {
    use cudarc::driver::result;
    result::init()?;
    let uuid = result::device::get_uuid(result::device::get(gpu as i32)?)?;
    let hex: Vec<String> = uuid.bytes.iter().map(|b| format!("{b:02x}")).collect();
    let (a, b, c, d, e) = (&hex[..4], &hex[4..6], &hex[6..8], &hex[8..10], &hex[10..]);
    Ok(format!("GPU-{}-{}-{}-{}-{}", a.concat(), b.concat(), c.concat(), d.concat(), e.concat()))
}

pub(crate) fn chunk_granularity(dev: i32) -> Result<usize> {
    granularity(&alloc_prop(dev, none_handle_type()))
}

/// A pooled state's reserved range: `positions` chunk positions of `chunk`
/// bytes, each mapped or not. Reserved once; the pointer the programs bake
/// in never moves.
pub(crate) struct Arena {
    va: sys::CUdeviceptr,
    chunk: usize,
    mapped: Vec<bool>,
    dev: i32,
}

impl Arena {
    pub(crate) fn reserve(dev: i32, chunk: usize, positions: usize) -> Result<Arena> {
        let mut va: sys::CUdeviceptr = 0;
        let size = chunk * positions.max(1);
        // Default alignment: the granularity, which `chunk` is a multiple
        // of (an alignment must be a power of two; a chunk need not be).
        cuda_check(unsafe { sys::cuMemAddressReserve(&mut va, size, 0, 0, 0) }, "cuMemAddressReserve")?;
        Ok(Arena { va, chunk, mapped: vec![false; positions], dev })
    }

    pub(crate) fn ptr(&self) -> u64 {
        self.va
    }

    fn at(&self, pos: usize) -> sys::CUdeviceptr {
        self.va + (pos * self.chunk) as u64
    }

    fn map(&mut self, pos: usize, handle: sys::CUmemGenericAllocationHandle) -> Result<()> {
        cuda_check(unsafe { sys::cuMemMap(self.at(pos), self.chunk, 0, handle, 0) }, "cuMemMap")?;
        self.mapped[pos] = true;
        Ok(())
    }

    fn unmap(&mut self, pos: usize) -> Result<()> {
        cuda_check(unsafe { sys::cuMemUnmap(self.at(pos), self.chunk) }, "cuMemUnmap")?;
        self.mapped[pos] = false;
        Ok(())
    }

    /// Grant this device read/write access over `positions`, all mapped.
    fn access(&self, positions: std::ops::Range<usize>) -> Result<()> {
        let mut access: sys::CUmemAccessDesc = unsafe { std::mem::zeroed() };
        access.location.type_ = sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = self.dev;
        access.flags = sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        cuda_check(
            unsafe { sys::cuMemSetAccess(self.at(positions.start), positions.len() * self.chunk, &access, 1) },
            "cuMemSetAccess",
        )
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        for pos in 0..self.mapped.len() {
            if self.mapped[pos] {
                unsafe { sys::cuMemUnmap(self.at(pos), self.chunk) };
            }
        }
        unsafe { sys::cuMemAddressFree(self.va, self.chunk * self.mapped.len().max(1)) };
    }
}

/// The physical chunks behind the pooled states: created once, released
/// once every arena has let go of them. No fabric handle: a pooled state
/// is not exportable.
pub(crate) struct Physical {
    handles: Vec<sys::CUmemGenericAllocationHandle>,
}

impl Physical {
    pub(crate) fn create(dev: i32, chunk: usize, count: usize) -> Result<Physical> {
        let prop = alloc_prop(dev, none_handle_type());
        let mut handles = Vec::with_capacity(count);
        for i in 0..count {
            let mut h: sys::CUmemGenericAllocationHandle = 0;
            if let Err(e) = cuda_check(unsafe { sys::cuMemCreate(&mut h, chunk, &prop, 0) }, "cuMemCreate") {
                drop(Physical { handles });
                return Err(Error::Cuda(format!("chunk {i} of {count} ({chunk} bytes): {e}")));
            }
            handles.push(h);
        }
        Ok(Physical { handles })
    }
}

impl Drop for Physical {
    fn drop(&mut self) {
        for &h in &self.handles {
            unsafe { sys::cuMemRelease(h) };
        }
    }
}

/// The shell of the pool's remaps: arenas in the pool's order over one set
/// of physical chunks. Arenas drop first, then the chunks.
pub(crate) struct Mapper {
    arenas: Vec<Arena>,
    #[allow(dead_code)]
    physical: Physical,
}

impl Mapper {
    pub(crate) fn new(arenas: Vec<Arena>, physical: Physical) -> Mapper {
        Mapper { arenas, physical }
    }

    /// Unmap, map, grant access — in that order, so a chunk a plan moves
    /// is off its old position before it is on its new one.
    pub(crate) fn run(&mut self, plan: &kern_pool::Remap) -> Result<()> {
        for &(a, p) in &plan.unmap {
            self.arenas[a].unmap(p)?;
        }
        for &(a, p, c) in &plan.map {
            let h = self.physical.handles[c as usize];
            self.arenas[a].map(p, h)?;
        }
        for (a, r) in plan.access_spans() {
            self.arenas[a].access(r)?;
        }
        Ok(())
    }
}

/// Synchronization is trivially correct for these raw views: the whole
/// runtime is single-stream.
impl DeviceSlice<u8> for DeviceBuf {
    fn len(&self) -> usize {
        self.bytes as usize
    }
    fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

impl DevicePtr<u8> for DeviceBuf {
    fn device_ptr<'a>(&'a self, _: &'a CudaStream) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        (self.ptr, SyncOnDrop::Record(None))
    }
}

impl DevicePtrMut<u8> for DeviceBuf {
    fn device_ptr_mut<'a>(&'a mut self, _: &'a CudaStream) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        (self.ptr, SyncOnDrop::Record(None))
    }
}

/// A byte range of a [`DeviceBuf`].
pub(crate) struct BufView {
    ptr: u64,
    len: usize,
    stream: Arc<CudaStream>,
}

impl BufView {
    pub(crate) fn ptr(&self) -> u64 {
        self.ptr
    }
}

impl DeviceSlice<u8> for BufView {
    fn len(&self) -> usize {
        self.len
    }
    fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

impl DevicePtr<u8> for BufView {
    fn device_ptr<'a>(&'a self, _: &'a CudaStream) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        (self.ptr, SyncOnDrop::Record(None))
    }
}

impl DevicePtrMut<u8> for BufView {
    fn device_ptr_mut<'a>(&'a mut self, _: &'a CudaStream) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        (self.ptr, SyncOnDrop::Record(None))
    }
}

/// A pool of timing events, for `kern test` and profiling.
pub(crate) struct Events(pub(crate) Vec<sys::CUevent>);

impl Events {
    pub(crate) fn new(n: usize) -> Result<Events> {
        let mut evs = Vec::with_capacity(n);
        for _ in 0..n {
            let mut ev: sys::CUevent = std::ptr::null_mut();
            cuda_check(unsafe { sys::cuEventCreate(&mut ev, 0) }, "cuEventCreate")?;
            evs.push(ev);
        }
        Ok(Events(evs))
    }

    pub(crate) fn record(&self, i: usize, stream: &CudaStream) -> Result<()> {
        cuda_check(unsafe { sys::cuEventRecord(self.0[i], stream.cu_stream()) }, "cuEventRecord")
    }

    pub(crate) fn elapsed_ms(&self, a: usize, b: usize) -> Result<f32> {
        let mut ms = 0f32;
        cuda_check(unsafe { sys::cuEventElapsedTime_v2(&mut ms, self.0[a], self.0[b]) }, "cuEventElapsedTime")?;
        Ok(ms)
    }
}

impl Drop for Events {
    fn drop(&mut self) {
        for ev in &self.0 {
            unsafe { sys::cuEventDestroy_v2(*ev) };
        }
    }
}

/// A fresh event recorded on `stream`; the caller destroys it.
pub(crate) fn record(stream: &CudaStream) -> Result<sys::CUevent> {
    let mut ev: sys::CUevent = std::ptr::null_mut();
    cuda_check(
        unsafe { sys::cuEventCreate(&mut ev, sys::CUevent_flags::CU_EVENT_DISABLE_TIMING as u32) },
        "cuEventCreate",
    )?;
    cuda_check(unsafe { sys::cuEventRecord(ev, stream.cu_stream()) }, "cuEventRecord")?;
    Ok(ev)
}

/// `stream` waits for `ev`, which is then destroyed.
pub(crate) fn wait_then_destroy(stream: &CudaStream, ev: sys::CUevent) -> Result<()> {
    let r = cuda_check(unsafe { sys::cuStreamWaitEvent(stream.cu_stream(), ev, 0) }, "cuStreamWaitEvent");
    unsafe { sys::cuEventDestroy_v2(ev) };
    r
}

/// Whether everything before `ev` on its stream has completed.
pub(crate) fn landed(ev: sys::CUevent) -> Result<bool> {
    match unsafe { sys::cuEventQuery(ev) } {
        sys::CUresult::CUDA_SUCCESS => Ok(true),
        sys::CUresult::CUDA_ERROR_NOT_READY => Ok(false),
        r => cuda_check(r, "cuEventQuery").map(|_| true),
    }
}
