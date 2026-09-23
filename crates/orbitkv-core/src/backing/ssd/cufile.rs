//! Optional cuFile reads and writes through bounded, registered GPU staging.
//!
//! The library is process-owned; ordinary io_uring deployments never load it.
//! cuFile may use its CPU compatibility path. Selecting this backend alone is
//! not evidence of native GDS; qualification must disable compatibility mode.

use std::ffi::c_void;
use std::fs::File;
use std::os::fd::AsRawFd;
use std::ptr::NonNull;
use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaStream, result};
use libloading::Library;

use super::GpuIo;
use crate::metrics::core_metrics;

pub(crate) const ALIGNMENT: usize = 4096;
pub(crate) const STAGING_BYTES: usize = 8 * 1024 * 1024;

#[repr(C)]
#[derive(Clone, Copy)]
struct Status {
    error: i32,
    cuda_error: i32,
}

impl Status {
    fn check(self, operation: &str) -> Result<(), String> {
        if self.error == 0 {
            Ok(())
        } else {
            Err(format!(
                "{operation}: cuFile error {}, CUDA error {}",
                self.error, self.cuda_error
            ))
        }
    }
}

#[repr(C)]
union FileHandle {
    fd: i32,
    handle: *mut c_void,
}

#[repr(C)]
struct Descriptor {
    kind: i32,
    handle: FileHandle,
    fs_ops: *const c_void,
}

type RegisterFile = unsafe extern "C" fn(*mut *mut c_void, *mut Descriptor) -> Status;
type DeregisterFile = unsafe extern "C" fn(*mut c_void);
type RegisterBuffer = unsafe extern "C" fn(*const c_void, usize, i32) -> Status;
type DeregisterBuffer = unsafe extern "C" fn(*const c_void) -> Status;
type Write = unsafe extern "C" fn(*mut c_void, *const c_void, usize, i64, i64) -> isize;
type Read = unsafe extern "C" fn(*mut c_void, *mut c_void, usize, i64, i64) -> isize;

pub(crate) struct Cufile {
    _library: Library,
    native_only: bool,
    register_file: RegisterFile,
    deregister_file: DeregisterFile,
    register_buffer: RegisterBuffer,
    deregister_buffer: DeregisterBuffer,
    read: Read,
    write: Write,
}

impl Cufile {
    fn get(native_only: bool) -> Result<Arc<Self>, String> {
        static DRIVER: OnceLock<Result<Arc<Cufile>, String>> = OnceLock::new();
        let driver = DRIVER
            .get_or_init(|| Self::load(native_only).map(Arc::new))
            .clone()?;
        if native_only && !driver.native_only {
            return Err("cuFile was already initialized without native-only configuration".into());
        }
        Ok(driver)
    }

    fn load(native_only: bool) -> Result<Self, String> {
        // SAFETY: public cuFile C ABI; the library outlives every copied symbol.
        unsafe {
            let library = Library::new("libcufile.so.0")
                .or_else(|_| Library::new("libcufile.so"))
                .map_err(|e| {
                    format!("cuFile unavailable: {e}; install NVIDIA GDS or select uring")
                })?;
            let open = *library
                .get::<unsafe extern "C" fn() -> Status>(b"cuFileDriverOpen\0")
                .map_err(|e| e.to_string())?;
            if native_only {
                // cuFile parameters are staged before DriverOpen. Avoid mutating
                // process environment after CUDA/Python worker threads exist.
                let set = *library
                    .get::<unsafe extern "C" fn(i32, bool) -> Status>(b"cuFileSetParameterBool\0")
                    .map_err(|e| format!("cuFile native-only configuration unavailable: {e}"))?;
                // CUFILE_PARAM_PROPERTIES_ALLOW_COMPAT_MODE / FORCE_COMPAT_MODE.
                set(1, false).check("disable cuFile compatibility")?;
                set(2, false).check("disable forced cuFile compatibility")?;
            }
            let driver = Self {
                native_only,
                register_file: *library
                    .get(b"cuFileHandleRegister\0")
                    .map_err(|e| e.to_string())?,
                deregister_file: *library
                    .get(b"cuFileHandleDeregister\0")
                    .map_err(|e| e.to_string())?,
                register_buffer: *library
                    .get(b"cuFileBufRegister\0")
                    .map_err(|e| e.to_string())?,
                deregister_buffer: *library
                    .get(b"cuFileBufDeregister\0")
                    .map_err(|e| e.to_string())?,
                read: *library.get(b"cuFileRead\0").map_err(|e| e.to_string())?,
                write: *library.get(b"cuFileWrite\0").map_err(|e| e.to_string())?,
                _library: library,
            };
            open().check("cuFileDriverOpen")?;
            log::info!(
                "SSD cuFile backend loaded (native_only={native_only}); qualify the hardware I/O path separately"
            );
            Ok(driver)
        }
    }
}

pub(crate) struct CufileFile {
    driver: Arc<Cufile>,
    handle: NonNull<c_void>,
    _file: File,
    pub(crate) gpu_io: Arc<GpuIo>,
}

// SAFETY: cuFile's file APIs are thread-safe. The handle and fd stay live until
// every SSD query/worker reference has been released.
unsafe impl Send for CufileFile {}
unsafe impl Sync for CufileFile {}

impl CufileFile {
    pub(crate) fn new(file: File, gpu_io: Arc<GpuIo>) -> Result<Self, String> {
        if gpu_io.automatic {
            let mut stat = std::mem::MaybeUninit::<libc::statfs>::uninit();
            // SAFETY: live descriptor and writable statfs output.
            if unsafe { libc::fstatfs(file.as_raw_fd(), stat.as_mut_ptr()) } != 0 {
                return Err(std::io::Error::last_os_error().to_string());
            }
            let stat = unsafe { stat.assume_init() };
            if !matches!(stat.f_type, libc::EXT4_SUPER_MAGIC | libc::XFS_SUPER_MAGIC) {
                return Err("automatic cuFile selection requires an ext4 or XFS mount".into());
            }
        }
        let driver = Cufile::get(gpu_io.automatic)?;
        // Initialize the complete C descriptor, including unused union bytes.
        let mut descriptor: Descriptor = unsafe { std::mem::zeroed() };
        descriptor.kind = 1;
        descriptor.handle.fd = file.as_raw_fd();
        let mut handle = std::ptr::null_mut();
        // SAFETY: descriptor refers to our owned file; outputs are valid stack storage.
        unsafe { (driver.register_file)(&raw mut handle, &raw mut descriptor) }
            .check("cuFileHandleRegister")?;
        let handle = NonNull::new(handle).ok_or("cuFile returned a null file handle")?;
        Ok(Self {
            driver,
            handle,
            _file: file,
            gpu_io,
        })
    }
}

impl Drop for CufileFile {
    fn drop(&mut self) {
        // SAFETY: all I/O borrows self and completes before the last owner drops.
        unsafe { (self.driver.deregister_file)(self.handle.as_ptr()) };
    }
}

/// One file interval and the corresponding engine-owned GPU range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CopyRange {
    pub file_offset: u64,
    pub device: u64,
    pub bytes: usize,
}

#[derive(Debug)]
pub(crate) struct IoBatch {
    pub file_offset: u64,
    pub bytes: usize,
    pub copies: Vec<CopyRange>,
}

/// Merge adjacent aligned intervals, splitting large state checkpoints to keep
/// staging bounded. Read amplification is restricted to the aligned edges.
pub(crate) fn plan_reads(mut copies: Vec<CopyRange>) -> Result<Vec<IoBatch>, String> {
    copies.sort_unstable_by_key(|copy| copy.file_offset);
    let mut batches: Vec<IoBatch> = Vec::new();
    for mut copy in copies {
        copy.file_offset
            .checked_add(copy.bytes as u64)
            .filter(|&end| i64::try_from(end).is_ok())
            .ok_or("SSD read offset overflow")?;
        copy.device
            .checked_add(copy.bytes as u64)
            .ok_or("GPU destination overflow")?;
        while copy.bytes != 0 {
            let aligned = copy.file_offset / ALIGNMENT as u64 * ALIGNMENT as u64;
            let append = batches.last().is_some_and(|batch| {
                copy.file_offset >= batch.file_offset
                    && aligned <= batch.file_offset + batch.bytes as u64
                    && copy.file_offset < batch.file_offset + STAGING_BYTES as u64
            });
            if !append {
                batches.push(IoBatch {
                    file_offset: aligned,
                    bytes: 0,
                    copies: Vec::new(),
                });
            }
            let batch = batches.last_mut().expect("batch inserted");
            let offset = (copy.file_offset - batch.file_offset) as usize;
            let bytes = copy.bytes.min(STAGING_BYTES - offset);
            batch.bytes = batch
                .bytes
                .max((offset + bytes).next_multiple_of(ALIGNMENT));
            batch.copies.push(CopyRange { bytes, ..copy });
            copy.file_offset += bytes as u64;
            copy.device += bytes as u64;
            copy.bytes -= bytes;
        }
    }
    Ok(batches)
}

/// A complete object owns its aligned extent, including padding. Gather only
/// its valid GPU bytes and zero padding before any chunk becomes visible.
pub(crate) fn plan_writes(
    file_offset: u64,
    bytes: u64,
    copies: Vec<CopyRange>,
) -> Result<Vec<IoBatch>, String> {
    let end = file_offset
        .checked_add(bytes)
        .filter(|&end| i64::try_from(end).is_ok())
        .ok_or("SSD write offset overflow")?;
    if !file_offset.is_multiple_of(ALIGNMENT as u64)
        || !bytes.is_multiple_of(ALIGNMENT as u64)
        || bytes == 0
    {
        return Err("SSD writes require an aligned, nonempty reservation".into());
    }
    for copy in &copies {
        if copy.file_offset < file_offset
            || copy
                .file_offset
                .checked_add(copy.bytes as u64)
                .is_none_or(|limit| limit > end)
            || copy.device.checked_add(copy.bytes as u64).is_none()
        {
            return Err("GPU write range exceeds its reservation".into());
        }
    }
    let mut batches = Vec::new();
    let mut start = file_offset;
    while start < end {
        let limit = end.min(start + STAGING_BYTES as u64);
        let pieces = copies
            .iter()
            .filter_map(|copy| {
                let begin = copy.file_offset.max(start);
                let finish = (copy.file_offset + copy.bytes as u64).min(limit);
                (begin < finish).then(|| CopyRange {
                    file_offset: begin,
                    device: copy.device + (begin - copy.file_offset),
                    bytes: (finish - begin) as usize,
                })
            })
            .collect();
        batches.push(IoBatch {
            file_offset: start,
            bytes: (limit - start) as usize,
            copies: pieces,
        });
        start = limit;
    }
    Ok(batches)
}

pub(crate) struct GpuBuffer {
    driver: Arc<Cufile>,
    stream: Arc<CudaStream>,
    pointer: u64,
}

impl GpuBuffer {
    pub(crate) fn new(stream: Arc<CudaStream>) -> Result<Self, String> {
        let driver = Cufile::get(false)?;
        stream
            .context()
            .bind_to_thread()
            .map_err(|e| e.to_string())?;
        // SAFETY: a synchronous device allocation, owned until Drop, avoids
        // allocator-pool/VMM registration differences between CUDA releases.
        let pointer = unsafe { result::malloc_sync(STAGING_BYTES) }.map_err(|e| e.to_string())?;
        // SAFETY: pointer owns STAGING_BYTES and the calling thread has its CUDA context.
        let registration =
            unsafe { (driver.register_buffer)(pointer as *const c_void, STAGING_BYTES, 0) };
        if let Err(error) = registration.check("cuFileBufRegister") {
            // SAFETY: no I/O was issued, so this allocation can be released.
            unsafe { result::free_sync(pointer) }.map_err(|e| e.to_string())?;
            return Err(error);
        }
        core_metrics()
            .ssd_gpu_staging_bytes
            .add(STAGING_BYTES as i64, &[]);
        Ok(Self {
            driver,
            stream,
            pointer,
        })
    }

    pub(crate) fn restore(&self, file: &CufileFile, batches: &[IoBatch]) -> Result<usize, String> {
        let mut restored = 0;
        for batch in batches {
            #[cfg(feature = "test-hooks")]
            crate::test_faults::pause_blocking("cufile");
            let started = std::time::Instant::now();
            // SAFETY: the plan bounds reads by our registered buffer; the SSD
            // lease pins the source extent. This API completes before returning.
            let bytes = unsafe {
                (self.driver.read)(
                    file.handle.as_ptr(),
                    self.pointer as *mut c_void,
                    batch.bytes,
                    batch.file_offset as i64,
                    0,
                )
            };
            core_metrics()
                .ssd_cufile_read_seconds
                .record(started.elapsed().as_secs_f64(), &[]);
            if bytes != batch.bytes as isize {
                core_metrics().ssd_cufile_read_failures.add(1, &[]);
                return Err(format!(
                    "cuFileRead returned {bytes}, expected {} at {}",
                    batch.bytes, batch.file_offset
                ));
            }
            core_metrics().ssd_cufile_read_bytes.add(bytes as u64, &[]);
            let submitted = (|| {
                for copy in &batch.copies {
                    // SAFETY: destination bounds were checked against the registered
                    // layout; the task owns its engine page lease through completion.
                    unsafe {
                        result::memcpy_dtod_async(
                            copy.device,
                            self.pointer + (copy.file_offset - batch.file_offset),
                            copy.bytes,
                            self.stream.cu_stream(),
                        )
                    }
                    .map_err(|e| e.to_string())?;
                    restored += copy.bytes;
                }
                Ok(())
            })();
            // Staging must not be overwritten while any scatter still reads it,
            // including a partially submitted scatter that failed.
            crate::transfer::finish_gpu_transfer(&self.stream, submitted)
                .map_err(|e| e.to_string())?;
        }
        Ok(restored)
    }

    pub(crate) fn save(&self, file: &CufileFile, batches: &[IoBatch]) -> Result<(), String> {
        for batch in batches {
            let submitted = (|| {
                // SAFETY: the buffer owns this bounded chunk. Every source is
                // covered by the publish call's engine-page ownership barrier.
                unsafe {
                    result::memset_d8_async(self.pointer, 0, batch.bytes, self.stream.cu_stream())
                }
                .map_err(|e| e.to_string())?;
                for copy in &batch.copies {
                    unsafe {
                        result::memcpy_dtod_async(
                            self.pointer + (copy.file_offset - batch.file_offset),
                            copy.device,
                            copy.bytes,
                            self.stream.cu_stream(),
                        )
                    }
                    .map_err(|e| e.to_string())?;
                }
                Ok(())
            })();
            crate::transfer::finish_gpu_transfer(&self.stream, submitted)
                .map_err(|e| e.to_string())?;
            #[cfg(feature = "test-hooks")]
            crate::test_faults::pause_blocking("cufile_write");
            let started = std::time::Instant::now();
            // SAFETY: the reservation exclusively owns the aligned file extent;
            // the gather completed and this synchronous write drains before return.
            let bytes = unsafe {
                (self.driver.write)(
                    file.handle.as_ptr(),
                    self.pointer as *const c_void,
                    batch.bytes,
                    batch.file_offset as i64,
                    0,
                )
            };
            #[cfg(feature = "test-hooks")]
            let bytes = if crate::test_faults::active("cufile_write_error") {
                -1
            } else {
                bytes
            };
            core_metrics()
                .ssd_cufile_write_seconds
                .record(started.elapsed().as_secs_f64(), &[]);
            if bytes != batch.bytes as isize {
                core_metrics().ssd_cufile_write_failures.add(1, &[]);
                return Err(format!(
                    "cuFileWrite returned {bytes}, expected {} at {}",
                    batch.bytes, batch.file_offset
                ));
            }
            core_metrics().ssd_cufile_write_bytes.add(bytes as u64, &[]);
        }
        Ok(())
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        let _ = crate::transfer::finish_gpu_transfer(&self.stream, Ok(()));
        if let Err(error) = self.stream.context().bind_to_thread() {
            log::error!("Cannot bind GPU context while releasing GDS buffer: {error}");
            std::process::abort();
        }
        // SAFETY: every read and scatter has drained before deregistration/free.
        let status = unsafe { (self.driver.deregister_buffer)(self.pointer as *const c_void) };
        if let Err(error) = status.check("cuFileBufDeregister") {
            log::error!("Cannot release registered GDS buffer: {error}");
            std::process::abort();
        }
        unsafe { result::free_sync(self.pointer) }.unwrap_or_else(|error| {
            log::error!("Cannot free GDS buffer: {error}");
            std::process::abort();
        });
        core_metrics()
            .ssd_gpu_staging_bytes
            .add(-(STAGING_BYTES as i64), &[]);
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/backing/ssd/cufile.rs"]
mod tests;
