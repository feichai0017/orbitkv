//! Optional nvCOMP 5.3 ABI. No nvCOMP binaries or headers are bundled.
use std::{ffi::c_void, sync::Arc};

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};

#[repr(C)]
#[derive(Clone, Copy)]
struct Options {
    algorithm: i32,
    data_type: i32,
    sub_chunks: u8,
    reserved: [u8; 55],
}
const OPTIONS: Options = Options {
    algorithm: 0,
    data_type: 0,
    sub_chunks: 0,
    reserved: [0; 55],
};
type Temp = unsafe extern "C" fn(usize, usize, Options, *mut usize, usize) -> i32;
type MaxOutput = unsafe extern "C" fn(usize, Options, *mut usize) -> i32;
type Compress = unsafe extern "C" fn(
    *const c_void,
    *const c_void,
    usize,
    usize,
    *mut c_void,
    usize,
    *const c_void,
    *mut c_void,
    Options,
    *mut c_void,
    *mut c_void,
) -> i32;
type Decompress = unsafe extern "C" fn(
    *const c_void,
    *const c_void,
    *const c_void,
    *mut c_void,
    usize,
    *mut c_void,
    usize,
    *const c_void,
    Options,
    *mut c_void,
    *mut c_void,
) -> i32;

pub(crate) struct Library {
    _library: libloading::Library,
    compress_temp: Temp,
    decompress_temp: Temp,
    max_output: MaxOutput,
    compress: Compress,
    decompress: Decompress,
}

fn check(status: i32) -> Result<(), String> {
    if status == 0 {
        Ok(())
    } else {
        Err(format!("nvCOMP ANS status {status}"))
    }
}

impl Library {
    pub(crate) fn load() -> Result<Self, String> {
        let path =
            std::env::var_os("ORBITKV_NVCOMP_LIBRARY").unwrap_or_else(|| "libnvcomp.so.5".into());
        // SAFETY: versioned NVIDIA C ABI; copied function pointers remain owned by the library.
        unsafe {
            let library = libloading::Library::new(&path)
                .map_err(|e| format!("ANS needs nvCOMP 5.3 (set ORBITKV_NVCOMP_LIBRARY): {e}"))?;
            let properties = library
                .get::<unsafe extern "C" fn(*mut u32) -> i32>(b"nvcompGetProperties\0")
                .map_err(|e| e.to_string())?;
            let mut version = [0u32; 2];
            check(properties(version.as_mut_ptr()))?;
            if version[0] / 1000 != 5 || version[0] / 100 % 10 != 3 {
                return Err(format!(
                    "nvCOMP 5.3 is required; library reports {}",
                    version[0]
                ));
            }
            Ok(Self {
                compress_temp: *library
                    .get(b"nvcompBatchedANSCompressGetTempSizeAsync\0")
                    .map_err(|e| e.to_string())?,
                decompress_temp: *library
                    .get(b"nvcompBatchedANSDecompressGetTempSizeAsync\0")
                    .map_err(|e| e.to_string())?,
                max_output: *library
                    .get(b"nvcompBatchedANSCompressGetMaxOutputChunkSize\0")
                    .map_err(|e| e.to_string())?,
                compress: *library
                    .get(b"nvcompBatchedANSCompressAsync\0")
                    .map_err(|e| e.to_string())?,
                decompress: *library
                    .get(b"nvcompBatchedANSDecompressAsync\0")
                    .map_err(|e| e.to_string())?,
                _library: library,
            })
        }
    }

    pub(crate) fn encode(
        &self,
        stream: &Arc<CudaStream>,
        source: u64,
        bytes: usize,
        budget: usize,
        data_type: i32,
    ) -> Result<Option<(CudaSlice<u8>, usize)>, String> {
        if !(4096..=16 * 1024 * 1024).contains(&bytes) || !source.is_multiple_of(8) {
            return Ok(None);
        }
        let options = Options {
            data_type,
            ..OPTIONS
        };
        let mut temp_size = 0;
        let mut max_size = 0;
        let mut decode_temp = 0;
        // SAFETY: host size queries with initialized options and output storage.
        unsafe {
            check((self.compress_temp)(
                1,
                bytes,
                options,
                &mut temp_size,
                bytes,
            ))?;
            check((self.max_output)(bytes, options, &mut max_size))?;
            check((self.decompress_temp)(
                1,
                bytes,
                OPTIONS,
                &mut decode_temp,
                bytes,
            ))?;
        }
        if temp_size
            .max(decode_temp)
            .saturating_add(max_size)
            .saturating_add(1024)
            > budget
        {
            return Ok(None);
        }
        let output = stream
            .alloc_zeros::<u8>(max_size)
            .map_err(|e| e.to_string())?;
        let temp = stream
            .alloc_zeros::<u8>(temp_size.max(1))
            .map_err(|e| e.to_string())?;
        let input_ptrs = stream.clone_htod(&[source]).map_err(|e| e.to_string())?;
        let input_sizes = stream
            .clone_htod(&[bytes as u64])
            .map_err(|e| e.to_string())?;
        let output_ptrs = stream
            .clone_htod(&[output.device_ptr(stream).0])
            .map_err(|e| e.to_string())?;
        let sizes = stream.alloc_zeros::<u64>(1).map_err(|e| e.to_string())?;
        let statuses = stream.alloc_zeros::<i32>(1).map_err(|e| e.to_string())?;
        // SAFETY: all pointers are device allocations on this stream and remain live through sync.
        let status = unsafe {
            (self.compress)(
                input_ptrs.device_ptr(stream).0 as _,
                input_sizes.device_ptr(stream).0 as _,
                bytes,
                1,
                temp.device_ptr(stream).0 as _,
                temp_size,
                output_ptrs.device_ptr(stream).0 as _,
                sizes.device_ptr(stream).0 as _,
                options,
                statuses.device_ptr(stream).0 as _,
                stream.cu_stream().cast(),
            )
        };
        stream.synchronize().map_err(|e| e.to_string())?;
        check(status)?;
        check(stream.clone_dtoh(&statuses).map_err(|e| e.to_string())?[0])?;
        let stored = stream.clone_dtoh(&sizes).map_err(|e| e.to_string())?[0] as usize;
        if stored == 0 || stored > max_size {
            return Err("nvCOMP returned invalid output size".into());
        }
        Ok(Some((output, stored)))
    }

    pub(crate) fn decode(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<u8>,
        bytes: usize,
        target: u64,
        logical: usize,
        budget: usize,
    ) -> Result<(), String> {
        if !target.is_multiple_of(8) {
            return Err("ANS destination is not 8-byte aligned".into());
        }
        let mut temp_size = 0;
        // SAFETY: host-only scratch query.
        unsafe {
            check((self.decompress_temp)(
                1,
                logical,
                OPTIONS,
                &mut temp_size,
                logical,
            ))?;
        }
        if temp_size.saturating_add(input.len()).saturating_add(64) > budget {
            return Err("ANS decode exceeds GPU codec budget".into());
        }
        let temp = stream
            .alloc_zeros::<u8>(temp_size.max(1))
            .map_err(|e| e.to_string())?;
        let input_ptrs = stream
            .clone_htod(&[input.device_ptr(stream).0])
            .map_err(|e| e.to_string())?;
        let input_sizes = stream
            .clone_htod(&[bytes as u64])
            .map_err(|e| e.to_string())?;
        let output_ptrs = stream.clone_htod(&[target]).map_err(|e| e.to_string())?;
        let capacities = stream
            .clone_htod(&[logical as u64])
            .map_err(|e| e.to_string())?;
        let sizes = stream.alloc_zeros::<u64>(1).map_err(|e| e.to_string())?;
        let statuses = stream.alloc_zeros::<i32>(1).map_err(|e| e.to_string())?;
        // SAFETY: caller validated checksum and bounded metadata before upload. Output capacity is explicit.
        let status = unsafe {
            (self.decompress)(
                input_ptrs.device_ptr(stream).0 as _,
                input_sizes.device_ptr(stream).0 as _,
                capacities.device_ptr(stream).0 as _,
                sizes.device_ptr(stream).0 as _,
                1,
                temp.device_ptr(stream).0 as _,
                temp_size,
                output_ptrs.device_ptr(stream).0 as _,
                OPTIONS,
                statuses.device_ptr(stream).0 as _,
                stream.cu_stream().cast(),
            )
        };
        stream.synchronize().map_err(|e| e.to_string())?;
        check(status)?;
        check(stream.clone_dtoh(&statuses).map_err(|e| e.to_string())?[0])?;
        if stream.clone_dtoh(&sizes).map_err(|e| e.to_string())?[0] != logical as u64 {
            return Err("ANS decoded size mismatch".into());
        }
        Ok(())
    }
}
