//! Optional nvCOMP 5.3 ABI. No nvCOMP binaries or headers are bundled.
use std::ffi::c_void;

use cudarc::driver::CudaStream;

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

    pub(super) fn max_output(&self, bytes: usize, data_type: i32) -> Result<usize, String> {
        let mut size = 0;
        // SAFETY: host-only query with initialized options and output storage.
        unsafe {
            check((self.max_output)(
                bytes,
                Options {
                    data_type,
                    ..OPTIONS
                },
                &mut size,
            ))?;
        }
        if size == 0 || size > 32 * 1024 * 1024 {
            return Err("nvCOMP returned invalid maximum output size".into());
        }
        Ok(size)
    }

    pub(super) fn workspace(
        &self,
        count: usize,
        max_bytes: usize,
        total_bytes: usize,
        data_type: i32,
        encode: bool,
    ) -> Result<usize, String> {
        let mut size = 0;
        let query = if encode {
            self.compress_temp
        } else {
            self.decompress_temp
        };
        // SAFETY: both nvCOMP 5.3 option structs have this 64-byte C layout.
        // Decompression's zero backend selects the default GPU backend.
        unsafe {
            check(query(
                count,
                max_bytes,
                Options {
                    data_type,
                    ..OPTIONS
                },
                &mut size,
                total_bytes,
            ))?;
        }
        Ok(size)
    }

    /// Enqueue a real nvCOMP batch into caller-owned pointer/size/status tables.
    /// All device arrays and scratch must survive the caller's stream drain,
    /// including when nvCOMP returns an error after partially enqueueing work.
    pub(super) unsafe fn encode_batch(
        &self,
        stream: &CudaStream,
        batch: DeviceBatch,
        data_type: i32,
    ) -> Result<(), String> {
        // SAFETY: supplied by the codec's checked, aligned arena layout.
        unsafe {
            check((self.compress)(
                batch.inputs as _,
                batch.input_sizes as _,
                batch.max_bytes,
                batch.count,
                batch.temp as _,
                batch.temp_bytes,
                batch.outputs as _,
                batch.sizes as _,
                Options {
                    data_type,
                    ..OPTIONS
                },
                batch.statuses as _,
                stream.cu_stream().cast(),
            ))
        }
    }

    pub(super) unsafe fn decode_batch(
        &self,
        stream: &CudaStream,
        batch: DeviceBatch,
        data_type: i32,
    ) -> Result<(), String> {
        // SAFETY: the entire batch has passed bounded metadata and payload CRC
        // checks; the codec owns the tables and drains before releasing ranges.
        unsafe {
            check((self.decompress)(
                batch.inputs as _,
                batch.input_sizes as _,
                batch.capacities as _,
                batch.sizes as _,
                batch.count,
                batch.temp as _,
                batch.temp_bytes,
                batch.outputs as _,
                Options {
                    data_type,
                    ..OPTIONS
                },
                batch.statuses as _,
                stream.cu_stream().cast(),
            ))
        }
    }
}

/// Views into the codec arena; never allocates or synchronizes independently.
pub(super) struct DeviceBatch {
    pub inputs: u64,
    pub input_sizes: u64,
    pub outputs: u64,
    pub capacities: u64,
    pub sizes: u64,
    pub statuses: u64,
    pub temp: u64,
    pub temp_bytes: usize,
    pub count: usize,
    pub max_bytes: usize,
}
