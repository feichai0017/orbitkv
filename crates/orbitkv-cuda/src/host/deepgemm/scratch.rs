//! Owned activation scratch whose addresses may be retained by CUDA graphs.
//!
//! Acquire raw pointers outside capture; each captured child graph retains the
//! exact `Arc<Scratch>` it references until its executable has retired. Growth
//! publishes a new allocation without invalidating earlier graph owners.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};

use super::contract::scratch_bytes;

#[derive(Debug)]
pub(super) struct Scratch {
    quantized: CudaSlice<u8>,
    scales: CudaSlice<u8>,
    pub(super) quantized_ptr: u64,
    pub(super) scales_ptr: u64,
}

impl Scratch {
    pub(super) fn prepare<'a>(
        current: &'a mut Option<Arc<Self>>,
        stream: &Arc<CudaStream>,
        rows: usize,
        input_features: usize,
    ) -> anyhow::Result<&'a Arc<Self>> {
        let (quantized_bytes, scale_bytes) =
            scratch_bytes(rows, input_features).map_err(|error| anyhow::anyhow!(error))?;
        if current.as_ref().is_none_or(|scratch| {
            scratch.quantized.len() < quantized_bytes || scratch.scales.len() < scale_bytes
        }) {
            *current = Some(Self::new(stream, quantized_bytes, scale_bytes)?);
        }
        Ok(current.as_ref().unwrap())
    }

    fn new(
        stream: &Arc<CudaStream>,
        quantized_bytes: usize,
        scale_bytes: usize,
    ) -> anyhow::Result<Arc<Self>> {
        anyhow::ensure!(
            stream.capture_status()?
                == cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE,
            "block-scaled scratch must be prepared outside CUDA graph capture"
        );
        let quantized = unsafe { stream.alloc::<u8>(quantized_bytes)? };
        let scales = unsafe { stream.alloc::<u8>(scale_bytes)? };
        // Acquire addresses outside capture. device_ptr records cudarc usage
        // events: recording them inside a child graph leaves Drop waiting on
        // an event belonging to a graph that may already have been destroyed.
        let quantized_ptr = quantized.device_ptr(stream).0;
        let scales_ptr = scales.device_ptr(stream).0;
        // Allocation occurs on the private capture stream, while the graph
        // executes on the runtime stream. Publish only completed allocations.
        stream.synchronize()?;
        Ok(Arc::new(Self {
            quantized,
            scales,
            quantized_ptr,
            scales_ptr,
        }))
    }

    pub(super) fn bytes(&self) -> usize {
        self.quantized.len() + self.scales.len()
    }
}
