//! Compile validated DMA ranges without copying gaps or reordering operations.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream, sys};

use super::{CopyDesc, TransferBackend};

/// One contiguous row or a run of explicit equal-width, constant-pitch rows.
/// Registration identities constrain CUDA access; the caller retains every
/// logical allocation covered by the original descriptors through stream drain.
#[derive(Debug)]
pub(super) struct DmaCopy {
    device: u64,
    host: *mut u8,
    width: usize,
    rows: usize,
    device_pitch: usize,
    host_pitch: usize,
    device_allocation: usize,
    host_registration: usize,
}

fn contiguous_ranges(mut copies: &[CopyDesc]) -> impl Iterator<Item = DmaCopy> + '_ {
    std::iter::from_fn(move || {
        let (&start, remaining) = copies.split_first()?;
        copies = remaining;
        let mut width = start.size;
        while let Some((&next, remaining)) = copies.split_first() {
            if start.device_allocation != next.device_allocation
                || start.host_registration != next.host_registration
                || start.device.checked_add(width as u64) != Some(next.device)
                || (start.host as usize).checked_add(width) != Some(next.host as usize)
                || next.device.checked_add(next.size as u64).is_none()
                || (next.host as usize).checked_add(next.size).is_none()
            {
                break;
            }
            let Some(combined) = width.checked_add(next.size) else {
                break;
            };
            width = combined;
            copies = remaining;
        }
        Some(DmaCopy {
            device: start.device,
            host: start.host,
            width,
            rows: 1,
            device_pitch: width,
            host_pitch: width,
            device_allocation: start.device_allocation,
            host_registration: start.host_registration,
        })
    })
}

/// Every row comes from validated descriptors. Neither a missing row nor a
/// gap between rows can become part of the copy. Descending or overlapping
/// rows remain separate submissions, preserving their original order.
pub(super) fn dma_copies(
    copies: &[CopyDesc],
    max_pitch: usize,
) -> impl Iterator<Item = DmaCopy> + '_ {
    let mut ranges = contiguous_ranges(copies).peekable();
    std::iter::from_fn(move || {
        let mut copy = ranges.next()?;
        let mut previous_device = copy.device;
        let mut previous_host = copy.host as usize;
        while let Some(next) = ranges.peek() {
            if next.width != copy.width
                || copy.width == 0
                || next.device_allocation != copy.device_allocation
                || next.host_registration != copy.host_registration
                || next.device.checked_add(next.width as u64).is_none()
                || (next.host as usize).checked_add(next.width).is_none()
            {
                break;
            }
            let Some(device_pitch) = next
                .device
                .checked_sub(previous_device)
                .and_then(|pitch| usize::try_from(pitch).ok())
            else {
                break;
            };
            let Some(host_pitch) = (next.host as usize).checked_sub(previous_host) else {
                break;
            };
            if device_pitch < copy.width
                || host_pitch < copy.width
                || device_pitch > max_pitch
                || host_pitch > max_pitch
                || (copy.rows > 1
                    && (device_pitch != copy.device_pitch || host_pitch != copy.host_pitch))
            {
                break;
            }
            copy.device_pitch = device_pitch;
            copy.host_pitch = host_pitch;
            copy.rows += 1;
            previous_device = next.device;
            previous_host = next.host as usize;
            ranges.next();
        }
        Some(copy)
    })
}

/// DMA copy-engine backend. The device pitch limit is read once at binding.
pub struct MemcpyBackend {
    pub(super) max_pitch: usize,
}

impl MemcpyBackend {
    pub fn new(context: &CudaContext) -> Result<Self, String> {
        let max_pitch = context
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_PITCH)
            .map_err(|error| error.to_string())?;
        let max_pitch = usize::try_from(max_pitch)
            .ok()
            .filter(|pitch| *pitch > 0)
            .ok_or("invalid CUDA maximum DMA pitch")?;
        Ok(Self { max_pitch })
    }

    fn submit(
        &self,
        copies: &[CopyDesc],
        stream: &Arc<CudaStream>,
        to_device: bool,
    ) -> Result<(), String> {
        for copy in dma_copies(copies, self.max_pitch) {
            // SAFETY: descriptors were bounds-checked and their owners remain
            // alive through stream drain, including on a partial enqueue error.
            // Compilation copies only explicit rows within one host registration
            // and one device allocation; row gaps are never accessed.
            let result = unsafe {
                if copy.rows == 1 {
                    if to_device {
                        sys::cuMemcpyHtoDAsync_v2(
                            copy.device,
                            copy.host.cast(),
                            copy.width,
                            stream.cu_stream(),
                        )
                    } else {
                        sys::cuMemcpyDtoHAsync_v2(
                            copy.host.cast(),
                            copy.device,
                            copy.width,
                            stream.cu_stream(),
                        )
                    }
                } else {
                    let host = sys::CUmemorytype::CU_MEMORYTYPE_HOST;
                    let device = sys::CUmemorytype::CU_MEMORYTYPE_DEVICE;
                    let params = sys::CUDA_MEMCPY2D {
                        srcXInBytes: 0,
                        srcY: 0,
                        srcMemoryType: if to_device { host } else { device },
                        srcHost: copy.host.cast(),
                        srcDevice: copy.device,
                        srcArray: std::ptr::null_mut(),
                        srcPitch: if to_device {
                            copy.host_pitch
                        } else {
                            copy.device_pitch
                        },
                        dstXInBytes: 0,
                        dstY: 0,
                        dstMemoryType: if to_device { device } else { host },
                        dstHost: copy.host.cast(),
                        dstDevice: copy.device,
                        dstArray: std::ptr::null_mut(),
                        dstPitch: if to_device {
                            copy.device_pitch
                        } else {
                            copy.host_pitch
                        },
                        WidthInBytes: copy.width,
                        Height: copy.rows,
                    };
                    sys::cuMemcpy2DAsync_v2(&params, stream.cu_stream())
                }
            };
            if result != sys::CUresult::CUDA_SUCCESS {
                return Err(format!(
                    "DMA copy failed (to_device={to_device}, rows={}): {result:?}",
                    copy.rows
                ));
            }
        }
        Ok(())
    }
}

impl TransferBackend for MemcpyBackend {
    fn h2d(&self, copies: &[CopyDesc], stream: &Arc<CudaStream>) -> Result<(), String> {
        self.submit(copies, stream, true)
    }

    fn d2h(&self, copies: &[CopyDesc], stream: &Arc<CudaStream>) -> Result<(), String> {
        self.submit(copies, stream, false)
    }

    fn name(&self) -> &'static str {
        "direct"
    }
}

#[cfg(test)]
#[path = "../../tests/unit/transfer/memcpy.rs"]
mod tests;
