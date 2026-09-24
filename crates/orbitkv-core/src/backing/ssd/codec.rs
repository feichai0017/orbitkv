use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::io;
use std::ptr::NonNull;
use std::sync::Arc;
use std::time::Instant;

use opentelemetry::KeyValue;
use orbitkv_state::StorageFormat;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::metrics::core_metrics;

mod fp8;

const ALIGNMENT: usize = 4096;

/// Each segment is independently decoded into its original, bounded allocation.
#[derive(Clone)]
pub(crate) enum Encoding {
    Raw,
    Fp8V1(Vec<EncodedSegment>),
}

#[derive(Clone)]
pub(crate) struct EncodedSegment {
    pub format: StorageFormat,
    pub bytes: usize,
    pub checksum: u32,
}

/// Host scratch is separate from pinned cache capacity and survives submitted I/O.
pub(super) struct Codec {
    budget: Arc<Semaphore>,
    capacity: usize,
}

pub(super) struct Buffer {
    ptr: NonNull<u8>,
    layout: Layout,
    _permit: OwnedSemaphorePermit,
    pub(super) len: usize,
}

// SAFETY: Buffer uniquely owns initialized host memory; it moves with its I/O job.
unsafe impl Send for Buffer {}

impl Buffer {
    fn new(len: usize, permit: OwnedSemaphorePermit) -> io::Result<Self> {
        let layout = Layout::from_size_align(len, ALIGNMENT).map_err(io::Error::other)?;
        // SAFETY: nonzero, validated layout; ownership is released by Drop.
        let ptr = NonNull::new(unsafe { alloc_zeroed(layout) })
            .ok_or_else(|| io::Error::other("SSD codec scratch allocation failed"))?;
        core_metrics().ssd_codec_scratch_bytes.add(len as i64, &[]);
        Ok(Self {
            ptr,
            layout,
            _permit: permit,
            len,
        })
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: the allocation is initialized and uniquely borrowed.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    pub(super) fn ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // SAFETY: the last I/O/decode owner releases the original allocation.
        unsafe { dealloc(self.ptr.as_ptr(), self.layout) };
        core_metrics()
            .ssd_codec_scratch_bytes
            .add(-(self.layout.size() as i64), &[]);
    }
}

impl Codec {
    pub(super) fn new(capacity: usize) -> io::Result<Self> {
        if capacity < ALIGNMENT || capacity > u32::MAX as usize {
            return Err(io::Error::other(
                "SSD codec budget must be between 4 KiB and 4 GiB - 1",
            ));
        }
        Ok(Self {
            budget: Arc::new(Semaphore::new(capacity)),
            capacity,
        })
    }

    /// Best-effort encoding never waits behind demand reads for scratch memory.
    pub(super) fn encode(
        &self,
        segments: &[(&[u8], StorageFormat)],
        alignment: usize,
    ) -> Option<(Encoding, Buffer)> {
        let start = Instant::now();
        let result = self.encode_block(segments, alignment);
        core_metrics().ssd_codec_seconds.record(
            start.elapsed().as_secs_f64(),
            &[KeyValue::new("operation", "encode")],
        );
        result
    }

    fn encode_block(
        &self,
        segments: &[(&[u8], StorageFormat)],
        alignment: usize,
    ) -> Option<(Encoding, Buffer)> {
        let skip = |reason: &'static str| {
            core_metrics()
                .ssd_codec_skips
                .add(1, &[KeyValue::new("reason", reason)]);
            None
        };
        if !segments
            .iter()
            .any(|(_, format)| *format != StorageFormat::Exact)
        {
            return skip("format");
        }
        let mut encoded_sizes = Vec::with_capacity(segments.len());
        let mut raw = 0usize;
        let mut stored = 0usize;
        for (input, format) in segments {
            let bytes = if *format == StorageFormat::Exact {
                input.len()
            } else if input.len().is_multiple_of(2) {
                input.len() / 2
            } else {
                return skip("layout");
            };
            encoded_sizes.push(bytes);
            raw = raw.checked_add(input.len())?;
            stored = stored.checked_add(bytes)?;
        }
        let stored = stored.checked_next_multiple_of(alignment)?;
        let raw = raw.checked_next_multiple_of(alignment)?;
        if stored > raw - raw / 8 {
            return skip("ratio");
        }
        let capacity = stored.checked_next_multiple_of(ALIGNMENT)?;
        if capacity == 0 || capacity > self.capacity {
            return skip("oversized");
        }
        let Ok(permit) = Arc::clone(&self.budget).try_acquire_many_owned(capacity as u32) else {
            return skip("budget");
        };
        let Ok(mut buffer) = Buffer::new(capacity, permit) else {
            return skip("allocation");
        };
        let mut metadata = Vec::with_capacity(segments.len());
        let mut end = 0;
        for ((input, format), bytes) in segments.iter().zip(encoded_sizes) {
            let output = &mut buffer.as_mut_slice()[end..end + bytes];
            if *format == StorageFormat::Exact {
                output.copy_from_slice(input);
            } else if !fp8::encode(*format, input, output) {
                return skip("nonfinite_or_range");
            }
            metadata.push(EncodedSegment {
                format: *format,
                bytes,
                checksum: crc32fast::hash(output),
            });
            end += bytes;
        }
        buffer.as_mut_slice()[end..stored].fill(0);
        buffer.len = stored;
        Some((Encoding::Fp8V1(metadata), buffer))
    }

    pub(super) async fn read_buffer(&self, len: usize) -> io::Result<Buffer> {
        let capacity = len
            .checked_next_multiple_of(ALIGNMENT)
            .filter(|n| *n != 0 && *n <= self.capacity)
            .ok_or_else(|| io::Error::other("encoded SSD object exceeds codec budget"))?;
        let permit = Arc::clone(&self.budget)
            .acquire_many_owned(capacity as u32)
            .await
            .map_err(io::Error::other)?;
        let mut buffer = Buffer::new(capacity, permit)?;
        buffer.len = len;
        Ok(buffer)
    }
}

/// The caller owns fresh slots exclusively until every segment passes validation.
pub(super) fn decode(
    encoding: &Encoding,
    input: &mut Buffer,
    segments: &mut [&mut [u8]],
) -> io::Result<()> {
    let start = Instant::now();
    let Encoding::Fp8V1(metadata) = encoding else {
        return Err(io::Error::other("unexpected raw SSD decode"));
    };
    let result = (|| {
        if segments.len() != metadata.len() {
            return Err(io::Error::other("SSD codec segment count mismatch"));
        }
        let mut offset = 0usize;
        for (output, meta) in segments.iter_mut().zip(metadata) {
            let end = offset
                .checked_add(meta.bytes)
                .filter(|end| *end <= input.len)
                .ok_or_else(|| io::Error::other("SSD codec segment exceeds extent"))?;
            let input = &input.as_mut_slice()[offset..end];
            if crc32fast::hash(input) != meta.checksum {
                return Err(io::Error::other("SSD codec checksum mismatch"));
            }
            let valid = match meta.format {
                StorageFormat::Exact => {
                    if input.len() == output.len() {
                        output.copy_from_slice(input);
                        true
                    } else {
                        false
                    }
                }
                format => fp8::decode(format, input, output),
            };
            if !valid {
                return Err(io::Error::other("SSD codec decoded length mismatch"));
            }
            offset = end;
        }
        Ok(())
    })();
    core_metrics().ssd_codec_seconds.record(
        start.elapsed().as_secs_f64(),
        &[KeyValue::new("operation", "decode")],
    );
    if result.is_err() {
        core_metrics().ssd_codec_decode_failures.add(1, &[]);
    }
    result
}

#[cfg(test)]
#[path = "../../../tests/unit/backing/ssd/codec.rs"]
mod tests;
