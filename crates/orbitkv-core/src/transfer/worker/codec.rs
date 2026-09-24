use std::{num::NonZeroU64, time::Instant};

use cudarc::driver::{DevicePtr, result};
use opentelemetry::KeyValue;
use orbitkv_state::StorageFormat;

use super::{LayerTransferData, TransferPayload, WorkerRuntime};
use crate::{
    EngineError,
    block::{RawBlock, Segment},
    codec::{EncodedSegment, gpu::GpuCodec, segment_format},
    memory::numa::NumaNode,
    metrics::core_metrics,
    storage::StorageEngine,
    transfer::{finish_gpu_transfer, layout::BlockCopies},
};

struct Reservation {
    bytes: usize,
    start: Instant,
    operation: &'static str,
}
impl Reservation {
    fn new(bytes: usize, operation: &'static str) -> Self {
        core_metrics()
            .storage_codec_reserved_bytes
            .add(bytes as i64, &[]);
        Self {
            bytes,
            start: Instant::now(),
            operation,
        }
    }
}
impl Drop for Reservation {
    fn drop(&mut self) {
        core_metrics()
            .storage_codec_reserved_bytes
            .add(-(self.bytes as i64), &[]);
        core_metrics().storage_codec_seconds.record(
            self.start.elapsed().as_secs_f64(),
            &[KeyValue::new("operation", self.operation)],
        );
    }
}

fn ranges(copies: BlockCopies) -> Vec<(u64, usize)> {
    match copies {
        BlockCopies::Contiguous(c) => vec![(c.addr, c.bytes)],
        BlockCopies::Split { k, v } => vec![(k.addr, k.bytes), (v.addr, v.bytes)],
    }
}

pub(super) fn save(
    runtime: &WorkerRuntime,
    layers: &mut [LayerTransferData],
    storage: Option<&StorageEngine>,
    numa: NumaNode,
) -> Result<(), EngineError> {
    if !layers.iter().any(|l| {
        l.blocks
            .iter()
            .any(|b| matches!(b.block, TransferPayload::Pending))
    }) {
        return Ok(());
    }
    let storage = storage.ok_or_else(|| EngineError::Storage("missing codec allocator".into()))?;
    let mut codec = runtime.codec.borrow_mut();
    if codec.is_none() {
        *codec = Some(GpuCodec::new(runtime.stream.context()).map_err(EngineError::Storage)?);
    }
    let codec = codec.as_mut().expect("initialized");
    let stream = &runtime.stream;
    let _reservation = Reservation::new(storage.codec_budget, "encode");
    let mut transferred = 0u64;
    for layer in layers {
        for block in &mut layer.blocks {
            if !matches!(block.block, TransferPayload::Pending) {
                continue;
            }
            let copies = ranges(
                layer
                    .layout
                    .block_copies(block.block_idx)
                    .map_err(EngineError::Storage)?,
            );
            let mut segments = Vec::with_capacity(copies.len());
            let mut metadata = Vec::with_capacity(copies.len());
            let mut compressed = false;
            for (index, (source, logical)) in copies.into_iter().enumerate() {
                let requested = segment_format(layer.layout.storage_format, index);
                let encoded =
                    codec.encode(stream, source, logical, requested, storage.codec_budget);
                // Also drains an unsuccessful/partially submitted codec operation before source release.
                finish_gpu_transfer(stream, encoded.as_ref().map(|_| ()).map_err(Clone::clone))?;
                let encoded = encoded.map_err(EngineError::Storage)?;
                let alignment = if storage.is_ssd_enabled() { 512 } else { 1 };
                let encoded = encoded.filter(|(_, len)| {
                    len.next_multiple_of(alignment)
                        <= logical.next_multiple_of(alignment)
                            - logical.next_multiple_of(alignment) / 8
                });
                let (device, len, format) = match &encoded {
                    Some((buffer, len)) => (buffer.device_ptr(stream).0, *len, requested),
                    None => (source, logical, StorageFormat::Exact),
                };
                let mut len = len;
                let mut format = format;
                let mut physical = len.next_multiple_of(alignment);
                let mut allocation = storage
                    .allocate(
                        NonZeroU64::new(physical as u64).expect("nonzero segment"),
                        Some(numa),
                    )
                    .ok_or_else(|| {
                        EngineError::Storage("pinned pool exhausted during encoded save".into())
                    })?;
                let mut ptr = allocation.mapped_ptr().host();
                // SAFETY: exclusive initialized host destination; engine/source GPU ownership lasts through sync.
                unsafe {
                    ptr.as_ptr().write_bytes(0, physical);
                }
                let submitted = unsafe {
                    result::memcpy_dtoh_async(
                        std::slice::from_raw_parts_mut(ptr.as_ptr(), len),
                        device,
                        stream.cu_stream(),
                    )
                }
                .map_err(|e| e.to_string());
                finish_gpu_transfer(stream, submitted)?;
                transferred += len as u64;
                if format == StorageFormat::Exact
                    && matches!(
                        requested,
                        StorageFormat::Fp8FromBf16 | StorageFormat::Fp8FromFp16
                    )
                    && logical / 2 + 1024 > storage.codec_budget
                    && logical <= 16 * 1024 * 1024
                {
                    let packed_size = (logical / 2).next_multiple_of(alignment);
                    if packed_size <= physical - physical / 8
                        && let Some(packed) = storage.allocate(
                            NonZeroU64::new(packed_size as u64).expect("nonzero"),
                            Some(numa),
                        )
                    {
                        // SAFETY: exclusive host buffers, all D2H already complete.
                        let input = unsafe { std::slice::from_raw_parts(ptr.as_ptr(), logical) };
                        let output = unsafe {
                            std::slice::from_raw_parts_mut(
                                packed.mapped_ptr().host().as_ptr(),
                                packed_size,
                            )
                        };
                        output.fill(0);
                        if crate::codec::cpu::encode(requested, input, &mut output[..logical / 2]) {
                            allocation = packed;
                            ptr = allocation.mapped_ptr().host();
                            len = logical / 2;
                            physical = packed_size;
                            format = requested;
                        }
                    }
                }
                let bytes = unsafe { std::slice::from_raw_parts(ptr.as_ptr(), len) };
                metadata.push(EncodedSegment {
                    version: 1,
                    format,
                    logical_bytes: logical,
                    stored_bytes: len,
                    checksum: crc32fast::hash(bytes),
                });
                compressed |= format != StorageFormat::Exact;
                if format == StorageFormat::Exact && requested != StorageFormat::Exact {
                    core_metrics()
                        .storage_codec_skips
                        .add(1, &[KeyValue::new("reason", "range_ratio_or_budget")]);
                }

                segments.push(Segment::new(ptr, physical, allocation));
            }
            let mut raw = RawBlock::new(segments);
            raw.storage_format = layer.layout.storage_format;
            if compressed {
                core_metrics().storage_codec_bytes.add(
                    metadata.iter().map(|m| m.logical_bytes as u64).sum(),
                    &[KeyValue::new("representation", "logical")],
                );
                core_metrics().storage_codec_bytes.add(
                    raw.memory_footprint(),
                    &[KeyValue::new("representation", "stored")],
                );
            }
            raw.encoding = compressed.then_some(metadata);
            block.block = TransferPayload::Owned(raw);
        }
    }
    core_metrics()
        .storage_codec_transfer_bytes
        .add(transferred, &[KeyValue::new("direction", "d2h")]);
    Ok(())
}

pub(super) fn restore(
    runtime: &WorkerRuntime,
    layers: &[LayerTransferData],
    budget: usize,
) -> Result<usize, EngineError> {
    let mut total = 0;
    let mut transferred = 0;
    for layer in layers {
        for block in &layer.blocks {
            if matches!(block.block, TransferPayload::Ssd { .. }) {
                continue;
            }
            let raw = block.block.raw();
            let Some(metadata) = &raw.encoding else {
                continue;
            };
            raw.validate_encoding().map_err(EngineError::Storage)?;
            let copies = ranges(
                layer
                    .layout
                    .block_copies(block.block_idx)
                    .map_err(EngineError::Storage)?,
            );
            if block.block.host_offset() != 0 || metadata.len() != copies.len() {
                return Err(EngineError::Storage(
                    "encoded restore layout mismatch".into(),
                ));
            }
            for (index, (meta, (_, bytes))) in metadata.iter().zip(&copies).enumerate() {
                if meta.logical_bytes != *bytes
                    || (meta.format != StorageFormat::Exact
                        && meta.format != segment_format(layer.layout.storage_format, index))
                {
                    return Err(EngineError::Storage(
                        "encoded restore representation mismatch".into(),
                    ));
                }
                if meta.stored_bytes.saturating_add(1024) > budget
                    && !matches!(
                        meta.format,
                        StorageFormat::Fp8FromBf16
                            | StorageFormat::Fp8FromFp16
                            | StorageFormat::Exact
                    )
                {
                    return Err(EngineError::Storage(
                        "restore exceeds codec scratch budget".into(),
                    ));
                }
            }
        }
    }
    let stream = &runtime.stream;
    let has_encoded = layers.iter().any(|l| {
        l.blocks.iter().any(|b| {
            !matches!(b.block, TransferPayload::Ssd { .. }) && b.block.raw().encoding.is_some()
        })
    });
    let _reservation = has_encoded.then(|| Reservation::new(budget, "decode"));
    for layer in layers {
        for block in &layer.blocks {
            if matches!(block.block, TransferPayload::Ssd { .. }) {
                continue;
            }
            let raw = block.block.raw();
            let Some(metadata) = &raw.encoding else {
                continue;
            };
            let copies = ranges(
                layer
                    .layout
                    .block_copies(block.block_idx)
                    .map_err(EngineError::Storage)?,
            );
            let mut codec = runtime.codec.borrow_mut();
            if codec.is_none() {
                *codec = Some(GpuCodec::new(stream.context()).map_err(EngineError::Storage)?);
            }
            for (index, (meta, (target, _))) in metadata.iter().zip(copies).enumerate() {
                // SAFETY: immutable, validated segment remains owned by the load task.
                let host = unsafe {
                    std::slice::from_raw_parts(
                        raw.segment_ptr(index).expect("validated").as_ptr(),
                        meta.stored_bytes,
                    )
                };
                let submitted = if meta.format == StorageFormat::Exact {
                    unsafe { result::memcpy_htod_async(target, host, stream.cu_stream()) }
                        .map_err(|e| e.to_string())
                } else if meta.stored_bytes.saturating_add(1024) > budget {
                    let mut decoded = vec![0u8; meta.logical_bytes];
                    if !crate::codec::cpu::decode(meta.format, host, &mut decoded) {
                        return Err(EngineError::Storage("CPU FP8 decode mismatch".into()));
                    }
                    let submitted =
                        unsafe { result::memcpy_htod_async(target, &decoded, stream.cu_stream()) }
                            .map_err(|e| e.to_string());
                    finish_gpu_transfer(stream, submitted)?;
                    Ok(())
                } else {
                    let input = stream
                        .clone_htod(host)
                        .map_err(|e| EngineError::Storage(e.to_string()))?;
                    codec
                        .as_mut()
                        .expect("initialized")
                        .decode(stream, &input, target, meta, budget)
                };
                finish_gpu_transfer(stream, submitted)?;
                total += meta.logical_bytes;
                transferred += if meta.stored_bytes.saturating_add(1024) > budget {
                    meta.logical_bytes
                } else {
                    meta.stored_bytes
                };
            }
        }
    }
    core_metrics()
        .storage_codec_transfer_bytes
        .add(transferred as u64, &[KeyValue::new("direction", "h2d")]);
    Ok(total)
}
