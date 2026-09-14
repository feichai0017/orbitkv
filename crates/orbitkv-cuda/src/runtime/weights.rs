//! Safetensor loading: validate encodings, borrow or convert bytes, upload, bind.
//! Checkpoint mappings and conversion buffers never become runtime host mirrors.

use std::{fs::File, path::Path};

use anyhow::{Context, Result};
use memmap2::MmapOptions;
use orbitkv_compiler::{hlir::Input, op::IntoEgglogOp, prelude::Graph};
use safetensors::SafeTensors;

use super::{CudaInput, CudaRuntimeImpl};

mod convert;
use convert::{Conversion, WeightData};

/// Actual graph bindings loaded from one shard. Extra checkpoint tensors are
/// ignored; a graph may intentionally load several shards or a model subset.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WeightLoadReport {
    pub tensors: usize,
    pub converted_tensors: usize,
    pub source_bytes: usize,
    pub device_bytes: usize,
}

impl<O: IntoEgglogOp> CudaRuntimeImpl<O> {
    /// Loads matching named inputs from an immutable safetensors checkpoint.
    ///
    /// Storage-compatible tensors borrow their mapped bytes. Floating-point
    /// conversions allocate one typed host buffer; neither path copies into a
    /// second byte vector. Uploads complete before the mapping is released.
    /// Call during initialization; a device failure may leave earlier bindings
    /// loaded. File/encoding validation completes before any bindings change.
    ///
    /// Returns contextual file, encoding, conversion or CUDA errors. The file
    /// must remain unchanged for the duration of the load.
    pub fn load_safetensors(
        &mut self,
        graph: &Graph,
        path: impl AsRef<Path>,
    ) -> Result<WeightLoadReport> {
        let path = path.as_ref();
        self.load_weight_shard(graph, path)
            .with_context(|| format!("loading weights from {}", path.display()))
    }

    fn load_weight_shard(&mut self, graph: &Graph, path: &Path) -> Result<WeightLoadReport> {
        let _file = tracing::info_span!(target: "orbitkv::stage", "cuda.weights.file", path = %path.display()).entered();
        let mmap =
            tracing::info_span!(target: "orbitkv::stage", "cuda.weights.map").in_scope(|| {
                let file = File::open(path)?;
                // Checkpoints are immutable while loading. The mapping remains
                // alive until all uploads on this stream have completed.
                unsafe { MmapOptions::new().map(&file) }
            })?;
        let tensors = tracing::info_span!(target: "orbitkv::stage", "cuda.weights.metadata")
            .in_scope(|| SafeTensors::deserialize(&mmap))?;
        let jobs = tracing::info_span!(target: "orbitkv::stage", "cuda.weights.validate")
            .in_scope(|| {
                graph
                    .graph
                    .node_indices()
                    .filter_map(|node| {
                        let input = (*graph.graph[node]).as_any().downcast_ref::<Input>()?;
                        let tensor = tensors.tensor(&input.label).ok()?;
                        Some(
                            Conversion::for_dtypes(tensor.dtype(), input.dtype)
                                .with_context(|| format!("tensor {}", input.label))
                                .map(|conversion| (node, input, tensor, conversion)),
                        )
                    })
                    .collect::<Result<Vec<_>>>()
            })?;
        let loaded = (|| -> Result<WeightLoadReport> {
            let mut report = WeightLoadReport::default();
            for (node, input, tensor, conversion) in jobs {
                let _tensor = tracing::info_span!(target: "orbitkv::stage", "cuda.weights.tensor",
                label = %input.label, source_dtype = ?tensor.dtype(), target_dtype = ?input.dtype,
                source_bytes = tensor.data().len(), converted = conversion.is_conversion())
                .entered();
                let data = if conversion.is_conversion() {
                    tracing::info_span!(target: "orbitkv::stage", "cuda.weights.convert")
                        .in_scope(|| conversion.apply(tensor.data()))?
                } else {
                    WeightData::Borrowed(tensor.data())
                };
                let bytes = data.as_bytes();
                let mut buffer = tracing::info_span!(target: "orbitkv::stage", "cuda.weights.allocate", bytes = bytes.len())
                .in_scope(|| {
                    if bytes.is_empty() {
                        self.cuda_stream.null::<u8>()
                    } else {
                        // Every byte is initialized by the upload before this
                        // allocation becomes a visible runtime input.
                        unsafe { self.cuda_stream.alloc::<u8>(bytes.len()) }
                    }
                })?;
                if !bytes.is_empty() {
                    tracing::info_span!(target: "orbitkv::stage", "cuda.weights.upload", bytes = bytes.len())
                    .in_scope(|| self.cuda_stream.memcpy_htod(bytes, &mut buffer))
                    .with_context(|| format!("uploading tensor {}", input.label))?;
                }
                self.hlir_host_mirrors.remove(&node);
                self.changed_hlir.insert(node);
                self.hlir_buffers.insert(
                    node,
                    CudaInput::Buffer {
                        buf: buffer,
                        len: bytes.len(),
                    },
                );
                report.tensors += 1;
                report.converted_tensors += usize::from(conversion.is_conversion());
                report.source_bytes += tensor.data().len();
                report.device_bytes += bytes.len();
            }
            Ok(report)
        })();
        // Drain even when an upload failed, before releasing the shard mapping.
        let completed = tracing::info_span!(target: "orbitkv::stage", "cuda.weights.complete")
            .in_scope(|| self.cuda_stream.synchronize());
        let report = loaded?;
        completed?;
        tracing::info_span!(target: "orbitkv::stage", "cuda.weights.loaded", tensors = report.tensors,
            converted_tensors = report.converted_tensors, source_bytes = report.source_bytes,
            device_bytes = report.device_bytes).in_scope(|| {});
        Ok(report)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/runtime/weights/mod.rs"]
mod tests;
