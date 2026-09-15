//! A captured plan depends on CSR contents, not just their addresses and sizes.

use super::{FlashInferPointers, PreparedFlashInferAttention};
use crate::cudarc::driver::{CudaStream, result};
use orbitkv_compiler::prelude::FxHashMap;
use std::sync::Arc;

/// One materialization's snapshots. Layers sharing metadata read it once;
/// snapshots never survive an execution or retain borrowed host pointers.
#[derive(Default)]
pub(crate) struct FlashInferMetadataCache {
    values: FxHashMap<(u64, usize), Vec<i32>>,
}

impl FlashInferMetadataCache {
    fn read(
        &mut self,
        stream: &Arc<CudaStream>,
        pointer: u64,
        count: usize,
    ) -> anyhow::Result<&[i32]> {
        if let std::collections::hash_map::Entry::Vacant(entry) =
            self.values.entry((pointer, count))
        {
            let mut values = vec![0_i32; count];
            // The prepared signature owns the validated row count. Its
            // runtime input allocation remains live throughout this call.
            unsafe { result::memcpy_dtoh_async(&mut values, pointer, stream.cu_stream())? };
            stream.synchronize()?;
            entry.insert(values);
        }
        Ok(&self.values[&(pointer, count)])
    }
}

impl PreparedFlashInferAttention {
    /// Call only while the capture's dimensions and buffer bindings still
    /// match. Other changes already require ordinary provider preparation.
    pub(crate) fn graph_metadata_matches(
        &self,
        stream: &Arc<CudaStream>,
        pointers: FlashInferPointers,
        snapshots: &mut FlashInferMetadataCache,
    ) -> anyhow::Result<bool> {
        let Some(kv) = pointers.explicit_kv_indptr else {
            return Ok(true);
        };
        let rows = self.spec.batch_size + 1;
        if snapshots.read(stream, kv, rows)? != self.spec.kv_indptr_host {
            return Ok(false);
        }
        if self.spec.uses_tensor_cores()
            && let Some(query) = pointers.explicit_qo_indptr
            && snapshots.read(stream, query, rows)? != self.spec.qo_indptr_host
        {
            return Ok(false);
        }
        Ok(true)
    }
}
