//! CUDA graph residency policy and preparation without model execution.

use orbitkv_compiler::{
    op::IntoEgglogOp,
    prelude::{DynMap, Symbol},
};

use super::{CudaGraphOp, CudaRuntimeImpl};

impl<O: IntoEgglogOp> CudaRuntimeImpl<O> {
    /// Bound the number of compiled buckets that retain materialized CUDA
    /// graph executables. This is useful for large serving models whose
    /// kernels and shared intermediate arena fit comfortably, but whose many
    /// scheduler shapes otherwise accumulate excessive CUDA driver graph
    /// memory. A value of `None` preserves the default unbounded residency;
    /// `Some(0)` is invalid because the active bucket must remain usable.
    pub fn set_max_materialized_buckets(&mut self, max_buckets: Option<usize>) {
        assert!(
            !self.cuda_graphs_frozen
                || max_buckets.is_none_or(|cap| cap >= self.compiled_buckets.len()),
            "cannot enable CUDA graph eviction after freezing residency"
        );
        assert!(
            max_buckets != Some(0),
            "materialized bucket capacity must be positive"
        );
        self.max_materialized_buckets = max_buckets;
        self.materialized_bucket_lru
            .retain(|bucket_idx| *bucket_idx < self.compiled_buckets.len());
    }

    /// Configured bound on materialized buckets; `None` permits every bucket.
    pub fn max_materialized_buckets(&self) -> Option<usize> {
        self.max_materialized_buckets
    }

    /// Snapshot of buckets that already own graph executables, in bucket order.
    /// Startup policy can preserve this work before filling unused residency
    /// slots. Multiple graph segments in a bucket contribute one index.
    pub fn materialized_bucket_indices(&self) -> Vec<usize> {
        (0..self.compiled_buckets.len())
            .filter(|&index| self.bucket_has_materialized_cuda_graphs(index))
            .collect()
    }

    pub fn with_max_materialized_buckets(mut self, max_buckets: usize) -> Self {
        self.set_max_materialized_buckets(Some(max_buckets));
        self
    }

    pub(super) fn cuda_graphs(&self) -> impl Iterator<Item = &CudaGraphOp> {
        self.compiled_buckets.iter().flat_map(|bucket| {
            bucket
                .exec_graph
                .node_weights()
                .filter_map(|op| op.internal.as_any().downcast_ref::<CudaGraphOp>())
        })
    }

    /// Begin optional CUDA graph warmup without restricting dynamic execution.
    ///
    /// `cached_dims` are performance hints: keep separate materializations for
    /// their values when the lowered graph benefits from doing so. Other
    /// dimensions may still change anywhere within the compiled bucket ranges;
    /// execution updates parameters or rematerializes affected library calls.
    /// Previously unseen cached values are also prepared on demand. Pass `[]`
    /// to reuse one mutable materialization per compiled graph.
    ///
    /// Call `prepare_cuda_graphs` with representative inputs to warm selected
    /// shapes. No finish/freeze call is needed. Use `begin_cuda_graph_preparation`
    /// only when explicitly requiring a finite, immutable set of captures.
    pub fn begin_cuda_graph_warmup(&mut self, cached_dims: &[Symbol]) {
        self.begin_cuda_graph_residency(cached_dims, false);
    }

    /// Begin strict startup capture of finite shape variants sharing one arena.
    /// Every capture-sensitive dimension of the selected lowering must be in
    /// `shape_dims`. After `finish_cuda_graph_preparation`, unseen capture shapes
    /// are errors even inside a compile bucket. This is an opt-in restriction;
    /// ordinary dynamic execution and `begin_cuda_graph_warmup` do not impose it.
    pub fn begin_cuda_graph_preparation(&mut self, shape_dims: &[Symbol]) {
        self.begin_cuda_graph_residency(shape_dims, true);
    }

    fn begin_cuda_graph_residency(&mut self, shape_dims: &[Symbol], strict: bool) {
        assert!(
            !self.cuda_graphs_frozen,
            "CUDA graph residency is already frozen"
        );
        self.cuda_stream.synchronize().unwrap();
        if strict {
            self.set_max_materialized_buckets(None);
        }
        for graph in self.cuda_graphs() {
            graph.enable_residency(shape_dims, strict);
        }
        self.materialized_bucket_lru.clear();
        self.release_pooled_memory();
    }

    /// Materialize one shape using the caller's current, valid input descriptors.
    /// Does not launch the model or modify its KV state. Warming a representative
    /// does not specialize the logical dimensions to that value or freeze graphs.
    pub fn prepare_cuda_graphs(&mut self, dyn_map: &DynMap) -> anyhow::Result<()> {
        anyhow::ensure!(
            !self.cuda_graphs_frozen,
            "CUDA graph preparation is already finished"
        );
        self.active_bucket = self.resolve_bucket(dyn_map);
        self.compiled_buckets[self.active_bucket].hlir_synced = false;
        self.prepare_bucket_buffers(self.active_bucket, dyn_map);
        self.update_shared_dyn_dims(self.active_bucket, dyn_map);
        self.apply_output_ptr_registrations();
        self.prepare_materialized_bucket_slot(self.active_bucket);
        self.materialize_bucket_cuda_graphs(self.active_bucket, dyn_map, false)?;
        self.cuda_stream.synchronize()?;
        Ok(())
    }

    /// Validate every retained bucket, then forbid recapture, eviction and arena growth.
    pub fn finish_cuda_graph_preparation(&mut self) -> anyhow::Result<()> {
        self.cuda_stream.synchronize()?;
        // Include all retained library plans in the final capacity check.
        self.validated_resource_signatures.clear();
        let capacity = self.compiled_buckets[self.active_bucket]
            .last_resource_validation_dyn_map
            .clone();
        self.validate_compiled_bucket_resources(self.active_bucket, &capacity)
            .map_err(|error| {
                anyhow::anyhow!("resident CUDA graph resources exceed capacity: {error}")
            })?;
        for graph in self.cuda_graphs() {
            graph.validate_residency()?;
        }
        for graph in self.cuda_graphs() {
            graph.freeze_residency();
        }
        self.cuda_graphs_frozen = true;
        Ok(())
    }

    /// Total full graph builds and currently retained executables. Cheap
    /// counters for asserting that serving never creates a new graph.
    pub fn cuda_graph_residency_stats(&self) -> (usize, usize) {
        self.cuda_graphs()
            .map(CudaGraphOp::residency_stats)
            .fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
    }
}

#[cfg(test)]
#[path = "../../tests/unit/runtime/residency/mod.rs"]
mod tests;
