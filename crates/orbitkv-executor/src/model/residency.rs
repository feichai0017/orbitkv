//! Deployment graph residency; independent of the selected computation and
//! artifact identity. Compiled kernels and state arenas retain their owners.

// OrbitKV symbols are stable interned dimension identities.
#![allow(clippy::mutable_key_type)]

use std::{
    collections::BTreeMap,
    num::NonZeroUsize,
    time::{Duration, Instant},
};

use orbitkv_compiler::prelude::{DynMap, tracing};
use serde::Serialize;

use super::{CompiledDecoder, DecoderError, representative::RepresentativeInputs};

/// Minimal residency until a deployment explicitly budgets additional buckets.
pub const DEFAULT_GRAPH_CACHE_CAPACITY: NonZeroUsize = NonZeroUsize::MIN;

/// Materialization counters for the currently loaded decoder program.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct DecoderGraphCacheStats {
    /// Cumulative full CUDA graph builds in the currently loaded program.
    pub graph_builds: usize,
    /// Currently materialized CUDA graphs, possibly several per bucket.
    pub materialized_graphs: usize,
}

/// Startup preparation evidence. These timings include host planning and
/// synchronization; they are not device kernel execution measurements.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct DecoderPreparationReport {
    pub elapsed: Duration,
    pub compiled_buckets: usize,
    pub buckets: Box<[PreparedDecoderBucket]>,
    pub cache: DecoderGraphCacheStats,
}

/// One artifact-derived representative prepared without launching the model.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PreparedDecoderBucket {
    pub bucket_index: usize,
    pub dimensions: BTreeMap<String, usize>,
    pub elapsed: Duration,
    pub graph_builds: usize,
}

impl CompiledDecoder {
    /// Prepares a bounded set of the selected artifact's representatives.
    ///
    /// Existing materializations take priority; unused slots are filled in
    /// artifact order. Preparation must not evict a retained bucket merely to
    /// replace it with a speculative representative. This deterministic policy
    /// makes no workload-frequency assumption. The graph cache bounds the set;
    /// other shapes remain available through ordinary on-demand preparation.
    /// Inputs use validated representative metadata in the existing stable
    /// allocations. No model execution, state mutation, or request is performed.
    /// Real request metadata is always uploaded before subsequent execution.
    ///
    /// # Errors
    /// Returns missing schedule, input allocation, provider preparation, or
    /// resource errors. A failed preparation must not publish engine readiness.
    pub fn prepare_execution(&mut self) -> Result<DecoderPreparationReport, DecoderError> {
        let _stage =
            tracing::info_span!(target: "orbitkv::stage", "orbitkv.decoder.prepare_execution")
                .entered();
        let started = Instant::now();
        let schedule = self
            .graph
            .selected_schedule()
            .ok_or_else(|| DecoderError::Preparation("selected schedule is missing".into()))?;
        let compiled_buckets = self.runtime.compiled_bucket_count();
        if schedule.bucket_representatives().len() != compiled_buckets || compiled_buckets == 0 {
            return Err(DecoderError::Preparation(
                "compiled bucket set is inconsistent".into(),
            ));
        }
        let capacity = self
            .runtime
            .max_materialized_buckets()
            .unwrap_or(compiled_buckets);
        // Keep the selected shapes independent of mutations to the runtime.
        let representatives = schedule
            .bucket_representatives()
            .cloned()
            .collect::<Vec<DynMap>>();
        let order = preparation_order(
            compiled_buckets,
            &self.runtime.materialized_bucket_indices(),
            capacity,
        );
        self.captured_decode = None;
        let inputs = RepresentativeInputs::new(&self.decoder, self.compile, self.page_tokens);
        let mut buckets = Vec::with_capacity(order.len());
        for bucket_index in order {
            let dims = &representatives[bucket_index];
            let _bucket = tracing::info_span!(target: "orbitkv::stage", "orbitkv.decoder.prepare_bucket", bucket = bucket_index).entered();
            let bucket_started = Instant::now();
            let before = self.graph_cache_stats();
            for (input, values) in inputs.values(dims) {
                self.runtime.set_data(input, values);
            }
            self.validate_input_allocations()?;
            self.runtime.prepare_cuda_graphs(dims).map_err(|error| {
                DecoderError::Preparation(format!("bucket {bucket_index}: {error:#}"))
            })?;
            if self.runtime.active_bucket_index() != bucket_index {
                return Err(DecoderError::Preparation(
                    "representative resolved to a different bucket".into(),
                ));
            }
            buckets.push(PreparedDecoderBucket {
                bucket_index,
                dimensions: dims
                    .iter()
                    .map(|(dim, &value)| (dim.to_string(), value))
                    .collect(),
                elapsed: bucket_started.elapsed(),
                graph_builds: self.graph_cache_stats().graph_builds - before.graph_builds,
            });
        }
        Ok(DecoderPreparationReport {
            elapsed: started.elapsed(),
            compiled_buckets,
            buckets: buckets.into_boxed_slice(),
            cache: self.graph_cache_stats(),
        })
    }

    /// Bounds materialized buckets, evicting the least recently used bucket
    /// before preparing a replacement. Does not change the compiled program.
    /// Changing policy invalidates an outer fixed-signature decode capture.
    pub fn set_graph_cache_capacity(&mut self, capacity: NonZeroUsize) {
        self.captured_decode = None;
        self.runtime
            .set_max_materialized_buckets(Some(capacity.get()));
    }

    /// Full builds and current residency; surgical provider recaptures are
    /// reported separately by `OrbitKV`'s per-operation diagnostics.
    #[must_use]
    pub fn graph_cache_stats(&self) -> DecoderGraphCacheStats {
        let (graph_builds, materialized_graphs) = self.runtime.cuda_graph_residency_stats();
        DecoderGraphCacheStats {
            graph_builds,
            materialized_graphs,
        }
    }
}

/// Prioritize existing work without interpreting model phases or dimensions.
/// Both the count and resident indices come from one validated runtime snapshot.
fn preparation_order(bucket_count: usize, resident: &[usize], capacity: usize) -> Vec<usize> {
    let mut order = Vec::with_capacity(capacity.min(bucket_count));
    for bucket in resident.iter().copied().chain(0..bucket_count) {
        if order.len() == capacity {
            break;
        }
        debug_assert!(bucket < bucket_count);
        if !order.contains(&bucket) {
            order.push(bucket);
        }
    }
    order
}

#[cfg(test)]
#[path = "../../tests/unit/model/residency/mod.rs"]
mod tests;
