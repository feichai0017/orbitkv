use std::{
    collections::hash_map::DefaultHasher,
    fmt::Write,
    hash::{Hash, Hasher},
};

use petgraph::{
    algo::toposort,
    stable_graph::NodeIndex,
    visit::{EdgeRef, NodeIndexable},
};
use rustc_hash::FxHashMap;

use super::{DimBucket, Graph, LLIRGraph};
use crate::{
    dtype::DType,
    egglog_utils::{LlirExtractor, SerializedEGraph},
    hlir::HLIROps,
    op::{IntoEgglogOp, Runtime},
    search::{SearchSpace, SelectedProgram, unroll_packed_llir},
    shape::{DynMap, Symbol},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Deserialize, serde::Serialize)]
struct LlirFingerprint(u64, u64);

/// The same semantic program identity used to validate selected schedules.
/// Independent of LLIR node allocation order; retains operation parameters and
/// ordered dependencies. This is a build-local diagnostic identity, not a
/// cryptographic digest of provider sources, weights, or a complete artifact.
pub fn llir_program_identity(llir: &LLIRGraph) -> String {
    let LlirFingerprint(first, second) = fingerprint_llir(llir);
    format!("{first:016x}{second:016x}")
}

fn fingerprint_llir(llir: &LLIRGraph) -> LlirFingerprint {
    fn hash_node(seed: u64, op: &crate::op::LLIROp, inputs: &[LlirFingerprint]) -> u64 {
        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        if let Some(input) = op.to_op::<crate::hlir::Input>() {
            "Input".hash(&mut hasher);
            input.label.hash(&mut hasher);
            write!(&mut HashWriter(&mut hasher), "{:?}", input.dtype).unwrap();
        } else if let Some(output) = op.to_op::<crate::hlir::Output>() {
            "Output".hash(&mut hasher);
            output.persist_only.hash(&mut hasher);
        } else {
            // Debug is the complete semantic representation of a type-erased
            // LLIR op. Runtime-backed ops must keep transient caches out of
            // their Debug implementation. Input/Output graph node ids are
            // bindings in the rebuilt HLIR rather than operation semantics,
            // so those two boundary ops are handled explicitly above.
            write!(&mut HashWriter(&mut hasher), "{op:?}").unwrap();
        }
        inputs.hash(&mut hasher);
        hasher.finish()
    }

    let mut node_fingerprints = vec![None; llir.node_bound()];
    for node in toposort(llir, None).expect("LLIR must be acyclic") {
        let mut incoming = llir
            .edges_directed(node, petgraph::Direction::Incoming)
            .map(|edge| (edge.id().index(), edge.source()))
            .collect::<Vec<_>>();
        incoming.sort_unstable_by_key(|(edge, _)| *edge);
        let inputs = incoming
            .into_iter()
            .map(|(_, source)| node_fingerprints[source.index()].unwrap())
            .collect::<Vec<_>>();
        let op = &llir[node];
        node_fingerprints[node.index()] = Some(LlirFingerprint(
            hash_node(0x243f_6a88_85a3_08d3, op, &inputs),
            hash_node(0x1319_8a2e_0370_7344, op, &inputs),
        ));
    }

    let mut nodes = node_fingerprints.into_iter().flatten().collect::<Vec<_>>();
    nodes.sort_unstable_by_key(|fingerprint| (fingerprint.0, fingerprint.1));
    let mut first = DefaultHasher::new();
    let mut second = DefaultHasher::new();
    0x243f_6a88_85a3_08d3_u64.hash(&mut first);
    0x1319_8a2e_0370_7344_u64.hash(&mut second);
    llir.edge_count().hash(&mut first);
    llir.edge_count().hash(&mut second);
    nodes.hash(&mut first);
    nodes.hash(&mut second);
    LlirFingerprint(first.finish(), second.finish())
}

struct HashWriter<'a>(&'a mut DefaultHasher);

impl std::fmt::Write for HashWriter<'_> {
    fn write_str(&mut self, value: &str) -> std::fmt::Result {
        self.0.write(value.as_bytes());
        Ok(())
    }
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct ScheduleBucket {
    egraph: SerializedEGraph,
    choices: Vec<(String, String)>,
    bucket_indices: DynMap,
    representative_dyn_map: DynMap,
    unrolled_llir_fingerprint: LlirFingerprint,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct SelectedSchedule {
    dim_buckets: FxHashMap<Symbol, Vec<DimBucket>>,
    buckets: Vec<ScheduleBucket>,
}

impl SelectedSchedule {
    /// Validated representative shapes in executable bucket order. Deployment
    /// preparation must use these points rather than independently choosing
    /// values from dimension intervals, whose Cartesian product can be invalid.
    pub fn bucket_representatives(&self) -> impl ExactSizeIterator<Item = &DynMap> {
        self.buckets
            .iter()
            .map(|bucket| &bucket.representative_dyn_map)
    }

    #[doc(hidden)]
    pub fn from_search(space: &SearchSpace, selected: &[SelectedProgram]) -> Option<Self> {
        if selected.len() != space.buckets.len() {
            return None;
        }
        let buckets = space
            .buckets
            .iter()
            .zip(selected)
            .map(|(bucket, selected)| {
                let mut extractor = LlirExtractor::new(&bucket.egraph, &space.ops);
                let choices = extractor.named_choices(&selected.genome);
                let indexed = extractor.index_named_choices(&choices);
                let llir = unroll_packed_llir(
                    extractor.extract_indexed_packed(&indexed, &space.custom_ops),
                );
                ScheduleBucket {
                    egraph: bucket.egraph.clone(),
                    choices,
                    bucket_indices: selected.bucket_indices.clone(),
                    representative_dyn_map: selected.representative_dyn_map.clone(),
                    unrolled_llir_fingerprint: fingerprint_llir(&llir),
                }
            })
            .collect();
        Some(Self {
            dim_buckets: space.dim_buckets.clone(),
            buckets,
        })
    }

    /// Re-extract every selected bucket and emit its exact LLIR through the
    /// standard diagnostic dumper. Intended for post-search correctness
    /// forensics; ordinary artifact serialization remains unchanged.
    #[doc(hidden)]
    pub fn dump_llirs(
        &self,
        ops: &[std::sync::Arc<Box<dyn crate::op::EgglogOp>>],
        custom_ops: &[crate::op::LLIROp],
    ) {
        for (bucket_index, bucket) in self.buckets.iter().enumerate() {
            let mut extractor = LlirExtractor::new(&bucket.egraph, ops);
            let choices = extractor.index_named_choices(&bucket.choices);
            let llir = unroll_packed_llir(extractor.extract_indexed_packed(&choices, custom_ops));
            crate::search::diagnostics::maybe_dump_selected_llir(
                &format!("artifact-bucket-{bucket_index}"),
                &bucket.representative_dyn_map,
                &llir,
            );
        }
    }
}

impl Graph {
    pub fn selected_schedule(&self) -> Option<&SelectedSchedule> {
        self.selected_schedule.as_ref()
    }

    /// Installs a previously selected schedule on this graph.
    ///
    /// Loading still re-extracts every bucket using this graph's current
    /// custom-op table and verifies the stored LLIR fingerprint. A schedule
    /// from a graph with different custom-op ordering or geometry therefore
    /// fails closed in [`Self::load_selected_schedule`].
    pub fn install_selected_schedule(&mut self, schedule: SelectedSchedule) {
        self.selected_schedule = Some(schedule);
    }

    pub fn from_selected_schedule(
        dyn_map: DynMap,
        input_meta: FxHashMap<NodeIndex, (String, DType)>,
        schedule: SelectedSchedule,
    ) -> Self {
        Self {
            dyn_map,
            input_meta,
            selected_schedule: Some(schedule),
            ..Self::default()
        }
    }

    pub fn load_selected_schedule<R: Runtime + 'static>(
        &self,
        runtime: &mut R,
    ) -> Result<(), String> {
        let schedule = self
            .selected_schedule
            .as_ref()
            .ok_or_else(|| "graph has no selected schedule".to_string())?;
        let custom_ops = self
            .custom_ops
            .iter()
            .map(|op| op.to_llir_op())
            .collect::<Vec<_>>();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut ops = R::Ops::into_vec();
            ops.extend(<HLIROps as IntoEgglogOp>::into_vec());
            let bucket_llirs = schedule
                .buckets
                .iter()
                .enumerate()
                .map(|(bucket_idx, bucket)| {
                    let mut extractor = LlirExtractor::new(&bucket.egraph, &ops);
                    let choices = extractor.index_named_choices(&bucket.choices);
                    let packed = extractor.extract_indexed_packed(&choices, &custom_ops);
                    let llir = unroll_packed_llir(packed);
                    assert_eq!(
                        fingerprint_llir(&llir),
                        bucket.unrolled_llir_fingerprint,
                        "selected schedule bucket {bucket_idx} unrolled LLIR fingerprint mismatch",
                    );
                    crate::search::diagnostics::maybe_dump_selected_llir(
                        &format!("artifact-load-bucket-{bucket_idx}"),
                        &bucket.representative_dyn_map,
                        &llir,
                    );
                    (
                        bucket.bucket_indices.clone(),
                        bucket.representative_dyn_map.clone(),
                        llir,
                    )
                })
                .collect::<Vec<_>>();
            runtime.load_llir_buckets(&schedule.dim_buckets, &bucket_llirs);
        }));
        result.map_err(|payload| {
            let detail = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or("non-string panic");
            format!("selected schedule could not be loaded: {detail}")
        })
    }
}

#[cfg(test)]
#[path = "../../tests/unit/graph/artifact/mod.rs"]
mod tests;
