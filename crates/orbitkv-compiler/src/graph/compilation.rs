//! Graph normalization, saturation, and handoff to the backend runtime.

use std::{collections::BTreeSet, sync::Arc};

use itertools::Itertools;
use rustc_hash::{FxHashMap, FxHashSet};

use super::{CompileOptions, DimBucket, Graph, joint_bucket_combinations};
use crate::{
    egglog_utils::{OpTextParts, PreparedEgglog, SerializedEGraph, hlir_to_egglog},
    op::{EgglogOp, IntoEgglogOp, Runtime},
    prelude::{DynMap, Symbol},
    search::{BucketSearchSpace, SearchSpace},
    shape::{DimInterval, DynDimIntervals},
};

impl Graph {
    #[tracing::instrument(skip_all)]
    pub fn build_search_space<Rt: Runtime>(&mut self, options: CompileOptions) {
        let mut ops = Rt::Ops::into_vec();
        ops.extend(<crate::hlir::HLIROps as IntoEgglogOp>::into_vec());
        self.build_search_space_with_ops::<Rt>(ops, options);
    }

    /// Applies the deterministic graph normalization that precedes search so
    /// a previously selected schedule can be loaded without saturating and
    /// profiling the search space again.
    ///
    /// The caller must still validate its own model/backend identity before
    /// installing an artifact. [`Graph::load_selected_schedule`] re-extracts
    /// the stored choices against this graph's current custom-op table and
    /// verifies every LLIR fingerprint.
    pub fn prepare_selected_schedule(&mut self, options: &CompileOptions) {
        self.run_auto_loop_rolling_prepass(options);
        for (&dim, &value) in &options.search_dims {
            self.set_dim(dim, value);
        }
    }

    #[tracing::instrument(skip_all)]
    pub fn build_search_space_exclude_ops<Rt: Runtime, Ex: IntoEgglogOp>(
        &mut self,
        options: CompileOptions,
    ) {
        let exclude_ops = Ex::into_vec()
            .into_iter()
            .map(|e| e.sort().name)
            .collect::<FxHashSet<_>>();
        let mut ops = Rt::Ops::into_vec();
        ops.retain(|o| !exclude_ops.contains(&o.sort().name));
        ops.extend(<crate::hlir::HLIROps as IntoEgglogOp>::into_vec());
        self.build_search_space_with_ops::<Rt>(ops, options);
    }

    /// Roll loops, then saturate one e-graph per bucket combination into a
    /// [`SearchSpace`] the runtime searches in [`Runtime::compile`].
    fn build_search_space_with_ops<Rt: Runtime>(
        &mut self,
        ops: Vec<Arc<Box<dyn EgglogOp>>>,
        options: CompileOptions,
    ) {
        let _build_stage =
            tracing::info_span!(target: "orbitkv::stage", "orbitkv.compiler.search_space.build")
                .entered();
        tracing::info_span!(target: "orbitkv::stage", "orbitkv.compiler.graph.normalize")
            .in_scope(|| self.run_auto_loop_rolling_prepass(&options));
        let declarations_stage =
            tracing::info_span!(target: "orbitkv::stage", "orbitkv.compiler.egglog.declarations")
                .entered();
        let dim_buckets = options.dim_buckets.clone();
        let late_pass_dyn_map = self.late_pass_dyn_map(&dim_buckets);
        let late_passes = Rt::late_egglog_passes(&ops, &options, &late_pass_dyn_map);
        let backend_declarations = ops
            .iter()
            .flat_map(|op| op.egglog_declarations())
            .collect::<BTreeSet<_>>();
        let custom_op_declarations = self
            .custom_ops
            .iter()
            .map(|op| op.compiler_declarations())
            .filter(|declaration| {
                !declaration.is_empty() && !backend_declarations.contains(*declaration)
            })
            .collect::<BTreeSet<_>>()
            .into_iter()
            .join("\n");
        let custom_op_facts = self
            .custom_ops
            .iter()
            .enumerate()
            .map(|(id, op)| op.compiler_facts(id))
            .filter(|facts| !facts.is_empty())
            .join("\n");
        let extra_egglog = [
            Rt::extra_egglog(),
            custom_op_declarations,
            options.compiler_facts.clone(),
            custom_op_facts,
        ]
        .into_iter()
        .filter(|part| !part.is_empty())
        .join("\n");

        drop(declarations_stage);
        let (program, root) =
            tracing::info_span!(target: "orbitkv::stage", "orbitkv.compiler.hlir.serialize")
                .in_scope(|| hlir_to_egglog(self));
        let op_parts = OpTextParts::new_with_late_passes(&ops, Rt::CLEANUP_HLIR, &late_passes)
            .with_extra_egglog(extra_egglog);
        let use_interval_analysis = !self.dim_intervals.is_empty() || !dim_buckets.is_empty();
        let prepared = PreparedEgglog::new(
            &program,
            &op_parts,
            use_interval_analysis,
            options.egglog_log_enabled(),
        )
        .unwrap();
        let buckets = joint_bucket_combinations(&options)
            .into_iter()
            .enumerate()
            .map(|(bucket, (bucket_indices, representative_override))| {
                let _stage = tracing::info_span!(target: "orbitkv::stage", "orbitkv.compiler.egglog.bucket", bucket, indices = ?bucket_indices).entered();
                let intervals = self.bucket_intervals(&dim_buckets, &bucket_indices);
                let facts = crate::egglog_utils::base::interval_facts_egglog(&intervals, []);
                let (egraph, _) = prepared.run_bucket(&facts, &root).unwrap();
                BucketSearchSpace {
                    egraph,
                    bucket_indices,
                    representative_override,
                    intervals,
                }
            })
            .collect();
        let custom_ops = self.custom_ops.iter().map(|op| op.to_llir_op()).collect();
        self.search_space = Some(SearchSpace {
            buckets,
            ops,
            custom_ops,
            dim_buckets,
        });
    }

    /// Graph-wide interval assumptions narrowed to one bucket combination.
    fn bucket_intervals(
        &self,
        dim_buckets: &FxHashMap<Symbol, Vec<DimBucket>>,
        bucket_indices: &DynMap,
    ) -> DynDimIntervals {
        let mut intervals = self.dim_intervals.clone();
        for (&dim, &idx) in bucket_indices {
            let bucket = &dim_buckets[&dim][idx];
            let min = i64::try_from(bucket.min)
                .expect("DimBucket min must fit into i64 for interval analysis");
            let max = i64::try_from(bucket.max)
                .expect("DimBucket max must fit into i64 for interval analysis");
            let bucket_interval = DimInterval::new(min, max);
            intervals
                .entry(dim)
                .and_modify(|existing| {
                    existing.min = existing.min.max(bucket_interval.min);
                    existing.max = existing.max.min(bucket_interval.max);
                    assert!(
                        existing.min <= existing.max,
                        "Bucket interval for dim '{dim}' does not overlap graph interval"
                    );
                })
                .or_insert(bucket_interval);
        }
        intervals
    }

    /// Dyn map handed to backend late passes: bucket maxima override the
    /// graph's values so passes plan for the largest shape.
    fn late_pass_dyn_map(&self, dim_buckets: &FxHashMap<Symbol, Vec<DimBucket>>) -> DynMap {
        let mut dyn_map = self.dyn_map.clone();
        for (&dim, buckets) in dim_buckets {
            if let Some(max) = buckets.iter().map(|bucket| bucket.max).max() {
                dyn_map.insert(dim, max);
            }
        }
        dyn_map
    }
    /// The built search space, if [`Graph::build_search_space`] has run.
    pub fn search_space(&self) -> Option<&SearchSpace> {
        self.search_space.as_ref()
    }

    /// Get a reference to the first e-graph search space (if built)
    pub fn egraph(&self) -> Option<&SerializedEGraph> {
        self.search_space
            .as_ref()
            .and_then(|space| space.buckets.first())
            .map(|bucket| &bucket.egraph)
    }

    /// Get a reference to the available ops (if search space is built)
    pub fn egglog_ops(&self) -> Option<&Vec<Arc<Box<dyn EgglogOp>>>> {
        self.search_space.as_ref().map(|space| &space.ops)
    }

    /// Build the search space and search it with one shared set of options.
    ///
    /// This is the usual compile entry point when runtime inputs such as
    /// weights have already been loaded. Use `build_search_space` and `search`
    /// directly when the two phases need to be separated.
    #[tracing::instrument(skip_all)]
    pub fn compile<R: Runtime>(&mut self, runtime: R, options: CompileOptions) -> R {
        let mut rng = rand::rng();
        self.compile_with_rng(runtime, options, &mut rng)
    }

    #[tracing::instrument(skip_all)]
    pub fn compile_with_rng<R: Runtime, G: rand::Rng>(
        &mut self,
        runtime: R,
        options: CompileOptions,
        rng: &mut G,
    ) -> R {
        let mut options = options;
        let backend_facts = runtime.compilation_facts();
        if !backend_facts.is_empty() {
            options.compiler_facts.push('\n');
            options.compiler_facts.push_str(&backend_facts);
        }
        self.build_search_space::<R>(options.clone());
        let runtime = self.search_with_rng(runtime, options, rng);
        // Legality-by-construction burn-down: any post-extraction mask that
        // fired during this compile is a contract violation to fix, not a
        // normal event — always report it.
        if let Some(report) = crate::mask_events::report() {
            println!("{report}");
        }
        runtime
    }

    #[tracing::instrument(skip_all)]
    pub fn search<R: Runtime>(&mut self, runtime: R, options: CompileOptions) -> R {
        let mut rng = rand::rng();
        self.search_with_rng(runtime, options, &mut rng)
    }

    /// Hand the built search space to the runtime, which searches it by
    /// whatever strategy it implements and loads the programs it selects.
    #[tracing::instrument(skip_all)]
    pub fn search_with_rng<R: Runtime, G: rand::Rng>(
        &mut self,
        mut runtime: R,
        options: CompileOptions,
        rng: &mut G,
    ) -> R {
        for (&dim, &value) in &options.search_dims {
            self.set_dim(dim, value);
        }
        let space = self
            .search_space
            .as_ref()
            .expect("build_search_space must run before search");
        assert!(
            options.dim_buckets.is_empty() || options.dim_buckets == space.dim_buckets,
            "dim buckets must be configured in CompileOptions before build_search_space; search cannot change buckets after build",
        );
        runtime.compile(space, &self.dyn_map, &options, rng);
        self.selected_schedule = runtime.selected_schedule();
        runtime
    }
}
