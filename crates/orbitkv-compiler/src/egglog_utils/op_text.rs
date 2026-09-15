//! Compile registered operation definitions into reusable egglog inputs.

use super::{
    EGraphPostprocess, EgglogSchedulePhase, LateEgglogPass, normalize_late_schedule,
    op_defs_string, primitives,
};
use crate::op::EgglogOp;
use itertools::Itertools;
use rustc_hash::FxHashSet;
use std::sync::Arc;

/// Immutable operation text and pure primitive definitions prepared once for
/// independent bucket saturation. This contains no non-Send operation objects.
pub struct OpTextParts {
    pub(super) primitives: Vec<primitives::EgglogPrimitive>,
    pub(super) op_defs: String,
    /// Declarations owned by registered ops, deduplicated and emitted before
    /// every per-op rewrite. Unlike backend extras, these travel with a raw op
    /// list through direct `run_egglog` callers as well as Runtime compilation.
    pub(super) op_declarations: String,
    /// Backend-provided egglog text (see [`crate::op::Runtime::extra_egglog`]),
    /// spliced after `op_defs` and `op_declarations`, before the rewrite rules.
    /// Empty for core / the reference backend.
    pub(super) extra_egglog: String,
    pub(super) cleanups: String,
    /// Names of op kinds that are eligible for cleanup (cleanup() == true).
    /// Used by the Rust post-processing pass to safely strip HLIR ops only
    /// when an alternative survives in the same eclass.
    pub(crate) cleanable_op_names: FxHashSet<String>,
    pub(super) late_program: String,
    pub(super) rewrites: String,
    pub(super) late_phases: Vec<EgglogSchedulePhase>,
    pub(super) late_postprocesses: Vec<EGraphPostprocess>,
}

impl OpTextParts {
    pub(crate) fn with_extra_egglog(mut self, extra_egglog: String) -> Self {
        self.extra_egglog = extra_egglog;
        self
    }

    pub fn new(ops: &[Arc<Box<dyn EgglogOp>>], cleanup: bool) -> Self {
        Self::new_with_late_passes(ops, cleanup, &[])
    }

    pub fn new_with_late_passes(
        ops: &[Arc<Box<dyn EgglogOp>>],
        cleanup: bool,
        late_passes: &[LateEgglogPass],
    ) -> Self {
        let cleanable_op_names: FxHashSet<String> = ops
            .iter()
            .filter(|op| op.cleanup())
            .map(|op| op.sort().name.to_string())
            .collect();
        let mut seen_declarations = FxHashSet::default();
        let op_declarations = ops
            .iter()
            .flat_map(|op| op.egglog_declarations())
            .filter(|declaration| seen_declarations.insert(declaration.clone()))
            .join("\n");
        Self {
            primitives: primitives::collect(ops.iter().flat_map(|op| op.egglog_primitives())),
            op_defs: op_defs_string(ops),
            op_declarations,
            // Default empty; the backend's Runtime::extra_egglog() is spliced in
            // by the Rt-aware callers (build_search_space) after construction.
            extra_egglog: String::new(),
            // The egglog `cleanup` ruleset deletes HLIR ops unconditionally,
            // even when no kernel rewrite fired in their eclass. On large
            // graphs (e.g. YOLO v11) that produces empty eclasses and the
            // post-processing cascade panics with "No valid graphs present".
            // We always emit an empty cleanup ruleset and instead do
            // conditional cleanup in Rust after egglog finishes.
            cleanups: String::new(),
            rewrites: ops
                .iter()
                .flat_map(|o| o.rewrites())
                .map(|r| r.to_egglog_string())
                .join("\n"),
            cleanable_op_names: if cleanup {
                cleanable_op_names
            } else {
                FxHashSet::default()
            },
            late_program: late_passes.iter().map(|p| p.program.as_str()).join("\n"),
            late_phases: late_passes
                .iter()
                .enumerate()
                .filter_map(|(i, pass)| {
                    let schedule = normalize_late_schedule(&pass.schedule);
                    (!schedule.is_empty()).then(|| EgglogSchedulePhase {
                        name: format!("late pass {:02}", i + 1),
                        schedule,
                    })
                })
                .collect(),
            late_postprocesses: late_passes
                .iter()
                .filter_map(|pass| pass.postprocess.clone())
                .collect(),
        }
    }
}
