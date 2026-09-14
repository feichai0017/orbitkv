//! Reusable, unsaturated compiler setup and independent bucket saturation.

use super::{
    EgglogRunReport, MAIN_SCHEDULE_MAX_CYCLES, MAIN_SCHEDULE_MAX_TUPLES, OpTextParts,
    SerializedEGraph, egglog_debug, egglog_final_phases, egglog_main_cycle_phases,
    egglog_setup_with_options, metric_duration, print_run_summary_with_log,
    print_serialized_shape_with_log, run_schedule_phase, stage_report, trace_stage_report,
};
use colored::Colorize;
use egglog::{EGraph, ast::Span, prelude::RustSpan, var};
use egglog_reports::ReportLevel;
use rustc_hash::{FxHashMap, FxHashSet};
use std::time::Instant;
use tracing::trace;

/// A model-local template. No bucket facts or saturation results enter this
/// e-graph. Forking copies the rule database, graph and backend facts while
/// preserving independent unions, analyses and rule execution state.
///
/// This is intentionally not a global cache: callers bind one operation table,
/// model program, backend facts and late-pass configuration to its lifetime.
pub(crate) struct PreparedEgglog<'a> {
    template: EGraph,
    parts: &'a OpTextParts,
    interval_analysis: bool,
    log: bool,
}

impl<'a> PreparedEgglog<'a> {
    pub(crate) fn new(
        program: &str,
        parts: &'a OpTextParts,
        interval_analysis: bool,
        log: bool,
    ) -> Result<Self, egglog::Error> {
        let _stage =
            tracing::info_span!(target: "orbitkv::stage", "orbitkv.compiler.egglog.prepare")
                .entered();
        #[cfg(debug_assertions)]
        {
            use std::sync::atomic::{AtomicBool, Ordering};
            static WARNED: AtomicBool = AtomicBool::new(false);
            if log && !WARNED.swap(true, Ordering::Relaxed) {
                eprintln!(
                    "Egglog is running in a debug build; use --release for model-sized compilation."
                );
            }
        }
        let started = Instant::now();
        let code =
            tracing::info_span!(target: "orbitkv::stage", "orbitkv.compiler.egglog.setup_text")
                .in_scope(|| egglog_setup_with_options(program, parts, interval_analysis));
        let mut template = EGraph::default();
        template.set_report_level(if log && egglog_debug() {
            ReportLevel::WithPlan
        } else {
            ReportLevel::TimeOnly
        });
        let commands =
            tracing::info_span!(target: "orbitkv::stage", "orbitkv.compiler.egglog.parse")
                .in_scope(|| template.parser.get_program_from_string(None, &code))?;
        tracing::info_span!(target: "orbitkv::stage", "orbitkv.compiler.egglog.setup_run")
            .in_scope(|| template.run_program(commands))?;
        if log {
            eprintln!(
                "   Egglog setup {} | {} bytes | {} tuples",
                metric_duration(started.elapsed()),
                code.len(),
                template.num_tuples()
            );
        }
        Ok(Self {
            template,
            parts,
            interval_analysis,
            log,
        })
    }

    /// Instantiate one bucket. Only facts local to this bucket are added here;
    /// every main and late schedule still runs to its original fixed point.
    pub(crate) fn run_bucket(
        &self,
        facts: &str,
        root: &str,
    ) -> Result<(SerializedEGraph, EgglogRunReport), egglog::Error> {
        let started = Instant::now();
        let egraph = self.instantiate(facts)?;
        saturate(
            egraph,
            root,
            self.parts,
            self.interval_analysis,
            self.log,
            started,
        )
    }

    fn instantiate(&self, facts: &str) -> Result<EGraph, egglog::Error> {
        let mut egraph =
            tracing::info_span!(target: "orbitkv::stage", "orbitkv.compiler.egglog.fork")
                .in_scope(|| self.template.clone());
        tracing::info_span!(target: "orbitkv::stage", "orbitkv.compiler.egglog.bucket_facts")
            .in_scope(|| egraph.parse_and_run_program(None, facts))?;
        Ok(egraph)
    }
}

pub(super) fn run_egglog_with_report_parts_impl(
    program: &str,
    root: &str,
    parts: &OpTextParts,
    interval_analysis: bool,
    log: bool,
) -> Result<(SerializedEGraph, EgglogRunReport), egglog::Error> {
    let started = Instant::now();
    let prepared = PreparedEgglog::new(program, parts, interval_analysis, log)?;
    // Single-run callers consume setup directly; they pay no template clone.
    saturate(
        prepared.template,
        root,
        parts,
        interval_analysis,
        log,
        started,
    )
}

fn saturate(
    mut egraph: EGraph,
    root: &str,
    op_parts: &OpTextParts,
    use_interval_analysis: bool,
    log: bool,
    started: Instant,
) -> Result<(SerializedEGraph, EgglogRunReport), egglog::Error> {
    let run_stage = tracing::info_span!(target: "orbitkv::stage", "orbitkv.compiler.egglog.run",
        interval_analysis = use_interval_analysis, plan_reports = log && egglog_debug(),
        schedule_count = tracing::field::Empty);
    let _run = run_stage.enter();
    trace!("{}", "Egglog running...".green());
    let mut phases = Vec::new();
    let mut reached_fixed_point = false;
    for cycle in 1..=MAIN_SCHEDULE_MAX_CYCLES {
        let mut cycle_updated = false;
        for phase in egglog_main_cycle_phases(cycle, use_interval_analysis) {
            cycle_updated |= run_schedule_phase(&mut egraph, &mut phases, &phase, log)?;
        }
        if egraph.num_tuples() > MAIN_SCHEDULE_MAX_TUPLES {
            return Err(egglog::Error::BackendError(format!(
                "egglog saturation exceeded tuple budget: {} > {}",
                egraph.num_tuples(),
                MAIN_SCHEDULE_MAX_TUPLES
            )));
        }
        if !cycle_updated {
            reached_fixed_point = true;
            break;
        }
    }
    if !reached_fixed_point {
        return Err(egglog::Error::BackendError(format!(
            "egglog saturation did not reach a fixed point within {MAIN_SCHEDULE_MAX_CYCLES} cycles"
        )));
    }
    for phase in egglog_final_phases(use_interval_analysis) {
        run_schedule_phase(&mut egraph, &mut phases, &phase, log)?;
    }
    for phase in &op_parts.late_phases {
        run_schedule_phase(&mut egraph, &mut phases, phase, log)?;
    }
    run_stage.record("schedule_count", phases.len());
    let full_report = stage_report(&egraph, started.elapsed());
    trace_stage_report("---- Egglog Rule Matches ----", &full_report);

    let run_report = EgglogRunReport {
        full: full_report,
        phases,
        total_time: started.elapsed(),
    };
    print_run_summary_with_log(&run_report, log);
    trace!(
        "{}",
        format!(
            "---- Egglog Total Took {} ----",
            pretty_duration::pretty_duration(&run_report.total_time, None).bold()
        )
        .green()
    );

    let _serialize_stage =
        tracing::info_span!(target: "orbitkv::stage", "orbitkv.compiler.egglog.serialize")
            .entered();
    let (sort, value) = egraph.eval_expr(&var!(root))?;
    let s = egraph.serialize(egglog::SerializeConfig {
        root_eclasses: vec![(sort, value)],
        max_functions: None,
        include_temporary_functions: false,
        max_calls_per_function: None,
    });
    print_serialized_shape_with_log(&s, log);
    // Convert to SerializedEGraph
    let mut classes = FxHashMap::default();
    for (node_id, node) in s.egraph.nodes.iter().filter(|(_, node)| !node.subsumed) {
        classes
            .entry(node.eclass.clone())
            .or_insert(vec![])
            .push(node_id.clone())
    }
    let mut egraph = SerializedEGraph {
        roots: s.egraph.root_eclasses,
        node_to_class: s
            .egraph
            .nodes
            .iter()
            .filter(|(_, enode)| !enode.subsumed)
            .map(|(n, enode)| (n.clone(), enode.eclass.clone()))
            .collect(),
        enodes: s
            .egraph
            .nodes
            .iter()
            .filter(|(_, enode)| !enode.subsumed)
            .map(|(n, enode)| {
                (
                    n.clone(),
                    (
                        enode.op.clone(),
                        enode
                            .children
                            .iter()
                            .map(|n| s.egraph.nodes[n].eclass.clone())
                            .collect(),
                    ),
                )
            })
            .collect(),
        eclasses: s
            .egraph
            .class_data
            .iter()
            .filter_map(|(c, eclass)| {
                classes
                    .get(c)
                    .map(|nodes| (c.clone(), (eclass.typ.clone().unwrap(), nodes.clone())))
            })
            .collect(),
    };
    // Strip out all [...] enodes
    egraph.enodes.retain(|_, (label, _)| label != "[...]");

    // Conditional cleanup: an `Op` enode in our IR has the shape
    // `Op (OpKind ...) (IList ...)`. The first child of `Op` is an OpKind
    // eclass; the OpKind enode's label tells us whether it's an HLIR op
    // (e.g. "MulKind") or a backend op (e.g. "FusionEndKind"). The egglog
    // `cleanup` ruleset can over-delete by removing HLIR variants that have
    // no kernel alternative (e.g. when dtype propagation didn't reach the
    // sub-expression). On conv-heavy graphs that drops Op eclasses to empty
    // and cascades all the way to the root with "No valid graphs present in
    // the e-graph!".
    //
    // We now perform cleanup here in Rust where it's safe: for every Op
    // eclass we look at the OpKind eclasses each Op points at, and strip
    // Op enodes whose kind is cleanable only if a non-cleanable kind exists
    // somewhere in the same Op eclass.
    let cleanable = &op_parts.cleanable_op_names;
    if !cleanable.is_empty() {
        // For each OpKind eclass, find the unique kind label (if any).
        // OpKind eclasses contain enodes like (MulKind ...) / (FusionEndKind ...).
        let mut opkind_class_kinds: FxHashMap<egraph_serialize::ClassId, FxHashSet<String>> =
            FxHashMap::default();
        for (nid, (label, _)) in &egraph.enodes {
            let cid = &egraph.node_to_class[nid];
            // The opkind labels come straight from the serializer's `op` field.
            opkind_class_kinds
                .entry(cid.clone())
                .or_default()
                .insert(label.clone());
        }

        let mut to_strip = Vec::new();
        // Walk Op enodes and decide which to drop.
        // For each Op eclass, group its Op enodes by (cleanable | survivor)
        // based on the OpKind in the first child eclass.
        let mut op_eclass_status: FxHashMap<
            egraph_serialize::ClassId,
            (Vec<egraph_serialize::NodeId>, bool),
        > = FxHashMap::default();
        for (nid, (label, children)) in &egraph.enodes {
            if label != "Op" {
                continue;
            }
            // Identify OpKind via first child eclass.
            let kind_class = match children.first() {
                Some(c) => c,
                None => continue,
            };
            let kinds = match opkind_class_kinds.get(kind_class) {
                Some(k) => k,
                None => continue,
            };
            let cleanable_kind = kinds.iter().all(|k| cleanable.contains(k));
            let op_class = &egraph.node_to_class[nid];
            let entry = op_eclass_status
                .entry(op_class.clone())
                .or_insert_with(|| (Vec::new(), false));
            if cleanable_kind {
                entry.0.push(nid.clone());
            } else {
                entry.1 = true;
            }
        }
        for (_op_class, (cleanable_nodes, has_survivor)) in op_eclass_status {
            if has_survivor {
                to_strip.extend(cleanable_nodes);
            }
        }
        for nid in to_strip {
            egraph.enodes.remove(&nid);
        }
    }

    for postprocess in &op_parts.late_postprocesses {
        postprocess(&mut egraph);
    }

    // Cascade: remove enodes whose children reference empty eclasses
    loop {
        let mut to_remove = vec![];
        for (id, (_, children)) in &egraph.enodes {
            if children.iter().any(|c| {
                egraph
                    .eclasses
                    .get(c)
                    .is_none_or(|(_, nodes)| !nodes.iter().any(|n| egraph.enodes.contains_key(n)))
            }) {
                to_remove.push(id.clone());
            }
        }
        for n in &to_remove {
            egraph.enodes.remove(n);
        }
        if to_remove.is_empty() {
            break;
        }
    }
    // Correct the eclass mapping
    for (_, enodes) in egraph.eclasses.values_mut() {
        enodes.retain(|n| egraph.enodes.contains_key(n));
    }
    egraph.eclasses.retain(|_, (_, c)| !c.is_empty());
    egraph
        .node_to_class
        .retain(|n, _| egraph.enodes.contains_key(n));
    assert!(
        egraph.roots.iter().all(|c| egraph.eclasses.contains_key(c)),
        "No valid graphs present in the e-graph!"
    );

    Ok((egraph, run_report))
}

#[cfg(test)]
#[path = "../../tests/unit/egglog_utils/saturation.rs"]
mod tests;
