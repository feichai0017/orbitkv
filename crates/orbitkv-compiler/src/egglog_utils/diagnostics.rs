//! Observe reports from the existing schedule without changing its execution.

use std::{collections::BTreeSet, time::Duration};

use egglog_reports::RunReport;

pub(super) fn record_schedule(report: &RunReport) {
    if !tracing::enabled!(target: "orbitkv::stage", tracing::Level::INFO) {
        return;
    }
    // Each report belongs to one run-schedule invocation. Reading the overall
    // e-graph report here would repeatedly count earlier phases.
    let rules = report
        .search_and_apply_time_per_rule
        .keys()
        .chain(report.num_matches_per_rule.keys())
        .collect::<BTreeSet<_>>();
    for rule in rules {
        let elapsed = report
            .search_and_apply_time_per_rule
            .get(rule)
            .copied()
            .unwrap_or(Duration::ZERO);
        let matches = report
            .num_matches_per_rule
            .get(rule)
            .copied()
            .unwrap_or_default();
        tracing::event!(name: "orbitkv.compiler.egglog.rule", target: "orbitkv::stage", tracing::Level::INFO,
            rule = rule.as_ref(), search_apply_ns = elapsed.as_nanos(), matches);
    }
    let rulesets = report
        .search_and_apply_time_per_ruleset
        .keys()
        .chain(report.merge_time_per_ruleset.keys())
        .chain(report.rebuild_time_per_ruleset.keys())
        .collect::<BTreeSet<_>>();
    for ruleset in rulesets {
        let search = report
            .search_and_apply_time_per_ruleset
            .get(ruleset)
            .copied()
            .unwrap_or(Duration::ZERO);
        let merge = report
            .merge_time_per_ruleset
            .get(ruleset)
            .copied()
            .unwrap_or(Duration::ZERO);
        let rebuild = report
            .rebuild_time_per_ruleset
            .get(ruleset)
            .copied()
            .unwrap_or(Duration::ZERO);
        tracing::event!(name: "orbitkv.compiler.egglog.ruleset", target: "orbitkv::stage", tracing::Level::INFO,
            ruleset = ruleset.as_ref(), search_apply_ns = search.as_nanos(),
            merge_ns = merge.as_nanos(), rebuild_ns = rebuild.as_nanos());
    }
}
