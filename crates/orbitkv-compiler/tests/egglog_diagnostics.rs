use std::collections::BTreeMap;

use orbitkv_compiler::{
    egglog_utils::{LateEgglogPass, run_egglog_with_report_and_late_passes},
    hlir::HLIROps,
    op::IntoEgglogOp,
};
use serde_json::Value;
use tracing_subscriber::layer::SubscriberExt;

#[test]
fn observed_phase_counters_equal_the_actual_run_report_and_preserve_the_egraph() {
    let program = "(let t0 (Input 0 \"x\" (F32))) (let t1 (Output t0 0 false))";
    let ops = <HLIROps as IntoEgglogOp>::into_vec();
    let long_prefix = "a-common-rule-prefix-".repeat(8);
    let rule_names = [
        format!("{long_prefix}first"),
        format!("{long_prefix}second"),
    ];
    let declarations = rule_names.iter().map(|name| format!(
        "(rule ((= ?out (Output ?inp ?id ?persist))) ((union ?out ?inp)) :ruleset observed_late :name \"{name}\")"
    )).collect::<Vec<_>>().join("\n");
    let passes = [LateEgglogPass::new(
        format!("(ruleset observed_late)\n{declarations}"),
        "(run-schedule (saturate observed_late))",
    )];
    let (control, _) =
        run_egglog_with_report_and_late_passes(program, "t1", &ops, false, &passes).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("stages.jsonl");
    let (layer, guard) = orbitkv_tracing::stage_trace_layer(&path).unwrap();
    // The standalone installer is process-global before compilation starts.
    // A separate integration-test process exercises that lifecycle without
    // competing thread-local subscribers in unrelated compiler unit tests.
    tracing::subscriber::set_global_default(tracing_subscriber::registry().with(layer)).unwrap();
    let (observed, report) =
        run_egglog_with_report_and_late_passes(program, "t1", &ops, false, &passes).unwrap();
    guard.finish().unwrap();
    assert_eq!(observed.enodes, control.enodes);
    assert_eq!(observed.eclasses, control.eclasses);
    assert_eq!(observed.roots, control.roots);

    let records = std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    let mut measured = BTreeMap::<String, (u64, u64)>::new();
    for row in records
        .iter()
        .filter(|row| row["name"] == "orbitkv.compiler.egglog.rule")
    {
        let fields = &row["fields"];
        let total = measured
            .entry(fields["rule"].as_str().unwrap().to_owned())
            .or_default();
        total.0 += fields["matches"].as_u64().unwrap();
        total.1 += fields["search_apply_ns"].as_u64().unwrap();
    }
    for (name, matches) in &report.full.num_matches_per_rule {
        assert_eq!(measured.get(name).expect(name).0, *matches as u64, "{name}");
    }
    for (name, elapsed) in &report.full.search_and_apply_time_per_rule {
        assert_eq!(u128::from(measured[name].1), elapsed.as_nanos(), "{name}");
    }
    for name in rule_names {
        assert!(
            measured[&name].0 > 0,
            "full rule identity must survive collection"
        );
    }
    assert_eq!(
        records
            .iter()
            .filter(|row| row["name"] == "orbitkv.compiler.egglog.schedule")
            .count(),
        report.phases.len()
    );
}
