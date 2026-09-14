use super::*;
use crate::egglog_utils::{LateEgglogPass, base};
use crate::{hlir::HLIROps, op::IntoEgglogOp, shape::DimInterval};

const PROGRAM: &str = "(let input (Input 0 \"x\" (F32))) (let output (Output input 0 false))";

fn facts(min: i64, max: i64) -> String {
    base::interval_facts_egglog(
        &[(crate::shape::Symbol::from('s'), DimInterval::new(min, max))]
            .into_iter()
            .collect(),
        [],
    )
}

#[test]
fn bucket_facts_and_unions_do_not_escape_their_instance() {
    let parts = OpTextParts::new(&HLIROps::into_vec(), false);
    let prepared = PreparedEgglog::new(PROGRAM, &parts, true, false).unwrap();
    for (min, max) in [(1, 1), (2, 8), (16, 32), (1, 1)] {
        let mut bucket = prepared.instantiate(&facts(min, max)).unwrap();
        bucket
            .parse_and_run_program(
                None,
                &format!(
                    "(run-schedule (saturate interval_expr))
             (check (= (lower (MVar \"s\")) {min}))
             (check (= (upper (MVar \"s\")) {max}))"
                ),
            )
            .unwrap();
        if min == max {
            bucket
                .parse_and_run_program(None, "(check (= (MVar \"s\") (MNum 1)))")
                .unwrap();
        } else {
            bucket
                .parse_and_run_program(None, "(fail (check (= (MVar \"s\") (MNum 1))))")
                .unwrap();
        }
        // Mutating a dtype or an alias in one instance must not change another.
        bucket
            .parse_and_run_program(None, "(union input (Input 1 \"private\" (F32)))")
            .unwrap();
    }
    let mut untouched = prepared.instantiate("").unwrap();
    untouched
        .parse_and_run_program(
            None,
            "(fail (check (= (lower (MVar \"s\")) 1)))
         (fail (check (= input (Input 1 \"private\" (F32)))))",
        )
        .unwrap();
}

#[test]
fn prepared_and_fresh_runs_preserve_choices_and_late_passes() {
    let passes = [LateEgglogPass::new(
        "(ruleset bucket_late)
         (rule ((= ?out (Output ?input ?id ?persist)))
               ((union ?out ?input)) :ruleset bucket_late)",
        "(run-schedule (saturate bucket_late))",
    )];
    let parts = OpTextParts::new_with_late_passes(&HLIROps::into_vec(), false, &passes);
    let prepared = PreparedEgglog::new(PROGRAM, &parts, true, false).unwrap();
    for bounds in [facts(1, 1), facts(2, 8), facts(1, 1)] {
        let (shared, report) = prepared.run_bucket(&bounds, "output").unwrap();
        let (fresh, fresh_report) = run_egglog_with_report_parts_impl(
            &format!("{PROGRAM}\n{bounds}"),
            "output",
            &parts,
            true,
            false,
        )
        .unwrap();
        assert_eq!(shared.enodes, fresh.enodes);
        assert_eq!(shared.eclasses, fresh.eclasses);
        assert_eq!(shared.roots, fresh.roots);
        assert_eq!(
            report.full.num_matches_per_rule,
            fresh_report.full.num_matches_per_rule
        );
        assert_eq!(report.phases.len(), fresh_report.phases.len());
    }
}
