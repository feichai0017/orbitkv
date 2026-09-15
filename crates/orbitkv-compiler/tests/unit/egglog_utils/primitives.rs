use super::*;
use egglog::{constraint::SimpleTypeConstraint, prelude::BaseSort, sort::I64Sort};

#[derive(Default)]
struct Double;

impl Primitive for Double {
    fn name(&self) -> &str {
        "backend-double"
    }
    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(self.name(), vec![I64Sort.to_arcsort(); 2], span.clone())
            .into_box()
    }
    fn apply(&self, state: &mut ExecutionState<'_>, args: &[Value]) -> Option<Value> {
        let value = state.base_values().unwrap::<i64>(args[0]).checked_mul(2)?;
        Some(state.base_values().get::<i64>(value))
    }
}

#[test]
fn duplicate_definitions_register_once_and_clones_keep_independent_facts() {
    let definitions = collect([
        EgglogPrimitive::new::<Double>(),
        EgglogPrimitive::new::<Double>(),
    ]);
    assert_eq!(definitions.len(), 1);
    let mut template = egglog::EGraph::default();
    template.add_primitive(definitions[0].clone());
    template.parse_and_run_program(None, "(relation value-in (i64)) (relation value-out (i64)) (rule ((value-in x)) ((value-out (backend-double x))))").unwrap();
    for value in [3, 7] {
        let mut bucket = template.clone();
        bucket
            .parse_and_run_program(
                None,
                &format!(
                    "(value-in {value}) (run 1) (check (value-out {})) (fail (check (value-out {})))",
                    value * 2,
                    (10 - value) * 2
                ),
            )
            .unwrap();
    }
    template
        .parse_and_run_program(
            None,
            "(fail (check (value-in 3))) (fail (check (value-in 7)))",
        )
        .unwrap();
}

#[derive(Default)]
struct Conflict;

impl Primitive for Conflict {
    fn name(&self) -> &str {
        Double.name()
    }
    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        Double.get_type_constraints(span)
    }
    fn apply(&self, _: &mut ExecutionState<'_>, _: &[Value]) -> Option<Value> {
        None
    }
}

#[test]
#[should_panic(expected = "conflicting egglog primitive implementations")]
fn names_cannot_silently_replace_another_implementation() {
    collect([
        EgglogPrimitive::new::<Double>(),
        EgglogPrimitive::new::<Conflict>(),
    ]);
}
