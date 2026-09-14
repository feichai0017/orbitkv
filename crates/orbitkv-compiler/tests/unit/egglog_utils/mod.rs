use super::{
    EGraphChoiceSet, LateEgglogPass, LlirExtractor, OpTextParts, SerializedEGraph,
    count_choice_sets_up_to, egglog_setup_with_options, random_initial_choice,
    run_egglog_with_late_passes, validate_choice_set,
};
use crate::egglog_utils::api::{Rule, SortDef, sort};
use crate::egglog_utils::base::OP_KIND;
use crate::prelude::FxHashMap;
use crate::{
    hlir::HLIROps,
    op::{EgglogOp, IntoEgglogOp},
};
use egraph_serialize::{ClassId, NodeId};
use rand::{SeedableRng, rngs::StdRng};

const TEST_OP_DECLARATION: &str = "(relation op_owned_test_relation ())";

#[derive(Debug, Default)]
struct OpOwnedDeclarationTest;

impl EgglogOp for OpOwnedDeclarationTest {
    fn sort(&self) -> SortDef {
        sort(OP_KIND, "OpOwnedDeclarationTest", &[])
    }

    fn egglog_declarations(&self) -> Vec<String> {
        vec![TEST_OP_DECLARATION.to_string()]
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![Rule::raw(
            "(rule ((op_owned_test_relation)) ()
                :ruleset expr
                :name \"consume op-owned declaration\")",
        )]
    }

    fn cleanup(&self) -> bool {
        false
    }
}

#[test]
fn op_owned_declarations_are_deduplicated_before_rewrites() {
    let ops = <(OpOwnedDeclarationTest, OpOwnedDeclarationTest)>::into_vec();
    let parts = OpTextParts::new(&ops, false);
    let program = egglog_setup_with_options("", &parts, false);

    assert_eq!(program.matches(TEST_OP_DECLARATION).count(), 1);
    assert!(
        program.find(TEST_OP_DECLARATION).unwrap()
            < program
                .find(":name \"consume op-owned declaration\"")
                .unwrap()
    );
}

// The backend extra-egglog hook (Runtime::extra_egglog) is carried on
// OpTextParts.extra_egglog and must be spliced into the program exactly once,
// after op-owned declarations and before the rewrite rules.
#[test]
fn extra_egglog_is_spliced_between_op_defs_and_rewrites() {
    let marker = "(function tron_is_attn_softmax (IR IR) bool :merge (or old new))";

    // Default: no extra egglog → marker absent.
    let parts = OpTextParts::new(&[], false);
    assert!(parts.extra_egglog.is_empty());
    let plain = egglog_setup_with_options("", &parts, false);
    assert!(!plain.contains(marker));

    // With a backend declaration set, it appears in the program exactly once.
    // (Its position — after op declarations, before rewrite rules — is
    // fixed by the array-literal order in egglog_setup_with_options.)
    let mut parts = OpTextParts::new(&[], false);
    parts.extra_egglog = marker.to_string();
    let program = egglog_setup_with_options("", &parts, false);
    assert_eq!(program.matches(marker).count(), 1, "spliced exactly once");
}

fn eclass(id: &str, label: &str, n_nodes: usize) -> (ClassId, (String, Vec<NodeId>)) {
    (
        ClassId::from(id),
        (
            label.to_string(),
            (0..n_nodes)
                .map(|i| NodeId::from(format!("{id}_{i}")))
                .collect(),
        ),
    )
}

fn egraph(eclasses: Vec<(ClassId, (String, Vec<NodeId>))>) -> SerializedEGraph {
    SerializedEGraph {
        enodes: FxHashMap::default(),
        eclasses: eclasses.into_iter().collect(),
        node_to_class: FxHashMap::default(),
        roots: Vec::new(),
    }
}

#[test]
fn llir_extractor_indexes_eclasses_by_id() {
    let root = ClassId::from("c");
    let mut egraph = egraph(vec![
        eclass("c", "Shape", 1),
        eclass("a", "Shape", 1),
        eclass("b", "Shape", 1),
    ]);
    egraph.roots.push(root);

    let extractor = LlirExtractor::new(&egraph, &[]);
    let ids = extractor
        .indexed_classes
        .iter()
        .map(|class| class.id.as_ref())
        .collect::<Vec<_>>();

    assert_eq!(ids, ["a", "b", "c"]);
}

#[test]
fn counts_ir_and_ilist_choice_sets() {
    let egraph = egraph(vec![
        eclass("a", "IR", 2),
        eclass("b", "IList", 3),
        eclass("op", "OpKind", 5),
        eclass("c", "Shape", 99),
    ]);

    assert_eq!(count_choice_sets_up_to(&egraph, 100), 30);
}

#[test]
fn caps_count_at_limit() {
    let egraph = egraph(vec![eclass("a", "IR", 1_000), eclass("b", "IList", 1_000)]);

    assert_eq!(count_choice_sets_up_to(&egraph, 10), 10);
}

fn dependency_egraph(cyclic: bool) -> SerializedEGraph {
    let a_class = ClassId::from("a");
    let b_class = ClassId::from("b");
    let a_node = NodeId::from("a_node");
    let b_node = NodeId::from("b_node");

    let mut egraph = SerializedEGraph {
        enodes: FxHashMap::default(),
        eclasses: FxHashMap::default(),
        node_to_class: FxHashMap::default(),
        roots: vec![a_class.clone()],
    };
    egraph
        .eclasses
        .insert(a_class.clone(), ("IR".into(), vec![a_node.clone()]));
    egraph
        .eclasses
        .insert(b_class.clone(), ("IR".into(), vec![b_node.clone()]));
    egraph
        .enodes
        .insert(a_node.clone(), ("Output".into(), vec![b_class.clone()]));
    egraph.enodes.insert(
        b_node.clone(),
        if cyclic {
            ("Output".into(), vec![a_class.clone()])
        } else {
            ("Input".into(), Vec::new())
        },
    );
    egraph.node_to_class.insert(a_node, a_class);
    egraph.node_to_class.insert(b_node, b_class);
    egraph
}

fn sole_choices(egraph: &SerializedEGraph) -> EGraphChoiceSet<'_> {
    egraph
        .eclasses
        .iter()
        .map(|(class, (_, nodes))| (class, &nodes[0]))
        .collect()
}

#[test]
fn choice_validation_accepts_acyclic_dependencies() {
    let egraph = dependency_egraph(false);
    let choices = sole_choices(&egraph);
    let ops = <HLIROps as IntoEgglogOp>::into_vec();

    assert_eq!(validate_choice_set(&egraph, &choices, &ops), Ok(()));
}

#[test]
fn choice_validation_rejects_correlated_dependency_cycles() {
    let egraph = dependency_egraph(true);
    let choices = sole_choices(&egraph);
    let ops = <HLIROps as IntoEgglogOp>::into_vec();

    let error = validate_choice_set(&egraph, &choices, &ops).unwrap_err();
    assert!(
        error.contains("dependency cycle"),
        "unexpected validation error: {error}"
    );
}

#[test]
fn random_initial_choice_repairs_reachable_cycles() {
    let mut egraph = dependency_egraph(true);
    let a_class = egraph.roots[0].clone();
    let leaf = NodeId::from("a_leaf");
    egraph
        .enodes
        .insert(leaf.clone(), ("Input".into(), Vec::new()));
    egraph
        .eclasses
        .get_mut(&a_class)
        .expect("root class")
        .1
        .push(leaf.clone());
    egraph.node_to_class.insert(leaf, a_class);
    let ops = <HLIROps as IntoEgglogOp>::into_vec();

    for seed in 0..32 {
        let mut rng = StdRng::seed_from_u64(seed);
        let choices = random_initial_choice(&egraph, &mut rng);
        assert_eq!(
            validate_choice_set(&egraph, &choices, &ops),
            Ok(()),
            "seed {seed} retained a reachable cycle"
        );
    }
}

#[test]
fn runs_late_pass_after_full_cleanup() {
    let ops = <HLIROps as IntoEgglogOp>::into_vec();
    let program = r#"
        (let t0 (Input 0 "" (F32)))
        (let t1 (Output t0 0 false))
    "#;
    let late_pass = LateEgglogPass::new(
        r#"
        (ruleset late_test)
        (rule ((= ?out (Output ?inp ?id ?persist_only)))
              ((union ?out ?inp))
              :ruleset late_test
              :name "late-output-to-input")
        "#,
        "(run-schedule (saturate late_test))",
    );

    let egraph = run_egglog_with_late_passes(program, "t1", &ops, false, &[late_pass])
        .expect("late pass should run");
    let root = egraph.roots.first().expect("root eclass");
    let root_labels: Vec<_> = egraph.eclasses[root]
        .1
        .iter()
        .map(|node| egraph.enodes[node].0.as_str())
        .collect();

    assert!(
        root_labels.contains(&"Input"),
        "late union should add Input to root eclass, got {root_labels:?}"
    );
}
