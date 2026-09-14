use super::*;
use crate::egglog_utils::{
    LlirExtractor, random_initial_choice_with_eligibility, reachable_choice_nodes,
    unresolved_choice_dependencies,
};
use rand::{SeedableRng, rngs::StdRng};

fn graph(entries: &[(&str, &str, &str, &[&str])], root: &str) -> SerializedEGraph {
    let mut graph = SerializedEGraph {
        enodes: Default::default(),
        eclasses: Default::default(),
        node_to_class: Default::default(),
        roots: vec![root.into()],
    };
    for &(class, node, kind, children) in entries {
        let class = ClassId::from(class);
        let node = NodeId::from(node);
        graph.enodes.insert(
            node.clone(),
            (
                kind.to_owned(),
                children.iter().map(|s| ClassId::from(*s)).collect(),
            ),
        );
        let label = if class.as_ref().starts_with("kind") || class.as_ref().starts_with("OpKind-") {
            "OpKind"
        } else if class.as_ref() == "id" || class.as_ref().starts_with("i64-") {
            "i64"
        } else if class.as_ref() == "dtype" || class.as_ref().starts_with("DType-") {
            "DType"
        } else if class.as_ref().starts_with("Expression-") {
            "Expression"
        } else if class.as_ref() == "args" {
            "IList"
        } else {
            "IR"
        };
        graph
            .eclasses
            .entry(class.clone())
            .or_insert_with(|| (label.to_owned(), vec![]))
            .1
            .push(node.clone());
        graph.node_to_class.insert(node, class);
    }
    graph
}

fn alternatives() -> SerializedEGraph {
    graph(
        &[
            ("id", "id0", "0", &[]),
            ("dtype", "dtype_f32", "F32", &[]),
            ("args", "args_nil", "INil", &[]),
            ("kind_bad", "placeholder", "CustomOpKind", &["id", "dtype"]),
            ("kind_good", "provider_a", "ProviderA", &[]),
            ("kind_good", "provider_b", "ProviderB", &[]),
            ("root", "bad", "Op", &["kind_bad", "args"]),
            ("root", "good", "Op", &["kind_good", "args"]),
            ("dead", "dead_op", "Op", &["kind_bad", "args"]),
            ("root", "dead_parent", "Wrapper", &["dead"]),
        ],
        "root",
    )
}

#[test]
fn eligibility_excludes_placeholders_and_required_parents_without_ranking_providers() {
    let graph = alternatives();
    let original = graph.enodes.clone();
    let eligibility = ChoiceEligibility::build(&graph, &[false]).unwrap();
    for allowed in ["provider_a", "provider_b", "good"] {
        assert!(eligibility.allows(&NodeId::from(allowed)));
    }
    for denied in ["placeholder", "bad", "dead_op", "dead_parent"] {
        assert!(!eligibility.allows(&NodeId::from(denied)));
    }
    assert_eq!(
        graph.enodes, original,
        "search eligibility must not rewrite the graph"
    );
    let permissive = ChoiceEligibility::build(&graph, &[true]).unwrap();
    assert!(permissive.allows(&NodeId::from("bad")));
}

#[test]
fn metadata_and_nonproductive_roots_fail_with_errors() {
    let mut graph = alternatives();
    assert!(
        ChoiceEligibility::build(&graph, &[])
            .err()
            .unwrap()
            .contains("metadata missing")
    );
    graph.roots = vec!["dead".into()];
    assert!(
        ChoiceEligibility::build(&graph, &[false])
            .err()
            .unwrap()
            .contains("no executable initial term")
    );
    graph.enodes.get_mut(&NodeId::from("id0")).unwrap().0 = "invalid".to_owned();
    assert!(
        ChoiceEligibility::build(&graph, &[true])
            .err()
            .unwrap()
            .contains("invalid ID")
    );
    graph
        .enodes
        .get_mut(&NodeId::from("placeholder"))
        .unwrap()
        .1 = vec![];
    assert!(
        ChoiceEligibility::build(&graph, &[true])
            .err()
            .unwrap()
            .contains("malformed metadata")
    );
    let graph = super::tests::graph(&[("root", "loop", "Cycle", &["root"])], "root");
    assert!(ChoiceEligibility::build(&graph, &[]).is_err());
}

#[test]
fn custom_op_ids_accept_integer_classes_with_real_model_bound_aliases() {
    // Exact ID-class structure from the Qwen 27B decoder artifact: the same
    // integer is both custom-op table index 24 and an expression bound. The
    // MNum/lower/upper cycle has a finite primitive-literal exit.
    let graph = graph(
        &[
            ("i64-24", "primitive-i64-24", "24", &[]),
            ("i64-24", "function-35-lower", "lower", &["Expression-9"]),
            ("i64-24", "function-35-upper", "upper", &["Expression-9"]),
            ("Expression-9", "function-7-MNum", "MNum", &["i64-24"]),
            ("DType-1", "function-0-Bf16", "Bf16", &[]),
            (
                "OpKind-6022",
                "function-8-CustomOpKind",
                "CustomOpKind",
                &["i64-24", "DType-1"],
            ),
            ("OpKind-6022", "provider", "Provider", &[]),
            ("args", "args_nil", "INil", &[]),
            ("root", "output", "Op", &["OpKind-6022", "args"]),
        ],
        "root",
    );
    let mut metadata = vec![true; 25];
    let custom = NodeId::from("function-8-CustomOpKind");
    let permissive = ChoiceEligibility::build(&graph, &metadata).unwrap();
    assert!(permissive.allows(&custom));
    for bound in ["function-35-lower", "function-35-upper", "function-7-MNum"] {
        assert!(permissive.allows(&NodeId::from(bound)));
    }
    metadata[24] = false;
    let restricted = ChoiceEligibility::build(&graph, &metadata).unwrap();
    assert!(!restricted.allows(&custom));
    assert!(restricted.allows(&NodeId::from("provider")));
    assert!(restricted.allows(&NodeId::from("output")));
}

#[test]
fn custom_op_ids_reject_missing_negative_and_conflicting_literals() {
    for (literals, reason) in [
        (vec![], "no integer literal"),
        (vec!["-1"], "invalid ID -1"),
        (vec!["0", "1"], "conflicting integer ID literals"),
    ] {
        let mut graph = alternatives();
        graph.enodes.remove(&NodeId::from("id0"));
        graph.node_to_class.remove(&NodeId::from("id0"));
        graph
            .eclasses
            .get_mut(&ClassId::from("id"))
            .unwrap()
            .1
            .clear();
        for (index, literal) in literals.into_iter().enumerate() {
            let node = NodeId::from(format!("literal-{index}"));
            graph
                .enodes
                .insert(node.clone(), (literal.to_owned(), vec![]));
            graph
                .eclasses
                .get_mut(&ClassId::from("id"))
                .unwrap()
                .1
                .push(node.clone());
            graph.node_to_class.insert(node, ClassId::from("id"));
        }
        assert!(
            ChoiceEligibility::build(&graph, &[true, true])
                .err()
                .unwrap()
                .contains(reason)
        );
    }
}

#[test]
#[ignore = "requires decoder artifacts in ORBITKV_ELIGIBILITY_ARTIFACT_PATHS"]
fn real_serialized_model_artifacts_have_productive_custom_op_eligibility() {
    let paths = std::env::var_os("ORBITKV_ELIGIBILITY_ARTIFACT_PATHS")
        .expect("provide a platform-separated list of decoder JSON artifact paths");
    let mut graph_count = 0;
    let mut alias_count = 0;
    for path in std::env::split_paths(&paths) {
        let bytes = std::fs::read(&path).unwrap();
        let artifact: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let buckets = artifact["schedule"]["buckets"].as_array().unwrap();
        assert!(!buckets.is_empty());
        for (bucket, entry) in buckets.iter().enumerate() {
            let graph: SerializedEGraph = serde_json::from_value(entry["egraph"].clone()).unwrap();
            // Derive metadata capacity independently from primitive ID leaves;
            // all-true flags exercise parsing/productivity without importing a
            // particular runtime's provider policy into this generic test.
            let mut table_len = 0;
            let mut custom_nodes = vec![];
            let mut aliases = 0;
            for (node, (kind, children)) in &graph.enodes {
                if kind != "CustomOpKind" {
                    continue;
                }
                assert_eq!(children.len(), 2);
                let (_, ids) = &graph.eclasses[&children[0]];
                aliases += usize::from(ids.len() > 1);
                for id in ids {
                    let (literal, dependencies) = &graph.enodes[id];
                    if dependencies.is_empty()
                        && let Ok(value) = literal.parse::<usize>()
                    {
                        table_len = table_len.max(value.checked_add(1).unwrap());
                    }
                }
                custom_nodes.push(node);
            }
            assert!(!custom_nodes.is_empty());
            let eligibility = ChoiceEligibility::build(&graph, &vec![true; table_len])
                .unwrap_or_else(|error| panic!("{} bucket {bucket}: {error}", path.display()));
            assert!(custom_nodes.iter().all(|node| eligibility.allows(node)));
            eprintln!(
                "artifact={} bucket={bucket} enodes={} custom_ops={} aliased_ids={aliases} metadata_entries={table_len}: eligible",
                path.display(),
                graph.enodes.len(),
                custom_nodes.len(),
            );
            graph_count += 1;
            alias_count += aliases;
        }
    }
    assert!(graph_count > 0);
    assert!(
        alias_count > 0,
        "fixtures must cover non-singleton ID classes"
    );
}

#[test]
fn initial_sampling_and_mutation_keep_all_and_only_eligible_reachable_choices() {
    let graph = alternatives();
    let mut extractor = LlirExtractor::new(&graph, &[]);
    extractor.set_custom_op_eligibility(&[false]).unwrap();
    let mut rng = StdRng::seed_from_u64(59);
    let mut providers = FxHashSet::default();
    for _ in 0..100 {
        let genome = extractor.random_indexed_choice(&mut rng);
        let named = extractor.named_choices(&genome);
        assert_eq!(
            named.iter().find(|(class, _)| class == "root").unwrap().1,
            "good"
        );
        providers.insert(
            named
                .iter()
                .find(|(class, _)| class == "kind_good")
                .unwrap()
                .1
                .clone(),
        );
        let offspring = extractor.extract_reachable_indexed_generation(
            &genome,
            4,
            3,
            &mut FxHashSet::default(),
            &mut rng,
        );
        for child in offspring {
            assert_eq!(
                extractor
                    .named_choices(&child)
                    .iter()
                    .find(|(class, _)| class == "root")
                    .unwrap()
                    .1,
                "good"
            );
        }
    }
    assert_eq!(
        providers.len(),
        2,
        "both legal provider alternatives must remain searchable"
    );
}

#[test]
fn cycle_repairs_never_restore_ineligible_escape_paths() {
    let mut graph = alternatives();
    let node = NodeId::from("cycle");
    let root = ClassId::from("root");
    graph
        .enodes
        .insert(node.clone(), ("Cycle".to_owned(), vec![root.clone()]));
    graph.eclasses.get_mut(&root).unwrap().1.push(node.clone());
    graph.node_to_class.insert(node, root);
    let eligibility = ChoiceEligibility::build(&graph, &[false]).unwrap();
    let mut rng = StdRng::seed_from_u64(127);
    for _ in 0..100 {
        let choices = random_initial_choice_with_eligibility(&graph, &mut rng, Some(&eligibility));
        let reachable = reachable_choice_nodes(&graph, &choices).unwrap();
        assert!(reachable.iter().all(|node| eligibility.allows(node)));
        assert!(
            unresolved_choice_dependencies(&graph, &choices, &reachable)
                .unwrap()
                .is_empty()
        );
    }
}
