use super::*;

#[test]
fn local_neighbors_preserve_every_unrelated_binding_and_exhaust_without_repeating() {
    let mut graph = SerializedEGraph {
        enodes: Default::default(),
        eclasses: Default::default(),
        node_to_class: Default::default(),
        roots: vec![ClassId::from("hot")],
    };
    for name in ["hot", "state"] {
        let class = ClassId::from(name);
        let nodes = (0..3)
            .map(|i| NodeId::from(format!("{name}-{i}")))
            .collect::<Vec<_>>();
        for node in &nodes {
            graph.enodes.insert(node.clone(), ("Input".into(), vec![]));
            graph.node_to_class.insert(node.clone(), class.clone());
        }
        graph.eclasses.insert(class, ("IR".into(), nodes));
    }
    let mut extractor = LlirExtractor::new(&graph, &[]);
    let base = extractor.index_named_choices(&[
        ("hot".into(), "hot-0".into()),
        ("state".into(), "state-2".into()),
    ]);
    let costs = [(
        ChoiceSite(extractor.class_to_index[&ClassId::from("hot")]),
        7.0,
    )];
    let mut neighborhood = Neighborhood::default();
    let mut chosen = Vec::new();
    while let Some(neighbor) = neighborhood.next(&mut extractor, &base, &costs, 1) {
        let named = extractor.named_choices(&neighbor.genome);
        assert_eq!(
            named.iter().find(|(class, _)| class == "state").unwrap().1,
            "state-2"
        );
        assert_eq!(
            neighbor
                .genome
                .choices
                .iter()
                .zip(&base.choices)
                .filter(|(a, b)| a != b)
                .count(),
            1
        );
        assert_eq!(
            neighbor.genome.hash,
            extractor.index_named_choices(&named).hash
        );
        chosen.push(neighbor.changes[0].to.clone());
    }
    assert_eq!(chosen, ["hot-1", "hot-2"]);
}

fn dependency_fixture() -> (SerializedEGraph, Vec<(String, String)>) {
    let mut graph = SerializedEGraph {
        enodes: Default::default(),
        eclasses: Default::default(),
        node_to_class: Default::default(),
        roots: vec![ClassId::from("consumer")],
    };
    for (name, children) in [
        ("consumer", vec!["producer"]),
        ("producer", vec!["shared", "shared"]),
        ("shared", vec![]),
        ("unrelated", vec![]),
    ] {
        let class = ClassId::from(name);
        let nodes = (0..2)
            .map(|i| NodeId::from(format!("{name}-{i}")))
            .collect::<Vec<_>>();
        for node in &nodes {
            graph.enodes.insert(
                node.clone(),
                (
                    "Input".into(),
                    children.iter().map(|&name| ClassId::from(name)).collect(),
                ),
            );
            graph.node_to_class.insert(node.clone(), class.clone());
        }
        graph.eclasses.insert(class, ("IR".into(), nodes));
    }
    let base = ["consumer", "producer", "shared", "unrelated"]
        .map(|name| (name.into(), format!("{name}-0")))
        .into();
    (graph, base)
}

#[test]
fn combinations_cross_a_coordinate_valley_and_preserve_unrelated_choices() {
    let (graph, named) = dependency_fixture();
    let mut extractor = LlirExtractor::new(&graph, &[]);
    let base = extractor.index_named_choices(&named);
    let costs = [(
        ChoiceSite(extractor.class_to_index[&ClassId::from("consumer")]),
        10.0,
    )];
    for width in [1, 2, 3] {
        let mut neighborhood = Neighborhood::default();
        let mut seen = FxHashSet::default();
        let mut best = 10;
        let mut longest = 0;
        while let Some(neighbor) = neighborhood.next(&mut extractor, &base, &costs, width) {
            assert!(seen.insert(neighbor.genome.hash));
            assert!(neighbor.changes.len() <= width);
            longest = longest.max(neighbor.changes.len());
            assert!(
                neighbor
                    .changes
                    .iter()
                    .all(|change| change.class != "unrelated")
            );
            let actual = extractor.named_choices(&neighbor.genome);
            assert_eq!(
                neighbor.genome.hash,
                extractor.index_named_choices(&actual).hash
            );
            // The consumer/producer representation only wins when both change.
            let paired = ["consumer", "producer"].iter().all(|name| {
                actual
                    .iter()
                    .any(|(class, node)| class == name && node.ends_with("-1"))
            });
            best = best.min(if paired { 1 } else { 11 });
        }
        assert_eq!(longest, width);
        assert_eq!(best, if width == 1 { 10 } else { 1 });
    }
}

#[test]
fn cyclic_and_newly_reachable_dependencies_have_finite_enumeration() {
    let (mut graph, named) = dependency_fixture();
    graph
        .enodes
        .get_mut(&NodeId::from("consumer-0"))
        .unwrap()
        .1
        .clear();
    graph
        .enodes
        .get_mut(&NodeId::from("shared-1"))
        .unwrap()
        .1
        .push(ClassId::from("consumer"));
    let mut extractor = LlirExtractor::new(&graph, &[]);
    let base = extractor.index_named_choices(&named);
    let costs = [(
        ChoiceSite(extractor.class_to_index[&ClassId::from("consumer")]),
        10.0,
    )];
    let mut neighborhood = Neighborhood::default();
    let mut count = 0;
    while let Some(neighbor) = neighborhood.next(&mut extractor, &base, &costs, 8) {
        count += 1;
        assert!(count <= 4);
        let classes = neighbor
            .changes
            .iter()
            .map(|change| &change.class)
            .collect::<FxHashSet<_>>();
        assert_eq!(classes.len(), neighbor.changes.len());
    }
    assert_eq!(count, 3);
}
