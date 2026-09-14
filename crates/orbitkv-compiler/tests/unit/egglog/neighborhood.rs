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
    let mut seen = FxHashSet::default();
    let mut chosen = Vec::new();
    while let Some(neighbor) = extractor.local_neighbor(&base, &costs, &mut seen) {
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
        chosen.push(neighbor.to);
    }
    assert_eq!(chosen, ["hot-1", "hot-2"]);
}
