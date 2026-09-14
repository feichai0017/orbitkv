use std::collections::{BTreeMap, BTreeSet};

use crate::egglog_utils::{LlirExtractor, random_initial_choice};

use rand::{SeedableRng, rngs::StdRng};

use super::*;

#[test]
fn coverage_visits_every_spelling_before_repeating_each_class() {
    let mut graph = SerializedEGraph {
        enodes: Default::default(),
        eclasses: Default::default(),
        node_to_class: Default::default(),
        roots: vec![],
    };
    let sizes = [2, 3, 5];
    for (site, size) in sizes.iter().enumerate() {
        let class = ClassId::from(format!("site-{site}"));
        let nodes = (0..*size)
            .map(|index| NodeId::from(format!("node-{site}-{index}")))
            .collect::<Vec<_>>();
        for node in &nodes {
            graph.enodes.insert(node.clone(), ("Input".into(), vec![]));
            graph.node_to_class.insert(node.clone(), class.clone());
        }
        graph.eclasses.insert(class, ("IR".into(), nodes));
    }
    let mut pools = ChoicePools::new(&graph, None);
    let mut rng = StdRng::seed_from_u64(101);
    let draws = (0..30)
        .map(|_| pools.coverage(&mut rng))
        .collect::<Vec<_>>();
    for (site, size) in sizes.iter().enumerate() {
        let class = ClassId::from(format!("site-{site}"));
        for cycle in draws.chunks_exact(*size) {
            let selected = cycle
                .iter()
                .map(|draw| draw[&class])
                .collect::<BTreeSet<_>>();
            assert_eq!(
                selected.len(),
                *size,
                "site {site} repeats before exhausting its pool"
            );
        }
    }
}

fn fixture(reverse: bool) -> SerializedEGraph {
    let mut graph = SerializedEGraph {
        enodes: Default::default(),
        eclasses: Default::default(),
        node_to_class: Default::default(),
        roots: vec![ClassId::from("site-0")],
    };
    let mut sites = (0..32).collect::<Vec<_>>();
    if reverse {
        sites.reverse();
        graph.eclasses.reserve(512);
    }
    for site in sites {
        let class = ClassId::from(format!("site-{site}"));
        let mut alternatives = (0..4)
            .map(|alternative| NodeId::from(format!("node-{site}-{alternative}")))
            .collect::<Vec<_>>();
        if reverse {
            alternatives.reverse();
        }
        for node in &alternatives {
            graph.enodes.insert(node.clone(), ("Input".into(), vec![]));
            graph.node_to_class.insert(node.clone(), class.clone());
        }
        graph.eclasses.insert(class, ("IR".into(), alternatives));
    }
    graph
}

fn draws(graph: &SerializedEGraph, seed: u64) -> Vec<BTreeMap<String, String>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..8)
        .map(|_| {
            random_initial_choice(graph, &mut rng)
                .iter()
                .map(|(class, node)| (class.to_string(), node.to_string()))
                .collect()
        })
        .collect()
}

#[test]
fn sampling_ignores_hash_capacity_insertion_and_alternative_order() {
    let forward = fixture(false);
    let reordered = fixture(true);
    assert_eq!(draws(&forward, 19), draws(&reordered, 19));
    assert_ne!(draws(&forward, 19), draws(&forward, 23));
}

#[test]
fn cyclic_choices_are_repaired_in_snapshot_order() {
    let mut graph = fixture(false);
    let root = ClassId::from("site-0");
    let other = ClassId::from("site-1");
    graph
        .enodes
        .get_mut(&NodeId::from("node-0-0"))
        .unwrap()
        .1
        .push(other);
    graph
        .enodes
        .get_mut(&NodeId::from("node-1-0"))
        .unwrap()
        .1
        .push(root);
    let mut reordered = graph.clone();
    reordered.eclasses.reserve(1024);
    for (_, nodes) in reordered.eclasses.values_mut() {
        nodes.reverse();
    }
    for seed in 0..32 {
        assert_eq!(draws(&graph, seed), draws(&reordered, seed), "seed {seed}");
    }
}

#[test]
fn indexed_coverage_uses_the_same_genomes_after_map_reallocation() {
    let graph = fixture(false);
    let mut moved = graph.clone();
    moved.eclasses.reserve(4096);
    let mut first = LlirExtractor::new(&graph, &[]);
    let mut second = LlirExtractor::new(&moved, &[]);
    let a = first.coverage_indexed_generation(
        8,
        &mut Default::default(),
        &mut StdRng::seed_from_u64(13),
    );
    let b = second.coverage_indexed_generation(
        8,
        &mut Default::default(),
        &mut StdRng::seed_from_u64(13),
    );
    assert_eq!(a.len(), 8);
    // Dense genomes are an indexed representation of identical named choices.
    let named_a = a
        .iter()
        .map(|genome| first.named_choices(genome))
        .collect::<Vec<_>>();
    let named_b = b
        .iter()
        .map(|genome| second.named_choices(genome))
        .collect::<Vec<_>>();
    assert_eq!(named_a, named_b);
}
