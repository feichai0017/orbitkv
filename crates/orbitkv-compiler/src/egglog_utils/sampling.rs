//! Sampling order belongs to an immutable search snapshot, never a hash table.
//! Coverage cycles visit each admitted spelling before repeating it. Repairs
//! may change a draw; only runtime evaluation establishes executable coverage.

use rand::{Rng, seq::SliceRandom};

use super::{
    ClassId, EGraphChoiceSet, NodeId, SerializedEGraph, eligibility::ChoiceEligibility,
    enode_is_loop_input_marker, is_search_choice_eclass, opkind_metadata_consistent,
};

struct Pool<'a> {
    class: &'a ClassId,
    nodes: Vec<&'a NodeId>,
    cycle: Vec<usize>,
}

pub(super) struct ChoicePools<'a> {
    pools: Vec<Pool<'a>>,
}

impl<'a> ChoicePools<'a> {
    pub fn new(graph: &'a SerializedEGraph, eligibility: Option<&ChoiceEligibility<'_>>) -> Self {
        let mut classes = graph.eclasses.iter().collect::<Vec<_>>();
        classes.sort_unstable_by(|(left, _), (right, _)| left.as_ref().cmp(right.as_ref()));
        let pools = classes
            .into_iter()
            .filter(|(_, (label, _))| is_search_choice_eclass(label))
            .map(|(class, (label, nodes))| {
                let mut allowed = nodes
                    .iter()
                    .filter(|node| eligibility.is_none_or(|e| e.allows(node)))
                    .collect::<Vec<_>>();
                if allowed.is_empty() {
                    // Ineligible, unreachable classes still need a binding in
                    // complete serialized genomes. They are never executable.
                    allowed.extend(nodes);
                }
                let consistent = allowed
                    .iter()
                    .copied()
                    .filter(|node| label == "OpKind" && opkind_metadata_consistent(graph, node))
                    .collect::<Vec<_>>();
                let inconsistent =
                    label == "OpKind" && !consistent.is_empty() && consistent.len() < allowed.len();
                let mut pool = if consistent.is_empty() {
                    allowed
                } else {
                    consistent
                };
                if inconsistent {
                    crate::mask_events::OPKIND_INCONSISTENT.record();
                }
                let marker_free = pool
                    .iter()
                    .copied()
                    .filter(|node| !enode_is_loop_input_marker(graph, node))
                    .collect::<Vec<_>>();
                if !marker_free.is_empty() {
                    if marker_free.len() < pool.len() {
                        crate::mask_events::MARKER_CHOICE_RESTRICTED.record();
                    }
                    pool = marker_free;
                }
                pool.sort_unstable_by(|left, right| left.as_ref().cmp(right.as_ref()));
                Pool {
                    class,
                    nodes: pool,
                    cycle: Vec::new(),
                }
            })
            .collect();
        Self { pools }
    }

    pub fn random(&self, rng: &mut (impl Rng + ?Sized)) -> EGraphChoiceSet<'a> {
        self.pools
            .iter()
            .map(|pool| {
                (
                    pool.class,
                    pool.nodes[rng.random_range(0..pool.nodes.len())],
                )
            })
            .collect()
    }

    pub fn coverage(&mut self, rng: &mut (impl Rng + ?Sized)) -> EGraphChoiceSet<'a> {
        self.pools
            .iter_mut()
            .map(|pool| {
                if pool.cycle.is_empty() {
                    pool.cycle.extend(0..pool.nodes.len());
                    pool.cycle.shuffle(rng);
                }
                (pool.class, pool.nodes[pool.cycle.pop().unwrap()])
            })
            .collect()
    }
}

#[cfg(test)]
#[path = "../../tests/unit/egglog/sampling.rs"]
mod tests;
