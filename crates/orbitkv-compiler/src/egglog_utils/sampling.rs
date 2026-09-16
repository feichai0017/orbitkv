//! Sampling order belongs to an immutable search snapshot, never a hash table.
//! Coverage cycles visit each admitted spelling before repeating it. Repairs
//! may change a draw; only runtime evaluation establishes executable coverage.

use rand::{Rng, seq::SliceRandom};
use rustc_hash::FxHashMap;

use super::{
    ClassId, EGraphChoiceSet, NodeId, SerializedEGraph, eligibility::ChoiceEligibility,
    enode_is_loop_input_marker, is_search_choice_eclass, opkind_metadata_consistent,
};

struct Pool<'a> {
    class: &'a ClassId,
    nodes: Vec<&'a NodeId>,
    priorities: Vec<i64>,
    cycle: Vec<usize>,
}

pub(super) struct ChoicePools<'a> {
    pools: Vec<Pool<'a>>,
    fallback_levels: Vec<i64>,
}

impl<'a> ChoicePools<'a> {
    pub fn new(graph: &'a SerializedEGraph, eligibility: Option<&ChoiceEligibility<'_>>) -> Self {
        let priorities = default_priorities(graph);
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
                pool.sort_unstable_by(|left, right| {
                    candidate_priority(graph, &priorities, left)
                        .cmp(&candidate_priority(graph, &priorities, right))
                        .then_with(|| left.as_ref().cmp(right.as_ref()))
                });
                let priorities = pool
                    .iter()
                    .map(|node| candidate_priority(graph, &priorities, node))
                    .collect();
                Pool {
                    class,
                    nodes: pool,
                    priorities,
                    cycle: Vec::new(),
                }
            })
            .collect::<Vec<_>>();
        let mut fallback_levels = pools
            .iter()
            .flat_map(|pool| pool.priorities.iter().copied())
            .filter(|priority| *priority != i64::MAX)
            .collect::<Vec<_>>();
        fallback_levels.sort_unstable();
        fallback_levels.dedup();
        Self {
            pools,
            fallback_levels,
        }
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

    /// Select explicit production defaults, then disable one declared
    /// priority tier per round. Undeclared e-classes never change merely
    /// because another class needs a legality fallback.
    pub fn stable(&self, round: usize) -> EGraphChoiceSet<'a> {
        let disabled = &self.fallback_levels[..round.min(self.fallback_levels.len())];
        self.pools
            .iter()
            .map(|pool| {
                let selected = pool
                    .priorities
                    .iter()
                    .position(|priority| !disabled.contains(priority))
                    .unwrap_or(0);
                (pool.class, pool.nodes[selected])
            })
            .collect()
    }

    pub fn stable_count(&self) -> usize {
        self.fallback_levels.len() + 1
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

fn default_priorities(graph: &SerializedEGraph) -> FxHashMap<&ClassId, i64> {
    graph
        .enodes
        .iter()
        .filter_map(|(node, (label, children))| {
            if label != "default-priority" || children.len() != 1 {
                return None;
            }
            let value_class = graph.node_to_class.get(node)?;
            let value = graph
                .eclasses
                .get(value_class)?
                .1
                .iter()
                .find_map(|candidate| {
                    let (literal, literal_children) = graph.enodes.get(candidate)?;
                    literal_children
                        .is_empty()
                        .then(|| literal.parse::<i64>().ok())?
                })?;
            Some((&children[0], value))
        })
        .fold(FxHashMap::default(), |mut priorities, (class, value)| {
            priorities
                .entry(class)
                .and_modify(|current| *current = (*current).min(value))
                .or_insert(value);
            priorities
        })
}

fn candidate_priority(
    graph: &SerializedEGraph,
    priorities: &FxHashMap<&ClassId, i64>,
    node: &NodeId,
) -> i64 {
    let (label, children) = &graph.enodes[node];
    if label != "Op" {
        return i64::MAX;
    }
    children
        .first()
        .and_then(|kind| priorities.get(kind))
        .copied()
        .unwrap_or(i64::MAX)
}

#[cfg(test)]
#[path = "../../tests/unit/egglog/sampling.rs"]
mod tests;
