//! Connected extraction choices over an immutable egglog snapshot. Unchanged
//! bindings remain those of the measured parent. This module neither recognizes
//! operations nor repairs or rewrites extracted programs.

use super::*;
use crate::search::profile::{ChoiceChange, ChoiceSite};
use std::sync::Arc;

pub(crate) struct LocalNeighbor {
    pub genome: IndexedChoiceSet,
    pub changes: Vec<ChoiceChange>,
    pub cost: f64,
}

struct Proposal {
    bindings: Vec<DenseNode>,
    cost: f64,
}

struct Frame {
    proposal: Proposal,
    extensions: std::vec::IntoIter<DenseNode>,
}

/// Lazy depth-first enumeration: propose an anchor change, then its connected
/// dependency combinations before advancing to the next anchor alternative.
/// No Cartesian product is materialized. The search's attempt/time budgets
/// also charge duplicate and invalid proposals, independently of GPU trials.
#[derive(Default)]
pub(crate) struct Neighborhood {
    site: usize,
    alternative: usize,
    expand: Option<Proposal>,
    stack: Vec<Frame>,
}

impl Neighborhood {
    pub(crate) fn next(
        &mut self,
        extractor: &mut LlirExtractor<'_>,
        base: &IndexedChoiceSet,
        costs: &[(ChoiceSite, f64)],
        max_changes: usize,
    ) -> Option<LocalNeighbor> {
        if let Some(proposal) = self.expand.take()
            && proposal.bindings.len() < max_changes
        {
            let extensions = extractor.dependency_extensions(base, &proposal.bindings);
            self.stack.push(Frame {
                proposal,
                extensions: extensions.into_iter(),
            });
        }
        let proposal = loop {
            if let Some(frame) = self.stack.last_mut() {
                if let Some(extension) = frame.extensions.next() {
                    let mut bindings = frame.proposal.bindings.clone();
                    bindings.push(extension);
                    break Proposal {
                        bindings,
                        cost: frame.proposal.cost,
                    };
                }
                self.stack.pop();
                continue;
            }
            let &(site, cost) = costs.get(self.site)?;
            let pool = extractor.mutation_pool(site.0);
            if let Some(&slot) = pool.get(self.alternative) {
                self.alternative += 1;
                if slot != base.choices[site.0 as usize] {
                    break Proposal {
                        bindings: vec![DenseNode {
                            class: site.0,
                            slot,
                        }],
                        cost,
                    };
                }
            } else {
                self.site += 1;
                self.alternative = 0;
            }
        };
        let neighbor = extractor.apply_proposal(base, &proposal);
        self.expand = Some(proposal);
        Some(neighbor)
    }
}

impl LlirExtractor<'_> {
    pub(super) fn operation_origins(
        &self,
        class: DenseIndex,
        dependencies: &[(DenseIndex, DenseIndex)],
    ) -> Arc<[ChoiceSite]> {
        std::iter::once(ChoiceSite(class))
            .chain(
                dependencies
                    .iter()
                    .filter(|(class, _)| self.indexed_classes[*class as usize].label != "IR")
                    .map(|&(class, _)| ChoiceSite(class)),
            )
            .collect::<Vec<_>>()
            .into()
    }

    pub(crate) fn mutable_site(&mut self, site: ChoiceSite) -> bool {
        self.mutation_pool(site.0).len() > 1
    }

    /// Find the next mutable decision on each selected dependency path. Walk
    /// through immutable metadata and already changed sites, stopping at other
    /// mutable sites. Newly reachable branches use the parent's saved bindings.
    fn dependency_extensions(
        &mut self,
        base: &IndexedChoiceSet,
        changed: &[DenseNode],
    ) -> Vec<DenseNode> {
        let mut pending = Vec::new();
        for &node in changed {
            pending.extend(self.indexed_node_parts(node).2);
        }
        let mut visited = FxHashSet::default();
        let mut sites = Vec::new();
        while let Some(class) = pending.pop() {
            if !visited.insert(class) {
                continue;
            }
            let replacement = changed.iter().find(|node| node.class == class).copied();
            if replacement.is_none()
                && self.indexed_classes[class as usize].searchable
                && self.mutable_site(ChoiceSite(class))
            {
                sites.push(class);
                continue;
            }
            let node = replacement.unwrap_or_else(|| self.indexed_selected(base, class));
            pending.extend(self.indexed_node_parts(node).2);
        }
        sites.sort_unstable();
        sites
            .into_iter()
            .flat_map(|class| {
                self.mutation_pool(class)
                    .iter()
                    .copied()
                    .filter(|&slot| slot != base.choices[class as usize])
                    .map(|slot| DenseNode { class, slot })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn apply_proposal(&self, base: &IndexedChoiceSet, proposal: &Proposal) -> LocalNeighbor {
        let mut genome = base.clone();
        let changes = proposal
            .bindings
            .iter()
            .map(|&DenseNode { class, slot }| {
                let info = &self.indexed_classes[class as usize];
                let from = &info.nodes[base.choices[class as usize] as usize];
                let to = &info.nodes[slot as usize];
                genome.choices[class as usize] = slot;
                genome.hash ^= hash_choice_entry(info.id, from) ^ hash_choice_entry(info.id, to);
                ChoiceChange {
                    class: info.id.to_string(),
                    from: from.to_string(),
                    to: to.to_string(),
                }
            })
            .collect();
        LocalNeighbor {
            genome,
            changes,
            cost: proposal.cost,
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/egglog/neighborhood.rs"]
mod tests;
