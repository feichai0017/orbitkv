//! Local extraction over existing egglog choices. No operation matching or
//! rewriting: unrelated bindings of the measured parent remain byte-identical.

use super::*;
use crate::search::profile::ChoiceSite;
use std::sync::Arc;

pub(crate) struct LocalNeighbor {
    pub genome: IndexedChoiceSet,
    pub class: String,
    pub from: String,
    pub to: String,
    pub cost: f64,
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

    pub(crate) fn local_neighbor(
        &mut self,
        base: &IndexedChoiceSet,
        costs: &[(ChoiceSite, f64)],
        seen: &mut FxHashSet<u64>,
    ) -> Option<LocalNeighbor> {
        for &(site, cost) in costs {
            let class = site.0;
            let old = base.choices[class as usize];
            let alternatives = self.mutation_pool(class).to_vec();
            for slot in alternatives {
                if slot == old {
                    continue;
                }
                let info = &self.indexed_classes[class as usize];
                let from = &info.nodes[old as usize];
                let to = &info.nodes[slot as usize];
                let hash =
                    base.hash ^ hash_choice_entry(info.id, from) ^ hash_choice_entry(info.id, to);
                if !seen.insert(hash) {
                    continue;
                }
                let mut neighbor = base.clone();
                neighbor.choices[class as usize] = slot;
                neighbor.hash = hash;
                return Some(LocalNeighbor {
                    genome: neighbor,
                    class: info.id.to_string(),
                    from: from.to_string(),
                    to: to.to_string(),
                    cost,
                });
            }
        }
        None
    }
}

#[cfg(test)]
#[path = "../../tests/unit/egglog/neighborhood.rs"]
mod tests;
