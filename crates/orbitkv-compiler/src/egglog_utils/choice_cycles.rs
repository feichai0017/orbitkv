//! Reachable choice-cycle repair shared by deterministic and tuned extraction.

use rand::Rng;
use rustc_hash::FxHashSet;

use super::{
    EGraphChoiceSet, NodeId, SerializedEGraph, cyclic_choice_components,
    eligibility::ChoiceEligibility, is_search_choice_eclass, opkind_metadata_consistent,
    reachable_choice_nodes,
};

pub(super) fn random<'a>(
    egraph: &'a SerializedEGraph,
    choices: &mut EGraphChoiceSet<'a>,
    rng: &mut (impl Rng + ?Sized),
    eligibility: Option<&ChoiceEligibility<'_>>,
) {
    repair_with(egraph, choices, eligibility, &mut |len| {
        rng.random_range(0..len)
    });
}

pub(super) fn deterministic<'a>(
    egraph: &'a SerializedEGraph,
    choices: &mut EGraphChoiceSet<'a>,
    eligibility: Option<&ChoiceEligibility<'_>>,
) {
    repair_with(egraph, choices, eligibility, &mut |_| 0);
}

fn repair_with<'a>(
    egraph: &'a SerializedEGraph,
    choices: &mut EGraphChoiceSet<'a>,
    eligibility: Option<&ChoiceEligibility<'_>>,
    pick: &mut impl FnMut(usize) -> usize,
) {
    // Repair only the reachable selected term. Unreachable eclasses still need
    // entries for a complete genome, but their cycles cannot enter the LLIR.
    for _ in 0..128 {
        let Ok(reachable) = reachable_choice_nodes(egraph, choices) else {
            return;
        };
        let Ok(mut components) = cyclic_choice_components(egraph, choices, &reachable) else {
            return;
        };
        if components.is_empty() {
            return;
        }

        for component in &mut components {
            component.sort_unstable_by(|left, right| left.as_ref().cmp(right.as_ref()));
        }
        components.sort_unstable_by(|left, right| left[0].as_ref().cmp(right[0].as_ref()));
        let mut repairs = Vec::new();
        for component in components {
            crate::mask_events::CHOICE_CYCLE_REPAIR.record();
            let component_set: FxHashSet<&NodeId> = component.iter().copied().collect();
            for selected_node in component {
                let Some(class) = egraph.node_to_class.get(selected_node) else {
                    continue;
                };
                let Some((label, alternatives)) = egraph.eclasses.get(class) else {
                    continue;
                };
                let dependency_score = |candidate: &NodeId| {
                    let mut blocked_classes = FxHashSet::default();
                    for child_class in &egraph.enodes[candidate].1 {
                        let Some((child_label, _)) = egraph.eclasses.get(child_class) else {
                            continue;
                        };
                        if is_search_choice_eclass(child_label)
                            && (child_class == class
                                || choices
                                    .get(child_class)
                                    .is_some_and(|node| component_set.contains(*node)))
                        {
                            blocked_classes.insert(child_class);
                        }
                    }
                    blocked_classes.len()
                };
                let selected_score = dependency_score(selected_node);
                let mut class_repairs = Vec::new();
                let mut best_reduction = 0usize;
                for alternative in alternatives {
                    if alternative == selected_node
                        || eligibility.is_some_and(|value| !value.allows(alternative))
                        || (label == "OpKind" && !opkind_metadata_consistent(egraph, alternative))
                    {
                        continue;
                    }
                    let reduction = selected_score.saturating_sub(dependency_score(alternative));
                    if reduction == 0 {
                        continue;
                    }
                    if reduction > best_reduction {
                        best_reduction = reduction;
                        class_repairs.clear();
                    }
                    if reduction == best_reduction {
                        class_repairs.push((class, alternative));
                    }
                }
                class_repairs.sort_unstable_by(|left, right| left.1.as_ref().cmp(right.1.as_ref()));
                if !class_repairs.is_empty() {
                    repairs.push(class_repairs[pick(class_repairs.len())]);
                }
            }
        }

        if repairs.is_empty() {
            return;
        }
        for (class, alternative) in repairs {
            choices.insert(class, alternative);
        }
    }
}
