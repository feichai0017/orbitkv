//! Runtime feedback orders extraction choices; it never substitutes for a
//! complete candidate's fitness measurement or its semantic/resource checks.

use petgraph::stable_graph::NodeIndex;
use rustc_hash::FxHashMap;
use std::sync::Arc;

use super::CandidateId;

/// Cost of one measured execution region. Units are runtime-defined and must
/// be consistent within a profile. CUDA reports seconds from device events.
#[derive(Debug, Clone)]
pub struct ProfiledRegion {
    pub nodes: Vec<NodeIndex>,
    pub cost: f64,
}

/// Exact snapshot-local decision that produced a local neighbor.
#[derive(Debug, Clone)]
pub struct TargetedChoice {
    pub parent: CandidateId,
    pub class: String,
    pub from: String,
    pub to: String,
    pub cost: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct ChoiceSite(pub(crate) u32);

pub(crate) struct ParentProfile {
    pub id: CandidateId,
    pub costs: Vec<(ChoiceSite, f64)>,
}

pub(crate) fn aggregate_costs(
    origins: &[Arc<[ChoiceSite]>],
    regions: &[ProfiledRegion],
    mut mutable: impl FnMut(ChoiceSite) -> bool,
) -> Vec<(ChoiceSite, f64)> {
    let mut costs = FxHashMap::<ChoiceSite, f64>::default();
    for region in regions {
        if !region.cost.is_finite() || region.cost <= 0.0 {
            continue;
        }
        let mut sites = region
            .nodes
            .iter()
            .filter_map(|node| origins.get(node.index()))
            .flat_map(|choices| choices.iter().copied())
            .filter(|&site| mutable(site))
            .collect::<Vec<_>>();
        sites.sort_unstable();
        sites.dedup();
        // A fused region has one measured duration, not an independent timing
        // for every constituent. Share its cost over distinct mutable choices.
        let share = region.cost / sites.len().max(1) as f64;
        for site in sites {
            *costs.entry(site).or_default() += share;
        }
    }
    let mut costs = costs
        .into_iter()
        .filter(|(_, cost)| cost.is_finite())
        .collect::<Vec<_>>();
    costs.sort_unstable_by(|(a, ac), (b, bc)| bc.total_cmp(ac).then_with(|| a.cmp(b)));
    costs
}

#[cfg(test)]
#[path = "../../tests/unit/search/profile.rs"]
mod tests;
