//! Static admissibility of custom-op metadata for a runtime's search.
//!
//! This does not rewrite the e-graph or rank implementations. It removes known
//! non-executable spellings from sampling pools and propagates that constraint
//! through required children. Every finite term made of eligible nodes remains
//! available. The unchanged graph can still be inspected or replayed by other
//! runtimes with different eligibility contracts.

use super::{ClassId, NodeId, SerializedEGraph};
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

pub(super) struct ChoiceEligibility<'a> {
    allowed: FxHashSet<&'a NodeId>,
}

impl<'a> ChoiceEligibility<'a> {
    pub fn build(egraph: &'a SerializedEGraph, custom_ops: &[bool]) -> Result<Self, String> {
        let mut owner: FxHashMap<&NodeId, &ClassId> = FxHashMap::default();
        for (class, (_, nodes)) in &egraph.eclasses {
            for node in nodes {
                if !egraph.enodes.contains_key(node) {
                    return Err(format!(
                        "eligibility: e-class {class} references missing node {node}"
                    ));
                }
                if owner.insert(node, class).is_some() {
                    return Err(format!(
                        "eligibility: node {node} belongs to multiple e-classes"
                    ));
                }
            }
        }
        let mut remaining = FxHashMap::default();
        let mut users: FxHashMap<&ClassId, Vec<&NodeId>> = FxHashMap::default();
        let mut productive = FxHashSet::default();
        let mut ready = VecDeque::new();
        let mut allowed = FxHashSet::default();
        for (node, (kind, children)) in &egraph.enodes {
            let class = *owner
                .get(node)
                .ok_or_else(|| format!("eligibility: node {node} has no owning e-class"))?;
            for child in children {
                if !egraph.eclasses.contains_key(child) {
                    return Err(format!(
                        "eligibility: node {node} references missing e-class {child}"
                    ));
                }
            }
            // CustomOpKind's integer is an immutable table index, not a
            // backend pattern or a preference among legal providers.
            if kind == "CustomOpKind" {
                let id = custom_op_id(egraph, node, children)?;
                let executable = custom_ops.get(id).ok_or_else(|| {
                    format!(
                        "eligibility metadata missing for custom op {id}; supplied {} entries",
                        custom_ops.len()
                    )
                })?;
                if !executable {
                    continue;
                }
            }
            remaining.insert(node, children.len());
            if children.is_empty() {
                allowed.insert(node);
                if productive.insert(class) {
                    ready.push_back(class);
                }
            }
            for child in children {
                users.entry(child).or_default().push(node);
            }
        }
        // A class becomes productive once any alternative has a finite
        // eligible term. Cycles without such a base case remain excluded.
        while let Some(class) = ready.pop_front() {
            for &node in users.get(class).into_iter().flatten() {
                let count = remaining
                    .get_mut(node)
                    .expect("eligible parent has a dependency count");
                *count -= 1;
                if *count == 0 {
                    allowed.insert(node);
                    let parent = owner[node];
                    if productive.insert(parent) {
                        ready.push_back(parent);
                    }
                }
            }
        }
        if egraph.roots.is_empty() {
            return Err("eligibility: search graph has no root".to_owned());
        }
        for root in &egraph.roots {
            if !productive.contains(root) {
                return Err(format!(
                    "no executable initial term for root {root}: missing an eligible custom-op implementation or an acyclic dependency path"
                ));
            }
        }
        Ok(Self { allowed })
    }

    pub fn allows(&self, node: &NodeId) -> bool {
        self.allowed.contains(node)
    }
}

fn custom_op_id(
    egraph: &SerializedEGraph,
    node: &NodeId,
    children: &[ClassId],
) -> Result<usize, String> {
    if children.len() != 2 {
        return Err(format!(
            "eligibility: custom-op node {node} has malformed metadata"
        ));
    }
    let id_class = children
        .first()
        .ok_or_else(|| format!("eligibility: custom-op node {node} has no ID"))?;
    let (_, ids) = &egraph.eclasses[id_class];
    // Integer classes are shared by value across the graph. A custom-op ID can
    // therefore also contain derived spellings such as lower/upper bounds of
    // an unrelated dimension. Only the primitive integer value identifies the
    // immutable custom_ops table entry; its e-class need not be a singleton.
    let mut literal = None;
    for candidate in ids {
        let (value, children) = &egraph.enodes[candidate];
        if !children.is_empty() {
            continue;
        }
        let Ok(value) = value.parse::<i64>() else {
            continue;
        };
        let value = usize::try_from(value)
            .map_err(|_| format!("eligibility: custom-op node {node} has invalid ID {value}"))?;
        if literal.is_some_and(|previous| previous != value) {
            return Err(format!(
                "eligibility: custom-op node {node} has conflicting integer ID literals"
            ));
        }
        literal = Some(value);
    }
    literal.ok_or_else(|| {
        format!("eligibility: custom-op node {node} has invalid ID: no integer literal")
    })
}

#[cfg(test)]
#[path = "../../tests/unit/egglog_utils/eligibility/mod.rs"]
mod tests;
