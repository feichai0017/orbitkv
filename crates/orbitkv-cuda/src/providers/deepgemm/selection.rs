//! Explicit kernel choices produced by egglog from shape bounds and device facts.
//!
//! Rank limits candidate generation only. The extracted operation owns the
//! complete tile and its admitted row range, never a rank to resolve at launch.

use orbitkv_compiler::egglog_utils::primitives::egglog::{
    ExecutionState, Primitive, Value,
    ast::Span,
    constraint::{SimpleTypeConstraint, TypeConstraint},
    prelude::BaseSort,
    sort::{I64Sort, S, StringSort},
};
use serde::{Deserialize, Serialize};

use super::tiling::{Config, candidates};

/// Search breadth, not a claim that the heuristic finds the optimal tile.
pub(super) const SEARCH_VARIANTS: usize = 4;

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Selection {
    pub(super) row_limit: usize,
    pub(super) config: Config,
}

impl Selection {
    pub(super) fn validate(self) -> anyhow::Result<()> {
        self.config.validate(self.row_limit)
    }

    pub(super) fn validate_rows(self, rows: usize) -> anyhow::Result<()> {
        anyhow::ensure!(
            rows > 0 && rows <= self.row_limit,
            "DeepGEMM rows {rows} outside selected kernel range 1..={}",
            self.row_limit
        );
        Ok(())
    }
}

/// No graph traversal, device queries or I/O: the rewrite supplies every input.
#[derive(Default)]
pub(super) struct TileCandidate;

impl Primitive for TileCandidate {
    fn name(&self) -> &str {
        "deepgemm-tile-candidate"
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        let mut signature = vec![I64Sort.to_arcsort(); 5];
        signature.push(StringSort.to_arcsort());
        SimpleTypeConstraint::new(self.name(), signature, span.clone()).into_box()
    }

    fn apply(&self, state: &mut ExecutionState<'_>, args: &[Value]) -> Option<Value> {
        let values = args
            .iter()
            .map(|value| usize::try_from(state.base_values().unwrap::<i64>(*value)).ok())
            .collect::<Option<Vec<_>>>()?;
        let [row_limit, n, k, num_sms, rank] = values.as_slice() else {
            return None;
        };
        if *rank >= SEARCH_VARIANTS {
            return None;
        }
        let config = *candidates(*row_limit, *n, *k, *num_sms).get(*rank)?;
        let selection = Selection {
            row_limit: *row_limit,
            config,
        };
        let descriptor = serde_json::to_string(&selection).ok()?;
        Some(state.base_values().get::<S>(descriptor.into()))
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/providers/deepgemm/selection.rs"]
mod tests;
