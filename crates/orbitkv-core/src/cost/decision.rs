use std::time::Instant;

use opentelemetry::KeyValue;

use super::estimates::{ESTIMATES, Estimate};
use super::{CostKey, ENABLED, SELECTION_ENABLED};
use crate::metrics::core_metrics;

const MAX_CANDIDATES: usize = 8;
const MIN_RELATIVE_GAIN: f64 = 0.05;

/// Select among routes already proven to have the same HostReady demand.
/// Missing, stale, incompatible or contended evidence preserves the planner's
/// deterministic default; this function never acquires execution resources.
pub(crate) fn select_route(candidates: &[CostKey], default: usize) -> usize {
    if !*ENABLED || !*SELECTION_ENABLED {
        return default;
    }
    let (selected, decision) =
        if default >= candidates.len() || candidates.len() < 2 || candidates.len() > MAX_CANDIDATES
        {
            (default, "unknown")
        } else if let Some(estimates) = ESTIMATES.try_lock() {
            let now = Instant::now();
            let predictions: [_; MAX_CANDIDATES] = std::array::from_fn(|index| {
                candidates
                    .get(index)
                    .and_then(|&key| estimates.predict(key, now))
            });
            choose(candidates, &predictions[..candidates.len()], default)
        } else {
            core_metrics().cost_estimate_dropped.add(1, &[]);
            (default, "contention")
        };
    core_metrics()
        .cost_route_decisions
        .add(1, &[KeyValue::new("decision", decision)]);
    selected
}

fn choose(
    candidates: &[CostKey],
    predictions: &[Option<Estimate>],
    default: usize,
) -> (usize, &'static str) {
    let Some(&default_key) = candidates.get(default) else {
        return (default, "unknown");
    };
    if candidates.len() < 2
        || candidates.len() != predictions.len()
        || predictions.iter().any(Option::is_none)
    {
        return (default, "unknown");
    }
    if candidates
        .iter()
        .any(|&candidate| !default_key.route_comparable(candidate))
    {
        return (default, "incomparable");
    }
    let Some(current) = predictions[default] else {
        return (default, "unknown");
    };
    let mut selected = default;
    let mut best_upper = current.seconds + current.absolute_error;
    let mut faster = false;
    for (index, prediction) in predictions.iter().enumerate() {
        let Some(candidate) = prediction else {
            return (default, "unknown");
        };
        if index == default || candidate.seconds >= current.seconds {
            continue;
        }
        faster = true;
        let gain = (current.seconds - current.absolute_error).max(0.0)
            - (candidate.seconds + candidate.absolute_error);
        let upper = candidate.seconds + candidate.absolute_error;
        if gain > current.seconds * MIN_RELATIVE_GAIN && upper < best_upper {
            selected = index;
            best_upper = upper;
        }
    }
    if selected == default {
        (default, if faster { "within_margin" } else { "default" })
    } else {
        (selected, "selected")
    }
}

#[cfg(test)]
#[path = "../../tests/unit/cost/decision.rs"]
mod tests;
