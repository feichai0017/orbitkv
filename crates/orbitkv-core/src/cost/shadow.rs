use super::estimates::{ESTIMATES, Estimate};
use super::{CostKey, ENABLED};
use crate::metrics::core_metrics;
use opentelemetry::KeyValue;
use std::time::Instant;

// A shadow-only threshold, not a qualified execution-selection policy.
const MIN_RELATIVE_GAIN: f64 = 0.05;

/// Inspect only candidates already proven feasible by the execution owner.
/// No source reads, alternative backend launches, or execution changes occur.
pub(crate) fn shadow(candidates: &[CostKey], selected: usize) {
    if !*ENABLED || selected >= candidates.len() || candidates.len() > 8 {
        return;
    }
    let now = Instant::now();
    let Some(estimates) = ESTIMATES.try_lock() else {
        core_metrics().cost_estimate_dropped.add(1, &[]);
        return;
    };
    // Stack-bounded by the execution owner's supported alternatives.
    let predictions: [_; 8] = std::array::from_fn(|i| {
        candidates
            .get(i)
            .and_then(|&key| estimates.predict(key, now))
    });
    drop(estimates);
    let metrics = core_metrics();
    for (key, prediction) in candidates.iter().zip(predictions) {
        let attributes = [
            KeyValue::new("path", key.path.label()),
            KeyValue::new(
                "evidence",
                if prediction.is_some() {
                    "known"
                } else {
                    "unknown"
                },
            ),
        ];
        metrics.cost_shadow_candidates.add(1, &attributes);
        if let Some(prediction) = prediction {
            metrics
                .cost_shadow_prediction_seconds
                .record(prediction.seconds, &attributes);
            metrics
                .cost_estimate_samples
                .record(prediction.count, &attributes);
            metrics.cost_estimate_age_seconds.record(
                now.saturating_duration_since(prediction.updated)
                    .as_secs_f64(),
                &attributes,
            );
            metrics
                .cost_estimate_error_seconds
                .record(prediction.absolute_error, &attributes);
        }
    }
    let decision = recommendation(candidates, &predictions[..candidates.len()], selected);
    metrics
        .cost_shadow_decisions
        .add(1, &[KeyValue::new("decision", decision)]);
}

fn recommendation(
    candidates: &[CostKey],
    predictions: &[Option<Estimate>],
    selected: usize,
) -> &'static str {
    let Some(&current_key) = candidates.get(selected) else {
        return "unknown";
    };
    if candidates.len() < 2
        || candidates.len() != predictions.len()
        || predictions.iter().any(Option::is_none)
    {
        return "unknown";
    }
    if candidates.iter().any(|&key| !current_key.comparable(key)) {
        return "incomparable";
    }
    let Some(current) = predictions[selected] else {
        return "unknown";
    };
    let mut faster = false;
    for (index, candidate) in predictions.iter().enumerate() {
        let Some(candidate) = candidate else {
            return "unknown";
        };
        if index == selected || candidate.seconds >= current.seconds {
            continue;
        }
        faster = true;
        let gain = (current.seconds - current.absolute_error).max(0.0)
            - (candidate.seconds + candidate.absolute_error);
        if gain > current.seconds * MIN_RELATIVE_GAIN {
            return "different";
        }
    }
    if faster { "within_margin" } else { "agree" }
}

#[cfg(test)]
#[path = "../../tests/unit/cost/shadow.rs"]
mod tests;
