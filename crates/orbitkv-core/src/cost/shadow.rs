use super::estimates::{ESTIMATES, Estimate};
use super::{CostKey, ENABLED};
use crate::metrics::core_metrics;
use opentelemetry::KeyValue;
use std::time::Instant;

/// Inspect only candidates already proven feasible by the execution owner.
/// No source reads, alternative backend launches, or execution changes occur.
pub(crate) fn shadow(candidates: &[CostKey], selected: usize) {
    if !*ENABLED || selected >= candidates.len() || candidates.is_empty() {
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
    if candidates.len() > predictions.len() {
        return;
    }
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
    let decision = recommendation(&predictions[..candidates.len()], selected);
    metrics
        .cost_shadow_decisions
        .add(1, &[KeyValue::new("decision", decision)]);
}

fn recommendation(predictions: &[Option<Estimate>], selected: usize) -> &'static str {
    if predictions.len() < 2 || predictions.iter().any(Option::is_none) {
        return "unknown";
    }
    let best = predictions
        .iter()
        .enumerate()
        .filter_map(|(i, e)| e.map(|e| (i, e.seconds)))
        .min_by(|a, b| a.1.total_cmp(&b.1));
    if best.is_some_and(|(i, _)| i == selected) {
        "agree"
    } else {
        "different"
    }
}

#[cfg(test)]
#[path = "../../tests/unit/cost/shadow.rs"]
mod tests;
