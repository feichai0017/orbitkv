use super::estimates::{ESTIMATES, Estimate};
use super::{CostKey, ENABLED};
use crate::metrics::core_metrics;
use opentelemetry::KeyValue;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Outcome {
    Completed,
    Failed,
    Cancelled,
    #[cfg(feature = "mooncake")]
    TimedOut,
    Abandoned,
}

impl Outcome {
    fn label(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
            #[cfg(feature = "mooncake")]
            Self::TimedOut => "timed_out",
            Self::Abandoned => "abandoned",
        }
    }
}

/// Lives with the physical operation, including a detached completion owner.
/// Dropping a caller's future is never evidence of physical service completion.
pub(crate) struct Observation(Option<Running>);

struct Running {
    key: CostKey,
    logical_bytes: Option<u64>,
    enqueued: Instant,
    admitted: Option<Instant>,
    submitted: Option<Instant>,
    prediction: Option<Estimate>,
}

impl Observation {
    pub(crate) fn disabled() -> Self {
        Self(None)
    }

    pub(crate) fn new(key: CostKey, logical_bytes: Option<u64>) -> Self {
        if !*ENABLED {
            return Self(None);
        }
        let now = Instant::now();
        let prediction = if key.path.is_restore_route() {
            ESTIMATES
                .try_lock()
                .and_then(|estimates| estimates.predict(key, now))
        } else {
            None
        };
        Self(Some(Running {
            key,
            logical_bytes,
            enqueued: now,
            admitted: None,
            submitted: None,
            prediction,
        }))
    }

    pub(crate) fn admitted(&mut self) {
        if let Some(running) = &mut self.0 {
            running.admitted.get_or_insert_with(Instant::now);
        }
    }

    /// Actual descriptors refine raw-copy shape without restarting queue timing.
    /// Composite paths and already submitted operations retain their own keys.
    pub(crate) fn refine_raw_copy(&mut self, key: CostKey, bytes: u64) -> bool {
        let Some(running) = &mut self.0 else {
            return false;
        };
        if !key.path.is_raw_copy() || running.key.path != key.path || running.submitted.is_some() {
            return false;
        }
        running.key = key;
        running.logical_bytes = Some(bytes);
        true
    }

    pub(crate) fn submitted(&mut self) {
        if let Some(running) = &mut self.0
            && running.submitted.is_none()
        {
            let now = Instant::now();
            running.admitted.get_or_insert(now);
            running.submitted = Some(now);
            if !running.key.path.is_restore_route() {
                running.prediction = ESTIMATES
                    .try_lock()
                    .and_then(|estimates| estimates.predict(running.key, now));
            }
        }
    }

    pub(crate) fn finish(mut self, outcome: Outcome, actual_io_bytes: Option<u64>) {
        if let Some(running) = self.0.take() {
            running.finish(outcome, actual_io_bytes, Instant::now());
        }
    }
}

impl Drop for Observation {
    fn drop(&mut self) {
        if let Some(running) = self.0.take() {
            let outcome = if running.submitted.is_some() {
                Outcome::Abandoned
            } else {
                Outcome::Cancelled
            };
            running.finish(outcome, None, Instant::now());
        }
    }
}

impl Running {
    fn estimate_sample(&self, outcome: Outcome, now: Instant) -> Option<f64> {
        self.service_sample(outcome, now).map(|service| {
            if self.key.path.is_restore_route() {
                now.saturating_duration_since(self.enqueued).as_secs_f64()
            } else {
                service
            }
        })
    }

    fn service_sample(&self, outcome: Outcome, now: Instant) -> Option<f64> {
        if outcome != Outcome::Completed {
            return None;
        }
        self.submitted
            .map(|submitted| now.saturating_duration_since(submitted).as_secs_f64())
    }

    fn finish(self, outcome: Outcome, actual_io_bytes: Option<u64>, now: Instant) {
        let metrics = core_metrics();
        let attributes = [
            KeyValue::new("path", self.key.path.label()),
            KeyValue::new("outcome", outcome.label()),
        ];
        metrics.cost_operations.add(1, &attributes);
        if let Some(bytes) = self.logical_bytes {
            metrics.cost_logical_bytes.add(bytes, &attributes);
        } else {
            metrics.cost_logical_unknown.add(1, &attributes);
        }
        if let Some(bytes) = actual_io_bytes {
            metrics.cost_io_bytes.add(bytes, &attributes);
        } else {
            metrics.cost_io_unknown.add(1, &attributes);
        }
        let record = |stage: &'static str, start: Instant, end: Instant| {
            metrics.cost_stage_seconds.record(
                end.saturating_duration_since(start).as_secs_f64(),
                &[
                    attributes[0].clone(),
                    attributes[1].clone(),
                    KeyValue::new("stage", stage),
                ],
            );
        };
        record("total", self.enqueued, now);
        if let Some(admitted) = self.admitted {
            record("queue", self.enqueued, admitted);
            record("admission", admitted, self.submitted.unwrap_or(now));
        }
        // Failure/cancellation elapsed time never becomes service evidence. It is
        // never treated as an exact completed service time or used for fitting.
        if let Some(seconds) = self.estimate_sample(outcome, now) {
            let submitted = self.submitted.unwrap_or(now);
            record("service", submitted, now);
            let path = [attributes[0].clone()];
            if let Some(prediction) = self.prediction {
                metrics
                    .cost_prediction_absolute_error_seconds
                    .record((seconds - prediction.seconds).abs(), &path);
            }
            if let Some(mut estimates) = ESTIMATES.try_lock() {
                let evicted = estimates.observe(self.key, seconds, now);
                drop(estimates);
                if evicted {
                    metrics.cost_estimate_evictions.add(1, &[]);
                }
            } else {
                metrics.cost_estimate_dropped.add(1, &[]);
            }
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/cost/observation.rs"]
mod tests;
