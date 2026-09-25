//! Bounded, host-observed costs. Observations never own execution resources.
//!
//! A path is one measurement boundary, not an additive edge: codec, prefetch
//! and SSD restore paths include child operations. No device timing is inferred.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::LazyLock;
use std::time::{Duration, Instant};

use opentelemetry::KeyValue;
use parking_lot::Mutex;

use crate::metrics::core_metrics;

const CAPACITY: usize = 512;
const MIN_SAMPLES: u64 = 4;
const MAX_AGE: Duration = Duration::from_secs(300);
const ALPHA: f64 = 0.2;

static ENABLED: LazyLock<bool> =
    LazyLock::new(|| std::env::var("ORBITKV_COST_OBSERVATIONS").as_deref() == Ok("1"));
static ESTIMATES: LazyLock<Mutex<Estimates>> = LazyLock::new(|| Mutex::new(Estimates::default()));

pub(crate) fn enabled() -> bool {
    *ENABLED
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum CostPath {
    GpuLoadDirect,
    GpuLoadKernel,
    GpuSaveDirect,
    GpuSaveKernel,
    GpuDecode,
    GpuEncode,
    GpuSsdLoad,
    SsdUringRestore,
    SsdCufileRestore,
    GpuSsdSave,
    SsdRead,
    SsdWrite,
    SsdWriteBatch,
    SsdCufileRead,
    SsdCufileWrite,
    SsdPrefetch,
    RemoteRead,
    RemoteAuthorization,
}

impl CostPath {
    fn is_raw_copy(self) -> bool {
        matches!(
            self,
            Self::GpuLoadDirect | Self::GpuLoadKernel | Self::GpuSaveDirect | Self::GpuSaveKernel
        )
    }

    fn is_restore_route(self) -> bool {
        matches!(self, Self::SsdUringRestore | Self::SsdCufileRestore)
    }

    fn label(self) -> &'static str {
        match self {
            Self::GpuLoadDirect => "gpu_load_direct",
            Self::GpuLoadKernel => "gpu_load_kernel",
            Self::GpuSaveDirect => "gpu_save_direct",
            Self::GpuSaveKernel => "gpu_save_kernel",
            Self::GpuDecode => "gpu_decode",
            Self::GpuEncode => "gpu_encode",
            Self::GpuSsdLoad => "gpu_ssd_load",
            Self::SsdUringRestore => "ssd_uring_restore",
            Self::SsdCufileRestore => "ssd_cufile_restore",
            Self::GpuSsdSave => "gpu_ssd_save",
            Self::SsdRead => "ssd_read",
            Self::SsdWrite => "ssd_write",
            Self::SsdWriteBatch => "ssd_write_batch",
            Self::SsdCufileRead => "ssd_cufile_read",
            Self::SsdCufileWrite => "ssd_cufile_write",
            Self::SsdPrefetch => "ssd_prefetch",
            Self::RemoteRead => "remote_read",
            Self::RemoteAuthorization => "remote_authorization",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum Representation {
    Raw,
    Ans,
    Fp8,
    TurboQuant,
    Mixed,
    Unknown,
}

impl From<orbitkv_state::StorageFormat> for Representation {
    fn from(format: orbitkv_state::StorageFormat) -> Self {
        use orbitkv_state::StorageFormat;
        match format {
            StorageFormat::Ans | StorageFormat::Ans16 | StorageFormat::AnsFp8 => Self::Ans,
            StorageFormat::Fp8FromBf16 | StorageFormat::Fp8FromFp16 => Self::Fp8,
            StorageFormat::TurboQuant { .. } => Self::TurboQuant,
            _ => Self::Raw,
        }
    }
}

/// Neither request IDs nor state keys belong here. Resources identify a GPU,
/// disk owner or peer incarnation; they are never exported as metric labels.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct CostKey {
    path: CostPath,
    resource: u64,
    representation: Representation,
    size: u8,
    fragments: u8,
    dma_ranges: u8,
    source_size: u8,
    source_fragments: u8,
    ssd_size: u8,
    ssd_fragments: u8,
}

impl CostKey {
    pub(crate) fn with_dma_ranges(self, ranges: usize) -> Self {
        Self {
            dma_ranges: bucket(ranges as u64),
            ..self
        }
    }

    pub(crate) fn with_path(self, path: CostPath) -> Self {
        Self { path, ..self }
    }
    pub(crate) fn with_ssd_shape(
        self,
        source_bytes: u64,
        source_fragments: usize,
        target_bytes: u64,
        target_fragments: usize,
    ) -> Self {
        Self {
            source_size: bucket(source_bytes),
            source_fragments: bucket(source_fragments as u64),
            ssd_size: bucket(target_bytes),
            ssd_fragments: bucket(target_fragments as u64),
            ..self
        }
    }
    pub(crate) fn with_path_resource(self, path: CostPath, resource: u64) -> Self {
        Self {
            path,
            resource,
            ..self
        }
    }
    pub(crate) fn new(
        path: CostPath,
        resource: u64,
        representation: Representation,
        bytes: u64,
        fragments: usize,
    ) -> Self {
        Self {
            path,
            resource,
            representation,
            size: bucket(bytes),
            fragments: bucket(fragments as u64),
            dma_ranges: 0,
            source_size: 0,
            source_fragments: 0,
            ssd_size: 0,
            ssd_fragments: 0,
        }
    }
}

fn bucket(value: u64) -> u8 {
    (u64::BITS - value.leading_zeros()) as u8
}

pub(crate) fn resource_id(value: &impl Hash) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Outcome {
    Completed,
    Failed,
    Cancelled,
    TimedOut,
    Abandoned,
}

impl Outcome {
    fn label(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
            Self::TimedOut => "timed_out",
            Self::Abandoned => "abandoned",
        }
    }
}

#[derive(Clone, Copy)]
struct Estimate {
    count: u64,
    seconds: f64,
    absolute_error: f64,
    updated: Instant,
}

impl Estimate {
    fn reliable(self, now: Instant) -> bool {
        self.count >= MIN_SAMPLES && now.saturating_duration_since(self.updated) <= MAX_AGE
    }
}

#[derive(Default)]
struct Estimates {
    entries: HashMap<CostKey, Estimate>,
}

impl Estimates {
    fn predict(&self, key: CostKey, now: Instant) -> Option<Estimate> {
        self.entries.get(&key).copied().filter(|e| e.reliable(now))
    }

    fn observe(&mut self, key: CostKey, seconds: f64, now: Instant) -> bool {
        let mut evicted = false;
        if !self.entries.contains_key(&key)
            && self.entries.len() == CAPACITY
            && let Some(oldest) = self
                .entries
                .iter()
                .min_by_key(|(_, e)| e.updated)
                .map(|(k, _)| *k)
        {
            self.entries.remove(&oldest);
            evicted = true;
        }
        let entry = self.entries.entry(key).or_insert(Estimate {
            count: 0,
            seconds,
            absolute_error: 0.0,
            updated: now,
        });
        // An idle path needs new evidence; old counts must not make it warm.
        if now.saturating_duration_since(entry.updated) > MAX_AGE {
            entry.count = 0;
        }
        if entry.count == 0 {
            entry.seconds = seconds;
            entry.absolute_error = 0.0;
        } else {
            entry.absolute_error +=
                ALPHA * ((seconds - entry.seconds).abs() - entry.absolute_error);
            entry.seconds += ALPHA * (seconds - entry.seconds);
        }
        entry.count = entry.count.saturating_add(1);
        entry.updated = now;
        evicted
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
#[path = "../tests/unit/cost.rs"]
mod tests;
