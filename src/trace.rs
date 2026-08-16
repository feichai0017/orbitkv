use std::io::BufRead;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::CompiledKvPlan;

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq)]
pub struct SglangTraceEvent {
    pub schema: Option<String>,
    pub event: Option<String>,
    pub size_swa: Option<u64>,
    pub swa_available_after: Option<u64>,
    pub size_full: Option<u64>,
    pub full_available_after: Option<u64>,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct TraceSummary {
    pub events: u64,
    pub peak_swa_used_tokens: u64,
    pub peak_full_used_tokens: u64,
    pub minimum_expected_swa_slots: u64,
}

#[derive(Debug, Error)]
pub enum TraceError {
    #[error("failed to read SGLang trace")]
    Io(#[from] std::io::Error),
    #[error("invalid JSON event on line {line}")]
    InvalidJson {
        line: usize,
        #[source]
        source: serde_json::Error,
    },
    #[error("max_active_requests must be positive")]
    ZeroActiveRequests,
    #[error("integer overflow while calculating trace summary")]
    ArithmeticOverflow,
}

/// Reads newline-delimited shadow events emitted by the `SGLang` plugin.
///
/// # Errors
///
/// Returns an error for I/O failure or malformed JSON.
pub fn read_jsonl(reader: impl BufRead) -> Result<Vec<SglangTraceEvent>, TraceError> {
    let mut events = Vec::new();
    for (index, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let event = serde_json::from_str(&line).map_err(|source| TraceError::InvalidJson {
            line: index + 1,
            source,
        })?;
        events.push(event);
    }
    Ok(events)
}

/// Summarizes peak `SGLang` pool occupancy against `OrbitKV`'s compiled SWA bound.
///
/// # Errors
///
/// Returns an error for zero active requests or checked arithmetic overflow.
pub fn summarize_sglang_trace(
    events: &[SglangTraceEvent],
    plan: &CompiledKvPlan,
    max_active_requests: u64,
) -> Result<TraceSummary, TraceError> {
    if max_active_requests == 0 {
        return Err(TraceError::ZeroActiveRequests);
    }

    let mut peak_swa = 0;
    let mut peak_full = 0;
    for event in events {
        if event.schema.as_deref() != Some("orbitkv.sglang-shadow.v1") {
            continue;
        }
        if let (Some(size), Some(available)) = (event.size_swa, event.swa_available_after) {
            peak_swa = peak_swa.max(size.saturating_sub(available));
        }
        if let (Some(size), Some(available)) = (event.size_full, event.full_available_after) {
            peak_full = peak_full.max(size.saturating_sub(available));
        }
    }

    let maximum_bounded_slots = plan
        .classes
        .iter()
        .filter_map(|class| class.slot_count)
        .max()
        .unwrap_or(0);
    let minimum = maximum_bounded_slots
        .checked_mul(plan.page_tokens)
        .and_then(|value| value.checked_mul(max_active_requests))
        .ok_or(TraceError::ArithmeticOverflow)?;
    Ok(TraceSummary {
        events: u64::try_from(events.len()).map_err(|_| TraceError::ArithmeticOverflow)?,
        peak_swa_used_tokens: peak_swa,
        peak_full_used_tokens: peak_full,
        minimum_expected_swa_slots: minimum,
    })
}

#[cfg(test)]
mod tests {
    use crate::{KvClassSpec, KvPlanInput, RetentionKind, compile_plan};

    use super::*;

    #[test]
    fn summarizes_sglang_allocator_headroom() {
        let plan = compile_plan(KvPlanInput {
            page_tokens: 16,
            classes: vec![
                KvClassSpec {
                    name: "full".into(),
                    layers: vec![0],
                    retention: RetentionKind::Full,
                    bytes_per_token_per_layer: 128,
                    window_tokens: None,
                },
                KvClassSpec {
                    name: "swa".into(),
                    layers: vec![1],
                    retention: RetentionKind::Sliding,
                    bytes_per_token_per_layer: 128,
                    window_tokens: Some(32),
                },
            ],
        })
        .unwrap();
        let events = vec![SglangTraceEvent {
            schema: Some("orbitkv.sglang-shadow.v1".into()),
            size_swa: Some(256),
            swa_available_after: Some(160),
            size_full: Some(1024),
            full_available_after: Some(800),
            ..SglangTraceEvent::default()
        }];
        let summary = summarize_sglang_trace(&events, &plan, 2).unwrap();
        assert_eq!(summary.peak_swa_used_tokens, 96);
        assert_eq!(summary.minimum_expected_swa_slots, 96);
    }
}
