//! Opt-in stage observations; durations are measured within one process.

use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};

const MAX_DIAGNOSTIC_EVENTS: u64 = 65_536;

pub static ENABLED: LazyLock<bool> =
    LazyLock::new(|| std::env::var("ORBITKV_TRACE_TRANSFERS").is_ok_and(|value| value == "1"));

static DIAGNOSTIC_LIMIT: LazyLock<u64> = LazyLock::new(|| {
    std::env::var("ORBITKV_DIAGNOSTIC_TIMELINE_LIMIT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
        .min(MAX_DIAGNOSTIC_EVENTS)
});
static DIAGNOSTIC_EVENTS: AtomicU64 = AtomicU64::new(0);

pub fn monotonic_ns() -> u64 {
    let now = rustix::time::clock_gettime(rustix::time::ClockId::Monotonic);
    u64::try_from(now.tv_sec)
        .unwrap_or(0)
        .saturating_mul(1_000_000_000)
        .saturating_add(u64::try_from(now.tv_nsec).unwrap_or(0))
}

pub fn diagnostic_enabled() -> bool {
    *ENABLED && *DIAGNOSTIC_LIMIT > 0
}

pub fn record(stage: &str, fields: impl FnOnce() -> serde_json::Value) {
    if !*ENABLED {
        return;
    }
    let mut fields = fields();
    fields["stage"] = stage.into();
    fields["pid"] = std::process::id().into();
    fields["at_unix_ns"] = (std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64)
        .into();
    fields["monotonic_ns"] = monotonic_ns().into();
    log::info!("cache_timeline {fields}");
}

pub fn record_diagnostic(stage: &str, fields: impl FnOnce() -> serde_json::Value) {
    if !diagnostic_enabled() {
        return;
    }
    let index = DIAGNOSTIC_EVENTS.fetch_add(1, Ordering::Relaxed);
    if index < *DIAGNOSTIC_LIMIT {
        record(stage, || {
            let mut fields = fields();
            fields["diagnostic_event"] = index.into();
            fields
        });
    } else if index == *DIAGNOSTIC_LIMIT {
        record(
            "diagnostic_timeline_limit",
            || serde_json::json!({"limit": *DIAGNOSTIC_LIMIT}),
        );
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn monotonic_clock_advances() {
        let before = super::monotonic_ns();
        let after = super::monotonic_ns();
        assert!(before > 0);
        assert!(after >= before);
    }
}
