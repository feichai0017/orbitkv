//! Opt-in stage observations; durations are measured within one process.

use std::sync::LazyLock;

static ENABLED: LazyLock<bool> =
    LazyLock::new(|| std::env::var("ORBITKV_TRACE_TRANSFERS").is_ok_and(|value| value == "1"));

pub(crate) fn record(stage: &str, fields: impl FnOnce() -> serde_json::Value) {
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
    log::info!("cache_timeline {fields}");
}
