//! Opt-in stage observations; durations are measured within one process.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError};
use std::sync::{LazyLock, mpsc};
use std::time::Duration;

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
static BUFFERED_EVENTS: AtomicU64 = AtomicU64::new(0);
static BUFFER_OVERFLOW: AtomicBool = AtomicBool::new(false);

enum BufferedEvent {
    Event(serde_json::Value),
    Flush(SyncSender<()>),
}

struct BufferedTimeline {
    sender: SyncSender<BufferedEvent>,
}

impl BufferedTimeline {
    fn start(limit: u64) -> Option<Self> {
        let capacity = usize::try_from(limit).unwrap_or(usize::MAX);
        let (sender, receiver) = mpsc::sync_channel(capacity);
        std::thread::Builder::new()
            .name("orbitkv-diagnostic-timeline".into())
            .spawn(move || write_buffered_events(receiver))
            .ok()?;
        Some(Self { sender })
    }

    fn record(&self, event: serde_json::Value) {
        let index = BUFFERED_EVENTS.fetch_add(1, Ordering::Relaxed);
        if index >= *DIAGNOSTIC_LIMIT {
            BUFFER_OVERFLOW.store(true, Ordering::Relaxed);
            return;
        }
        if let Err(TrySendError::Full(_) | TrySendError::Disconnected(_)) =
            self.sender.try_send(BufferedEvent::Event(event))
        {
            BUFFER_OVERFLOW.store(true, Ordering::Relaxed);
        }
    }

    fn flush(&self) {
        if BUFFER_OVERFLOW.swap(false, Ordering::Relaxed) {
            let event = diagnostic_limit_event();
            if self.sender.send(BufferedEvent::Event(event)).is_err() {
                return;
            }
        }
        let (complete, completed) = mpsc::sync_channel(0);
        if self.sender.send(BufferedEvent::Flush(complete)).is_ok() {
            let _ = completed.recv_timeout(Duration::from_secs(30));
        }
    }
}

static BUFFERED_TIMELINE: LazyLock<Option<BufferedTimeline>> = LazyLock::new(|| {
    (*DIAGNOSTIC_LIMIT > 0)
        .then(|| BufferedTimeline::start(*DIAGNOSTIC_LIMIT))
        .flatten()
});

fn write_buffered_events(receiver: Receiver<BufferedEvent>) {
    while let Ok(event) = receiver.recv() {
        match event {
            BufferedEvent::Event(fields) => log::info!("cache_timeline {fields}"),
            BufferedEvent::Flush(complete) => {
                log::logger().flush();
                let _ = complete.send(());
            }
        }
    }
}

fn diagnostic_limit_event() -> serde_json::Value {
    serde_json::json!({
        "stage": "diagnostic_timeline_limit",
        "pid": std::process::id(),
        "monotonic_ns": monotonic_ns(),
        "limit": *DIAGNOSTIC_LIMIT,
    })
}

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
    if let Some(timeline) = BUFFERED_TIMELINE.as_ref() {
        timeline.record(fields);
    } else {
        log::info!("cache_timeline {fields}");
    }
}

/// Drain diagnostic events after request, storage and lifecycle owners stop.
pub fn flush_diagnostic() {
    if let Some(timeline) = BUFFERED_TIMELINE.as_ref() {
        timeline.flush();
    }
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
