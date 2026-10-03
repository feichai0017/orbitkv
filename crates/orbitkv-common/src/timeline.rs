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
static BUFFERED_EVENTS: AtomicU64 = AtomicU64::new(0);
static BUFFER_OVERFLOW: AtomicBool = AtomicBool::new(false);

const HAS_BLOCKS: u16 = 1 << 0;
const HAS_BYTES: u16 = 1 << 1;
const HAS_GROUP: u16 = 1 << 2;
const HAS_PENDING_BLOCKS: u16 = 1 << 3;
const HAS_INFLIGHT_WRITES: u16 = 1 << 4;
const HAS_MAX_INFLIGHT_WRITES: u16 = 1 << 5;
const HAS_ELAPSED_US: u16 = 1 << 6;

#[derive(Clone, Copy, Debug, Default)]
pub struct DiagnosticFields {
    request_id: u64,
    session_epoch: u64,
    session_token: u64,
    blocks: u64,
    bytes: u64,
    group: u64,
    pending_blocks: u64,
    inflight_writes: u64,
    max_inflight_writes: u64,
    elapsed_us: u64,
    present: u16,
    success: Option<bool>,
    outcome: Option<&'static str>,
}

impl DiagnosticFields {
    pub const fn operation(request_id: u64, session_epoch: u64, session_token: u64) -> Self {
        Self {
            request_id,
            session_epoch,
            session_token,
            blocks: 0,
            bytes: 0,
            group: 0,
            pending_blocks: 0,
            inflight_writes: 0,
            max_inflight_writes: 0,
            elapsed_us: 0,
            present: 0,
            success: None,
            outcome: None,
        }
    }

    pub const fn with_operation(
        mut self,
        request_id: u64,
        session_epoch: u64,
        session_token: u64,
    ) -> Self {
        self.request_id = request_id;
        self.session_epoch = session_epoch;
        self.session_token = session_token;
        self
    }

    pub const fn blocks(mut self, blocks: usize) -> Self {
        self.blocks = blocks as u64;
        self.present |= HAS_BLOCKS;
        self
    }

    pub const fn bytes(mut self, bytes: u64) -> Self {
        self.bytes = bytes;
        self.present |= HAS_BYTES;
        self
    }

    pub const fn group(mut self, group: u32) -> Self {
        self.group = group as u64;
        self.present |= HAS_GROUP;
        self
    }

    pub const fn pending_blocks(mut self, pending_blocks: usize) -> Self {
        self.pending_blocks = pending_blocks as u64;
        self.present |= HAS_PENDING_BLOCKS;
        self
    }

    pub const fn inflight_writes(mut self, inflight_writes: usize) -> Self {
        self.inflight_writes = inflight_writes as u64;
        self.present |= HAS_INFLIGHT_WRITES;
        self
    }

    pub const fn max_inflight_writes(mut self, max_inflight_writes: usize) -> Self {
        self.max_inflight_writes = max_inflight_writes as u64;
        self.present |= HAS_MAX_INFLIGHT_WRITES;
        self
    }

    pub const fn elapsed_us(mut self, elapsed_us: u64) -> Self {
        self.elapsed_us = elapsed_us;
        self.present |= HAS_ELAPSED_US;
        self
    }

    pub const fn success(mut self, success: bool) -> Self {
        self.success = Some(success);
        self
    }

    pub const fn outcome(mut self, outcome: &'static str) -> Self {
        self.outcome = Some(outcome);
        self
    }
}

struct DiagnosticEvent {
    index: u64,
    stage: &'static str,
    pid: u32,
    monotonic_ns: u64,
    fields: DiagnosticFields,
}

const FIXED_TEXT_BYTES: usize = 96;

struct FixedText {
    bytes: [u8; FIXED_TEXT_BYTES],
    len: u8,
    truncated: bool,
}

impl FixedText {
    fn new(value: &str) -> Self {
        let mut bytes = [0; FIXED_TEXT_BYTES];
        let len = value.len().min(FIXED_TEXT_BYTES);
        bytes[..len].copy_from_slice(&value.as_bytes()[..len]);
        Self {
            bytes,
            len: len as u8,
            truncated: value.len() > len,
        }
    }

    fn as_str(&self) -> std::borrow::Cow<'_, str> {
        String::from_utf8_lossy(&self.bytes[..self.len as usize])
    }
}

struct QueryPathEvent {
    stage: &'static str,
    pid: u32,
    monotonic_ns: u64,
    request_id: FixedText,
    instance_id: FixedText,
    group_id: u32,
    warmup: bool,
    prepare: bool,
    elapsed_us: u64,
    hit_blocks: usize,
}

enum BufferedEvent {
    Json(serde_json::Value),
    Diagnostic(DiagnosticEvent),
    QueryPath(QueryPathEvent),
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

    fn reserve(&self) -> Option<u64> {
        let index = BUFFERED_EVENTS.fetch_add(1, Ordering::Relaxed);
        if index >= *DIAGNOSTIC_LIMIT {
            BUFFER_OVERFLOW.store(true, Ordering::Relaxed);
            return None;
        }
        Some(index)
    }

    fn send(&self, event: BufferedEvent) {
        if let Err(TrySendError::Full(_) | TrySendError::Disconnected(_)) =
            self.sender.try_send(event)
        {
            BUFFER_OVERFLOW.store(true, Ordering::Relaxed);
        }
    }

    fn record_json(&self, event: serde_json::Value) {
        if self.reserve().is_some() {
            self.send(BufferedEvent::Json(event));
        }
    }

    fn record_diagnostic(&self, stage: &'static str, fields: DiagnosticFields) {
        if let Some(index) = self.reserve() {
            self.send(BufferedEvent::Diagnostic(DiagnosticEvent {
                index,
                stage,
                pid: std::process::id(),
                monotonic_ns: monotonic_ns(),
                fields,
            }));
        }
    }

    #[allow(clippy::too_many_arguments, reason = "fixed diagnostic event schema")]
    fn record_query_path(
        &self,
        stage: &'static str,
        request_id: &str,
        instance_id: &str,
        group_id: u32,
        warmup: bool,
        prepare: bool,
        elapsed_us: u64,
        hit_blocks: usize,
    ) {
        if self.reserve().is_some() {
            self.send(BufferedEvent::QueryPath(QueryPathEvent {
                stage,
                pid: std::process::id(),
                monotonic_ns: monotonic_ns(),
                request_id: FixedText::new(request_id),
                instance_id: FixedText::new(instance_id),
                group_id,
                warmup,
                prepare,
                elapsed_us,
                hit_blocks,
            }));
        }
    }

    fn flush(&self) {
        if BUFFER_OVERFLOW.swap(false, Ordering::Relaxed) {
            let event = diagnostic_limit_event();
            if self.sender.send(BufferedEvent::Json(event)).is_err() {
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
            BufferedEvent::Json(fields) => log::info!("cache_timeline {fields}"),
            BufferedEvent::Diagnostic(event) => {
                let fields = diagnostic_event_json(event);
                log::info!("cache_timeline {fields}");
            }
            BufferedEvent::QueryPath(event) => {
                let fields = query_path_event_json(&event);
                log::info!("cache_timeline {fields}");
            }
            BufferedEvent::Flush(complete) => {
                log::logger().flush();
                let _ = complete.send(());
            }
        }
    }
}

fn diagnostic_event_json(event: DiagnosticEvent) -> serde_json::Value {
    let mut value = serde_json::json!({
        "diagnostic_event": event.index,
        "stage": event.stage,
        "pid": event.pid,
        "monotonic_ns": event.monotonic_ns,
        "request_id": event.fields.request_id,
        "session_epoch": event.fields.session_epoch,
        "session_token": event.fields.session_token,
    });
    let optional = [
        (HAS_BLOCKS, "blocks", event.fields.blocks),
        (HAS_BYTES, "bytes", event.fields.bytes),
        (HAS_GROUP, "group", event.fields.group),
        (
            HAS_PENDING_BLOCKS,
            "pending_blocks",
            event.fields.pending_blocks,
        ),
        (
            HAS_INFLIGHT_WRITES,
            "inflight_writes",
            event.fields.inflight_writes,
        ),
        (
            HAS_MAX_INFLIGHT_WRITES,
            "max_inflight_writes",
            event.fields.max_inflight_writes,
        ),
        (HAS_ELAPSED_US, "elapsed_us", event.fields.elapsed_us),
    ];
    for (mask, name, field) in optional {
        if event.fields.present & mask != 0 {
            value[name] = field.into();
        }
    }
    if let Some(success) = event.fields.success {
        value["success"] = success.into();
    }
    if let Some(outcome) = event.fields.outcome {
        value["outcome"] = outcome.into();
    }
    value
}

fn query_path_event_json(event: &QueryPathEvent) -> serde_json::Value {
    serde_json::json!({
        "stage": event.stage,
        "pid": event.pid,
        "monotonic_ns": event.monotonic_ns,
        "request_id": event.request_id.as_str(),
        "request_id_truncated": event.request_id.truncated,
        "instance_id": event.instance_id.as_str(),
        "instance_id_truncated": event.instance_id.truncated,
        "group_id": event.group_id,
        "warmup": event.warmup,
        "prepare": event.prepare,
        "elapsed_us": event.elapsed_us,
        "hit_blocks": event.hit_blocks,
    })
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
        timeline.record_json(fields);
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

pub fn record_diagnostic(stage: &'static str, fields: DiagnosticFields) {
    if !diagnostic_enabled() {
        return;
    }
    if let Some(timeline) = BUFFERED_TIMELINE.as_ref() {
        timeline.record_diagnostic(stage, fields);
    } else {
        let event = DiagnosticEvent {
            index: BUFFERED_EVENTS.fetch_add(1, Ordering::Relaxed),
            stage,
            pid: std::process::id(),
            monotonic_ns: monotonic_ns(),
            fields,
        };
        let fields = diagnostic_event_json(event);
        log::info!("cache_timeline {fields}");
    }
}

#[allow(clippy::too_many_arguments, reason = "existing query timeline schema")]
pub fn record_query_path(
    stage: &'static str,
    request_id: &str,
    instance_id: &str,
    group_id: u32,
    warmup: bool,
    prepare: bool,
    elapsed_us: u64,
    hit_blocks: usize,
) {
    if !*ENABLED {
        return;
    }
    if let Some(timeline) = BUFFERED_TIMELINE.as_ref() {
        timeline.record_query_path(
            stage,
            request_id,
            instance_id,
            group_id,
            warmup,
            prepare,
            elapsed_us,
            hit_blocks,
        );
    } else {
        record(stage, || {
            serde_json::json!({
                "request_id": request_id,
                "instance_id": instance_id,
                "group_id": group_id,
                "warmup": warmup,
                "prepare": prepare,
                "elapsed_us": elapsed_us,
                "hit_blocks": hit_blocks,
            })
        });
    }
}

#[cfg(test)]
mod tests {
    use super::{DiagnosticEvent, DiagnosticFields};

    #[test]
    fn monotonic_clock_advances() {
        let before = super::monotonic_ns();
        let after = super::monotonic_ns();
        assert!(before > 0);
        assert!(after >= before);
    }

    #[test]
    fn fixed_diagnostic_event_preserves_stage_correlation_and_queue_fields() {
        let event = DiagnosticEvent {
            index: 7,
            stage: "publish_ssd_dequeue",
            pid: 11,
            monotonic_ns: 13,
            fields: DiagnosticFields::operation(17, 19, 23)
                .blocks(8)
                .pending_blocks(3)
                .inflight_writes(2)
                .max_inflight_writes(4),
        };
        assert_eq!(
            super::diagnostic_event_json(event),
            serde_json::json!({
                "diagnostic_event": 7,
                "stage": "publish_ssd_dequeue",
                "pid": 11,
                "monotonic_ns": 13,
                "request_id": 17,
                "session_epoch": 19,
                "session_token": 23,
                "blocks": 8,
                "pending_blocks": 3,
                "inflight_writes": 2,
                "max_inflight_writes": 4,
            })
        );
    }
}
