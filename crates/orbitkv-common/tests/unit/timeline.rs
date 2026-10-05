use std::sync::Arc;

use super::{
    BufferedTimeline, DiagnosticEvent, DiagnosticFields, FlushSummary, RestoreEvent,
    RestoreTimelineFields,
};

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
        super::diagnostic_event_json(&event),
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

#[test]
fn fixed_restore_event_preserves_existing_timing_schema() {
    let event = RestoreEvent {
        stage: "local_restore_complete",
        pid: 11,
        at_unix_ns: 12,
        monotonic_ns: 13,
        fields: RestoreTimelineFields::operation(17, 19, 23)
            .success(true)
            .drain_timing(29, 31, 37, 41, 43, 47),
    };
    assert_eq!(
        super::restore_event_json(&event),
        serde_json::json!({
            "stage": "local_restore_complete",
            "pid": 11,
            "at_unix_ns": 12,
            "monotonic_ns": 13,
            "restore_key": "manager:17:19:23",
            "success": true,
            "readiness_ns": 29,
            "dispatched_ns": 31,
            "dequeued_ns": 37,
            "claimed_ns": 41,
            "submitted_ns": 43,
            "drained_ns": 47,
        })
    );
}

#[test]
fn zero_limit_does_not_allocate_buffer() {
    assert!(BufferedTimeline::start(0).is_none());
}

#[test]
fn concurrent_writers_claim_each_slot_once_and_flush_once() {
    const THREADS: u64 = 8;
    const EVENTS_PER_THREAD: u64 = 128;
    let timeline = Arc::new(BufferedTimeline::start(THREADS * EVENTS_PER_THREAD).unwrap());
    let mut writers = Vec::new();
    for thread in 0..THREADS {
        let timeline = Arc::clone(&timeline);
        writers.push(std::thread::spawn(move || {
            for event in 0..EVENTS_PER_THREAD {
                timeline.record_diagnostic(
                    "concurrent",
                    DiagnosticFields::operation(thread * EVENTS_PER_THREAD + event, thread, event),
                );
            }
        }));
    }
    for writer in writers {
        writer.join().unwrap();
    }

    let mut events = Vec::new();
    assert_eq!(
        timeline.flush_with(|event| events.push(event)),
        Some(FlushSummary {
            reserved: THREADS * EVENTS_PER_THREAD,
            ready: THREADS * EVENTS_PER_THREAD,
            overflow: false,
        })
    );
    let mut indices = events
        .iter()
        .map(|event| event["diagnostic_event"].as_u64().unwrap())
        .collect::<Vec<_>>();
    indices.sort_unstable();
    assert_eq!(
        indices,
        (0..THREADS * EVENTS_PER_THREAD).collect::<Vec<_>>()
    );

    assert_eq!(timeline.flush_with(|event| events.push(event)), None);
    assert_eq!(events.len() as u64, THREADS * EVENTS_PER_THREAD);
}

#[test]
fn overflow_emits_invalidating_limit_event() {
    let timeline = BufferedTimeline::start(1).unwrap();
    timeline.record_diagnostic("first", DiagnosticFields::default());
    timeline.record_diagnostic("overflow", DiagnosticFields::default());

    let mut events = Vec::new();
    assert_eq!(
        timeline.flush_with(|event| events.push(event)),
        Some(FlushSummary {
            reserved: 1,
            ready: 1,
            overflow: true,
        })
    );
    assert_eq!(events.len(), 2);
    assert_eq!(events[1]["stage"], "diagnostic_timeline_limit");
    assert_eq!(events[1]["reason"], "capacity_exceeded_or_post_seal");
}

#[test]
fn incomplete_slot_emits_invalidating_limit_event() {
    let timeline = BufferedTimeline::start(2).unwrap();
    assert_eq!(timeline.reserve(), Some(0));
    timeline.record_diagnostic("ready", DiagnosticFields::default());

    let mut events = Vec::new();
    assert_eq!(
        timeline.flush_with(|event| events.push(event)),
        Some(FlushSummary {
            reserved: 2,
            ready: 1,
            overflow: false,
        })
    );
    assert_eq!(events.len(), 2);
    assert_eq!(events[1]["stage"], "diagnostic_timeline_limit");
    assert_eq!(events[1]["reason"], "slot_not_ready");
    assert_eq!(events[1]["reserved"], 2);
    assert_eq!(events[1]["ready"], 1);
}
