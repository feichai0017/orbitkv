use std::time::Duration;

use orbitkv_state::ReplicaRepresentation;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CompletionIntent {
    HostReady,
    EngineRestore,
    SourceRelease,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CompletionPath {
    PrefillToDecodeHandoff,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompletionAdmission {
    Admitted,
    Rejected,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompletionOutcome {
    Completed,
    Failed,
    Cancelled,
    TimedOut,
}

/// One measured physical completion interval reported by its execution owner.
///
/// Request identifiers and state keys are intentionally absent. The TENT
/// notification generation proves that the report came from a fenced transfer,
/// but it is freshness evidence rather than a cost-model dimension.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompletionObservation {
    pub instance_id: String,
    pub destination_device_id: i32,
    pub source_endpoint: String,
    pub notification_generation: u64,
    pub intent: CompletionIntent,
    pub path: CompletionPath,
    pub representation: ReplicaRepresentation,
    pub logical_bytes: u64,
    pub wire_bytes: u64,
    pub fragment_count: u32,
    pub elapsed: Duration,
    pub admission: CompletionAdmission,
    pub outcome: CompletionOutcome,
}
