use super::*;
use std::time::Duration;

fn observation() -> CompletionObservation {
    CompletionObservation {
        instance_id: "decode-instance".into(),
        destination_device_id: 2,
        source_endpoint: "tent://prefill-3".into(),
        notification_generation: 7,
        intent: CompletionIntent::EngineRestore,
        route: CompletionRoute::PrefillToDecodeHandoff,
        representation: ReplicaRepresentation::Raw,
        logical_bytes: 8192,
        wire_bytes: 8192,
        fragment_count: 4,
        elapsed: Duration::from_micros(250),
        admission: CompletionAdmission::Admitted,
        outcome: CompletionOutcome::Completed,
    }
}

#[test]
fn completion_validation_requires_fresh_consistent_evidence() {
    assert!(validate_observation(&observation()).is_ok());

    let mut invalid = observation();
    invalid.notification_generation = 0;
    assert!(validate_observation(&invalid).is_err());

    invalid = observation();
    invalid.intent = CompletionIntent::HostReady;
    assert!(validate_observation(&invalid).is_err());

    invalid = observation();
    invalid.admission = CompletionAdmission::Rejected;
    assert!(validate_observation(&invalid).is_err());

    invalid.outcome = CompletionOutcome::TimedOut;
    invalid.wire_bytes = 0;
    assert!(validate_observation(&invalid).is_ok());
}
