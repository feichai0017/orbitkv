use orbitkv_state::ReplicaRepresentation;

use super::{EngineError, OrbitKVEngine};
use crate::completion::{
    CompletionAdmission, CompletionIntent, CompletionObservation, CompletionOutcome,
    CompletionRoute,
};
use crate::cost::{
    self, CostEstimateKey, CostObservationKind, ExecutionResource, Outcome, resource_id,
};

impl OrbitKVEngine {
    /// Accept measured completion evidence from a registered local engine.
    ///
    /// This records evidence only; route selection remains disabled until the
    /// competing direct-restore path reports the same completion target.
    pub fn observe_completion(
        &self,
        observation: CompletionObservation,
    ) -> Result<(), EngineError> {
        validate_observation(&observation)?;
        let instance = self.get_instance(&observation.instance_id)?;
        if instance
            .get_gpu(observation.destination_device_id)
            .is_none()
        {
            return Err(EngineError::WorkerMissing(
                observation.instance_id,
                observation.destination_device_id,
            ));
        }

        let source_endpoint_hash = resource_id(&observation.source_endpoint);
        let key = CostEstimateKey::new(
            CostObservationKind::PrefillToDecodeHandoff,
            ExecutionResource::PrefillToDecodeHandoff {
                source_endpoint_hash,
                destination_device: observation.destination_device_id as u64,
            },
            observation.representation,
            observation.logical_bytes,
            observation.fragment_count as usize,
        )
        .with_wire_bytes(observation.wire_bytes);
        let outcome = match observation.outcome {
            CompletionOutcome::Completed => Outcome::Completed,
            CompletionOutcome::Failed => Outcome::Failed,
            CompletionOutcome::Cancelled => Outcome::Cancelled,
            CompletionOutcome::TimedOut => Outcome::TimedOut,
        };
        cost::record_completion_observation(
            key,
            observation.logical_bytes,
            observation.wire_bytes,
            observation.elapsed,
            observation.admission == CompletionAdmission::Admitted,
            outcome,
        );
        Ok(())
    }
}

fn validate_observation(observation: &CompletionObservation) -> Result<(), EngineError> {
    let invalid = |message: &str| EngineError::InvalidArgument(message.into());
    if observation.instance_id.is_empty() {
        return Err(invalid("completion instance_id must not be empty"));
    }
    if observation.destination_device_id < 0 {
        return Err(invalid(
            "completion destination_device_id must be non-negative",
        ));
    }
    if observation.source_endpoint.is_empty() {
        return Err(invalid("completion source_endpoint must not be empty"));
    }
    if observation.notification_generation == 0 {
        return Err(invalid(
            "completion notification_generation must be non-zero",
        ));
    }
    if observation.logical_bytes == 0 {
        return Err(invalid("completion logical_bytes must be non-zero"));
    }
    if observation.fragment_count == 0 {
        return Err(invalid("completion fragment_count must be non-zero"));
    }
    if observation.elapsed.is_zero() {
        return Err(invalid("completion elapsed duration must be non-zero"));
    }
    if observation.representation == ReplicaRepresentation::Unknown {
        return Err(invalid("completion representation must be known"));
    }
    if observation.route == CompletionRoute::PrefillToDecodeHandoff
        && observation.intent != CompletionIntent::EngineRestore
    {
        return Err(invalid(
            "prefill-to-decode handoff must complete an engine restore",
        ));
    }
    match (
        observation.admission,
        observation.outcome,
        observation.wire_bytes,
    ) {
        (CompletionAdmission::Admitted, CompletionOutcome::Completed, 0)
        | (CompletionAdmission::Rejected, CompletionOutcome::Completed, _)
        | (CompletionAdmission::Rejected, _, 1..) => Err(invalid(
            "completion admission, outcome and wire_bytes are inconsistent",
        )),
        _ => Ok(()),
    }
}

#[cfg(test)]
#[path = "../../tests/unit/engine/completion.rs"]
mod tests;
