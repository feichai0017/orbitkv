use serde::{Deserialize, Serialize};

use crate::StateLayoutAlternative;

use super::{KvManagerError, RelocationPlan};

const MINIMUM_PROFILE_SAMPLES: u32 = 3;
const NANOS_PER_SECOND: u128 = 1_000_000_000;

/// Static identity and byte geometry known by the canonical manager.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct RelocationPlanningContext {
    pub plan_fingerprint: [u8; 32],
    pub page_tokens: u32,
    pub page_payload_bytes: u64,
}

/// Identity of a matched source-layout/target-layout measurement.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RelocationCostIdentity {
    pub plan_fingerprint: [u8; 32],
    pub compiler_facts_digest: [u8; 32],
    pub bucket_fingerprint: [u8; 32],
    pub artifact_fingerprint: [u8; 32],
    pub schedule_fingerprint: [u8; 32],
    pub bucket_program_fingerprint: [u8; 32],
    pub source_execution_fingerprint: [u8; 32],
    pub target_execution_fingerprint: [u8; 32],
    pub class_id: u16,
    pub source_layout: StateLayoutAlternative,
    pub target_layout: StateLayoutAlternative,
}

/// Proposal range covered by the measured copy bandwidth and execution costs.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RelocationCostEnvelope {
    pub minimum_fragmentation_milli: u16,
    pub maximum_fragmentation_milli: u16,
    pub minimum_moved_bytes: u64,
    pub maximum_moved_bytes: u64,
}

/// Measured source/target step costs and relocation-copy throughput.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RelocationCostMeasurement {
    pub source_step_ns: u64,
    pub target_step_ns: u64,
    pub relocation_bytes_per_second: u64,
    pub source_samples: u32,
    pub target_samples: u32,
    pub relocation_samples: u32,
}

/// A backend measurement usable by the manager without importing backend types.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RelocationCostProfile {
    pub identity: RelocationCostIdentity,
    pub envelope: RelocationCostEnvelope,
    pub measurement: RelocationCostMeasurement,
}

/// Profitability evidence required before a legal relocation is admitted.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RelocationAdmission {
    /// Default: no backend measurement means no relocation.
    #[default]
    Disabled,
    /// Explicit non-production mode retained for deterministic planner and
    /// correctness qualification.
    StaticFragmentation { minimum_fragmentation_milli: u16 },
    /// Require exact measured evidence and a positive amortized benefit.
    Measured {
        profile: Box<RelocationCostProfile>,
        expected_reuse_steps: u64,
        minimum_net_savings_milli: u16,
    },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RelocationPolicy {
    pub admission: RelocationAdmission,
    pub maximum_source_pages: u32,
    pub evacuation_headroom_pages: u32,
    pub full_evacuation: bool,
}

impl Default for RelocationPolicy {
    fn default() -> Self {
        Self {
            admission: RelocationAdmission::Disabled,
            maximum_source_pages: 32,
            evacuation_headroom_pages: 8,
            full_evacuation: false,
        }
    }
}

impl RelocationPolicy {
    #[must_use]
    pub const fn static_fragmentation(
        minimum_fragmentation_milli: u16,
        maximum_source_pages: u32,
        evacuation_headroom_pages: u32,
        full_evacuation: bool,
    ) -> Self {
        Self {
            admission: RelocationAdmission::StaticFragmentation {
                minimum_fragmentation_milli,
            },
            maximum_source_pages,
            evacuation_headroom_pages,
            full_evacuation,
        }
    }

    #[must_use]
    pub fn measured(
        profile: RelocationCostProfile,
        expected_reuse_steps: u64,
        minimum_net_savings_milli: u16,
        maximum_source_pages: u32,
        evacuation_headroom_pages: u32,
        full_evacuation: bool,
    ) -> Self {
        Self {
            admission: RelocationAdmission::Measured {
                profile: Box::new(profile),
                expected_reuse_steps,
                minimum_net_savings_milli,
            },
            maximum_source_pages,
            evacuation_headroom_pages,
            full_evacuation,
        }
    }
}

pub(super) fn minimum_fragmentation(
    policy: &RelocationPolicy,
) -> Result<Option<u16>, KvManagerError> {
    if policy.maximum_source_pages == 0 || policy.evacuation_headroom_pages == 0 {
        return Err(KvManagerError::InvalidRelocationPolicy);
    }
    match &policy.admission {
        RelocationAdmission::Disabled => Ok(None),
        RelocationAdmission::StaticFragmentation {
            minimum_fragmentation_milli,
        } if *minimum_fragmentation_milli <= 1000 => Ok(Some(*minimum_fragmentation_milli)),
        RelocationAdmission::StaticFragmentation { .. } => {
            Err(KvManagerError::InvalidRelocationPolicy)
        }
        RelocationAdmission::Measured {
            profile,
            expected_reuse_steps,
            minimum_net_savings_milli,
        } => {
            validate_profile_shape(profile)?;
            if *expected_reuse_steps == 0 || *minimum_net_savings_milli > 1000 {
                return Err(KvManagerError::InvalidRelocationPolicy);
            }
            Ok(Some(profile.envelope.minimum_fragmentation_milli))
        }
    }
}

pub(super) fn admits_relocation(
    context: RelocationPlanningContext,
    policy: &RelocationPolicy,
    plan: &RelocationPlan,
) -> Result<bool, KvManagerError> {
    match &policy.admission {
        RelocationAdmission::Disabled => Ok(false),
        RelocationAdmission::StaticFragmentation { .. } => Ok(true),
        RelocationAdmission::Measured {
            profile,
            expected_reuse_steps,
            minimum_net_savings_milli,
        } => {
            validate_profile_shape(profile)?;
            validate_profile_identity(context, &profile.identity, plan)?;
            let moved_bytes = proposal_moved_bytes(context, plan)?;
            if plan.fragmentation_milli > profile.envelope.maximum_fragmentation_milli
                || moved_bytes < profile.envelope.minimum_moved_bytes
                || moved_bytes > profile.envelope.maximum_moved_bytes
            {
                return Ok(false);
            }
            let measurement = profile.measurement;
            let per_step_saving = measurement
                .source_step_ns
                .saturating_sub(measurement.target_step_ns);
            let gross_saving = u128::from(per_step_saving)
                .checked_mul(u128::from(*expected_reuse_steps))
                .ok_or(KvManagerError::ArithmeticOverflow("relocation benefit"))?;
            let relocation_ns = u128::from(moved_bytes)
                .checked_mul(NANOS_PER_SECOND)
                .ok_or(KvManagerError::ArithmeticOverflow("relocation cost"))?
                .div_ceil(u128::from(measurement.relocation_bytes_per_second));
            let required = relocation_ns
                .checked_mul(u128::from(1000_u16 + *minimum_net_savings_milli))
                .ok_or(KvManagerError::ArithmeticOverflow("relocation cost margin"))?
                .div_ceil(1000);
            Ok(gross_saving > required)
        }
    }
}

fn validate_profile_identity(
    context: RelocationPlanningContext,
    identity: &RelocationCostIdentity,
    plan: &RelocationPlan,
) -> Result<(), KvManagerError> {
    if identity.plan_fingerprint != context.plan_fingerprint
        || identity.class_id != plan.class_id
        || identity.source_layout != StateLayoutAlternative::TokenSelectionMask
        || identity.target_layout != StateLayoutAlternative::PackedTokenSlots
    {
        return Err(KvManagerError::RelocationCostProfileMismatch);
    }
    Ok(())
}

fn proposal_moved_bytes(
    context: RelocationPlanningContext,
    plan: &RelocationPlan,
) -> Result<u64, KvManagerError> {
    let bytes_per_token = context
        .page_payload_bytes
        .checked_div(u64::from(context.page_tokens))
        .filter(|bytes| {
            context.page_tokens > 0
                && bytes
                    .checked_mul(u64::from(context.page_tokens))
                    .is_some_and(|total| total == context.page_payload_bytes)
        })
        .ok_or(KvManagerError::RelocationCostProfileMismatch)?;
    bytes_per_token
        .checked_mul(
            u64::try_from(plan.moves.len())
                .map_err(|_| KvManagerError::ArithmeticOverflow("relocation token count"))?,
        )
        .ok_or(KvManagerError::ArithmeticOverflow("relocation byte cost"))
}

fn validate_profile_shape(profile: &RelocationCostProfile) -> Result<(), KvManagerError> {
    let identity = &profile.identity;
    let envelope = &profile.envelope;
    let measurement = &profile.measurement;
    if identity.plan_fingerprint == [0; 32]
        || identity.compiler_facts_digest == [0; 32]
        || identity.bucket_fingerprint == [0; 32]
        || identity.artifact_fingerprint == [0; 32]
        || identity.schedule_fingerprint == [0; 32]
        || identity.bucket_program_fingerprint == [0; 32]
        || identity.source_execution_fingerprint == [0; 32]
        || identity.target_execution_fingerprint == [0; 32]
        || identity.source_execution_fingerprint == identity.target_execution_fingerprint
        || envelope.minimum_fragmentation_milli > envelope.maximum_fragmentation_milli
        || envelope.maximum_fragmentation_milli > 1000
        || envelope.minimum_moved_bytes == 0
        || envelope.minimum_moved_bytes > envelope.maximum_moved_bytes
        || measurement.source_step_ns == 0
        || measurement.target_step_ns == 0
        || measurement.relocation_bytes_per_second == 0
        || measurement.source_samples < MINIMUM_PROFILE_SAMPLES
        || measurement.target_samples < MINIMUM_PROFILE_SAMPLES
        || measurement.relocation_samples < MINIMUM_PROFILE_SAMPLES
    {
        return Err(KvManagerError::InvalidRelocationCostProfile);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_manager::{PageLease, TokenLocation, TokenMove, ViewVersion};

    fn page(page_id: u32) -> PageLease {
        PageLease {
            engine_epoch: 1,
            pool_epoch: 1,
            generation: 1,
            page_id,
            pool_id: 1,
        }
    }

    fn candidate() -> RelocationPlan {
        let location = |page_id, backend_index, offset| TokenLocation {
            page: page(page_id),
            backend_index,
            offset,
            reserved: 0,
        };
        RelocationPlan {
            class_id: 0,
            base_version: ViewVersion(1),
            target_version: ViewVersion(2),
            fragmentation_milli: 500,
            source_pages: vec![page(1), page(2)].into_boxed_slice(),
            destination_pages: vec![page(3)].into_boxed_slice(),
            moves: vec![
                TokenMove {
                    token_id: 0,
                    source: location(1, 0, 0),
                    destination: location(3, 2, 0),
                },
                TokenMove {
                    token_id: 4,
                    source: location(2, 1, 0),
                    destination: location(3, 2, 1),
                },
            ]
            .into_boxed_slice(),
            projected_reclaimed_pages: 1,
        }
    }

    fn context() -> RelocationPlanningContext {
        RelocationPlanningContext {
            plan_fingerprint: [1; 32],
            page_tokens: 4,
            page_payload_bytes: 512,
        }
    }

    fn profile() -> RelocationCostProfile {
        RelocationCostProfile {
            identity: RelocationCostIdentity {
                plan_fingerprint: [1; 32],
                compiler_facts_digest: [2; 32],
                bucket_fingerprint: [4; 32],
                artifact_fingerprint: [5; 32],
                schedule_fingerprint: [6; 32],
                bucket_program_fingerprint: [7; 32],
                source_execution_fingerprint: [9; 32],
                target_execution_fingerprint: [10; 32],
                class_id: 0,
                source_layout: StateLayoutAlternative::TokenSelectionMask,
                target_layout: StateLayoutAlternative::PackedTokenSlots,
            },
            envelope: RelocationCostEnvelope {
                minimum_fragmentation_milli: 250,
                maximum_fragmentation_milli: 750,
                minimum_moved_bytes: 128,
                maximum_moved_bytes: 512,
            },
            measurement: RelocationCostMeasurement {
                source_step_ns: 100,
                target_step_ns: 70,
                relocation_bytes_per_second: 2_560_000_000,
                source_samples: 3,
                target_samples: 3,
                relocation_samples: 3,
            },
        }
    }

    fn measured(profile: &RelocationCostProfile, expected_reuse_steps: u64) -> RelocationPolicy {
        RelocationPolicy::measured(*profile, expected_reuse_steps, 100, 8, 2, true)
    }

    #[test]
    fn default_policy_disables_relocation_without_measurement() {
        assert_eq!(
            minimum_fragmentation(&RelocationPolicy::default()),
            Ok(None)
        );
        assert_eq!(
            admits_relocation(context(), &RelocationPolicy::default(), &candidate()),
            Ok(false)
        );
    }

    #[test]
    fn measured_policy_requires_positive_amortized_net_savings() {
        assert_eq!(
            admits_relocation(context(), &measured(&profile(), 5), &candidate()),
            Ok(true)
        );
        assert_eq!(
            admits_relocation(context(), &measured(&profile(), 3), &candidate()),
            Ok(false)
        );

        let mut slower_target = profile();
        slower_target.measurement.target_step_ns = 110;
        assert_eq!(
            admits_relocation(context(), &measured(&slower_target, 100), &candidate()),
            Ok(false)
        );
    }

    #[test]
    fn measured_policy_rejects_stale_or_weak_evidence() {
        let mut wrong_plan = profile();
        wrong_plan.identity.plan_fingerprint[0] ^= 1;
        assert_eq!(
            admits_relocation(context(), &measured(&wrong_plan, 5), &candidate()),
            Err(KvManagerError::RelocationCostProfileMismatch)
        );

        let mut outside_envelope = profile();
        outside_envelope.envelope.maximum_moved_bytes = 200;
        assert_eq!(
            admits_relocation(context(), &measured(&outside_envelope, 5), &candidate()),
            Ok(false)
        );

        let mut weak = profile();
        weak.measurement.target_samples = MINIMUM_PROFILE_SAMPLES - 1;
        assert_eq!(
            minimum_fragmentation(&measured(&weak, 5)),
            Err(KvManagerError::InvalidRelocationCostProfile)
        );
    }
}
