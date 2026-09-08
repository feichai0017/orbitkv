use std::time::Duration;

use luminal_cuda_lite::runtime::SelectedBucketProfile;
use orbitkv::{
    StateLayoutAlternative,
    kv_manager::{
        RelocationCostEnvelope, RelocationCostIdentity, RelocationCostMeasurement,
        RelocationCostProfile,
    },
};
use sha2::{Digest, Sha256};

use crate::{ExecutorArena, ExecutorPlan};

use super::{
    CompiledDecoder, DecoderArtifactArena, DecoderCompileConfig, DecoderConfig, DecoderError,
    DecoderWeightFeatures,
};

const MINIMUM_PROFILE_SAMPLES: u32 = 3;
const NANOS_PER_SECOND: u128 = 1_000_000_000;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct DecoderProfileIdentity {
    pub matched_execution_fingerprint: [u8; 32],
    pub manager_plan_fingerprint: [u8; 32],
    pub compiler_facts_digest: [u8; 32],
    pub artifact_fingerprint: [u8; 32],
    pub schedule_fingerprint: [u8; 32],
    pub bucket_program_fingerprints: Box<[[u64; 2]]>,
    pub class_layouts: Box<[(u16, StateLayoutAlternative)]>,
}

pub(super) fn decoder_measurement_identity(
    config: &DecoderConfig,
    plan: &ExecutorPlan,
    arenas: &[ExecutorArena],
    weights: DecoderWeightFeatures,
    compile: DecoderCompileConfig,
) -> Result<[u8; 32], DecoderError> {
    #[derive(serde::Serialize)]
    struct MeasurementIdentity<'a> {
        manifest_fingerprint: &'a str,
        manager_plan_fingerprint: [u8; 32],
        page_tokens: u32,
        decoder: &'a DecoderConfig,
        weights: DecoderWeightFeatures,
        arenas: Vec<DecoderArtifactArena>,
        compile: DecoderCompileConfig,
    }

    let manager_plan_fingerprint =
        plan.state_layout_facts
            .manager_plan_fingerprint
            .ok_or(DecoderError::RelocationProfile(
                "manager plan identity missing",
            ))?;
    let identity = MeasurementIdentity {
        manifest_fingerprint: &plan.manifest_fingerprint,
        manager_plan_fingerprint,
        page_tokens: plan.page_tokens,
        decoder: config,
        weights,
        arenas: arenas
            .iter()
            .map(|arena| DecoderArtifactArena {
                class_id: arena.class_id,
                backend_base_index: arena.backend_base_index,
                page_count: arena.page_count,
            })
            .collect(),
        compile,
    };
    Ok(Sha256::digest(serde_json::to_vec(&identity)?).into())
}

/// One real relocation-copy observation measured with CUDA events.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RelocationBandwidthSample {
    pub bytes: u64,
    pub device_time: Duration,
}

/// Aggregate device-to-device relocation throughput used by the cost model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RelocationBandwidthProfile {
    bytes_per_second: u64,
    samples: u32,
}

/// Proposal range for which matched relocation measurements may be reused.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RelocationProfileEnvelopeInput {
    pub class_id: u16,
    pub minimum_fragmentation_milli: u16,
    pub maximum_fragmentation_milli: u16,
    pub minimum_moved_bytes: u64,
    pub maximum_moved_bytes: u64,
}

impl RelocationBandwidthProfile {
    /// Aggregates real CUDA-event samples as total bytes divided by total time.
    ///
    /// # Errors
    ///
    /// Rejects fewer than three samples, zero-byte copies, zero durations, and
    /// arithmetic overflow.
    pub fn from_samples(samples: &[RelocationBandwidthSample]) -> Result<Self, DecoderError> {
        let sample_count = u32::try_from(samples.len())
            .map_err(|_| DecoderError::RelocationProfile("too many relocation samples"))?;
        if sample_count < MINIMUM_PROFILE_SAMPLES
            || samples
                .iter()
                .any(|sample| sample.bytes == 0 || sample.device_time.is_zero())
        {
            return Err(DecoderError::RelocationProfile(
                "at least three non-zero relocation samples are required",
            ));
        }
        let bytes = samples.iter().try_fold(0_u128, |total, sample| {
            total
                .checked_add(u128::from(sample.bytes))
                .ok_or(DecoderError::RelocationProfile(
                    "relocation byte total overflow",
                ))
        })?;
        let nanoseconds =
            samples.iter().try_fold(0_u128, |total, sample| {
                total.checked_add(sample.device_time.as_nanos()).ok_or(
                    DecoderError::RelocationProfile("relocation duration total overflow"),
                )
            })?;
        let bytes_per_second = bytes
            .checked_mul(NANOS_PER_SECOND)
            .and_then(|scaled| scaled.checked_div(nanoseconds))
            .and_then(|throughput| u64::try_from(throughput).ok())
            .filter(|throughput| *throughput > 0)
            .ok_or(DecoderError::RelocationProfile(
                "relocation throughput is not representable",
            ))?;
        Ok(Self {
            bytes_per_second,
            samples: sample_count,
        })
    }
}

/// Builds manager-consumable evidence from two freshly searched executables.
///
/// The source and target must differ only in the selected physical layout for
/// `class_id`. Bucket geometry, model/weight/arena identity, and manager plan
/// must match exactly. Loading a stored schedule yields no fresh selected
/// profile and is therefore rejected.
///
/// # Errors
///
/// Fails closed for stale or weak profiles, mismatched executables or buckets,
/// illegal source/target layouts, and invalid proposal envelopes.
pub fn build_relocation_cost_profile(
    source: &CompiledDecoder,
    target: &CompiledDecoder,
    bucket_index: usize,
    envelope: RelocationProfileEnvelopeInput,
    relocation: RelocationBandwidthProfile,
) -> Result<RelocationCostProfile, DecoderError> {
    let source_profile = source
        .runtime
        .selected_bucket_profiles()
        .iter()
        .find(|profile| profile.bucket_index == bucket_index)
        .ok_or(DecoderError::RelocationProfile(
            "source bucket has no fresh selected profile",
        ))?;
    let target_profile = target
        .runtime
        .selected_bucket_profiles()
        .iter()
        .find(|profile| profile.bucket_index == bucket_index)
        .ok_or(DecoderError::RelocationProfile(
            "target bucket has no fresh selected profile",
        ))?;
    build_profile_from_parts(
        &source.profile_identity,
        source_profile,
        &target.profile_identity,
        target_profile,
        envelope,
        relocation,
    )
}

fn build_profile_from_parts(
    source: &DecoderProfileIdentity,
    source_profile: &SelectedBucketProfile,
    target: &DecoderProfileIdentity,
    target_profile: &SelectedBucketProfile,
    envelope: RelocationProfileEnvelopeInput,
    relocation: RelocationBandwidthProfile,
) -> Result<RelocationCostProfile, DecoderError> {
    if source.matched_execution_fingerprint != target.matched_execution_fingerprint
        || source.manager_plan_fingerprint != target.manager_plan_fingerprint
        || source.compiler_facts_digest == target.compiler_facts_digest
        || source.schedule_fingerprint == [0; 32]
        || target.schedule_fingerprint == [0; 32]
        || source.artifact_fingerprint == target.artifact_fingerprint
        || source.schedule_fingerprint == target.schedule_fingerprint
        || source.bucket_program_fingerprints.len() != target.bucket_program_fingerprints.len()
        || source
            .bucket_program_fingerprints
            .get(source_profile.bucket_index)
            .zip(
                target
                    .bucket_program_fingerprints
                    .get(target_profile.bucket_index),
            )
            .is_none_or(|(source_program, target_program)| source_program == target_program)
        || source.class_layouts.len() != target.class_layouts.len()
    {
        return Err(DecoderError::RelocationProfile(
            "source and target executable identities do not match",
        ));
    }
    let source_layout = class_layout(source, envelope.class_id)?;
    let target_layout = class_layout(target, envelope.class_id)?;
    if source_layout != StateLayoutAlternative::Compiled
        || target_layout != StateLayoutAlternative::PackedTokenSlots
        || source
            .class_layouts
            .iter()
            .zip(target.class_layouts.iter())
            .any(
                |(&(source_id, source_layout), &(target_id, target_layout))| {
                    source_id != target_id
                        || (source_id != envelope.class_id && source_layout != target_layout)
                },
            )
    {
        return Err(DecoderError::RelocationProfile(
            "executables do not represent one compiled-to-packed layout change",
        ));
    }
    if envelope.minimum_fragmentation_milli > envelope.maximum_fragmentation_milli
        || envelope.maximum_fragmentation_milli > 1000
        || envelope.minimum_moved_bytes == 0
        || envelope.minimum_moved_bytes > envelope.maximum_moved_bytes
    {
        return Err(DecoderError::RelocationProfile(
            "relocation proposal envelope is invalid",
        ));
    }
    if source_profile.bucket_index != target_profile.bucket_index
        || source_profile.bucket_dimensions != target_profile.bucket_dimensions
        || source_profile.representative_dimensions != target_profile.representative_dimensions
        || source_profile.trials < usize::try_from(MINIMUM_PROFILE_SAMPLES).unwrap()
        || target_profile.trials < usize::try_from(MINIMUM_PROFILE_SAMPLES).unwrap()
        || source_profile.device_time.is_zero()
        || target_profile.device_time.is_zero()
    {
        return Err(DecoderError::RelocationProfile(
            "selected bucket measurements are weak or unmatched",
        ));
    }
    let source_step_ns = duration_ns(source_profile.device_time)?;
    let target_step_ns = duration_ns(target_profile.device_time)?;
    let source_samples = u32::try_from(source_profile.trials)
        .map_err(|_| DecoderError::RelocationProfile("source sample count overflow"))?;
    let target_samples = u32::try_from(target_profile.trials)
        .map_err(|_| DecoderError::RelocationProfile("target sample count overflow"))?;
    let bucket_fingerprint = bucket_fingerprint(source_profile);

    Ok(RelocationCostProfile {
        identity: RelocationCostIdentity {
            plan_fingerprint: source.manager_plan_fingerprint,
            source_compiler_facts_digest: source.compiler_facts_digest,
            target_compiler_facts_digest: target.compiler_facts_digest,
            bucket_fingerprint,
            source_artifact_fingerprint: source.artifact_fingerprint,
            target_artifact_fingerprint: target.artifact_fingerprint,
            source_schedule_fingerprint: source.schedule_fingerprint,
            target_schedule_fingerprint: target.schedule_fingerprint,
            class_id: envelope.class_id,
            source_layout,
            target_layout,
        },
        envelope: RelocationCostEnvelope {
            minimum_fragmentation_milli: envelope.minimum_fragmentation_milli,
            maximum_fragmentation_milli: envelope.maximum_fragmentation_milli,
            minimum_moved_bytes: envelope.minimum_moved_bytes,
            maximum_moved_bytes: envelope.maximum_moved_bytes,
        },
        measurement: RelocationCostMeasurement {
            source_step_ns,
            target_step_ns,
            relocation_bytes_per_second: relocation.bytes_per_second,
            source_samples,
            target_samples,
            relocation_samples: relocation.samples,
        },
    })
}

fn class_layout(
    identity: &DecoderProfileIdentity,
    class_id: u16,
) -> Result<StateLayoutAlternative, DecoderError> {
    identity
        .class_layouts
        .iter()
        .find_map(|&(candidate, layout)| (candidate == class_id).then_some(layout))
        .ok_or(DecoderError::RelocationProfile(
            "relocation class is absent from compiler facts",
        ))
}

fn duration_ns(duration: Duration) -> Result<u64, DecoderError> {
    u64::try_from(duration.as_nanos())
        .ok()
        .filter(|duration| *duration > 0)
        .ok_or(DecoderError::RelocationProfile(
            "selected bucket duration is not representable",
        ))
}

fn bucket_fingerprint(profile: &SelectedBucketProfile) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update((profile.bucket_index as u64).to_le_bytes());
    hash.update((profile.bucket_dimensions.len() as u64).to_le_bytes());
    for dimension in &profile.bucket_dimensions {
        hash_bytes(&mut hash, dimension.name.as_bytes());
        hash.update((dimension.bucket_index as u64).to_le_bytes());
        hash.update((dimension.minimum as u64).to_le_bytes());
        hash.update((dimension.maximum as u64).to_le_bytes());
        hash.update((dimension.representative as u64).to_le_bytes());
    }
    hash.update((profile.representative_dimensions.len() as u64).to_le_bytes());
    for (name, value) in &profile.representative_dimensions {
        hash_bytes(&mut hash, name.as_bytes());
        hash.update((*value as u64).to_le_bytes());
    }
    hash.finalize().into()
}

fn hash_bytes(hash: &mut Sha256, bytes: &[u8]) {
    hash.update((bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
}

#[cfg(test)]
mod tests {
    use luminal_cuda_lite::runtime::SelectedBucketDimension;

    use super::*;

    fn identity(layout: StateLayoutAlternative, facts: u8) -> DecoderProfileIdentity {
        DecoderProfileIdentity {
            matched_execution_fingerprint: [1; 32],
            manager_plan_fingerprint: [2; 32],
            compiler_facts_digest: [facts; 32],
            artifact_fingerprint: [facts + 5; 32],
            schedule_fingerprint: [facts + 10; 32],
            bucket_program_fingerprints: vec![[u64::from(facts), 1]].into_boxed_slice(),
            class_layouts: vec![(0, layout), (1, StateLayoutAlternative::Compiled)]
                .into_boxed_slice(),
        }
    }

    fn bucket(time_ns: u64) -> SelectedBucketProfile {
        SelectedBucketProfile {
            bucket_index: 0,
            bucket_dimensions: vec![SelectedBucketDimension {
                name: "s".into(),
                bucket_index: 0,
                minimum: 1,
                maximum: 1,
                representative: 1,
            }],
            representative_dimensions: vec![("s".into(), 1)],
            device_time: Duration::from_nanos(time_ns),
            trials: 3,
        }
    }

    fn envelope() -> RelocationProfileEnvelopeInput {
        RelocationProfileEnvelopeInput {
            class_id: 0,
            minimum_fragmentation_milli: 250,
            maximum_fragmentation_milli: 750,
            minimum_moved_bytes: 1024,
            maximum_moved_bytes: 4096,
        }
    }

    fn bandwidth() -> RelocationBandwidthProfile {
        RelocationBandwidthProfile::from_samples(
            &[RelocationBandwidthSample {
                bytes: 1024,
                device_time: Duration::from_nanos(100),
            }; 3],
        )
        .unwrap()
    }

    #[test]
    fn builds_identity_bound_matched_cost_profile() {
        let profile = build_profile_from_parts(
            &identity(StateLayoutAlternative::Compiled, 3),
            &bucket(100),
            &identity(StateLayoutAlternative::PackedTokenSlots, 4),
            &bucket(70),
            envelope(),
            bandwidth(),
        )
        .unwrap();

        assert_eq!(profile.identity.plan_fingerprint, [2; 32]);
        assert_eq!(profile.measurement.source_step_ns, 100);
        assert_eq!(profile.measurement.target_step_ns, 70);
        assert_eq!(profile.measurement.relocation_samples, 3);
        assert_ne!(profile.identity.bucket_fingerprint, [0; 32]);
    }

    #[test]
    fn rejects_geometry_layout_and_sample_mismatches() {
        let source = identity(StateLayoutAlternative::Compiled, 3);
        let target = identity(StateLayoutAlternative::PackedTokenSlots, 4);
        let mut wrong_bucket = bucket(70);
        wrong_bucket.bucket_dimensions[0].maximum = 2;
        assert!(
            build_profile_from_parts(
                &source,
                &bucket(100),
                &target,
                &wrong_bucket,
                envelope(),
                bandwidth(),
            )
            .is_err()
        );

        let wrong_layout = identity(StateLayoutAlternative::Compiled, 4);
        assert!(
            build_profile_from_parts(
                &source,
                &bucket(100),
                &wrong_layout,
                &bucket(70),
                envelope(),
                bandwidth(),
            )
            .is_err()
        );

        let mut no_execution_change = target.clone();
        no_execution_change.bucket_program_fingerprints =
            source.bucket_program_fingerprints.clone();
        assert!(
            build_profile_from_parts(
                &source,
                &bucket(100),
                &no_execution_change,
                &bucket(70),
                envelope(),
                bandwidth(),
            )
            .is_err()
        );

        let mut weak = bucket(70);
        weak.trials = 2;
        assert!(
            build_profile_from_parts(
                &source,
                &bucket(100),
                &target,
                &weak,
                envelope(),
                bandwidth(),
            )
            .is_err()
        );
    }

    #[test]
    fn bandwidth_requires_multiple_real_nonzero_samples() {
        assert!(RelocationBandwidthProfile::from_samples(&[]).is_err());
        assert!(
            RelocationBandwidthProfile::from_samples(
                &[RelocationBandwidthSample {
                    bytes: 1,
                    device_time: Duration::ZERO,
                }; 3]
            )
            .is_err()
        );
    }
}
