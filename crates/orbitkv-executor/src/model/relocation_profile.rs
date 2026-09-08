use std::time::Duration;

use luminal_cuda_lite::runtime::{RuntimeExecutionProfile, SelectedBucketDimension};
use orbitkv::{
    StateLayoutAlternative,
    kv_manager::{
        RelocationCostEnvelope, RelocationCostIdentity, RelocationCostMeasurement,
        RelocationCostProfile,
    },
};
use sha2::{Digest, Sha256};

use super::{CompiledDecoder, DecoderError, DecoderStep};

const MINIMUM_PROFILE_SAMPLES: u32 = 3;
const NANOS_PER_SECOND: u128 = 1_000_000_000;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct DecoderProfileIdentity {
    pub manager_plan_fingerprint: [u8; 32],
    pub compiler_facts_digest: [u8; 32],
    pub artifact_fingerprint: [u8; 32],
    pub schedule_fingerprint: [u8; 32],
    pub bucket_program_fingerprints: Box<[[u64; 2]]>,
}

impl DecoderProfileIdentity {
    pub(super) fn fingerprint(&self) -> [u8; 32] {
        let mut hash = Sha256::new();
        hash.update(self.manager_plan_fingerprint);
        hash.update(self.compiler_facts_digest);
        hash.update(self.artifact_fingerprint);
        hash.update(self.schedule_fingerprint);
        hash.update((self.bucket_program_fingerprints.len() as u64).to_le_bytes());
        for program in &self.bucket_program_fingerprints {
            hash.update(program[0].to_le_bytes());
            hash.update(program[1].to_le_bytes());
        }
        hash.finalize().into()
    }
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

/// Fresh profile of one deployed decoder at an exact request geometry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionGeometryProfile {
    executable_fingerprint: [u8; 32],
    bucket_index: usize,
    bucket_dimensions: Vec<SelectedBucketDimension>,
    execution_dimensions: Vec<(String, usize)>,
    state_geometry: Box<[ExecutionStateGeometry]>,
    fingerprint: [u8; 32],
    device_time: Duration,
    trials: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ExecutionStateGeometry {
    class_id: u16,
    page_tokens: u32,
    page_count: usize,
    token_count: u64,
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

pub(super) fn execution_geometry_profile(
    step: DecoderStep<'_>,
    profile: RuntimeExecutionProfile,
    executable_fingerprint: [u8; 32],
) -> Result<ExecutionGeometryProfile, DecoderError> {
    let state_geometry = step
        .classes
        .iter()
        .map(|class| {
            let mut token_count = 0_u64;
            for row in class.attention.page_indptr.windows(2) {
                let page_count = i64::from(row[1]) - i64::from(row[0]);
                if page_count <= 0 {
                    return Err(DecoderError::RelocationProfile(
                        "execution profile contains an empty state row",
                    ));
                }
            }
            for (row, &last_page_len) in class
                .attention
                .page_indptr
                .windows(2)
                .zip(class.attention.last_page_len.iter())
            {
                let page_count = u64::try_from(i64::from(row[1]) - i64::from(row[0]))
                    .map_err(|_| DecoderError::RelocationProfile("invalid state page count"))?;
                token_count = token_count
                    .checked_add(
                        page_count
                            .saturating_sub(1)
                            .checked_mul(u64::from(class.attention.page_tokens))
                            .and_then(|tokens| {
                                tokens.checked_add(u64::try_from(last_page_len).ok()?)
                            })
                            .ok_or(DecoderError::RelocationProfile(
                                "state token count overflow",
                            ))?,
                    )
                    .ok_or(DecoderError::RelocationProfile(
                        "state token count overflow",
                    ))?;
            }
            Ok(ExecutionStateGeometry {
                class_id: class.class_id,
                page_tokens: class.attention.page_tokens,
                page_count: class.attention.page_indices.len(),
                token_count,
            })
        })
        .collect::<Result<Vec<_>, DecoderError>>()?
        .into_boxed_slice();
    let fingerprint = execution_fingerprint(
        profile.bucket_index,
        &profile.execution_dimensions,
        &state_geometry,
    );
    Ok(ExecutionGeometryProfile {
        executable_fingerprint,
        bucket_index: profile.bucket_index,
        bucket_dimensions: profile.bucket_dimensions,
        execution_dimensions: profile.execution_dimensions,
        state_geometry,
        fingerprint,
        device_time: profile.device_time,
        trials: profile.trials,
    })
}

/// Builds manager-consumable evidence from two exact runtime geometries.
///
/// Both profiles must come from the same executable and bucket. The target
/// must preserve every unrelated state class while representing the selected
/// class with no more tokens or pages than the source.
///
/// # Errors
///
/// Fails closed for stale or weak profiles, mismatched executables or buckets,
/// illegal source/target layouts, and invalid proposal envelopes.
pub fn build_relocation_cost_profile(
    decoder: &CompiledDecoder,
    source: &ExecutionGeometryProfile,
    target: &ExecutionGeometryProfile,
    envelope: RelocationProfileEnvelopeInput,
    relocation: RelocationBandwidthProfile,
) -> Result<RelocationCostProfile, DecoderError> {
    build_profile_from_parts(
        &decoder.profile_identity,
        source,
        target,
        envelope,
        relocation,
    )
}

fn build_profile_from_parts(
    identity: &DecoderProfileIdentity,
    source: &ExecutionGeometryProfile,
    target: &ExecutionGeometryProfile,
    envelope: RelocationProfileEnvelopeInput,
    relocation: RelocationBandwidthProfile,
) -> Result<RelocationCostProfile, DecoderError> {
    if source.executable_fingerprint != identity.fingerprint()
        || target.executable_fingerprint != identity.fingerprint()
        || source.bucket_index != target.bucket_index
        || non_layout_dimensions(&source.bucket_dimensions, &source.state_geometry)
            != non_layout_dimensions(&target.bucket_dimensions, &target.state_geometry)
        || non_layout_execution_dimensions(&source.execution_dimensions, &source.state_geometry)
            != non_layout_execution_dimensions(&target.execution_dimensions, &target.state_geometry)
        || source.state_geometry.len() != target.state_geometry.len()
        || source.fingerprint == target.fingerprint
        || source.trials < usize::try_from(MINIMUM_PROFILE_SAMPLES).unwrap()
        || target.trials < usize::try_from(MINIMUM_PROFILE_SAMPLES).unwrap()
        || source.device_time.is_zero()
        || target.device_time.is_zero()
    {
        return Err(DecoderError::RelocationProfile(
            "source and target execution measurements are weak or unmatched",
        ));
    }
    let source_class = state_geometry(source, envelope.class_id)?;
    let target_class = state_geometry(target, envelope.class_id)?;
    if source_class.page_tokens != 1
        || target_class.page_tokens <= 1
        || target_class.page_count > source_class.page_count
        || target_class.token_count != source_class.token_count
        || source
            .state_geometry
            .iter()
            .zip(target.state_geometry.iter())
            .any(|(source, target)| {
                source.class_id != target.class_id
                    || (source.class_id != envelope.class_id && source != target)
            })
    {
        return Err(DecoderError::RelocationProfile(
            "measurements do not represent one token-selection-to-packed transition",
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
    let source_step_ns = duration_ns(source.device_time)?;
    let target_step_ns = duration_ns(target.device_time)?;
    let source_samples = u32::try_from(source.trials)
        .map_err(|_| DecoderError::RelocationProfile("source sample count overflow"))?;
    let target_samples = u32::try_from(target.trials)
        .map_err(|_| DecoderError::RelocationProfile("target sample count overflow"))?;
    let bucket_fingerprint = bucket_fingerprint(source);
    let bucket_program_fingerprint = identity
        .bucket_program_fingerprints
        .get(source.bucket_index)
        .copied()
        .map(program_fingerprint)
        .ok_or(DecoderError::RelocationProfile("bucket program missing"))?;
    Ok(RelocationCostProfile {
        identity: RelocationCostIdentity {
            plan_fingerprint: identity.manager_plan_fingerprint,
            compiler_facts_digest: identity.compiler_facts_digest,
            bucket_fingerprint,
            artifact_fingerprint: identity.artifact_fingerprint,
            schedule_fingerprint: identity.schedule_fingerprint,
            bucket_program_fingerprint,
            source_execution_fingerprint: source.fingerprint,
            target_execution_fingerprint: target.fingerprint,
            class_id: envelope.class_id,
            source_layout: StateLayoutAlternative::TokenSelectionMask,
            target_layout: StateLayoutAlternative::PackedTokenSlots,
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

fn state_geometry(
    profile: &ExecutionGeometryProfile,
    class_id: u16,
) -> Result<ExecutionStateGeometry, DecoderError> {
    profile
        .state_geometry
        .iter()
        .find(|state| state.class_id == class_id)
        .copied()
        .ok_or(DecoderError::RelocationProfile(
            "relocation class is absent from execution geometry",
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

fn bucket_fingerprint(profile: &ExecutionGeometryProfile) -> [u8; 32] {
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
    hash.finalize().into()
}

fn non_layout_dimensions<'a>(
    dimensions: &'a [SelectedBucketDimension],
    states: &[ExecutionStateGeometry],
) -> Vec<(&'a str, usize, usize, usize, usize)> {
    dimensions
        .iter()
        .filter(|dimension| !is_layout_dimension(&dimension.name, states))
        .map(|dimension| {
            (
                dimension.name.as_str(),
                dimension.bucket_index,
                dimension.minimum,
                dimension.maximum,
                dimension.representative,
            )
        })
        .collect()
}

fn non_layout_execution_dimensions<'a>(
    dimensions: &'a [(String, usize)],
    states: &[ExecutionStateGeometry],
) -> Vec<(&'a str, usize)> {
    dimensions
        .iter()
        .filter(|(name, _)| !is_layout_dimension(name, states))
        .map(|(name, value)| (name.as_str(), *value))
        .collect()
}

fn is_layout_dimension(name: &str, states: &[ExecutionStateGeometry]) -> bool {
    states.iter().any(|state| {
        name == format!("c_{}", state.class_id) || name == format!("p_{}", state.class_id)
    })
}

fn execution_fingerprint(
    bucket_index: usize,
    dimensions: &[(String, usize)],
    states: &[ExecutionStateGeometry],
) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update((bucket_index as u64).to_le_bytes());
    for (name, value) in dimensions {
        hash_bytes(&mut hash, name.as_bytes());
        hash.update((*value as u64).to_le_bytes());
    }
    for state in states {
        hash.update(state.class_id.to_le_bytes());
        hash.update(state.page_tokens.to_le_bytes());
        hash.update((state.page_count as u64).to_le_bytes());
        hash.update(state.token_count.to_le_bytes());
    }
    hash.finalize().into()
}

fn program_fingerprint(program: [u64; 2]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(program[0].to_le_bytes());
    hash.update(program[1].to_le_bytes());
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

    fn identity() -> DecoderProfileIdentity {
        DecoderProfileIdentity {
            manager_plan_fingerprint: [2; 32],
            compiler_facts_digest: [3; 32],
            artifact_fingerprint: [4; 32],
            schedule_fingerprint: [5; 32],
            bucket_program_fingerprints: vec![[6, 1]].into_boxed_slice(),
        }
    }

    fn geometry(page_tokens: u32, pages: usize, time_ns: u64) -> ExecutionGeometryProfile {
        let identity = identity();
        let bucket_dimensions = vec![SelectedBucketDimension {
            name: "s".into(),
            bucket_index: 0,
            minimum: 1,
            maximum: 1,
            representative: 1,
        }];
        let execution_dimensions = vec![
            ("c_0".into(), pages),
            ("p_0".into(), usize::try_from(page_tokens).unwrap()),
            ("s".into(), 1),
        ];
        let state_geometry = vec![ExecutionStateGeometry {
            class_id: 0,
            page_tokens,
            page_count: pages,
            token_count: 8,
        }]
        .into_boxed_slice();
        ExecutionGeometryProfile {
            executable_fingerprint: identity.fingerprint(),
            bucket_index: 0,
            bucket_dimensions,
            fingerprint: execution_fingerprint(0, &execution_dimensions, &state_geometry),
            execution_dimensions,
            state_geometry,
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
        let source_identity = identity();
        let profile = build_profile_from_parts(
            &source_identity,
            &geometry(1, 8, 100),
            &geometry(16, 1, 70),
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
        let source_identity = identity();
        let source = geometry(1, 8, 100);
        let target = geometry(16, 1, 70);
        let mut wrong_bucket = target.clone();
        wrong_bucket.bucket_dimensions[0].maximum = 2;
        assert!(
            build_profile_from_parts(
                &source_identity,
                &source,
                &wrong_bucket,
                envelope(),
                bandwidth(),
            )
            .is_err()
        );

        let wrong_layout = geometry(1, 8, 70);
        assert!(
            build_profile_from_parts(
                &source_identity,
                &source,
                &wrong_layout,
                envelope(),
                bandwidth(),
            )
            .is_err()
        );

        let no_execution_change = source.clone();
        assert!(
            build_profile_from_parts(
                &source_identity,
                &source,
                &no_execution_change,
                envelope(),
                bandwidth(),
            )
            .is_err()
        );

        let mut weak = target;
        weak.trials = 2;
        assert!(
            build_profile_from_parts(&source_identity, &source, &weak, envelope(), bandwidth(),)
                .is_err()
        );

        let mut foreign = geometry(16, 1, 70);
        foreign.executable_fingerprint = [99; 32];
        assert!(
            build_profile_from_parts(&source_identity, &source, &foreign, envelope(), bandwidth(),)
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
