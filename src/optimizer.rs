use serde::Serialize;
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{CompiledKvPlan, PlanError, RetentionKind, SglangPolicy};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PhysicalPlanObjective {
    CapacityUnderReclamationBudget,
    ReclamationUnderAdmissionTarget,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SglangPhysicalOptimizationInput {
    pub available_kv_bytes: u64,
    pub max_running_requests: u64,
    pub attention_data_parallel_size: u64,
    pub chunked_prefill_tokens: u64,
    pub workload_requests: u64,
    pub prompt_tokens_per_request: u64,
    pub decode_tokens_per_request: u64,
    pub candidate_eviction_intervals: Vec<u64>,
    pub maximum_reclamation_calls_per_request: Option<u64>,
    pub minimum_admitted_requests: Option<u64>,
    pub objective: PhysicalPlanObjective,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SglangPhysicalContract {
    pub cache_kind: &'static str,
    pub require_radix_cache_disabled: bool,
    pub require_overlap_schedule_disabled: bool,
    pub require_speculative_decoding_disabled: bool,
    pub disaggregation_mode: &'static str,
    pub chunks_in_flight: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SglangPhysicalCost {
    pub semantic_swa_token_slots_per_request: u64,
    pub physical_swa_token_slots: u64,
    pub physical_swa_bytes: u64,
    pub full_token_capacity: u64,
    pub physical_full_bytes: u64,
    pub allocated_kv_bytes: u64,
    pub unallocated_kv_bytes: u64,
    pub retention_amplification_milli: u64,
    pub admitted_requests: u64,
    pub admission_waves: u64,
    pub estimated_prefill_reclamation_calls_per_request: u64,
    pub estimated_decode_reclamation_calls_per_request: u64,
    pub estimated_reclamation_calls_per_request: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SglangPhysicalCandidate {
    pub eviction_interval_tokens: u64,
    pub feasible: bool,
    pub rejection_reasons: Vec<String>,
    pub cost: Option<SglangPhysicalCost>,
    pub policy: Option<SglangPolicy>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SglangPhysicalPlan {
    pub schema: &'static str,
    pub plan_fingerprint: String,
    pub physical_plan_fingerprint: String,
    pub objective: PhysicalPlanObjective,
    pub input: SglangPhysicalOptimizationInput,
    pub contract: SglangPhysicalContract,
    pub selected_eviction_interval_tokens: u64,
    pub selected: SglangPhysicalCandidate,
    pub candidates: Vec<SglangPhysicalCandidate>,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum OptimizerError {
    #[error("physical optimizer available_kv_bytes must be positive")]
    ZeroKvBudget,
    #[error("physical optimizer max_running_requests must be positive")]
    ZeroMaxRunningRequests,
    #[error("physical optimizer attention_data_parallel_size must be positive")]
    ZeroAttentionDpSize,
    #[error(
        "physical optimizer attention_data_parallel_size {attention_dp_size} exceeds max_running_requests {max_running_requests}"
    )]
    AttentionDpExceedsRequests {
        attention_dp_size: u64,
        max_running_requests: u64,
    },
    #[error("physical optimizer chunked_prefill_tokens must be positive")]
    ZeroChunkedPrefill,
    #[error("physical optimizer workload_requests must be positive")]
    ZeroWorkloadRequests,
    #[error("physical optimizer prompt+decode tokens must be positive")]
    EmptyRequest,
    #[error("physical optimizer requires at least one eviction interval candidate")]
    EmptyCandidates,
    #[error("physical optimizer requires at least one Full class")]
    MissingFullClass,
    #[error("physical optimizer requires exactly one sliding class, found {0}")]
    SlidingClassCount(usize),
    #[error("physical optimizer does not support partitioned block domains")]
    PartitionedBlockDomain,
    #[error("physical optimizer found no feasible candidate")]
    NoFeasibleCandidate,
    #[error("integer overflow while calculating {0}")]
    ArithmeticOverflow(&'static str),
    #[error(transparent)]
    Plan(#[from] PlanError),
}

#[derive(Clone, Copy)]
struct PlanGeometry {
    page_tokens: u64,
    window_tokens: u64,
    full_bytes_per_token: u64,
    swa_bytes_per_token: u64,
}

/// Synthesizes an engine-specific physical plan from a compiled lifetime plan,
/// an explicit KV budget, and a request workload.
///
/// The first optimizer target exactly models `SGLang`'s non-overlap SWA chunk
/// cache. Every candidate is page aligned and includes per-request window,
/// eviction overshoot, decode-page slack, chunked-prefill staging, and the
/// final sentinel page. Selection is constraint based, not a hidden learned
/// score.
///
/// # Errors
///
/// Returns an error for unsupported plan geometry, invalid workload inputs,
/// checked arithmetic failure, or if no candidate satisfies the declared
/// reclamation and admission constraints.
pub fn optimize_sglang_physical_plan(
    plan: &CompiledKvPlan,
    input: &SglangPhysicalOptimizationInput,
) -> Result<SglangPhysicalPlan, OptimizerError> {
    validate_input(input)?;
    let geometry = plan_geometry(plan)?;
    let mut canonical_input = input.clone();
    canonical_input.candidate_eviction_intervals.sort_unstable();
    canonical_input.candidate_eviction_intervals.dedup();
    let mut candidates = canonical_input
        .candidate_eviction_intervals
        .iter()
        .map(|&interval| evaluate_candidate(plan, geometry, &canonical_input, interval))
        .collect::<Result<Vec<_>, OptimizerError>>()?;
    candidates.sort_by_key(|candidate| candidate.eviction_interval_tokens);
    let selected_index = candidates
        .iter()
        .enumerate()
        .filter(|(_, candidate)| candidate.feasible)
        .min_by(|(_, lhs), (_, rhs)| compare_candidates(lhs, rhs, canonical_input.objective))
        .map(|(index, _)| index)
        .ok_or(OptimizerError::NoFeasibleCandidate)?;
    let selected = candidates[selected_index].clone();
    let physical_plan_fingerprint = physical_fingerprint(
        &plan.fingerprint(),
        &canonical_input,
        selected.eviction_interval_tokens,
    );
    Ok(SglangPhysicalPlan {
        schema: "orbitkv.sglang-physical-plan.v1",
        plan_fingerprint: plan.fingerprint(),
        physical_plan_fingerprint,
        objective: canonical_input.objective,
        input: canonical_input,
        contract: SglangPhysicalContract {
            cache_kind: "swa_chunk_cache",
            require_radix_cache_disabled: true,
            require_overlap_schedule_disabled: true,
            require_speculative_decoding_disabled: true,
            disaggregation_mode: "none",
            chunks_in_flight: 1,
        },
        selected_eviction_interval_tokens: selected.eviction_interval_tokens,
        selected,
        candidates,
    })
}

fn physical_fingerprint(
    plan_fingerprint: &str,
    input: &SglangPhysicalOptimizationInput,
    selected_interval: u64,
) -> String {
    let mut hash = Sha256::new();
    update_bytes(&mut hash, plan_fingerprint.as_bytes());
    for value in [
        input.available_kv_bytes,
        input.max_running_requests,
        input.attention_data_parallel_size,
        input.chunked_prefill_tokens,
        input.workload_requests,
        input.prompt_tokens_per_request,
        input.decode_tokens_per_request,
        input
            .maximum_reclamation_calls_per_request
            .unwrap_or(u64::MAX),
        input.minimum_admitted_requests.unwrap_or(u64::MAX),
        selected_interval,
    ] {
        hash.update(value.to_le_bytes());
    }
    hash.update(
        match input.objective {
            PhysicalPlanObjective::CapacityUnderReclamationBudget => 0_u64,
            PhysicalPlanObjective::ReclamationUnderAdmissionTarget => 1_u64,
        }
        .to_le_bytes(),
    );
    hash.update(
        u64::try_from(input.candidate_eviction_intervals.len())
            .unwrap_or(u64::MAX)
            .to_le_bytes(),
    );
    for &interval in &input.candidate_eviction_intervals {
        hash.update(interval.to_le_bytes());
    }
    format!("sha256:{:x}", hash.finalize())
}

fn update_bytes(hash: &mut Sha256, bytes: &[u8]) {
    hash.update((bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
}

fn validate_input(input: &SglangPhysicalOptimizationInput) -> Result<(), OptimizerError> {
    if input.available_kv_bytes == 0 {
        return Err(OptimizerError::ZeroKvBudget);
    }
    if input.max_running_requests == 0 {
        return Err(OptimizerError::ZeroMaxRunningRequests);
    }
    if input.attention_data_parallel_size == 0 {
        return Err(OptimizerError::ZeroAttentionDpSize);
    }
    if input.attention_data_parallel_size > input.max_running_requests {
        return Err(OptimizerError::AttentionDpExceedsRequests {
            attention_dp_size: input.attention_data_parallel_size,
            max_running_requests: input.max_running_requests,
        });
    }
    if input.chunked_prefill_tokens == 0 {
        return Err(OptimizerError::ZeroChunkedPrefill);
    }
    if input.workload_requests == 0 {
        return Err(OptimizerError::ZeroWorkloadRequests);
    }
    if input.prompt_tokens_per_request == 0 && input.decode_tokens_per_request == 0 {
        return Err(OptimizerError::EmptyRequest);
    }
    if input.candidate_eviction_intervals.is_empty() {
        return Err(OptimizerError::EmptyCandidates);
    }
    Ok(())
}

fn plan_geometry(plan: &CompiledKvPlan) -> Result<PlanGeometry, OptimizerError> {
    if plan
        .classes
        .iter()
        .any(|class| !class.block_domain.is_all())
    {
        return Err(OptimizerError::PartitionedBlockDomain);
    }
    let full = plan
        .classes
        .iter()
        .filter(|class| class.spec.retention == RetentionKind::Full)
        .collect::<Vec<_>>();
    if full.is_empty() {
        return Err(OptimizerError::MissingFullClass);
    }
    let sliding = plan
        .classes
        .iter()
        .filter(|class| class.spec.retention == RetentionKind::Sliding)
        .collect::<Vec<_>>();
    if sliding.len() != 1 {
        return Err(OptimizerError::SlidingClassCount(sliding.len()));
    }
    let full_bytes_per_token = class_bytes_per_token(&full)?;
    let swa_bytes_per_token = class_bytes_per_token(&sliding)?;
    let window_tokens =
        sliding[0]
            .spec
            .window_tokens
            .ok_or_else(|| PlanError::InvalidCompiledClass {
                class: sliding[0].spec.name.clone(),
            })?;
    Ok(PlanGeometry {
        page_tokens: plan.page_tokens,
        window_tokens,
        full_bytes_per_token,
        swa_bytes_per_token,
    })
}

fn class_bytes_per_token(classes: &[&crate::CompiledKvClass]) -> Result<u64, OptimizerError> {
    classes.iter().try_fold(0_u64, |total, class| {
        let layers = u64::try_from(class.spec.layers.len())
            .map_err(|_| OptimizerError::ArithmeticOverflow("class layer count"))?;
        let bytes = layers
            .checked_mul(class.spec.bytes_per_token_per_layer)
            .ok_or(OptimizerError::ArithmeticOverflow("class bytes per token"))?;
        total
            .checked_add(bytes)
            .ok_or(OptimizerError::ArithmeticOverflow(
                "total class bytes per token",
            ))
    })
}

fn evaluate_candidate(
    plan: &CompiledKvPlan,
    geometry: PlanGeometry,
    input: &SglangPhysicalOptimizationInput,
    interval: u64,
) -> Result<SglangPhysicalCandidate, OptimizerError> {
    let mut rejection_reasons = Vec::new();
    if interval == 0 || !interval.is_multiple_of(geometry.page_tokens) {
        rejection_reasons.push(format!(
            "interval must be a positive multiple of page_tokens {}",
            geometry.page_tokens
        ));
        return Ok(SglangPhysicalCandidate {
            eviction_interval_tokens: interval,
            feasible: false,
            rejection_reasons,
            cost: None,
            policy: None,
        });
    }
    let policy = plan.sglang_policy_with_eviction_interval(interval)?;
    let cost = calculate_candidate_cost(geometry, input, interval, &policy)?;
    rejection_reasons.extend(candidate_rejections(&cost, input));
    Ok(SglangPhysicalCandidate {
        eviction_interval_tokens: interval,
        feasible: rejection_reasons.is_empty(),
        rejection_reasons,
        cost: Some(cost),
        policy: Some(policy),
    })
}

fn calculate_candidate_cost(
    geometry: PlanGeometry,
    input: &SglangPhysicalOptimizationInput,
    interval: u64,
    policy: &SglangPolicy,
) -> Result<SglangPhysicalCost, OptimizerError> {
    let physical_swa_token_slots = sglang_chunk_cache_swa_slots(
        geometry,
        input.max_running_requests,
        input.attention_data_parallel_size,
        input.chunked_prefill_tokens,
        interval,
    )?;
    let physical_swa_bytes = physical_swa_token_slots
        .checked_mul(geometry.swa_bytes_per_token)
        .ok_or(OptimizerError::ArithmeticOverflow("physical SWA bytes"))?;
    let available_full_bytes = input.available_kv_bytes.saturating_sub(physical_swa_bytes);
    let raw_full_tokens = available_full_bytes / geometry.full_bytes_per_token;
    let full_token_capacity = align_down(raw_full_tokens, geometry.page_tokens);
    let physical_full_bytes = full_token_capacity
        .checked_mul(geometry.full_bytes_per_token)
        .ok_or(OptimizerError::ArithmeticOverflow("physical Full bytes"))?;
    let allocated_kv_bytes = physical_swa_bytes
        .checked_add(physical_full_bytes)
        .ok_or(OptimizerError::ArithmeticOverflow("allocated KV bytes"))?;
    let unallocated_kv_bytes = input.available_kv_bytes.saturating_sub(allocated_kv_bytes);

    let request_tokens = input
        .prompt_tokens_per_request
        .checked_add(input.decode_tokens_per_request)
        .ok_or(OptimizerError::ArithmeticOverflow(
            "request sequence tokens",
        ))?;
    let request_full_slots = align_up(request_tokens, geometry.page_tokens)?;
    let admitted_per_attention_dp = full_token_capacity / request_full_slots;
    let admitted_requests = admitted_per_attention_dp
        .checked_mul(input.attention_data_parallel_size)
        .ok_or(OptimizerError::ArithmeticOverflow(
            "global admitted requests",
        ))?
        .min(input.max_running_requests)
        .min(input.workload_requests);
    let admission_waves = if admitted_requests == 0 {
        u64::MAX
    } else {
        ceil_div(input.workload_requests, admitted_requests)?
    };

    let prefill_calls = ceil_div(
        input.prompt_tokens_per_request,
        input.chunked_prefill_tokens,
    )?;
    let decode_calls = ceil_div(input.decode_tokens_per_request, interval)?;
    let total_calls =
        prefill_calls
            .checked_add(decode_calls)
            .ok_or(OptimizerError::ArithmeticOverflow(
                "reclamation calls per request",
            ))?;
    let requests_per_attention_dp = input.max_running_requests / input.attention_data_parallel_size;
    let semantic_request_slots = policy
        .max_persistent_swa_token_slots_per_request
        .checked_mul(requests_per_attention_dp)
        .and_then(|value| value.checked_add(input.chunked_prefill_tokens))
        .and_then(|value| value.checked_add(geometry.page_tokens))
        .ok_or(OptimizerError::ArithmeticOverflow(
            "semantic SWA pool floor",
        ))?;
    let retention_amplification_milli =
        physical_swa_token_slots
            .checked_mul(1000)
            .ok_or(OptimizerError::ArithmeticOverflow(
                "retention amplification",
            ))?
            / semantic_request_slots.max(1);
    Ok(SglangPhysicalCost {
        semantic_swa_token_slots_per_request: policy.max_persistent_swa_token_slots_per_request,
        physical_swa_token_slots,
        physical_swa_bytes,
        full_token_capacity,
        physical_full_bytes,
        allocated_kv_bytes,
        unallocated_kv_bytes,
        retention_amplification_milli,
        admitted_requests,
        admission_waves,
        estimated_prefill_reclamation_calls_per_request: prefill_calls,
        estimated_decode_reclamation_calls_per_request: decode_calls,
        estimated_reclamation_calls_per_request: total_calls,
    })
}

fn candidate_rejections(
    cost: &SglangPhysicalCost,
    input: &SglangPhysicalOptimizationInput,
) -> Vec<String> {
    let mut reasons = Vec::new();
    if input
        .maximum_reclamation_calls_per_request
        .is_some_and(|maximum| cost.estimated_reclamation_calls_per_request > maximum)
    {
        reasons.push(format!(
            "estimated reclamation calls {} exceed maximum {}",
            cost.estimated_reclamation_calls_per_request,
            input
                .maximum_reclamation_calls_per_request
                .unwrap_or_default()
        ));
    }
    if input
        .minimum_admitted_requests
        .is_some_and(|minimum| cost.admitted_requests < minimum)
    {
        reasons.push(format!(
            "admitted requests {} below minimum {}",
            cost.admitted_requests,
            input.minimum_admitted_requests.unwrap_or_default()
        ));
    }
    if cost.admitted_requests == 0 {
        reasons.push("Full pool cannot admit one workload request".into());
    }
    if cost.physical_swa_bytes > input.available_kv_bytes {
        reasons.push(format!(
            "physical SWA bytes {} exceed KV budget {}",
            cost.physical_swa_bytes, input.available_kv_bytes
        ));
    }
    reasons
}

fn sglang_chunk_cache_swa_slots(
    geometry: PlanGeometry,
    max_running_requests: u64,
    attention_data_parallel_size: u64,
    chunked_prefill_tokens: u64,
    interval: u64,
) -> Result<u64, OptimizerError> {
    let sglang_window_left = geometry.window_tokens - 1;
    let per_request = sglang_window_left
        .checked_add(interval)
        .and_then(|value| value.checked_add(2 * geometry.page_tokens))
        .ok_or(OptimizerError::ArithmeticOverflow(
            "SGLang SWA slots per request",
        ))?;
    let requests_per_attention_dp = max_running_requests / attention_data_parallel_size;
    let unaligned = per_request
        .checked_mul(requests_per_attention_dp)
        .and_then(|value| value.checked_add(chunked_prefill_tokens))
        .and_then(|value| value.checked_add(geometry.page_tokens))
        .ok_or(OptimizerError::ArithmeticOverflow("SGLang SWA pool slots"))?;
    align_up(unaligned, geometry.page_tokens)
}

fn compare_candidates(
    lhs: &SglangPhysicalCandidate,
    rhs: &SglangPhysicalCandidate,
    objective: PhysicalPlanObjective,
) -> std::cmp::Ordering {
    let lhs_cost = lhs.cost.as_ref().expect("feasible candidate has cost");
    let rhs_cost = rhs.cost.as_ref().expect("feasible candidate has cost");
    let admission = lhs_cost.admission_waves.cmp(&rhs_cost.admission_waves);
    if admission != std::cmp::Ordering::Equal {
        return admission;
    }
    match objective {
        PhysicalPlanObjective::CapacityUnderReclamationBudget => rhs_cost
            .full_token_capacity
            .cmp(&lhs_cost.full_token_capacity)
            .then_with(|| {
                lhs_cost
                    .estimated_reclamation_calls_per_request
                    .cmp(&rhs_cost.estimated_reclamation_calls_per_request)
            })
            .then_with(|| {
                lhs.eviction_interval_tokens
                    .cmp(&rhs.eviction_interval_tokens)
            }),
        PhysicalPlanObjective::ReclamationUnderAdmissionTarget => lhs_cost
            .estimated_reclamation_calls_per_request
            .cmp(&rhs_cost.estimated_reclamation_calls_per_request)
            .then_with(|| {
                rhs.eviction_interval_tokens
                    .cmp(&lhs.eviction_interval_tokens)
            })
            .then_with(|| {
                rhs_cost
                    .full_token_capacity
                    .cmp(&lhs_cost.full_token_capacity)
            }),
    }
}

const fn align_down(value: u64, alignment: u64) -> u64 {
    value / alignment * alignment
}

fn align_up(value: u64, alignment: u64) -> Result<u64, OptimizerError> {
    ceil_div(value, alignment)?
        .checked_mul(alignment)
        .ok_or(OptimizerError::ArithmeticOverflow("aligned value"))
}

fn ceil_div(value: u64, divisor: u64) -> Result<u64, OptimizerError> {
    if value == 0 {
        return Ok(0);
    }
    value
        .checked_add(divisor - 1)
        .map(|sum| sum / divisor)
        .ok_or(OptimizerError::ArithmeticOverflow("ceiling division"))
}

#[cfg(test)]
mod tests {
    use crate::{KvClassSpec, KvPlanInput, RetentionKind, compile_plan};

    use super::*;

    fn gpt_oss_plan() -> CompiledKvPlan {
        compile_plan(KvPlanInput {
            page_tokens: 16,
            classes: vec![
                KvClassSpec {
                    name: "full".into(),
                    layers: (0..12).collect(),
                    retention: RetentionKind::Full,
                    bytes_per_token_per_layer: 2048,
                    window_tokens: None,
                },
                KvClassSpec {
                    name: "swa".into(),
                    layers: (12..24).collect(),
                    retention: RetentionKind::Sliding,
                    bytes_per_token_per_layer: 2048,
                    window_tokens: Some(128),
                },
            ],
        })
        .unwrap()
    }

    fn pressure_input() -> SglangPhysicalOptimizationInput {
        SglangPhysicalOptimizationInput {
            available_kv_bytes: 2_123_759_616,
            max_running_requests: 128,
            attention_data_parallel_size: 1,
            chunked_prefill_tokens: 2048,
            workload_requests: 8,
            prompt_tokens_per_request: 6000,
            decode_tokens_per_request: 32,
            candidate_eviction_intervals: vec![16, 32, 64, 128],
            maximum_reclamation_calls_per_request: Some(4),
            minimum_admitted_requests: Some(8),
            objective: PhysicalPlanObjective::CapacityUnderReclamationBudget,
        }
    }

    #[test]
    fn reproduces_sglang_pool_capacities_and_selects_interval_32() {
        let optimized = optimize_sglang_physical_plan(&gpt_oss_plan(), &pressure_input()).unwrap();
        assert_eq!(optimized.selected_eviction_interval_tokens, 32);
        assert!(optimized.physical_plan_fingerprint.starts_with("sha256:"));
        let by_interval = optimized
            .candidates
            .iter()
            .map(|candidate| (candidate.eviction_interval_tokens, candidate))
            .collect::<std::collections::BTreeMap<_, _>>();
        assert!(!by_interval[&16].feasible);
        assert_eq!(
            by_interval[&16].rejection_reasons,
            vec!["estimated reclamation calls 5 exceed maximum 4"]
        );
        let stock = by_interval[&128].cost.as_ref().unwrap();
        assert_eq!(stock.physical_swa_token_slots, 38_800);
        assert_eq!(stock.full_token_capacity, 47_616);
        assert_eq!(stock.admitted_requests, 7);
        assert_eq!(stock.admission_waves, 2);
        let selected = by_interval[&32].cost.as_ref().unwrap();
        assert_eq!(selected.physical_swa_token_slots, 26_512);
        assert_eq!(selected.full_token_capacity, 59_904);
        assert_eq!(selected.admitted_requests, 8);
        assert_eq!(selected.admission_waves, 1);
    }

    #[test]
    fn reclamation_objective_prefers_larger_interval_after_admission_target() {
        let mut input = pressure_input();
        input.objective = PhysicalPlanObjective::ReclamationUnderAdmissionTarget;
        input.maximum_reclamation_calls_per_request = None;
        let optimized = optimize_sglang_physical_plan(&gpt_oss_plan(), &input).unwrap();
        assert_eq!(optimized.selected_eviction_interval_tokens, 64);
    }

    #[test]
    fn partitioned_domains_fail_closed() {
        let mut plan = gpt_oss_plan();
        plan.classes[1].block_domain.start_block = 1;
        assert_eq!(
            optimize_sglang_physical_plan(&plan, &pressure_input()),
            Err(OptimizerError::PartitionedBlockDomain)
        );
    }

    #[test]
    fn physical_fingerprint_is_canonical_and_strategy_specific() {
        let mut reordered = pressure_input();
        reordered.candidate_eviction_intervals = vec![128, 32, 16, 64, 32];
        let first = optimize_sglang_physical_plan(&gpt_oss_plan(), &pressure_input()).unwrap();
        let second = optimize_sglang_physical_plan(&gpt_oss_plan(), &reordered).unwrap();
        assert_eq!(
            first.physical_plan_fingerprint,
            second.physical_plan_fingerprint
        );

        let mut different_budget = pressure_input();
        different_budget.available_kv_bytes += 16 * 12 * 2048;
        let different = optimize_sglang_physical_plan(&gpt_oss_plan(), &different_budget).unwrap();
        assert_ne!(
            first.physical_plan_fingerprint,
            different.physical_plan_fingerprint
        );
    }
}
