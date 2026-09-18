//! Qualification-aware plan storage and deterministic selection.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use aletheia_contracts::{
    ContractError, HardwareFingerprint, ModelIdentity, PhysicalPlan, QualificationCertificate, ServiceLevelObjective,
    WorkloadPoint,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QualifiedPlan {
    pub plan: PhysicalPlan,
    pub certificate: QualificationCertificate,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SelectionRequest {
    pub model: ModelIdentity,
    pub hardware: HardwareFingerprint,
    pub workload: WorkloadPoint,
    pub slo: ServiceLevelObjective,
    #[serde(default)]
    pub provider_preference: Vec<String>,
    #[serde(default)]
    pub available_providers: BTreeSet<String>,
}

impl SelectionRequest {
    pub fn validate(&self) -> Result<(), ContractError> {
        self.model.validate()?;
        self.hardware.validate()?;
        self.slo.validate()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Selection {
    pub primary: QualifiedPlan,
    pub fallbacks: Vec<QualifiedPlan>,
}

impl Selection {
    pub fn plans(&self) -> impl Iterator<Item = &QualifiedPlan> {
        std::iter::once(&self.primary).chain(&self.fallbacks)
    }
}

#[derive(Debug, Error)]
pub enum RegistryError {
    #[error(transparent)]
    InvalidContract(#[from] ContractError),
    #[error("plan {0:?} is already registered")]
    DuplicatePlan(String),
    #[error("plan {plan:?} references missing fallback {fallback:?}; register fallbacks first")]
    MissingFallback { plan: String, fallback: String },
    #[error("fallback {fallback:?} has different model semantics from plan {plan:?}")]
    IncompatibleFallback { plan: String, fallback: String },
    #[error("no qualified plan satisfies the selection request")]
    NoEligiblePlan,
}

#[derive(Default)]
pub struct PlanRegistry {
    plans: BTreeMap<String, QualifiedPlan>,
}

impl PlanRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, plan: PhysicalPlan, certificate: QualificationCertificate) -> Result<(), RegistryError> {
        plan.validate()?;
        certificate.validate_for(&plan)?;
        if self.plans.contains_key(&plan.id) {
            return Err(RegistryError::DuplicatePlan(plan.id));
        }
        for fallback in &plan.fallback_plan_ids {
            let Some(record) = self.plans.get(fallback) else {
                return Err(RegistryError::MissingFallback { plan: plan.id, fallback: fallback.clone() });
            };
            if record.plan.model.semantics_sha256 != plan.model.semantics_sha256 {
                return Err(RegistryError::IncompatibleFallback { plan: plan.id, fallback: fallback.clone() });
            }
        }
        self.plans.insert(plan.id.clone(), QualifiedPlan { plan, certificate });
        Ok(())
    }

    pub fn get(&self, id: &str) -> Option<&QualifiedPlan> {
        self.plans.get(id)
    }

    pub fn len(&self) -> usize {
        self.plans.len()
    }

    pub fn is_empty(&self) -> bool {
        self.plans.is_empty()
    }

    pub fn select(&self, request: &SelectionRequest) -> Result<Selection, RegistryError> {
        request.validate()?;
        let mut candidates = self.plans.values().filter(|record| eligible(record, request)).collect::<Vec<_>>();
        candidates.sort_by(|left, right| rank(left, right, request));
        let primary = (*candidates.first().ok_or(RegistryError::NoEligiblePlan)?).clone();
        let fallbacks = primary
            .plan
            .fallback_plan_ids
            .iter()
            .filter_map(|id| self.plans.get(id))
            .filter(|record| eligible(record, request))
            .cloned()
            .collect();
        Ok(Selection { primary, fallbacks })
    }
}

fn eligible(record: &QualifiedPlan, request: &SelectionRequest) -> bool {
    let plan = &record.plan;
    let evidence = &record.certificate;
    plan.model.semantics_sha256 == request.model.semantics_sha256
        && plan.hardware.supports(&request.hardware)
        && evidence.hardware == request.hardware
        && plan.workload.contains(&request.workload)
        && evidence.workload.contains(&request.workload)
        && plan.resources.peak_device_memory_bytes <= request.slo.max_device_memory_bytes
        && plan.resources.peak_device_memory_bytes <= request.hardware.device_memory_bytes
        && evidence.resources.peak_device_memory_bytes <= request.slo.max_device_memory_bytes
        && evidence.performance.p99_latency_micros <= request.slo.max_p99_latency_micros
        && evidence.performance.goodput_tokens_per_second >= request.slo.min_goodput_tokens_per_second
        && evidence.numerical.max_abs_error <= request.slo.max_abs_error
        && evidence.numerical.min_output_agreement >= request.slo.min_output_agreement
        && plan
            .steps
            .iter()
            .all(|step| request.available_providers.is_empty() || request.available_providers.contains(&step.provider))
}

fn rank(left: &QualifiedPlan, right: &QualifiedPlan, request: &SelectionRequest) -> Ordering {
    let provider_rank = |record: &QualifiedPlan| {
        record
            .plan
            .steps
            .iter()
            .map(|step| {
                request
                    .provider_preference
                    .iter()
                    .position(|provider| provider == &step.provider)
                    .unwrap_or(request.provider_preference.len())
            })
            .min()
            .unwrap_or(request.provider_preference.len())
    };

    provider_rank(left)
        .cmp(&provider_rank(right))
        .then_with(|| {
            right
                .certificate
                .performance
                .goodput_tokens_per_second
                .total_cmp(&left.certificate.performance.goodput_tokens_per_second)
        })
        .then_with(|| {
            left.certificate.performance.p99_latency_micros.cmp(&right.certificate.performance.p99_latency_micros)
        })
        .then_with(|| left.plan.resources.peak_device_memory_bytes.cmp(&right.plan.resources.peak_device_memory_bytes))
        .then_with(|| left.plan.id.cmp(&right.plan.id))
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use aletheia_contracts::{
        CaptureMode, HardwareTarget, InclusiveRangeU32, NumericalEvidence, PerformanceEvidence, Phase, ProviderStep,
        QualificationStatus, ResilienceEvidence, ResourceEvidence, ResourceRequirements, SCHEMA_VERSION,
        WorkloadDomain,
    };

    use super::*;

    fn hash(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn hardware() -> HardwareFingerprint {
        HardwareFingerprint {
            accelerator: "cpu".into(),
            architecture: "reference".into(),
            device_count: 1,
            device_memory_bytes: 4096,
            driver: "none".into(),
            runtime: "rust".into(),
        }
    }

    fn plan(id: &str, provider: &str, fallback: Vec<String>) -> PhysicalPlan {
        PhysicalPlan {
            schema_version: SCHEMA_VERSION,
            id: id.into(),
            model: ModelIdentity { name: "toy".into(), semantics_sha256: hash('a') },
            hardware: HardwareTarget {
                accelerator: "cpu".into(),
                architecture: "reference".into(),
                device_count: 1,
                min_device_memory_bytes: 1024,
            },
            workload: WorkloadDomain {
                phases: vec![Phase::Decode],
                batch: InclusiveRangeU32 { min: 1, max: 8 },
                rows_per_sequence: InclusiveRangeU32 { min: 1, max: 1 },
                context_tokens: InclusiveRangeU32 { min: 0, max: 4096 },
            },
            resources: ResourceRequirements {
                peak_device_memory_bytes: 2048,
                workspace_bytes: 128,
                stable_addresses: false,
                capture: CaptureMode::Eager,
            },
            steps: vec![ProviderStep {
                id: "forward".into(),
                provider: provider.into(),
                operation: "identity".into(),
                artifact_sha256: hash('b'),
                inputs: vec!["input".into()],
                outputs: vec!["output".into()],
                config: BTreeMap::new(),
            }],
            fallback_plan_ids: fallback,
            labels: BTreeMap::new(),
        }
    }

    fn certificate(plan: &PhysicalPlan, goodput: f64, p99: u64) -> QualificationCertificate {
        QualificationCertificate {
            schema_version: SCHEMA_VERSION,
            plan_id: plan.id.clone(),
            plan_sha256: plan.digest().unwrap(),
            status: QualificationStatus::Qualified,
            hardware: hardware(),
            workload: plan.workload.clone(),
            numerical: NumericalEvidence {
                oracle: "reference".into(),
                cases: 32,
                atol: 0.0,
                rtol: 0.0,
                max_abs_error: 0.0,
                min_output_agreement: 1.0,
            },
            performance: PerformanceEvidence {
                workload_sha256: hash('c'),
                samples: 32,
                p50_latency_micros: p99 / 2,
                p99_latency_micros: p99,
                goodput_tokens_per_second: goodput,
            },
            resources: ResourceEvidence { peak_device_memory_bytes: 1024, peak_workspace_bytes: 64 },
            resilience: ResilienceEvidence { soak_seconds: 60, passed_faults: vec!["provider_failure".into()] },
            software: BTreeMap::new(),
        }
    }

    fn request() -> SelectionRequest {
        SelectionRequest {
            model: ModelIdentity { name: "toy".into(), semantics_sha256: hash('a') },
            hardware: hardware(),
            workload: WorkloadPoint { phase: Phase::Decode, batch: 4, rows_per_sequence: 1, context_tokens: 1024 },
            slo: ServiceLevelObjective {
                max_p99_latency_micros: 100,
                min_goodput_tokens_per_second: 1.0,
                max_device_memory_bytes: 4096,
                max_abs_error: 0.0,
                min_output_agreement: 1.0,
            },
            provider_preference: Vec::new(),
            available_providers: BTreeSet::new(),
        }
    }

    #[test]
    fn selects_highest_goodput_and_keeps_declared_fallback() {
        let mut registry = PlanRegistry::new();
        let fallback = plan("safe", "reference", Vec::new());
        registry.register(fallback.clone(), certificate(&fallback, 10.0, 80)).unwrap();
        let primary = plan("fast", "experimental", vec![fallback.id.clone()]);
        registry.register(primary.clone(), certificate(&primary, 20.0, 70)).unwrap();

        let selected = registry.select(&request()).unwrap();
        assert_eq!(selected.primary.plan.id, "fast");
        assert_eq!(selected.fallbacks[0].plan.id, "safe");
    }

    #[test]
    fn exact_hardware_evidence_is_required() {
        let mut registry = PlanRegistry::new();
        let plan = plan("safe", "reference", Vec::new());
        registry.register(plan.clone(), certificate(&plan, 10.0, 80)).unwrap();
        let mut request = request();
        request.hardware.driver = "different".into();
        assert!(matches!(registry.select(&request), Err(RegistryError::NoEligiblePlan)));
    }

    #[test]
    fn registration_rejects_tampered_plan() {
        let mut registry = PlanRegistry::new();
        let mut plan = plan("safe", "reference", Vec::new());
        let certificate = certificate(&plan, 10.0, 80);
        plan.steps[0].operation = "tampered".into();
        assert!(registry.register(plan, certificate).unwrap_err().to_string().contains("does not match"));
    }
}
