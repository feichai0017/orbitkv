//! Provider-independent preparation, execution, and whole-plan fallback.

use std::collections::BTreeMap;
use std::sync::Arc;

use aletheia_contracts::{ContractError, ProviderStep, WorkloadDomain, WorkloadPoint};
use aletheia_control::{QualifiedPlan, Selection};
use thiserror::Error;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    F32(Vec<f32>),
    I64(Vec<i64>),
    Bytes(Vec<u8>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExecutionState {
    pub workload: WorkloadPoint,
    pub values: BTreeMap<String, Value>,
}

impl ExecutionState {
    pub fn new(workload: WorkloadPoint) -> Self {
        Self { workload, values: BTreeMap::new() }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProviderFailure {
    pub provider: String,
    pub step: String,
    pub message: String,
}

impl ProviderFailure {
    pub fn new(provider: impl Into<String>, step: impl Into<String>, message: impl Into<String>) -> Self {
        Self { provider: provider.into(), step: step.into(), message: message.into() }
    }
}

pub trait PreparedStep: Send + Sync {
    fn execute(&self, state: &mut ExecutionState) -> Result<(), ProviderFailure>;
}

pub trait Provider: Send + Sync {
    fn name(&self) -> &str;
    fn prepare(&self, step: &ProviderStep) -> Result<Box<dyn PreparedStep>, ProviderFailure>;
}

#[derive(Debug, Error)]
pub enum PrepareError {
    #[error("plan {plan:?} is not a valid qualified plan: {source}")]
    InvalidQualifiedPlan { plan: String, source: ContractError },
    #[error("provider {provider:?} required by plan {plan:?} is not installed")]
    MissingProvider { plan: String, provider: String },
    #[error("failed to prepare plan {plan:?}: {failure:?}")]
    Provider { plan: String, failure: ProviderFailure },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FailedAttempt {
    pub plan_id: String,
    pub failure: ProviderFailure,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExecutionReport {
    pub plan_id: String,
    pub failed_attempts: Vec<FailedAttempt>,
    pub state: ExecutionState,
}

#[derive(Debug, Error)]
#[error("every qualified plan failed")]
pub struct ExecutionError {
    pub attempts: Vec<FailedAttempt>,
}

#[derive(Default)]
pub struct Executor {
    providers: BTreeMap<String, Arc<dyn Provider>>,
}

impl Executor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_provider(&mut self, provider: Arc<dyn Provider>) -> bool {
        self.providers.insert(provider.name().to_owned(), provider).is_none()
    }

    pub fn provider_names(&self) -> impl Iterator<Item = &str> {
        self.providers.keys().map(String::as_str)
    }

    pub fn prepare(&self, selection: Selection) -> Result<PreparedSelection, PrepareError> {
        let plans = selection.plans().map(|qualified| self.prepare_plan(qualified)).collect::<Result<Vec<_>, _>>()?;
        Ok(PreparedSelection { plans })
    }

    fn prepare_plan(&self, qualified: &QualifiedPlan) -> Result<PreparedPlan, PrepareError> {
        qualified
            .plan
            .validate()
            .map_err(|source| PrepareError::InvalidQualifiedPlan { plan: qualified.plan.id.clone(), source })?;
        qualified
            .certificate
            .validate_for(&qualified.plan)
            .map_err(|source| PrepareError::InvalidQualifiedPlan { plan: qualified.plan.id.clone(), source })?;
        let mut steps = Vec::with_capacity(qualified.plan.steps.len());
        for step in &qualified.plan.steps {
            let provider = self.providers.get(&step.provider).ok_or_else(|| PrepareError::MissingProvider {
                plan: qualified.plan.id.clone(),
                provider: step.provider.clone(),
            })?;
            let prepared = provider
                .prepare(step)
                .map_err(|failure| PrepareError::Provider { plan: qualified.plan.id.clone(), failure })?;
            steps.push(prepared);
        }
        Ok(PreparedPlan { id: qualified.plan.id.clone(), workload: qualified.plan.workload.clone(), steps })
    }
}

struct PreparedPlan {
    id: String,
    workload: WorkloadDomain,
    steps: Vec<Box<dyn PreparedStep>>,
}

pub struct PreparedSelection {
    plans: Vec<PreparedPlan>,
}

impl PreparedSelection {
    pub fn execute(&self, initial: &ExecutionState) -> Result<ExecutionReport, ExecutionError> {
        let mut failed_attempts = Vec::new();
        for plan in &self.plans {
            let mut state = initial.clone();
            match execute_plan(plan, &mut state) {
                Ok(()) => {
                    return Ok(ExecutionReport { plan_id: plan.id.clone(), failed_attempts, state });
                }
                Err(failure) => failed_attempts.push(FailedAttempt { plan_id: plan.id.clone(), failure }),
            }
        }
        Err(ExecutionError { attempts: failed_attempts })
    }
}

fn execute_plan(plan: &PreparedPlan, state: &mut ExecutionState) -> Result<(), ProviderFailure> {
    if !plan.workload.contains(&state.workload) {
        return Err(ProviderFailure::new(
            "executor",
            "workload_domain",
            "execution input is outside the qualified workload domain",
        ));
    }
    for step in &plan.steps {
        step.execute(state)?;
    }
    Ok(())
}

pub mod reference {
    use super::*;

    pub struct ReferenceProvider;

    impl Provider for ReferenceProvider {
        fn name(&self) -> &str {
            "reference"
        }

        fn prepare(&self, step: &ProviderStep) -> Result<Box<dyn PreparedStep>, ProviderFailure> {
            let prepared = match step.operation.as_str() {
                "identity" => ReferenceStep::Identity {
                    step: step.id.clone(),
                    input: exactly_one(&step.inputs, self.name(), &step.id, "input")?,
                    output: exactly_one(&step.outputs, self.name(), &step.id, "output")?,
                },
                "add_scalar" => {
                    let scalar = step
                        .config
                        .get("scalar")
                        .ok_or_else(|| ProviderFailure::new(self.name(), &step.id, "missing scalar config"))?
                        .parse::<f32>()
                        .map_err(|_| ProviderFailure::new(self.name(), &step.id, "scalar is not f32"))?;
                    ReferenceStep::AddScalar {
                        step: step.id.clone(),
                        input: exactly_one(&step.inputs, self.name(), &step.id, "input")?,
                        output: exactly_one(&step.outputs, self.name(), &step.id, "output")?,
                        scalar,
                    }
                }
                operation => {
                    return Err(ProviderFailure::new(
                        self.name(),
                        &step.id,
                        format!("unsupported operation {operation:?}"),
                    ));
                }
            };
            Ok(Box::new(prepared))
        }
    }

    fn exactly_one(values: &[String], provider: &str, step: &str, kind: &str) -> Result<String, ProviderFailure> {
        if let [value] = values {
            Ok(value.clone())
        } else {
            Err(ProviderFailure::new(provider, step, format!("expected exactly one {kind}")))
        }
    }

    enum ReferenceStep {
        Identity { step: String, input: String, output: String },
        AddScalar { step: String, input: String, output: String, scalar: f32 },
    }

    impl PreparedStep for ReferenceStep {
        fn execute(&self, state: &mut ExecutionState) -> Result<(), ProviderFailure> {
            match self {
                Self::Identity { step, input, output } => {
                    let value =
                        state.values.get(input).cloned().ok_or_else(|| {
                            ProviderFailure::new("reference", step, format!("missing input {input:?}"))
                        })?;
                    state.values.insert(output.clone(), value);
                }
                Self::AddScalar { step, input, output, scalar } => {
                    let Value::F32(values) =
                        state.values.get(input).cloned().ok_or_else(|| {
                            ProviderFailure::new("reference", step, format!("missing input {input:?}"))
                        })?
                    else {
                        return Err(ProviderFailure::new("reference", step, "input must be f32"));
                    };
                    state
                        .values
                        .insert(output.clone(), Value::F32(values.into_iter().map(|value| value + scalar).collect()));
                }
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use aletheia_contracts::{
        CaptureMode, HardwareFingerprint, HardwareTarget, InclusiveRangeU32, ModelIdentity, NumericalEvidence,
        PerformanceEvidence, Phase, PhysicalPlan, QualificationCertificate, QualificationStatus, ResilienceEvidence,
        ResourceEvidence, ResourceRequirements, SCHEMA_VERSION, WorkloadDomain,
    };

    use super::reference::ReferenceProvider;
    use super::*;

    fn hash(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn qualified(id: &str, provider: &str, operation: &str, fallbacks: Vec<String>) -> QualifiedPlan {
        let plan = PhysicalPlan {
            schema_version: SCHEMA_VERSION,
            id: id.into(),
            model: ModelIdentity { name: "toy".into(), semantics_sha256: hash('a') },
            hardware: HardwareTarget {
                accelerator: "cpu".into(),
                architecture: "reference".into(),
                device_count: 1,
                min_device_memory_bytes: 1,
            },
            workload: WorkloadDomain {
                phases: vec![Phase::Decode],
                batch: InclusiveRangeU32 { min: 1, max: 1 },
                rows_per_sequence: InclusiveRangeU32 { min: 1, max: 1 },
                context_tokens: InclusiveRangeU32 { min: 1, max: 1 },
            },
            resources: ResourceRequirements {
                peak_device_memory_bytes: 1,
                workspace_bytes: 0,
                stable_addresses: false,
                capture: CaptureMode::Eager,
            },
            steps: vec![ProviderStep {
                id: "step".into(),
                provider: provider.into(),
                operation: operation.into(),
                artifact_sha256: hash('b'),
                inputs: vec!["input".into()],
                outputs: vec!["output".into()],
                config: BTreeMap::new(),
            }],
            fallback_plan_ids: fallbacks,
            labels: BTreeMap::new(),
        };
        let certificate = QualificationCertificate {
            schema_version: SCHEMA_VERSION,
            plan_id: id.into(),
            plan_sha256: plan.digest().unwrap(),
            status: QualificationStatus::Qualified,
            hardware: HardwareFingerprint {
                accelerator: "cpu".into(),
                architecture: "reference".into(),
                device_count: 1,
                device_memory_bytes: 1,
                driver: "none".into(),
                runtime: "rust".into(),
            },
            workload: plan.workload.clone(),
            numerical: NumericalEvidence {
                oracle: "reference".into(),
                cases: 1,
                atol: 0.0,
                rtol: 0.0,
                max_abs_error: 0.0,
                min_output_agreement: 1.0,
            },
            performance: PerformanceEvidence {
                workload_sha256: hash('c'),
                samples: 1,
                p50_latency_micros: 1,
                p99_latency_micros: 1,
                goodput_tokens_per_second: 1.0,
            },
            resources: ResourceEvidence { peak_device_memory_bytes: 1, peak_workspace_bytes: 0 },
            resilience: ResilienceEvidence { soak_seconds: 0, passed_faults: Vec::new() },
            software: BTreeMap::new(),
        };
        QualifiedPlan { plan, certificate }
    }

    struct FailingProvider;
    struct FailingStep;

    impl Provider for FailingProvider {
        fn name(&self) -> &str {
            "failing"
        }

        fn prepare(&self, _step: &ProviderStep) -> Result<Box<dyn PreparedStep>, ProviderFailure> {
            Ok(Box::new(FailingStep))
        }
    }

    impl PreparedStep for FailingStep {
        fn execute(&self, state: &mut ExecutionState) -> Result<(), ProviderFailure> {
            state.values.insert("leaked".into(), Value::Bytes(vec![1]));
            Err(ProviderFailure::new("failing", "step", "injected failure"))
        }
    }

    #[test]
    fn fallback_restarts_from_clean_input() {
        let primary = qualified("fast", "failing", "identity", vec!["safe".into()]);
        let safe = qualified("safe", "reference", "identity", Vec::new());
        let selection = Selection { primary, fallbacks: vec![safe] };
        let mut executor = Executor::new();
        assert!(executor.register_provider(Arc::new(FailingProvider)));
        assert!(executor.register_provider(Arc::new(ReferenceProvider)));
        let prepared = executor.prepare(selection).unwrap();
        let mut input = ExecutionState::new(WorkloadPoint {
            phase: Phase::Decode,
            batch: 1,
            rows_per_sequence: 1,
            context_tokens: 1,
        });
        input.values.insert("input".into(), Value::F32(vec![1.0, 2.0]));

        let report = prepared.execute(&input).unwrap();
        assert_eq!(report.plan_id, "safe");
        assert_eq!(report.failed_attempts.len(), 1);
        assert_eq!(report.state.values.get("output"), Some(&Value::F32(vec![1.0, 2.0])));
        assert!(!report.state.values.contains_key("leaked"));
    }

    #[test]
    fn prepare_rejects_missing_provider() {
        let selection =
            Selection { primary: qualified("missing", "gpu", "identity", Vec::new()), fallbacks: Vec::new() };
        assert!(matches!(Executor::new().prepare(selection), Err(PrepareError::MissingProvider { .. })));
    }

    #[test]
    fn executor_revalidates_qualified_plan_digest() {
        let mut primary = qualified("tampered", "reference", "identity", Vec::new());
        primary.plan.steps[0].operation = "add_scalar".into();
        let selection = Selection { primary, fallbacks: Vec::new() };
        let mut executor = Executor::new();
        executor.register_provider(Arc::new(ReferenceProvider));
        assert!(matches!(executor.prepare(selection), Err(PrepareError::InvalidQualifiedPlan { .. })));
    }

    #[test]
    fn execution_rejects_workload_outside_qualified_domain() {
        let selection =
            Selection { primary: qualified("safe", "reference", "identity", Vec::new()), fallbacks: Vec::new() };
        let mut executor = Executor::new();
        executor.register_provider(Arc::new(ReferenceProvider));
        let prepared = executor.prepare(selection).unwrap();
        let state = ExecutionState::new(WorkloadPoint {
            phase: Phase::Decode,
            batch: 2,
            rows_per_sequence: 1,
            context_tokens: 1,
        });
        let error = prepared.execute(&state).unwrap_err();
        assert_eq!(error.attempts[0].failure.provider, "executor");
    }
}
