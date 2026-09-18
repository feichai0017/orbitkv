//! Portable Aletheia contracts shared by candidate generators, qualification workers,
//! registries, executors, and serving integrations. This crate deliberately has
//! no dependency on a device or serving framework.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use thiserror::Error;

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, Error, PartialEq)]
pub enum ContractError {
    #[error("{path}: {message}")]
    Invalid { path: &'static str, message: String },
    #[error("failed to serialize a canonical contract: {0}")]
    Serialization(String),
}

fn invalid(path: &'static str, message: impl Into<String>) -> ContractError {
    ContractError::Invalid { path, message: message.into() }
}

fn non_empty(path: &'static str, value: &str) -> Result<(), ContractError> {
    if value.trim().is_empty() { Err(invalid(path, "must not be empty")) } else { Ok(()) }
}

fn identifier(path: &'static str, value: &str) -> Result<(), ContractError> {
    non_empty(path, value)?;
    if value.len() > 128
        || !value.bytes().all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
        || !value.as_bytes()[0].is_ascii_alphanumeric()
        || value == "."
        || value == ".."
    {
        return Err(invalid(path, "must be a portable identifier of at most 128 characters"));
    }
    Ok(())
}

fn digest(path: &'static str, value: &str) -> Result<(), ContractError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()) {
        return Err(invalid(path, "must be a 64-character lowercase hexadecimal SHA-256 digest"));
    }
    Ok(())
}

pub fn sha256<T: Serialize>(value: &T) -> Result<String, ContractError> {
    let bytes = serde_json::to_vec(value).map_err(|error| ContractError::Serialization(error.to_string()))?;
    Ok(hex::encode(Sha256::digest(bytes)))
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelIdentity {
    pub name: String,
    pub semantics_sha256: String,
}

impl ModelIdentity {
    pub fn validate(&self) -> Result<(), ContractError> {
        non_empty("model.name", &self.name)?;
        digest("model.semantics_sha256", &self.semantics_sha256)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HardwareTarget {
    pub accelerator: String,
    pub architecture: String,
    pub device_count: u16,
    pub min_device_memory_bytes: u64,
}

impl HardwareTarget {
    pub fn validate(&self) -> Result<(), ContractError> {
        non_empty("hardware.accelerator", &self.accelerator)?;
        non_empty("hardware.architecture", &self.architecture)?;
        if self.device_count == 0 {
            return Err(invalid("hardware.device_count", "must be positive"));
        }
        Ok(())
    }

    pub fn supports(&self, hardware: &HardwareFingerprint) -> bool {
        self.accelerator == hardware.accelerator
            && self.architecture == hardware.architecture
            && self.device_count == hardware.device_count
            && self.min_device_memory_bytes <= hardware.device_memory_bytes
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HardwareFingerprint {
    pub accelerator: String,
    pub architecture: String,
    pub device_count: u16,
    pub device_memory_bytes: u64,
    pub driver: String,
    pub runtime: String,
}

impl HardwareFingerprint {
    pub fn validate(&self) -> Result<(), ContractError> {
        non_empty("hardware.accelerator", &self.accelerator)?;
        non_empty("hardware.architecture", &self.architecture)?;
        non_empty("hardware.driver", &self.driver)?;
        non_empty("hardware.runtime", &self.runtime)?;
        if self.device_count == 0 || self.device_memory_bytes == 0 {
            return Err(invalid("hardware", "device count and memory must be positive"));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Prefill,
    Decode,
    Verify,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InclusiveRangeU32 {
    pub min: u32,
    pub max: u32,
}

impl InclusiveRangeU32 {
    pub fn validate(&self, path: &'static str, allow_zero: bool) -> Result<(), ContractError> {
        if (!allow_zero && self.min == 0) || self.min > self.max {
            return Err(invalid(
                path,
                if allow_zero { "must be a non-empty inclusive range" } else { "must be a positive inclusive range" },
            ));
        }
        Ok(())
    }

    pub fn contains(&self, value: u32) -> bool {
        (self.min..=self.max).contains(&value)
    }

    pub fn contains_range(&self, other: Self) -> bool {
        self.min <= other.min && other.max <= self.max
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkloadDomain {
    pub phases: Vec<Phase>,
    pub batch: InclusiveRangeU32,
    pub rows_per_sequence: InclusiveRangeU32,
    pub context_tokens: InclusiveRangeU32,
}

impl WorkloadDomain {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.phases.is_empty() {
            return Err(invalid("workload.phases", "must not be empty"));
        }
        let mut phases = self.phases.clone();
        phases.sort_unstable();
        phases.dedup();
        if phases.len() != self.phases.len() {
            return Err(invalid("workload.phases", "must not contain duplicates"));
        }
        self.batch.validate("workload.batch", false)?;
        self.rows_per_sequence.validate("workload.rows_per_sequence", false)?;
        self.context_tokens.validate("workload.context_tokens", true)
    }

    pub fn contains(&self, point: &WorkloadPoint) -> bool {
        self.phases.contains(&point.phase)
            && self.batch.contains(point.batch)
            && self.rows_per_sequence.contains(point.rows_per_sequence)
            && self.context_tokens.contains(point.context_tokens)
    }

    pub fn contains_domain(&self, other: &Self) -> bool {
        other.phases.iter().all(|phase| self.phases.contains(phase))
            && self.batch.contains_range(other.batch)
            && self.rows_per_sequence.contains_range(other.rows_per_sequence)
            && self.context_tokens.contains_range(other.context_tokens)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkloadPoint {
    pub phase: Phase,
    pub batch: u32,
    pub rows_per_sequence: u32,
    pub context_tokens: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureMode {
    Eager,
    CudaGraph,
    Persistent,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceRequirements {
    pub peak_device_memory_bytes: u64,
    pub workspace_bytes: u64,
    pub stable_addresses: bool,
    pub capture: CaptureMode,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderStep {
    pub id: String,
    pub provider: String,
    pub operation: String,
    pub artifact_sha256: String,
    #[serde(default)]
    pub inputs: Vec<String>,
    #[serde(default)]
    pub outputs: Vec<String>,
    #[serde(default)]
    pub config: BTreeMap<String, String>,
}

impl ProviderStep {
    fn validate(&self) -> Result<(), ContractError> {
        non_empty("steps.id", &self.id)?;
        non_empty("steps.provider", &self.provider)?;
        non_empty("steps.operation", &self.operation)?;
        digest("steps.artifact_sha256", &self.artifact_sha256)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhysicalPlan {
    pub schema_version: u32,
    pub id: String,
    pub model: ModelIdentity,
    pub hardware: HardwareTarget,
    pub workload: WorkloadDomain,
    pub resources: ResourceRequirements,
    pub steps: Vec<ProviderStep>,
    #[serde(default)]
    pub fallback_plan_ids: Vec<String>,
    #[serde(default)]
    pub labels: BTreeMap<String, String>,
}

impl PhysicalPlan {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(invalid("schema_version", format!("expected {SCHEMA_VERSION}")));
        }
        identifier("id", &self.id)?;
        self.model.validate()?;
        self.hardware.validate()?;
        self.workload.validate()?;
        if self.steps.is_empty() {
            return Err(invalid("steps", "must not be empty"));
        }
        let mut step_ids = Vec::with_capacity(self.steps.len());
        for step in &self.steps {
            step.validate()?;
            step_ids.push(step.id.as_str());
        }
        step_ids.sort_unstable();
        if step_ids.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(invalid("steps.id", "must be unique within a plan"));
        }
        if self.fallback_plan_ids.iter().any(|fallback| fallback == &self.id) {
            return Err(invalid("fallback_plan_ids", "a plan cannot fall back to itself"));
        }
        for fallback in &self.fallback_plan_ids {
            identifier("fallback_plan_ids", fallback)?;
        }
        let mut fallbacks = self.fallback_plan_ids.clone();
        fallbacks.sort();
        fallbacks.dedup();
        if fallbacks.len() != self.fallback_plan_ids.len() {
            return Err(invalid("fallback_plan_ids", "must not contain duplicates"));
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String, ContractError> {
        sha256(self)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericalEvidence {
    pub oracle: String,
    pub cases: u64,
    pub atol: f64,
    pub rtol: f64,
    pub max_abs_error: f64,
    /// Fraction of outputs that must satisfy the declared numerical envelope.
    /// At model scope this may be top-1/token agreement; at operator scope it
    /// is the fraction of tensor elements within tolerance.
    pub min_output_agreement: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerformanceEvidence {
    pub workload_sha256: String,
    pub samples: u64,
    pub p50_latency_micros: u64,
    pub p99_latency_micros: u64,
    pub goodput_tokens_per_second: f64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceEvidence {
    pub peak_device_memory_bytes: u64,
    pub peak_workspace_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResilienceEvidence {
    pub soak_seconds: u64,
    #[serde(default)]
    pub passed_faults: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QualificationStatus {
    Candidate,
    Qualified,
    Revoked,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualificationCertificate {
    pub schema_version: u32,
    pub plan_id: String,
    pub plan_sha256: String,
    pub status: QualificationStatus,
    pub hardware: HardwareFingerprint,
    pub workload: WorkloadDomain,
    pub numerical: NumericalEvidence,
    pub performance: PerformanceEvidence,
    pub resources: ResourceEvidence,
    pub resilience: ResilienceEvidence,
    #[serde(default)]
    pub software: BTreeMap<String, String>,
}

impl QualificationCertificate {
    pub fn validate_for(&self, plan: &PhysicalPlan) -> Result<(), ContractError> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(invalid("certificate.schema_version", format!("expected {SCHEMA_VERSION}")));
        }
        if self.status != QualificationStatus::Qualified {
            return Err(invalid("certificate.status", "must be qualified"));
        }
        if self.plan_id != plan.id {
            return Err(invalid("certificate.plan_id", "does not name the supplied plan"));
        }
        digest("certificate.plan_sha256", &self.plan_sha256)?;
        if self.plan_sha256 != plan.digest()? {
            return Err(invalid("certificate.plan_sha256", "does not match the supplied plan"));
        }
        self.hardware.validate()?;
        if !plan.hardware.supports(&self.hardware) {
            return Err(invalid("certificate.hardware", "is outside the plan hardware target"));
        }
        self.workload.validate()?;
        if !plan.workload.contains_domain(&self.workload) {
            return Err(invalid("certificate.workload", "is outside the plan workload domain"));
        }
        non_empty("certificate.numerical.oracle", &self.numerical.oracle)?;
        if self.numerical.cases == 0
            || !self.numerical.atol.is_finite()
            || self.numerical.atol < 0.0
            || !self.numerical.rtol.is_finite()
            || self.numerical.rtol < 0.0
            || !self.numerical.max_abs_error.is_finite()
            || self.numerical.max_abs_error < 0.0
            || !(0.0..=1.0).contains(&self.numerical.min_output_agreement)
        {
            return Err(invalid("certificate.numerical", "contains invalid or empty evidence"));
        }
        digest("certificate.performance.workload_sha256", &self.performance.workload_sha256)?;
        if self.performance.samples == 0
            || self.performance.p50_latency_micros > self.performance.p99_latency_micros
            || !self.performance.goodput_tokens_per_second.is_finite()
            || self.performance.goodput_tokens_per_second <= 0.0
        {
            return Err(invalid("certificate.performance", "contains invalid or empty evidence"));
        }
        if self.resources.peak_device_memory_bytes > plan.resources.peak_device_memory_bytes
            || self.resources.peak_workspace_bytes > plan.resources.workspace_bytes
        {
            return Err(invalid("certificate.resources", "observed usage exceeds the plan declaration"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServiceLevelObjective {
    pub max_p99_latency_micros: u64,
    pub min_goodput_tokens_per_second: f64,
    pub max_device_memory_bytes: u64,
    pub max_abs_error: f64,
    pub min_output_agreement: f64,
}

impl ServiceLevelObjective {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.max_p99_latency_micros == 0
            || !self.min_goodput_tokens_per_second.is_finite()
            || self.min_goodput_tokens_per_second < 0.0
            || self.max_device_memory_bytes == 0
            || !self.max_abs_error.is_finite()
            || self.max_abs_error < 0.0
            || !(0.0..=1.0).contains(&self.min_output_agreement)
        {
            return Err(invalid("slo", "contains an invalid bound"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TraceEvent {
    pub schema_version: u32,
    pub timestamp_micros: u64,
    pub request_id: String,
    pub model_semantics_sha256: String,
    pub workload: WorkloadPoint,
    pub queue_micros: u64,
    pub execute_micros: u64,
    pub plan_id: String,
    pub outcome: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn plan() -> PhysicalPlan {
        PhysicalPlan {
            schema_version: SCHEMA_VERSION,
            id: "baseline".into(),
            model: ModelIdentity { name: "toy".into(), semantics_sha256: hash('a') },
            hardware: HardwareTarget {
                accelerator: "cpu".into(),
                architecture: "reference".into(),
                device_count: 1,
                min_device_memory_bytes: 1,
            },
            workload: WorkloadDomain {
                phases: vec![Phase::Decode],
                batch: InclusiveRangeU32 { min: 1, max: 8 },
                rows_per_sequence: InclusiveRangeU32 { min: 1, max: 1 },
                context_tokens: InclusiveRangeU32 { min: 0, max: 4096 },
            },
            resources: ResourceRequirements {
                peak_device_memory_bytes: 1024,
                workspace_bytes: 128,
                stable_addresses: false,
                capture: CaptureMode::Eager,
            },
            steps: vec![ProviderStep {
                id: "forward".into(),
                provider: "reference".into(),
                operation: "identity".into(),
                artifact_sha256: hash('b'),
                inputs: vec!["input".into()],
                outputs: vec!["output".into()],
                config: BTreeMap::new(),
            }],
            fallback_plan_ids: Vec::new(),
            labels: BTreeMap::new(),
        }
    }

    #[test]
    fn plan_validation_rejects_duplicate_steps() {
        let mut plan = plan();
        plan.steps.push(plan.steps[0].clone());
        assert!(plan.validate().unwrap_err().to_string().contains("unique"));
    }

    #[test]
    fn plan_id_is_safe_as_a_registry_directory() {
        let mut plan = plan();
        plan.id = "../../escape".into();
        assert!(plan.validate().unwrap_err().to_string().contains("portable identifier"));
    }

    #[test]
    fn plan_digest_changes_with_execution_contract() {
        let first = plan();
        let mut second = first.clone();
        second.steps[0].operation = "add_scalar".into();
        assert_ne!(first.digest().unwrap(), second.digest().unwrap());
    }

    #[test]
    fn workload_domain_checks_every_axis() {
        let domain = plan().workload;
        assert!(domain.contains(&WorkloadPoint {
            phase: Phase::Decode,
            batch: 4,
            rows_per_sequence: 1,
            context_tokens: 1024,
        }));
        assert!(domain.contains(&WorkloadPoint {
            phase: Phase::Decode,
            batch: 1,
            rows_per_sequence: 1,
            context_tokens: 0,
        }));
        assert!(!domain.contains(&WorkloadPoint {
            phase: Phase::Prefill,
            batch: 4,
            rows_per_sequence: 1,
            context_tokens: 1024,
        }));
    }
}
