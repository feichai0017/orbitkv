use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{KvPlanSource, PlanError};

const RUNTIME_STATE_PLAN_SCHEMA: &str = "orbitkv.runtime-state-plan.v1";

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeExecutionMode {
    Policy,
    Owner,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeOwnerTransport {
    Ffi,
    Sidecar,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeUniformStatePlanMode {
    Execute,
    KernelReference,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimePrefixMode {
    CapsuleBackedSwaRadix,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimePrefixContract {
    pub mode: RuntimePrefixMode,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeExecutionContract {
    pub mode: RuntimeExecutionMode,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub owner_transport: Option<RuntimeOwnerTransport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uniform_state_plan_mode: Option<RuntimeUniformStatePlanMode>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeCapsuleContract {
    pub enabled: bool,
    pub chunk_tokens: u64,
    pub maximum_payload_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeStatePlan {
    pub schema: String,
    pub artifact_fingerprint: String,
    pub plan_fingerprint: String,
    pub semantic_source: KvPlanSource,
    pub layout: Value,
    pub sglang_policy: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub physical_plan: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uniform_state_plan: Option<Value>,
    pub execution: RuntimeExecutionContract,
    pub capsule: RuntimeCapsuleContract,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix: Option<RuntimePrefixContract>,
}

#[derive(Clone, Debug)]
pub struct RuntimeStatePlanOptions {
    pub eviction_interval_tokens: u64,
    pub physical_plan: Option<Value>,
    pub uniform_state_plan: Option<Value>,
    pub execution: RuntimeExecutionContract,
    pub capsule: RuntimeCapsuleContract,
    pub prefix: Option<RuntimePrefixContract>,
}

#[derive(Debug, Error)]
pub enum RuntimeStatePlanError {
    #[error("runtime StatePlan schema is unsupported")]
    UnsupportedSchema,
    #[error("runtime StatePlan fingerprint does not match its contents")]
    FingerprintMismatch,
    #[error("runtime StatePlan layout does not match its semantic source")]
    LayoutMismatch,
    #[error("runtime StatePlan SGLang policy is invalid")]
    InvalidSglangPolicy,
    #[error("runtime StatePlan physical plan is invalid")]
    InvalidPhysicalPlan,
    #[error("runtime StatePlan uniform-SWA plan is invalid")]
    InvalidUniformStatePlan,
    #[error("runtime StatePlan owner mode requires an owner transport")]
    MissingOwnerTransport,
    #[error("runtime StatePlan policy mode must not declare an owner transport")]
    UnexpectedOwnerTransport,
    #[error("runtime StatePlan uniform mode requires a uniform-SWA plan")]
    MissingUniformStatePlan,
    #[error("runtime StatePlan Capsule execution requires owner mode")]
    CapsuleRequiresOwner,
    #[error("runtime StatePlan Capsule limits must be positive")]
    InvalidCapsuleContract,
    #[error("runtime StatePlan Prefix contract requires owner Capsule execution")]
    InvalidPrefixContract,
    #[error("runtime StatePlan JSON encoding failed: {0}")]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Plan(#[from] PlanError),
}

/// Compiles all semantic and engine contracts consumed by the runtime into one
/// fingerprinted artifact.
///
/// Deployment resources such as the Capsule store root and FFI library path
/// remain external, but must bind to this artifact's fingerprint.
///
/// # Errors
///
/// Returns an error for incompatible nested artifacts or unsupported runtime
/// contracts.
pub fn compile_runtime_state_plan(
    semantic_source: KvPlanSource,
    options: RuntimeStatePlanOptions,
) -> Result<RuntimeStatePlan, RuntimeStatePlanError> {
    validate_execution(
        &options.execution,
        options.uniform_state_plan.is_some(),
        options.capsule.enabled,
    )?;
    validate_capsule(&options.capsule)?;
    let compiled = semantic_source.clone().compile()?;
    let plan_fingerprint = compiled.fingerprint();
    let layout = serde_json::to_value(compiled.layout_program()?)?;
    let policy = compiled.sglang_policy_with_eviction_interval(options.eviction_interval_tokens)?;
    let sglang_policy = serde_json::to_value(policy)?;
    validate_prefix(
        options.prefix.as_ref(),
        &options.execution,
        &options.capsule,
        &sglang_policy,
        options.uniform_state_plan.is_some(),
    )?;
    validate_physical_plan(
        options.physical_plan.as_ref(),
        &plan_fingerprint,
        &sglang_policy,
    )?;
    validate_uniform_state_plan(
        options.uniform_state_plan.as_ref(),
        &plan_fingerprint,
        options.execution.uniform_state_plan_mode,
    )?;
    let mut artifact = RuntimeStatePlan {
        schema: RUNTIME_STATE_PLAN_SCHEMA.into(),
        artifact_fingerprint: String::new(),
        plan_fingerprint,
        semantic_source,
        layout,
        sglang_policy,
        physical_plan: options.physical_plan,
        uniform_state_plan: options.uniform_state_plan,
        execution: options.execution,
        capsule: options.capsule,
        prefix: options.prefix,
    };
    artifact.artifact_fingerprint = artifact.compute_fingerprint()?;
    Ok(artifact)
}

impl RuntimeStatePlan {
    /// Validates every nested identity against the semantic source and the
    /// top-level artifact fingerprint.
    ///
    /// # Errors
    ///
    /// Returns an error for any mismatch or malformed nested contract.
    pub fn validate(&self) -> Result<(), RuntimeStatePlanError> {
        if self.schema != RUNTIME_STATE_PLAN_SCHEMA {
            return Err(RuntimeStatePlanError::UnsupportedSchema);
        }
        validate_execution(
            &self.execution,
            self.uniform_state_plan.is_some(),
            self.capsule.enabled,
        )?;
        validate_capsule(&self.capsule)?;
        let compiled = self.semantic_source.clone().compile()?;
        if compiled.fingerprint() != self.plan_fingerprint {
            return Err(RuntimeStatePlanError::LayoutMismatch);
        }
        if serde_json::to_value(compiled.layout_program()?)? != self.layout {
            return Err(RuntimeStatePlanError::LayoutMismatch);
        }
        validate_sglang_policy(&self.sglang_policy, &self.plan_fingerprint)?;
        validate_prefix(
            self.prefix.as_ref(),
            &self.execution,
            &self.capsule,
            &self.sglang_policy,
            self.uniform_state_plan.is_some(),
        )?;
        validate_physical_plan(
            self.physical_plan.as_ref(),
            &self.plan_fingerprint,
            &self.sglang_policy,
        )?;
        validate_uniform_state_plan(
            self.uniform_state_plan.as_ref(),
            &self.plan_fingerprint,
            self.execution.uniform_state_plan_mode,
        )?;
        if self.compute_fingerprint()? != self.artifact_fingerprint {
            return Err(RuntimeStatePlanError::FingerprintMismatch);
        }
        Ok(())
    }

    fn compute_fingerprint(&self) -> Result<String, RuntimeStatePlanError> {
        let mut payload = serde_json::json!({
            "schema": self.schema,
            "plan_fingerprint": self.plan_fingerprint,
            "semantic_source": self.semantic_source,
            "layout": self.layout,
            "sglang_policy": self.sglang_policy,
            "physical_plan": self.physical_plan,
            "uniform_state_plan": self.uniform_state_plan,
            "execution": self.execution,
            "capsule": self.capsule,
        });
        if let Some(prefix) = &self.prefix {
            payload
                .as_object_mut()
                .expect("runtime StatePlan fingerprint payload must be an object")
                .insert("prefix".into(), serde_json::to_value(prefix)?);
        }
        let bytes = serde_json::to_vec(&payload)?;
        Ok(format!("sha256:{:x}", Sha256::digest(bytes)))
    }
}

fn validate_execution(
    execution: &RuntimeExecutionContract,
    has_uniform_plan: bool,
    capsule_enabled: bool,
) -> Result<(), RuntimeStatePlanError> {
    match (execution.mode, execution.owner_transport) {
        (RuntimeExecutionMode::Owner, None) => {
            return Err(RuntimeStatePlanError::MissingOwnerTransport);
        }
        (RuntimeExecutionMode::Policy, Some(_)) => {
            return Err(RuntimeStatePlanError::UnexpectedOwnerTransport);
        }
        _ => {}
    }
    if execution.uniform_state_plan_mode.is_some() && !has_uniform_plan {
        return Err(RuntimeStatePlanError::MissingUniformStatePlan);
    }
    if capsule_enabled && execution.mode != RuntimeExecutionMode::Owner {
        return Err(RuntimeStatePlanError::CapsuleRequiresOwner);
    }
    Ok(())
}

fn validate_capsule(capsule: &RuntimeCapsuleContract) -> Result<(), RuntimeStatePlanError> {
    if capsule.chunk_tokens == 0 || capsule.maximum_payload_bytes == 0 {
        return Err(RuntimeStatePlanError::InvalidCapsuleContract);
    }
    Ok(())
}

fn validate_prefix(
    prefix: Option<&RuntimePrefixContract>,
    execution: &RuntimeExecutionContract,
    capsule: &RuntimeCapsuleContract,
    policy: &Value,
    has_uniform_plan: bool,
) -> Result<(), RuntimeStatePlanError> {
    if prefix.is_some() {
        let bounded = policy
            .get("bounded_classes")
            .and_then(Value::as_array)
            .ok_or(RuntimeStatePlanError::InvalidPrefixContract)?;
        let unbounded = policy
            .get("unbounded_classes")
            .and_then(Value::as_array)
            .ok_or(RuntimeStatePlanError::InvalidPrefixContract)?;
        if !capsule.enabled
            || execution.mode != RuntimeExecutionMode::Owner
            || execution.owner_transport != Some(RuntimeOwnerTransport::Sidecar)
            || has_uniform_plan
            || bounded.len() != 1
            || unbounded.len() != 1
        {
            return Err(RuntimeStatePlanError::InvalidPrefixContract);
        }
    }
    Ok(())
}

fn validate_sglang_policy(
    policy: &Value,
    plan_fingerprint: &str,
) -> Result<(), RuntimeStatePlanError> {
    if policy.get("schema").and_then(Value::as_str) != Some("orbitkv.sglang-policy.v1")
        || policy.get("plan_fingerprint").and_then(Value::as_str) != Some(plan_fingerprint)
    {
        return Err(RuntimeStatePlanError::InvalidSglangPolicy);
    }
    Ok(())
}

fn validate_physical_plan(
    wrapper: Option<&Value>,
    plan_fingerprint: &str,
    policy: &Value,
) -> Result<(), RuntimeStatePlanError> {
    let Some(wrapper) = wrapper else {
        return Ok(());
    };
    let physical = wrapper
        .get("physical_plan")
        .ok_or(RuntimeStatePlanError::InvalidPhysicalPlan)?;
    if wrapper.get("schema").and_then(Value::as_str) != Some("orbitkv.hf-physical-compilation.v1")
        || physical.get("schema").and_then(Value::as_str) != Some("orbitkv.sglang-physical-plan.v1")
        || physical.get("plan_fingerprint").and_then(Value::as_str) != Some(plan_fingerprint)
        || physical.pointer("/selected/policy") != Some(policy)
    {
        return Err(RuntimeStatePlanError::InvalidPhysicalPlan);
    }
    Ok(())
}

fn validate_uniform_state_plan(
    plan: Option<&Value>,
    plan_fingerprint: &str,
    mode: Option<RuntimeUniformStatePlanMode>,
) -> Result<(), RuntimeStatePlanError> {
    let Some(plan) = plan else {
        return if mode.is_none() {
            Ok(())
        } else {
            Err(RuntimeStatePlanError::MissingUniformStatePlan)
        };
    };
    if plan.get("schema").and_then(Value::as_str) != Some("orbitkv.hf-state-plan.v4")
        || plan
            .pointer("/layout/plan_fingerprint")
            .and_then(Value::as_str)
            != Some(plan_fingerprint)
    {
        return Err(RuntimeStatePlanError::InvalidUniformStatePlan);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::{KvClassSpec, KvPlanInput, RetentionKind};

    use super::*;

    fn source() -> KvPlanSource {
        KvPlanSource::Legacy(KvPlanInput {
            page_tokens: 16,
            classes: vec![
                KvClassSpec {
                    name: "full".into(),
                    layers: vec![1],
                    retention: RetentionKind::Full,
                    bytes_per_token_per_layer: 512,
                    window_tokens: None,
                },
                KvClassSpec {
                    name: "swa".into(),
                    layers: vec![0],
                    retention: RetentionKind::Sliding,
                    bytes_per_token_per_layer: 512,
                    window_tokens: Some(128),
                },
            ],
        })
    }

    #[test]
    fn runtime_state_plan_round_trips_and_detects_tampering() {
        let artifact = compile_runtime_state_plan(
            source(),
            RuntimeStatePlanOptions {
                eviction_interval_tokens: 32,
                physical_plan: None,
                uniform_state_plan: None,
                execution: RuntimeExecutionContract {
                    mode: RuntimeExecutionMode::Owner,
                    owner_transport: Some(RuntimeOwnerTransport::Ffi),
                    uniform_state_plan_mode: None,
                },
                capsule: RuntimeCapsuleContract {
                    enabled: true,
                    chunk_tokens: 128,
                    maximum_payload_bytes: 1 << 30,
                },
                prefix: None,
            },
        )
        .unwrap();
        artifact.validate().unwrap();
        let encoded = serde_json::to_vec(&artifact).unwrap();
        let mut decoded: RuntimeStatePlan = serde_json::from_slice(&encoded).unwrap();
        decoded.validate().unwrap();
        assert!(decoded.prefix.is_none());
        decoded.capsule.chunk_tokens = 256;
        assert!(matches!(
            decoded.validate(),
            Err(RuntimeStatePlanError::FingerprintMismatch)
        ));
    }

    #[test]
    fn capsule_contract_requires_owner_execution() {
        let error = compile_runtime_state_plan(
            source(),
            RuntimeStatePlanOptions {
                eviction_interval_tokens: 16,
                physical_plan: None,
                uniform_state_plan: None,
                execution: RuntimeExecutionContract {
                    mode: RuntimeExecutionMode::Policy,
                    owner_transport: None,
                    uniform_state_plan_mode: None,
                },
                capsule: RuntimeCapsuleContract {
                    enabled: true,
                    chunk_tokens: 16,
                    maximum_payload_bytes: 1 << 20,
                },
                prefix: None,
            },
        )
        .unwrap_err();
        assert!(matches!(error, RuntimeStatePlanError::CapsuleRequiresOwner));
    }

    #[test]
    fn radix_prefix_contract_requires_sidecar_and_is_fingerprinted() {
        let error = compile_runtime_state_plan(
            source(),
            RuntimeStatePlanOptions {
                eviction_interval_tokens: 16,
                physical_plan: None,
                uniform_state_plan: None,
                execution: RuntimeExecutionContract {
                    mode: RuntimeExecutionMode::Owner,
                    owner_transport: Some(RuntimeOwnerTransport::Ffi),
                    uniform_state_plan_mode: None,
                },
                capsule: RuntimeCapsuleContract {
                    enabled: true,
                    chunk_tokens: 16,
                    maximum_payload_bytes: 1 << 20,
                },
                prefix: Some(RuntimePrefixContract {
                    mode: RuntimePrefixMode::CapsuleBackedSwaRadix,
                }),
            },
        )
        .unwrap_err();
        assert!(matches!(
            error,
            RuntimeStatePlanError::InvalidPrefixContract
        ));

        let artifact = compile_runtime_state_plan(
            source(),
            RuntimeStatePlanOptions {
                eviction_interval_tokens: 16,
                physical_plan: None,
                uniform_state_plan: None,
                execution: RuntimeExecutionContract {
                    mode: RuntimeExecutionMode::Owner,
                    owner_transport: Some(RuntimeOwnerTransport::Sidecar),
                    uniform_state_plan_mode: None,
                },
                capsule: RuntimeCapsuleContract {
                    enabled: true,
                    chunk_tokens: 16,
                    maximum_payload_bytes: 1 << 20,
                },
                prefix: Some(RuntimePrefixContract {
                    mode: RuntimePrefixMode::CapsuleBackedSwaRadix,
                }),
            },
        )
        .unwrap();
        artifact.validate().unwrap();
        assert_eq!(
            artifact.prefix,
            Some(RuntimePrefixContract {
                mode: RuntimePrefixMode::CapsuleBackedSwaRadix
            })
        );
    }
}
