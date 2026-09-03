use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::attention_state::{
    AttentionStateBackend, AttentionStateError, AttentionStatePlanInput, AttentionStateSpec,
    AttentionStateStorage, CompiledAttentionState, RecurrentFamily, compile_attention_state_plan,
};
use crate::plan::{
    AddressProgram, BlockDomain, PlanError, RetentionKind, RetirementProgram, TokenStorageKind,
    compile_plan,
};
use crate::retention::{InferredRetention, KvHeadRange, RetentionError, analyze_state};
use crate::runtime_manifest::{
    RUNTIME_MANIFEST_SCHEMA, RUNTIME_MANIFEST_VERSION, RuntimeCapability, RuntimeManifest,
    RuntimeManifestError, RuntimeManifestSource, is_sha256_fingerprint, write_canonical_json,
};

const RUNTIME_TARGET_SCHEMA: &str = "orbitkv.runtime-target";
const RUNTIME_TARGET_VERSION: u32 = 1;
pub const EXECUTION_SIGNATURE_SCHEMA: &str = "orbitkv.execution-signature";
pub const EXECUTION_SIGNATURE_VERSION: u32 = 1;
pub const RUNTIME_BINDING_SCHEMA: &str = "orbitkv.runtime-binding";
pub const RUNTIME_BINDING_VERSION: u32 = 1;
const RUNTIME_TARGET_ARTIFACT_MAX_BYTES: usize = 16 * 1024 * 1024;

const SGLANG_TARGET_ID: &str = "sglang";
const SGLANG_TARGET_CONTRACT_VERSION: u32 = 4;
const SGLANG_ADMISSION_PROFILE_ID: &str = "eager-single-device-bf16-nhd";
const SGLANG_ADMISSION_PROFILE_VERSION: u32 = 1;
const REQUIRED_WIRE_VERSION: u32 = 14;

/// Stable identity of one executor contract. The identifier names the target,
/// while `contract_version` changes whenever its runtime contract changes.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeTargetIdentity {
    pub id: String,
    pub contract_version: u32,
}

/// Stable identity of the structural admission algorithm used by a target.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeAdmissionProfile {
    pub id: String,
    pub version: u32,
}

/// Closed topology vocabulary understood by the admission implementation.
///
/// These are structural shapes, not model or hardware names. Each variant is
/// recognized from the complete compiled layout and state geometry.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionTopology {
    WholeDomainChunkedTokenKv,
    WholeDomainFullLatentKv,
    WholeDomainFullSlidingTokenKv,
    WholeDomainFullTokenKv,
    WholeDomainFullTokenKvGdnConvolution,
    WholeDomainFullTokenKvMamba,
    WholeDomainSlidingTokenKv,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct RuntimeTarget {
    schema: String,
    version: u32,
    fingerprint: String,
    target: RuntimeTargetIdentity,
    admission_profile: RuntimeAdmissionProfile,
    page_tokens: u64,
    supported_manifest_versions: Vec<u32>,
    required_wire_version: u32,
    supported_capabilities: Vec<RuntimeCapability>,
    supported_topologies: Vec<ExecutionTopology>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionKvHeadRange {
    pub start: u32,
    pub end_exclusive: u32,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionBlockDomain {
    pub start_block: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_block_exclusive: Option<u64>,
}

impl ExecutionBlockDomain {
    const fn is_all(&self) -> bool {
        self.start_block == 0 && self.end_block_exclusive.is_none()
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExecutionAddressProgram {
    AppendOnly,
    Pinned,
    Periodic {
        period_blocks: u64,
    },
    PeriodicFrom {
        period_blocks: u64,
        origin_block: u64,
    },
    ResettableArena {
        blocks_per_epoch: u64,
    },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExecutionRetirementProgram {
    Never,
    BlockEndPlus { offset_tokens: u64 },
    EpochEnd { blocks_per_epoch: u64 },
}

/// Complete physical class shape copied from a validated layout program.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionTokenClass {
    pub name: String,
    pub layers: Vec<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_head_range: Option<ExecutionKvHeadRange>,
    pub bytes_per_token_per_layer: u64,
    pub address: ExecutionAddressProgram,
    pub retirement: ExecutionRetirementProgram,
    pub minimum_slots_per_request: Option<u64>,
    #[serde(default, skip_serializing_if = "ExecutionBlockDomain::is_all")]
    pub block_domain: ExecutionBlockDomain,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionStateComponent {
    pub name: String,
    pub bytes_per_token_per_layer: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExecutionTokenBackend {
    TokenSlots {
        storage: TokenStorageKind,
        components: Vec<ExecutionStateComponent>,
        bytes_per_token_per_layer: u64,
        page_bytes_per_layer: u64,
        retention: RetentionKind,
        window_tokens: Option<u64>,
        token_relocatable: bool,
    },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionTokenState {
    pub name: String,
    pub layers: Vec<u32>,
    pub backend: ExecutionTokenBackend,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExecutionFixedBackend {
    RecurrentCheckpoints {
        family: RecurrentFamily,
        state_bytes_per_layer: u64,
        checkpoint_slots_per_request: u32,
        checkpoint_bytes_per_request: u64,
        token_relocatable: bool,
    },
    ConvolutionRing {
        state_bytes_per_layer: u64,
        kernel_width: u32,
        checkpoint_slots_per_request: u32,
        checkpoint_bytes_per_request: u64,
        token_relocatable: bool,
    },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionFixedState {
    pub name: String,
    pub layers: Vec<u32>,
    pub backend: ExecutionFixedBackend,
}

/// Complete structural projection used for target admission.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionSignature {
    pub schema: String,
    pub version: u32,
    pub fingerprint: String,
    pub manifest_schema: String,
    pub manifest_version: u32,
    pub manifest_fingerprint: String,
    pub page_tokens: u64,
    pub token_classes: Vec<ExecutionTokenClass>,
    pub token_states: Vec<ExecutionTokenState>,
    pub fixed_states: Vec<ExecutionFixedState>,
}

/// Proof that one exact manifest was structurally admitted by one exact target
/// contract. Dynamic device and engine checks remain the adapter's duty.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeBinding {
    pub schema: String,
    pub version: u32,
    pub fingerprint: String,
    pub manifest_fingerprint: String,
    pub target: RuntimeTargetIdentity,
    pub admission_profile: RuntimeAdmissionProfile,
    pub target_contract_fingerprint: String,
    pub required_wire_version: u32,
    pub execution_topology: ExecutionTopology,
    pub execution_signature: ExecutionSignature,
}

#[derive(Debug, Error)]
pub enum TargetAdmissionError {
    #[error("{artifact} schema {actual:?} is unsupported")]
    UnsupportedSchema {
        artifact: &'static str,
        actual: String,
    },
    #[error("{artifact} version {actual} is unsupported")]
    UnsupportedVersion { artifact: &'static str, actual: u32 },
    #[error("{artifact} fingerprint is malformed")]
    MalformedFingerprint { artifact: &'static str },
    #[error("{artifact} fingerprint does not match its contents")]
    FingerprintMismatch { artifact: &'static str },
    #[error("{artifact} JSON is not in its canonical typed shape")]
    NonCanonicalJson { artifact: &'static str },
    #[error("{artifact} is {actual} bytes, above the {maximum}-byte limit")]
    ArtifactTooLarge {
        artifact: &'static str,
        actual: usize,
        maximum: usize,
    },
    #[error("target contract {field} must be a nonempty stable identifier")]
    InvalidIdentifier { field: &'static str },
    #[error("target contract versions and page_tokens must be positive")]
    InvalidTargetContractGeometry,
    #[error("target contract lists must be nonempty, sorted, and unique")]
    NonCanonicalTargetLists,
    #[error(
        "runtime target contract {target}@{target_contract_version} with admission profile {admission_profile}@{admission_profile_version} does not support manifest version {manifest_version}"
    )]
    UnsupportedTargetManifestVersion {
        target: String,
        target_contract_version: u32,
        admission_profile: String,
        admission_profile_version: u32,
        manifest_version: u32,
    },
    #[error("target does not support manifest capability {0:?}")]
    UnsupportedTargetCapability(RuntimeCapability),
    #[error("execution signature is not a compiler-derived structural projection: {0}")]
    InvalidExecutionSignature(&'static str),
    #[error("target requires page_tokens={target}, but the manifest uses {manifest}")]
    PageTokensMismatch { target: u64, manifest: u64 },
    #[error("manifest topology is not one of the closed execution shapes")]
    UnsupportedManifestTopology,
    #[error("target does not support manifest topology {0:?}")]
    UnsupportedTargetTopology(ExecutionTopology),
    #[error("runtime target binding does not match its embedded identities")]
    BindingIdentityMismatch,
    #[error("runtime target binding does not match the supplied manifest and contract")]
    BindingMismatch,
    #[error(transparent)]
    Manifest(#[from] RuntimeManifestError),
    #[error(transparent)]
    AttentionState(#[from] AttentionStateError),
    #[error(transparent)]
    Plan(#[from] PlanError),
    #[error(transparent)]
    Retention(#[from] RetentionError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

impl RuntimeTarget {
    fn new(
        target: RuntimeTargetIdentity,
        admission_profile: RuntimeAdmissionProfile,
        page_tokens: u64,
        supported_manifest_versions: Vec<u32>,
        required_wire_version: u32,
        supported_capabilities: Vec<RuntimeCapability>,
        supported_topologies: Vec<ExecutionTopology>,
    ) -> Result<Self, TargetAdmissionError> {
        let mut contract = Self {
            schema: RUNTIME_TARGET_SCHEMA.to_owned(),
            version: RUNTIME_TARGET_VERSION,
            fingerprint: String::new(),
            target,
            admission_profile,
            page_tokens,
            supported_manifest_versions,
            required_wire_version,
            supported_capabilities,
            supported_topologies,
        };
        contract.fingerprint = computed_fingerprint(&contract)?;
        contract.validate()?;
        Ok(contract)
    }

    fn packaged_sglang() -> Result<Self, TargetAdmissionError> {
        Self::new(
            RuntimeTargetIdentity {
                id: SGLANG_TARGET_ID.into(),
                contract_version: SGLANG_TARGET_CONTRACT_VERSION,
            },
            RuntimeAdmissionProfile {
                id: SGLANG_ADMISSION_PROFILE_ID.into(),
                version: SGLANG_ADMISSION_PROFILE_VERSION,
            },
            16,
            vec![RUNTIME_MANIFEST_VERSION],
            REQUIRED_WIRE_VERSION,
            vec![
                RuntimeCapability::AppendOnlyAddressing,
                RuntimeCapability::BlockDomainPartitioning,
                RuntimeCapability::ConvolutionState,
                RuntimeCapability::FixedStateCheckpoints,
                RuntimeCapability::KvHeadPartitioning,
                RuntimeCapability::PeriodicAddressing,
                RuntimeCapability::PeriodicFromAddressing,
                RuntimeCapability::PinnedAddressing,
                RuntimeCapability::RecurrentState,
                RuntimeCapability::ResettableArenaAddressing,
                RuntimeCapability::SemanticRetirement,
                RuntimeCapability::TokenComponentGeometry,
                RuntimeCapability::TokenManager,
            ],
            vec![
                ExecutionTopology::WholeDomainChunkedTokenKv,
                ExecutionTopology::WholeDomainFullLatentKv,
                ExecutionTopology::WholeDomainFullSlidingTokenKv,
                ExecutionTopology::WholeDomainFullTokenKv,
                ExecutionTopology::WholeDomainFullTokenKvGdnConvolution,
                ExecutionTopology::WholeDomainFullTokenKvMamba,
                ExecutionTopology::WholeDomainSlidingTokenKv,
            ],
        )
    }

    /// Computes the canonical content fingerprint excluding `fingerprint`.
    ///
    /// # Errors
    ///
    /// Returns an error if the typed contract cannot be serialized.
    #[cfg(test)]
    fn computed_fingerprint(&self) -> Result<String, TargetAdmissionError> {
        computed_fingerprint(self)
    }

    /// Validates the envelope, canonical topology list, and fingerprint.
    ///
    /// # Errors
    ///
    /// Returns an error for any unsupported or noncanonical contract field.
    fn validate(&self) -> Result<(), TargetAdmissionError> {
        validate_envelope(
            "runtime target contract",
            &self.schema,
            self.version,
            RUNTIME_TARGET_SCHEMA,
            RUNTIME_TARGET_VERSION,
            &self.fingerprint,
        )?;
        if !is_stable_identifier(&self.target.id) {
            return Err(TargetAdmissionError::InvalidIdentifier { field: "target.id" });
        }
        if !is_stable_identifier(&self.admission_profile.id) {
            return Err(TargetAdmissionError::InvalidIdentifier {
                field: "admission_profile.id",
            });
        }
        if self.target.contract_version == 0
            || self.admission_profile.version == 0
            || self.page_tokens == 0
            || self.required_wire_version == 0
        {
            return Err(TargetAdmissionError::InvalidTargetContractGeometry);
        }
        if self.supported_manifest_versions.is_empty()
            || !strictly_sorted(&self.supported_manifest_versions)
            || self.supported_capabilities.is_empty()
            || !strictly_sorted(&self.supported_capabilities)
            || self.supported_topologies.is_empty()
            || !self
                .supported_topologies
                .windows(2)
                .all(|pair| pair[0] < pair[1])
        {
            return Err(TargetAdmissionError::NonCanonicalTargetLists);
        }
        validate_fingerprint("runtime target contract", &self.fingerprint, self)
    }
}

fn strictly_sorted<T: Ord>(values: &[T]) -> bool {
    !values.is_empty() && values.windows(2).all(|pair| pair[0] < pair[1])
}

impl ExecutionSignature {
    /// Parses and validates a serialized execution signature.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed, oversized, noncanonical, or forged
    /// signature data.
    pub fn from_json(bytes: &[u8]) -> Result<Self, TargetAdmissionError> {
        from_json(bytes, "execution signature")
    }

    /// Computes the canonical content fingerprint excluding `fingerprint`.
    ///
    /// # Errors
    ///
    /// Returns an error if the typed signature cannot be serialized.
    pub fn computed_fingerprint(&self) -> Result<String, TargetAdmissionError> {
        computed_fingerprint(self)
    }

    /// Recompiles the structural projection and validates its fingerprint.
    ///
    /// # Errors
    ///
    /// Returns an error when the signature is not compiler-derived or its
    /// envelope or fingerprint is invalid.
    pub fn validate(&self) -> Result<(), TargetAdmissionError> {
        validate_envelope(
            "execution signature",
            &self.schema,
            self.version,
            EXECUTION_SIGNATURE_SCHEMA,
            EXECUTION_SIGNATURE_VERSION,
            &self.fingerprint,
        )?;
        if self.manifest_schema != RUNTIME_MANIFEST_SCHEMA
            || !is_sha256_fingerprint(&self.manifest_fingerprint)
            || self.page_tokens == 0
        {
            return Err(TargetAdmissionError::InvalidExecutionSignature(
                "invalid manifest identity or page geometry",
            ));
        }
        if self.manifest_version != RUNTIME_MANIFEST_VERSION {
            return Err(TargetAdmissionError::InvalidExecutionSignature(
                "unsupported manifest version",
            ));
        }
        validate_signature_projection(self)?;
        validate_fingerprint("execution signature", &self.fingerprint, self)
    }
}

impl RuntimeBinding {
    /// Parses and validates a serialized runtime target binding.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed, oversized, noncanonical, or forged
    /// binding data.
    pub fn from_json(bytes: &[u8]) -> Result<Self, TargetAdmissionError> {
        from_json(bytes, "runtime target binding")
    }

    /// Computes the canonical content fingerprint excluding `fingerprint`.
    ///
    /// # Errors
    ///
    /// Returns an error if the typed binding cannot be serialized.
    pub fn computed_fingerprint(&self) -> Result<String, TargetAdmissionError> {
        computed_fingerprint(self)
    }

    /// Validates the binding envelope and every embedded identity.
    ///
    /// # Errors
    ///
    /// Returns an error when the signature, topology, identities, or
    /// fingerprint disagree.
    pub fn validate(&self) -> Result<(), TargetAdmissionError> {
        self.validate_shape()?;
        let target = RuntimeTarget::packaged_sglang()?;
        if self.target != target.target
            || self.admission_profile != target.admission_profile
            || self.target_contract_fingerprint != target.fingerprint
            || self.required_wire_version != target.required_wire_version
        {
            return Err(TargetAdmissionError::BindingIdentityMismatch);
        }
        Ok(())
    }

    fn validate_shape(&self) -> Result<(), TargetAdmissionError> {
        validate_envelope(
            "runtime target binding",
            &self.schema,
            self.version,
            RUNTIME_BINDING_SCHEMA,
            RUNTIME_BINDING_VERSION,
            &self.fingerprint,
        )?;
        self.execution_signature.validate()?;
        if !is_stable_identifier(&self.target.id)
            || self.target.contract_version == 0
            || !is_stable_identifier(&self.admission_profile.id)
            || self.admission_profile.version == 0
            || !is_sha256_fingerprint(&self.manifest_fingerprint)
            || !is_sha256_fingerprint(&self.target_contract_fingerprint)
            || self.manifest_fingerprint != self.execution_signature.manifest_fingerprint
        {
            return Err(TargetAdmissionError::BindingIdentityMismatch);
        }
        if self.required_wire_version == 0 {
            return Err(TargetAdmissionError::BindingIdentityMismatch);
        }
        if classify_execution_signature(&self.execution_signature)? != self.execution_topology {
            return Err(TargetAdmissionError::BindingIdentityMismatch);
        }
        validate_fingerprint("runtime target binding", &self.fingerprint, self)
    }

    /// Recomputes admission against the packaged `SGLang` target and requires
    /// byte-meaningful typed equality.
    ///
    /// # Errors
    ///
    /// Returns an error when the binding is invalid or differs from a freshly
    /// admitted product binding for the supplied manifest.
    pub fn validate_against(&self, manifest: &RuntimeManifest) -> Result<(), TargetAdmissionError> {
        let target = RuntimeTarget::packaged_sglang()?;
        self.validate_against_target(manifest, &target)
    }

    fn validate_against_target(
        &self,
        manifest: &RuntimeManifest,
        target: &RuntimeTarget,
    ) -> Result<(), TargetAdmissionError> {
        self.validate_shape()?;
        let expected = admit_runtime_manifest_for_target(manifest, target)?;
        if *self != expected {
            return Err(TargetAdmissionError::BindingMismatch);
        }
        Ok(())
    }
}

/// Derives the complete structural signature from a validated manifest.
/// # Errors
/// Returns an error when the manifest is invalid or its compiler-derived
/// state/layout projection cannot be represented consistently.
#[allow(clippy::too_many_lines)]
pub fn derive_execution_signature(
    manifest: &RuntimeManifest,
) -> Result<ExecutionSignature, TargetAdmissionError> {
    manifest.validate()?;
    let token_classes = manifest
        .token_manager_plan
        .as_ref()
        .map_or_else(Vec::new, |manager| {
            manager
                .layout
                .classes
                .iter()
                .map(ExecutionTokenClass::from)
                .collect()
        });
    let (page_tokens, token_states, fixed_states) = match &manifest.source {
        RuntimeManifestSource::AttentionState { .. } => {
            let attention = manifest.attention_state_plan.as_ref().ok_or(
                TargetAdmissionError::InvalidExecutionSignature(
                    "attention-state source has no compiled plan",
                ),
            )?;
            let mut token_states = Vec::new();
            let mut fixed_states = Vec::new();
            for state in &attention.states {
                match state.backend {
                    AttentionStateBackend::TokenSlots { .. } => {
                        token_states.push(ExecutionTokenState::from(state));
                    }
                    AttentionStateBackend::RecurrentCheckpoints { .. }
                    | AttentionStateBackend::ConvolutionRing { .. } => {
                        fixed_states.push(ExecutionFixedState::from(state));
                    }
                }
            }
            (attention.page_tokens, token_states, fixed_states)
        }
        RuntimeManifestSource::RetentionIr { program } => {
            let layout = &manifest
                .token_manager_plan
                .as_ref()
                .ok_or(TargetAdmissionError::UnsupportedManifestTopology)?
                .layout;
            if program.states.len() != 1 || layout.classes.len() != 1 {
                return Err(TargetAdmissionError::UnsupportedManifestTopology);
            }
            let state = &program.states[0];
            let class = &layout.classes[0];
            let InferredRetention::Chunked { chunk_tokens } = analyze_state(state)?.inferred else {
                return Err(TargetAdmissionError::UnsupportedManifestTopology);
            };
            let AddressProgram::ResettableArena { blocks_per_epoch } = class.address else {
                return Err(TargetAdmissionError::UnsupportedManifestTopology);
            };
            let expected_chunk = blocks_per_epoch
                .checked_mul(program.page_tokens)
                .ok_or(TargetAdmissionError::UnsupportedManifestTopology)?;
            if blocks_per_epoch == 0
                || chunk_tokens != expected_chunk
                || class.name != state.name
                || class.layers != state.layers
                || class.kv_head_range.is_some()
                || state.kv_head_range.is_some()
                || !class.block_domain.is_all()
                || class.bytes_per_token_per_layer != state.bytes_per_token_per_layer
                || class.minimum_slots_per_request != Some(blocks_per_epoch)
                || !matches!(
                    class.retirement,
                    RetirementProgram::EpochEnd {
                        blocks_per_epoch: retirement_blocks
                    } if retirement_blocks == blocks_per_epoch
                )
                || !covers_all_layers([class.layers.as_slice()])
            {
                return Err(TargetAdmissionError::UnsupportedManifestTopology);
            }
            let page_bytes_per_layer = state
                .bytes_per_token_per_layer
                .checked_mul(program.page_tokens)
                .ok_or(TargetAdmissionError::InvalidExecutionSignature(
                    "token page geometry overflows",
                ))?;
            let token_state = ExecutionTokenState {
                name: state.name.clone(),
                layers: state.layers.clone(),
                backend: ExecutionTokenBackend::TokenSlots {
                    storage: TokenStorageKind::TokenKv,
                    components: Vec::new(),
                    bytes_per_token_per_layer: state.bytes_per_token_per_layer,
                    page_bytes_per_layer,
                    retention: RetentionKind::Chunked,
                    window_tokens: None,
                    token_relocatable: true,
                },
            };
            (program.page_tokens, vec![token_state], Vec::new())
        }
    };
    let mut signature = ExecutionSignature {
        schema: EXECUTION_SIGNATURE_SCHEMA.to_owned(),
        version: EXECUTION_SIGNATURE_VERSION,
        fingerprint: String::new(),
        manifest_schema: manifest.schema.clone(),
        manifest_version: manifest.version,
        manifest_fingerprint: manifest.fingerprint.clone(),
        page_tokens,
        token_classes,
        token_states,
        fixed_states,
    };
    signature.fingerprint = signature.computed_fingerprint()?;
    signature.validate()?;
    Ok(signature)
}

/// Structurally admits a manifest and emits a binding to the packaged `SGLang`
/// target contract. No device, stream, tensor, or engine object is touched.
/// # Errors
/// Returns an error when the manifest is invalid, page geometry differs, or
/// its topology is unsupported by the product target.
pub fn admit_runtime_manifest(
    manifest: &RuntimeManifest,
) -> Result<RuntimeBinding, TargetAdmissionError> {
    let target = RuntimeTarget::packaged_sglang()?;
    admit_runtime_manifest_for_target(manifest, &target)
}

fn admit_runtime_manifest_for_target(
    manifest: &RuntimeManifest,
    target: &RuntimeTarget,
) -> Result<RuntimeBinding, TargetAdmissionError> {
    target.validate()?;
    validate_target_manifest_version(target, manifest.version)?;
    for capability in &manifest.capability_requirements {
        if target
            .supported_capabilities
            .binary_search(capability)
            .is_err()
        {
            return Err(TargetAdmissionError::UnsupportedTargetCapability(
                *capability,
            ));
        }
    }
    let signature = derive_execution_signature(manifest)?;
    if signature.page_tokens != target.page_tokens {
        return Err(TargetAdmissionError::PageTokensMismatch {
            target: target.page_tokens,
            manifest: signature.page_tokens,
        });
    }
    let execution_topology = classify_execution_signature(&signature)?;
    if target
        .supported_topologies
        .binary_search(&execution_topology)
        .is_err()
    {
        return Err(TargetAdmissionError::UnsupportedTargetTopology(
            execution_topology,
        ));
    }
    let mut binding = RuntimeBinding {
        schema: RUNTIME_BINDING_SCHEMA.to_owned(),
        version: RUNTIME_BINDING_VERSION,
        fingerprint: String::new(),
        manifest_fingerprint: manifest.fingerprint.clone(),
        target: target.target.clone(),
        admission_profile: target.admission_profile.clone(),
        target_contract_fingerprint: target.fingerprint.clone(),
        required_wire_version: target.required_wire_version,
        execution_topology,
        execution_signature: signature,
    };
    binding.fingerprint = binding.computed_fingerprint()?;
    binding.validate_shape()?;
    Ok(binding)
}

fn validate_target_manifest_version(
    target: &RuntimeTarget,
    manifest_version: u32,
) -> Result<(), TargetAdmissionError> {
    if target
        .supported_manifest_versions
        .binary_search(&manifest_version)
        .is_ok()
    {
        return Ok(());
    }
    Err(TargetAdmissionError::UnsupportedTargetManifestVersion {
        target: target.target.id.clone(),
        target_contract_version: target.target.contract_version,
        admission_profile: target.admission_profile.id.clone(),
        admission_profile_version: target.admission_profile.version,
        manifest_version,
    })
}

fn validate_signature_projection(
    signature: &ExecutionSignature,
) -> Result<(), TargetAdmissionError> {
    if signature.token_states.iter().any(|state| {
        matches!(
            state.backend,
            ExecutionTokenBackend::TokenSlots {
                retention: RetentionKind::Chunked,
                ..
            }
        )
    }) {
        return validate_chunked_signature_projection(signature);
    }
    let mut states =
        Vec::with_capacity(signature.token_states.len() + signature.fixed_states.len());
    states.extend(
        signature
            .token_states
            .iter()
            .map(ExecutionTokenState::to_input),
    );
    states.extend(
        signature
            .fixed_states
            .iter()
            .map(ExecutionFixedState::to_input),
    );
    let states = states.into_iter().collect::<Result<Vec<_>, _>>()?;
    let compiled = compile_attention_state_plan(AttentionStatePlanInput {
        page_tokens: signature.page_tokens,
        states,
    })?;
    let expected_tokens = compiled
        .states
        .iter()
        .filter(|state| matches!(state.backend, AttentionStateBackend::TokenSlots { .. }))
        .map(ExecutionTokenState::from)
        .collect::<Vec<_>>();
    let expected_fixed = compiled
        .states
        .iter()
        .filter(|state| !matches!(state.backend, AttentionStateBackend::TokenSlots { .. }))
        .map(ExecutionFixedState::from)
        .collect::<Vec<_>>();
    if signature.token_states != expected_tokens || signature.fixed_states != expected_fixed {
        return Err(TargetAdmissionError::InvalidExecutionSignature(
            "compiled state geometry differs",
        ));
    }
    let expected_classes = match compiled.token_manager_plan() {
        Ok(input) => compile_plan(input)?
            .layout_program()?
            .classes
            .iter()
            .map(ExecutionTokenClass::from)
            .collect(),
        Err(AttentionStateError::NoTokenState) => Vec::new(),
        Err(error) => return Err(error.into()),
    };
    if signature.token_classes != expected_classes {
        return Err(TargetAdmissionError::InvalidExecutionSignature(
            "compiled token layout differs",
        ));
    }
    Ok(())
}

fn validate_chunked_signature_projection(
    signature: &ExecutionSignature,
) -> Result<(), TargetAdmissionError> {
    if !signature.fixed_states.is_empty()
        || signature.token_classes.len() != 1
        || signature.token_states.len() != 1
        || !chunked_class(
            &signature.token_classes[0],
            &signature.token_states[0],
            signature.page_tokens,
        )
        || !covers_all_layers([signature.token_states[0].layers.as_slice()])
    {
        return Err(TargetAdmissionError::InvalidExecutionSignature(
            "chunked token projection differs",
        ));
    }
    Ok(())
}

fn classify_execution_signature(
    signature: &ExecutionSignature,
) -> Result<ExecutionTopology, TargetAdmissionError> {
    signature.validate()?;
    let classes = &signature.token_classes;
    let tokens = &signature.token_states;
    let fixed = &signature.fixed_states;

    if fixed.is_empty()
        && classes.len() == 1
        && tokens.len() == 1
        && chunked_class(&classes[0], &tokens[0], signature.page_tokens)
        && covers_all_layers([tokens[0].layers.as_slice()])
    {
        return Ok(ExecutionTopology::WholeDomainChunkedTokenKv);
    }

    if fixed.is_empty() && classes.len() == 1 && tokens.len() == 1 {
        if full_class(&classes[0], &tokens[0], TokenStorageKind::LatentKv)
            && covers_all_layers([tokens[0].layers.as_slice()])
        {
            return Ok(ExecutionTopology::WholeDomainFullLatentKv);
        }
        if full_class(&classes[0], &tokens[0], TokenStorageKind::TokenKv)
            && covers_all_layers([tokens[0].layers.as_slice()])
        {
            return Ok(ExecutionTopology::WholeDomainFullTokenKv);
        }
        if sliding_class(&classes[0], &tokens[0], signature.page_tokens)
            && covers_all_layers([tokens[0].layers.as_slice()])
        {
            return Ok(ExecutionTopology::WholeDomainSlidingTokenKv);
        }
    }

    if fixed.is_empty()
        && classes.len() == 2
        && tokens.len() == 2
        && full_class(&classes[0], &tokens[0], TokenStorageKind::TokenKv)
        && sliding_class(&classes[1], &tokens[1], signature.page_tokens)
        && covers_all_layers([tokens[0].layers.as_slice(), tokens[1].layers.as_slice()])
    {
        return Ok(ExecutionTopology::WholeDomainFullSlidingTokenKv);
    }

    if classes.len() == 1
        && tokens.len() == 1
        && full_class(&classes[0], &tokens[0], TokenStorageKind::TokenKv)
    {
        if !fixed.is_empty()
            && fixed
                .iter()
                .all(|state| recurrent_state(state, RecurrentFamily::Mamba, 2))
            && covers_layer_groups(
                std::iter::once(tokens[0].layers.as_slice())
                    .chain(fixed.iter().map(|state| state.layers.as_slice())),
            )
        {
            return Ok(ExecutionTopology::WholeDomainFullTokenKvMamba);
        }
        if fixed.len() == 2 {
            let recurrent = fixed
                .iter()
                .find(|state| recurrent_state(state, RecurrentFamily::Gdn, 2));
            let convolution = fixed.iter().find(|state| convolution_state(state, 2));
            if let (Some(recurrent), Some(convolution)) = (recurrent, convolution)
                && recurrent.layers == convolution.layers
                && disjoint_cover(&tokens[0].layers, &recurrent.layers)
            {
                return Ok(ExecutionTopology::WholeDomainFullTokenKvGdnConvolution);
            }
        }
    }
    Err(TargetAdmissionError::UnsupportedManifestTopology)
}

fn chunked_class(
    class: &ExecutionTokenClass,
    state: &ExecutionTokenState,
    page_tokens: u64,
) -> bool {
    if class.name != state.name || class.layers != state.layers {
        return false;
    }
    let ExecutionAddressProgram::ResettableArena { blocks_per_epoch } = &class.address else {
        return false;
    };
    let blocks_per_epoch = *blocks_per_epoch;
    let ExecutionTokenBackend::TokenSlots {
        storage,
        components,
        retention,
        window_tokens,
        token_relocatable,
        bytes_per_token_per_layer,
        page_bytes_per_layer,
    } = &state.backend;
    let Some(expected_page_bytes) = bytes_per_token_per_layer.checked_mul(page_tokens) else {
        return false;
    };
    matches!(
        &class.retirement,
        ExecutionRetirementProgram::EpochEnd {
            blocks_per_epoch: retirement_blocks
        } if *retirement_blocks == blocks_per_epoch
    ) && blocks_per_epoch > 0
        && whole_domain(class)
        && class.minimum_slots_per_request == Some(blocks_per_epoch)
        && *storage == TokenStorageKind::TokenKv
        && components.is_empty()
        && *retention == RetentionKind::Chunked
        && window_tokens.is_none()
        && *token_relocatable
        && class.bytes_per_token_per_layer == *bytes_per_token_per_layer
        && *page_bytes_per_layer == expected_page_bytes
}

fn full_class(
    class: &ExecutionTokenClass,
    state: &ExecutionTokenState,
    storage: TokenStorageKind,
) -> bool {
    whole_domain(class)
        && class.name == state.name
        && class.layers == state.layers
        && matches!(class.address, ExecutionAddressProgram::AppendOnly)
        && matches!(class.retirement, ExecutionRetirementProgram::Never)
        && class.minimum_slots_per_request.is_none()
        && matches!(
            state.backend,
            ExecutionTokenBackend::TokenSlots {
                storage: actual,
                retention: RetentionKind::Full,
                window_tokens: None,
                token_relocatable: true,
                ..
            } if actual == storage
        )
}

fn sliding_class(
    class: &ExecutionTokenClass,
    state: &ExecutionTokenState,
    page_tokens: u64,
) -> bool {
    let ExecutionTokenBackend::TokenSlots {
        storage,
        retention,
        window_tokens: Some(window_tokens),
        token_relocatable,
        ..
    } = state.backend
    else {
        return false;
    };
    if storage != TokenStorageKind::TokenKv
        || retention != RetentionKind::Sliding
        || !token_relocatable
        || class.name != state.name
        || class.layers != state.layers
        || !whole_domain(class)
    {
        return false;
    }
    let Some(period_blocks) = window_tokens.checked_sub(1).and_then(|value| {
        let rounded = value / page_tokens + u64::from(value % page_tokens != 0);
        rounded.checked_add(1)
    }) else {
        return false;
    };
    matches!(
        class.address,
        ExecutionAddressProgram::Periodic { period_blocks: actual }
            if actual == period_blocks
    ) && matches!(
        class.retirement,
        ExecutionRetirementProgram::BlockEndPlus { offset_tokens }
            if offset_tokens == window_tokens - 1
    ) && class.minimum_slots_per_request == Some(period_blocks)
}

fn whole_domain(class: &ExecutionTokenClass) -> bool {
    class.kv_head_range.is_none() && class.block_domain.is_all()
}

fn recurrent_state(state: &ExecutionFixedState, family: RecurrentFamily, slots: u32) -> bool {
    matches!(
        state.backend,
        ExecutionFixedBackend::RecurrentCheckpoints {
            family: actual,
            checkpoint_slots_per_request,
            token_relocatable: false,
            ..
        } if actual == family && checkpoint_slots_per_request == slots
    )
}

fn convolution_state(state: &ExecutionFixedState, slots: u32) -> bool {
    matches!(
        state.backend,
        ExecutionFixedBackend::ConvolutionRing {
            checkpoint_slots_per_request,
            token_relocatable: false,
            ..
        } if checkpoint_slots_per_request == slots
    )
}

fn covers_all_layers<const N: usize>(groups: [&[u32]; N]) -> bool {
    covers_layer_groups(groups)
}

fn covers_layer_groups<'a>(groups: impl IntoIterator<Item = &'a [u32]>) -> bool {
    let groups = groups.into_iter().collect::<Vec<_>>();
    let mut layers = groups
        .iter()
        .flat_map(|group| group.iter().copied())
        .collect::<Vec<_>>();
    if layers.is_empty()
        || groups
            .iter()
            .any(|group| group.is_empty() || !group.windows(2).all(|pair| pair[0] < pair[1]))
    {
        return false;
    }
    layers.sort_unstable();
    layers
        .iter()
        .copied()
        .eq(0..u32::try_from(layers.len()).unwrap_or(u32::MAX))
}

fn disjoint_cover(first: &[u32], second: &[u32]) -> bool {
    covers_all_layers([first, second])
}

impl ExecutionTokenState {
    fn to_input(&self) -> Result<AttentionStateSpec, TargetAdmissionError> {
        let ExecutionTokenBackend::TokenSlots {
            storage,
            components,
            retention,
            window_tokens,
            token_relocatable,
            ..
        } = &self.backend;
        if !token_relocatable {
            return Err(TargetAdmissionError::InvalidExecutionSignature(
                "token state is not relocatable",
            ));
        }
        let component = |first: &str, second: &str| {
            if components.len() != 2 || components[0].name != first || components[1].name != second
            {
                return Err(TargetAdmissionError::InvalidExecutionSignature(
                    "token component order differs",
                ));
            }
            Ok((
                components[0].bytes_per_token_per_layer,
                components[1].bytes_per_token_per_layer,
            ))
        };
        let storage = match storage {
            TokenStorageKind::TokenKv => {
                let (key, value) = component("key", "value")?;
                AttentionStateStorage::TokenKv {
                    key_bytes_per_token_per_layer: key,
                    value_bytes_per_token_per_layer: value,
                    retention: *retention,
                    window_tokens: *window_tokens,
                }
            }
            TokenStorageKind::LatentKv => {
                let (latent, rope) = component("latent", "rope")?;
                AttentionStateStorage::LatentKv {
                    latent_bytes_per_token_per_layer: latent,
                    rope_bytes_per_token_per_layer: rope,
                    retention: *retention,
                    window_tokens: *window_tokens,
                }
            }
        };
        Ok(AttentionStateSpec {
            name: self.name.clone(),
            layers: self.layers.clone(),
            storage,
        })
    }
}

impl ExecutionFixedState {
    fn to_input(&self) -> Result<AttentionStateSpec, TargetAdmissionError> {
        let storage = match self.backend {
            ExecutionFixedBackend::RecurrentCheckpoints {
                family,
                state_bytes_per_layer,
                checkpoint_slots_per_request,
                token_relocatable,
                ..
            } => {
                if token_relocatable {
                    return Err(TargetAdmissionError::InvalidExecutionSignature(
                        "fixed state is relocatable",
                    ));
                }
                AttentionStateStorage::Recurrent {
                    family,
                    state_bytes_per_layer,
                    checkpoint_slots_per_request,
                }
            }
            ExecutionFixedBackend::ConvolutionRing {
                state_bytes_per_layer,
                kernel_width,
                checkpoint_slots_per_request,
                token_relocatable,
                ..
            } => {
                if token_relocatable {
                    return Err(TargetAdmissionError::InvalidExecutionSignature(
                        "fixed state is relocatable",
                    ));
                }
                AttentionStateStorage::Convolution {
                    state_bytes_per_layer,
                    kernel_width,
                    checkpoint_slots_per_request,
                }
            }
        };
        Ok(AttentionStateSpec {
            name: self.name.clone(),
            layers: self.layers.clone(),
            storage,
        })
    }
}

impl From<&crate::plan::ClassLayoutProgram> for ExecutionTokenClass {
    fn from(value: &crate::plan::ClassLayoutProgram) -> Self {
        Self {
            name: value.name.clone(),
            layers: value.layers.clone(),
            kv_head_range: value.kv_head_range.as_ref().map(Into::into),
            bytes_per_token_per_layer: value.bytes_per_token_per_layer,
            address: (&value.address).into(),
            retirement: (&value.retirement).into(),
            minimum_slots_per_request: value.minimum_slots_per_request,
            block_domain: (&value.block_domain).into(),
        }
    }
}

impl From<&KvHeadRange> for ExecutionKvHeadRange {
    fn from(value: &KvHeadRange) -> Self {
        Self {
            start: value.start,
            end_exclusive: value.end_exclusive,
        }
    }
}

impl From<&BlockDomain> for ExecutionBlockDomain {
    fn from(value: &BlockDomain) -> Self {
        Self {
            start_block: value.start_block,
            end_block_exclusive: value.end_block_exclusive,
        }
    }
}

impl From<&AddressProgram> for ExecutionAddressProgram {
    fn from(value: &AddressProgram) -> Self {
        match *value {
            AddressProgram::AppendOnly => Self::AppendOnly,
            AddressProgram::Pinned => Self::Pinned,
            AddressProgram::Periodic { period_blocks } => Self::Periodic { period_blocks },
            AddressProgram::PeriodicFrom {
                period_blocks,
                origin_block,
            } => Self::PeriodicFrom {
                period_blocks,
                origin_block,
            },
            AddressProgram::ResettableArena { blocks_per_epoch } => {
                Self::ResettableArena { blocks_per_epoch }
            }
        }
    }
}

impl From<&RetirementProgram> for ExecutionRetirementProgram {
    fn from(value: &RetirementProgram) -> Self {
        match *value {
            RetirementProgram::Never => Self::Never,
            RetirementProgram::BlockEndPlus { offset_tokens } => {
                Self::BlockEndPlus { offset_tokens }
            }
            RetirementProgram::EpochEnd { blocks_per_epoch } => Self::EpochEnd { blocks_per_epoch },
        }
    }
}

impl From<&CompiledAttentionState> for ExecutionTokenState {
    fn from(value: &CompiledAttentionState) -> Self {
        let AttentionStateBackend::TokenSlots {
            storage,
            components,
            bytes_per_token_per_layer,
            page_bytes_per_layer,
            retention,
            window_tokens,
            token_relocatable,
        } = &value.backend
        else {
            unreachable!("token conversion requires token state");
        };
        Self {
            name: value.name.clone(),
            layers: value.layers.clone(),
            backend: ExecutionTokenBackend::TokenSlots {
                storage: *storage,
                components: components
                    .iter()
                    .map(|component| ExecutionStateComponent {
                        name: component.name.to_owned(),
                        bytes_per_token_per_layer: component.bytes_per_token_per_layer,
                    })
                    .collect(),
                bytes_per_token_per_layer: *bytes_per_token_per_layer,
                page_bytes_per_layer: *page_bytes_per_layer,
                retention: *retention,
                window_tokens: *window_tokens,
                token_relocatable: *token_relocatable,
            },
        }
    }
}

impl From<&CompiledAttentionState> for ExecutionFixedState {
    fn from(value: &CompiledAttentionState) -> Self {
        let backend = match value.backend {
            AttentionStateBackend::RecurrentCheckpoints {
                family,
                state_bytes_per_layer,
                checkpoint_slots_per_request,
                checkpoint_bytes_per_request,
                token_relocatable,
            } => ExecutionFixedBackend::RecurrentCheckpoints {
                family,
                state_bytes_per_layer,
                checkpoint_slots_per_request,
                checkpoint_bytes_per_request,
                token_relocatable,
            },
            AttentionStateBackend::ConvolutionRing {
                state_bytes_per_layer,
                kernel_width,
                checkpoint_slots_per_request,
                checkpoint_bytes_per_request,
                token_relocatable,
            } => ExecutionFixedBackend::ConvolutionRing {
                state_bytes_per_layer,
                kernel_width,
                checkpoint_slots_per_request,
                checkpoint_bytes_per_request,
                token_relocatable,
            },
            AttentionStateBackend::TokenSlots { .. } => {
                unreachable!("fixed conversion requires fixed state")
            }
        };
        Self {
            name: value.name.clone(),
            layers: value.layers.clone(),
            backend,
        }
    }
}

fn validate_envelope(
    artifact: &'static str,
    schema: &str,
    version: u32,
    expected_schema: &str,
    expected_version: u32,
    fingerprint: &str,
) -> Result<(), TargetAdmissionError> {
    if schema != expected_schema {
        return Err(TargetAdmissionError::UnsupportedSchema {
            artifact,
            actual: schema.to_owned(),
        });
    }
    if version != expected_version {
        return Err(TargetAdmissionError::UnsupportedVersion {
            artifact,
            actual: version,
        });
    }
    if !is_sha256_fingerprint(fingerprint) {
        return Err(TargetAdmissionError::MalformedFingerprint { artifact });
    }
    Ok(())
}

fn validate_fingerprint<T: Serialize>(
    artifact: &'static str,
    actual: &str,
    value: &T,
) -> Result<(), TargetAdmissionError> {
    if actual != computed_fingerprint(value)? {
        return Err(TargetAdmissionError::FingerprintMismatch { artifact });
    }
    Ok(())
}

fn computed_fingerprint<T: Serialize>(value: &T) -> Result<String, TargetAdmissionError> {
    let mut payload = serde_json::to_value(value)?;
    let Some(object) = payload.as_object_mut() else {
        return Err(TargetAdmissionError::NonCanonicalJson {
            artifact: "target artifact",
        });
    };
    object.remove("fingerprint");
    let mut canonical = Vec::new();
    write_canonical_json(&payload, &mut canonical)?;
    Ok(format!("sha256:{:x}", Sha256::digest(canonical)))
}

fn from_json<T>(bytes: &[u8], artifact: &'static str) -> Result<T, TargetAdmissionError>
where
    T: for<'de> Deserialize<'de> + Serialize + ValidateArtifact,
{
    if bytes.len() > RUNTIME_TARGET_ARTIFACT_MAX_BYTES {
        return Err(TargetAdmissionError::ArtifactTooLarge {
            artifact,
            actual: bytes.len(),
            maximum: RUNTIME_TARGET_ARTIFACT_MAX_BYTES,
        });
    }
    let raw = serde_json::from_slice::<Value>(bytes)?;
    let value = serde_json::from_slice::<T>(bytes)?;
    if serde_json::to_value(&value)? != raw {
        return Err(TargetAdmissionError::NonCanonicalJson { artifact });
    }
    value.validate_artifact()?;
    Ok(value)
}

trait ValidateArtifact {
    fn validate_artifact(&self) -> Result<(), TargetAdmissionError>;
}

impl ValidateArtifact for ExecutionSignature {
    fn validate_artifact(&self) -> Result<(), TargetAdmissionError> {
        self.validate()
    }
}

impl ValidateArtifact for RuntimeBinding {
    fn validate_artifact(&self) -> Result<(), TargetAdmissionError> {
        self.validate()
    }
}

fn is_stable_identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'.' | b'-' | b'_')
        })
        && value.as_bytes()[0].is_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    include!("runtime_target_tests.rs");
}
