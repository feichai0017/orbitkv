use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::attention_state::{
    AttentionStateBackend, AttentionStateError, AttentionStatePlanInput, AttentionStateSpec,
    AttentionStateStorage, CompiledAttentionState, CompiledAttentionStatePlan, RecurrentFamily,
    StateComponentGeometry, compile_attention_state_plan,
};
use crate::hf_config::{
    HfConfigError, HfManagerPlanError, HfRetentionOptions, compile_hf_attention_state_input,
    compile_hf_token_manager_plan,
};
use crate::plan::{
    AddressProgram, BlockDomain, ClassLayoutProgram, KvClassSpec, KvPlanInput, LayoutProgram,
    PlanError, RetirementProgram, TokenStorageKind, compile_plan,
};
use crate::retention::KvHeadRange;

pub const RUNTIME_MANIFEST_SCHEMA: &str = "orbitkv.runtime-manifest";
pub const RUNTIME_MANIFEST_VERSION: u32 = 1;
/// Upper bound for serialized manifests admitted by runtime consumers.
pub const RUNTIME_MANIFEST_MAX_BYTES: usize = 16 * 1024 * 1024;

/// Closed v1 capability vocabulary serialized as the listed snake-case names.
///
/// The compiler emits this vector in lexical order with no duplicates. These
/// values describe what the serialized physical plan requires to be admitted;
/// optional runtime policies such as token relocation and future lifecycle
/// operations are deliberately not inferred from storage geometry alone.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeCapability {
    AppendOnlyAddressing,
    ConvolutionState,
    FixedStateCheckpoints,
    PeriodicAddressing,
    RecurrentState,
    SemanticRetirement,
    TokenComponentGeometry,
    TokenManager,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RuntimeTokenManagerPlan {
    pub input: KvPlanInput,
    pub layout: LayoutProgram,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RuntimeManifest {
    pub schema: String,
    pub version: u32,
    pub fingerprint: String,
    pub token_manager_plan: Option<RuntimeTokenManagerPlan>,
    pub attention_state_plan: CompiledAttentionStatePlan,
    pub capability_requirements: Vec<RuntimeCapability>,
}

impl<'de> Deserialize<'de> for RuntimeManifest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = RuntimeManifestWire::deserialize(deserializer)?;
        let manifest = Self {
            schema: wire.schema,
            version: wire.version,
            fingerprint: wire.fingerprint,
            token_manager_plan: wire
                .token_manager_plan
                .map(RuntimeTokenManagerPlanWire::into_runtime)
                .transpose()
                .map_err(serde::de::Error::custom)?,
            attention_state_plan: wire
                .attention_state_plan
                .into_compiled()
                .map_err(serde::de::Error::custom)?,
            capability_requirements: wire.capability_requirements,
        };
        manifest.validate().map_err(serde::de::Error::custom)?;
        Ok(manifest)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeManifestWire {
    schema: String,
    version: u32,
    fingerprint: String,
    token_manager_plan: Option<RuntimeTokenManagerPlanWire>,
    attention_state_plan: CompiledAttentionStatePlanWire,
    capability_requirements: Vec<RuntimeCapability>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeTokenManagerPlanWire {
    input: KvPlanInput,
    layout: LayoutProgramWire,
}

impl RuntimeTokenManagerPlanWire {
    fn into_runtime(self) -> Result<RuntimeTokenManagerPlan, RuntimeManifestError> {
        Ok(RuntimeTokenManagerPlan {
            input: self.input,
            layout: self.layout.into_layout()?,
        })
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LayoutProgramWire {
    schema: String,
    plan_fingerprint: String,
    page_tokens: u64,
    classes: Vec<ClassLayoutProgramWire>,
}

impl LayoutProgramWire {
    fn into_layout(self) -> Result<LayoutProgram, RuntimeManifestError> {
        if self.schema != "orbitkv.layout-program.v1" {
            return Err(RuntimeManifestError::TokenManagerLayoutMismatch);
        }
        Ok(LayoutProgram {
            schema: "orbitkv.layout-program.v1",
            plan_fingerprint: self.plan_fingerprint,
            page_tokens: self.page_tokens,
            classes: self
                .classes
                .into_iter()
                .map(ClassLayoutProgramWire::into_layout)
                .collect(),
        })
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ClassLayoutProgramWire {
    name: String,
    layers: Vec<u32>,
    #[serde(default)]
    kv_head_range: Option<KvHeadRange>,
    bytes_per_token_per_layer: u64,
    address: AddressProgramWire,
    retirement: RetirementProgramWire,
    minimum_slots_per_request: Option<u64>,
    #[serde(default)]
    block_domain: BlockDomainWire,
}

impl ClassLayoutProgramWire {
    fn into_layout(self) -> ClassLayoutProgram {
        ClassLayoutProgram {
            name: self.name,
            layers: self.layers,
            kv_head_range: self.kv_head_range,
            bytes_per_token_per_layer: self.bytes_per_token_per_layer,
            address: self.address.into_runtime(),
            retirement: self.retirement.into_runtime(),
            minimum_slots_per_request: self.minimum_slots_per_request,
            block_domain: self.block_domain.into_runtime(),
        }
    }
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum AddressProgramWire {
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

impl AddressProgramWire {
    fn into_runtime(self) -> AddressProgram {
        match self {
            Self::AppendOnly => AddressProgram::AppendOnly,
            Self::Pinned => AddressProgram::Pinned,
            Self::Periodic { period_blocks } => AddressProgram::Periodic { period_blocks },
            Self::PeriodicFrom {
                period_blocks,
                origin_block,
            } => AddressProgram::PeriodicFrom {
                period_blocks,
                origin_block,
            },
            Self::ResettableArena { blocks_per_epoch } => {
                AddressProgram::ResettableArena { blocks_per_epoch }
            }
        }
    }
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum RetirementProgramWire {
    Never,
    BlockEndPlus { offset_tokens: u64 },
    EpochEnd { blocks_per_epoch: u64 },
}

impl RetirementProgramWire {
    fn into_runtime(self) -> RetirementProgram {
        match self {
            Self::Never => RetirementProgram::Never,
            Self::BlockEndPlus { offset_tokens } => {
                RetirementProgram::BlockEndPlus { offset_tokens }
            }
            Self::EpochEnd { blocks_per_epoch } => RetirementProgram::EpochEnd { blocks_per_epoch },
        }
    }
}

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct BlockDomainWire {
    start_block: u64,
    end_block_exclusive: Option<u64>,
}

impl BlockDomainWire {
    fn into_runtime(self) -> BlockDomain {
        BlockDomain {
            start_block: self.start_block,
            end_block_exclusive: self.end_block_exclusive,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CompiledAttentionStatePlanWire {
    schema: String,
    page_tokens: u64,
    states: Vec<CompiledAttentionStateWire>,
}

impl CompiledAttentionStatePlanWire {
    fn into_compiled(self) -> Result<CompiledAttentionStatePlan, RuntimeManifestError> {
        if self.schema != "orbitkv.attention-state-plan.v1" {
            return Err(RuntimeManifestError::AttentionStatePlanMismatch);
        }
        Ok(CompiledAttentionStatePlan {
            schema: "orbitkv.attention-state-plan.v1",
            page_tokens: self.page_tokens,
            states: self
                .states
                .into_iter()
                .map(CompiledAttentionStateWire::into_compiled)
                .collect::<Result<Vec<_>, _>>()?,
        })
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CompiledAttentionStateWire {
    name: String,
    layers: Vec<u32>,
    backend: AttentionStateBackendWire,
}

impl CompiledAttentionStateWire {
    fn into_compiled(self) -> Result<CompiledAttentionState, RuntimeManifestError> {
        Ok(CompiledAttentionState {
            name: self.name,
            layers: self.layers,
            backend: self.backend.into_backend()?,
        })
    }
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum AttentionStateBackendWire {
    TokenSlots {
        storage: TokenStorageKind,
        components: Vec<StateComponentGeometryWire>,
        bytes_per_token_per_layer: u64,
        page_bytes_per_layer: u64,
        retention: crate::plan::RetentionKind,
        window_tokens: Option<u64>,
        token_relocatable: bool,
    },
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

impl AttentionStateBackendWire {
    fn into_backend(self) -> Result<AttentionStateBackend, RuntimeManifestError> {
        Ok(match self {
            Self::TokenSlots {
                storage,
                components,
                bytes_per_token_per_layer,
                page_bytes_per_layer,
                retention,
                window_tokens,
                token_relocatable,
            } => AttentionStateBackend::TokenSlots {
                storage,
                components: components
                    .into_iter()
                    .map(StateComponentGeometryWire::into_geometry)
                    .collect::<Result<Vec<_>, _>>()?,
                bytes_per_token_per_layer,
                page_bytes_per_layer,
                retention,
                window_tokens,
                token_relocatable,
            },
            Self::RecurrentCheckpoints {
                family,
                state_bytes_per_layer,
                checkpoint_slots_per_request,
                checkpoint_bytes_per_request,
                token_relocatable,
            } => AttentionStateBackend::RecurrentCheckpoints {
                family,
                state_bytes_per_layer,
                checkpoint_slots_per_request,
                checkpoint_bytes_per_request,
                token_relocatable,
            },
            Self::ConvolutionRing {
                state_bytes_per_layer,
                kernel_width,
                checkpoint_slots_per_request,
                checkpoint_bytes_per_request,
                token_relocatable,
            } => AttentionStateBackend::ConvolutionRing {
                state_bytes_per_layer,
                kernel_width,
                checkpoint_slots_per_request,
                checkpoint_bytes_per_request,
                token_relocatable,
            },
        })
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StateComponentGeometryWire {
    name: String,
    bytes_per_token_per_layer: u64,
}

impl StateComponentGeometryWire {
    fn into_geometry(self) -> Result<StateComponentGeometry, RuntimeManifestError> {
        let name = match self.name.as_str() {
            "key" => "key",
            "value" => "value",
            "latent" => "latent",
            "rope" => "rope",
            _ => return Err(RuntimeManifestError::AttentionStatePlanMismatch),
        };
        Ok(StateComponentGeometry {
            name,
            bytes_per_token_per_layer: self.bytes_per_token_per_layer,
        })
    }
}

#[derive(Debug, Error)]
pub enum RuntimeManifestError {
    #[error("runtime manifest schema {0:?} is unsupported")]
    UnsupportedSchema(String),
    #[error("runtime manifest version {0} is unsupported")]
    UnsupportedVersion(u32),
    #[error(
        "runtime manifest fingerprint must be sha256 followed by 64 lowercase hexadecimal digits"
    )]
    MalformedFingerprint,
    #[error("runtime manifest fingerprint does not match its contents")]
    FingerprintMismatch,
    #[error("runtime manifest attention-state plan does not match its compiled contracts")]
    AttentionStatePlanMismatch,
    #[error("runtime manifest token-manager plan is missing or unexpectedly present")]
    TokenManagerPresenceMismatch,
    #[error("runtime manifest token-manager input does not match its attention-state projection")]
    TokenManagerInputMismatch,
    #[error("runtime manifest token-manager layout does not match its input")]
    TokenManagerLayoutMismatch,
    #[error("runtime manifest capability requirements are not canonical")]
    CapabilityRequirementsMismatch,
    #[error("runtime manifest JSON is not in its canonical typed shape")]
    NonCanonicalJson,
    #[error("runtime manifest is {actual} bytes, above the {maximum}-byte limit")]
    ManifestTooLarge { actual: usize, maximum: usize },
    #[error("HF token class {class:?} cannot be represented as attention state")]
    UnsupportedHfTokenClass { class: String },
    #[error(transparent)]
    AttentionState(#[from] AttentionStateError),
    #[error(transparent)]
    Plan(#[from] PlanError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

#[derive(Debug, Error)]
pub enum HfRuntimeManifestError {
    #[error(transparent)]
    Hf(#[from] HfManagerPlanError),
    #[error(transparent)]
    Manifest(#[from] RuntimeManifestError),
}

/// Compiles a hand-authored heterogeneous attention-state plan into the
/// versioned artifact consumed by runtime adapters.
///
/// # Errors
///
/// Returns an error if either the heterogeneous state contracts or their
/// token-manager projection cannot be compiled and self-validated.
pub fn compile_runtime_manifest(
    input: AttentionStatePlanInput,
) -> Result<RuntimeManifest, RuntimeManifestError> {
    compile_runtime_manifest_from_plan(compile_attention_state_plan(input)?)
}

/// Packages an existing compiled heterogeneous plan into a self-validating
/// executable artifact. The supplied plan is recompiled from its descriptors
/// before it is trusted.
///
/// # Errors
///
/// Returns an error if a derived field has been forged or the token layout
/// cannot be generated.
pub fn compile_runtime_manifest_from_plan(
    attention_state_plan: CompiledAttentionStatePlan,
) -> Result<RuntimeManifest, RuntimeManifestError> {
    let rebuilt = recompile_attention_state_plan(&attention_state_plan)?;
    if rebuilt != attention_state_plan {
        return Err(RuntimeManifestError::AttentionStatePlanMismatch);
    }
    let token_manager_plan = build_token_manager_plan(&attention_state_plan)?;
    let capability_requirements =
        derive_capability_requirements(&attention_state_plan, token_manager_plan.as_ref());
    let mut manifest = RuntimeManifest {
        schema: RUNTIME_MANIFEST_SCHEMA.to_owned(),
        version: RUNTIME_MANIFEST_VERSION,
        fingerprint: String::new(),
        token_manager_plan,
        attention_state_plan,
        capability_requirements,
    };
    manifest.fingerprint = manifest.computed_fingerprint()?;
    manifest.validate()?;
    manifest.validate_serialized_size_with_limit(RUNTIME_MANIFEST_MAX_BYTES)?;
    Ok(manifest)
}

/// Compiles either a supported heterogeneous HF config or an existing
/// token-only HF frontend profile into the same runtime artifact.
///
/// # Errors
///
/// Returns an error when the HF frontend cannot prove the config semantics or
/// when its compiled token geometry cannot be represented by attention state.
pub fn compile_hf_runtime_manifest(
    config_json: &[u8],
    options: HfRetentionOptions,
) -> Result<RuntimeManifest, HfRuntimeManifestError> {
    let token_input = compile_hf_token_manager_plan(config_json, options)?;
    let state_input = match compile_hf_attention_state_input(config_json, options) {
        Ok(input) => input,
        Err(error) if is_not_heterogeneous_profile(&error) => lift_token_plan(&token_input)?,
        Err(error) => return Err(HfManagerPlanError::Config(error).into()),
    };
    Ok(compile_runtime_manifest(state_input)?)
}

impl RuntimeManifest {
    /// Parses and validates a serialized runtime manifest before returning it.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed JSON, unknown fields or enum variants,
    /// unsupported versions, inconsistent compiler output, or fingerprint
    /// mismatch.
    pub fn from_json(bytes: &[u8]) -> Result<Self, RuntimeManifestError> {
        if bytes.len() > RUNTIME_MANIFEST_MAX_BYTES {
            return Err(RuntimeManifestError::ManifestTooLarge {
                actual: bytes.len(),
                maximum: RUNTIME_MANIFEST_MAX_BYTES,
            });
        }
        let raw = serde_json::from_slice::<Value>(bytes)?;
        let manifest = serde_json::from_slice::<Self>(bytes)?;
        if serde_json::to_value(&manifest)? != raw {
            return Err(RuntimeManifestError::NonCanonicalJson);
        }
        Ok(manifest)
    }

    /// Reconstructs every compiler-derived section and verifies the artifact
    /// fingerprint. Runtime adapters must call this before resource allocation.
    ///
    /// # Errors
    ///
    /// Returns an error for an unsupported envelope, forged derived field,
    /// non-canonical capability list, or content fingerprint mismatch.
    pub fn validate(&self) -> Result<(), RuntimeManifestError> {
        if self.schema != RUNTIME_MANIFEST_SCHEMA {
            return Err(RuntimeManifestError::UnsupportedSchema(self.schema.clone()));
        }
        if self.version != RUNTIME_MANIFEST_VERSION {
            return Err(RuntimeManifestError::UnsupportedVersion(self.version));
        }
        if !is_sha256_fingerprint(&self.fingerprint) {
            return Err(RuntimeManifestError::MalformedFingerprint);
        }

        let rebuilt = recompile_attention_state_plan(&self.attention_state_plan)?;
        if rebuilt != self.attention_state_plan {
            return Err(RuntimeManifestError::AttentionStatePlanMismatch);
        }
        validate_token_manager_plan(&self.attention_state_plan, self.token_manager_plan.as_ref())?;

        let requirements = derive_capability_requirements(
            &self.attention_state_plan,
            self.token_manager_plan.as_ref(),
        );
        if self.capability_requirements != requirements {
            return Err(RuntimeManifestError::CapabilityRequirementsMismatch);
        }
        if self.computed_fingerprint()? != self.fingerprint {
            return Err(RuntimeManifestError::FingerprintMismatch);
        }
        Ok(())
    }

    /// Computes the manifest fingerprint over canonical compact JSON. Object
    /// keys are recursively sorted, array order is preserved, and the top-level
    /// `fingerprint` member is omitted rather than encoded as an empty value.
    /// Version 1 uses UTF-8 JSON with no insignificant whitespace, `serde_json`
    /// string escaping, and `serde_json` integer rendering. These rules, together
    /// with the schema's integer-only numeric fields, define the v1 canonical
    /// byte profile used by every consumer.
    ///
    /// # Errors
    ///
    /// Returns an error if the typed manifest cannot be represented as JSON.
    pub fn computed_fingerprint(&self) -> Result<String, RuntimeManifestError> {
        let mut payload = serde_json::to_value(self)?;
        let Some(object) = payload.as_object_mut() else {
            return Err(RuntimeManifestError::NonCanonicalJson);
        };
        object.remove("fingerprint");
        let mut canonical = Vec::new();
        write_canonical_json(&payload, &mut canonical)?;
        Ok(format!("sha256:{:x}", Sha256::digest(canonical)))
    }

    fn validate_serialized_size_with_limit(
        &self,
        maximum: usize,
    ) -> Result<(), RuntimeManifestError> {
        // CLI artifacts use pretty JSON. Bounding that representation also
        // guarantees that the compact representation fits every consumer's
        // identical byte limit.
        let actual = serde_json::to_vec_pretty(self)?.len() + 1;
        if actual > maximum {
            return Err(RuntimeManifestError::ManifestTooLarge { actual, maximum });
        }
        Ok(())
    }
}

fn build_token_manager_plan(
    attention: &CompiledAttentionStatePlan,
) -> Result<Option<RuntimeTokenManagerPlan>, RuntimeManifestError> {
    match attention.token_manager_plan() {
        Ok(input) => {
            let layout = compile_plan(input.clone())?.layout_program()?;
            Ok(Some(RuntimeTokenManagerPlan { input, layout }))
        }
        Err(AttentionStateError::NoTokenState) => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn validate_token_manager_plan(
    attention: &CompiledAttentionStatePlan,
    actual: Option<&RuntimeTokenManagerPlan>,
) -> Result<(), RuntimeManifestError> {
    match (attention.token_manager_plan(), actual) {
        (Err(AttentionStateError::NoTokenState), None) => Ok(()),
        (Err(AttentionStateError::NoTokenState), Some(_)) | (Ok(_), None) => {
            Err(RuntimeManifestError::TokenManagerPresenceMismatch)
        }
        (Err(error), _) => Err(error.into()),
        (Ok(expected_input), Some(actual)) => {
            if expected_input != actual.input {
                return Err(RuntimeManifestError::TokenManagerInputMismatch);
            }
            let expected_layout = compile_plan(actual.input.clone())?.layout_program()?;
            if expected_layout != actual.layout {
                return Err(RuntimeManifestError::TokenManagerLayoutMismatch);
            }
            Ok(())
        }
    }
}

fn recompile_attention_state_plan(
    compiled: &CompiledAttentionStatePlan,
) -> Result<CompiledAttentionStatePlan, RuntimeManifestError> {
    let states = compiled
        .states
        .iter()
        .map(|state| {
            let storage = match &state.backend {
                AttentionStateBackend::TokenSlots {
                    storage,
                    components,
                    retention,
                    window_tokens,
                    ..
                } => match storage {
                    TokenStorageKind::TokenKv => AttentionStateStorage::TokenKv {
                        key_bytes_per_token_per_layer: component_bytes(components, "key")?,
                        value_bytes_per_token_per_layer: component_bytes(components, "value")?,
                        retention: *retention,
                        window_tokens: *window_tokens,
                    },
                    TokenStorageKind::LatentKv => AttentionStateStorage::LatentKv {
                        latent_bytes_per_token_per_layer: component_bytes(components, "latent")?,
                        rope_bytes_per_token_per_layer: component_bytes(components, "rope")?,
                        retention: *retention,
                        window_tokens: *window_tokens,
                    },
                },
                AttentionStateBackend::RecurrentCheckpoints {
                    family,
                    state_bytes_per_layer,
                    checkpoint_slots_per_request,
                    ..
                } => AttentionStateStorage::Recurrent {
                    family: *family,
                    state_bytes_per_layer: *state_bytes_per_layer,
                    checkpoint_slots_per_request: *checkpoint_slots_per_request,
                },
                AttentionStateBackend::ConvolutionRing {
                    state_bytes_per_layer,
                    kernel_width,
                    checkpoint_slots_per_request,
                    ..
                } => AttentionStateStorage::Convolution {
                    state_bytes_per_layer: *state_bytes_per_layer,
                    kernel_width: *kernel_width,
                    checkpoint_slots_per_request: *checkpoint_slots_per_request,
                },
            };
            Ok(AttentionStateSpec {
                name: state.name.clone(),
                layers: state.layers.clone(),
                storage,
            })
        })
        .collect::<Result<Vec<_>, RuntimeManifestError>>()?;
    Ok(compile_attention_state_plan(AttentionStatePlanInput {
        page_tokens: compiled.page_tokens,
        states,
    })?)
}

fn component_bytes(
    components: &[StateComponentGeometry],
    name: &str,
) -> Result<u64, RuntimeManifestError> {
    if components.len() != 2 {
        return Err(RuntimeManifestError::AttentionStatePlanMismatch);
    }
    components
        .iter()
        .find(|component| component.name == name)
        .map(|component| component.bytes_per_token_per_layer)
        .ok_or(RuntimeManifestError::AttentionStatePlanMismatch)
}

fn derive_capability_requirements(
    attention: &CompiledAttentionStatePlan,
    token_manager: Option<&RuntimeTokenManagerPlan>,
) -> Vec<RuntimeCapability> {
    let mut requirements = BTreeSet::new();
    if let Some(token_manager) = token_manager {
        requirements.insert(RuntimeCapability::TokenManager);
        if token_manager
            .input
            .classes
            .iter()
            .any(|class| !class.components.is_empty())
        {
            requirements.insert(RuntimeCapability::TokenComponentGeometry);
        }
        for class in &token_manager.layout.classes {
            match class.address {
                AddressProgram::AppendOnly | AddressProgram::Pinned => {
                    requirements.insert(RuntimeCapability::AppendOnlyAddressing);
                }
                AddressProgram::Periodic { .. } | AddressProgram::PeriodicFrom { .. } => {
                    requirements.insert(RuntimeCapability::PeriodicAddressing);
                }
                AddressProgram::ResettableArena { .. } => {}
            }
            if class.retirement != RetirementProgram::Never {
                requirements.insert(RuntimeCapability::SemanticRetirement);
            }
        }
    }

    let mut has_fixed_state = false;
    for state in &attention.states {
        match state.backend {
            AttentionStateBackend::TokenSlots { .. } => {}
            AttentionStateBackend::RecurrentCheckpoints { .. } => {
                has_fixed_state = true;
                requirements.insert(RuntimeCapability::RecurrentState);
            }
            AttentionStateBackend::ConvolutionRing { .. } => {
                has_fixed_state = true;
                requirements.insert(RuntimeCapability::ConvolutionState);
            }
        }
    }
    if has_fixed_state {
        requirements.insert(RuntimeCapability::FixedStateCheckpoints);
    }
    requirements.into_iter().collect()
}

fn lift_token_plan(input: &KvPlanInput) -> Result<AttentionStatePlanInput, RuntimeManifestError> {
    Ok(AttentionStatePlanInput {
        page_tokens: input.page_tokens,
        states: input
            .classes
            .iter()
            .map(lift_token_class)
            .collect::<Result<Vec<_>, _>>()?,
    })
}

fn lift_token_class(class: &KvClassSpec) -> Result<AttentionStateSpec, RuntimeManifestError> {
    let storage = match class.storage {
        TokenStorageKind::TokenKv => {
            let (key, value) = if class.components.is_empty() {
                let key = class.bytes_per_token_per_layer / 2;
                (key, class.bytes_per_token_per_layer - key)
            } else {
                (
                    token_component_bytes(class, "key")?,
                    token_component_bytes(class, "value")?,
                )
            };
            if key == 0 || value == 0 {
                return Err(RuntimeManifestError::UnsupportedHfTokenClass {
                    class: class.name.clone(),
                });
            }
            AttentionStateStorage::TokenKv {
                key_bytes_per_token_per_layer: key,
                value_bytes_per_token_per_layer: value,
                retention: class.retention,
                window_tokens: class.window_tokens,
            }
        }
        TokenStorageKind::LatentKv => AttentionStateStorage::LatentKv {
            latent_bytes_per_token_per_layer: token_component_bytes(class, "latent")?,
            rope_bytes_per_token_per_layer: token_component_bytes(class, "rope")?,
            retention: class.retention,
            window_tokens: class.window_tokens,
        },
    };
    Ok(AttentionStateSpec {
        name: class.name.clone(),
        layers: class.layers.clone(),
        storage,
    })
}

fn token_component_bytes(class: &KvClassSpec, name: &str) -> Result<u64, RuntimeManifestError> {
    class
        .components
        .iter()
        .find(|component| component.name == name)
        .map(|component| component.bytes_per_token_per_layer)
        .filter(|bytes| *bytes > 0)
        .ok_or_else(|| RuntimeManifestError::UnsupportedHfTokenClass {
            class: class.name.clone(),
        })
}

fn is_not_heterogeneous_profile(error: &HfConfigError) -> bool {
    matches!(
        error,
        HfConfigError::UnsupportedQwenHybridGdnArchitecture { .. }
            | HfConfigError::UnsupportedQwenHybridGdnModelType { .. }
            | HfConfigError::MissingQwenHybridGdnTextConfig
    )
}

fn is_sha256_fingerprint(value: &str) -> bool {
    let Some(digest) = value.strip_prefix("sha256:") else {
        return false;
    };
    digest.len() == 64
        && digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn write_canonical_json(value: &Value, output: &mut Vec<u8>) -> Result<(), serde_json::Error> {
    match value {
        Value::Null => output.extend_from_slice(b"null"),
        Value::Bool(value) => {
            output.extend_from_slice(if *value { &b"true"[..] } else { &b"false"[..] });
        }
        Value::Number(value) => output.extend_from_slice(value.to_string().as_bytes()),
        Value::String(value) => serde_json::to_writer(output, value)?,
        Value::Array(values) => {
            output.push(b'[');
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    output.push(b',');
                }
                write_canonical_json(value, output)?;
            }
            output.push(b']');
        }
        Value::Object(values) => {
            output.push(b'{');
            let mut entries = values.iter().collect::<Vec<_>>();
            entries.sort_unstable_by_key(|(key, _)| *key);
            for (index, (key, value)) in entries.into_iter().enumerate() {
                if index != 0 {
                    output.push(b',');
                }
                serde_json::to_writer(&mut *output, key)?;
                output.push(b':');
                write_canonical_json(value, output)?;
            }
            output.push(b'}');
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention_state::{AttentionStateSpec, RecurrentFamily};
    use crate::plan::RetentionKind;

    fn mixed_input() -> AttentionStatePlanInput {
        AttentionStatePlanInput {
            page_tokens: 16,
            states: vec![
                AttentionStateSpec {
                    name: "full".into(),
                    layers: vec![0],
                    storage: AttentionStateStorage::TokenKv {
                        key_bytes_per_token_per_layer: 256,
                        value_bytes_per_token_per_layer: 256,
                        retention: RetentionKind::Full,
                        window_tokens: None,
                    },
                },
                AttentionStateSpec {
                    name: "local".into(),
                    layers: vec![1],
                    storage: AttentionStateStorage::LatentKv {
                        latent_bytes_per_token_per_layer: 96,
                        rope_bytes_per_token_per_layer: 32,
                        retention: RetentionKind::Sliding,
                        window_tokens: Some(18),
                    },
                },
                AttentionStateSpec {
                    name: "recurrent".into(),
                    layers: vec![2],
                    storage: AttentionStateStorage::Recurrent {
                        family: RecurrentFamily::LinearAttention,
                        state_bytes_per_layer: 4_096,
                        checkpoint_slots_per_request: 2,
                    },
                },
                AttentionStateSpec {
                    name: "convolution".into(),
                    layers: vec![2],
                    storage: AttentionStateStorage::Convolution {
                        state_bytes_per_layer: 2_048,
                        kernel_width: 4,
                        checkpoint_slots_per_request: 2,
                    },
                },
            ],
        }
    }

    #[test]
    fn mixed_manifest_round_trips_with_canonical_capabilities() {
        let manifest = compile_runtime_manifest(mixed_input()).unwrap();
        manifest.validate().unwrap();
        assert_eq!(manifest.schema, RUNTIME_MANIFEST_SCHEMA);
        assert_eq!(manifest.version, RUNTIME_MANIFEST_VERSION);
        assert_eq!(
            manifest.capability_requirements,
            vec![
                RuntimeCapability::AppendOnlyAddressing,
                RuntimeCapability::ConvolutionState,
                RuntimeCapability::FixedStateCheckpoints,
                RuntimeCapability::PeriodicAddressing,
                RuntimeCapability::RecurrentState,
                RuntimeCapability::SemanticRetirement,
                RuntimeCapability::TokenComponentGeometry,
                RuntimeCapability::TokenManager,
            ]
        );
        let token = manifest.token_manager_plan.as_ref().unwrap();
        assert_eq!(token.input.classes.len(), 2);
        assert_eq!(
            token.layout.classes[1].address,
            AddressProgram::Periodic { period_blocks: 3 }
        );

        let encoded = serde_json::to_vec(&manifest).unwrap();
        let decoded = RuntimeManifest::from_json(&encoded).unwrap();
        assert_eq!(decoded, manifest);
    }

    #[test]
    fn fixed_only_manifest_uses_null_token_manager() {
        let manifest = compile_runtime_manifest(AttentionStatePlanInput {
            page_tokens: 16,
            states: vec![AttentionStateSpec {
                name: "state".into(),
                layers: vec![0, 1],
                storage: AttentionStateStorage::Recurrent {
                    family: RecurrentFamily::Mamba,
                    state_bytes_per_layer: 64,
                    checkpoint_slots_per_request: 2,
                },
            }],
        })
        .unwrap();
        assert!(manifest.token_manager_plan.is_none());
        assert_eq!(
            manifest.capability_requirements,
            vec![
                RuntimeCapability::FixedStateCheckpoints,
                RuntimeCapability::RecurrentState,
            ]
        );
        RuntimeManifest::from_json(&serde_json::to_vec(&manifest).unwrap()).unwrap();
    }

    #[test]
    fn fingerprint_is_stable_and_covers_array_order() {
        let manifest = compile_runtime_manifest(mixed_input()).unwrap();
        assert_eq!(
            manifest.fingerprint,
            manifest.computed_fingerprint().unwrap()
        );
        assert_eq!(
            manifest.fingerprint,
            "sha256:b936744c23c8fc874f477900bef47c45c2b027ea7bb175f943438eeacfe76946"
        );

        let mut reordered = manifest.clone();
        reordered.attention_state_plan.states.swap(0, 1);
        assert_ne!(
            reordered.computed_fingerprint().unwrap(),
            manifest.fingerprint
        );
    }

    #[test]
    fn canonical_json_recursively_sorts_object_keys() {
        let value = serde_json::json!({
            "z": {"b": 2, "a": 1},
            "a": ["状态", {"y": false, "x": null}]
        });
        let mut encoded = Vec::new();
        write_canonical_json(&value, &mut encoded).unwrap();
        assert_eq!(
            String::from_utf8(encoded).unwrap(),
            r#"{"a":["状态",{"x":null,"y":false}],"z":{"a":1,"b":2}}"#
        );
    }

    #[test]
    fn canonical_json_interoperability_vector_covers_escaping_and_unicode() {
        let value = serde_json::json!({
            "z": ["状态", "line\nquote\"slash\\", null, true, 42],
            "a": {"é": "值", "\u{0001}": "\u{0008}\u{000c}\n\r\t"}
        });
        let mut encoded = Vec::new();
        write_canonical_json(&value, &mut encoded).unwrap();
        assert_eq!(
            format!("sha256:{:x}", Sha256::digest(&encoded)),
            "sha256:95762173915ab37233eb16469bdb42ad0c2efcc5aad56052a87ec2ec33819b7f"
        );
    }

    #[test]
    fn rejects_manifest_above_the_serialized_size_limit() {
        let oversized = vec![b' '; RUNTIME_MANIFEST_MAX_BYTES + 1];
        assert!(matches!(
            RuntimeManifest::from_json(&oversized),
            Err(RuntimeManifestError::ManifestTooLarge { .. })
        ));
    }

    #[test]
    fn compiled_manifest_must_fit_the_emitted_size_limit() {
        let manifest = compile_runtime_manifest(mixed_input()).unwrap();
        assert!(matches!(
            manifest.validate_serialized_size_with_limit(64),
            Err(RuntimeManifestError::ManifestTooLarge { maximum: 64, .. })
        ));
    }

    #[test]
    fn rejects_tampering_even_with_a_recomputed_fingerprint() {
        let original = compile_runtime_manifest(mixed_input()).unwrap();

        let mut state = original.clone();
        let AttentionStateBackend::TokenSlots {
            page_bytes_per_layer,
            ..
        } = &mut state.attention_state_plan.states[0].backend
        else {
            panic!("fixture begins with token state");
        };
        *page_bytes_per_layer += 1;
        state.fingerprint = state.computed_fingerprint().unwrap();
        assert!(matches!(
            state.validate(),
            Err(RuntimeManifestError::AttentionStatePlanMismatch)
        ));

        let mut input = original.clone();
        input.token_manager_plan.as_mut().unwrap().input.classes[0].name = "forged".into();
        input.fingerprint = input.computed_fingerprint().unwrap();
        assert!(matches!(
            input.validate(),
            Err(RuntimeManifestError::TokenManagerInputMismatch)
        ));

        let mut layout = original.clone();
        layout.token_manager_plan.as_mut().unwrap().layout.classes[0].bytes_per_token_per_layer +=
            1;
        layout.fingerprint = layout.computed_fingerprint().unwrap();
        assert!(matches!(
            layout.validate(),
            Err(RuntimeManifestError::TokenManagerLayoutMismatch)
        ));

        let mut capabilities = original;
        capabilities.capability_requirements.swap(0, 1);
        capabilities.fingerprint = capabilities.computed_fingerprint().unwrap();
        assert!(matches!(
            capabilities.validate(),
            Err(RuntimeManifestError::CapabilityRequirementsMismatch)
        ));
    }

    #[test]
    fn rejects_fingerprint_schema_version_and_section_presence_changes() {
        let original = compile_runtime_manifest(mixed_input()).unwrap();

        let mut malformed = original.clone();
        malformed.fingerprint = "SHA256:1234".into();
        assert!(matches!(
            malformed.validate(),
            Err(RuntimeManifestError::MalformedFingerprint)
        ));

        let mut stale = original.clone();
        stale.fingerprint.replace_range(7..8, "0");
        if stale.fingerprint == original.fingerprint {
            stale.fingerprint.replace_range(7..8, "1");
        }
        assert!(matches!(
            stale.validate(),
            Err(RuntimeManifestError::FingerprintMismatch)
        ));

        let mut schema = original.clone();
        schema.schema.push_str(".v1");
        assert!(matches!(
            schema.validate(),
            Err(RuntimeManifestError::UnsupportedSchema(_))
        ));

        let mut version = original.clone();
        version.version = 2;
        assert!(matches!(
            version.validate(),
            Err(RuntimeManifestError::UnsupportedVersion(2))
        ));

        let mut missing_token_plan = original;
        missing_token_plan.token_manager_plan = None;
        missing_token_plan.fingerprint = missing_token_plan.computed_fingerprint().unwrap();
        assert!(matches!(
            missing_token_plan.validate(),
            Err(RuntimeManifestError::TokenManagerPresenceMismatch)
        ));
    }

    #[test]
    fn rejects_unknown_fields_at_every_manifest_layer() {
        let manifest = compile_runtime_manifest(mixed_input()).unwrap();
        let base = serde_json::to_value(manifest).unwrap();

        let mut top = base.clone();
        top.as_object_mut()
            .unwrap()
            .insert("unknown".into(), true.into());
        assert!(matches!(
            RuntimeManifest::from_json(&serde_json::to_vec(&top).unwrap()),
            Err(RuntimeManifestError::Json(_))
        ));

        let mut address = base.clone();
        address
            .pointer_mut("/token_manager_plan/layout/classes/0/address")
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert("unknown".into(), true.into());
        assert!(matches!(
            RuntimeManifest::from_json(&serde_json::to_vec(&address).unwrap()),
            Err(RuntimeManifestError::Json(_) | RuntimeManifestError::NonCanonicalJson)
        ));

        let mut retirement = base.clone();
        retirement
            .pointer_mut("/token_manager_plan/layout/classes/0/retirement")
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert("unknown".into(), true.into());
        assert!(matches!(
            RuntimeManifest::from_json(&serde_json::to_vec(&retirement).unwrap()),
            Err(RuntimeManifestError::Json(_) | RuntimeManifestError::NonCanonicalJson)
        ));

        let mut backend = base;
        backend
            .pointer_mut("/attention_state_plan/states/2/backend")
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert("unknown".into(), true.into());
        assert!(matches!(
            RuntimeManifest::from_json(&serde_json::to_vec(&backend).unwrap()),
            Err(RuntimeManifestError::Json(_) | RuntimeManifestError::NonCanonicalJson)
        ));
    }

    #[test]
    fn rejects_duplicate_root_and_nested_fields() {
        let manifest = compile_runtime_manifest(mixed_input()).unwrap();
        let encoded = serde_json::to_string(&manifest).unwrap();
        let duplicate_root =
            encoded.replacen("{\"schema\":", "{\"schema\":\"duplicate\",\"schema\":", 1);
        assert!(matches!(
            RuntimeManifest::from_json(duplicate_root.as_bytes()),
            Err(RuntimeManifestError::Json(_))
        ));

        let duplicate_nested = encoded.replacen(
            "\"input\":{\"page_tokens\":",
            "\"input\":{\"page_tokens\":8,\"page_tokens\":",
            1,
        );
        assert!(matches!(
            RuntimeManifest::from_json(duplicate_nested.as_bytes()),
            Err(RuntimeManifestError::Json(_))
        ));
    }
}
