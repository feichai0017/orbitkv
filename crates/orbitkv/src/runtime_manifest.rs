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
    PlanError, RetirementProgram, TokenStorageKind, compile_plan, compile_retention_program,
};
use crate::retention::{KvHeadRange, RetentionError, RetentionProgramInput};

pub const RUNTIME_MANIFEST_SCHEMA: &str = "orbitkv.runtime-manifest";
pub const RUNTIME_MANIFEST_VERSION: u32 = 3;
/// Upper bound for serialized manifests admitted by runtime consumers.
pub const RUNTIME_MANIFEST_MAX_BYTES: usize = 16 * 1024 * 1024;

/// Closed capability vocabulary serialized as the listed snake-case names.
///
/// The compiler emits this vector in lexical order with no duplicates. These
/// values describe what the serialized physical plan requires to be admitted;
/// optional runtime policies and future lifecycle operations are deliberately
/// not inferred from storage geometry alone.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeCapability {
    AppendOnlyAddressing,
    BlockDomainPartitioning,
    ConvolutionState,
    FixedStateCheckpoints,
    KvHeadPartitioning,
    PeriodicAddressing,
    PeriodicFromAddressing,
    PinnedAddressing,
    RecurrentState,
    ResettableArenaAddressing,
    SemanticRetirement,
    TokenComponentGeometry,
    TokenManager,
}

/// The declarative source from which all executable manifest sections are
/// deterministically compiled.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RuntimeManifestSource {
    AttentionState { input: AttentionStatePlanInput },
    RetentionIr { program: RetentionProgramInput },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RuntimeTokenManagerPlan {
    pub layout: LayoutProgram,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RuntimeManifest {
    pub schema: String,
    pub version: u32,
    pub fingerprint: String,
    pub source: RuntimeManifestSource,
    pub token_manager_plan: Option<RuntimeTokenManagerPlan>,
    pub attention_state_plan: Option<CompiledAttentionStatePlan>,
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
            source: wire.source,
            token_manager_plan: wire
                .token_manager_plan
                .map(RuntimeTokenManagerPlanWire::into_runtime)
                .transpose()
                .map_err(serde::de::Error::custom)?,
            attention_state_plan: wire
                .attention_state_plan
                .map(CompiledAttentionStatePlanWire::into_compiled)
                .transpose()
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
    source: RuntimeManifestSource,
    token_manager_plan: Option<RuntimeTokenManagerPlanWire>,
    attention_state_plan: Option<CompiledAttentionStatePlanWire>,
    capability_requirements: Vec<RuntimeCapability>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeTokenManagerPlanWire {
    layout: LayoutProgramWire,
}

impl RuntimeTokenManagerPlanWire {
    fn into_runtime(self) -> Result<RuntimeTokenManagerPlan, RuntimeManifestError> {
        Ok(RuntimeTokenManagerPlan {
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
    },
    RecurrentCheckpoints {
        family: RecurrentFamily,
        state_bytes_per_layer: u64,
        checkpoint_slots_per_request: u32,
        checkpoint_bytes_per_request: u64,
    },
    ConvolutionRing {
        state_bytes_per_layer: u64,
        kernel_width: u32,
        checkpoint_slots_per_request: u32,
        checkpoint_bytes_per_request: u64,
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
            },
            Self::RecurrentCheckpoints {
                family,
                state_bytes_per_layer,
                checkpoint_slots_per_request,
                checkpoint_bytes_per_request,
            } => AttentionStateBackend::RecurrentCheckpoints {
                family,
                state_bytes_per_layer,
                checkpoint_slots_per_request,
                checkpoint_bytes_per_request,
            },
            Self::ConvolutionRing {
                state_bytes_per_layer,
                kernel_width,
                checkpoint_slots_per_request,
                checkpoint_bytes_per_request,
            } => AttentionStateBackend::ConvolutionRing {
                state_bytes_per_layer,
                kernel_width,
                checkpoint_slots_per_request,
                checkpoint_bytes_per_request,
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
    #[error("runtime manifest source and derived sections do not match")]
    SourceMismatch,
    #[error("runtime manifest token-manager plan is missing or unexpectedly present")]
    TokenManagerPresenceMismatch,
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
    Retention(#[from] RetentionError),
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
    let attention_state_plan = compile_attention_state_plan(input.clone())?;
    build_attention_state_manifest(input, attention_state_plan)
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
    let input = attention_state_input(&attention_state_plan)?;
    let rebuilt = compile_attention_state_plan(input.clone())?;
    if rebuilt != attention_state_plan {
        return Err(RuntimeManifestError::AttentionStatePlanMismatch);
    }
    build_attention_state_manifest(input, attention_state_plan)
}

fn build_attention_state_manifest(
    input: AttentionStatePlanInput,
    attention_state_plan: CompiledAttentionStatePlan,
) -> Result<RuntimeManifest, RuntimeManifestError> {
    let token_manager_plan = build_attention_token_manager_plan(&attention_state_plan)?;
    let capability_requirements = derive_attention_capability_requirements(
        &attention_state_plan,
        token_manager_plan.as_ref(),
    );
    let mut manifest = RuntimeManifest {
        schema: RUNTIME_MANIFEST_SCHEMA.to_owned(),
        version: RUNTIME_MANIFEST_VERSION,
        fingerprint: String::new(),
        source: RuntimeManifestSource::AttentionState { input },
        token_manager_plan,
        attention_state_plan: Some(attention_state_plan),
        capability_requirements,
    };
    manifest.fingerprint = manifest.computed_fingerprint()?;
    manifest.validate()?;
    manifest.validate_serialized_size_with_limit(RUNTIME_MANIFEST_MAX_BYTES)?;
    Ok(manifest)
}

/// Compiles Retention IR into the same canonical runtime manifest type.
///
/// # Errors
///
/// Returns an error when retention analysis or physical lowering fails.
pub fn compile_retention_runtime_manifest(
    program: RetentionProgramInput,
) -> Result<RuntimeManifest, RuntimeManifestError> {
    let compiled = compile_retention_program(program.clone())?;
    let layout = compiled.layout_program()?;
    let capability_requirements = derive_retention_capability_requirements(&layout);
    let mut manifest = RuntimeManifest {
        schema: RUNTIME_MANIFEST_SCHEMA.to_owned(),
        version: RUNTIME_MANIFEST_VERSION,
        fingerprint: String::new(),
        source: RuntimeManifestSource::RetentionIr { program },
        token_manager_plan: Some(RuntimeTokenManagerPlan { layout }),
        attention_state_plan: None,
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

        let requirements = match &self.source {
            RuntimeManifestSource::AttentionState { input } => {
                let rebuilt = compile_attention_state_plan(input.clone())?;
                if self.attention_state_plan.as_ref() != Some(&rebuilt) {
                    return Err(RuntimeManifestError::AttentionStatePlanMismatch);
                }
                validate_attention_token_manager_plan(&rebuilt, self.token_manager_plan.as_ref())?;
                derive_attention_capability_requirements(&rebuilt, self.token_manager_plan.as_ref())
            }
            RuntimeManifestSource::RetentionIr { program } => {
                if self.attention_state_plan.is_some() {
                    return Err(RuntimeManifestError::SourceMismatch);
                }
                let Some(manager) = &self.token_manager_plan else {
                    return Err(RuntimeManifestError::TokenManagerPresenceMismatch);
                };
                let expected = compile_retention_program(program.clone())?.layout_program()?;
                if manager.layout != expected {
                    return Err(RuntimeManifestError::TokenManagerLayoutMismatch);
                }
                derive_retention_capability_requirements(&expected)
            }
        };
        if self.capability_requirements != requirements {
            return Err(RuntimeManifestError::CapabilityRequirementsMismatch);
        }
        if self.computed_fingerprint()? != self.fingerprint {
            return Err(RuntimeManifestError::FingerprintMismatch);
        }
        Ok(())
    }

    /// Reconstructs the manager input when the source has an exact
    /// lossless projection. Region/head-specialized Retention IR intentionally
    /// has no lossy fallback.
    ///
    /// # Errors
    /// Returns an error when validation fails or Retention IR cannot be
    /// represented by the manager input.
    pub fn token_manager_input(&self) -> Result<Option<KvPlanInput>, RuntimeManifestError> {
        self.validate()?;
        match &self.source {
            RuntimeManifestSource::AttentionState { input } => Ok(
                match compile_attention_state_plan(input.clone())?.token_manager_plan() {
                    Ok(input) => Some(input),
                    Err(AttentionStateError::NoTokenState) => None,
                    Err(error) => return Err(error.into()),
                },
            ),
            RuntimeManifestSource::RetentionIr { program } => {
                let compiled = compile_retention_program(program.clone())?;
                if compiled
                    .classes
                    .iter()
                    .any(|class| class.kv_head_range.is_some() || !class.block_domain.is_all())
                {
                    return Err(RuntimeManifestError::SourceMismatch);
                }
                Ok(Some(KvPlanInput {
                    page_tokens: compiled.page_tokens,
                    classes: compiled
                        .classes
                        .into_iter()
                        .map(|class| class.spec)
                        .collect(),
                }))
            }
        }
    }

    /// Computes the manifest fingerprint over canonical compact JSON. Object
    /// keys are recursively sorted, array order is preserved, and the top-level
    /// `fingerprint` member is omitted rather than encoded as an empty value.
    /// The canonical profile uses UTF-8 JSON with no insignificant whitespace, `serde_json`
    /// string escaping, and `serde_json` integer rendering. These rules, together
    /// with the schema's integer-only numeric fields, define the canonical
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

fn build_attention_token_manager_plan(
    attention: &CompiledAttentionStatePlan,
) -> Result<Option<RuntimeTokenManagerPlan>, RuntimeManifestError> {
    match attention.token_manager_plan() {
        Ok(input) => {
            let layout = compile_plan(input.clone())?.layout_program()?;
            Ok(Some(RuntimeTokenManagerPlan { layout }))
        }
        Err(AttentionStateError::NoTokenState) => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn validate_attention_token_manager_plan(
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
            let expected_layout = compile_plan(expected_input)?.layout_program()?;
            if expected_layout != actual.layout {
                return Err(RuntimeManifestError::TokenManagerLayoutMismatch);
            }
            Ok(())
        }
    }
}

fn attention_state_input(
    compiled: &CompiledAttentionStatePlan,
) -> Result<AttentionStatePlanInput, RuntimeManifestError> {
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
    Ok(AttentionStatePlanInput {
        page_tokens: compiled.page_tokens,
        states,
    })
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

fn derive_attention_capability_requirements(
    attention: &CompiledAttentionStatePlan,
    token_manager: Option<&RuntimeTokenManagerPlan>,
) -> Vec<RuntimeCapability> {
    let mut requirements = BTreeSet::new();
    if let Some(token_manager) = token_manager {
        requirements.insert(RuntimeCapability::TokenManager);
        if attention.states.iter().any(|state| {
            matches!(
                &state.backend,
                AttentionStateBackend::TokenSlots { components, .. } if !components.is_empty()
            )
        }) {
            requirements.insert(RuntimeCapability::TokenComponentGeometry);
        }
        add_layout_capabilities(&token_manager.layout, &mut requirements);
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

fn derive_retention_capability_requirements(layout: &LayoutProgram) -> Vec<RuntimeCapability> {
    let mut requirements = BTreeSet::from([RuntimeCapability::TokenManager]);
    add_layout_capabilities(layout, &mut requirements);
    requirements.into_iter().collect()
}

fn add_layout_capabilities(layout: &LayoutProgram, requirements: &mut BTreeSet<RuntimeCapability>) {
    for class in &layout.classes {
        if class.kv_head_range.is_some() {
            requirements.insert(RuntimeCapability::KvHeadPartitioning);
        }
        if !class.block_domain.is_all() {
            requirements.insert(RuntimeCapability::BlockDomainPartitioning);
        }
        match class.address {
            AddressProgram::AppendOnly => {
                requirements.insert(RuntimeCapability::AppendOnlyAddressing);
            }
            AddressProgram::Pinned => {
                requirements.insert(RuntimeCapability::PinnedAddressing);
            }
            AddressProgram::Periodic { .. } => {
                requirements.insert(RuntimeCapability::PeriodicAddressing);
            }
            AddressProgram::PeriodicFrom { .. } => {
                requirements.insert(RuntimeCapability::PeriodicFromAddressing);
            }
            AddressProgram::ResettableArena { .. } => {
                requirements.insert(RuntimeCapability::ResettableArenaAddressing);
            }
        }
        if class.retirement != RetirementProgram::Never {
            requirements.insert(RuntimeCapability::SemanticRetirement);
        }
    }
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
    matches!(error, HfConfigError::NotHeterogeneousAttentionState)
}

pub(crate) fn is_sha256_fingerprint(value: &str) -> bool {
    let Some(digest) = value.strip_prefix("sha256:") else {
        return false;
    };
    digest.len() == 64
        && digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub(crate) fn write_canonical_json(
    value: &Value,
    output: &mut Vec<u8>,
) -> Result<(), serde_json::Error> {
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
    include!("runtime_manifest_tests.rs");
}
