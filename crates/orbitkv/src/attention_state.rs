use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::plan::{KvClassSpec, KvPlanInput, RetentionKind, TokenComponentSpec, TokenStorageKind};

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RecurrentFamily {
    Mamba,
    Gdn,
    Kda,
    LinearAttention,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum AttentionStateStorage {
    TokenKv {
        key_bytes_per_token_per_layer: u64,
        value_bytes_per_token_per_layer: u64,
        retention: RetentionKind,
        #[serde(default)]
        window_tokens: Option<u64>,
    },
    LatentKv {
        latent_bytes_per_token_per_layer: u64,
        rope_bytes_per_token_per_layer: u64,
        retention: RetentionKind,
        #[serde(default)]
        window_tokens: Option<u64>,
    },
    Recurrent {
        family: RecurrentFamily,
        state_bytes_per_layer: u64,
        checkpoint_slots_per_request: u32,
    },
    Convolution {
        state_bytes_per_layer: u64,
        kernel_width: u32,
        checkpoint_slots_per_request: u32,
    },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AttentionStateSpec {
    pub name: String,
    pub layers: Vec<u32>,
    pub storage: AttentionStateStorage,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AttentionStatePlanInput {
    pub page_tokens: u64,
    pub states: Vec<AttentionStateSpec>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct StateComponentGeometry {
    pub name: &'static str,
    pub bytes_per_token_per_layer: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AttentionStateBackend {
    TokenSlots {
        storage: TokenStorageKind,
        components: Vec<StateComponentGeometry>,
        bytes_per_token_per_layer: u64,
        page_bytes_per_layer: u64,
        retention: RetentionKind,
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

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CompiledAttentionState {
    pub name: String,
    pub layers: Vec<u32>,
    pub backend: AttentionStateBackend,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CompiledAttentionStatePlan {
    pub schema: &'static str,
    pub page_tokens: u64,
    pub states: Vec<CompiledAttentionState>,
}

impl CompiledAttentionStatePlan {
    /// Projects token-addressable state into the canonical manager input.
    ///
    /// Recurrent and convolution state remain owned by their checkpoint
    /// backends and are deliberately absent from the returned token plan.
    /// Component geometry remains available on each compiled token backend so
    /// an engine can copy `MLA` latent and `RoPE` payloads independently.
    ///
    /// # Errors
    ///
    /// Returns [`AttentionStateError::NoTokenState`] when the heterogeneous
    /// plan contains only fixed-width recurrent or convolution state.
    pub fn token_manager_plan(&self) -> Result<KvPlanInput, AttentionStateError> {
        let classes = self
            .states
            .iter()
            .filter_map(|state| match &state.backend {
                AttentionStateBackend::TokenSlots {
                    storage,
                    components,
                    bytes_per_token_per_layer,
                    retention,
                    window_tokens,
                    ..
                } => Some(KvClassSpec {
                    name: state.name.clone(),
                    layers: state.layers.clone(),
                    retention: *retention,
                    bytes_per_token_per_layer: *bytes_per_token_per_layer,
                    window_tokens: *window_tokens,
                    storage: *storage,
                    components: components
                        .iter()
                        .map(|component| TokenComponentSpec {
                            name: component.name.to_owned(),
                            bytes_per_token_per_layer: component.bytes_per_token_per_layer,
                        })
                        .collect(),
                }),
                AttentionStateBackend::RecurrentCheckpoints { .. }
                | AttentionStateBackend::ConvolutionRing { .. } => None,
            })
            .collect::<Vec<_>>();
        if classes.is_empty() {
            return Err(AttentionStateError::NoTokenState);
        }
        Ok(KvPlanInput {
            page_tokens: self.page_tokens,
            classes,
        })
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum AttentionStateError {
    #[error("attention-state page_tokens must be positive")]
    ZeroPageTokens,
    #[error("attention-state plan must not be empty")]
    EmptyPlan,
    #[error("attention-state name must not be empty")]
    EmptyName,
    #[error("attention-state name {0:?} is duplicated")]
    DuplicateName(String),
    #[error("attention-state {0:?} must own at least one layer")]
    EmptyLayers(String),
    #[error("attention-state {state:?} duplicates layer {layer}")]
    DuplicateLayer { state: String, layer: u32 },
    #[error("layer {layer} has multiple {role} state owners")]
    OverlappingRole { layer: u32, role: &'static str },
    #[error("attention-state geometry must be positive")]
    ZeroGeometry,
    #[error("token state retention/window contract is invalid")]
    InvalidRetention,
    #[error("generation-checked state needs at least two checkpoint slots")]
    InsufficientCheckpointSlots,
    #[error("integer overflow while compiling attention state")]
    ArithmeticOverflow,
    #[error("attention-state plan has no token-addressable state")]
    NoTokenState,
}

/// Compiles heterogeneous attention state into backend-specific ownership
/// contracts. Token-slot and fixed-state backends retain separate lifecycle
/// contracts.
///
/// # Errors
///
/// Rejects missing geometry, ambiguous same-role layer ownership, invalid
/// retention windows, and state backends without double-buffer headroom.
pub fn compile_attention_state_plan(
    input: AttentionStatePlanInput,
) -> Result<CompiledAttentionStatePlan, AttentionStateError> {
    if input.page_tokens == 0 {
        return Err(AttentionStateError::ZeroPageTokens);
    }
    if input.states.is_empty() {
        return Err(AttentionStateError::EmptyPlan);
    }
    let mut names = BTreeSet::new();
    let mut roles = BTreeMap::<(u32, &'static str), String>::new();
    let mut states = Vec::with_capacity(input.states.len());
    for state in input.states {
        validate_identity(&state, &mut names)?;
        let layer_count = u64::try_from(state.layers.len())
            .map_err(|_| AttentionStateError::ArithmeticOverflow)?;
        let (role, backend) = compile_storage(&state.storage, input.page_tokens, layer_count)?;
        for &layer in &state.layers {
            if roles.insert((layer, role), state.name.clone()).is_some() {
                return Err(AttentionStateError::OverlappingRole { layer, role });
            }
        }
        states.push(CompiledAttentionState {
            name: state.name,
            layers: state.layers,
            backend,
        });
    }
    Ok(CompiledAttentionStatePlan {
        schema: "orbitkv.attention-state-plan.v1",
        page_tokens: input.page_tokens,
        states,
    })
}

/// Compiles heterogeneous state and extracts the token-addressable manager
/// classes. Fixed-width recurrent and convolution state are not projected.
///
/// # Errors
///
/// Returns the heterogeneous compiler error, including
/// [`AttentionStateError::NoTokenState`] for a state-only model.
pub fn compile_attention_state_manager_plan(
    input: AttentionStatePlanInput,
) -> Result<KvPlanInput, AttentionStateError> {
    compile_attention_state_plan(input)?.token_manager_plan()
}

fn validate_identity(
    state: &AttentionStateSpec,
    names: &mut BTreeSet<String>,
) -> Result<(), AttentionStateError> {
    if state.name.is_empty() {
        return Err(AttentionStateError::EmptyName);
    }
    if !names.insert(state.name.clone()) {
        return Err(AttentionStateError::DuplicateName(state.name.clone()));
    }
    if state.layers.is_empty() {
        return Err(AttentionStateError::EmptyLayers(state.name.clone()));
    }
    let unique = state.layers.iter().copied().collect::<BTreeSet<_>>();
    if unique.len() != state.layers.len() {
        let layer = state
            .layers
            .iter()
            .copied()
            .find(|layer| state.layers.iter().filter(|value| *value == layer).count() > 1)
            .expect("duplicate layer exists");
        return Err(AttentionStateError::DuplicateLayer {
            state: state.name.clone(),
            layer,
        });
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn compile_storage(
    storage: &AttentionStateStorage,
    page_tokens: u64,
    layer_count: u64,
) -> Result<(&'static str, AttentionStateBackend), AttentionStateError> {
    match storage {
        AttentionStateStorage::TokenKv {
            key_bytes_per_token_per_layer,
            value_bytes_per_token_per_layer,
            retention,
            window_tokens,
        } => {
            validate_token_geometry(
                &[
                    *key_bytes_per_token_per_layer,
                    *value_bytes_per_token_per_layer,
                ],
                *retention,
                *window_tokens,
            )?;
            let bytes_per_token_per_layer = key_bytes_per_token_per_layer
                .checked_add(*value_bytes_per_token_per_layer)
                .ok_or(AttentionStateError::ArithmeticOverflow)?;
            Ok((
                "token_addressable",
                AttentionStateBackend::TokenSlots {
                    storage: TokenStorageKind::TokenKv,
                    components: vec![
                        StateComponentGeometry {
                            name: "key",
                            bytes_per_token_per_layer: *key_bytes_per_token_per_layer,
                        },
                        StateComponentGeometry {
                            name: "value",
                            bytes_per_token_per_layer: *value_bytes_per_token_per_layer,
                        },
                    ],
                    bytes_per_token_per_layer,
                    page_bytes_per_layer: bytes_per_token_per_layer
                        .checked_mul(page_tokens)
                        .ok_or(AttentionStateError::ArithmeticOverflow)?,
                    retention: *retention,
                    window_tokens: *window_tokens,
                },
            ))
        }
        AttentionStateStorage::LatentKv {
            latent_bytes_per_token_per_layer,
            rope_bytes_per_token_per_layer,
            retention,
            window_tokens,
        } => {
            validate_token_geometry(
                &[
                    *latent_bytes_per_token_per_layer,
                    *rope_bytes_per_token_per_layer,
                ],
                *retention,
                *window_tokens,
            )?;
            let bytes_per_token_per_layer = latent_bytes_per_token_per_layer
                .checked_add(*rope_bytes_per_token_per_layer)
                .ok_or(AttentionStateError::ArithmeticOverflow)?;
            Ok((
                "token_addressable",
                AttentionStateBackend::TokenSlots {
                    storage: TokenStorageKind::LatentKv,
                    components: vec![
                        StateComponentGeometry {
                            name: "latent",
                            bytes_per_token_per_layer: *latent_bytes_per_token_per_layer,
                        },
                        StateComponentGeometry {
                            name: "rope",
                            bytes_per_token_per_layer: *rope_bytes_per_token_per_layer,
                        },
                    ],
                    bytes_per_token_per_layer,
                    page_bytes_per_layer: bytes_per_token_per_layer
                        .checked_mul(page_tokens)
                        .ok_or(AttentionStateError::ArithmeticOverflow)?,
                    retention: *retention,
                    window_tokens: *window_tokens,
                },
            ))
        }
        AttentionStateStorage::Recurrent {
            family,
            state_bytes_per_layer,
            checkpoint_slots_per_request,
        } => {
            validate_checkpoint_geometry(*state_bytes_per_layer, *checkpoint_slots_per_request)?;
            let checkpoint_bytes_per_request = state_bytes_per_layer
                .checked_mul(layer_count)
                .ok_or(AttentionStateError::ArithmeticOverflow)?
                .checked_mul(u64::from(*checkpoint_slots_per_request))
                .ok_or(AttentionStateError::ArithmeticOverflow)?;
            Ok((
                "recurrent",
                AttentionStateBackend::RecurrentCheckpoints {
                    family: *family,
                    state_bytes_per_layer: *state_bytes_per_layer,
                    checkpoint_slots_per_request: *checkpoint_slots_per_request,
                    checkpoint_bytes_per_request,
                },
            ))
        }
        AttentionStateStorage::Convolution {
            state_bytes_per_layer,
            kernel_width,
            checkpoint_slots_per_request,
        } => {
            if *kernel_width == 0 {
                return Err(AttentionStateError::ZeroGeometry);
            }
            validate_checkpoint_geometry(*state_bytes_per_layer, *checkpoint_slots_per_request)?;
            let checkpoint_bytes_per_request = state_bytes_per_layer
                .checked_mul(layer_count)
                .ok_or(AttentionStateError::ArithmeticOverflow)?
                .checked_mul(u64::from(*checkpoint_slots_per_request))
                .ok_or(AttentionStateError::ArithmeticOverflow)?;
            Ok((
                "convolution",
                AttentionStateBackend::ConvolutionRing {
                    state_bytes_per_layer: *state_bytes_per_layer,
                    kernel_width: *kernel_width,
                    checkpoint_slots_per_request: *checkpoint_slots_per_request,
                    checkpoint_bytes_per_request,
                },
            ))
        }
    }
}

fn validate_token_geometry(
    components: &[u64],
    retention: RetentionKind,
    window_tokens: Option<u64>,
) -> Result<(), AttentionStateError> {
    if components.contains(&0) {
        return Err(AttentionStateError::ZeroGeometry);
    }
    match retention {
        RetentionKind::Full if window_tokens.is_none() => Ok(()),
        RetentionKind::Sliding if window_tokens.is_some_and(|window| window > 0) => Ok(()),
        _ => Err(AttentionStateError::InvalidRetention),
    }
}

fn validate_checkpoint_geometry(bytes: u64, slots: u32) -> Result<(), AttentionStateError> {
    if bytes == 0 {
        return Err(AttentionStateError::ZeroGeometry);
    }
    if slots < 2 {
        return Err(AttentionStateError::InsufficientCheckpointSlots);
    }
    Ok(())
}

#[cfg(test)]
#[path = "../tests/unit/attention_state/mod.rs"]
mod tests;
