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
/// contracts. Only token-slot backends admit token relocation.
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
                    token_relocatable: true,
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
                    token_relocatable: true,
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
                    token_relocatable: false,
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
                    token_relocatable: false,
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
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::too_many_lines)]
    fn compiles_token_latent_recurrent_and_convolution_backends() {
        let output = compile_attention_state_plan(AttentionStatePlanInput {
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
                    name: "mla".into(),
                    layers: vec![1],
                    storage: AttentionStateStorage::LatentKv {
                        latent_bytes_per_token_per_layer: 1024,
                        rope_bytes_per_token_per_layer: 128,
                        retention: RetentionKind::Full,
                        window_tokens: None,
                    },
                },
                AttentionStateSpec {
                    name: "gdn".into(),
                    layers: vec![2],
                    storage: AttentionStateStorage::Recurrent {
                        family: RecurrentFamily::Gdn,
                        state_bytes_per_layer: 4096,
                        checkpoint_slots_per_request: 2,
                    },
                },
                AttentionStateSpec {
                    name: "shortconv".into(),
                    layers: vec![2],
                    storage: AttentionStateStorage::Convolution {
                        state_bytes_per_layer: 2048,
                        kernel_width: 4,
                        checkpoint_slots_per_request: 2,
                    },
                },
            ],
        })
        .unwrap();
        assert_eq!(output.schema, "orbitkv.attention-state-plan.v1");
        let AttentionStateBackend::TokenSlots {
            components,
            bytes_per_token_per_layer,
            page_bytes_per_layer,
            ..
        } = &output.states[1].backend
        else {
            panic!("MLA state must lower to token slots");
        };
        assert_eq!(*bytes_per_token_per_layer, 1_152);
        assert_eq!(*page_bytes_per_layer, 18_432);
        assert_eq!(
            components,
            &[
                StateComponentGeometry {
                    name: "latent",
                    bytes_per_token_per_layer: 1_024,
                },
                StateComponentGeometry {
                    name: "rope",
                    bytes_per_token_per_layer: 128,
                },
            ]
        );
        assert!(matches!(
            &output.states[1].backend,
            AttentionStateBackend::TokenSlots {
                token_relocatable: true,
                ..
            }
        ));
        assert!(matches!(
            &output.states[2].backend,
            AttentionStateBackend::RecurrentCheckpoints {
                token_relocatable: false,
                ..
            }
        ));
        assert!(matches!(
            &output.states[3].backend,
            AttentionStateBackend::ConvolutionRing {
                token_relocatable: false,
                ..
            }
        ));

        let manager = output.token_manager_plan().unwrap();
        assert_eq!(manager.page_tokens, 16);
        assert_eq!(manager.classes.len(), 2);
        assert_eq!(manager.classes[0].name, "full");
        assert_eq!(manager.classes[0].bytes_per_token_per_layer, 512);
        assert_eq!(manager.classes[1].name, "mla");
        assert_eq!(manager.classes[1].bytes_per_token_per_layer, 1_152);
        assert_eq!(manager.classes[1].layers, vec![1]);
        assert_eq!(manager.classes[1].storage, TokenStorageKind::LatentKv);
        assert_eq!(
            manager.classes[1].components,
            vec![
                TokenComponentSpec {
                    name: "latent".into(),
                    bytes_per_token_per_layer: 1_024,
                },
                TokenComponentSpec {
                    name: "rope".into(),
                    bytes_per_token_per_layer: 128,
                },
            ]
        );
    }

    #[test]
    fn page_geometry_uses_the_declared_page_size() {
        let output = compile_attention_state_plan(AttentionStatePlanInput {
            page_tokens: 7,
            states: vec![AttentionStateSpec {
                name: "full".into(),
                layers: vec![0],
                storage: AttentionStateStorage::TokenKv {
                    key_bytes_per_token_per_layer: 3,
                    value_bytes_per_token_per_layer: 5,
                    retention: RetentionKind::Full,
                    window_tokens: None,
                },
            }],
        })
        .unwrap();
        assert!(matches!(
            output.states[0].backend,
            AttentionStateBackend::TokenSlots {
                bytes_per_token_per_layer: 8,
                page_bytes_per_layer: 56,
                ..
            }
        ));
    }

    #[test]
    fn same_role_overlap_and_single_checkpoint_fail_closed() {
        let duplicated = AttentionStatePlanInput {
            page_tokens: 16,
            states: vec![
                AttentionStateSpec {
                    name: "mha".into(),
                    layers: vec![0],
                    storage: AttentionStateStorage::TokenKv {
                        key_bytes_per_token_per_layer: 1,
                        value_bytes_per_token_per_layer: 1,
                        retention: RetentionKind::Full,
                        window_tokens: None,
                    },
                },
                AttentionStateSpec {
                    name: "mla".into(),
                    layers: vec![0],
                    storage: AttentionStateStorage::LatentKv {
                        latent_bytes_per_token_per_layer: 1,
                        rope_bytes_per_token_per_layer: 1,
                        retention: RetentionKind::Full,
                        window_tokens: None,
                    },
                },
            ],
        };
        assert!(matches!(
            compile_attention_state_plan(duplicated),
            Err(AttentionStateError::OverlappingRole { .. })
        ));
        let recurrent = AttentionStatePlanInput {
            page_tokens: 16,
            states: vec![AttentionStateSpec {
                name: "mamba".into(),
                layers: vec![0],
                storage: AttentionStateStorage::Recurrent {
                    family: RecurrentFamily::Mamba,
                    state_bytes_per_layer: 16,
                    checkpoint_slots_per_request: 1,
                },
            }],
        };
        assert_eq!(
            compile_attention_state_plan(recurrent),
            Err(AttentionStateError::InsufficientCheckpointSlots)
        );
    }

    #[test]
    fn state_only_plan_has_no_token_manager_projection() {
        let output = compile_attention_state_plan(AttentionStatePlanInput {
            page_tokens: 16,
            states: vec![AttentionStateSpec {
                name: "mamba".into(),
                layers: vec![0, 1, 2],
                storage: AttentionStateStorage::Recurrent {
                    family: RecurrentFamily::Mamba,
                    state_bytes_per_layer: 16,
                    checkpoint_slots_per_request: 2,
                },
            }],
        })
        .unwrap();
        assert!(matches!(
            output.states[0].backend,
            AttentionStateBackend::RecurrentCheckpoints {
                checkpoint_bytes_per_request: 96,
                ..
            }
        ));
        assert_eq!(
            output.token_manager_plan(),
            Err(AttentionStateError::NoTokenState)
        );
    }
}
