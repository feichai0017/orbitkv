use serde::Serialize;
use thiserror::Error;

use crate::{
    AttentionStateBackend, RecurrentFamily, RuntimeManifest, RuntimeManifestError,
    RuntimeManifestSource, StateComponentGeometry, TokenStorageKind,
    plan::{AddressProgram, BlockDomain, RetentionKind, RetirementProgram},
};

/// Backend-neutral persistent-state facts derived from one validated manifest.
///
/// These facts are the stable boundary for compiler backends. They describe
/// semantic and physical choices without exposing Luminal, egglog, CUDA, or
/// device-pointer types. Dynamic facts such as current Prefix ownership and
/// physical contiguity are intentionally supplied by the runtime binding layer.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct StateLayoutFacts {
    pub manifest_fingerprint: String,
    pub page_tokens: u64,
    pub classes: Box<[StateClassLayoutFacts]>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct StateClassLayoutFacts {
    /// Present only for token-addressable classes owned by the KV manager.
    pub manager_class_id: Option<u16>,
    pub name: String,
    pub layers: Box<[u32]>,
    pub storage: StateStorageFacts,
    pub retention: Option<RetentionKind>,
    pub window_tokens: Option<u64>,
    pub address: Option<AddressProgram>,
    pub retirement: Option<RetirementProgram>,
    pub block_domain: Option<BlockDomain>,
    pub legal_layouts: Box<[StateLayoutAlternative]>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StateStorageFacts {
    TokenSlots {
        storage: StateStorageKind,
        components: Box<[StateComponentFact]>,
        bytes_per_token_per_layer: u64,
        page_bytes_per_layer: u64,
        token_relocatable: bool,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StateStorageKind {
    TokenKv,
    LatentKv,
    GenericTokenState,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct StateComponentFact {
    pub name: String,
    pub bytes_per_token_per_layer: u64,
}

/// A semantically legal physical realization. Profitability is deliberately
/// absent: an executor cost profile must choose among these alternatives.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StateLayoutAlternative {
    Compiled,
    /// Pack retained token payload into fewer pages. The runtime must prove
    /// private ownership and completed component copies before publication.
    PackedTokenSlots,
}

#[derive(Debug, Error)]
pub enum StateLayoutFactsError {
    #[error(transparent)]
    Manifest(#[from] RuntimeManifestError),
    #[error("state layout facts do not match the compiled manifest")]
    ManifestMismatch,
    #[error("manager class count exceeds the layout-facts identity range")]
    ClassIdOverflow,
}

impl RuntimeManifest {
    /// Derives the complete static state/layout contract for compiler backends.
    ///
    /// # Errors
    ///
    /// Returns an error if the manifest is invalid or its state and manager
    /// sections cannot be joined exactly.
    pub fn state_layout_facts(&self) -> Result<StateLayoutFacts, StateLayoutFactsError> {
        self.validate()?;
        let mut classes = Vec::new();
        if let Some(manager) = &self.token_manager_plan {
            for (class_id, layout) in manager.layout.classes.iter().enumerate() {
                let manager_class_id =
                    u16::try_from(class_id).map_err(|_| StateLayoutFactsError::ClassIdOverflow)?;
                let state = self.attention_state_plan.as_ref().and_then(|plan| {
                    plan.states
                        .iter()
                        .find(|state| state.name == layout.name && state.layers == layout.layers)
                });
                let (storage, retention, window_tokens) = match (&self.source, state) {
                    (RuntimeManifestSource::AttentionState { .. }, Some(state)) => {
                        token_storage_facts(&state.backend)?
                    }
                    (RuntimeManifestSource::RetentionIr { .. }, None) => {
                        retention_ir_storage_facts(layout, manager.layout.page_tokens)?
                    }
                    _ => return Err(StateLayoutFactsError::ManifestMismatch),
                };
                let mut legal_layouts = vec![StateLayoutAlternative::Compiled];
                if matches!(
                    storage,
                    StateStorageFacts::TokenSlots {
                        token_relocatable: true,
                        ..
                    }
                ) && retention == RetentionKind::Full
                    && matches!(layout.address, AddressProgram::AppendOnly)
                    && layout.retirement == RetirementProgram::Never
                    && layout.block_domain.is_all()
                {
                    legal_layouts.push(StateLayoutAlternative::PackedTokenSlots);
                }
                classes.push(StateClassLayoutFacts {
                    manager_class_id: Some(manager_class_id),
                    name: layout.name.clone(),
                    layers: layout.layers.clone().into_boxed_slice(),
                    storage,
                    retention: Some(retention),
                    window_tokens,
                    address: Some(layout.address.clone()),
                    retirement: Some(layout.retirement.clone()),
                    block_domain: Some(layout.block_domain.clone()),
                    legal_layouts: legal_layouts.into_boxed_slice(),
                });
            }
        }
        if let Some(plan) = &self.attention_state_plan {
            for state in &plan.states {
                if matches!(state.backend, AttentionStateBackend::TokenSlots { .. }) {
                    continue;
                }
                classes.push(fixed_state_facts(state));
            }
        }
        Ok(StateLayoutFacts {
            manifest_fingerprint: self.fingerprint.clone(),
            page_tokens: self.token_manager_plan.as_ref().map_or_else(
                || {
                    self.attention_state_plan
                        .as_ref()
                        .map_or(0, |plan| plan.page_tokens)
                },
                |manager| manager.layout.page_tokens,
            ),
            classes: classes.into_boxed_slice(),
        })
    }
}

fn token_storage_facts(
    backend: &AttentionStateBackend,
) -> Result<(StateStorageFacts, RetentionKind, Option<u64>), StateLayoutFactsError> {
    let AttentionStateBackend::TokenSlots {
        storage,
        components,
        bytes_per_token_per_layer,
        page_bytes_per_layer,
        retention,
        window_tokens,
        token_relocatable,
    } = backend
    else {
        return Err(StateLayoutFactsError::ManifestMismatch);
    };
    Ok((
        StateStorageFacts::TokenSlots {
            storage: match storage {
                TokenStorageKind::TokenKv => StateStorageKind::TokenKv,
                TokenStorageKind::LatentKv => StateStorageKind::LatentKv,
            },
            components: component_facts(components),
            bytes_per_token_per_layer: *bytes_per_token_per_layer,
            page_bytes_per_layer: *page_bytes_per_layer,
            token_relocatable: *token_relocatable,
        },
        *retention,
        *window_tokens,
    ))
}

fn retention_ir_storage_facts(
    layout: &crate::plan::ClassLayoutProgram,
    page_tokens: u64,
) -> Result<(StateStorageFacts, RetentionKind, Option<u64>), StateLayoutFactsError> {
    let retention = match (&layout.address, &layout.retirement) {
        (AddressProgram::AppendOnly | AddressProgram::Pinned, RetirementProgram::Never) => {
            RetentionKind::Full
        }
        (
            AddressProgram::Periodic { .. } | AddressProgram::PeriodicFrom { .. },
            RetirementProgram::BlockEndPlus { .. },
        ) => RetentionKind::Sliding,
        (
            AddressProgram::ResettableArena { blocks_per_epoch },
            RetirementProgram::EpochEnd {
                blocks_per_epoch: retired_blocks,
            },
        ) if blocks_per_epoch == retired_blocks => RetentionKind::Chunked,
        _ => return Err(StateLayoutFactsError::ManifestMismatch),
    };
    let page_bytes_per_layer = layout
        .bytes_per_token_per_layer
        .checked_mul(page_tokens)
        .ok_or(StateLayoutFactsError::ManifestMismatch)?;
    let window_tokens = match layout.retirement {
        RetirementProgram::BlockEndPlus { offset_tokens } => Some(
            offset_tokens
                .checked_add(1)
                .ok_or(StateLayoutFactsError::ManifestMismatch)?,
        ),
        RetirementProgram::Never | RetirementProgram::EpochEnd { .. } => None,
    };
    Ok((
        StateStorageFacts::TokenSlots {
            storage: StateStorageKind::GenericTokenState,
            components: vec![StateComponentFact {
                name: "state".into(),
                bytes_per_token_per_layer: layout.bytes_per_token_per_layer,
            }]
            .into_boxed_slice(),
            bytes_per_token_per_layer: layout.bytes_per_token_per_layer,
            page_bytes_per_layer,
            token_relocatable: false,
        },
        retention,
        window_tokens,
    ))
}

fn component_facts(components: &[StateComponentGeometry]) -> Box<[StateComponentFact]> {
    components
        .iter()
        .map(|component| StateComponentFact {
            name: component.name.to_owned(),
            bytes_per_token_per_layer: component.bytes_per_token_per_layer,
        })
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

fn fixed_state_facts(state: &crate::CompiledAttentionState) -> StateClassLayoutFacts {
    let storage = match &state.backend {
        AttentionStateBackend::RecurrentCheckpoints {
            family,
            state_bytes_per_layer,
            checkpoint_slots_per_request,
            checkpoint_bytes_per_request,
            ..
        } => StateStorageFacts::RecurrentCheckpoints {
            family: *family,
            state_bytes_per_layer: *state_bytes_per_layer,
            checkpoint_slots_per_request: *checkpoint_slots_per_request,
            checkpoint_bytes_per_request: *checkpoint_bytes_per_request,
        },
        AttentionStateBackend::ConvolutionRing {
            state_bytes_per_layer,
            kernel_width,
            checkpoint_slots_per_request,
            checkpoint_bytes_per_request,
            ..
        } => StateStorageFacts::ConvolutionRing {
            state_bytes_per_layer: *state_bytes_per_layer,
            kernel_width: *kernel_width,
            checkpoint_slots_per_request: *checkpoint_slots_per_request,
            checkpoint_bytes_per_request: *checkpoint_bytes_per_request,
        },
        AttentionStateBackend::TokenSlots { .. } => unreachable!("token state filtered above"),
    };
    StateClassLayoutFacts {
        manager_class_id: None,
        name: state.name.clone(),
        layers: state.layers.clone().into_boxed_slice(),
        storage,
        retention: None,
        window_tokens: None,
        address: None,
        retirement: None,
        block_domain: None,
        legal_layouts: vec![StateLayoutAlternative::Compiled].into_boxed_slice(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AttentionStatePlanInput, AttentionStateSpec, AttentionStateStorage,
        compile_runtime_manifest,
    };

    #[test]
    fn facts_preserve_static_geometry_and_only_offer_legal_compaction() {
        let manifest = compile_runtime_manifest(AttentionStatePlanInput {
            page_tokens: 16,
            states: vec![
                AttentionStateSpec {
                    name: "global".into(),
                    layers: vec![0, 2],
                    storage: AttentionStateStorage::TokenKv {
                        key_bytes_per_token_per_layer: 128,
                        value_bytes_per_token_per_layer: 128,
                        retention: RetentionKind::Full,
                        window_tokens: None,
                    },
                },
                AttentionStateSpec {
                    name: "local".into(),
                    layers: vec![1, 3],
                    storage: AttentionStateStorage::TokenKv {
                        key_bytes_per_token_per_layer: 128,
                        value_bytes_per_token_per_layer: 128,
                        retention: RetentionKind::Sliding,
                        window_tokens: Some(64),
                    },
                },
                AttentionStateSpec {
                    name: "recurrent".into(),
                    layers: vec![4],
                    storage: AttentionStateStorage::Recurrent {
                        family: RecurrentFamily::Gdn,
                        state_bytes_per_layer: 256,
                        checkpoint_slots_per_request: 2,
                    },
                },
            ],
        })
        .unwrap();

        let facts = manifest.state_layout_facts().unwrap();
        assert_eq!(facts.manifest_fingerprint, manifest.fingerprint);
        assert_eq!(facts.page_tokens, 16);
        assert_eq!(facts.classes.len(), 3);
        assert_eq!(facts.classes[0].manager_class_id, Some(0));
        assert_eq!(
            facts.classes[0].legal_layouts.as_ref(),
            [
                StateLayoutAlternative::Compiled,
                StateLayoutAlternative::PackedTokenSlots,
            ]
        );
        assert_eq!(facts.classes[1].manager_class_id, Some(1));
        assert_eq!(facts.classes[1].window_tokens, Some(64));
        assert_eq!(
            facts.classes[1].legal_layouts.as_ref(),
            [StateLayoutAlternative::Compiled]
        );
        assert_eq!(facts.classes[2].manager_class_id, None);
        assert!(matches!(
            facts.classes[2].storage,
            StateStorageFacts::RecurrentCheckpoints {
                family: RecurrentFamily::Gdn,
                ..
            }
        ));
    }
}
