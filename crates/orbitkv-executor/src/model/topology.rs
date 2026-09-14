//! Joint decoder-layer ownership derived from model semantics and `OrbitKV` state classes.

use std::collections::BTreeSet;

use super::{DecoderConfig, DecoderError, DecoderLayerKind, GatedDeltaConfig};
use crate::{AttentionVisibility, ExecutorPlan, FixedStateStorage};

/// State contract selected for one decoder layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum DecoderLayerState {
    TokenKv {
        class_id: u16,
    },
    GatedDelta {
        recurrent_state_id: u16,
        convolution_state_id: u16,
        geometry: GatedDeltaConfig,
    },
}

/// Complete, uniquely-owned persistent-state topology for a decoder graph.
pub(super) struct DecoderTopology {
    layers: Box<[DecoderLayerState]>,
}

impl DecoderTopology {
    /// Compiles model layer semantics and manager-authored state classes into
    /// one exact layer ownership map.
    pub(super) fn compile(
        config: &DecoderConfig,
        plan: &ExecutorPlan,
    ) -> Result<Self, DecoderError> {
        let declared = config.layer_kinds.as_deref();
        if declared.is_some_and(|layers| layers.len() != config.layers) {
            return Err(DecoderError::UnsupportedPlan);
        }
        let token_owners = token_owners(config, plan, declared)?;
        let fixed = fixed_state_owners(config, plan, declared)?;
        let layers = (0..config.layers)
            .map(|layer| match declared.and_then(|kinds| kinds.get(layer)) {
                Some(DecoderLayerKind::Linear) => fixed.ok_or(DecoderError::UnsupportedPlan),
                Some(DecoderLayerKind::Full | DecoderLayerKind::Sliding) | None => token_owners
                    .get(layer)
                    .and_then(|owner| *owner)
                    .map(|class_id| DecoderLayerState::TokenKv { class_id })
                    .ok_or(DecoderError::UnsupportedPlan),
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            layers: layers.into_boxed_slice(),
        })
    }

    pub(super) fn has_fixed_state(&self) -> bool {
        self.layers
            .iter()
            .any(|layer| matches!(layer, DecoderLayerState::GatedDelta { .. }))
    }

    pub(super) fn gated_delta_state(&self) -> Option<(u16, u16, GatedDeltaConfig)> {
        self.layers.iter().find_map(|layer| match layer {
            DecoderLayerState::GatedDelta {
                recurrent_state_id,
                convolution_state_id,
                geometry,
            } => Some((*recurrent_state_id, *convolution_state_id, *geometry)),
            DecoderLayerState::TokenKv { .. } => None,
        })
    }

    pub(super) fn token_layers(&self) -> BTreeSet<u32> {
        self.layers
            .iter()
            .enumerate()
            .filter_map(|(layer, state)| {
                matches!(state, DecoderLayerState::TokenKv { .. })
                    .then(|| u32::try_from(layer).ok())
                    .flatten()
            })
            .collect()
    }

    pub(super) fn layer(&self, layer: usize) -> Option<DecoderLayerState> {
        self.layers.get(layer).copied()
    }
}

fn token_owners(
    config: &DecoderConfig,
    plan: &ExecutorPlan,
    declared: Option<&[DecoderLayerKind]>,
) -> Result<Vec<Option<u16>>, DecoderError> {
    let mut owners = vec![None; config.layers];
    for class in &plan.classes {
        if class.layers.is_empty() {
            return Err(DecoderError::UnsupportedPlan);
        }
        for &layer in &class.layers {
            let index = usize::try_from(layer).map_err(|_| DecoderError::UnsupportedPlan)?;
            let owner = owners.get_mut(index).ok_or(DecoderError::UnsupportedPlan)?;
            if owner.replace(class.class_id).is_some()
                || !visibility_matches(
                    declared.and_then(|layers| layers.get(index)),
                    class.visibility,
                )
            {
                return Err(DecoderError::UnsupportedPlan);
            }
        }
    }
    for (layer, owner) in owners.iter().enumerate() {
        let stateful = declared
            .and_then(|layers| layers.get(layer))
            .is_some_and(|kind| *kind == DecoderLayerKind::Linear);
        if stateful == owner.is_some() {
            return Err(DecoderError::UnsupportedPlan);
        }
    }
    Ok(owners)
}

fn visibility_matches(
    declared: Option<&DecoderLayerKind>,
    visibility: AttentionVisibility,
) -> bool {
    match declared {
        None => true,
        Some(DecoderLayerKind::Full) => visibility == AttentionVisibility::Full,
        Some(DecoderLayerKind::Sliding) => {
            matches!(visibility, AttentionVisibility::Sliding { .. })
        }
        Some(DecoderLayerKind::Linear) => false,
    }
}

fn fixed_state_owners(
    config: &DecoderConfig,
    plan: &ExecutorPlan,
    declared: Option<&[DecoderLayerKind]>,
) -> Result<Option<DecoderLayerState>, DecoderError> {
    let stateful_layers = declared
        .unwrap_or_default()
        .iter()
        .enumerate()
        .filter_map(|(layer, kind)| {
            (*kind == DecoderLayerKind::Linear)
                .then(|| u32::try_from(layer).ok())
                .flatten()
        })
        .collect::<BTreeSet<_>>();
    if stateful_layers.is_empty() {
        return if plan.fixed_states.is_empty() && config.gated_delta.is_none() {
            Ok(None)
        } else {
            Err(DecoderError::UnsupportedPlan)
        };
    }
    let geometry = config.gated_delta.ok_or(DecoderError::UnsupportedPlan)?;
    let recurrent_bytes = geometry
        .recurrent_bytes()
        .ok_or(DecoderError::UnsupportedPlan)?;
    let convolution_bytes = geometry
        .convolution_bytes()
        .ok_or(DecoderError::UnsupportedPlan)?;
    let mut recurrent = None;
    let mut convolution = None;
    for state in &plan.fixed_states {
        if state.layers.iter().copied().collect::<BTreeSet<_>>() != stateful_layers {
            return Err(DecoderError::UnsupportedPlan);
        }
        match state.storage {
            FixedStateStorage::Recurrent {
                family: orbitkv::RecurrentFamily::Gdn,
                bytes_per_layer,
                ..
            } if bytes_per_layer == recurrent_bytes
                && recurrent.replace(state.state_id).is_none() => {}
            FixedStateStorage::Convolution {
                bytes_per_layer,
                kernel_width,
                ..
            } if bytes_per_layer == convolution_bytes
                && usize::try_from(kernel_width).ok()
                    == Some(geometry.convolution_kernel_width)
                && convolution.replace(state.state_id).is_none() => {}
            _ => return Err(DecoderError::UnsupportedPlan),
        }
    }
    match (recurrent, convolution) {
        (Some(recurrent_state_id), Some(convolution_state_id)) if plan.fixed_states.len() == 2 => {
            Ok(Some(DecoderLayerState::GatedDelta {
                recurrent_state_id,
                convolution_state_id,
                geometry,
            }))
        }
        _ => Err(DecoderError::UnsupportedPlan),
    }
}

#[cfg(test)]
#[path = "../../tests/unit/model/topology/mod.rs"]
mod tests;
