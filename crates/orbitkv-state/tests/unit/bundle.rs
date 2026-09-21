use super::*;
use crate::{StateDType, StateFormat, StateLayout, TokenRange};

fn descriptor(component: StateComponent) -> StateDescriptor {
    StateDescriptor {
        content: [1; 32],
        span: TokenRange::new(0, 16).unwrap(),
        component,
        format: StateFormat {
            model: [2; 32],
            implementation: [3; 32],
            dtype: StateDType::Bf16,
            layout: StateLayout::MhaPageFirst,
            block_tokens: 16,
            tensor_parallel_size: 1,
            pipeline_parallel_size: 1,
            context_parallel_size: 1,
        },
    }
}

#[test]
fn bundle_tracks_presence_of_every_declared_component() {
    let attention = StateComponent::AttentionKv;
    let recurrent = StateComponent::RecurrentCheckpoint;
    let mut bundle = StateBundle {
        boundary: 16,
        components: vec![
            BundleComponent {
                descriptor: descriptor(attention.clone()),
                available: true,
            },
            BundleComponent {
                descriptor: descriptor(recurrent.clone()),
                available: false,
            },
        ],
        recovery: RecoveryContract::all([attention, recurrent]),
    };

    assert!(!bundle.has_required_components());
    bundle.components[1].available = true;
    assert!(bundle.has_required_components());
}
