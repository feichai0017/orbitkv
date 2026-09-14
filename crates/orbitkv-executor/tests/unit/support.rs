use orbitkv::{
    StateClassLayoutFacts, StateComponentFact, StateLayoutFacts, StateStorageFacts,
    StateStorageKind,
    plan::{AddressProgram, BlockDomain, RetentionKind, RetirementProgram},
};

use crate::{AttentionClass, AttentionVisibility, ExecutorPlan, FixedStateClass};

pub(crate) fn executor_plan(
    manifest_fingerprint: &str,
    page_tokens: u32,
    classes: Vec<AttentionClass>,
) -> ExecutorPlan {
    executor_plan_with_fixed_states(manifest_fingerprint, page_tokens, classes, Vec::new())
}

pub(crate) fn executor_plan_with_fixed_states(
    manifest_fingerprint: &str,
    page_tokens: u32,
    classes: Vec<AttentionClass>,
    fixed_states: Vec<FixedStateClass>,
) -> ExecutorPlan {
    let state_classes = classes
        .iter()
        .map(|class| {
            let (retention, window_tokens, address, retirement) = match class.visibility {
                AttentionVisibility::Full => (
                    RetentionKind::Full,
                    None,
                    AddressProgram::AppendOnly,
                    RetirementProgram::Never,
                ),
                AttentionVisibility::Sliding { window_tokens } => {
                    let period_blocks = 1 + (window_tokens - 1).div_ceil(u64::from(page_tokens));
                    (
                        RetentionKind::Sliding,
                        Some(window_tokens),
                        AddressProgram::Periodic { period_blocks },
                        RetirementProgram::BlockEndPlus {
                            offset_tokens: window_tokens - 1,
                        },
                    )
                }
                AttentionVisibility::Chunked { blocks_per_epoch } => (
                    RetentionKind::Chunked,
                    None,
                    AddressProgram::ResettableArena { blocks_per_epoch },
                    RetirementProgram::EpochEnd { blocks_per_epoch },
                ),
            };
            let bytes_per_token_per_layer = class
                .key_bytes_per_token_per_layer
                .checked_add(class.value_bytes_per_token_per_layer)
                .unwrap();
            StateClassLayoutFacts {
                manager_class_id: Some(class.class_id),
                name: class.name.clone(),
                layers: class.layers.clone(),
                storage: StateStorageFacts::TokenSlots {
                    storage: StateStorageKind::TokenKv,
                    components: vec![
                        StateComponentFact {
                            name: "key".into(),
                            bytes_per_token_per_layer: class.key_bytes_per_token_per_layer,
                        },
                        StateComponentFact {
                            name: "value".into(),
                            bytes_per_token_per_layer: class.value_bytes_per_token_per_layer,
                        },
                    ]
                    .into_boxed_slice(),
                    bytes_per_token_per_layer,
                    page_bytes_per_layer: bytes_per_token_per_layer * u64::from(page_tokens),
                },
                retention: Some(retention),
                window_tokens,
                address: Some(address),
                retirement: Some(retirement),
                block_domain: Some(BlockDomain::all()),
            }
        })
        .collect::<Vec<_>>()
        .into_boxed_slice();
    ExecutorPlan {
        manifest_fingerprint: manifest_fingerprint.into(),
        page_tokens,
        state_layout_facts: StateLayoutFacts {
            manifest_fingerprint: manifest_fingerprint.into(),
            page_tokens: u64::from(page_tokens),
            classes: state_classes,
        },
        classes: classes.into_boxed_slice(),
        fixed_states: fixed_states.into_boxed_slice(),
    }
}
