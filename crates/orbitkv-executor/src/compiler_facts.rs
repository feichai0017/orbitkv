use std::fmt::Write;

use orbitkv::{
    StateClassLayoutFacts, StateLayoutAlternative, StateLayoutFacts, StateStorageFacts,
    plan::{AddressProgram, RetentionKind},
};
use sha2::{Digest, Sha256};

use crate::{ExecutorArena, ExecutorError, ExecutorPlan};

/// `OrbitKV` state facts lowered into Luminal's compile-time vocabulary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LuminalCompilerFacts {
    manifest_fingerprint: String,
    digest: String,
    classes: Box<[LuminalStateClassFacts]>,
    egglog: String,
}

/// One manager-owned state class plus its stable runtime arena binding.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LuminalStateClassFacts {
    pub state: StateClassLayoutFacts,
    pub backend_base_index: u64,
    pub page_count: u32,
    pub address_stable: bool,
}

impl LuminalCompilerFacts {
    #[must_use]
    pub fn manifest_fingerprint(&self) -> &str {
        &self.manifest_fingerprint
    }

    #[must_use]
    pub fn digest(&self) -> &str {
        &self.digest
    }

    #[must_use]
    pub fn classes(&self) -> &[LuminalStateClassFacts] {
        &self.classes
    }

    #[must_use]
    pub fn egglog(&self) -> &str {
        &self.egglog
    }
}

impl ExecutorPlan {
    /// Binds static state/layout facts to stable executor arenas and lowers
    /// them into declarations consumed during Luminal e-graph construction.
    ///
    /// # Errors
    ///
    /// Rejects missing, reordered, or geometrically inconsistent class/arena
    /// bindings before graph compilation.
    pub fn luminal_compiler_facts(
        &self,
        arenas: &[ExecutorArena],
    ) -> Result<LuminalCompilerFacts, ExecutorError> {
        crate::validate_arenas(arenas, self.classes.len())?;
        let state_layout_facts = &self.state_layout_facts;
        if state_layout_facts.manifest_fingerprint != self.manifest_fingerprint
            || state_layout_facts.page_tokens != u64::from(self.page_tokens)
        {
            return Err(ExecutorError::CompilerFactsMismatch);
        }
        let token_states = state_layout_facts
            .classes
            .iter()
            .filter(|state| state.manager_class_id.is_some())
            .collect::<Vec<_>>();
        if token_states.len() != self.classes.len() {
            return Err(ExecutorError::CompilerFactsMismatch);
        }

        let classes = token_states
            .into_iter()
            .zip(&self.classes)
            .zip(arenas)
            .map(|((state, class), arena)| {
                if state.manager_class_id != Some(class.class_id)
                    || state.name != class.name
                    || state.layers.as_ref() != class.layers.as_ref()
                    || arena.class_id != class.class_id
                {
                    return Err(ExecutorError::CompilerFactsMismatch);
                }
                Ok(LuminalStateClassFacts {
                    state: state.clone(),
                    backend_base_index: arena.backend_base_index,
                    page_count: arena.page_count,
                    address_stable: true,
                })
            })
            .collect::<Result<Vec<_>, ExecutorError>>()?
            .into_boxed_slice();
        let egglog = lower_egglog(state_layout_facts, &classes)?;
        let digest = format!("sha256:{:x}", Sha256::digest(egglog.as_bytes()));
        Ok(LuminalCompilerFacts {
            manifest_fingerprint: self.manifest_fingerprint.clone(),
            digest,
            classes,
            egglog,
        })
    }
}

fn lower_egglog(
    facts: &StateLayoutFacts,
    classes: &[LuminalStateClassFacts],
) -> Result<String, ExecutorError> {
    let mut output = String::from(
        r"(relation persistent-state-manifest (String))
(relation persistent-state-class (i64))
(relation persistent-state-name (i64 String))
(relation persistent-state-page-tokens (i64 i64))
(relation persistent-state-layer-count (i64 i64))
(relation persistent-state-token-bytes (i64 i64))
(relation persistent-state-page-bytes (i64 i64))
(relation persistent-state-storage-token-kv (i64))
(relation persistent-state-storage-latent-kv (i64))
(relation persistent-state-storage-generic-token-state (i64))
(relation persistent-state-component-bytes (i64 String i64))
(relation persistent-state-layer (i64 i64))
; The final block-domain value is -1 when the domain is unbounded.
(relation persistent-state-block-domain (i64 i64 i64))
(relation persistent-state-arena (i64 i64 i64))
(relation persistent-state-address-stable (i64))
(relation persistent-state-token-relocatable (i64))
(relation persistent-state-retention-full (i64))
(relation persistent-state-retention-sliding (i64 i64))
(relation persistent-state-retention-chunked (i64 i64))
(relation persistent-state-retirement-never (i64))
(relation persistent-state-retirement-block-end-plus (i64 i64))
(relation persistent-state-retirement-epoch-end (i64 i64))
(relation persistent-state-address-append-only (i64))
(relation persistent-state-address-pinned (i64))
(relation persistent-state-address-periodic (i64 i64 i64))
(relation persistent-state-address-resettable (i64 i64))
(relation persistent-state-layout-compiled (i64))
(relation persistent-state-layout-packed-token-slots (i64))
",
    );
    writeln!(
        output,
        "(persistent-state-manifest {})",
        egglog_string(&facts.manifest_fingerprint)?
    )
    .expect("writing to String cannot fail");
    for class in classes {
        write_class_facts(&mut output, class)?;
    }
    Ok(output)
}

fn write_class_facts(
    output: &mut String,
    class: &LuminalStateClassFacts,
) -> Result<(), ExecutorError> {
    let class_id = class
        .state
        .manager_class_id
        .ok_or(ExecutorError::CompilerFactsMismatch)?;
    let class_id = i64::from(class_id);
    writeln!(output, "(persistent-state-class {class_id})").expect("writing to String cannot fail");
    writeln!(
        output,
        "(persistent-state-name {class_id} {})",
        egglog_string(&class.state.name)?
    )
    .expect("writing to String cannot fail");
    write_integer_fact(
        output,
        "persistent-state-page-tokens",
        class_id,
        class_page_tokens(class)?,
    )?;
    write_integer_fact(
        output,
        "persistent-state-layer-count",
        class_id,
        u64::try_from(class.state.layers.len())
            .map_err(|_| ExecutorError::CompilerFactsMismatch)?,
    )?;
    for &layer in &class.state.layers {
        write_integer_fact(output, "persistent-state-layer", class_id, u64::from(layer))?;
    }
    let domain = class
        .state
        .block_domain
        .as_ref()
        .ok_or(ExecutorError::CompilerFactsMismatch)?;
    let domain_start = to_egglog_i64(domain.start_block)?;
    let domain_end = domain
        .end_block_exclusive
        .map(to_egglog_i64)
        .transpose()?
        .unwrap_or(-1);
    writeln!(
        output,
        "(persistent-state-block-domain {class_id} {domain_start} {domain_end})"
    )
    .expect("writing to String cannot fail");
    let backend_base_index = to_egglog_i64(class.backend_base_index)?;
    let page_count = i64::from(class.page_count);
    writeln!(
        output,
        "(persistent-state-arena {class_id} {backend_base_index} {page_count})"
    )
    .expect("writing to String cannot fail");
    if class.address_stable {
        writeln!(output, "(persistent-state-address-stable {class_id})")
            .expect("writing to String cannot fail");
    }
    write_storage_facts(output, class_id, &class.state.storage)?;
    write_retention_facts(output, class_id, class)?;
    write_address_facts(output, class_id, class)?;
    write_retirement_facts(output, class_id, class)?;
    for layout in &class.state.legal_layouts {
        let relation = match layout {
            StateLayoutAlternative::Compiled => "persistent-state-layout-compiled",
            StateLayoutAlternative::PackedTokenSlots => {
                "persistent-state-layout-packed-token-slots"
            }
        };
        write_unary_fact(output, relation, class_id);
    }
    Ok(())
}

fn write_storage_facts(
    output: &mut String,
    class_id: i64,
    storage_facts: &StateStorageFacts,
) -> Result<(), ExecutorError> {
    let StateStorageFacts::TokenSlots {
        components,
        bytes_per_token_per_layer,
        page_bytes_per_layer,
        token_relocatable,
        storage,
    } = storage_facts
    else {
        return Err(ExecutorError::CompilerFactsMismatch);
    };
    write_unary_fact(
        output,
        match storage {
            orbitkv::StateStorageKind::TokenKv => "persistent-state-storage-token-kv",
            orbitkv::StateStorageKind::LatentKv => "persistent-state-storage-latent-kv",
            orbitkv::StateStorageKind::GenericTokenState => {
                "persistent-state-storage-generic-token-state"
            }
        },
        class_id,
    );
    write_integer_fact(
        output,
        "persistent-state-token-bytes",
        class_id,
        *bytes_per_token_per_layer,
    )?;
    write_integer_fact(
        output,
        "persistent-state-page-bytes",
        class_id,
        *page_bytes_per_layer,
    )?;
    for component in components {
        writeln!(
            output,
            "(persistent-state-component-bytes {class_id} {} {})",
            egglog_string(&component.name)?,
            component.bytes_per_token_per_layer
        )
        .expect("writing to String cannot fail");
    }
    if *token_relocatable {
        write_unary_fact(output, "persistent-state-token-relocatable", class_id);
    }
    Ok(())
}

fn write_retention_facts(
    output: &mut String,
    class_id: i64,
    class: &LuminalStateClassFacts,
) -> Result<(), ExecutorError> {
    match (class.state.retention, class.state.window_tokens) {
        (Some(RetentionKind::Full), None) => {
            writeln!(output, "(persistent-state-retention-full {class_id})")
                .expect("writing to String cannot fail");
        }
        (Some(RetentionKind::Sliding), Some(window)) => {
            writeln!(
                output,
                "(persistent-state-retention-sliding {class_id} {window})"
            )
            .expect("writing to String cannot fail");
        }
        (Some(RetentionKind::Chunked), None) => {
            let Some(AddressProgram::ResettableArena {
                blocks_per_epoch: blocks,
            }) = class.state.address
            else {
                return Err(ExecutorError::CompilerFactsMismatch);
            };
            writeln!(
                output,
                "(persistent-state-retention-chunked {class_id} {blocks})"
            )
            .expect("writing to String cannot fail");
        }
        _ => return Err(ExecutorError::CompilerFactsMismatch),
    }
    Ok(())
}

fn write_address_facts(
    output: &mut String,
    class_id: i64,
    class: &LuminalStateClassFacts,
) -> Result<(), ExecutorError> {
    match class
        .state
        .address
        .as_ref()
        .ok_or(ExecutorError::CompilerFactsMismatch)?
    {
        AddressProgram::AppendOnly => {
            write_unary_fact(output, "persistent-state-address-append-only", class_id);
        }
        AddressProgram::Pinned => {
            write_unary_fact(output, "persistent-state-address-pinned", class_id);
        }
        AddressProgram::Periodic { period_blocks } => {
            writeln!(
                output,
                "(persistent-state-address-periodic {class_id} {period_blocks} 0)"
            )
            .expect("writing to String cannot fail");
        }
        AddressProgram::PeriodicFrom {
            period_blocks,
            origin_block,
        } => {
            writeln!(
                output,
                "(persistent-state-address-periodic {class_id} {period_blocks} {origin_block})"
            )
            .expect("writing to String cannot fail");
        }
        AddressProgram::ResettableArena { blocks_per_epoch } => {
            writeln!(
                output,
                "(persistent-state-address-resettable {class_id} {blocks_per_epoch})"
            )
            .expect("writing to String cannot fail");
        }
    }
    Ok(())
}

fn write_retirement_facts(
    output: &mut String,
    class_id: i64,
    class: &LuminalStateClassFacts,
) -> Result<(), ExecutorError> {
    match class
        .state
        .retirement
        .as_ref()
        .ok_or(ExecutorError::CompilerFactsMismatch)?
    {
        orbitkv::plan::RetirementProgram::Never => {
            write_unary_fact(output, "persistent-state-retirement-never", class_id);
        }
        orbitkv::plan::RetirementProgram::BlockEndPlus { offset_tokens } => {
            write_integer_fact(
                output,
                "persistent-state-retirement-block-end-plus",
                class_id,
                *offset_tokens,
            )?;
        }
        orbitkv::plan::RetirementProgram::EpochEnd { blocks_per_epoch } => {
            write_integer_fact(
                output,
                "persistent-state-retirement-epoch-end",
                class_id,
                *blocks_per_epoch,
            )?;
        }
    }
    Ok(())
}

fn class_page_tokens(class: &LuminalStateClassFacts) -> Result<u64, ExecutorError> {
    let StateStorageFacts::TokenSlots {
        bytes_per_token_per_layer,
        page_bytes_per_layer,
        ..
    } = &class.state.storage
    else {
        return Err(ExecutorError::CompilerFactsMismatch);
    };
    page_bytes_per_layer
        .checked_div(*bytes_per_token_per_layer)
        .filter(|tokens| {
            *tokens > 0
                && tokens
                    .checked_mul(*bytes_per_token_per_layer)
                    .is_some_and(|bytes| bytes == *page_bytes_per_layer)
        })
        .ok_or(ExecutorError::CompilerFactsMismatch)
}

fn write_unary_fact(output: &mut String, relation: &str, class_id: i64) {
    writeln!(output, "({relation} {class_id})").expect("writing to String cannot fail");
}

fn write_integer_fact(
    output: &mut String,
    relation: &str,
    class_id: i64,
    value: u64,
) -> Result<(), ExecutorError> {
    let value = to_egglog_i64(value)?;
    writeln!(output, "({relation} {class_id} {value})").expect("writing to String cannot fail");
    Ok(())
}

fn to_egglog_i64(value: u64) -> Result<i64, ExecutorError> {
    i64::try_from(value).map_err(|_| ExecutorError::CompilerFactsMismatch)
}

fn egglog_string(value: &str) -> Result<String, ExecutorError> {
    serde_json::to_string(value).map_err(|_| ExecutorError::CompilerFactsMismatch)
}

#[cfg(test)]
mod tests {
    use orbitkv::{
        AttentionStatePlanInput, AttentionStateSpec, AttentionStateStorage,
        compile_runtime_manifest, plan::RetentionKind,
    };

    use super::*;

    fn plan_and_arenas() -> (ExecutorPlan, [ExecutorArena; 2]) {
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
            ],
        })
        .unwrap();
        let plan = ExecutorPlan::compile(&manifest).unwrap();
        let arena = |class_id, pool_id, backend_base_index| ExecutorArena {
            engine_epoch: 1,
            pool_epoch: 1,
            pool_id,
            class_id,
            backend_domain: class_id + 1,
            first_page_id: u32::from(class_id) * 8 + 1,
            page_count: 8,
            backend_base_index,
        };
        (plan, [arena(0, 1, 0), arena(1, 2, 8)])
    }

    #[test]
    fn lowers_manifest_and_arena_contract_to_deterministic_facts() {
        let (plan, arenas) = plan_and_arenas();
        let facts = plan.luminal_compiler_facts(&arenas).unwrap();
        let repeated = plan.luminal_compiler_facts(&arenas).unwrap();
        assert_eq!(facts.digest(), repeated.digest());
        assert_eq!(facts.manifest_fingerprint(), plan.manifest_fingerprint);
        assert_eq!(facts.classes().len(), 2);
        assert!(
            facts
                .egglog()
                .contains("(persistent-state-retention-full 0)")
        );
        assert!(
            facts
                .egglog()
                .contains("(persistent-state-layout-packed-token-slots 0)")
        );
        assert!(
            facts
                .egglog()
                .contains("(persistent-state-retention-sliding 1 64)")
        );
        assert!(
            !facts
                .egglog()
                .contains("(persistent-state-layout-packed-token-slots 1)")
        );
        assert!(facts.egglog().contains("(persistent-state-arena 1 8 8)"));
    }

    #[test]
    fn rejects_reordered_or_overflowing_arena_bindings() {
        let (plan, mut arenas) = plan_and_arenas();
        arenas.swap(0, 1);
        assert!(matches!(
            plan.luminal_compiler_facts(&arenas),
            Err(ExecutorError::PreparedGeometryMismatch)
        ));

        let (plan, mut arenas) = plan_and_arenas();
        arenas[0].backend_base_index = u64::MAX;
        assert!(matches!(
            plan.luminal_compiler_facts(&arenas),
            Err(ExecutorError::CompilerFactsMismatch)
        ));
    }
}
