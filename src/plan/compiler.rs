use std::collections::{BTreeMap, BTreeSet};

use crate::retention::{
    AtomicRetention, InferredRegion, InferredRetention, KvHeadRange, RetentionAnalysis,
    RetentionError, RetentionProgramInput, RetentionStateDecl, analyze_state,
};

use super::{
    BlockDomain, CompiledKvClass, CompiledKvPlan, KvClassSpec, KvPlanInput, PlanError,
    RetentionKind, ceil_div,
};

/// Compiles declarative `may_read(query, key)` relations into the current
/// executable lifetime and layout plan.
///
/// # Errors
///
/// Returns an error for an unsupported schema, invalid state geometry, or
/// failed lifetime analysis.
pub fn compile_retention_program(
    program: RetentionProgramInput,
) -> Result<CompiledKvPlan, PlanError> {
    if program.schema != "orbitkv.retention-ir.v1" {
        return Err(RetentionError::UnsupportedSchema(program.schema).into());
    }
    if program.page_tokens == 0 {
        return Err(PlanError::ZeroPageTokens);
    }
    let mut classes = Vec::new();
    for state in program.states {
        validate_head_range(&state)?;
        let RetentionAnalysis { inferred, .. } = analyze_state(&state)?;
        match inferred {
            InferredRetention::Unbounded => classes.push(InferredClassInput::atomic(
                &state,
                state.name.clone(),
                RetentionKind::Full,
                None,
                BlockDomain::all(),
                None,
            )),
            InferredRetention::FixedWindow { window_tokens } => {
                classes.push(InferredClassInput::atomic(
                    &state,
                    state.name.clone(),
                    RetentionKind::Sliding,
                    Some(window_tokens),
                    BlockDomain::all(),
                    None,
                ));
            }
            InferredRetention::Chunked { chunk_tokens } => {
                if !chunk_tokens.is_multiple_of(program.page_tokens) {
                    return Err(PlanError::ChunkNotPageAligned {
                        chunk_tokens,
                        page_tokens: program.page_tokens,
                    });
                }
                classes.push(InferredClassInput::chunked(&state, chunk_tokens));
            }
            InferredRetention::Partitioned { regions } => {
                classes.extend(lower_partitioned_state(
                    &state,
                    regions,
                    program.page_tokens,
                )?);
            }
        }
    }
    compile_inferred_classes(program.page_tokens, classes)
}

/// Compiles checked Full and sliding-window classes into a KV block plan.
///
/// # Errors
///
/// Returns an error for invalid class geometry, overlapping layers, zero sizes,
/// or checked arithmetic overflow.
pub fn compile_plan(input: KvPlanInput) -> Result<CompiledKvPlan, PlanError> {
    validate_plan_input(&input)?;
    compile_retention_program(input.into_retention_program()?)
}

struct InferredClassInput {
    spec: KvClassSpec,
    kv_head_range: Option<KvHeadRange>,
    chunk_tokens: Option<u64>,
    source_state: Option<String>,
    block_domain: BlockDomain,
}

impl InferredClassInput {
    fn atomic(
        state: &RetentionStateDecl,
        name: String,
        retention: RetentionKind,
        window_tokens: Option<u64>,
        block_domain: BlockDomain,
        source_state: Option<String>,
    ) -> Self {
        Self {
            spec: KvClassSpec {
                name,
                layers: state.layers.clone(),
                retention,
                bytes_per_token_per_layer: state.bytes_per_token_per_layer,
                window_tokens,
            },
            kv_head_range: state.kv_head_range.clone(),
            chunk_tokens: None,
            source_state,
            block_domain,
        }
    }

    fn chunked(state: &RetentionStateDecl, chunk_tokens: u64) -> Self {
        Self {
            spec: KvClassSpec {
                name: state.name.clone(),
                layers: state.layers.clone(),
                retention: RetentionKind::Chunked,
                bytes_per_token_per_layer: state.bytes_per_token_per_layer,
                window_tokens: None,
            },
            kv_head_range: state.kv_head_range.clone(),
            chunk_tokens: Some(chunk_tokens),
            source_state: None,
            block_domain: BlockDomain::all(),
        }
    }
}

fn lower_partitioned_state(
    state: &RetentionStateDecl,
    regions: Vec<InferredRegion>,
    page_tokens: u64,
) -> Result<Vec<InferredClassInput>, PlanError> {
    regions
        .into_iter()
        .map(|region| {
            if !region.start_token.is_multiple_of(page_tokens)
                || region
                    .end_token_exclusive
                    .is_some_and(|end| !end.is_multiple_of(page_tokens))
            {
                return Err(PlanError::SinkBoundaryNotPageAligned {
                    sink_tokens: region.end_token_exclusive.unwrap_or(region.start_token),
                    page_tokens,
                });
            }
            let (retention, window_tokens) = match region.retention {
                AtomicRetention::Unbounded => (RetentionKind::Full, None),
                AtomicRetention::FixedWindow { window_tokens } => {
                    (RetentionKind::Sliding, Some(window_tokens))
                }
            };
            Ok(InferredClassInput::atomic(
                state,
                format!("{}::{}", state.name, region.label),
                retention,
                window_tokens,
                BlockDomain {
                    start_block: region.start_token / page_tokens,
                    end_block_exclusive: region.end_token_exclusive.map(|end| end / page_tokens),
                },
                Some(state.name.clone()),
            ))
        })
        .collect()
}

fn validate_plan_input(input: &KvPlanInput) -> Result<(), PlanError> {
    if input.page_tokens == 0 {
        return Err(PlanError::ZeroPageTokens);
    }
    if input.classes.is_empty() {
        return Err(PlanError::EmptyPlan);
    }

    let mut claimed_layers = BTreeMap::<u32, String>::new();
    for class in &input.classes {
        validate_class(class)?;
        for &layer in &class.layers {
            if let Some(first) = claimed_layers.insert(layer, class.name.clone()) {
                return Err(PlanError::LayerOverlap {
                    layer,
                    first,
                    second: class.name.clone(),
                });
            }
        }
    }
    Ok(())
}

fn compile_inferred_classes(
    page_tokens: u64,
    classes: Vec<InferredClassInput>,
) -> Result<CompiledKvPlan, PlanError> {
    if page_tokens == 0 {
        return Err(PlanError::ZeroPageTokens);
    }
    if classes.is_empty() {
        return Err(PlanError::EmptyPlan);
    }
    let mut names = BTreeSet::new();
    let mut claimed_layers = BTreeMap::<u32, Vec<(String, Option<KvHeadRange>)>>::new();
    let mut compiled = Vec::with_capacity(classes.len());
    for class in classes {
        validate_class(&class.spec)?;
        if !names.insert(class.spec.name.clone()) {
            return Err(PlanError::DuplicateClassName(class.spec.name));
        }
        let source = class
            .source_state
            .clone()
            .unwrap_or_else(|| class.spec.name.clone());
        for &layer in &class.spec.layers {
            let claims = claimed_layers.entry(layer).or_default();
            if let Some((first, _)) = claims.iter().find(|(first, range)| {
                first != &source
                    && head_ranges_overlap(range.as_ref(), class.kv_head_range.as_ref())
            }) {
                if class.kv_head_range.is_some() {
                    return Err(PlanError::KvHeadOverlap {
                        layer,
                        first: first.clone(),
                        second: source.clone(),
                    });
                }
                return Err(PlanError::LayerOverlap {
                    layer,
                    first: first.clone(),
                    second: source.clone(),
                });
            }
            claims.push((source.clone(), class.kv_head_range.clone()));
        }
        let slot_count =
            match class.spec.retention {
                RetentionKind::Full => class
                    .block_domain
                    .end_block_exclusive
                    .map(|end| end.saturating_sub(class.block_domain.start_block)),
                RetentionKind::Sliding => {
                    let window = class.spec.window_tokens.ok_or_else(|| {
                        PlanError::InvalidCompiledClass {
                            class: class.spec.name.clone(),
                        }
                    })?;
                    Some(
                        1_u64
                            .checked_add(ceil_div(window - 1, page_tokens)?)
                            .ok_or(PlanError::ArithmeticOverflow {
                                calculation: "sliding slot count",
                            })?,
                    )
                }
                RetentionKind::Chunked => {
                    let chunk_tokens =
                        class
                            .chunk_tokens
                            .ok_or_else(|| PlanError::InvalidCompiledChunk {
                                class: class.spec.name.clone(),
                            })?;
                    if !chunk_tokens.is_multiple_of(page_tokens) {
                        return Err(PlanError::ChunkNotPageAligned {
                            chunk_tokens,
                            page_tokens,
                        });
                    }
                    Some(chunk_tokens / page_tokens)
                }
            };
        compiled.push(CompiledKvClass {
            spec: class.spec,
            slot_count,
            kv_head_range: class.kv_head_range,
            chunk_tokens: class.chunk_tokens,
            source_state: class.source_state,
            block_domain: class.block_domain,
        });
    }
    Ok(CompiledKvPlan {
        page_tokens,
        classes: compiled,
    })
}

fn validate_class(class: &KvClassSpec) -> Result<(), PlanError> {
    if class.name.is_empty() {
        return Err(PlanError::EmptyClassName {
            class: class.name.clone(),
        });
    }
    if class.layers.is_empty() {
        return Err(PlanError::EmptyLayers {
            class: class.name.clone(),
        });
    }
    let unique = class.layers.iter().copied().collect::<BTreeSet<_>>();
    if unique.len() != class.layers.len() {
        return Err(PlanError::DuplicateLayerInClass {
            class: class.name.clone(),
        });
    }
    if class.bytes_per_token_per_layer == 0 {
        return Err(PlanError::ZeroBytesPerToken {
            class: class.name.clone(),
        });
    }
    match class.retention {
        RetentionKind::Full | RetentionKind::Chunked if class.window_tokens.is_some() => {
            Err(PlanError::FullHasWindow {
                class: class.name.clone(),
            })
        }
        RetentionKind::Sliding if class.window_tokens.is_none_or(|window| window == 0) => {
            Err(PlanError::SlidingWithoutWindow {
                class: class.name.clone(),
            })
        }
        _ => Ok(()),
    }
}

fn validate_head_range(state: &RetentionStateDecl) -> Result<(), PlanError> {
    if state
        .kv_head_range
        .as_ref()
        .is_some_and(|range| range.start >= range.end_exclusive)
    {
        return Err(PlanError::EmptyKvHeadRange {
            state: state.name.clone(),
        });
    }
    Ok(())
}

fn head_ranges_overlap(lhs: Option<&KvHeadRange>, rhs: Option<&KvHeadRange>) -> bool {
    match (lhs, rhs) {
        (Some(lhs), Some(rhs)) => lhs.start < rhs.end_exclusive && rhs.start < lhs.end_exclusive,
        _ => true,
    }
}
