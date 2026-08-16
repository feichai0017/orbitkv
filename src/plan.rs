use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RetentionKind {
    Full,
    Sliding,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KvClassSpec {
    pub name: String,
    pub layers: Vec<u32>,
    pub retention: RetentionKind,
    pub bytes_per_token_per_layer: u64,
    #[serde(default)]
    pub window_tokens: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KvPlanInput {
    pub page_tokens: u64,
    pub classes: Vec<KvClassSpec>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CompiledKvClass {
    pub spec: KvClassSpec,
    pub slot_count: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ClassCapacity {
    pub name: String,
    pub semantic_live_tokens: u64,
    pub physical_token_slots: u64,
    pub resident_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BlockRange {
    pub start: u64,
    pub end_exclusive: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SglangBoundedClassPolicy {
    pub name: String,
    pub window_tokens: u64,
    pub block_slots: u64,
    pub token_slots: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SglangPolicy {
    pub schema: &'static str,
    pub plan_fingerprint: String,
    pub page_tokens: u64,
    pub swa_eviction_interval_tokens: u64,
    pub max_persistent_swa_token_slots_per_request: u64,
    pub bounded_classes: Vec<SglangBoundedClassPolicy>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CompiledKvPlan {
    pub page_tokens: u64,
    pub classes: Vec<CompiledKvClass>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AddressProgram {
    AppendOnly,
    Periodic { period_blocks: u64 },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RetirementProgram {
    Never,
    BlockEndPlus { offset_tokens: u64 },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ClassLayoutProgram {
    pub name: String,
    pub layers: Vec<u32>,
    pub bytes_per_token_per_layer: u64,
    pub address: AddressProgram,
    pub retirement: RetirementProgram,
    pub minimum_slots_per_request: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct LayoutProgram {
    pub schema: &'static str,
    pub plan_fingerprint: String,
    pub page_tokens: u64,
    pub classes: Vec<ClassLayoutProgram>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PhysicalBackend {
    Paged,
    CudaVmm,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BackendRequirements {
    pub logical_bytes: u64,
    pub cuda_vmm_supported: bool,
    pub cuda_vmm_granularity_bytes: u64,
    pub require_stable_virtual_address: bool,
    pub maximum_rounding_amplification_milli: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BackendDecision {
    pub backend: PhysicalBackend,
    pub logical_bytes: u64,
    pub physical_bytes: u64,
    pub rounding_amplification_milli: u64,
    pub reason: &'static str,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum PlanError {
    #[error("page_tokens must be positive")]
    ZeroPageTokens,
    #[error("compiled plan must contain at least one class")]
    EmptyPlan,
    #[error("{class}: class name must not be empty")]
    EmptyClassName { class: String },
    #[error("{class}: layers must not be empty")]
    EmptyLayers { class: String },
    #[error("{class}: layer ids must be unique")]
    DuplicateLayerInClass { class: String },
    #[error("{class}: bytes_per_token_per_layer must be positive")]
    ZeroBytesPerToken { class: String },
    #[error("{class}: full retention cannot have a window")]
    FullHasWindow { class: String },
    #[error("{class}: sliding retention requires a positive window")]
    SlidingWithoutWindow { class: String },
    #[error("layer {layer} appears in both {first:?} and {second:?}")]
    LayerOverlap {
        layer: u32,
        first: String,
        second: String,
    },
    #[error("boundary must be non-negative")]
    InvalidBoundary,
    #[error("integer overflow while calculating {calculation}")]
    ArithmeticOverflow { calculation: &'static str },
    #[error("compiled sliding class {class:?} is missing its window")]
    InvalidCompiledClass { class: String },
    #[error(
        "SGLang eviction interval {interval} must be a positive multiple of page_tokens {page_tokens}"
    )]
    InvalidSglangEvictionInterval { interval: u64, page_tokens: u64 },
    #[error("backend logical_bytes must be positive")]
    ZeroBackendBytes,
    #[error("CUDA VMM granularity must be positive when VMM is supported")]
    ZeroVmmGranularity,
}

impl CompiledKvPlan {
    #[must_use]
    pub fn fingerprint(&self) -> String {
        let mut hash = Sha256::new();
        hash.update(self.page_tokens.to_le_bytes());
        hash.update((self.classes.len() as u64).to_le_bytes());
        for class in &self.classes {
            update_bytes(&mut hash, class.spec.name.as_bytes());
            hash.update((class.spec.layers.len() as u64).to_le_bytes());
            for &layer in &class.spec.layers {
                hash.update(u64::from(layer).to_le_bytes());
            }
            hash.update(class.spec.bytes_per_token_per_layer.to_le_bytes());
            hash.update(
                match class.spec.retention {
                    RetentionKind::Full => 0_u64,
                    RetentionKind::Sliding => 1_u64,
                }
                .to_le_bytes(),
            );
            hash.update(class.spec.window_tokens.unwrap_or(0).to_le_bytes());
            hash.update(class.slot_count.unwrap_or(0).to_le_bytes());
        }
        format!("sha256:{:x}", hash.finalize())
    }

    /// Emits the temporal address and retirement program consumed by a block
    /// manager backend.
    ///
    /// # Errors
    ///
    /// Returns an error if a compiled sliding class is missing its window or
    /// finite slot count.
    pub fn layout_program(&self) -> Result<LayoutProgram, PlanError> {
        let classes = self
            .classes
            .iter()
            .map(|class| {
                let (address, retirement) = match class.spec.retention {
                    RetentionKind::Full => (AddressProgram::AppendOnly, RetirementProgram::Never),
                    RetentionKind::Sliding => {
                        let window = class.spec.window_tokens.ok_or_else(|| {
                            PlanError::InvalidCompiledClass {
                                class: class.spec.name.clone(),
                            }
                        })?;
                        let period_blocks =
                            class
                                .slot_count
                                .ok_or_else(|| PlanError::InvalidCompiledClass {
                                    class: class.spec.name.clone(),
                                })?;
                        (
                            AddressProgram::Periodic { period_blocks },
                            RetirementProgram::BlockEndPlus {
                                offset_tokens: window - 1,
                            },
                        )
                    }
                };
                Ok(ClassLayoutProgram {
                    name: class.spec.name.clone(),
                    layers: class.spec.layers.clone(),
                    bytes_per_token_per_layer: class.spec.bytes_per_token_per_layer,
                    address,
                    retirement,
                    minimum_slots_per_request: class.slot_count,
                })
            })
            .collect::<Result<Vec<_>, PlanError>>()?;
        Ok(LayoutProgram {
            schema: "orbitkv.layout-program.v1",
            plan_fingerprint: self.fingerprint(),
            page_tokens: self.page_tokens,
            classes,
        })
    }

    /// Calculates semantic and physical capacity for each KV class.
    ///
    /// # Errors
    ///
    /// Returns an error if any checked byte or slot calculation overflows.
    pub fn capacity_at(&self, boundary: u64) -> Result<Vec<ClassCapacity>, PlanError> {
        self.classes
            .iter()
            .map(|class| self.class_capacity_at(class, boundary))
            .collect()
    }

    /// Calculates total resident bytes at a logical boundary.
    ///
    /// # Errors
    ///
    /// Returns an error if any checked byte calculation overflows.
    pub fn resident_bytes_at(&self, boundary: u64) -> Result<u64, PlanError> {
        self.capacity_at(boundary)?
            .into_iter()
            .try_fold(0_u64, |total, capacity| {
                total
                    .checked_add(capacity.resident_bytes)
                    .ok_or(PlanError::ArithmeticOverflow {
                        calculation: "total resident bytes",
                    })
            })
    }

    /// Calculates a diagnostic baseline that retains every class as full attention.
    ///
    /// # Errors
    ///
    /// Returns an error if any checked byte or slot calculation overflows.
    pub fn all_full_baseline_bytes_at(&self, boundary: u64) -> Result<u64, PlanError> {
        let pages = ceil_div(boundary, self.page_tokens)?;
        let token_slots =
            pages
                .checked_mul(self.page_tokens)
                .ok_or(PlanError::ArithmeticOverflow {
                    calculation: "all-full token slots",
                })?;
        self.classes.iter().try_fold(0_u64, |total, class| {
            let layer_count = u64::try_from(class.spec.layers.len()).map_err(|_| {
                PlanError::ArithmeticOverflow {
                    calculation: "layer count",
                }
            })?;
            let bytes = token_slots
                .checked_mul(class.spec.bytes_per_token_per_layer)
                .and_then(|value| value.checked_mul(layer_count))
                .ok_or(PlanError::ArithmeticOverflow {
                    calculation: "all-full baseline bytes",
                })?;
            total
                .checked_add(bytes)
                .ok_or(PlanError::ArithmeticOverflow {
                    calculation: "total all-full baseline bytes",
                })
        })
    }

    /// Returns the logical blocks required to continue from a pre-query boundary.
    ///
    /// # Errors
    ///
    /// Returns an error if the compiled plan contains an invalid sliding class.
    pub fn continuation_blocks(
        &self,
        boundary: u64,
    ) -> Result<BTreeMap<String, Vec<u64>>, PlanError> {
        self.classes
            .iter()
            .map(|class| {
                let start_token = match class.spec.retention {
                    RetentionKind::Full => 0,
                    RetentionKind::Sliding => {
                        let window = class.spec.window_tokens.ok_or_else(|| {
                            PlanError::InvalidCompiledClass {
                                class: class.spec.name.clone(),
                            }
                        })?;
                        boundary.saturating_sub(window.saturating_sub(1))
                    }
                };
                let blocks = if start_token >= boundary {
                    Vec::new()
                } else {
                    let first = start_token / self.page_tokens;
                    let last = (boundary - 1) / self.page_tokens;
                    (first..=last).collect()
                };
                Ok((class.spec.name.clone(), blocks))
            })
            .collect()
    }

    /// Returns the compact ranges required to continue from a pre-query boundary.
    ///
    /// # Errors
    ///
    /// Returns an error if the compiled plan contains an invalid sliding class.
    pub fn continuation_ranges(
        &self,
        boundary: u64,
    ) -> Result<BTreeMap<String, Vec<BlockRange>>, PlanError> {
        Ok(self
            .continuation_blocks(boundary)?
            .into_iter()
            .map(|(name, blocks)| (name, compact_ranges(&blocks)))
            .collect())
    }

    /// Lowers bounded block lifetimes to `SGLang`'s page-granular SWA policy.
    ///
    /// # Errors
    ///
    /// Returns an error if a checked slot calculation overflows or a compiled
    /// sliding class is missing its window.
    pub fn sglang_policy(&self) -> Result<SglangPolicy, PlanError> {
        self.sglang_policy_with_eviction_interval(self.page_tokens)
    }

    /// Lowers bounded block lifetimes with an explicit `SGLang` reclamation
    /// interval used by the physical-plan cost model.
    ///
    /// # Errors
    ///
    /// Returns an error if the interval is not a positive page multiple, a
    /// checked slot calculation overflows, or a sliding class is invalid.
    pub fn sglang_policy_with_eviction_interval(
        &self,
        eviction_interval_tokens: u64,
    ) -> Result<SglangPolicy, PlanError> {
        if eviction_interval_tokens == 0
            || !eviction_interval_tokens.is_multiple_of(self.page_tokens)
        {
            return Err(PlanError::InvalidSglangEvictionInterval {
                interval: eviction_interval_tokens,
                page_tokens: self.page_tokens,
            });
        }
        let bounded_classes =
            self.classes
                .iter()
                .filter_map(|class| class.slot_count.map(|slots| (class, slots)))
                .map(|(class, block_slots)| {
                    let window_tokens = class.spec.window_tokens.ok_or_else(|| {
                        PlanError::InvalidCompiledClass {
                            class: class.spec.name.clone(),
                        }
                    })?;
                    let token_slots = block_slots.checked_mul(self.page_tokens).ok_or(
                        PlanError::ArithmeticOverflow {
                            calculation: "SGLang policy token slots",
                        },
                    )?;
                    Ok(SglangBoundedClassPolicy {
                        name: class.spec.name.clone(),
                        window_tokens,
                        block_slots,
                        token_slots,
                    })
                })
                .collect::<Result<Vec<_>, PlanError>>()?;
        let max_persistent_swa_token_slots_per_request = bounded_classes
            .iter()
            .map(|class| class.token_slots)
            .max()
            .unwrap_or(0);
        Ok(SglangPolicy {
            schema: "orbitkv.sglang-policy.v1",
            plan_fingerprint: self.fingerprint(),
            page_tokens: self.page_tokens,
            swa_eviction_interval_tokens: eviction_interval_tokens,
            max_persistent_swa_token_slots_per_request,
            bounded_classes,
        })
    }

    fn class_capacity_at(
        &self,
        class: &CompiledKvClass,
        boundary: u64,
    ) -> Result<ClassCapacity, PlanError> {
        let (semantic_live_tokens, slots) =
            match class.spec.retention {
                RetentionKind::Full => (boundary, ceil_div(boundary, self.page_tokens)?),
                RetentionKind::Sliding => {
                    let window = class.spec.window_tokens.ok_or_else(|| {
                        PlanError::InvalidCompiledClass {
                            class: class.spec.name.clone(),
                        }
                    })?;
                    let semantic = boundary.min(window.saturating_sub(1));
                    let existing = ceil_div(boundary, self.page_tokens)?;
                    (semantic, existing.min(class.slot_count.unwrap_or(0)))
                }
            };
        let physical_token_slots =
            slots
                .checked_mul(self.page_tokens)
                .ok_or(PlanError::ArithmeticOverflow {
                    calculation: "physical token slots",
                })?;
        let layer_count =
            u64::try_from(class.spec.layers.len()).map_err(|_| PlanError::ArithmeticOverflow {
                calculation: "layer count",
            })?;
        let resident_bytes = physical_token_slots
            .checked_mul(class.spec.bytes_per_token_per_layer)
            .and_then(|value| value.checked_mul(layer_count))
            .ok_or(PlanError::ArithmeticOverflow {
                calculation: "class resident bytes",
            })?;
        Ok(ClassCapacity {
            name: class.spec.name.clone(),
            semantic_live_tokens,
            physical_token_slots,
            resident_bytes,
        })
    }
}

/// Chooses a physical backend using explicit VMM rounding and address-stability
/// costs.
///
/// # Errors
///
/// Returns an error for zero logical bytes, a zero VMM granularity on a
/// VMM-capable device, or checked arithmetic overflow.
pub fn choose_physical_backend(
    requirements: &BackendRequirements,
) -> Result<BackendDecision, PlanError> {
    if requirements.logical_bytes == 0 {
        return Err(PlanError::ZeroBackendBytes);
    }
    if !requirements.cuda_vmm_supported {
        return Ok(BackendDecision {
            backend: PhysicalBackend::Paged,
            logical_bytes: requirements.logical_bytes,
            physical_bytes: requirements.logical_bytes,
            rounding_amplification_milli: 1000,
            reason: "cuda_vmm_unsupported",
        });
    }
    if requirements.cuda_vmm_granularity_bytes == 0 {
        return Err(PlanError::ZeroVmmGranularity);
    }
    let physical_bytes = ceil_div(
        requirements.logical_bytes,
        requirements.cuda_vmm_granularity_bytes,
    )?
    .checked_mul(requirements.cuda_vmm_granularity_bytes)
    .ok_or(PlanError::ArithmeticOverflow {
        calculation: "VMM rounded physical bytes",
    })?;
    let rounding_amplification_milli =
        physical_bytes
            .checked_mul(1000)
            .ok_or(PlanError::ArithmeticOverflow {
                calculation: "VMM rounding amplification",
            })?
            / requirements.logical_bytes;
    if !requirements.require_stable_virtual_address {
        return Ok(BackendDecision {
            backend: PhysicalBackend::Paged,
            logical_bytes: requirements.logical_bytes,
            physical_bytes: requirements.logical_bytes,
            rounding_amplification_milli: 1000,
            reason: "stable_virtual_address_not_required",
        });
    }
    if rounding_amplification_milli > requirements.maximum_rounding_amplification_milli {
        return Ok(BackendDecision {
            backend: PhysicalBackend::Paged,
            logical_bytes: requirements.logical_bytes,
            physical_bytes: requirements.logical_bytes,
            rounding_amplification_milli: 1000,
            reason: "cuda_vmm_rounding_too_expensive",
        });
    }
    Ok(BackendDecision {
        backend: PhysicalBackend::CudaVmm,
        logical_bytes: requirements.logical_bytes,
        physical_bytes,
        rounding_amplification_milli,
        reason: "stable_virtual_address_within_rounding_budget",
    })
}

/// Compiles checked Full and sliding-window classes into a KV block plan.
///
/// # Errors
///
/// Returns an error for invalid class geometry, overlapping layers, zero sizes,
/// or checked arithmetic overflow.
pub fn compile_plan(input: KvPlanInput) -> Result<CompiledKvPlan, PlanError> {
    if input.page_tokens == 0 {
        return Err(PlanError::ZeroPageTokens);
    }
    if input.classes.is_empty() {
        return Err(PlanError::EmptyPlan);
    }

    let mut claimed_layers = BTreeMap::<u32, String>::new();
    let mut compiled = Vec::with_capacity(input.classes.len());
    for class in input.classes {
        validate_class(&class)?;
        for &layer in &class.layers {
            if let Some(first) = claimed_layers.insert(layer, class.name.clone()) {
                return Err(PlanError::LayerOverlap {
                    layer,
                    first,
                    second: class.name,
                });
            }
        }
        let slot_count = match class.retention {
            RetentionKind::Full => None,
            RetentionKind::Sliding => {
                let window =
                    class
                        .window_tokens
                        .ok_or_else(|| PlanError::InvalidCompiledClass {
                            class: class.name.clone(),
                        })?;
                Some(
                    1_u64
                        .checked_add(ceil_div(window - 1, input.page_tokens)?)
                        .ok_or(PlanError::ArithmeticOverflow {
                            calculation: "sliding slot count",
                        })?,
                )
            }
        };
        compiled.push(CompiledKvClass {
            spec: class,
            slot_count,
        });
    }
    Ok(CompiledKvPlan {
        page_tokens: input.page_tokens,
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
        RetentionKind::Full if class.window_tokens.is_some() => Err(PlanError::FullHasWindow {
            class: class.name.clone(),
        }),
        RetentionKind::Sliding if class.window_tokens.is_none_or(|window| window == 0) => {
            Err(PlanError::SlidingWithoutWindow {
                class: class.name.clone(),
            })
        }
        _ => Ok(()),
    }
}

fn ceil_div(value: u64, divisor: u64) -> Result<u64, PlanError> {
    if value == 0 {
        return Ok(0);
    }
    value
        .checked_add(divisor - 1)
        .map(|sum| sum / divisor)
        .ok_or(PlanError::ArithmeticOverflow {
            calculation: "ceiling division",
        })
}

fn compact_ranges(blocks: &[u64]) -> Vec<BlockRange> {
    let Some((&first, rest)) = blocks.split_first() else {
        return Vec::new();
    };
    let mut ranges = Vec::new();
    let mut start = first;
    let mut previous = first;
    for &block in rest {
        if block == previous + 1 {
            previous = block;
            continue;
        }
        ranges.push(BlockRange {
            start,
            end_exclusive: previous + 1,
        });
        start = block;
        previous = block;
    }
    ranges.push(BlockRange {
        start,
        end_exclusive: previous + 1,
    });
    ranges
}

fn update_bytes(hash: &mut Sha256, bytes: &[u8]) {
    hash.update((bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sliding_input(window: u64, page: u64) -> KvPlanInput {
        KvPlanInput {
            page_tokens: page,
            classes: vec![KvClassSpec {
                name: "swa".into(),
                layers: vec![0],
                retention: RetentionKind::Sliding,
                bytes_per_token_per_layer: 128,
                window_tokens: Some(window),
            }],
        }
    }

    #[test]
    fn sliding_formula_matches_exhaustive_overlap() {
        for page in [1, 4, 16] {
            for window in 1..=65 {
                let plan = compile_plan(sliding_input(window, page)).unwrap();
                let mut maximum = 0;
                for query in 0..window + 2 * page {
                    let first_key = query.saturating_sub(window - 1);
                    let first_block = first_key / page;
                    let last_block = query / page;
                    maximum = maximum.max(last_block - first_block + 1);
                }
                assert_eq!(plan.classes[0].slot_count, Some(maximum));
            }
        }
    }

    #[test]
    fn continuation_cut_has_w_minus_one_old_tokens() {
        let plan = compile_plan(sliding_input(32, 16)).unwrap();
        assert_eq!(plan.continuation_blocks(64).unwrap()["swa"], vec![2, 3]);
    }

    #[test]
    fn full_and_swa_capacity_matches_page_geometry() {
        let plan = compile_plan(KvPlanInput {
            page_tokens: 16,
            classes: vec![
                KvClassSpec {
                    name: "full".into(),
                    layers: (0..10).collect(),
                    retention: RetentionKind::Full,
                    bytes_per_token_per_layer: 4096,
                    window_tokens: None,
                },
                KvClassSpec {
                    name: "swa".into(),
                    layers: (10..62).collect(),
                    retention: RetentionKind::Sliding,
                    bytes_per_token_per_layer: 4096,
                    window_tokens: Some(1024),
                },
            ],
        })
        .unwrap();
        let resident = plan.resident_bytes_at(32_768).unwrap();
        let baseline = plan.all_full_baseline_bytes_at(32_768).unwrap();
        assert_eq!(resident, 1_563_688_960);
        assert_eq!(baseline, 8_321_499_136);
        let policy = plan.sglang_policy().unwrap();
        assert_eq!(policy.swa_eviction_interval_tokens, 16);
        assert_eq!(policy.max_persistent_swa_token_slots_per_request, 1040);
        assert_eq!(policy.bounded_classes[0].block_slots, 65);
        assert_eq!(
            plan.sglang_policy_with_eviction_interval(64)
                .unwrap()
                .swa_eviction_interval_tokens,
            64
        );
        assert!(matches!(
            plan.sglang_policy_with_eviction_interval(24),
            Err(PlanError::InvalidSglangEvictionInterval { .. })
        ));
        let layout = plan.layout_program().unwrap();
        assert_eq!(layout.schema, "orbitkv.layout-program.v1");
        assert_eq!(layout.plan_fingerprint, plan.fingerprint());
        assert_eq!(layout.classes[0].address, AddressProgram::AppendOnly);
        assert_eq!(
            layout.classes[1].address,
            AddressProgram::Periodic { period_blocks: 65 }
        );
        assert_eq!(
            layout.classes[1].retirement,
            RetirementProgram::BlockEndPlus {
                offset_tokens: 1023
            }
        );
    }

    #[test]
    fn vmm_backend_is_selected_only_when_rounding_is_affordable() {
        let small = choose_physical_backend(&BackendRequirements {
            logical_bytes: 64 * 1024,
            cuda_vmm_supported: true,
            cuda_vmm_granularity_bytes: 2 * 1024 * 1024,
            require_stable_virtual_address: true,
            maximum_rounding_amplification_milli: 1250,
        })
        .unwrap();
        assert_eq!(small.backend, PhysicalBackend::Paged);
        assert_eq!(small.reason, "cuda_vmm_rounding_too_expensive");

        let large = choose_physical_backend(&BackendRequirements {
            logical_bytes: 16 * 1024 * 1024,
            cuda_vmm_supported: true,
            cuda_vmm_granularity_bytes: 2 * 1024 * 1024,
            require_stable_virtual_address: true,
            maximum_rounding_amplification_milli: 1250,
        })
        .unwrap();
        assert_eq!(large.backend, PhysicalBackend::CudaVmm);
        assert_eq!(large.rounding_amplification_milli, 1000);
    }
}
