use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::retention::{
    IntExpr, KvHeadRange, Predicate, RetentionError, RetentionProgramInput, RetentionStateDecl,
};

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RetentionKind {
    Full,
    Sliding,
    Chunked,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_head_range: Option<KvHeadRange>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_state: Option<String>,
    #[serde(default, skip_serializing_if = "BlockDomain::is_all")]
    pub block_domain: BlockDomain,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ClassCapacity {
    pub name: String,
    pub semantic_live_tokens: u64,
    pub physical_token_slots: u64,
    pub resident_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct LifetimeNormalizedClass {
    pub name: String,
    pub layers: Vec<u32>,
    pub kv_head_range: KvHeadRange,
    pub head_count: u64,
    pub slot_count: u64,
    pub bytes_per_head_per_token: u64,
    pub normalized_bytes_per_request: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct LifetimeNormalizationReport {
    pub schema: &'static str,
    pub plan_fingerprint: String,
    pub page_tokens: u64,
    pub normalized_classes: Vec<LifetimeNormalizedClass>,
    pub normalized_bytes_per_request: u64,
    pub max_window_baseline_bytes_per_request: u64,
    pub savings_bytes_per_request: u64,
    pub savings_percent_milli: u64,
    pub retention_amplification_milli: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ApplicabilityClass {
    SafeFallback,
    UniformBounded,
    HybridLifetimes,
    RegionSpecialization,
    LifetimeNormalization,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ApplicabilityClassGeometry {
    pub name: String,
    pub retention: RetentionKind,
    pub layer_count: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_head_range: Option<KvHeadRange>,
    pub semantic_live_tokens: u64,
    pub physical_token_slots: u64,
    pub resident_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ApplicabilityReport {
    pub schema: &'static str,
    pub plan_fingerprint: String,
    pub page_tokens: u64,
    pub boundary_tokens: u64,
    pub applicability: ApplicabilityClass,
    pub lifetime_class_count: u64,
    pub bounded_class_count: u64,
    pub unbounded_class_count: u64,
    pub generated_layouts: Vec<&'static str>,
    pub classes: Vec<ApplicabilityClassGeometry>,
    pub semantically_live_bytes: u64,
    pub physical_resident_bytes: u64,
    pub all_full_baseline_bytes: u64,
    pub reclaimable_baseline_bytes: u64,
    pub static_reduction_percent_milli: u64,
    pub bounded_resident_bytes: u64,
    pub bounded_resident_fraction_milli: u64,
    pub physical_to_semantic_amplification_milli: Option<u64>,
    pub claim_boundary: [&'static str; 3],
}

type HeadStripe = (KvHeadRange, u64, u64);

struct NormalizedHeadGeometry {
    classes: Vec<LifetimeNormalizedClass>,
    per_layer: BTreeMap<u32, Vec<HeadStripe>>,
    bytes_per_request: u64,
}

struct ApplicabilityGeometry {
    classes: Vec<ApplicabilityClassGeometry>,
    semantically_live_bytes: u64,
    physical_resident_bytes: u64,
    bounded_resident_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BlockRange {
    pub start: u64,
    pub end_exclusive: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CompiledKvPlan {
    pub page_tokens: u64,
    pub classes: Vec<CompiledKvClass>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AddressProgram {
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

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RetirementProgram {
    Never,
    BlockEndPlus { offset_tokens: u64 },
    EpochEnd { blocks_per_epoch: u64 },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ClassLayoutProgram {
    pub name: String,
    pub layers: Vec<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_head_range: Option<KvHeadRange>,
    pub bytes_per_token_per_layer: u64,
    pub address: AddressProgram,
    pub retirement: RetirementProgram,
    pub minimum_slots_per_request: Option<u64>,
    #[serde(default, skip_serializing_if = "BlockDomain::is_all")]
    pub block_domain: BlockDomain,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct LayoutProgram {
    pub schema: &'static str,
    pub plan_fingerprint: String,
    pub page_tokens: u64,
    pub classes: Vec<ClassLayoutProgram>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct LogicalCellId {
    pub request_id: String,
    pub class_name: String,
    pub cell_index: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct CellVersion {
    pub cycle: u64,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct TemporalAddress {
    pub cell: LogicalCellId,
    pub version: CellVersion,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct BlockDomain {
    pub start_block: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_block_exclusive: Option<u64>,
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
    #[error("compiled class name {0:?} is duplicated")]
    DuplicateClassName(String),
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
    #[error("KV head ranges overlap in layer {layer}: {first:?} conflicts with {second:?}")]
    KvHeadOverlap {
        layer: u32,
        first: String,
        second: String,
    },
    #[error("{state}: KV head range must be non-empty")]
    EmptyKvHeadRange { state: String },
    #[error("lifetime normalization requires KV head ranges on every class")]
    MissingKvHeadGeometry,
    #[error("lifetime normalization requires a uniform byte width per KV head in layer {layer}")]
    InconsistentKvHeadWidth { layer: u32 },
    #[error(
        "lifetime normalization requires contiguous KV head coverage in layer {layer}, expected head {expected}, found {actual}"
    )]
    KvHeadCoverageGap {
        layer: u32,
        expected: u32,
        actual: u32,
    },
    #[error("integer overflow while calculating {calculation}")]
    ArithmeticOverflow { calculation: &'static str },
    #[error("compiled sliding class {class:?} is missing its window")]
    InvalidCompiledClass { class: String },
    #[error("compiled chunked class {class:?} is missing its chunk size")]
    InvalidCompiledChunk { class: String },
    #[error("periodic address program must have a positive period")]
    ZeroAddressPeriod,
    #[error("layout program does not contain class {0:?}")]
    UnknownLayoutClass(String),
    #[error("logical block {ordinal} is outside class {class:?} block domain")]
    AddressOutsideBlockDomain { class: String, ordinal: u64 },
    #[error("sink boundary {sink_tokens} must be aligned to page_tokens {page_tokens}")]
    SinkBoundaryNotPageAligned { sink_tokens: u64, page_tokens: u64 },
    #[error("canonical manager plans cannot declare chunked retention; use Retention IR")]
    CanonicalChunkedUnsupported,
    #[error("chunk size {chunk_tokens} must be aligned to page_tokens {page_tokens}")]
    ChunkNotPageAligned { chunk_tokens: u64, page_tokens: u64 },
    #[error("{class}: window {window} does not fit Retention IR i64 constants")]
    WindowOutOfRange { class: String, window: u64 },
    #[error(transparent)]
    Retention(#[from] RetentionError),
}

impl AddressProgram {
    /// Evaluates the dense cell index and temporal version without allocating
    /// logical identity strings.
    ///
    /// # Errors
    ///
    /// Returns an error if a periodic program has a zero period or its origin
    /// exceeds the logical ordinal.
    pub fn evaluate_dense(&self, ordinal: u64) -> Result<(u64, CellVersion), PlanError> {
        let (cell_index, cycle) = match *self {
            Self::AppendOnly | Self::Pinned => (ordinal, 0),
            Self::Periodic { period_blocks } => {
                if period_blocks == 0 {
                    return Err(PlanError::ZeroAddressPeriod);
                }
                (ordinal % period_blocks, ordinal / period_blocks)
            }
            Self::PeriodicFrom {
                period_blocks,
                origin_block,
            } => {
                if period_blocks == 0 {
                    return Err(PlanError::ZeroAddressPeriod);
                }
                let relative =
                    ordinal
                        .checked_sub(origin_block)
                        .ok_or(PlanError::ArithmeticOverflow {
                            calculation: "periodic address origin subtraction",
                        })?;
                (relative % period_blocks, relative / period_blocks)
            }
            Self::ResettableArena { blocks_per_epoch } => {
                if blocks_per_epoch == 0 {
                    return Err(PlanError::ZeroAddressPeriod);
                }
                (ordinal % blocks_per_epoch, ordinal / blocks_per_epoch)
            }
        };
        Ok((cell_index, CellVersion { cycle }))
    }

    /// Maps one logical block ordinal to its compiler-defined cell and version.
    ///
    /// # Errors
    ///
    /// Returns an error if a periodic program has a zero period.
    pub fn evaluate(
        &self,
        request_id: &str,
        class_name: &str,
        ordinal: u64,
    ) -> Result<TemporalAddress, PlanError> {
        let (cell_index, version) = self.evaluate_dense(ordinal)?;
        Ok(TemporalAddress {
            cell: LogicalCellId {
                request_id: request_id.to_owned(),
                class_name: class_name.to_owned(),
                cell_index,
            },
            version,
        })
    }
}

impl BlockDomain {
    #[must_use]
    pub const fn all() -> Self {
        Self {
            start_block: 0,
            end_block_exclusive: None,
        }
    }

    #[must_use]
    pub const fn is_all(&self) -> bool {
        self.start_block == 0 && self.end_block_exclusive.is_none()
    }

    #[must_use]
    pub fn contains(&self, ordinal: u64) -> bool {
        ordinal >= self.start_block && self.end_block_exclusive.is_none_or(|end| ordinal < end)
    }

    #[must_use]
    pub fn blocks_before(&self, end_exclusive: u64) -> u64 {
        let end = self
            .end_block_exclusive
            .map_or(end_exclusive, |domain_end| domain_end.min(end_exclusive));
        end.saturating_sub(self.start_block)
    }
}

impl RetirementProgram {
    /// Evaluates the semantic death boundary of one logical block.
    ///
    /// # Errors
    ///
    /// Returns an error if block or token arithmetic overflows.
    pub fn death_boundary(&self, page_tokens: u64, ordinal: u64) -> Result<Option<u64>, PlanError> {
        match *self {
            Self::Never => Ok(None),
            Self::BlockEndPlus { offset_tokens } => ordinal
                .checked_add(1)
                .and_then(|block_end| block_end.checked_mul(page_tokens))
                .and_then(|block_end| block_end.checked_add(offset_tokens))
                .map(Some)
                .ok_or(PlanError::ArithmeticOverflow {
                    calculation: "retirement program death boundary",
                }),
            Self::EpochEnd { blocks_per_epoch } => {
                if blocks_per_epoch == 0 {
                    return Err(PlanError::ZeroAddressPeriod);
                }
                ordinal
                    .checked_div(blocks_per_epoch)
                    .and_then(|epoch| epoch.checked_add(1))
                    .and_then(|epoch| epoch.checked_mul(blocks_per_epoch))
                    .and_then(|end_block| end_block.checked_mul(page_tokens))
                    .map(Some)
                    .ok_or(PlanError::ArithmeticOverflow {
                        calculation: "epoch retirement boundary",
                    })
            }
        }
    }
}

impl ClassLayoutProgram {
    /// Evaluates this class's temporal address for one request block.
    ///
    /// # Errors
    ///
    /// Returns an error if the address program is invalid.
    pub fn temporal_address(
        &self,
        request_id: &str,
        ordinal: u64,
    ) -> Result<TemporalAddress, PlanError> {
        if !self.block_domain.contains(ordinal) {
            return Err(PlanError::AddressOutsideBlockDomain {
                class: self.name.clone(),
                ordinal,
            });
        }
        self.address.evaluate(request_id, &self.name, ordinal)
    }
}

impl LayoutProgram {
    /// Looks up and evaluates one class address program.
    ///
    /// # Errors
    ///
    /// Returns an error for an unknown class or invalid address program.
    pub fn temporal_address(
        &self,
        request_id: &str,
        class_name: &str,
        ordinal: u64,
    ) -> Result<TemporalAddress, PlanError> {
        self.class(class_name)?
            .temporal_address(request_id, ordinal)
    }

    /// Returns one compiled class layout.
    ///
    /// # Errors
    ///
    /// Returns an error if the class does not exist.
    pub fn class(&self, class_name: &str) -> Result<&ClassLayoutProgram, PlanError> {
        self.classes
            .iter()
            .find(|class| class.name == class_name)
            .ok_or_else(|| PlanError::UnknownLayoutClass(class_name.to_owned()))
    }
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
                    RetentionKind::Chunked => 2_u64,
                }
                .to_le_bytes(),
            );
            hash.update(class.spec.window_tokens.unwrap_or(0).to_le_bytes());
            if let Some(chunk_tokens) = class.chunk_tokens {
                hash.update(1_u64.to_le_bytes());
                hash.update(chunk_tokens.to_le_bytes());
            }
            hash.update(class.slot_count.unwrap_or(0).to_le_bytes());
            if let Some(range) = &class.kv_head_range {
                hash.update(1_u64.to_le_bytes());
                hash.update(u64::from(range.start).to_le_bytes());
                hash.update(u64::from(range.end_exclusive).to_le_bytes());
            }
            if !class.block_domain.is_all() {
                hash.update(1_u64.to_le_bytes());
                hash.update(class.block_domain.start_block.to_le_bytes());
                hash.update(
                    class
                        .block_domain
                        .end_block_exclusive
                        .unwrap_or(u64::MAX)
                        .to_le_bytes(),
                );
            }
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
                    RetentionKind::Full if class.block_domain.end_block_exclusive.is_some() => {
                        (AddressProgram::Pinned, RetirementProgram::Never)
                    }
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
                        let address = if class.block_domain.start_block == 0 {
                            AddressProgram::Periodic { period_blocks }
                        } else {
                            AddressProgram::PeriodicFrom {
                                period_blocks,
                                origin_block: class.block_domain.start_block,
                            }
                        };
                        (
                            address,
                            RetirementProgram::BlockEndPlus {
                                offset_tokens: window - 1,
                            },
                        )
                    }
                    RetentionKind::Chunked => {
                        class
                            .chunk_tokens
                            .ok_or_else(|| PlanError::InvalidCompiledChunk {
                                class: class.spec.name.clone(),
                            })?;
                        let blocks_per_epoch =
                            class
                                .slot_count
                                .ok_or_else(|| PlanError::InvalidCompiledChunk {
                                    class: class.spec.name.clone(),
                                })?;
                        (
                            AddressProgram::ResettableArena { blocks_per_epoch },
                            RetirementProgram::EpochEnd { blocks_per_epoch },
                        )
                    }
                };
                Ok(ClassLayoutProgram {
                    name: class.spec.name.clone(),
                    layers: class.spec.layers.clone(),
                    kv_head_range: class.kv_head_range.clone(),
                    bytes_per_token_per_layer: class.spec.bytes_per_token_per_layer,
                    address,
                    retirement,
                    minimum_slots_per_request: class.slot_count,
                    block_domain: class.block_domain.clone(),
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
        let mut seen_sources = BTreeSet::new();
        self.classes.iter().try_fold(0_u64, |total, class| {
            let source = class.source_state.as_deref().unwrap_or(&class.spec.name);
            if !seen_sources.insert(source.to_owned()) {
                return Ok(total);
            }
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
                Ok((
                    class.spec.name.clone(),
                    self.class_live_blocks(class, boundary)?,
                ))
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

    /// Compares lifetime-normalized head stripes against a baseline that uses
    /// each layer's maximum slot count for every KV head.
    ///
    /// # Errors
    ///
    /// Returns an error unless every class has a non-overlapping, contiguous
    /// KV-head range, every class is bounded, and byte width per head is
    /// uniform within each layer.
    pub fn lifetime_normalization_report(&self) -> Result<LifetimeNormalizationReport, PlanError> {
        let geometry = self.normalized_head_stripes()?;
        let max_window_baseline_bytes_per_request =
            self.max_window_baseline_bytes(geometry.per_layer)?;
        let savings_bytes_per_request =
            max_window_baseline_bytes_per_request - geometry.bytes_per_request;
        let savings_percent_milli = savings_bytes_per_request.checked_mul(100_000).ok_or(
            PlanError::ArithmeticOverflow {
                calculation: "lifetime-normalization savings percent",
            },
        )? / max_window_baseline_bytes_per_request;
        let retention_amplification_milli = max_window_baseline_bytes_per_request
            .checked_mul(1000)
            .ok_or(PlanError::ArithmeticOverflow {
                calculation: "retention amplification",
            })?
            / geometry.bytes_per_request;
        Ok(LifetimeNormalizationReport {
            schema: "orbitkv.lifetime-normalization.v1",
            plan_fingerprint: self.fingerprint(),
            page_tokens: self.page_tokens,
            normalized_classes: geometry.classes,
            normalized_bytes_per_request: geometry.bytes_per_request,
            max_window_baseline_bytes_per_request,
            savings_bytes_per_request,
            savings_percent_milli,
            retention_amplification_milli,
        })
    }

    /// Summarizes whether compiled lifetime semantics create a static KV
    /// residency opportunity at one logical boundary.
    ///
    /// The report is a block-geometry result. It deliberately does not predict
    /// kernel time, scheduler behavior, or end-to-end throughput.
    ///
    /// # Errors
    ///
    /// Returns an error if checked byte, layer, or percentage calculations
    /// overflow.
    pub fn applicability_report(&self, boundary: u64) -> Result<ApplicabilityReport, PlanError> {
        let layout = self.layout_program()?;
        let capacities = self.capacity_at(boundary)?;
        let bounded_class_count = u64::try_from(
            self.classes
                .iter()
                .filter(|class| class.spec.retention != RetentionKind::Full)
                .count(),
        )
        .map_err(|_| PlanError::ArithmeticOverflow {
            calculation: "bounded class count",
        })?;
        let lifetime_class_count =
            u64::try_from(self.classes.len()).map_err(|_| PlanError::ArithmeticOverflow {
                calculation: "lifetime class count",
            })?;
        let unbounded_class_count = lifetime_class_count - bounded_class_count;
        let applicability = self.classify_applicability(bounded_class_count, unbounded_class_count);
        let generated_layouts = layout_kinds(&layout);
        let geometry = self.applicability_geometry(capacities)?;
        let all_full_baseline_bytes = self.all_full_baseline_bytes_at(boundary)?;
        let reclaimable_baseline_bytes =
            all_full_baseline_bytes.saturating_sub(geometry.physical_resident_bytes);
        let static_reduction_percent_milli = checked_scaled_ratio(
            reclaimable_baseline_bytes,
            100_000,
            all_full_baseline_bytes,
            "applicability reduction percent",
        )?
        .unwrap_or(0);
        let bounded_resident_fraction_milli = checked_scaled_ratio(
            geometry.bounded_resident_bytes,
            1000,
            geometry.physical_resident_bytes,
            "bounded resident fraction",
        )?
        .unwrap_or(0);
        let physical_to_semantic_amplification_milli = checked_scaled_ratio(
            geometry.physical_resident_bytes,
            1000,
            geometry.semantically_live_bytes,
            "physical to semantic amplification",
        )?;
        Ok(ApplicabilityReport {
            schema: "orbitkv.applicability-report.v1",
            plan_fingerprint: self.fingerprint(),
            page_tokens: self.page_tokens,
            boundary_tokens: boundary,
            applicability,
            lifetime_class_count,
            bounded_class_count,
            unbounded_class_count,
            generated_layouts,
            classes: geometry.classes,
            semantically_live_bytes: geometry.semantically_live_bytes,
            physical_resident_bytes: geometry.physical_resident_bytes,
            all_full_baseline_bytes,
            reclaimable_baseline_bytes,
            static_reduction_percent_milli,
            bounded_resident_bytes: geometry.bounded_resident_bytes,
            bounded_resident_fraction_milli,
            physical_to_semantic_amplification_milli,
            claim_boundary: [
                "static equal-page KV geometry at the requested logical boundary",
                "not a kernel, scheduler, admission, or end-to-end speedup prediction",
                "unsupported retention semantics must fail closed before this report",
            ],
        })
    }

    fn classify_applicability(
        &self,
        bounded_class_count: u64,
        unbounded_class_count: u64,
    ) -> ApplicabilityClass {
        if self
            .classes
            .iter()
            .any(|class| class.kv_head_range.is_some())
        {
            return ApplicabilityClass::LifetimeNormalization;
        }
        if self
            .classes
            .iter()
            .any(|class| !class.block_domain.is_all())
        {
            return ApplicabilityClass::RegionSpecialization;
        }
        if bounded_class_count == 0 {
            return ApplicabilityClass::SafeFallback;
        }
        let bounded_signatures = self
            .classes
            .iter()
            .filter(|class| class.spec.retention != RetentionKind::Full)
            .map(|class| {
                (
                    class.spec.retention,
                    class.spec.window_tokens,
                    class.chunk_tokens,
                )
            })
            .collect::<BTreeSet<_>>();
        if unbounded_class_count == 0 && bounded_signatures.len() == 1 {
            ApplicabilityClass::UniformBounded
        } else {
            ApplicabilityClass::HybridLifetimes
        }
    }

    fn applicability_geometry(
        &self,
        capacities: Vec<ClassCapacity>,
    ) -> Result<ApplicabilityGeometry, PlanError> {
        let mut semantically_live_bytes = 0_u64;
        let mut classes = Vec::with_capacity(self.classes.len());
        for (class, capacity) in self.classes.iter().zip(capacities) {
            let layer_count = u64::try_from(class.spec.layers.len()).map_err(|_| {
                PlanError::ArithmeticOverflow {
                    calculation: "applicability layer count",
                }
            })?;
            let semantic_bytes = capacity
                .semantic_live_tokens
                .checked_mul(class.spec.bytes_per_token_per_layer)
                .and_then(|value| value.checked_mul(layer_count))
                .ok_or(PlanError::ArithmeticOverflow {
                    calculation: "semantically live bytes",
                })?;
            semantically_live_bytes = semantically_live_bytes.checked_add(semantic_bytes).ok_or(
                PlanError::ArithmeticOverflow {
                    calculation: "total semantically live bytes",
                },
            )?;
            classes.push(ApplicabilityClassGeometry {
                name: class.spec.name.clone(),
                retention: class.spec.retention,
                layer_count,
                kv_head_range: class.kv_head_range.clone(),
                semantic_live_tokens: capacity.semantic_live_tokens,
                physical_token_slots: capacity.physical_token_slots,
                resident_bytes: capacity.resident_bytes,
            });
        }
        let physical_resident_bytes = classes.iter().try_fold(0_u64, |total, class| {
            total
                .checked_add(class.resident_bytes)
                .ok_or(PlanError::ArithmeticOverflow {
                    calculation: "applicability resident bytes",
                })
        })?;
        let bounded_resident_bytes = classes
            .iter()
            .filter(|class| class.retention != RetentionKind::Full)
            .try_fold(0_u64, |total, class| {
                total
                    .checked_add(class.resident_bytes)
                    .ok_or(PlanError::ArithmeticOverflow {
                        calculation: "bounded resident bytes",
                    })
            })?;
        Ok(ApplicabilityGeometry {
            classes,
            semantically_live_bytes,
            physical_resident_bytes,
            bounded_resident_bytes,
        })
    }

    fn normalized_head_stripes(&self) -> Result<NormalizedHeadGeometry, PlanError> {
        let mut normalized_classes = Vec::with_capacity(self.classes.len());
        let mut normalized_bytes_per_request = 0_u64;
        let mut per_layer = BTreeMap::<u32, Vec<HeadStripe>>::new();
        for class in &self.classes {
            let range = class
                .kv_head_range
                .clone()
                .ok_or(PlanError::MissingKvHeadGeometry)?;
            let head_count = u64::from(range.end_exclusive - range.start);
            let slot_count = class
                .slot_count
                .ok_or_else(|| PlanError::InvalidCompiledClass {
                    class: class.spec.name.clone(),
                })?;
            if !class
                .spec
                .bytes_per_token_per_layer
                .is_multiple_of(head_count)
            {
                return Err(PlanError::InconsistentKvHeadWidth {
                    layer: class.spec.layers[0],
                });
            }
            let bytes_per_head_per_token = class.spec.bytes_per_token_per_layer / head_count;
            let class_bytes = slot_count
                .checked_mul(self.page_tokens)
                .and_then(|value| value.checked_mul(class.spec.bytes_per_token_per_layer))
                .and_then(|value| value.checked_mul(u64::try_from(class.spec.layers.len()).ok()?))
                .ok_or(PlanError::ArithmeticOverflow {
                    calculation: "lifetime-normalized class bytes",
                })?;
            normalized_bytes_per_request = normalized_bytes_per_request
                .checked_add(class_bytes)
                .ok_or(PlanError::ArithmeticOverflow {
                    calculation: "lifetime-normalized total bytes",
                })?;
            normalized_classes.push(LifetimeNormalizedClass {
                name: class.spec.name.clone(),
                layers: class.spec.layers.clone(),
                kv_head_range: range.clone(),
                head_count,
                slot_count,
                bytes_per_head_per_token,
                normalized_bytes_per_request: class_bytes,
            });
            for &layer in &class.spec.layers {
                per_layer.entry(layer).or_default().push((
                    range.clone(),
                    slot_count,
                    bytes_per_head_per_token,
                ));
            }
        }
        Ok(NormalizedHeadGeometry {
            classes: normalized_classes,
            per_layer,
            bytes_per_request: normalized_bytes_per_request,
        })
    }

    fn max_window_baseline_bytes(
        &self,
        per_layer: BTreeMap<u32, Vec<HeadStripe>>,
    ) -> Result<u64, PlanError> {
        per_layer
            .into_iter()
            .try_fold(0_u64, |total, (layer, mut stripes)| {
                stripes.sort_by_key(|(range, _, _)| range.start);
                let mut expected_head = 0_u32;
                let mut total_heads = 0_u64;
                let mut maximum_slots = 0_u64;
                let expected_width = stripes[0].2;
                for (range, slots, width) in stripes {
                    if range.start != expected_head {
                        return Err(PlanError::KvHeadCoverageGap {
                            layer,
                            expected: expected_head,
                            actual: range.start,
                        });
                    }
                    if width != expected_width {
                        return Err(PlanError::InconsistentKvHeadWidth { layer });
                    }
                    total_heads = total_heads
                        .checked_add(u64::from(range.end_exclusive - range.start))
                        .ok_or(PlanError::ArithmeticOverflow {
                            calculation: "KV head count",
                        })?;
                    maximum_slots = maximum_slots.max(slots);
                    expected_head = range.end_exclusive;
                }
                let layer_bytes = maximum_slots
                    .checked_mul(self.page_tokens)
                    .and_then(|value| value.checked_mul(total_heads))
                    .and_then(|value| value.checked_mul(expected_width))
                    .ok_or(PlanError::ArithmeticOverflow {
                        calculation: "max-window baseline bytes",
                    })?;
                total
                    .checked_add(layer_bytes)
                    .ok_or(PlanError::ArithmeticOverflow {
                        calculation: "total max-window baseline bytes",
                    })
            })
    }

    fn class_capacity_at(
        &self,
        class: &CompiledKvClass,
        boundary: u64,
    ) -> Result<ClassCapacity, PlanError> {
        let domain_start = class
            .block_domain
            .start_block
            .checked_mul(self.page_tokens)
            .ok_or(PlanError::ArithmeticOverflow {
                calculation: "class domain start token",
            })?;
        let domain_end = class
            .block_domain
            .end_block_exclusive
            .map(|end| {
                end.checked_mul(self.page_tokens)
                    .ok_or(PlanError::ArithmeticOverflow {
                        calculation: "class domain end token",
                    })
            })
            .transpose()?
            .unwrap_or(boundary);
        let live_start = match class.spec.retention {
            RetentionKind::Full => domain_start,
            RetentionKind::Sliding => {
                let window =
                    class
                        .spec
                        .window_tokens
                        .ok_or_else(|| PlanError::InvalidCompiledClass {
                            class: class.spec.name.clone(),
                        })?;
                domain_start.max(boundary.saturating_sub(window.saturating_sub(1)))
            }
            RetentionKind::Chunked => {
                let chunk = class
                    .chunk_tokens
                    .ok_or_else(|| PlanError::InvalidCompiledChunk {
                        class: class.spec.name.clone(),
                    })?;
                boundary / chunk * chunk
            }
        };
        let live_end = boundary.min(domain_end);
        let semantic_live_tokens = live_end.saturating_sub(live_start);
        let existing_blocks = ceil_div(boundary, self.page_tokens)?;
        let existing_domain_blocks = class.block_domain.blocks_before(existing_blocks);
        let slots = class.slot_count.map_or(existing_domain_blocks, |capacity| {
            existing_domain_blocks.min(capacity)
        });
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

    fn class_live_blocks(
        &self,
        class: &CompiledKvClass,
        boundary: u64,
    ) -> Result<Vec<u64>, PlanError> {
        let start_token = match class.spec.retention {
            RetentionKind::Full => 0,
            RetentionKind::Sliding => {
                let window =
                    class
                        .spec
                        .window_tokens
                        .ok_or_else(|| PlanError::InvalidCompiledClass {
                            class: class.spec.name.clone(),
                        })?;
                boundary.saturating_sub(window.saturating_sub(1))
            }
            RetentionKind::Chunked => {
                let chunk = class
                    .chunk_tokens
                    .ok_or_else(|| PlanError::InvalidCompiledChunk {
                        class: class.spec.name.clone(),
                    })?;
                boundary / chunk * chunk
            }
        };
        if start_token >= boundary {
            return Ok(Vec::new());
        }
        let first = (start_token / self.page_tokens).max(class.block_domain.start_block);
        let last = (boundary - 1) / self.page_tokens;
        let last = class
            .block_domain
            .end_block_exclusive
            .map_or(last, |end| last.min(end.saturating_sub(1)));
        if first > last {
            return Ok(Vec::new());
        }
        Ok((first..=last).collect())
    }
}

impl KvClassSpec {
    /// Desugars a validated canonical manager class into Retention IR.
    ///
    /// # Errors
    ///
    /// Returns an error if a sliding window does not fit the IR constant type.
    pub fn to_retention_state(&self) -> Result<RetentionStateDecl, PlanError> {
        let may_read = match self.retention {
            RetentionKind::Full => Predicate::True,
            RetentionKind::Sliding => Predicate::LessThan {
                lhs: IntExpr::Sub {
                    lhs: Box::new(IntExpr::QueryPosition),
                    rhs: Box::new(IntExpr::KeyPosition),
                },
                rhs: IntExpr::Constant {
                    value: i64::try_from(self.window_tokens.unwrap_or(0)).map_err(|_| {
                        PlanError::WindowOutOfRange {
                            class: self.name.clone(),
                            window: self.window_tokens.unwrap_or(0),
                        }
                    })?,
                },
            },
            RetentionKind::Chunked => return Err(PlanError::CanonicalChunkedUnsupported),
        };
        Ok(RetentionStateDecl {
            name: self.name.clone(),
            layers: self.layers.clone(),
            kv_head_range: None,
            bytes_per_token_per_layer: self.bytes_per_token_per_layer,
            may_read,
        })
    }
}

impl KvPlanInput {
    /// Desugars the validated canonical manager plan into Retention IR.
    ///
    /// # Errors
    ///
    /// Returns an error if a window cannot be represented by the IR.
    pub fn into_retention_program(self) -> Result<RetentionProgramInput, PlanError> {
        Ok(RetentionProgramInput {
            schema: "orbitkv.retention-ir.v1".into(),
            page_tokens: self.page_tokens,
            states: self
                .classes
                .into_iter()
                .map(|class| class.to_retention_state())
                .collect::<Result<Vec<_>, PlanError>>()?,
        })
    }
}

mod compiler;

pub use compiler::{compile_plan, compile_retention_program};

fn layout_kinds(layout: &LayoutProgram) -> Vec<&'static str> {
    let mut kinds = Vec::new();
    for class in &layout.classes {
        let kind = match class.address {
            AddressProgram::AppendOnly => "append_only",
            AddressProgram::Pinned => "pinned",
            AddressProgram::Periodic { .. } | AddressProgram::PeriodicFrom { .. } => "periodic",
            AddressProgram::ResettableArena { .. } => "resettable_arena",
        };
        if !kinds.contains(&kind) {
            kinds.push(kind);
        }
    }
    kinds
}

fn checked_scaled_ratio(
    numerator: u64,
    scale: u64,
    denominator: u64,
    calculation: &'static str,
) -> Result<Option<u64>, PlanError> {
    numerator
        .checked_mul(scale)
        .ok_or(PlanError::ArithmeticOverflow { calculation })
        .map(|scaled| scaled.checked_div(denominator))
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
mod tests;
