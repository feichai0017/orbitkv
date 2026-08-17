use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::retention::{
    AtomicRetention, InferredRegion, InferredRetention, IntExpr, KvHeadRange, Predicate,
    RetentionAnalysis, RetentionError, RetentionProgramInput, RetentionStateDecl, analyze_state,
};

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
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

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(untagged)]
pub enum KvPlanSource {
    Retention(RetentionProgramInput),
    Legacy(KvPlanInput),
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

type HeadStripe = (KvHeadRange, u64, u64);

struct NormalizedHeadGeometry {
    classes: Vec<LifetimeNormalizedClass>,
    per_layer: BTreeMap<u32, Vec<HeadStripe>>,
    bytes_per_request: u64,
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

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
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

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct CellVersion {
    pub cycle: u64,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct TemporalAddress {
    pub cell: LogicalCellId,
    pub version: CellVersion,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct BlockDomain {
    pub start_block: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_block_exclusive: Option<u64>,
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
    #[error("boundary must be non-negative")]
    InvalidBoundary,
    #[error("integer overflow while calculating {calculation}")]
    ArithmeticOverflow { calculation: &'static str },
    #[error("compiled sliding class {class:?} is missing its window")]
    InvalidCompiledClass { class: String },
    #[error("compiled chunked class {class:?} is missing its chunk size")]
    InvalidCompiledChunk { class: String },
    #[error(
        "SGLang eviction interval {interval} must be a positive multiple of page_tokens {page_tokens}"
    )]
    InvalidSglangEvictionInterval { interval: u64, page_tokens: u64 },
    #[error("backend logical_bytes must be positive")]
    ZeroBackendBytes,
    #[error("CUDA VMM granularity must be positive when VMM is supported")]
    ZeroVmmGranularity,
    #[error("periodic address program must have a positive period")]
    ZeroAddressPeriod,
    #[error("layout program does not contain class {0:?}")]
    UnknownLayoutClass(String),
    #[error("logical block {ordinal} is outside class {class:?} block domain")]
    AddressOutsideBlockDomain { class: String, ordinal: u64 },
    #[error("sink boundary {sink_tokens} must be aligned to page_tokens {page_tokens}")]
    SinkBoundaryNotPageAligned { sink_tokens: u64, page_tokens: u64 },
    #[error("SGLang lowering does not support partitioned block domains")]
    UnsupportedSglangBlockDomain,
    #[error("SGLang lowering does not support retention class {0:?}")]
    UnsupportedSglangRetention(String),
    #[error("legacy syntax cannot declare chunked retention; use Retention IR")]
    LegacyChunkedUnsupported,
    #[error("chunk size {chunk_tokens} must be aligned to page_tokens {page_tokens}")]
    ChunkNotPageAligned { chunk_tokens: u64, page_tokens: u64 },
    #[error("{class}: legacy window {window} does not fit Retention IR i64 constants")]
    LegacyWindowOutOfRange { class: String, window: u64 },
    #[error(transparent)]
    Retention(#[from] RetentionError),
}

impl AddressProgram {
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
        Ok(TemporalAddress {
            cell: LogicalCellId {
                request_id: request_id.to_owned(),
                class_name: class_name.to_owned(),
                cell_index,
            },
            version: CellVersion { cycle },
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
        if let Some(class) = self
            .classes
            .iter()
            .find(|class| class.spec.retention == RetentionKind::Chunked)
        {
            return Err(PlanError::UnsupportedSglangRetention(
                class.spec.name.clone(),
            ));
        }
        if let Some(class) = self
            .classes
            .iter()
            .find(|class| class.kv_head_range.is_some())
        {
            return Err(PlanError::UnsupportedSglangRetention(
                class.spec.name.clone(),
            ));
        }
        let bounded_classes =
            self.classes
                .iter()
                .filter_map(|class| class.slot_count.map(|slots| (class, slots)))
                .map(|(class, block_slots)| {
                    if !class.block_domain.is_all() {
                        return Err(PlanError::UnsupportedSglangBlockDomain);
                    }
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

impl KvClassSpec {
    /// Desugars a validated legacy class into the declarative Retention IR.
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
                        PlanError::LegacyWindowOutOfRange {
                            class: self.name.clone(),
                            window: self.window_tokens.unwrap_or(0),
                        }
                    })?,
                },
            },
            RetentionKind::Chunked => return Err(PlanError::LegacyChunkedUnsupported),
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
    /// Desugars validated legacy syntax into the declarative Retention IR.
    ///
    /// # Errors
    ///
    /// Returns an error if a legacy window cannot be represented by the IR.
    pub fn to_retention_program(&self) -> Result<RetentionProgramInput, PlanError> {
        Ok(RetentionProgramInput {
            schema: "orbitkv.retention-ir.v1".into(),
            page_tokens: self.page_tokens,
            states: self
                .classes
                .iter()
                .map(KvClassSpec::to_retention_state)
                .collect::<Result<Vec<_>, PlanError>>()?,
        })
    }
}

impl KvPlanSource {
    /// Converts either frontend into the declarative Retention IR.
    ///
    /// # Errors
    ///
    /// Returns an error if a legacy class is invalid or cannot be represented.
    pub fn into_retention_program(self) -> Result<RetentionProgramInput, PlanError> {
        match self {
            Self::Retention(program) => Ok(program),
            Self::Legacy(input) => {
                validate_plan_input(&input)?;
                input.to_retention_program()
            }
        }
    }

    /// Compiles either the declarative Retention IR or the legacy Full/SWA
    /// syntax through the same lifetime compiler.
    ///
    /// # Errors
    ///
    /// Returns an error for an unsupported schema, invalid legacy input, or
    /// failed lifetime inference.
    pub fn compile(self) -> Result<CompiledKvPlan, PlanError> {
        compile_retention_program(self.into_retention_program()?)
    }
}

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
    KvPlanSource::Legacy(input).compile()
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

    #[test]
    fn periodic_address_program_derives_cell_and_cycle() {
        let address = AddressProgram::Periodic { period_blocks: 65 }
            .evaluate("request-7", "swa", 131)
            .unwrap();
        assert_eq!(address.cell.request_id, "request-7");
        assert_eq!(address.cell.class_name, "swa");
        assert_eq!(address.cell.cell_index, 1);
        assert_eq!(address.version.cycle, 2);
    }

    #[test]
    fn retirement_program_matches_sliding_death_formula() {
        assert_eq!(
            RetirementProgram::BlockEndPlus {
                offset_tokens: 1023
            }
            .death_boundary(16, 0)
            .unwrap(),
            Some(1039)
        );
        assert_eq!(
            RetirementProgram::Never
                .death_boundary(16, u64::MAX)
                .unwrap(),
            None
        );
    }

    #[test]
    fn legacy_and_retention_ir_compile_identically() {
        let legacy = KvPlanInput {
            page_tokens: 16,
            classes: vec![
                KvClassSpec {
                    name: "full".into(),
                    layers: vec![0],
                    retention: RetentionKind::Full,
                    bytes_per_token_per_layer: 128,
                    window_tokens: None,
                },
                KvClassSpec {
                    name: "swa".into(),
                    layers: vec![1],
                    retention: RetentionKind::Sliding,
                    bytes_per_token_per_layer: 128,
                    window_tokens: Some(1024),
                },
            ],
        };
        let legacy_plan = compile_plan(legacy.clone()).unwrap();
        let retention_plan =
            compile_retention_program(legacy.to_retention_program().unwrap()).unwrap();
        assert_eq!(legacy_plan, retention_plan);
        assert_eq!(
            legacy_plan.layout_program().unwrap(),
            retention_plan.layout_program().unwrap()
        );
        assert_eq!(legacy_plan.fingerprint(), retention_plan.fingerprint());
    }

    #[test]
    fn sink_sliding_relation_synthesizes_pinned_and_periodic_regions() {
        let program = RetentionProgramInput {
            schema: "orbitkv.retention-ir.v1".into(),
            page_tokens: 4,
            states: vec![RetentionStateDecl {
                name: "attention".into(),
                layers: vec![0],
                kv_head_range: None,
                bytes_per_token_per_layer: 128,
                may_read: Predicate::Or {
                    terms: vec![
                        Predicate::LessThan {
                            lhs: IntExpr::KeyPosition,
                            rhs: IntExpr::Constant { value: 4 },
                        },
                        Predicate::LessThan {
                            lhs: IntExpr::Sub {
                                lhs: Box::new(IntExpr::QueryPosition),
                                rhs: Box::new(IntExpr::KeyPosition),
                            },
                            rhs: IntExpr::Constant { value: 8 },
                        },
                    ],
                },
            }],
        };
        let plan = compile_retention_program(program).unwrap();
        assert_eq!(plan.classes.len(), 2);
        assert_eq!(plan.classes[0].spec.name, "attention::sink");
        assert_eq!(plan.classes[0].slot_count, Some(1));
        assert_eq!(
            plan.classes[0].block_domain,
            BlockDomain {
                start_block: 0,
                end_block_exclusive: Some(1)
            }
        );
        assert_eq!(plan.classes[1].spec.name, "attention::local");
        assert_eq!(plan.classes[1].slot_count, Some(3));
        assert_eq!(plan.classes[1].block_domain.start_block, 1);

        let layout = plan.layout_program().unwrap();
        assert_eq!(layout.classes[0].address, AddressProgram::Pinned);
        assert_eq!(
            layout.classes[1].address,
            AddressProgram::PeriodicFrom {
                period_blocks: 3,
                origin_block: 1
            }
        );
        let continuation = plan.continuation_blocks(20).unwrap();
        assert_eq!(continuation["attention::sink"], vec![0]);
        assert_eq!(continuation["attention::local"], vec![3, 4]);
        let capacity = plan.capacity_at(20).unwrap();
        assert_eq!(capacity[0].semantic_live_tokens, 4);
        assert_eq!(capacity[0].physical_token_slots, 4);
        assert_eq!(capacity[1].semantic_live_tokens, 7);
        assert_eq!(capacity[1].physical_token_slots, 12);
        assert_eq!(plan.all_full_baseline_bytes_at(20).unwrap(), 20 * 128);
        assert_eq!(
            plan.sglang_policy(),
            Err(PlanError::UnsupportedSglangBlockDomain)
        );

        assert!(matches!(
            layout.temporal_address("request", "attention::sink", 1),
            Err(PlanError::AddressOutsideBlockDomain {
                class,
                ordinal: 1
            }) if class == "attention::sink"
        ));
        assert!(matches!(
            layout.temporal_address("request", "attention::local", 0),
            Err(PlanError::AddressOutsideBlockDomain {
                class,
                ordinal: 0
            }) if class == "attention::local"
        ));
    }

    #[test]
    fn sink_boundary_must_align_with_reclamation_page() {
        let program = RetentionProgramInput {
            schema: "orbitkv.retention-ir.v1".into(),
            page_tokens: 4,
            states: vec![RetentionStateDecl {
                name: "attention".into(),
                layers: vec![0],
                kv_head_range: None,
                bytes_per_token_per_layer: 128,
                may_read: Predicate::Or {
                    terms: vec![
                        Predicate::LessThan {
                            lhs: IntExpr::KeyPosition,
                            rhs: IntExpr::Constant { value: 3 },
                        },
                        Predicate::LessThan {
                            lhs: IntExpr::Sub {
                                lhs: Box::new(IntExpr::QueryPosition),
                                rhs: Box::new(IntExpr::KeyPosition),
                            },
                            rhs: IntExpr::Constant { value: 8 },
                        },
                    ],
                },
            }],
        };
        assert!(matches!(
            compile_retention_program(program),
            Err(PlanError::SinkBoundaryNotPageAligned {
                sink_tokens: 3,
                page_tokens: 4
            })
        ));
    }

    #[test]
    fn same_chunk_relation_synthesizes_resettable_arena() {
        let program = RetentionProgramInput {
            schema: "orbitkv.retention-ir.v1".into(),
            page_tokens: 4,
            states: vec![RetentionStateDecl {
                name: "chunked".into(),
                layers: vec![0],
                kv_head_range: None,
                bytes_per_token_per_layer: 128,
                may_read: Predicate::Equal {
                    lhs: IntExpr::FloorDiv {
                        value: Box::new(IntExpr::QueryPosition),
                        divisor: 16,
                    },
                    rhs: IntExpr::FloorDiv {
                        value: Box::new(IntExpr::KeyPosition),
                        divisor: 16,
                    },
                },
            }],
        };
        let plan = compile_retention_program(program).unwrap();
        assert_eq!(plan.classes[0].spec.retention, RetentionKind::Chunked);
        assert_eq!(plan.classes[0].chunk_tokens, Some(16));
        assert_eq!(plan.classes[0].slot_count, Some(4));
        let layout = plan.layout_program().unwrap();
        assert_eq!(
            layout.classes[0].address,
            AddressProgram::ResettableArena {
                blocks_per_epoch: 4
            }
        );
        assert_eq!(
            layout.classes[0].retirement,
            RetirementProgram::EpochEnd {
                blocks_per_epoch: 4
            }
        );
        for ordinal in 0..4 {
            assert_eq!(
                layout.classes[0]
                    .retirement
                    .death_boundary(4, ordinal)
                    .unwrap(),
                Some(16)
            );
        }
        assert_eq!(
            layout.classes[0].retirement.death_boundary(4, 4).unwrap(),
            Some(32)
        );
        assert_eq!(plan.continuation_blocks(20).unwrap()["chunked"], vec![4]);
        assert_eq!(plan.capacity_at(20).unwrap()[0].semantic_live_tokens, 4);
        assert!(matches!(
            plan.sglang_policy(),
            Err(PlanError::UnsupportedSglangRetention(class)) if class == "chunked"
        ));
    }

    #[test]
    fn chunk_size_must_align_with_reclamation_page() {
        let program = RetentionProgramInput {
            schema: "orbitkv.retention-ir.v1".into(),
            page_tokens: 4,
            states: vec![RetentionStateDecl {
                name: "chunked".into(),
                layers: vec![0],
                kv_head_range: None,
                bytes_per_token_per_layer: 128,
                may_read: Predicate::Equal {
                    lhs: IntExpr::FloorDiv {
                        value: Box::new(IntExpr::QueryPosition),
                        divisor: 10,
                    },
                    rhs: IntExpr::FloorDiv {
                        value: Box::new(IntExpr::KeyPosition),
                        divisor: 10,
                    },
                },
            }],
        };
        assert_eq!(
            compile_retention_program(program),
            Err(PlanError::ChunkNotPageAligned {
                chunk_tokens: 10,
                page_tokens: 4
            })
        );
    }

    #[test]
    fn lifetime_normalization_matches_multi_scale_head_theory() {
        let state =
            |name: &str, start: u32, end_exclusive: u32, window: i64| -> RetentionStateDecl {
                RetentionStateDecl {
                    name: name.into(),
                    layers: vec![0],
                    kv_head_range: Some(KvHeadRange {
                        start,
                        end_exclusive,
                    }),
                    bytes_per_token_per_layer: u64::from(end_exclusive - start) * 512,
                    may_read: Predicate::LessThan {
                        lhs: IntExpr::Sub {
                            lhs: Box::new(IntExpr::QueryPosition),
                            rhs: Box::new(IntExpr::KeyPosition),
                        },
                        rhs: IntExpr::Constant { value: window },
                    },
                }
            };
        let plan = compile_retention_program(RetentionProgramInput {
            schema: "orbitkv.retention-ir.v1".into(),
            page_tokens: 16,
            states: vec![
                state("w512", 0, 8, 512),
                state("w2048", 8, 16, 2048),
                state("w8192", 16, 32, 8192),
            ],
        })
        .unwrap();
        assert_eq!(plan.classes[0].slot_count, Some(33));
        assert_eq!(plan.classes[1].slot_count, Some(129));
        assert_eq!(plan.classes[2].slot_count, Some(513));
        let report = plan.lifetime_normalization_report().unwrap();
        let unit_bytes = 16 * 512;
        assert_eq!(report.normalized_bytes_per_request, 9504 * unit_bytes);
        assert_eq!(
            report.max_window_baseline_bytes_per_request,
            16416 * unit_bytes
        );
        assert_eq!(report.savings_bytes_per_request, 6912 * unit_bytes);
        assert_eq!(report.savings_percent_milli, 42_105);
        assert_eq!(report.retention_amplification_milli, 1727);
    }

    #[test]
    fn overlapping_head_ranges_fail_closed() {
        let state = |name: &str, start: u32, end_exclusive: u32| RetentionStateDecl {
            name: name.into(),
            layers: vec![0],
            kv_head_range: Some(KvHeadRange {
                start,
                end_exclusive,
            }),
            bytes_per_token_per_layer: u64::from(end_exclusive - start) * 512,
            may_read: Predicate::LessThan {
                lhs: IntExpr::Sub {
                    lhs: Box::new(IntExpr::QueryPosition),
                    rhs: Box::new(IntExpr::KeyPosition),
                },
                rhs: IntExpr::Constant { value: 512 },
            },
        };
        assert!(matches!(
            compile_retention_program(RetentionProgramInput {
                schema: "orbitkv.retention-ir.v1".into(),
                page_tokens: 16,
                states: vec![state("a", 0, 8), state("b", 4, 12)],
            }),
            Err(PlanError::KvHeadOverlap {
                layer: 0,
                first,
                second
            }) if first == "a" && second == "b"
        ));
    }

    #[test]
    fn partitioned_retention_rejects_zero_page_tokens_before_lowering() {
        let program = RetentionProgramInput {
            schema: "orbitkv.retention-ir.v1".into(),
            page_tokens: 0,
            states: vec![RetentionStateDecl {
                name: "attention".into(),
                layers: vec![0],
                kv_head_range: None,
                bytes_per_token_per_layer: 128,
                may_read: Predicate::True,
            }],
        };
        assert_eq!(
            compile_retention_program(program),
            Err(PlanError::ZeroPageTokens)
        );
    }

    #[test]
    fn sink_sliding_continuation_matches_declared_relation_exhaustively() {
        for page_tokens in [1_u64, 2, 4, 8] {
            for window_tokens in 1_u64..=17 {
                let sink_tokens = 2 * page_tokens;
                let declaration = RetentionStateDecl {
                    name: "attention".into(),
                    layers: vec![0],
                    kv_head_range: None,
                    bytes_per_token_per_layer: 128,
                    may_read: Predicate::Or {
                        terms: vec![
                            Predicate::LessThan {
                                lhs: IntExpr::KeyPosition,
                                rhs: IntExpr::Constant {
                                    value: i64::try_from(sink_tokens).unwrap(),
                                },
                            },
                            Predicate::LessThan {
                                lhs: IntExpr::Sub {
                                    lhs: Box::new(IntExpr::QueryPosition),
                                    rhs: Box::new(IntExpr::KeyPosition),
                                },
                                rhs: IntExpr::Constant {
                                    value: i64::try_from(window_tokens).unwrap(),
                                },
                            },
                        ],
                    },
                };
                let plan = compile_retention_program(RetentionProgramInput {
                    schema: "orbitkv.retention-ir.v1".into(),
                    page_tokens,
                    states: vec![declaration.clone()],
                })
                .unwrap();
                let layout = plan.layout_program().unwrap();

                for boundary in 0_u64..=8 * page_tokens + 2 * window_tokens {
                    let continuation = plan.continuation_blocks(boundary).unwrap();
                    let mut expected_sink = BTreeSet::new();
                    let mut expected_local = BTreeSet::new();
                    for key in 0..boundary {
                        if !declaration.may_read.may_read(
                            i64::try_from(boundary).unwrap(),
                            i64::try_from(key).unwrap(),
                        ) {
                            continue;
                        }
                        let block = key / page_tokens;
                        if key < sink_tokens {
                            expected_sink.insert(block);
                        } else {
                            expected_local.insert(block);
                        }
                    }
                    assert_eq!(
                        continuation["attention::sink"],
                        expected_sink.into_iter().collect::<Vec<_>>()
                    );
                    assert_eq!(
                        continuation["attention::local"],
                        expected_local.iter().copied().collect::<Vec<_>>()
                    );

                    let mut live_cells = BTreeSet::new();
                    for ordinal in expected_local {
                        let address = layout
                            .temporal_address("request", "attention::local", ordinal)
                            .unwrap();
                        assert!(
                            live_cells.insert(address.cell.cell_index),
                            "live local blocks collided at page={page_tokens}, window={window_tokens}, boundary={boundary}"
                        );
                    }
                }
            }
        }
    }
}
