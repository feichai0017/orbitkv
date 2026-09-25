use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{StateComponent, TokenRange};

/// Persistence needed by a registered group to resume at a token boundary.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecoveryRule {
    Prefix,
    Window { tokens: u64 },
    Checkpoint,
}

/// All components in a group are stored and leased together by the data plane.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateRequirement {
    pub group: u32,
    pub components: BTreeSet<StateComponent>,
    pub rule: RecoveryRule,
}

/// Exact group pages held by a query lease, identified by absolute token ends.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BundleComponent {
    pub group: u32,
    pub page_ends: Vec<u64>,
}

/// Evidence for a queried tail. The engine must hold a valid prefix at span.start.
/// Page identity and lease provenance must already be checked by the data plane.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateBundle {
    pub namespace: String,
    pub span: TokenRange,
    pub components: Vec<BundleComponent>,
}

/// Complete page demand at an engine-selected recovery boundary.
/// The existing query instance and session bind this demand to registered state;
/// it does not carry a second namespace or own engine destinations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecoveryDemand {
    pub page_tokens: u64,
    pub span: TokenRange,
    pub groups: Vec<(u32, TokenRange)>,
}

impl RecoveryDemand {
    /// Validate the declared ranges and the selected group's transmitted hashes.
    /// The Manager additionally checks the exact registered group set.
    pub fn validate(&self, group: u32, blocks: usize) -> Result<(), RecoveryError> {
        if self.page_tokens == 0
            || self.span.end < self.span.start
            || !self.span.start.is_multiple_of(self.page_tokens)
            || !self.span.end.is_multiple_of(self.page_tokens)
        {
            return Err(RecoveryError::InvalidSpan);
        }
        if self.groups.first() != Some(&(0, self.span)) {
            return Err(RecoveryError::InvalidGroup(0));
        }
        let mut previous = None;
        let mut selected = None;
        for &(id, range) in &self.groups {
            if previous.is_some_and(|previous| id <= previous) {
                return Err(RecoveryError::InvalidGroup(id));
            }
            previous = Some(id);
            if range.start < self.span.start
                || range.start > range.end
                || range.end != self.span.end
                || !range.start.is_multiple_of(self.page_tokens)
                || (range.is_empty() && !self.span.is_empty())
            {
                return Err(RecoveryError::InvalidCoverage(id));
            }
            if id == group {
                selected = Some((range.end - range.start) / self.page_tokens);
            }
        }
        let pages = selected.ok_or(RecoveryError::InvalidGroup(group))?;
        if u64::try_from(blocks).ok() != Some(pages) {
            return Err(RecoveryError::InvalidCoverage(group));
        }
        Ok(())
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum RecoveryError {
    #[error("invalid recovery contract: {0}")]
    InvalidContract(&'static str),
    #[error("recovery evidence belongs to a different model/storage namespace")]
    IncompatibleNamespace,
    #[error("recovery span must be ordered and aligned to the registered page size")]
    InvalidSpan,
    #[error("missing, duplicate, or unknown recovery group {0}")]
    InvalidGroup(u32),
    #[error("group {0} page ends must be unique, ordered, aligned and within the queried tail")]
    InvalidCoverage(u32),
}

/// Compiles engine-declared state requirements into page demand and validation.
/// This does not analyze the model's numerical implementation or predict requests.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecoveryContract {
    namespace: String,
    page_tokens: u64,
    // None retains the full queried tail; Some(n) retains its last n pages.
    requirements: BTreeMap<u32, Option<u64>>,
}

impl RecoveryContract {
    pub fn compile(
        namespace: String,
        page_tokens: u64,
        requirements: Vec<StateRequirement>,
    ) -> Result<Self, RecoveryError> {
        if namespace.is_empty() || page_tokens == 0 || requirements.is_empty() {
            return Err(RecoveryError::InvalidContract(
                "empty identity, page size or requirements",
            ));
        }
        let mut rules = BTreeMap::new();
        for requirement in requirements {
            if requirement.components.is_empty() {
                return Err(RecoveryError::InvalidContract(
                    "group has no state components",
                ));
            }
            for component in &requirement.components {
                let compatible = matches!(
                    (&requirement.rule, component),
                    (
                        RecoveryRule::Prefix,
                        StateComponent::AttentionKv | StateComponent::MlaKv
                    ) | (RecoveryRule::Window { .. }, StateComponent::SlidingWindowKv)
                        | (
                            RecoveryRule::Checkpoint,
                            StateComponent::RecurrentCheckpoint | StateComponent::ConvolutionState
                        )
                );
                if !compatible {
                    return Err(RecoveryError::InvalidContract(
                        "unsupported component/recovery rule",
                    ));
                }
            }
            if let RecoveryRule::Window { tokens } = requirement.rule
                && (tokens == 0 || tokens.checked_add(page_tokens - 1).is_none())
            {
                return Err(RecoveryError::InvalidContract("invalid sliding window"));
            }
            let pages = match requirement.rule {
                RecoveryRule::Prefix => None,
                RecoveryRule::Window { tokens } => Some(tokens.div_ceil(page_tokens)),
                RecoveryRule::Checkpoint => Some(1),
            };
            if rules.insert(requirement.group, pages).is_some() {
                return Err(RecoveryError::InvalidGroup(requirement.group));
            }
        }
        if rules.get(&0) != Some(&None) {
            return Err(RecoveryError::InvalidContract(
                "group zero must supply the attention prefix",
            ));
        }
        Ok(Self {
            namespace,
            page_tokens,
            requirements: rules,
        })
    }

    /// Minimal page-aligned demand per group for one declared recovery boundary.
    /// The engine must retain valid state at span.start. These ranges describe
    /// required state, not its availability, leases, or permission to reclaim it.
    pub fn required_ranges(
        &self,
        namespace: &str,
        span: TokenRange,
    ) -> Result<Vec<(u32, TokenRange)>, RecoveryError> {
        self.validate_span(namespace, span)?;
        Ok(self
            .requirements
            .iter()
            .map(|(&group, &pages)| (group, self.required_span(span, pages)))
            .collect())
    }

    /// Carry all compiled group ranges with each selected group read.
    pub fn demand(
        &self,
        namespace: &str,
        span: TokenRange,
    ) -> Result<RecoveryDemand, RecoveryError> {
        Ok(RecoveryDemand {
            page_tokens: self.page_tokens,
            span,
            groups: self.required_ranges(namespace, span)?,
        })
    }

    /// Translate the compiled demand into a slice of the original hash batch.
    pub fn read_range(
        &self,
        namespace: &str,
        span: TokenRange,
        group: u32,
        batch_len: usize,
    ) -> Result<std::ops::Range<usize>, RecoveryError> {
        self.validate_span(namespace, span)?;
        let pages = *self
            .requirements
            .get(&group)
            .ok_or(RecoveryError::InvalidGroup(group))?;
        let needed = self.required_span(span, pages);
        let start = usize::try_from((needed.start - span.start) / self.page_tokens)
            .map_err(|_| RecoveryError::InvalidSpan)?;
        let end = usize::try_from((needed.end - span.start) / self.page_tokens)
            .map_err(|_| RecoveryError::InvalidSpan)?;
        if end > batch_len {
            return Err(RecoveryError::InvalidSpan);
        }
        Ok(start..end)
    }

    /// Keep page expansion and cross-rank intersection out of engine Python loops.
    /// Candidate evidence is provisional; use read_range and validate the leases
    /// again before admitting a selected boundary.
    pub fn common_boundaries(
        &self,
        namespace: &str,
        span: TokenRange,
        shards: &[Vec<(u32, Vec<u32>)>],
    ) -> Result<Vec<u64>, RecoveryError> {
        self.validate_span(namespace, span)?;
        let mut common: Option<BTreeSet<u64>> = None;
        for groups in shards {
            let components = groups
                .iter()
                .map(|(group, positions)| {
                    let page_ends = positions
                        .iter()
                        .map(|&position| {
                            u64::from(position)
                                .checked_add(1)
                                .and_then(|n| n.checked_mul(self.page_tokens))
                                .and_then(|n| n.checked_add(span.start))
                                .ok_or(RecoveryError::InvalidCoverage(*group))
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok(BundleComponent {
                        group: *group,
                        page_ends,
                    })
                })
                .collect::<Result<Vec<_>, RecoveryError>>()?;
            let legal: BTreeSet<_> = self
                .restorable_boundaries(&StateBundle {
                    namespace: namespace.into(),
                    span,
                    components,
                })?
                .into_iter()
                .collect();
            if let Some(common) = &mut common {
                common.retain(|end| legal.contains(end));
            } else {
                common = Some(legal);
            }
        }
        Ok(common.unwrap_or_default().into_iter().collect())
    }

    /// Select one rank-common boundary under the engine's usable-token limit.
    /// Returning a scalar avoids exporting every candidate to Python for vLLM.
    pub fn select_boundary(
        &self,
        namespace: &str,
        span: TokenRange,
        shards: &[Vec<(u32, Vec<u32>)>],
        limit: u64,
    ) -> Result<Option<u64>, RecoveryError> {
        Ok(self
            .common_boundaries(namespace, span, shards)?
            .into_iter()
            .rfind(|&end| end <= limit))
    }

    fn required_span(&self, span: TokenRange, pages: Option<u64>) -> TokenRange {
        let count = ((span.end - span.start) / self.page_tokens).min(pages.unwrap_or(u64::MAX));
        TokenRange {
            start: span.end - count * self.page_tokens,
            end: span.end,
        }
    }

    fn validate_span(&self, namespace: &str, span: TokenRange) -> Result<(), RecoveryError> {
        if namespace != self.namespace {
            return Err(RecoveryError::IncompatibleNamespace);
        }
        if span.end < span.start
            || !span.start.is_multiple_of(self.page_tokens)
            || !span.end.is_multiple_of(self.page_tokens)
        {
            return Err(RecoveryError::InvalidSpan);
        }
        Ok(())
    }

    /// Return every legal boundary: checkpoint/window hits need not be dense.
    /// Intersect these sets across ranks; taking the minimum of maxima is unsafe.
    pub fn restorable_boundaries(&self, bundle: &StateBundle) -> Result<Vec<u64>, RecoveryError> {
        let span = bundle.span;
        self.validate_span(&bundle.namespace, span)?;
        let mut coverage = BTreeMap::new();
        for component in &bundle.components {
            if !self.requirements.contains_key(&component.group)
                || coverage
                    .insert(component.group, component.page_ends.as_slice())
                    .is_some()
            {
                return Err(RecoveryError::InvalidGroup(component.group));
            }
            let mut previous = span.start;
            for &end in &component.page_ends {
                if end <= previous || end > span.end || !end.is_multiple_of(self.page_tokens) {
                    return Err(RecoveryError::InvalidCoverage(component.group));
                }
                previous = end;
            }
        }
        for group in self.requirements.keys() {
            if !coverage.contains_key(group) {
                return Err(RecoveryError::InvalidGroup(*group));
            }
        }
        let mut result = Vec::new();
        for (index, &boundary) in coverage[&0].iter().enumerate() {
            if (boundary - span.start) / self.page_tokens != index as u64 + 1 {
                break;
            }
            let complete = self.requirements.iter().all(|(group, &pages)| {
                let ends = coverage[group];
                let needed = self.required_span(
                    TokenRange {
                        start: span.start,
                        end: boundary,
                    },
                    pages,
                );
                let first = ends.partition_point(|&end| end <= needed.start);
                let last = ends.partition_point(|&end| end <= needed.end);
                (last - first) as u64 == (needed.end - needed.start) / self.page_tokens
            });
            if complete {
                result.push(boundary);
            }
        }
        Ok(result)
    }
}

#[cfg(test)]
#[path = "../tests/unit/bundle.rs"]
mod tests;
