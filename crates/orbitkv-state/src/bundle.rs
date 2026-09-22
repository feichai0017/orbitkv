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
