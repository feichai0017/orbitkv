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

/// Compiles engine-declared state requirements once at registration.
/// This validates recovery evidence, not the model's numerical implementation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecoveryContract {
    namespace: String,
    page_tokens: u64,
    requirements: BTreeMap<u32, RecoveryRule>,
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
            if rules.insert(requirement.group, requirement.rule).is_some() {
                return Err(RecoveryError::InvalidGroup(requirement.group));
            }
        }
        if rules.get(&0) != Some(&RecoveryRule::Prefix) {
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

    /// Return every legal boundary: checkpoint/window hits need not be dense.
    /// Intersect these sets across ranks; taking the minimum of maxima is unsafe.
    pub fn restorable_boundaries(&self, bundle: &StateBundle) -> Result<Vec<u64>, RecoveryError> {
        if bundle.namespace != self.namespace {
            return Err(RecoveryError::IncompatibleNamespace);
        }
        let span = bundle.span;
        if span.end < span.start
            || !span.start.is_multiple_of(self.page_tokens)
            || !span.end.is_multiple_of(self.page_tokens)
        {
            return Err(RecoveryError::InvalidSpan);
        }
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
            let complete = self.requirements.iter().all(|(group, rule)| {
                let ends = coverage[group];
                match rule {
                    RecoveryRule::Checkpoint => ends.binary_search(&boundary).is_ok(),
                    RecoveryRule::Prefix | RecoveryRule::Window { .. } => {
                        let count = match rule {
                            RecoveryRule::Window { tokens } => tokens.div_ceil(self.page_tokens),
                            _ => (boundary - span.start) / self.page_tokens,
                        }
                        .min((boundary - span.start) / self.page_tokens);
                        let begin = boundary - count * self.page_tokens;
                        let first = ends.partition_point(|&end| end <= begin);
                        let last = ends.partition_point(|&end| end <= boundary);
                        last - first == count as usize
                    }
                }
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
