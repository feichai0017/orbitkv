//! Declarative admission for attention providers. These records generate
//! egglog eligibility rules; Rust does not match or replace model subgraphs.
//! Launch/workspace ownership remains in each provider's HostOp implementation.

use std::ops::RangeInclusive;

use orbitkv_compiler::{dtype::DType, egglog_utils::api::Rule};
use orbitkv_ops::ops::attention::{
    ATTENTION_DECLARATIONS, AttentionMask, AttentionSpec, PagedKvLayout,
};

const CAPABILITY_DECLARATIONS: &str = include_str!("attention/declarations.egg");

/// Translation of the CUDA providers' window ABI into logical visibility.
pub(crate) fn mask_from_window(window_left: i64) -> Option<AttentionMask> {
    if window_left == -1 {
        Some(AttentionMask::Causal)
    } else {
        usize::try_from(window_left)
            .ok()
            .map(|window_left| AttentionMask::Sliding { window_left })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PageSizes {
    /// Any positive size representable by the provider's signed i32 page ABI.
    Any,
    Exact(usize),
}

#[derive(Clone, Copy, Debug)]
pub struct AttentionKernelCapability {
    /// Stable implementation identity, independent of the request's phase.
    pub algorithm: &'static str,
    pub dtype: DType,
    /// Admitted (query/key dimension, value dimension) pairs.
    pub head_dimensions: &'static [(usize, usize)],
    pub layout: PagedKvLayout,
    pub page_sizes: PageSizes,
    /// Decode-only implementations require proof of one query per request.
    pub supports_prefill: bool,
}

#[derive(Clone, Debug)]
pub struct AttentionProviderCapabilities {
    pub name: &'static str,
    /// Admitted instruction families, possibly narrower than theoretical support.
    pub compute_majors: RangeInclusive<i32>,
    /// Causal and sliding attention with separate unquantized K/V allocations.
    pub kernels: &'static [AttentionKernelCapability],
}

impl AttentionProviderCapabilities {
    pub fn supports_target(&self, compute_major: i32) -> bool {
        self.compute_majors.contains(&compute_major)
    }

    /// Pure shape/semantic admission for native entry points and diagnostics.
    /// Availability, target, preparation resources and dynamic buffers are
    /// checked separately; this method never selects an implementation.
    pub fn supports_geometry(
        &self,
        algorithm: &str,
        spec: AttentionSpec,
        layout: PagedKvLayout,
        page_size: usize,
        prefill: bool,
    ) -> bool {
        let mask_supported = match spec.mask {
            AttentionMask::Causal => true,
            AttentionMask::Sliding { window_left } => i32::try_from(window_left).is_ok(),
            AttentionMask::Unmasked => false,
        };
        mask_supported
            && page_size > 0
            && i32::try_from(page_size).is_ok()
            && spec.query_heads > 0
            && spec.kv_heads > 0
            && spec.query_heads.is_multiple_of(spec.kv_heads)
            && i32::try_from(spec.query_heads).is_ok()
            && i32::try_from(spec.kv_heads).is_ok()
            && self.kernels.iter().any(|kernel| {
                kernel.algorithm == algorithm
                    && kernel.dtype == spec.dtype
                    && kernel
                        .head_dimensions
                        .contains(&(spec.query_key_dim, spec.value_dim))
                    && kernel.layout == layout
                    && (!prefill || kernel.supports_prefill)
                    && match kernel.page_sizes {
                        PageSizes::Any => true,
                        PageSizes::Exact(size) => page_size == size,
                    }
            })
    }

    pub(crate) fn declarations() -> Vec<String> {
        vec![
            ATTENTION_DECLARATIONS.to_owned(),
            CAPABILITY_DECLARATIONS.to_owned(),
            crate::target::DECLARATIONS.to_owned(),
        ]
    }

    /// Admit complete supported combinations before provider preparation/JIT.
    /// Equal query/request expressions prove decode-only eligibility for the
    /// semantic contract, which requires nonempty queries for each request.
    pub(crate) fn eligibility_rules(&self) -> Vec<Rule> {
        let mut rules = ["all", self.name]
            .map(|policy| {
                Rule::raw(format!(
                    include_str!("attention/target.egg.in"),
                    provider = self.name,
                    policy = policy,
                    minimum_major = self.compute_majors.start(),
                    maximum_major = self.compute_majors.end(),
                ))
            })
            .into_iter()
            .collect::<Vec<_>>();
        rules.push(Rule::raw(format!(
            include_str!("attention/causal_window.egg.in"),
            i32::MAX,
            self.name,
        )));
        for (index, kernel) in self.kernels.iter().enumerate() {
            for &(query_key_dim, value_dim) in kernel.head_dimensions {
                let page_guard = match kernel.page_sizes {
                    PageSizes::Any => String::new(),
                    PageSizes::Exact(size) => format!("(= ?page_tokens {size})"),
                };
                let phase_guard = if kernel.supports_prefill {
                    ""
                } else {
                    "(= ?q ?requests)"
                };
                rules.push(Rule::raw(format!(
                    include_str!("attention/eligibility.egg.in"),
                    kernel.dtype,
                    kernel.layout.to_egglog(),
                    self.name,
                    kernel.algorithm,
                    self.name,
                    max_index = i32::MAX,
                    provider = self.name,
                    index = index,
                    page_guard = page_guard,
                    phase_guard = phase_guard,
                    query_key_dim = query_key_dim,
                    value_dim = value_dim,
                )));
            }
        }
        rules
    }
}

#[cfg(test)]
#[path = "../../tests/unit/providers/attention/mod.rs"]
pub(crate) mod tests;
