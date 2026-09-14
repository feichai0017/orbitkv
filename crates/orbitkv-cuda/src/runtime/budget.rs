//! Arena sizing, search headroom, and materialized-bucket budget policies.

use std::collections::VecDeque;

pub(super) const ARENA_ALIGNMENT: usize = 256;
pub(super) const MIN_ARENA_ALLOCATION_BYTES: usize = 16 * 1024 * 1024;
pub(super) const MIN_SEARCH_DEVICE_HEADROOM_BYTES: usize = 512 * 1024 * 1024;
pub(super) const SEARCH_DEVICE_HEADROOM_DIVISOR: usize = 200;
pub(super) const MIN_SEARCH_CACHE_EVICTION_HEADROOM_BYTES: usize = 1024 * 1024 * 1024;
pub(super) const SEARCH_CACHE_EVICTION_HEADROOM_DIVISOR: usize = 50;
pub(super) const MIN_SEARCH_CANDIDATE_NODE_ALLOWANCE: usize = 1024;

pub(super) fn materialized_bucket_evictions(
    materialized: &[bool],
    lru: &VecDeque<usize>,
    keep: usize,
    capacity: usize,
) -> Vec<usize> {
    let projected = materialized.iter().filter(|resident| **resident).count()
        + usize::from(!materialized.get(keep).copied().unwrap_or(false));
    let mut remaining = projected.saturating_sub(capacity);
    let mut evictions = Vec::with_capacity(remaining);

    for candidate in lru.iter().copied().chain(0..materialized.len()) {
        if remaining == 0 {
            break;
        }
        if candidate != keep
            && materialized.get(candidate).copied().unwrap_or(false)
            && !evictions.contains(&candidate)
        {
            evictions.push(candidate);
            remaining -= 1;
        }
    }
    evictions
}

pub(super) fn search_candidate_node_limit(baseline_nodes: usize) -> usize {
    baseline_nodes.saturating_add(MIN_SEARCH_CANDIDATE_NODE_ALLOWANCE)
}

pub(super) fn bounded_search_intermediate_bytes(
    configured: Option<usize>,
    free_device_bytes: usize,
    total_device_bytes: usize,
) -> usize {
    // Candidate plans account for arenas and declared HostOp workspaces, but
    // the CUDA context, loaded modules, library internals, and allocator
    // metadata also consume VRAM. Reserve a small device-relative margin so a
    // graph that cannot physically allocate is rejected by planning instead
    // of reaching cuMemAlloc and panicking during search.
    let headroom =
        (total_device_bytes / SEARCH_DEVICE_HEADROOM_DIVISOR).max(MIN_SEARCH_DEVICE_HEADROOM_BYTES);
    let available = free_device_bytes.saturating_sub(headroom);
    configured.map_or(available, |limit| limit.min(available))
}

pub(super) fn search_cache_under_pressure(
    free_device_bytes: usize,
    total_device_bytes: usize,
) -> bool {
    let minimum_free = (total_device_bytes / SEARCH_CACHE_EVICTION_HEADROOM_DIVISOR)
        .max(MIN_SEARCH_CACHE_EVICTION_HEADROOM_BYTES);
    free_device_bytes < minimum_free
}
