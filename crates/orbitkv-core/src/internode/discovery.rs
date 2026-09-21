use std::time::{Duration, Instant};

use hashlink::LinkedHashMap;
use orbitkv_state::{BlockCandidates, ReplicaLocation, StateKey};

pub(super) const CANDIDATE_CACHE_BYTES: usize = 16 * 1024 * 1024;
const CANDIDATE_TTL: Duration = Duration::from_secs(5);

struct Entry {
    candidates: BlockCandidates,
    expires: Instant,
    bytes: usize,
}

/// Positive location hints only. The source is always authoritative.
pub(super) struct CandidateIndex {
    entries: LinkedHashMap<StateKey, Entry>,
    bytes: usize,
    capacity: usize,
}

impl CandidateIndex {
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            entries: LinkedHashMap::new(),
            bytes: 0,
            capacity,
        }
    }

    pub(super) fn get(&mut self, key: &StateKey, now: Instant) -> Option<BlockCandidates> {
        if self
            .entries
            .get(key)
            .is_some_and(|entry| entry.expires <= now)
        {
            self.remove(key);
        }
        self.entries
            .to_back(key)
            .map(|entry| entry.candidates.clone())
    }

    pub(super) fn insert(&mut self, candidates: BlockCandidates, now: Instant) {
        self.remove(&candidates.key);
        // Include the map's owned key and entry metadata in the logical budget.
        let bytes = candidates.estimated_size()
            + candidates.key.estimated_size() as usize
            + std::mem::size_of::<Entry>();
        if candidates.replicas.is_empty() || bytes > self.capacity {
            return;
        }
        while self.bytes + bytes > self.capacity {
            if let Some((_, entry)) = self.entries.pop_front() {
                self.bytes -= entry.bytes;
            }
        }
        self.bytes += bytes;
        self.entries.insert(
            candidates.key.clone(),
            Entry {
                candidates,
                expires: now + CANDIDATE_TTL,
                bytes,
            },
        );
    }

    pub(super) fn reject(&mut self, key: &StateKey, replica: &ReplicaLocation) {
        if let Some(entry) = self.entries.get_mut(key) {
            // A late rejection must not remove a newer residency or incarnation.
            entry
                .candidates
                .replicas
                .retain(|candidate| candidate != replica);
            if entry.candidates.replicas.is_empty() {
                self.remove(key);
            }
        }
    }

    fn remove(&mut self, key: &StateKey) {
        if let Some(entry) = self.entries.remove(key) {
            self.bytes -= entry.bytes;
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/internode/discovery.rs"]
mod tests;
