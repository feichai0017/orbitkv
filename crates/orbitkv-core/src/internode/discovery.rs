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
mod tests {
    use super::*;
    use orbitkv_state::CacheOwner;

    fn row(hash: u8, sequence: u64) -> BlockCandidates {
        BlockCandidates {
            key: StateKey::new("ns".into(), vec![hash]),
            replicas: vec![ReplicaLocation {
                owner: CacheOwner {
                    endpoint: "owner:50055".into(),
                    incarnation: uuid::Uuid::from_u128(1),
                },
                sequence,
            }],
        }
    }

    #[test]
    fn bounded_lru_expiry_and_version_specific_rejection() {
        let now = Instant::now();
        let first = row(1, 1);
        let mut index = CandidateIndex::new(4096);
        index.insert(first.clone(), now);
        let one_entry = index.bytes;
        index.capacity = one_entry * 2;
        index.insert(row(2, 2), now);
        assert!(index.get(&first.key, now).is_some());
        index.insert(row(3, 3), now);
        assert!(index.get(&row(2, 2).key, now).is_none());
        let replacement = row(1, 4);
        index.insert(replacement.clone(), now);
        index.reject(&first.key, &first.replicas[0]);
        assert_eq!(index.get(&first.key, now), Some(replacement.clone()));
        index.reject(&first.key, &replacement.replicas[0]);
        assert!(index.get(&first.key, now).is_none());
        assert!(index.get(&row(3, 3).key, now + CANDIDATE_TTL).is_none());
        assert_eq!(index.bytes, 0);
        let mut negative = row(1, 1);
        negative.replicas.clear();
        index.insert(negative, now);
        index.capacity = 1;
        index.insert(first, now);
        assert_eq!(index.bytes, 0);
    }
}
