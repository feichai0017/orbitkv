use orbitkv_state::{CacheOwner, DISCOVERY_MAX_BYTES, DISCOVERY_MAX_KEYS, InventoryRecord};

use super::replica::ReplicaSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PeerSource {
    Dram,
    Ssd,
}

impl PeerSource {
    pub(crate) fn medium(self) -> orbitkv_state::ReplicaMedium {
        match self {
            Self::Dram => orbitkv_state::ReplicaMedium::Dram,
            Self::Ssd => orbitkv_state::ReplicaMedium::Ssd,
        }
    }
}

pub(crate) struct FetchSegment {
    pub(crate) owner: CacheOwner,
    pub(crate) source: PeerSource,
    pub(crate) records: Vec<InventoryRecord>,
    pub(crate) stored_bytes: Option<u64>,
    pub(crate) representation: orbitkv_state::ReplicaRepresentation,
}

pub(crate) struct FetchPlan<'a> {
    rows: &'a mut [ReplicaSet],
    source: PeerSource,
}

impl<'a> FetchPlan<'a> {
    #[cfg(test)]
    pub(crate) fn new(
        rows: &'a mut [ReplicaSet],
        required: usize,
        source: PeerSource,
    ) -> Option<Self> {
        let prefix = Self::prefix_len(rows, source);
        (prefix > 0 && prefix >= required).then(|| Self::from_prefix(rows, prefix, source))
    }

    pub(super) fn prefix_len(rows: &[ReplicaSet], source: PeerSource) -> usize {
        rows.iter()
            .take_while(|row| row.peer(source.medium()).next().is_some())
            .count()
    }

    pub(super) fn from_prefix(
        rows: &'a mut [ReplicaSet],
        prefix: usize,
        source: PeerSource,
    ) -> Self {
        Self {
            rows: &mut rows[..prefix],
            source,
        }
    }

    pub(crate) fn reject(&mut self, start: usize, segment: &FetchSegment) {
        for row in &mut self.rows[start..][..segment.records.len()] {
            row.reject_peer(&segment.owner);
        }
    }

    pub(crate) fn block_count(&self) -> usize {
        self.rows.len()
    }

    pub(crate) fn next_segment(&self, start: usize) -> Option<FetchSegment> {
        let rows = &self.rows;

        let row = rows.get(start)?;
        let mut best: Option<(&CacheOwner, usize)> = None;
        for candidate in row.peer(self.source.medium()) {
            let mut bytes = row.key.namespace.len();
            let count = rows[start..]
                .iter()
                .take(DISCOVERY_MAX_KEYS)
                .map_while(|row| {
                    row.peer(self.source.medium())
                        .find(|r| r.owner == candidate.owner)?;
                    bytes = bytes.saturating_add(row.key.hash.len());
                    (bytes <= DISCOVERY_MAX_BYTES).then_some(())
                })
                .count();
            if count == 0 {
                continue;
            }
            if best.is_none_or(|(owner, best_count)| {
                count > best_count || (count == best_count && candidate.owner < *owner)
            }) {
                best = Some((&candidate.owner, count));
            }
        }
        let (owner, count) = best?;
        let mut records = Vec::with_capacity(count);
        let mut stored_bytes = Some(0u64);
        let mut representation = None;
        for row in &rows[start..start + count] {
            let replica = row
                .peer(self.source.medium())
                .find(|replica| &replica.owner == owner)?;
            stored_bytes = stored_bytes
                .zip(replica.metadata.stored_bytes)
                .and_then(|(total, bytes)| total.checked_add(bytes));
            let next = replica.metadata.representation;
            representation = Some(match representation {
                None => next,
                Some(orbitkv_state::ReplicaRepresentation::Unknown) => {
                    orbitkv_state::ReplicaRepresentation::Unknown
                }
                Some(_) if next == orbitkv_state::ReplicaRepresentation::Unknown => {
                    orbitkv_state::ReplicaRepresentation::Unknown
                }
                Some(previous) if previous == next => previous,
                Some(_) => orbitkv_state::ReplicaRepresentation::Mixed,
            });
            records.push(InventoryRecord {
                key: row.key.clone(),
                sequence: replica.sequence,
                present: true,
                metadata: Some(replica.metadata),
            });
        }
        Some(FetchSegment {
            owner: owner.clone(),
            source: self.source,
            records,
            stored_bytes,
            representation: representation.unwrap_or_default(),
        })
    }
}

#[cfg(test)]
#[path = "../../tests/unit/planning/peer.rs"]
mod tests;
