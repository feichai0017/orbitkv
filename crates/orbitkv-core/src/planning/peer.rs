use orbitkv_state::{CacheOwner, DISCOVERY_MAX_BYTES, DISCOVERY_MAX_KEYS, InventoryRecord};

use super::replica::ReplicaSet;

pub(crate) struct FetchSegment {
    pub(crate) owner: CacheOwner,
    pub(crate) records: Vec<InventoryRecord>,
}

pub(crate) struct FetchPlan<'a> {
    rows: &'a mut [ReplicaSet],
}

impl<'a> FetchPlan<'a> {
    #[cfg(test)]
    pub(crate) fn new(rows: &'a mut [ReplicaSet], required: usize) -> Option<Self> {
        let prefix = Self::prefix_len(rows);
        (prefix > 0 && prefix >= required).then(|| Self::from_prefix(rows, prefix))
    }

    pub(super) fn prefix_len(rows: &[ReplicaSet]) -> usize {
        rows.iter()
            .take_while(|row| row.peer_dram().next().is_some())
            .count()
    }

    pub(super) fn from_prefix(rows: &'a mut [ReplicaSet], prefix: usize) -> Self {
        Self {
            rows: &mut rows[..prefix],
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
        for candidate in row.peer_dram() {
            let mut bytes = row.key.namespace.len();
            let count = rows[start..]
                .iter()
                .take(DISCOVERY_MAX_KEYS)
                .map_while(|row| {
                    row.peer_dram().find(|r| r.owner == candidate.owner)?;
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
        for row in &rows[start..start + count] {
            let replica = row.peer_dram().find(|replica| &replica.owner == owner)?;
            records.push(InventoryRecord {
                key: row.key.clone(),
                sequence: replica.sequence,
                present: true,
            });
        }
        Some(FetchSegment {
            owner: owner.clone(),
            records,
        })
    }
}

#[cfg(test)]
#[path = "../../tests/unit/planning/peer.rs"]
mod tests;
