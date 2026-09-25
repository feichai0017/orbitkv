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
    pub(crate) fn new(rows: &'a mut [ReplicaSet], required: usize) -> Option<Self> {
        let prefix = rows
            .iter()
            .take_while(|row| row.peer_dram().next().is_some())
            .count();
        (prefix > 0 && prefix >= required).then(|| Self {
            rows: &mut rows[..prefix],
        })
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
        let mut best: Option<FetchSegment> = None;
        for candidate in row.peer_dram() {
            let mut bytes = row.key.namespace.len();
            let records: Vec<_> = rows[start..]
                .iter()
                .take(DISCOVERY_MAX_KEYS)
                .map_while(|row| {
                    let replica = row.peer_dram().find(|r| r.owner == candidate.owner)?;
                    bytes = bytes.saturating_add(row.key.hash.len());
                    (bytes <= DISCOVERY_MAX_BYTES).then(|| InventoryRecord {
                        key: row.key.clone(),
                        sequence: replica.sequence,
                        present: true,
                    })
                })
                .collect();
            if records.is_empty() {
                continue;
            }
            if best.as_ref().is_none_or(|best| {
                records.len() > best.records.len()
                    || (records.len() == best.records.len() && candidate.owner < best.owner)
            }) {
                best = Some(FetchSegment {
                    owner: candidate.owner.clone(),
                    records,
                });
            }
        }
        best
    }
}

#[cfg(test)]
#[path = "../../tests/unit/planning/peer.rs"]
mod tests;
