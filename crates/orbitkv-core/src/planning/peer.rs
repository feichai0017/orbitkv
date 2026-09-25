use orbitkv_state::{
    BlockCandidates, CacheOwner, DISCOVERY_MAX_BYTES, DISCOVERY_MAX_KEYS, InventoryRecord,
};

use super::replica::ReplicaSet;

pub(crate) struct FetchSegment {
    pub(crate) owner: CacheOwner,
    pub(crate) records: Vec<InventoryRecord>,
}

pub(crate) struct FetchPlan {
    rows: Vec<ReplicaSet>,
}

impl FetchPlan {
    pub(crate) fn new(rows: Vec<BlockCandidates>) -> Option<Self> {
        let mut rows: Vec<_> = rows
            .into_iter()
            .map(|row| {
                let mut replicas = ReplicaSet::new(row.key);
                replicas.set_peer_dram(row.replicas);
                replicas
            })
            .collect();
        let prefix = rows
            .iter()
            .take_while(|row| row.peer_dram().next().is_some())
            .count();
        rows.truncate(prefix);
        (!rows.is_empty()).then_some(Self { rows })
    }

    pub(crate) fn next_segment(&self, start: usize) -> Option<FetchSegment> {
        next_segment(&self.rows, start)
    }

    pub(crate) fn reject(&mut self, start: usize, segment: &FetchSegment) {
        for row in &mut self.rows[start..][..segment.records.len()] {
            row.reject_peer(&segment.owner);
        }
    }

    pub(crate) fn block_count(&self) -> usize {
        self.rows.len()
    }

    pub(crate) fn matches(&self, namespace: &str, hashes: &[Vec<u8>]) -> bool {
        self.rows.len() <= hashes.len()
            && self
                .rows
                .iter()
                .zip(hashes)
                .all(|(row, hash)| row.key.namespace == namespace && row.key.hash == *hash)
    }
}

fn next_segment(rows: &[ReplicaSet], start: usize) -> Option<FetchSegment> {
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

#[cfg(test)]
#[path = "../../tests/unit/planning/peer.rs"]
mod tests;
