use std::collections::HashMap;
use std::sync::Arc;

use orbitkv_state::{
    BlockCandidates, DISCOVERY_MAX_REPLICAS_PER_MEDIUM, InventoryRecord, ReplicaLocation,
    ReplicaMedium, ReplicaMetadata, ReplicaRepresentation, StateKey,
};
use parking_lot::RwLock;
use uuid::Uuid;

use crate::MembershipView;

pub const DEFAULT_INDEX_BYTES: usize = 256 * 1024 * 1024;

pub enum IndexUpdate {
    Residency {
        owner: Uuid,
        record: InventoryRecord,
    },
    Publisher {
        owner: Uuid,
        ready: bool,
    },
    RemoveOwner(Uuid),
}

struct Entry {
    owner: Uuid,
    sequence: u64,
    metadata: ReplicaMetadata,
}

#[derive(Default)]
struct View {
    blocks: HashMap<StateKey, Vec<Entry>>,
    publishers: HashMap<Uuid, bool>,
    bytes: usize,
    revision: i64,
    ready: bool,
}

#[derive(serde::Serialize)]
pub struct IndexStatus {
    pub revision: Option<i64>,
    pub accounted_bytes: usize,
    pub registration_valid: bool,
    pub available: bool,
}

pub struct GlobalIndex {
    membership: Arc<MembershipView>,
    view: RwLock<View>,
    byte_limit: usize,
}

impl GlobalIndex {
    pub fn new(membership: Arc<MembershipView>, byte_limit: usize) -> Self {
        Self {
            membership,
            view: RwLock::new(View::default()),
            byte_limit,
        }
    }

    pub fn reset(&self) {
        *self.view.write() = View::default();
    }

    pub fn revision(&self) -> Option<i64> {
        let view = self.view.read();
        view.ready.then_some(view.revision)
    }

    pub fn status(&self) -> IndexStatus {
        let view = self.view.read();
        IndexStatus {
            revision: view.ready.then_some(view.revision),
            accounted_bytes: view.bytes,
            registration_valid: self.membership.registration_valid(),
            available: view.ready && self.membership.permits(self.membership.owner()),
        }
    }

    pub fn bytes(&self) -> usize {
        self.view.read().bytes
    }

    pub fn finish_snapshot(&self, revision: i64) -> Result<(), String> {
        let mut view = self.view.write();
        if revision <= 0 || view.revision > revision {
            return Err("invalid global index snapshot revision".into());
        }
        view.revision = revision;
        view.ready = true;
        Ok(())
    }

    pub fn apply(&self, revision: i64, updates: Vec<IndexUpdate>) -> Result<(), String> {
        let mut view = self.view.write();
        if revision <= 0 || (view.ready && revision < view.revision) {
            return Err("global index revision moved backwards".into());
        }
        let result = (|| {
            for update in updates {
                match update {
                    IndexUpdate::Publisher { owner, ready } => {
                        if owner.is_nil() {
                            return Err("nil metadata publisher".into());
                        }
                        if !view.publishers.contains_key(&owner) {
                            view.bytes += 128;
                        }
                        view.publishers.insert(owner, ready);
                    }
                    IndexUpdate::RemoveOwner(owner) => {
                        if view.publishers.remove(&owner).is_some() {
                            view.bytes -= 128;
                        }
                        let mut removed = 0;
                        view.blocks.retain(|key, entries| {
                            entries.retain(|entry| {
                                if entry.owner == owner {
                                    removed += entry_bytes(key);
                                    false
                                } else {
                                    true
                                }
                            });
                            !entries.is_empty()
                        });
                        view.bytes -= removed;
                    }
                    IndexUpdate::Residency { owner, record } => {
                        let metadata = record.metadata.ok_or("missing residency metadata")?;
                        if owner.is_nil()
                            || record.sequence == 0
                            || record.key.namespace.is_empty()
                            || record.key.hash.is_empty()
                            || !matches!(metadata.medium, ReplicaMedium::Dram | ReplicaMedium::Ssd)
                            || metadata.representation == ReplicaRepresentation::Unknown
                        {
                            return Err("invalid global residency record".into());
                        }
                        let size = entry_bytes(&record.key);
                        let entries = view.blocks.entry(record.key.clone()).or_default();
                        let previous = entries.iter().position(|entry| {
                            entry.owner == owner && entry.metadata.medium == metadata.medium
                        });
                        let mut delta = 0isize;
                        if let Some(i) = previous {
                            if entries[i].sequence > record.sequence {
                                continue;
                            }
                            if record.present {
                                if entries[i].sequence == record.sequence {
                                    if entries[i].metadata != metadata {
                                        return Err("conflicting residency generation".into());
                                    }
                                    continue;
                                }
                                entries[i] = Entry {
                                    owner,
                                    sequence: record.sequence,
                                    metadata,
                                };
                            } else {
                                entries.swap_remove(i);
                                delta = -(size as isize);
                            }
                        } else if record.present {
                            entries.push(Entry {
                                owner,
                                sequence: record.sequence,
                                metadata,
                            });
                            delta = size as isize;
                        }
                        if entries.is_empty() {
                            view.blocks.remove(&record.key);
                        }
                        view.bytes = view
                            .bytes
                            .checked_add_signed(delta)
                            .ok_or("index byte overflow")?;
                    }
                }
                if view.bytes > self.byte_limit {
                    return Err("complete global index exceeds metadata budget".into());
                }
            }
            Ok(())
        })();
        if result.is_err() {
            *view = View::default();
        } else {
            view.revision = revision;
        }
        result
    }

    pub fn lookup(&self, keys: &[StateKey]) -> Vec<BlockCandidates> {
        let view = self.view.read();
        let permitted = view.ready && self.membership.permits(self.membership.owner());
        keys.iter()
            .map(|key| {
                let mut replicas = Vec::new();
                if permitted && let Some(entries) = view.blocks.get(key) {
                    for medium in [ReplicaMedium::Dram, ReplicaMedium::Ssd] {
                        let start = replicas.len();
                        for entry in entries
                            .iter()
                            .filter(|entry| entry.metadata.medium == medium)
                        {
                            if view.publishers.get(&entry.owner) != Some(&true) {
                                continue;
                            }
                            if let Some(owner) = self.membership.resolve(entry.owner)
                                && owner.incarnation != self.membership.owner().incarnation
                            {
                                replicas.push(ReplicaLocation {
                                    owner,
                                    sequence: entry.sequence,
                                    metadata: entry.metadata,
                                });
                            }
                            if replicas.len() - start == DISCOVERY_MAX_REPLICAS_PER_MEDIUM {
                                break;
                            }
                        }
                    }
                }
                BlockCandidates {
                    key: key.clone(),
                    replicas,
                }
            })
            .collect()
    }
}

fn entry_bytes(key: &StateKey) -> usize {
    // Include map/vector allocation headroom and shared-key storage in every entry.
    256 + key.namespace.len() + key.hash.len()
}

#[cfg(test)]
#[path = "../tests/unit/index.rs"]
mod tests;
