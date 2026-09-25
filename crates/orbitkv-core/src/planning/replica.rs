use std::sync::Weak;

#[cfg(feature = "mooncake")]
use orbitkv_state::{DISCOVERY_MAX_REPLICAS, ReplicaLocation};
use smallvec::SmallVec;

use crate::backing::ssd::SsdReadCandidate;
use crate::block::{SealedBlock, StateKey};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Medium {
    Dram,
    Ssd,
}

enum Evidence {
    Memory(Weak<SealedBlock>),
    Extent(SsdReadCandidate),
    #[cfg(feature = "mooncake")]
    Peer(ReplicaLocation),
}

/// A location/version hint. Its evidence owns no payload or transfer permission.
/// Medium and acquisition authority are independent: peer evidence is not a tier.
pub(crate) struct ReplicaCandidate {
    medium: Medium,
    evidence: Evidence,
}

impl ReplicaCandidate {
    pub(crate) fn ssd(candidate: SsdReadCandidate) -> Self {
        Self {
            medium: Medium::Ssd,
            evidence: Evidence::Extent(candidate),
        }
    }

    fn is_available(&self) -> bool {
        match &self.evidence {
            Evidence::Memory(block) => block.strong_count() > 0,
            Evidence::Extent(_) => true,
            #[cfg(feature = "mooncake")]
            Evidence::Peer(_) => true,
        }
    }

    pub(crate) fn local_ssd(&self) -> Option<&SsdReadCandidate> {
        match &self.evidence {
            Evidence::Extent(candidate) if self.medium == Medium::Ssd => Some(candidate),
            _ => None,
        }
    }

    fn is_peer(&self) -> bool {
        match self.evidence {
            #[cfg(feature = "mooncake")]
            Evidence::Peer(_) => true,
            _ => false,
        }
    }

    #[cfg(feature = "mooncake")]
    fn peer_dram(&self) -> Option<&ReplicaLocation> {
        match &self.evidence {
            Evidence::Peer(location) if self.medium == Medium::Dram => Some(location),
            _ => None,
        }
    }
}

/// At most one replica per local store plus the directory's bounded peer set.
pub(crate) struct ReplicaSet {
    pub(crate) key: StateKey,
    replicas: SmallVec<[ReplicaCandidate; 2]>,
}

impl ReplicaSet {
    pub(crate) fn new(key: StateKey) -> Self {
        Self {
            key,
            replicas: SmallVec::new(),
        }
    }

    pub(crate) fn set_memory(&mut self, block: Weak<SealedBlock>) {
        self.set_local(ReplicaCandidate {
            medium: Medium::Dram,
            evidence: Evidence::Memory(block),
        });
    }

    pub(crate) fn set_ssd(&mut self, candidate: SsdReadCandidate) {
        self.set_local(ReplicaCandidate::ssd(candidate));
    }

    fn set_local(&mut self, candidate: ReplicaCandidate) {
        self.replicas
            .retain(|existing| existing.medium != candidate.medium || existing.is_peer());
        self.replicas.push(candidate);
    }

    /// Today's catalog describes DRAM only. Unknown peer sizes/encoding remain
    /// absent from the evidence; no HBM/SSD capability is inferred from TE support.
    #[cfg(feature = "mooncake")]
    pub(crate) fn set_peer_dram(&mut self, replicas: Vec<ReplicaLocation>) {
        self.replicas.retain(|replica| !replica.is_peer());
        for location in replicas.into_iter().take(DISCOVERY_MAX_REPLICAS) {
            if !self.peer_dram().any(|peer| peer.owner == location.owner) {
                self.replicas.push(ReplicaCandidate {
                    medium: Medium::Dram,
                    evidence: Evidence::Peer(location),
                });
            }
        }
    }

    #[cfg(feature = "mooncake")]
    pub(crate) fn peer_dram(&self) -> impl Iterator<Item = &ReplicaLocation> {
        self.replicas.iter().filter_map(ReplicaCandidate::peer_dram)
    }

    #[cfg(feature = "mooncake")]
    pub(crate) fn reject_peer(&mut self, owner: &orbitkv_state::CacheOwner) {
        self.replicas
            .retain(|replica| replica.peer_dram().is_none_or(|peer| &peer.owner != owner));
    }

    pub(crate) fn is_available(&self) -> bool {
        self.replicas.iter().any(ReplicaCandidate::is_available)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/planning/replica.rs"]
mod tests;
