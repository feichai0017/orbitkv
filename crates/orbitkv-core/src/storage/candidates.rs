use std::sync::Weak;

use orbitkv_state::ReplicaLocation;

use crate::backing::ssd::SsdReadCandidate;
use crate::block::{SealedBlock, StateKey};

/// Metadata evidence, not permission to read. Local DRAM exposes its immutable
/// image through a weak reference; SSD retains index metadata without pinning
/// the extent. Peer DRAM needs authoritative authorization before any TE read.
pub(crate) struct ResidencyCandidates {
    pub(crate) key: StateKey,
    pub(crate) dram: Option<Weak<SealedBlock>>,
    pub(crate) ssd: Option<SsdReadCandidate>,
    pub(crate) peer_dram: Vec<ReplicaLocation>,
}

impl ResidencyCandidates {
    pub(crate) fn is_available(&self) -> bool {
        self.dram
            .as_ref()
            .is_some_and(|block| block.strong_count() > 0)
            || self.ssd.is_some()
            || !self.peer_dram.is_empty()
    }
}
