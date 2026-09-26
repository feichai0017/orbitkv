use std::sync::Arc;

use crate::QueryMode;
use crate::block::StateKey;
use crate::storage::ssd::SsdStore;

#[cfg(feature = "mooncake")]
use super::peer::{FetchPlan, PeerSource};
use super::replica::ReplicaSet;
use super::ssd::SsdReadPlan;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReadTarget {
    HostReady,
    EngineRestore,
}

/// A metadata-only route whose result is host-resident state. Source leases
/// and transfer grants are acquired by the selected execution owner.
pub(crate) enum HostReadRoute<'a> {
    Ssd(SsdReadPlan<'a>),
    #[cfg(feature = "mooncake")]
    Peer(FetchPlan<'a>),
}

enum HostReadSelection {
    Ssd,
    #[cfg(feature = "mooncake")]
    Peer {
        prefix: usize,
        source: PeerSource,
    },
}

/// Unresolved sources for one admitted query batch. DRAM prefix holds already
/// belong to the caller; these records remain metadata until route acquisition.
pub(crate) struct ReadPlan {
    pub(crate) rows: Vec<ReplicaSet>,
    pub(crate) target: ReadTarget,
    pub(crate) required: usize,
    #[cfg(feature = "mooncake")]
    pub(crate) wait_for_full_prefix: bool,
}

impl ReadPlan {
    pub(crate) fn new(keys: &[StateKey], mode: QueryMode, ssd: Option<&Arc<SsdStore>>) -> Self {
        let mut rows: Vec<_> = keys.iter().cloned().map(ReplicaSet::new).collect();
        if let Some(ssd) = ssd {
            for (row, candidate) in rows.iter_mut().zip(ssd.discover(keys)) {
                if let Some(candidate) = candidate {
                    row.set_ssd(candidate);
                }
            }
        }
        Self {
            rows,
            #[cfg(feature = "mooncake")]
            wait_for_full_prefix: mode == QueryMode::WaitForFullPrefix,
            target: if matches!(mode, QueryMode::Warmup | QueryMode::Prepare) {
                ReadTarget::HostReady
            } else {
                ReadTarget::EngineRestore
            },
            required: if mode == QueryMode::WaitForFullPrefix {
                keys.len()
            } else {
                1
            },
        }
    }

    pub(crate) fn host_route(
        &mut self,
        peer_available: bool,
        allow_ssd: bool,
        codec_budget: usize,
    ) -> Option<HostReadRoute<'_>> {
        #[cfg(feature = "mooncake")]
        let peer_dram = if peer_available {
            let prefix = FetchPlan::prefix_len(&self.rows, PeerSource::Dram);
            (prefix > 0 && prefix >= self.required).then_some(prefix)
        } else {
            None
        };
        #[cfg(not(feature = "mooncake"))]
        let _ = peer_available;

        #[cfg(feature = "mooncake")]
        let peer_ssd = if peer_available {
            let prefix = FetchPlan::prefix_len(&self.rows, PeerSource::Ssd);
            (prefix > 0 && prefix >= self.required).then_some(prefix)
        } else {
            None
        };
        let local_ssd = allow_ssd && self.ssd(crate::SsdReadPath::Uring, codec_budget).is_some();

        #[cfg(feature = "mooncake")]
        let selected = peer_dram
            .map(|prefix| HostReadSelection::Peer {
                prefix,
                source: PeerSource::Dram,
            })
            .or_else(|| local_ssd.then_some(HostReadSelection::Ssd))
            .or_else(|| {
                peer_ssd.map(|prefix| HostReadSelection::Peer {
                    prefix,
                    source: PeerSource::Ssd,
                })
            });
        #[cfg(not(feature = "mooncake"))]
        let selected = local_ssd.then_some(HostReadSelection::Ssd);

        match selected? {
            HostReadSelection::Ssd => self
                .ssd(crate::SsdReadPath::Uring, codec_budget)
                .map(HostReadRoute::Ssd),
            #[cfg(feature = "mooncake")]
            HostReadSelection::Peer { prefix, source } => Some(HostReadRoute::Peer(
                FetchPlan::from_prefix(&mut self.rows, prefix, source),
            )),
        }
    }
}
