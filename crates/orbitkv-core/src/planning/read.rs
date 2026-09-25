use std::sync::Arc;

use crate::QueryMode;
use crate::block::StateKey;
use crate::storage::ssd::SsdStore;

use super::replica::ReplicaSet;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReadTarget {
    HostReady,
    EngineRestore,
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
}
