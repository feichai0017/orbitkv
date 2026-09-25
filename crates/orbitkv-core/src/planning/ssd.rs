use std::sync::Arc;

use crate::backing::ssd::SsdBackingStore;
use crate::block::{RestoreSource, StateKey};
use crate::{QueryMode, SsdReadPath};

use super::replica::ReplicaCandidate;

/// Metadata-only plan for one homogeneous SSD route. Acquisition revalidates
/// each version; the returned sources hand ownership to existing query leases.
pub(crate) struct SsdReadPlan {
    candidates: Vec<ReplicaCandidate>,
    path: SsdReadPath,
    required: usize,
}

impl SsdReadPlan {
    pub(crate) fn discover(
        store: &Arc<SsdBackingStore>,
        keys: &[StateKey],
        mode: QueryMode,
        codec_budget: usize,
    ) -> Option<Self> {
        if matches!(mode, QueryMode::Warmup | QueryMode::Prepare)
            || (store.read_path.is_none() && !store.gpu_io.available())
        {
            return None;
        }
        let candidates: Vec<_> = store
            .discover_prefix(keys)
            .into_iter()
            .map(ReplicaCandidate::ssd)
            .collect();
        let required = if mode == QueryMode::WaitForFullPrefix {
            keys.len()
        } else {
            1
        };
        if candidates.is_empty() || candidates.len() < required {
            return None;
        }
        let path = match store.read_path {
            Some(SsdReadPath::Uring) => SsdReadPath::Uring,
            _ if candidates.iter().all(|candidate| {
                candidate
                    .local_ssd()
                    .is_some_and(|source| source.cufile_eligible(codec_budget))
            }) =>
            {
                SsdReadPath::Cufile
            }
            _ => return None,
        };
        Some(Self {
            candidates,
            path,
            required,
        })
    }

    pub(crate) fn acquire(self, codec_budget: usize) -> Option<Vec<RestoreSource>> {
        let leases: Vec<_> = self
            .candidates
            .iter()
            .map_while(|candidate| candidate.local_ssd()?.pin())
            .collect();
        if leases.len() < self.required
            || (self.path == SsdReadPath::Cufile
                && leases
                    .iter()
                    .any(|lease| !lease.cufile_eligible(codec_budget)))
        {
            return None;
        }
        Some(
            leases
                .into_iter()
                .map(|lease| RestoreSource::Ssd {
                    lease,
                    path: self.path,
                })
                .collect(),
        )
    }
}
