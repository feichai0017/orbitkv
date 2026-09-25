use std::sync::Arc;

use crate::SsdReadPath;
use crate::backing::ssd::{SsdBackingStore, SsdReadLease};

use super::read::{ReadPlan, ReadTarget};
use super::replica::ReplicaSet;

/// A route over request-owned evidence, not another copy of the SSD inventory.
pub(crate) struct SsdReadPlan<'a> {
    rows: &'a [ReplicaSet],
    pub(crate) path: SsdReadPath,
    required: usize,
}

impl ReadPlan {
    pub(crate) fn deferred_ssd(
        &self,
        store: &SsdBackingStore,
        codec_budget: usize,
    ) -> Option<SsdReadPlan<'_>> {
        if self.target != ReadTarget::EngineRestore
            || (store.read_path.is_none() && !store.gpu_io.available())
        {
            return None;
        }
        self.ssd(store.read_path.unwrap_or(SsdReadPath::Cufile), codec_budget)
    }

    pub(crate) fn ssd(&self, path: SsdReadPath, codec_budget: usize) -> Option<SsdReadPlan<'_>> {
        let count = self
            .rows
            .iter()
            .take_while(|row| row.local_ssd().is_some())
            .count();
        if count == 0 || count < self.required {
            return None;
        }
        let rows = &self.rows[..count];
        if path == SsdReadPath::Cufile
            && rows.iter().any(|row| {
                !row.local_ssd()
                    .is_some_and(|source| source.cufile_eligible(codec_budget))
            })
        {
            return None;
        }
        Some(SsdReadPlan {
            rows,
            path,
            required: self.required,
        })
    }
}

impl SsdReadPlan<'_> {
    pub(crate) fn acquire(self, codec_budget: usize) -> Option<Vec<Arc<SsdReadLease>>> {
        let leases: Vec<_> = self
            .rows
            .iter()
            .map_while(|row| row.local_ssd()?.pin())
            .collect();
        if leases.len() < self.required
            || (self.path == SsdReadPath::Cufile
                && leases
                    .iter()
                    .any(|lease| !lease.cufile_eligible(codec_budget)))
        {
            return None;
        }
        Some(leases)
    }
}
