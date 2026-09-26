use std::sync::Arc;

use crate::SsdReadPath;
use crate::storage::ssd::{SsdReadLease, SsdStore};

use super::read::{ReadPlan, ReadTarget};
use super::replica::ReplicaSet;

/// A route over request-owned evidence, not another copy of the SSD inventory.
pub(crate) struct SsdReadPlan<'a> {
    rows: &'a [ReplicaSet],
    pub(crate) path: SsdReadPath,
    pub(crate) allow_uring_fallback: bool,
    required: usize,
}

impl ReadPlan {
    pub(crate) fn deferred_ssd(
        &self,
        store: &SsdStore,
        codec_budget: usize,
    ) -> Option<SsdReadPlan<'_>> {
        if self.target != ReadTarget::EngineRestore {
            return None;
        }
        let (path, allow_uring_fallback) =
            deferred_route(store.read_path, store.gpu_io.available())?;
        self.ssd_route(path, codec_budget, allow_uring_fallback)
    }

    pub(crate) fn ssd(&self, path: SsdReadPath, codec_budget: usize) -> Option<SsdReadPlan<'_>> {
        self.ssd_route(path, codec_budget, false)
    }

    fn ssd_route(
        &self,
        path: SsdReadPath,
        codec_budget: usize,
        allow_uring_fallback: bool,
    ) -> Option<SsdReadPlan<'_>> {
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
            allow_uring_fallback,
            required: self.required,
        })
    }
}

fn deferred_route(
    explicit: Option<SsdReadPath>,
    cufile_available: bool,
) -> Option<(SsdReadPath, bool)> {
    match explicit {
        Some(path) => Some((path, false)),
        None if cufile_available => Some((SsdReadPath::Cufile, true)),
        None => None,
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

#[cfg(test)]
#[path = "../../tests/unit/planning/ssd.rs"]
mod tests;
