use std::collections::HashSet;

use crate::SsdReadPath;
use crate::block::RestoreSource;

/// Physical restore intent after the engine has allocated destination pages.
/// It owns no source or device resource; execution owners acquire those next.
#[derive(Debug)]
pub(crate) struct RestorePlan {
    device_id: i32,
    ssd_path: Option<SsdReadPath>,
    allow_uring_fallback: bool,
    ssd_source_bytes: u64,
    ssd_source_fragments: usize,
    has_memory: bool,
}

impl RestorePlan {
    pub(crate) fn new<'a>(
        device_id: i32,
        sources: impl IntoIterator<Item = (usize, &'a RestoreSource)>,
    ) -> Result<Self, String> {
        if device_id < 0 {
            return Err("restore target device must be non-negative".into());
        }
        let mut seen = HashSet::new();
        let mut plan = Self {
            device_id,
            ssd_path: None,
            allow_uring_fallback: false,
            ssd_source_bytes: 0,
            ssd_source_fragments: 0,
            has_memory: false,
        };
        for (source_id, source) in sources {
            if let RestoreSource::Ssd {
                path,
                allow_uring_fallback,
                ..
            } = source
            {
                if plan.ssd_path.is_some_and(|selected| selected != *path) {
                    return Err("one restore plan cannot mix SSD read routes".into());
                }
                if plan.ssd_path.is_some() && plan.allow_uring_fallback != *allow_uring_fallback {
                    return Err("one restore plan cannot mix SSD fallback policies".into());
                }
                plan.ssd_path = Some(*path);
                plan.allow_uring_fallback = *allow_uring_fallback;
            }
            if !seen.insert(source_id) {
                continue;
            }
            match source {
                RestoreSource::Memory(_) => plan.has_memory = true,
                RestoreSource::Ssd { lease, .. } => {
                    for slot in &lease.entry.slots {
                        plan.ssd_source_bytes = plan
                            .ssd_source_bytes
                            .checked_add(slot.total_size())
                            .ok_or("restore source bytes overflow")?;
                        plan.ssd_source_fragments = plan
                            .ssd_source_fragments
                            .checked_add(slot.num_segments())
                            .ok_or("restore source fragment count overflow")?;
                    }
                }
            }
        }
        Ok(plan)
    }

    pub(crate) fn device_id(&self) -> i32 {
        self.device_id
    }

    pub(crate) fn ssd_path(&self) -> Option<SsdReadPath> {
        self.ssd_path
    }

    pub(crate) fn fallback_from_cufile(&mut self) -> Result<bool, String> {
        if self.ssd_path != Some(SsdReadPath::Cufile) {
            return Ok(false);
        }
        if !self.allow_uring_fallback {
            return Err("explicit cuFile restore has no device staging owner".into());
        }
        self.ssd_path = Some(SsdReadPath::Uring);
        self.allow_uring_fallback = false;
        Ok(true)
    }

    pub(crate) fn ssd_source_bytes(&self) -> u64 {
        self.ssd_source_bytes
    }

    pub(crate) fn ssd_source_fragments(&self) -> usize {
        self.ssd_source_fragments
    }

    pub(crate) fn has_memory(&self) -> bool {
        self.has_memory
    }
}

#[cfg(test)]
#[path = "../../tests/unit/planning/restore.rs"]
mod tests;
