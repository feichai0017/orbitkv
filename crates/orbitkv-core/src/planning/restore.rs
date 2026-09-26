use std::collections::HashSet;

use crate::SsdReadPath;
use crate::block::RestoreSource;

/// Physical restore intent after the engine has allocated destination pages.
/// It owns no source or device resource; execution owners acquire those next.
#[derive(Debug)]
pub(crate) struct RestorePlan {
    device_id: i32,
    ssd_path: Option<SsdReadPath>,
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
            ssd_source_bytes: 0,
            ssd_source_fragments: 0,
            has_memory: false,
        };
        for (source_id, source) in sources {
            if let RestoreSource::Ssd { path, .. } = source {
                if plan.ssd_path.is_some_and(|selected| selected != *path) {
                    return Err("one restore plan cannot mix SSD read routes".into());
                }
                plan.ssd_path = Some(*path);
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
