use crate::plan::RetentionKind;
use std::ops::{Deref, DerefMut};

use super::error::KvManagerError;
use super::identity::{ReclamationLease, StepLease, SubmissionLease};
use super::protocol::BackendArenaRegistration;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct RuntimeClass {
    pub(super) class_id: u16,
    pub(super) retention: RetentionKind,
    pub(super) window_tokens: Option<u64>,
    pub(super) period_blocks: Option<u64>,
    pub(super) backend: BackendArenaRegistration,
    pub(super) first_page_id: u32,
}

impl RuntimeClass {
    pub(super) fn candidate_start(self, previous_boundary: u64) -> u64 {
        match self.retention {
            RetentionKind::Full => 0,
            RetentionKind::Sliding => previous_boundary.saturating_sub(self.history_tokens()),
            RetentionKind::Chunked => unreachable!("canonical profile rejects chunked retention"),
        }
    }

    pub(super) fn retained_start(self, target_boundary: u64) -> u64 {
        match self.retention {
            RetentionKind::Full => 0,
            RetentionKind::Sliding => target_boundary.saturating_sub(self.history_tokens()),
            RetentionKind::Chunked => unreachable!("canonical profile rejects chunked retention"),
        }
    }

    pub(super) fn history_tokens(self) -> u64 {
        self.window_tokens
            .expect("validated sliding class has a window")
            .saturating_sub(1)
    }

    pub(super) fn temporal_address(self, ordinal: u64) -> (u64, u64) {
        match self.retention {
            RetentionKind::Full => (ordinal, 0),
            RetentionKind::Sliding => {
                let period = self
                    .period_blocks
                    .expect("validated sliding class has a period");
                (ordinal % period, ordinal / period)
            }
            RetentionKind::Chunked => unreachable!("canonical profile rejects chunked retention"),
        }
    }

    pub(super) fn contains_page(self, page_id: u32) -> bool {
        page_id >= self.first_page_id && page_id - self.first_page_id < self.backend.page_count
    }

    pub(super) fn backend_index(self, page_id: u32) -> Result<u64, KvManagerError> {
        if !self.contains_page(page_id) {
            return Err(KvManagerError::WrongPageArena);
        }
        self.backend
            .backend_base_index
            .checked_add(u64::from(page_id - self.first_page_id))
            .ok_or(KvManagerError::ArithmeticOverflow("backend page index"))
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PagePhase {
    Free,
    Reserved { step: StepLease },
    Live,
    Retiring { reclamation: ReclamationLease },
    Quarantined,
    Exhausted,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct PageState {
    pub(super) class_id: u16,
    pub(super) generation: u64,
    pub(super) request_refs: u32,
    pub(super) prefix_refs: u32,
    pub(super) reader_pins: u32,
    pub(super) writer: Option<SubmissionLease>,
    pub(super) completion_domain: u64,
    pub(super) completion_value: u64,
    pub(super) phase: PagePhase,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct PageCounts {
    pub(super) free: u64,
    pub(super) reserved: u64,
    pub(super) writing: u64,
    pub(super) active: u64,
    pub(super) retiring: u64,
    pub(super) quarantined: u64,
    pub(super) exhausted: u64,
    pub(super) request_refs: u64,
    pub(super) prefix_refs: u64,
    pub(super) reader_pins: u64,
}

impl PageCounts {
    pub(super) fn increment(&mut self, phase: PagePhase) {
        *self.counter_mut(phase) += 1;
    }

    pub(super) fn decrement(&mut self, phase: PagePhase) {
        let counter = self.counter_mut(phase);
        debug_assert!(*counter > 0);
        *counter -= 1;
    }

    fn counter_mut(&mut self, phase: PagePhase) -> &mut u64 {
        match phase {
            PagePhase::Free => &mut self.free,
            PagePhase::Reserved { .. } => &mut self.reserved,
            PagePhase::Live => &mut self.active,
            PagePhase::Retiring { .. } => &mut self.retiring,
            PagePhase::Quarantined => &mut self.quarantined,
            PagePhase::Exhausted => &mut self.exhausted,
        }
    }

    pub(super) fn apply_page_change(&mut self, before: PageState, after: PageState) {
        debug_assert_eq!(before.class_id, after.class_id);
        if std::mem::discriminant(&before.phase) != std::mem::discriminant(&after.phase) {
            self.decrement(before.phase);
            self.increment(after.phase);
        }
        Self::replace_total(
            &mut self.writing,
            u64::from(before.phase == PagePhase::Live && before.writer.is_some()),
            u64::from(after.phase == PagePhase::Live && after.writer.is_some()),
        );
        Self::replace_total(
            &mut self.request_refs,
            u64::from(before.request_refs),
            u64::from(after.request_refs),
        );
        Self::replace_total(
            &mut self.prefix_refs,
            u64::from(before.prefix_refs),
            u64::from(after.prefix_refs),
        );
        Self::replace_total(
            &mut self.reader_pins,
            u64::from(before.reader_pins),
            u64::from(after.reader_pins),
        );
    }

    #[cfg(test)]
    pub(super) fn increment_page(&mut self, page: PageState) {
        self.increment(page.phase);
        self.writing += u64::from(page.phase == PagePhase::Live && page.writer.is_some());
        self.request_refs += u64::from(page.request_refs);
        self.prefix_refs += u64::from(page.prefix_refs);
        self.reader_pins += u64::from(page.reader_pins);
    }

    fn replace_total(counter: &mut u64, before: u64, after: u64) {
        if after >= before {
            *counter = counter
                .checked_add(after - before)
                .expect("page census cannot overflow u64");
        } else {
            *counter = counter
                .checked_sub(before - after)
                .expect("page census cannot underflow");
        }
    }
}

pub(super) struct PageMut<'a> {
    page: &'a mut PageState,
    counts: &'a mut PageCounts,
    before: PageState,
}

impl<'a> PageMut<'a> {
    pub(super) fn new(page: &'a mut PageState, counts: &'a mut PageCounts) -> Self {
        Self {
            before: *page,
            page,
            counts,
        }
    }
}

impl Deref for PageMut<'_> {
    type Target = PageState;

    fn deref(&self) -> &Self::Target {
        self.page
    }
}

impl DerefMut for PageMut<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.page
    }
}

impl Drop for PageMut<'_> {
    fn drop(&mut self) {
        self.counts.apply_page_change(self.before, *self.page);
    }
}

impl PageState {
    pub(super) const fn free(class_id: u16) -> Self {
        Self {
            class_id,
            generation: 0,
            request_refs: 0,
            prefix_refs: 0,
            reader_pins: 0,
            writer: None,
            completion_domain: 0,
            completion_value: 0,
            phase: PagePhase::Free,
        }
    }
}
#[derive(Debug)]
pub(super) struct ArenaSlot<T> {
    pub(super) generation: u32,
    pub(super) value: Option<T>,
}

#[derive(Debug)]
pub(super) struct Arena<T> {
    pub(super) label: &'static str,
    pub(super) slots: Vec<ArenaSlot<T>>,
    pub(super) free: Vec<u32>,
    pub(super) active: usize,
}

impl<T> Arena<T> {
    pub(super) fn new(label: &'static str, capacity: u32) -> Result<Self, KvManagerError> {
        if capacity == 0 {
            return Err(KvManagerError::ZeroCapacity(label));
        }
        let capacity = usize::try_from(capacity)
            .map_err(|_| KvManagerError::ArithmeticOverflow("arena capacity"))?;
        Ok(Self {
            label,
            slots: (0..capacity)
                .map(|_| ArenaSlot {
                    generation: 0,
                    value: None,
                })
                .collect(),
            free: (0..u32::try_from(capacity)
                .map_err(|_| KvManagerError::ArithmeticOverflow("arena capacity"))?)
                .rev()
                .collect(),
            active: 0,
        })
    }

    pub(super) fn plan_many(&self, count: usize) -> Result<Vec<(u32, u32)>, KvManagerError> {
        if self.free.len() < count {
            return Err(KvManagerError::ArenaExhausted(self.label));
        }
        Ok(self
            .free
            .iter()
            .rev()
            .take(count)
            .map(|&slot| {
                let state = &self.slots[slot as usize];
                let generation = state
                    .generation
                    .checked_add(1)
                    .expect("generation-exhausted slots are not free");
                (slot, generation)
            })
            .collect())
    }

    pub(super) fn insert_planned(&mut self, planned: (u32, u32), value: T) {
        let (slot, generation) = planned;
        let index = usize::try_from(slot).expect("planned arena slot fits usize");
        let popped = self.free.pop();
        assert_eq!(popped, Some(slot), "planned arena slot remains stack head");
        let state = &mut self.slots[index];
        debug_assert!(state.value.is_none());
        debug_assert_eq!(state.generation.checked_add(1), Some(generation));
        state.generation = generation;
        state.value = Some(value);
        self.active += 1;
    }

    pub(super) fn get(&self, slot: u32, generation: u32) -> Result<&T, KvManagerError> {
        let state = self.slot(slot)?;
        if state.generation != generation {
            return Err(KvManagerError::StaleLease(self.label));
        }
        state
            .value
            .as_ref()
            .ok_or(KvManagerError::StaleLease(self.label))
    }

    pub(super) fn get_mut(&mut self, slot: u32, generation: u32) -> Result<&mut T, KvManagerError> {
        let label = self.label;
        let state = self.slot_mut(slot)?;
        if state.generation != generation {
            return Err(KvManagerError::StaleLease(label));
        }
        state
            .value
            .as_mut()
            .ok_or(KvManagerError::StaleLease(label))
    }

    pub(super) fn remove(&mut self, slot: u32, generation: u32) -> Result<T, KvManagerError> {
        let label = self.label;
        let state = self.slot_mut(slot)?;
        if state.generation != generation {
            return Err(KvManagerError::StaleLease(label));
        }
        let value = state
            .value
            .take()
            .ok_or(KvManagerError::StaleLease(label))?;
        if state.generation != u32::MAX {
            self.free.push(slot);
        }
        self.active -= 1;
        Ok(value)
    }

    fn slot(&self, slot: u32) -> Result<&ArenaSlot<T>, KvManagerError> {
        self.slots
            .get(usize::try_from(slot).map_err(|_| KvManagerError::StaleLease(self.label))?)
            .ok_or(KvManagerError::StaleLease(self.label))
    }

    fn slot_mut(&mut self, slot: u32) -> Result<&mut ArenaSlot<T>, KvManagerError> {
        self.slots
            .get_mut(usize::try_from(slot).map_err(|_| KvManagerError::StaleLease(self.label))?)
            .ok_or(KvManagerError::StaleLease(self.label))
    }

    pub(super) fn active_len(&self) -> usize {
        self.active
    }
}
