//! Session-owned restore grants. Source ownership outlives engine DMA, and slot
//! reuse requires Manager reaping followed by engine acknowledgement.

use std::collections::BTreeMap;
use std::fs::File;
use std::ops::Range;
use std::os::fd::OwnedFd;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use memmap2::{MmapMut, MmapOptions};
use rustix::event::{EventfdFlags, eventfd};
use rustix::fs::{MemfdFlags, SealFlags, fcntl_add_seals, ftruncate, memfd_create};
use thiserror::Error;

use crate::{RestoreResponse, RestoreState};

pub const RESTORE_COMPLETION_SLOTS: usize = 1024;
pub const RESTORE_ERROR_BYTES: usize = 88;
pub const RESTORE_PLAN_BYTES: usize = 1024 * 1024;
const HEADER_BYTES: usize = 4096;
const RECORD_BYTES: usize = 128;
const PLAN_OFFSET: usize = HEADER_BYTES + RESTORE_COMPLETION_SLOTS * RECORD_BYTES;
const MAPPING_BYTES: usize = PLAN_OFFSET + RESTORE_PLAN_BYTES;
const MAGIC_VERSION: u64 = 0x0003_4f52_4243;
const NEXT_OPERATION_OFFSET: usize = 24;
const DIRTY_OFFSET: usize = 64;
const DIRTY_WORDS: usize = RESTORE_COMPLETION_SLOTS / 64;
const STATE_BITS: u32 = 4;
const STATE_MASK: u64 = (1 << STATE_BITS) - 1;
const MAX_OPERATION_ID: u64 = u64::MAX >> STATE_BITS;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u64)]
pub enum GrantState {
    Reserved = 0,
    Preparing = 1,
    CancelRequested = 2,
    Granted = 3,
    Active = 4,
    Drained = 5,
    Revoked = 6,
    Managed = 7,
    Reaped = 8,
    Acknowledged = 9,
}

impl GrantState {
    fn decode(tag: u64) -> Result<Self, CompletionError> {
        match tag & STATE_MASK {
            0 => Ok(Self::Reserved),
            1 => Ok(Self::Preparing),
            2 => Ok(Self::CancelRequested),
            3 => Ok(Self::Granted),
            4 => Ok(Self::Active),
            5 => Ok(Self::Drained),
            6 => Ok(Self::Revoked),
            7 => Ok(Self::Managed),
            8 => Ok(Self::Reaped),
            9 => Ok(Self::Acknowledged),
            _ => Err(CompletionError::InvalidPayload),
        }
    }
}

#[derive(Debug, Error)]
pub enum CompletionError {
    #[error("restore grant memory operation failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid restore grant mapping or session identity")]
    InvalidMapping,
    #[error("all restore grant slots are unconsumed")]
    Full,
    #[error("restore plan bank exhausted")]
    PlanFull,
    #[error("restore operation ids exhausted")]
    Exhausted,
    #[error("unknown, consumed, or invalid-state restore operation {0}")]
    Stale(u64),
    #[error("invalid restore grant payload")]
    InvalidPayload,
}

/// Exactly one Manager instance allocates plan-bank ranges. Every shared access
/// is atomic so rejected stale readers cannot race record or plan recycling.
pub struct RestoreCompletions {
    file: File,
    map: MmapMut,
    notification: OwnedFd,
    manager_notification: OwnedFd,
    plans: Mutex<BTreeMap<u64, Range<usize>>>,
}

impl RestoreCompletions {
    pub(crate) fn create(
        epoch: u64,
        token: u64,
        notification: OwnedFd,
    ) -> Result<Self, CompletionError> {
        let fd = memfd_create(
            "orbitkv-restore-grants",
            MemfdFlags::CLOEXEC | MemfdFlags::ALLOW_SEALING,
        )
        .map_err(std::io::Error::from)?;
        ftruncate(&fd, MAPPING_BYTES as u64).map_err(std::io::Error::from)?;
        fcntl_add_seals(&fd, SealFlags::GROW | SealFlags::SHRINK | SealFlags::SEAL)
            .map_err(std::io::Error::from)?;
        let manager_notification = eventfd(0, EventfdFlags::CLOEXEC | EventfdFlags::NONBLOCK)
            .map_err(std::io::Error::from)?;
        let this = Self::map(File::from(fd), notification, manager_notification)?;
        this.word(0).store(MAGIC_VERSION, Ordering::Relaxed);
        this.word(8).store(epoch, Ordering::Relaxed);
        this.word(NEXT_OPERATION_OFFSET).store(1, Ordering::Relaxed);
        this.word(16).store(token, Ordering::Release);
        Ok(this)
    }

    pub(crate) fn open(
        fd: OwnedFd,
        notification: OwnedFd,
        manager_notification: OwnedFd,
        epoch: u64,
        token: u64,
    ) -> Result<Self, CompletionError> {
        let this = Self::map(File::from(fd), notification, manager_notification)?;
        if this.word(16).load(Ordering::Acquire) != token
            || this.word(8).load(Ordering::Relaxed) != epoch
            || this.word(0).load(Ordering::Relaxed) != MAGIC_VERSION
        {
            return Err(CompletionError::InvalidMapping);
        }
        Ok(this)
    }

    fn map(
        file: File,
        notification: OwnedFd,
        manager_notification: OwnedFd,
    ) -> Result<Self, CompletionError> {
        if file.metadata()?.len() != MAPPING_BYTES as u64 {
            return Err(CompletionError::InvalidMapping);
        }
        let seals = rustix::fs::fcntl_get_seals(&file).map_err(std::io::Error::from)?;
        if !seals.contains(SealFlags::GROW | SealFlags::SHRINK | SealFlags::SEAL) {
            return Err(CompletionError::InvalidMapping);
        }
        // The sealed memfd cannot shrink. All offsets are aligned AtomicU64s.
        let map = unsafe { MmapOptions::new().len(MAPPING_BYTES).map_mut(&file)? };
        Ok(Self {
            file,
            map,
            notification,
            manager_notification,
            plans: Mutex::new(BTreeMap::new()),
        })
    }

    pub(crate) fn file(&self) -> &File {
        &self.file
    }
    pub fn notification_fd(&self) -> &OwnedFd {
        &self.notification
    }
    pub fn manager_notification_fd(&self) -> &OwnedFd {
        &self.manager_notification
    }
    pub fn notify(&self) -> Result<(), CompletionError> {
        Self::signal(&self.notification)
    }

    fn signal(fd: &OwnedFd) -> Result<(), CompletionError> {
        loop {
            match rustix::io::write(fd, &1u64.to_ne_bytes()) {
                Ok(8) => return Ok(()),
                Ok(_) => return Err(std::io::Error::from(std::io::ErrorKind::WriteZero).into()),
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => return Ok(()),
                Err(error) => return Err(std::io::Error::from(error).into()),
            }
        }
    }

    fn word(&self, offset: usize) -> &AtomicU64 {
        assert!(offset.is_multiple_of(8) && offset + 8 <= self.map.len());
        unsafe { &*self.map.as_ptr().add(offset).cast::<AtomicU64>() }
    }

    fn offset(id: u64) -> Result<usize, CompletionError> {
        if id == 0 || id > MAX_OPERATION_ID {
            return Err(CompletionError::Stale(id));
        }
        Ok(HEADER_BYTES + id as usize % RESTORE_COMPLETION_SLOTS * RECORD_BYTES)
    }

    fn transition(&self, id: u64, from: GrantState, to: GrantState) -> Result<(), CompletionError> {
        self.word(Self::offset(id)?)
            .compare_exchange(
                (id << STATE_BITS) | from as u64,
                (id << STATE_BITS) | to as u64,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .map_err(|_| CompletionError::Stale(id))?;
        Ok(())
    }

    pub fn state(&self, id: u64) -> Result<GrantState, CompletionError> {
        let tag = self.word(Self::offset(id)?).load(Ordering::Acquire);
        if tag >> STATE_BITS != id {
            return Err(CompletionError::Stale(id));
        }
        GrantState::decode(tag)
    }

    pub fn reserve(&self) -> Result<u64, CompletionError> {
        for _ in 0..RESTORE_COMPLETION_SLOTS {
            let id = self
                .word(NEXT_OPERATION_OFFSET)
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
                    (next != 0 && next <= MAX_OPERATION_ID).then(|| next + 1)
                })
                .map_err(|_| CompletionError::Exhausted)?;
            let record = Self::offset(id)?;
            let tag = self.word(record).load(Ordering::Acquire);
            if (tag == 0 || tag & STATE_MASK == GrantState::Acknowledged as u64)
                && id > tag >> STATE_BITS
                && self
                    .word(record)
                    .compare_exchange(tag, id << STATE_BITS, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
            {
                return Ok(id);
            }
        }
        Err(CompletionError::Full)
    }

    /// Deduplicate admission before consuming any leases.
    pub fn claim(&self, id: u64) -> Result<(), CompletionError> {
        self.transition(id, GrantState::Reserved, GrantState::Preparing)
    }

    /// Only cancellation before admission proves the operation never ran.
    pub fn cancel(&self, id: u64) -> Result<bool, CompletionError> {
        loop {
            match self.state(id)? {
                GrantState::Reserved => {
                    if self
                        .transition(id, GrantState::Reserved, GrantState::Acknowledged)
                        .is_ok()
                    {
                        return Ok(true);
                    }
                }
                GrantState::Preparing => {
                    if self
                        .transition(id, GrantState::Preparing, GrantState::CancelRequested)
                        .is_ok()
                    {
                        return Ok(false);
                    }
                }
                GrantState::Acknowledged => return Err(CompletionError::Stale(id)),
                _ => return Ok(false),
            }
        }
    }

    pub fn start_managed(&self, id: u64) -> Result<(), CompletionError> {
        loop {
            let state = self.state(id)?;
            if !matches!(state, GrantState::Preparing | GrantState::CancelRequested) {
                return Err(CompletionError::Stale(id));
            }
            if self.transition(id, state, GrantState::Managed).is_ok() {
                return Ok(());
            }
        }
    }

    /// The caller installs source owners before making a plan claimable.
    pub fn publish_local(&self, id: u64, plan: &[u8]) -> Result<bool, CompletionError> {
        if self.state(id)? == GrantState::CancelRequested {
            return Ok(false);
        }
        if self.state(id)? != GrantState::Preparing {
            return Err(CompletionError::Stale(id));
        }
        if plan.is_empty() || plan.len() > RESTORE_PLAN_BYTES {
            return Err(CompletionError::PlanFull);
        }
        let size = plan.len().next_multiple_of(8);
        let mut plans = self
            .plans
            .lock()
            .map_err(|_| CompletionError::InvalidMapping)?;
        if plans.contains_key(&id) {
            return Err(CompletionError::Stale(id));
        }
        let mut ranges: Vec<_> = plans.values().collect();
        ranges.sort_unstable_by_key(|range| range.start);
        let mut start = 0;
        for range in ranges {
            if range.start - start >= size {
                break;
            }
            start = range.end;
        }
        if start + size > RESTORE_PLAN_BYTES {
            return Err(CompletionError::PlanFull);
        }
        for (index, chunk) in plan.chunks(8).enumerate() {
            let mut word = [0; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            self.word(PLAN_OFFSET + start + index * 8)
                .store(u64::from_ne_bytes(word), Ordering::Relaxed);
        }
        let record = Self::offset(id)?;
        self.word(record + 8).store(start as u64, Ordering::Relaxed);
        self.word(record + 16)
            .store(plan.len() as u64, Ordering::Relaxed);
        self.word(record + 24).store(0, Ordering::Relaxed);
        plans.insert(id, start..start + size);
        match self.transition(id, GrantState::Preparing, GrantState::Granted) {
            Ok(()) => Ok(true),
            Err(_) if self.state(id)? == GrantState::CancelRequested => {
                plans.remove(&id);
                Ok(false)
            }
            Err(error) => {
                plans.remove(&id);
                Err(error)
            }
        }
    }

    /// Claim precedes every read of the plan. The returned bytes no longer refer
    /// to the shared bank, allowing independent plan and source reclamation.
    pub fn claim_local(&self, id: u64) -> Result<Option<Vec<u8>>, CompletionError> {
        match self.state(id)? {
            GrantState::Granted => {}
            GrantState::Acknowledged => return Err(CompletionError::Stale(id)),
            _ => return Ok(None),
        }
        self.transition(id, GrantState::Granted, GrantState::Active)?;
        let record = Self::offset(id)?;
        let offset = self.word(record + 8).load(Ordering::Relaxed) as usize;
        let len = self.word(record + 16).load(Ordering::Relaxed) as usize;
        if len == 0
            || !offset.is_multiple_of(8)
            || len > RESTORE_PLAN_BYTES
            || offset > RESTORE_PLAN_BYTES - len
        {
            return Err(CompletionError::InvalidPayload);
        }
        let mut plan = Vec::with_capacity(len.next_multiple_of(8));
        for part in (0..len).step_by(8) {
            plan.extend_from_slice(
                &self
                    .word(PLAN_OFFSET + offset + part)
                    .load(Ordering::Relaxed)
                    .to_ne_bytes(),
            );
        }
        plan.truncate(len);
        self.word(record + 24).store(1, Ordering::Release);
        self.dirty(id)?;
        Ok(Some(plan))
    }

    fn dirty(&self, id: u64) -> Result<(), CompletionError> {
        let slot = id as usize % RESTORE_COMPLETION_SLOTS;
        self.word(DIRTY_OFFSET + slot / 64 * 8)
            .fetch_or(1 << (slot % 64), Ordering::Release);
        Self::signal(&self.manager_notification)
    }

    /// A bounded bitset cannot overflow. Clearing precedes state inspection;
    /// concurrent updates either appear in that inspection or set the next bit.
    pub fn manager_updates(&self) -> Result<Vec<(u64, GrantState)>, CompletionError> {
        let mut counter = [0; 8];
        loop {
            match rustix::io::read(&self.manager_notification, &mut counter) {
                Ok(_) => break,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(std::io::Error::from(error).into()),
            }
        }
        let mut updates = Vec::new();
        for index in 0..DIRTY_WORDS {
            let mut bits = self
                .word(DIRTY_OFFSET + index * 8)
                .swap(0, Ordering::AcqRel);
            while bits != 0 {
                let slot = index * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let tag = self
                    .word(HEADER_BYTES + slot * RECORD_BYTES)
                    .load(Ordering::Acquire);
                if tag != 0 {
                    updates.push((tag >> STATE_BITS, GrantState::decode(tag)?));
                }
            }
        }
        Ok(updates)
    }

    pub fn release_plan(&self, id: u64) -> Result<(), CompletionError> {
        let record = Self::offset(id)?;
        let state = self.state(id)?;
        if !matches!(state, GrantState::Revoked | GrantState::Drained)
            && !(state == GrantState::Active && self.word(record + 24).load(Ordering::Acquire) == 1)
        {
            return Err(CompletionError::Stale(id));
        }
        self.plans
            .lock()
            .map_err(|_| CompletionError::InvalidMapping)?
            .remove(&id);
        Ok(())
    }

    pub fn revoke(&self, id: u64) -> Result<bool, CompletionError> {
        match self.transition(id, GrantState::Granted, GrantState::Revoked) {
            Ok(()) => {
                self.dirty(id)?;
                Ok(true)
            }
            Err(_)
                if matches!(
                    self.state(id)?,
                    GrantState::Active
                        | GrantState::Drained
                        | GrantState::Reaped
                        | GrantState::Acknowledged
                ) =>
            {
                Ok(false)
            }
            Err(error) => Err(error),
        }
    }

    /// Only the native executor may publish this, after no-submit proof or drain.
    pub fn drained(&self, id: u64, result: Result<(), String>) -> Result<(), CompletionError> {
        if self.state(id)? != GrantState::Active {
            return Err(CompletionError::Stale(id));
        }
        self.write_result(id, result)?;
        self.transition(id, GrantState::Active, GrantState::Drained)?;
        self.dirty(id)
    }

    pub fn drain_succeeded(&self, id: u64) -> Result<bool, CompletionError> {
        match self.state(id)? {
            GrantState::Drained => {
                Ok(self.word(Self::offset(id)? + 32).load(Ordering::Relaxed) & 1 == 0)
            }
            GrantState::Revoked => Ok(false),
            _ => Err(CompletionError::Stale(id)),
        }
    }

    /// Source owners must already have been released by the Manager caller.
    pub fn reap(&self, id: u64) -> Result<(), CompletionError> {
        let state = self.state(id)?;
        if !matches!(state, GrantState::Drained | GrantState::Revoked) {
            return Err(CompletionError::Stale(id));
        }
        self.plans
            .lock()
            .map_err(|_| CompletionError::InvalidMapping)?
            .remove(&id);
        if state == GrantState::Revoked {
            self.write_result(id, Err("restore grant revoked".into()))?;
        }
        self.transition(id, state, GrantState::Reaped)?;
        self.notify()
    }

    pub fn finish_cancelled(&self, id: u64) -> Result<(), CompletionError> {
        self.reject(id, "restore preparation cancelled".into())
    }

    /// Preparation owns no submitted GPU work when this method is called.
    pub fn reject(&self, id: u64, message: String) -> Result<(), CompletionError> {
        loop {
            let state = self.state(id)?;
            if !matches!(state, GrantState::Preparing | GrantState::CancelRequested) {
                return Err(CompletionError::Stale(id));
            }
            self.write_result(id, Err(message.clone()))?;
            if self.transition(id, state, GrantState::Reaped).is_ok() {
                return self.notify();
            }
        }
    }

    /// The Manager worker has drained and released its own payload owners.
    pub fn complete(&self, id: u64, result: Result<(), String>) -> Result<(), CompletionError> {
        if self.state(id)? != GrantState::Managed {
            return Err(CompletionError::Stale(id));
        }
        self.write_result(id, result)?;
        self.transition(id, GrantState::Managed, GrantState::Reaped)
    }

    fn write_result(&self, id: u64, result: Result<(), String>) -> Result<(), CompletionError> {
        let record = Self::offset(id)?;
        let (failed, mut message) = match result {
            Ok(()) => (0, String::new()),
            Err(message) => (1, message),
        };
        let mut end = message.len().min(RESTORE_ERROR_BYTES);
        while !message.is_char_boundary(end) {
            end -= 1;
        }
        message.truncate(end);
        for (index, chunk) in message.as_bytes().chunks(8).enumerate() {
            let mut word = [0; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            self.word(record + 40 + index * 8)
                .store(u64::from_ne_bytes(word), Ordering::Relaxed);
        }
        self.word(record + 32)
            .store((message.len() as u64) << 1 | failed, Ordering::Relaxed);
        Ok(())
    }

    pub fn poll(&self, id: u64) -> Result<RestoreResponse, CompletionError> {
        let record = Self::offset(id)?;
        let state = self.state(id)?;
        if state == GrantState::Acknowledged {
            return Err(CompletionError::Stale(id));
        }
        if state != GrantState::Reaped {
            return Ok(RestoreResponse {
                operation_id: id,
                state: RestoreState::Pending,
                message: String::new(),
            });
        }
        let result = self.word(record + 32).load(Ordering::Relaxed);
        let len = (result >> 1) as usize;
        if len > RESTORE_ERROR_BYTES {
            return Err(CompletionError::InvalidPayload);
        }
        let mut message = Vec::with_capacity(len.next_multiple_of(8));
        for offset in (0..len).step_by(8) {
            message.extend_from_slice(
                &self
                    .word(record + 40 + offset)
                    .load(Ordering::Relaxed)
                    .to_ne_bytes(),
            );
        }
        message.truncate(len);
        self.transition(id, GrantState::Reaped, GrantState::Acknowledged)?;
        Ok(RestoreResponse {
            operation_id: id,
            state: if result & 1 == 0 {
                RestoreState::Succeeded
            } else {
                RestoreState::Failed
            },
            message: String::from_utf8(message).map_err(|_| CompletionError::InvalidPayload)?,
        })
    }
}

#[cfg(test)]
#[path = "../tests/unit/completion.rs"]
mod tests;
