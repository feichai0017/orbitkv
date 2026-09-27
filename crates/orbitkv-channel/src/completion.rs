//! Session-owned restore admission and results. Client cancellation can reclaim an
//! unclaimed reservation; only consuming a drained result reclaims a claimed slot.

use std::fs::File;
use std::os::fd::OwnedFd;
use std::sync::atomic::{AtomicU64, Ordering};

use memmap2::{MmapMut, MmapOptions};
use rustix::fs::{MemfdFlags, SealFlags, fcntl_add_seals, ftruncate, memfd_create};
use thiserror::Error;

use crate::{RestoreResponse, RestoreState};

pub const RESTORE_COMPLETION_SLOTS: usize = 1024;
pub const RESTORE_ERROR_BYTES: usize = 4096;
const HEADER_BYTES: usize = 4096;
const RECORD_BYTES: usize = 64;
const ERROR_OFFSET: usize = HEADER_BYTES + RESTORE_COMPLETION_SLOTS * RECORD_BYTES;
const MAPPING_BYTES: usize = ERROR_OFFSET + RESTORE_COMPLETION_SLOTS * RESTORE_ERROR_BYTES;
const MAGIC_VERSION: u64 = 0x0002_4f52_4243;
const NEXT_OPERATION_OFFSET: usize = 24;
const STATE_BITS: u32 = 3;
const STATE_MASK: u64 = (1 << STATE_BITS) - 1;
const RESERVED: u64 = 0;
const EXECUTING: u64 = 1;
const SUCCEEDED: u64 = 2;
const FAILED: u64 = 3;
const ACKNOWLEDGED: u64 = 4;
const MAX_OPERATION_ID: u64 = u64::MAX >> STATE_BITS;

#[derive(Debug, Error)]
pub enum CompletionError {
    #[error("restore completion memory operation failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid restore completion mapping or session identity")]
    InvalidMapping,
    #[error("all restore completion slots are unconsumed")]
    Full,
    #[error("restore operation ids exhausted")]
    Exhausted,
    #[error("unknown or consumed restore operation {0}")]
    Stale(u64),
    #[error("invalid restore completion payload")]
    InvalidPayload,
}

/// One session's bounded result storage, shared with exactly one client.
///
/// Successful operations touch only the compact record area. Error pages are
/// demand-zero and are touched only when storing errors. Every shared field,
/// including error bytes, is atomic: a stale reader cannot race slot recycling.
pub struct RestoreCompletions {
    file: File,
    map: MmapMut,
    notification: OwnedFd,
}

impl RestoreCompletions {
    pub(crate) fn create(
        epoch: u64,
        token: u64,
        notification: OwnedFd,
    ) -> Result<Self, CompletionError> {
        let fd = memfd_create(
            "orbitkv-restore-completions",
            MemfdFlags::CLOEXEC | MemfdFlags::ALLOW_SEALING,
        )
        .map_err(std::io::Error::from)?;
        ftruncate(&fd, MAPPING_BYTES as u64).map_err(std::io::Error::from)?;
        fcntl_add_seals(&fd, SealFlags::GROW | SealFlags::SHRINK | SealFlags::SEAL)
            .map_err(std::io::Error::from)?;
        let this = Self::map(File::from(fd), notification)?;
        this.word(0).store(MAGIC_VERSION, Ordering::Relaxed);
        this.word(8).store(epoch, Ordering::Relaxed);
        this.word(NEXT_OPERATION_OFFSET).store(1, Ordering::Relaxed);
        this.word(16).store(token, Ordering::Release);
        Ok(this)
    }

    pub(crate) fn open(
        fd: OwnedFd,
        notification: OwnedFd,
        epoch: u64,
        token: u64,
    ) -> Result<Self, CompletionError> {
        let this = Self::map(File::from(fd), notification)?;
        if this.word(16).load(Ordering::Acquire) != token
            || this.word(8).load(Ordering::Relaxed) != epoch
            || this.word(0).load(Ordering::Relaxed) != MAGIC_VERSION
        {
            return Err(CompletionError::InvalidMapping);
        }
        Ok(this)
    }

    fn map(file: File, notification: OwnedFd) -> Result<Self, CompletionError> {
        if file.metadata()?.len() != MAPPING_BYTES as u64 {
            return Err(CompletionError::InvalidMapping);
        }
        // The sealed memfd cannot shrink. All shared accesses use aligned
        // AtomicU64 words; no ordinary references to mutable payloads escape.
        let map = unsafe { MmapOptions::new().len(MAPPING_BYTES).map_mut(&file)? };
        Ok(Self {
            file,
            map,
            notification,
        })
    }

    pub(crate) fn file(&self) -> &File {
        &self.file
    }

    pub fn notification_fd(&self) -> &OwnedFd {
        &self.notification
    }

    pub fn notify(&self) -> Result<(), CompletionError> {
        match rustix::io::write(&self.notification, &1u64.to_ne_bytes()) {
            Ok(8) => Ok(()),
            Ok(_) => Err(std::io::Error::from(std::io::ErrorKind::WriteZero).into()),
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => Ok(()),
            Err(error) => Err(std::io::Error::from(error).into()),
        }
    }

    fn word(&self, offset: usize) -> &AtomicU64 {
        assert!(offset.is_multiple_of(8) && offset + 8 <= self.map.len());
        // mmap starts at a page boundary and every offset is 8-byte aligned.
        unsafe { &*self.map.as_ptr().add(offset).cast::<AtomicU64>() }
    }

    fn offsets(operation_id: u64) -> Result<(usize, usize), CompletionError> {
        if operation_id == 0 || operation_id > MAX_OPERATION_ID {
            return Err(CompletionError::Stale(operation_id));
        }
        let slot = operation_id as usize % RESTORE_COMPLETION_SLOTS;
        Ok((
            HEADER_BYTES + slot * RECORD_BYTES,
            ERROR_OFFSET + slot * RESTORE_ERROR_BYTES,
        ))
    }

    /// The client reserves an identity before sending the restore request. IDs
    /// never repeat across either mapping during this session.
    pub fn reserve(&self) -> Result<u64, CompletionError> {
        for _ in 0..RESTORE_COMPLETION_SLOTS {
            let id = self
                .word(NEXT_OPERATION_OFFSET)
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
                    (next != 0 && next <= MAX_OPERATION_ID).then(|| next + 1)
                })
                .map_err(|_| CompletionError::Exhausted)?;
            let (record, _) = Self::offsets(id)?;
            let tag = self.word(record).load(Ordering::Acquire);
            if (tag == 0 || tag & STATE_MASK == ACKNOWLEDGED)
                && id > (tag >> STATE_BITS)
                && self
                    .word(record)
                    .compare_exchange(
                        tag,
                        (id << STATE_BITS) | RESERVED,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .is_ok()
            {
                return Ok(id);
            }
        }
        Err(CompletionError::Full)
    }

    /// The Manager must claim exactly once before consuming leases or submitting
    /// work. A request cancelled before this CAS can never acquire DMA ownership.
    pub fn claim(&self, operation_id: u64) -> Result<(), CompletionError> {
        let (record, _) = Self::offsets(operation_id)?;
        self.word(record)
            .compare_exchange(
                (operation_id << STATE_BITS) | RESERVED,
                (operation_id << STATE_BITS) | EXECUTING,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .map_err(|_| CompletionError::Stale(operation_id))?;
        Ok(())
    }

    /// Cancel a request only while it is unclaimed. A false result leaves the
    /// record owned by the Manager until its drained result is consumed.
    pub fn cancel(&self, operation_id: u64) -> Result<bool, CompletionError> {
        let (record, _) = Self::offsets(operation_id)?;
        match self.word(record).compare_exchange(
            (operation_id << STATE_BITS) | RESERVED,
            (operation_id << STATE_BITS) | ACKNOWLEDGED,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => Ok(true),
            Err(tag)
                if tag >> STATE_BITS == operation_id
                    && matches!(tag & STATE_MASK, EXECUTING | SUCCEEDED | FAILED) =>
            {
                Ok(false)
            }
            Err(_) => Err(CompletionError::Stale(operation_id)),
        }
    }

    /// Called once by the restore outcome owner, after submitted work has drained.
    pub fn complete(
        &self,
        operation_id: u64,
        result: Result<(), String>,
    ) -> Result<(), CompletionError> {
        let (record, error) = Self::offsets(operation_id)?;
        let pending = (operation_id << STATE_BITS) | EXECUTING;
        if self.word(record).load(Ordering::Acquire) != pending {
            return Err(CompletionError::Stale(operation_id));
        }
        let (status, message) = match result {
            Ok(()) => (SUCCEEDED, String::new()),
            Err(mut message) => {
                let mut end = message.len().min(RESTORE_ERROR_BYTES);
                while !message.is_char_boundary(end) {
                    end -= 1;
                }
                message.truncate(end);
                (FAILED, message)
            }
        };
        for (index, chunk) in message.as_bytes().chunks(8).enumerate() {
            let mut word = [0; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            self.word(error + index * 8)
                .store(u64::from_ne_bytes(word), Ordering::Relaxed);
        }
        self.word(record + 8)
            .store(message.len() as u64, Ordering::Relaxed);
        self.word(record)
            .compare_exchange(
                pending,
                (operation_id << STATE_BITS) | status,
                Ordering::Release,
                Ordering::Relaxed,
            )
            .map_err(|_| CompletionError::Stale(operation_id))?;
        Ok(())
    }

    /// Acquire a result and acknowledge it only after copying the full payload.
    /// Pending reads never acknowledge, including after a caller's wait timeout.
    pub fn poll(&self, operation_id: u64) -> Result<RestoreResponse, CompletionError> {
        let (record, error) = Self::offsets(operation_id)?;
        let tag = self.word(record).load(Ordering::Acquire);
        if tag >> STATE_BITS != operation_id || tag & STATE_MASK == ACKNOWLEDGED {
            return Err(CompletionError::Stale(operation_id));
        }
        let state = match tag & STATE_MASK {
            RESERVED | EXECUTING => RestoreState::Pending,
            SUCCEEDED => RestoreState::Succeeded,
            FAILED => RestoreState::Failed,
            _ => return Err(CompletionError::InvalidPayload),
        };
        let mut message = Vec::new();
        if state == RestoreState::Failed {
            let len = self.word(record + 8).load(Ordering::Relaxed) as usize;
            if self.word(record).load(Ordering::Acquire) != tag {
                return Err(CompletionError::Stale(operation_id));
            }
            if len > RESTORE_ERROR_BYTES {
                return Err(CompletionError::InvalidPayload);
            }
            message.reserve(len.next_multiple_of(8));
            for offset in (0..len).step_by(8) {
                message.extend_from_slice(
                    &self
                        .word(error + offset)
                        .load(Ordering::Relaxed)
                        .to_ne_bytes(),
                );
            }
            message.truncate(len);
            if self.word(record).load(Ordering::Acquire) != tag {
                return Err(CompletionError::Stale(operation_id));
            }
        }
        if state != RestoreState::Pending {
            self.word(record)
                .compare_exchange(
                    tag,
                    (operation_id << STATE_BITS) | ACKNOWLEDGED,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .map_err(|_| CompletionError::Stale(operation_id))?;
        }
        Ok(RestoreResponse {
            operation_id,
            state,
            message: String::from_utf8(message).map_err(|_| CompletionError::InvalidPayload)?,
        })
    }
}

#[cfg(test)]
#[path = "../tests/unit/completion.rs"]
mod tests;
