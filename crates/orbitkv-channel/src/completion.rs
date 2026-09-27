//! Session-owned restore results. Acknowledgement, never a deadline, reclaims a slot.

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
const MAGIC_VERSION: u64 = 0x0001_4f52_4243;
const SUCCEEDED: u64 = 1;
const FAILED: u64 = 2;
const ACKNOWLEDGED: u64 = 3;
const MAX_OPERATION_ID: u64 = u64::MAX >> 2;

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

    /// Reserve before submitting restore work. IDs never repeat during a session.
    pub fn reserve(&self, next_id: &mut u64) -> Result<u64, CompletionError> {
        for _ in 0..RESTORE_COMPLETION_SLOTS {
            let id = *next_id;
            if id == 0 || id > MAX_OPERATION_ID {
                return Err(CompletionError::Exhausted);
            }
            *next_id += 1;
            let (record, _) = Self::offsets(id)?;
            let tag = self.word(record).load(Ordering::Acquire);
            if (tag == 0 || tag & 3 == ACKNOWLEDGED)
                && id > (tag >> 2)
                && self
                    .word(record)
                    .compare_exchange(tag, id << 2, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
            {
                return Ok(id);
            }
        }
        Err(CompletionError::Full)
    }

    /// Roll back admission only when no restore was submitted.
    pub fn abandon(&self, operation_id: u64) -> Result<(), CompletionError> {
        let (record, _) = Self::offsets(operation_id)?;
        self.word(record)
            .compare_exchange(
                operation_id << 2,
                (operation_id << 2) | ACKNOWLEDGED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .map_err(|_| CompletionError::Stale(operation_id))?;
        Ok(())
    }

    /// Called once by the restore outcome owner, after submitted work has drained.
    pub fn complete(
        &self,
        operation_id: u64,
        result: Result<(), String>,
    ) -> Result<(), CompletionError> {
        let (record, error) = Self::offsets(operation_id)?;
        let pending = operation_id << 2;
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
                pending | status,
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
        if tag >> 2 != operation_id || tag & 3 == ACKNOWLEDGED {
            return Err(CompletionError::Stale(operation_id));
        }
        let state = match tag & 3 {
            0 => RestoreState::Pending,
            SUCCEEDED => RestoreState::Succeeded,
            FAILED => RestoreState::Failed,
            _ => unreachable!("acknowledgement checked above"),
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
                    (operation_id << 2) | ACKNOWLEDGED,
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
