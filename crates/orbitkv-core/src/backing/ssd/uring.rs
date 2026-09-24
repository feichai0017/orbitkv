// Copyright 2025 foyer Project Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Adapted for OrbitKV: simplified to a single-file engine with blocking read/write APIs.

use io_uring::{IoUring, opcode, types::Fd};
use log::{info, warn};
use std::io;
use std::os::unix::io::RawFd;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread::JoinHandle;
use tokio::sync::oneshot;

use super::SSD_ALIGNMENT;
use crate::cost::{CostKey, CostPath, Observation, Outcome, Representation, resource_id};

const DEFAULT_URING_THREADS: usize = 16;
static NEXT_ENGINE: AtomicU64 = AtomicU64::new(1);

/// Configuration for io_uring engine.
#[derive(Debug, Clone)]
pub(super) struct UringConfig {
    pub threads: usize,
    pub io_depth: usize,
    /// Enable SQ polling; requires kernel support. Off by default.
    pub sqpoll: bool,
    /// Idle time in ms before the kernel SQ poll thread sleeps.
    pub sqpoll_idle: u32,
}

impl Default for UringConfig {
    fn default() -> Self {
        Self {
            threads: DEFAULT_URING_THREADS,
            io_depth: 128,
            sqpoll: false,
            sqpoll_idle: 10,
        }
    }
}

#[derive(Clone, Copy)]
enum IoType {
    Readv,
    Writev,
}

struct IoCtx {
    io_type: IoType,
    fd: RawFd,
    len: usize,
    offset: u64,
    complete: oneshot::Sender<io::Result<usize>>,
    iovecs: Option<Box<[libc::iovec]>>,
    requested_bytes: u64,
    observation: Observation,
}

impl IoCtx {
    fn complete(self, result: io::Result<usize>) {
        let (outcome, actual) = match &result {
            Ok(bytes) if *bytes as u64 == self.requested_bytes => {
                (Outcome::Completed, Some(*bytes as u64))
            }
            Ok(bytes) => (Outcome::Failed, Some(*bytes as u64)),
            Err(_) => (Outcome::Failed, None),
        };
        self.observation.finish(outcome, actual);
        let _ = self.complete.send(result);
    }
}

// SAFETY: IoCtx is created on one thread and sent to the io_uring shard thread
// via mpsc channel. The raw pointers in iovecs reference pinned memory that
// outlives the I/O operation (guaranteed by the caller).
unsafe impl Send for IoCtx {}

struct UringShard {
    rx: mpsc::Receiver<IoCtx>,
    uring: IoUring,
    io_depth: usize,
}

impl UringShard {
    fn run(mut self) {
        let mut inflight = 0usize;
        let mut channel_closed = false;

        loop {
            // Try to prepare as many as possible up to io_depth.
            while inflight < self.io_depth && !channel_closed {
                let next = if inflight == 0 {
                    // If idle, block until we have at least one ctx.
                    match self.rx.recv() {
                        Ok(ctx) => Some(ctx),
                        Err(e) => {
                            warn!("io_uring shard recv closed: {e}");
                            channel_closed = true;
                            None
                        }
                    }
                } else {
                    match self.rx.try_recv() {
                        Ok(ctx) => Some(ctx),
                        Err(mpsc::TryRecvError::Disconnected) => {
                            channel_closed = true;
                            None
                        }
                        Err(mpsc::TryRecvError::Empty) => None,
                    }
                };
                let mut ctx = match next {
                    Some(ctx) => ctx,
                    None => break,
                };
                ctx.observation.admitted();

                let fd = Fd(ctx.fd);
                let sqe = match ctx.io_type {
                    IoType::Readv => {
                        let iovecs_ptr = ctx
                            .iovecs
                            .as_ref()
                            .expect("readv must have iovecs")
                            .as_ptr();
                        opcode::Readv::new(fd, iovecs_ptr, ctx.len as _)
                            .offset(ctx.offset)
                            .build()
                    }
                    IoType::Writev => {
                        // Safety: iovecs must remain valid until completion
                        let iovecs_ptr = ctx
                            .iovecs
                            .as_ref()
                            .expect("writev must have iovecs")
                            .as_ptr();
                        opcode::Writev::new(fd, iovecs_ptr, ctx.len as _)
                            .offset(ctx.offset)
                            .build()
                    }
                };

                // This measures host submission through observed CQE, including
                // SQ batching; it is not an isolated device service timer.
                ctx.observation.submitted();
                let data = Box::into_raw(Box::new(ctx)) as u64;
                let sqe = sqe.user_data(data);
                // SAFETY: The Box<IoCtx> is leaked via into_raw and recovered in the
                // completion path via Box::from_raw. The iovec pointers within IoCtx
                // reference pinned memory guaranteed by the caller to remain valid
                // until the oneshot completes.
                let push_result = unsafe { self.uring.submission().push(&sqe) };
                if push_result.is_err() {
                    // Recover the leaked IoCtx to avoid memory leak.
                    let ctx = unsafe { Box::from_raw(data as *mut IoCtx) };
                    ctx.observation.finish(Outcome::Failed, Some(0));
                    let _ = ctx
                        .complete
                        .send(Err(io::Error::other("submission queue full")));
                    continue;
                }
                inflight += 1;
            }

            // If channel is closed and no inflight requests, exit gracefully.
            if channel_closed && inflight == 0 {
                info!("io_uring shard shutting down gracefully");
                return;
            }

            if inflight == 0 {
                continue;
            }

            // Submit and wait for at least one completion.
            if let Err(e) = self.uring.submit_and_wait(1) {
                // Fatal error; drop all inflight requests.
                warn!("io_uring submit_and_wait failed: {}, shutting down", e);
                while let Some(cqe) = self.uring.completion().next() {
                    let data = cqe.user_data();
                    if data != 0 {
                        // Safety: data was produced from Box::into_raw.
                        let ctx = unsafe { Box::from_raw(data as *mut IoCtx) };
                        ctx.complete(Err(io::Error::other(format!(
                            "io_uring submit failed: {e}"
                        ))));
                    }
                }
                info!("io_uring shard shut down due to submit error");
                return;
            }

            let mut completed = 0usize;
            for cqe in self.uring.completion() {
                completed += 1;
                let data = cqe.user_data();
                if data == 0 {
                    warn!("io_uring completion with user_data=0, res={}", cqe.result());
                    continue;
                }
                let ctx = unsafe { Box::from_raw(data as *mut IoCtx) };
                let res = cqe.result();
                let send_res = if res < 0 {
                    Err(io::Error::from_raw_os_error(-res))
                } else {
                    Ok(res as usize)
                };
                ctx.complete(send_res);
            }
            inflight = inflight.saturating_sub(completed);
        }
    }
}

/// io_uring based engine for read/write against one or more cache files.
pub(super) struct UringIoEngine {
    fds: Vec<RawFd>,
    resources: Vec<u64>,
    pub(super) cost_resource: u64,
    txs: Vec<mpsc::SyncSender<IoCtx>>,
    write_shards: usize,
    next_read: AtomicUsize,
    #[allow(
        dead_code,
        reason = "join handles keep io_uring shard threads alive until Drop"
    )]
    handles: Vec<JoinHandle<()>>,
}

impl UringIoEngine {
    pub(super) fn new_multi(fds: Vec<RawFd>, cfg: UringConfig) -> io::Result<Self> {
        if cfg.threads == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "threads must be > 0",
            ));
        }
        if fds.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "at least one fd is required",
            ));
        }

        let incarnation = NEXT_ENGINE.fetch_add(1, Ordering::Relaxed);
        let resources = fds
            .iter()
            .map(|&fd| {
                if !crate::cost::enabled() {
                    return 0;
                }
                let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
                // SAFETY: fds remain owned by SsdBackingStore through engine drain.
                if unsafe { libc::fstat(fd, stat.as_mut_ptr()) } != 0 {
                    return resource_id(&(incarnation, fd));
                }
                let stat = unsafe { stat.assume_init() };
                resource_id(&(incarnation, stat.st_dev, stat.st_ino))
            })
            .collect::<Vec<_>>();

        let mut txs = Vec::with_capacity(cfg.threads);
        let mut handles = Vec::with_capacity(cfg.threads);

        for idx in 0..cfg.threads {
            let (tx, rx) = mpsc::sync_channel(cfg.io_depth * 2);
            let mut builder = IoUring::builder();
            if cfg.sqpoll {
                builder.setup_sqpoll(cfg.sqpoll_idle);
            }
            let uring = builder.build(cfg.io_depth as u32)?;
            let shard = UringShard {
                rx,
                uring,
                io_depth: cfg.io_depth,
            };
            let handle = std::thread::Builder::new()
                .name(format!("orbitkv-uring-{idx}"))
                .spawn(move || shard.run())?;
            txs.push(tx);
            handles.push(handle);
        }

        // Keep each file's writes on one queue. Reads may use any of the
        // remaining queues: tying them to the file id leaves all but one
        // worker idle for a single cache file, and submit_and_wait can strand
        // a newly queued read behind an unrelated write already in flight.
        let write_shards = fds.len().min((cfg.threads / 2).max(1));
        let cost_resource = resource_id(&resources);
        Ok(Self {
            fds,
            resources,
            cost_resource,
            txs,
            write_shards,
            next_read: AtomicUsize::new(0),
            handles,
        })
    }

    fn pick_tx(&self, shard_id: usize, io_type: IoType) -> &mpsc::SyncSender<IoCtx> {
        let readers = self.txs.len() - self.write_shards;
        let idx = match io_type {
            IoType::Readv if readers > 0 => {
                self.write_shards + self.next_read.fetch_add(1, Ordering::Relaxed) % readers
            }
            _ => shard_id % self.write_shards,
        };
        &self.txs[idx]
    }

    fn fd(&self, shard_id: usize) -> io::Result<RawFd> {
        self.fds.get(shard_id).copied().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("invalid SSD shard id {shard_id}"),
            )
        })
    }

    fn validate_direct_io(
        iovecs: impl IntoIterator<Item = (usize, usize)>,
        offset: u64,
    ) -> io::Result<()> {
        Self::ensure_aligned("offset", offset as usize)?;
        for (addr, len) in iovecs {
            Self::ensure_aligned("buffer address", addr)?;
            Self::ensure_aligned("iovec length", len)?;
        }
        Ok(())
    }

    fn ensure_aligned(name: &str, value: usize) -> io::Result<()>
    where
        Self: Sized,
    {
        if value.is_multiple_of(SSD_ALIGNMENT) {
            Ok(())
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("O_DIRECT {name} {value:#x} is not {SSD_ALIGNMENT}-byte aligned"),
            ))
        }
    }

    /// Vectorized read (readv) - reads into multiple buffers in a single syscall.
    ///
    /// # Arguments
    /// * `iovecs` - Array of (ptr, len) pairs to read into sequentially
    /// * `offset` - File offset to start reading
    ///
    /// # Safety
    /// Caller must ensure all buffer pointers remain valid until the returned receiver completes.
    pub(super) fn readv_at_async(
        &self,
        shard_id: usize,
        iovecs: Vec<(*mut u8, usize)>,
        offset: u64,
    ) -> io::Result<oneshot::Receiver<io::Result<usize>>> {
        if iovecs.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "readv requires at least one iovec",
            ));
        }
        Self::validate_direct_io(
            iovecs.iter().map(|(ptr, len)| (*ptr as usize, *len)),
            offset,
        )?;

        let iovecs_libc: Box<[libc::iovec]> = iovecs
            .iter()
            .map(|(ptr, len)| libc::iovec {
                iov_base: *ptr as *mut libc::c_void,
                iov_len: *len,
            })
            .collect();

        let iovec_count = iovecs_libc.len();
        let requested_bytes = if crate::cost::enabled() {
            iovecs.iter().map(|(_, len)| *len as u64).sum()
        } else {
            0
        };
        let fd = self.fd(shard_id)?;
        let (tx, rx) = oneshot::channel();
        let ctx = IoCtx {
            io_type: IoType::Readv,
            fd,
            len: iovec_count,
            offset,
            complete: tx,
            iovecs: Some(iovecs_libc),
            requested_bytes,
            observation: Observation::new(
                CostKey::new(
                    CostPath::SsdRead,
                    self.resources[shard_id],
                    Representation::Unknown,
                    requested_bytes,
                    iovec_count,
                ),
                None,
            ),
        };

        self.pick_tx(shard_id, IoType::Readv)
            .send(ctx)
            .map_err(|e| {
                e.0.observation.finish(Outcome::Failed, Some(0));
                io::Error::new(io::ErrorKind::BrokenPipe, "io_uring readv queue closed")
            })?;
        Ok(rx)
    }

    /// Vectorized write (writev) - writes multiple buffers in a single syscall.
    ///
    /// # Arguments
    /// * `iovecs` - Array of (ptr, len) pairs to write sequentially
    /// * `offset` - File offset to start writing
    ///
    /// # Safety
    /// Caller must ensure all buffer pointers remain valid until the returned receiver completes.
    pub(super) fn writev_at_async(
        &self,
        shard_id: usize,
        iovecs: Vec<(*const u8, usize)>,
        offset: u64,
    ) -> io::Result<oneshot::Receiver<io::Result<usize>>> {
        if iovecs.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "writev requires at least one iovec",
            ));
        }
        Self::validate_direct_io(
            iovecs.iter().map(|(ptr, len)| (*ptr as usize, *len)),
            offset,
        )?;

        // Convert to libc::iovec
        let iovecs_libc: Box<[libc::iovec]> = iovecs
            .iter()
            .map(|(ptr, len)| libc::iovec {
                iov_base: *ptr as *mut libc::c_void,
                iov_len: *len,
            })
            .collect();

        let iovec_count = iovecs_libc.len();
        let requested_bytes = if crate::cost::enabled() {
            iovecs.iter().map(|(_, len)| *len as u64).sum()
        } else {
            0
        };
        let fd = self.fd(shard_id)?;
        let (tx, rx) = oneshot::channel();
        let ctx = IoCtx {
            io_type: IoType::Writev,
            fd,
            len: iovec_count, // number of iovecs
            offset,
            complete: tx,
            iovecs: Some(iovecs_libc),
            requested_bytes,
            observation: Observation::new(
                CostKey::new(
                    CostPath::SsdWrite,
                    self.resources[shard_id],
                    Representation::Unknown,
                    requested_bytes,
                    iovec_count,
                ),
                None,
            ),
        };

        self.pick_tx(shard_id, IoType::Writev)
            .send(ctx)
            .map_err(|e| {
                e.0.observation.finish(Outcome::Failed, Some(0));
                io::Error::new(io::ErrorKind::BrokenPipe, "io_uring writev queue closed")
            })?;
        Ok(rx)
    }
}

impl Drop for UringIoEngine {
    fn drop(&mut self) {
        // Drop senders to unblock shards, then join.
        self.txs.clear();
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/backing/ssd/uring.rs"]
mod tests;
