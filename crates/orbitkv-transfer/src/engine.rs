//! OrbitKV-facing Mooncake Transfer Engine lifecycle and batch execution.
//!
//! OrbitKV owns state identity, leases, generation checks, and transfer plans.
//! Mooncake exclusively owns transport mechanics: registered segments, RDMA/TCP
//! selection, rails, endpoints, retries, and batch completion.

use std::collections::HashMap;
use std::ffi::{CStr, CString, OsString, c_char, c_void};
use std::os::unix::ffi::OsStrExt;
use std::ptr::NonNull;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use orbitkv_mooncake_sys as native;

use crate::error::{MooncakeError, Result};
use crate::types::{
    INVALID_BATCH, Notification, STATUS_COMPLETED, STATUS_PENDING, STATUS_WAITING, TransferOp,
    TransferSlice,
};

static ENGINE_CREATE_LOCK: Mutex<()> = Mutex::new(());

pub struct TransferEngine {
    native: native::NativeEngine,
    segments: Mutex<HashMap<String, native::SegmentId>>,
}

unsafe impl Send for TransferEngine {}
unsafe impl Sync for TransferEngine {}

impl TransferEngine {
    pub fn new(
        metadata: &str,
        local_server_name: &str,
        bind_host: &str,
        rpc_port: u64,
        nic_filter: &[String],
    ) -> Result<Self> {
        native::ensure_loaded().map_err(MooncakeError::NativeRuntime)?;
        // The upstream C ABI takes its NIC filter from a process-global
        // environment variable. Serialize initialization and restore the
        // caller's environment so multiple engines cannot leak filters into
        // one another.
        let _create_guard = ENGINE_CREATE_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let _filter_guard = ScopedNicFilter::apply(nic_filter)?;
        let metadata = CString::new(metadata)?;
        let local_server_name = CString::new(local_server_name)?;
        let bind_host = CString::new(bind_host)?;
        let native = unsafe {
            native::create(
                metadata.as_ptr(),
                local_server_name.as_ptr(),
                bind_host.as_ptr(),
                rpc_port,
            )
        };
        if native.is_null() {
            return Err(MooncakeError::Create);
        }
        Ok(Self {
            native,
            segments: Mutex::new(HashMap::new()),
        })
    }

    pub fn local_segment_name(&self) -> Result<String> {
        let mut output = [0u8; 256];
        check("getLocalIpAndPort", unsafe {
            native::local_ip_and_port(
                self.native,
                output.as_mut_ptr().cast::<c_char>(),
                output.len(),
            )
        })?;
        Ok(unsafe { CStr::from_ptr(output.as_ptr().cast::<c_char>()) }
            .to_string_lossy()
            .into_owned())
    }

    /// Register memory under a Mooncake topology location such as `cpu:0`.
    ///
    /// # Safety
    ///
    /// The memory must remain valid and pinned until it is unregistered.
    pub unsafe fn register_memory(
        &self,
        address: NonNull<u8>,
        length: usize,
        location: &str,
    ) -> Result<()> {
        let location = CString::new(location)?;
        check("registerLocalMemory", unsafe {
            native::register_memory(
                self.native,
                address.as_ptr().cast::<c_void>(),
                length,
                location.as_ptr(),
            )
        })
    }

    /// # Safety
    ///
    /// `address` must identify a currently registered region.
    pub unsafe fn unregister_memory(&self, address: NonNull<u8>) -> Result<()> {
        check("unregisterLocalMemory", unsafe {
            native::unregister_memory(self.native, address.as_ptr().cast::<c_void>())
        })
    }

    pub fn submit_and_wait(
        &self,
        operation: TransferOp,
        remote_segment: &str,
        slices: &[TransferSlice],
        timeout: Duration,
    ) -> Result<usize> {
        self.submit_and_wait_inner(operation, remote_segment, slices, timeout, None)
    }

    pub fn submit_and_notify(
        &self,
        operation: TransferOp,
        remote_segment: &str,
        slices: &[TransferSlice],
        timeout: Duration,
        notification: &Notification,
    ) -> Result<usize> {
        self.submit_and_wait_inner(
            operation,
            remote_segment,
            slices,
            timeout,
            Some(notification),
        )
    }

    fn submit_and_wait_inner(
        &self,
        operation: TransferOp,
        remote_segment: &str,
        slices: &[TransferSlice],
        timeout: Duration,
        notification: Option<&Notification>,
    ) -> Result<usize> {
        if slices.is_empty() {
            return Ok(0);
        }
        let notification = notification
            .map(|notification| {
                Ok::<_, MooncakeError>((
                    CString::new(notification.name.as_str())?,
                    CString::new(notification.message.as_str())?,
                ))
            })
            .transpose()?;
        let segment = self.open_segment(remote_segment)?;
        let batch = unsafe { native::allocate_batch(self.native, slices.len()) };
        if batch == INVALID_BATCH {
            return Err(MooncakeError::InvalidBatch);
        }
        let mut requests = slices
            .iter()
            .map(|slice| native::TransferRequest {
                opcode: operation.code(),
                source: slice.local.as_ptr().cast::<c_void>(),
                target_id: segment,
                target_offset: slice.remote_address,
                length: slice.length as u64,
            })
            .collect::<Vec<_>>();
        let submitted = match notification.as_ref() {
            Some((name, message)) => check("submitTransferWithNotify", unsafe {
                native::submit_with_notify(
                    self.native,
                    batch,
                    &mut requests,
                    native::Notify {
                        name: name.as_ptr().cast_mut(),
                        message: message.as_ptr().cast_mut(),
                    },
                )
            }),
            None => check("submitTransfer", unsafe {
                native::submit(self.native, batch, &mut requests)
            }),
        };
        drain_batch(
            slices.len(),
            timeout,
            submitted,
            |task| {
                let mut status = native::TransferStatus {
                    status: STATUS_WAITING,
                    transferred_bytes: 0,
                };
                check("getTransferStatus", unsafe {
                    native::transfer_status(self.native, batch, task, &mut status)
                })?;
                Ok(status)
            },
            || unsafe { native::free_batch(self.native, batch) == 0 },
        )
    }

    pub fn take_notifications(&self) -> Result<Vec<Notification>> {
        let mut count = 0;
        let messages = unsafe { native::take_notifies(self.native, &mut count) };
        if count < 0 {
            return Err(MooncakeError::InvalidNotificationCount(count));
        }
        if count == 0 {
            return Ok(Vec::new());
        }
        if messages.is_null() {
            return Err(MooncakeError::InvalidNotificationBuffer(count));
        }
        let messages_slice = unsafe { std::slice::from_raw_parts(messages, count as usize) };
        let notifications = messages_slice
            .iter()
            .map(|message| Notification {
                name: unsafe { CStr::from_ptr(message.name) }
                    .to_string_lossy()
                    .into_owned(),
                message: unsafe { CStr::from_ptr(message.message) }
                    .to_string_lossy()
                    .into_owned(),
            })
            .collect();
        let _ = unsafe { native::free_notifies(messages, count) };
        Ok(notifications)
    }

    pub fn send_notification(
        &self,
        remote_segment: &str,
        notification: &Notification,
    ) -> Result<()> {
        let segment = self.open_segment(remote_segment)?;
        let name = CString::new(notification.name.as_str())?;
        let message = CString::new(notification.message.as_str())?;
        check("genNotifyInEngine", unsafe {
            native::notify(
                self.native,
                segment,
                native::Notify {
                    name: name.as_ptr().cast_mut(),
                    message: message.as_ptr().cast_mut(),
                },
            )
        })
    }

    fn open_segment(&self, remote_segment: &str) -> Result<native::SegmentId> {
        let mut segments = self
            .segments
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(segment) = segments.get(remote_segment) {
            return Ok(*segment);
        }
        let name = CString::new(remote_segment)?;
        let segment = unsafe { native::open_segment(self.native, name.as_ptr()) };
        if segment < 0 {
            return Err(MooncakeError::Operation {
                operation: "openSegment",
                status: segment,
            });
        }
        segments.insert(remote_segment.to_string(), segment);
        Ok(segment)
    }

    /// Drop a cached peer segment after a failed transfer or peer restart.
    pub fn invalidate_segment(&self, remote_segment: &str) {
        let segment = self
            .segments
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .remove(remote_segment);
        if let Some(segment) = segment {
            let _ = unsafe { native::close_segment(self.native, segment) };
        }
    }
}

fn set_nic_filter(nics: &[String]) -> Result<()> {
    if nics.is_empty() {
        return Ok(());
    }
    let value = CString::new(nics.join(","))?;
    if unsafe { libc::setenv(c"MC_TE_FILTERS".as_ptr(), value.as_ptr(), 1) } == 0 {
        Ok(())
    } else {
        Err(MooncakeError::Operation {
            operation: "setenv(MC_TE_FILTERS)",
            status: -1,
        })
    }
}

struct ScopedNicFilter {
    previous: Option<OsString>,
    changed: bool,
}

impl ScopedNicFilter {
    fn apply(nics: &[String]) -> Result<Self> {
        let previous = std::env::var_os("MC_TE_FILTERS");
        let changed = !nics.is_empty();
        if changed {
            set_nic_filter(nics)?;
        }
        Ok(Self { previous, changed })
    }
}

impl Drop for ScopedNicFilter {
    fn drop(&mut self) {
        if !self.changed {
            return;
        }
        match self.previous.as_ref() {
            Some(previous) => {
                if let Ok(value) = CString::new(previous.as_bytes()) {
                    unsafe {
                        libc::setenv(c"MC_TE_FILTERS".as_ptr(), value.as_ptr(), 1);
                    }
                }
            }
            None => unsafe {
                libc::unsetenv(c"MC_TE_FILTERS".as_ptr());
            },
        }
    }
}

impl Drop for TransferEngine {
    fn drop(&mut self) {
        if let Ok(segments) = self.segments.get_mut() {
            for segment in segments.values() {
                let _ = unsafe { native::close_segment(self.native, *segment) };
            }
        }
        unsafe { native::destroy(self.native) };
    }
}

/// Native submission can partially succeed. Status errors and TIMEOUT are not
/// fences; the pinned Mooncake implementation frees a batch only when every
/// task's `is_finished` is set. Keep descriptors and caller-owned memory alive
/// until that succeeds, even when the operation will ultimately return an error.
fn drain_batch(
    tasks: usize,
    timeout: Duration,
    submitted: Result<()>,
    mut poll: impl FnMut(usize) -> Result<native::TransferStatus>,
    mut free: impl FnMut() -> bool,
) -> Result<usize> {
    let started = Instant::now();
    let mut completed = vec![false; tasks];
    let mut total = 0usize;
    let mut failure = submitted.err();
    let mut timed_out = false;
    loop {
        for (task, done) in completed.iter_mut().enumerate() {
            if *done {
                continue;
            }
            match poll(task) {
                Ok(status) if status.status == STATUS_COMPLETED => {
                    total = total.saturating_add(status.transferred_bytes as usize);
                    *done = true;
                }
                Ok(status) if matches!(status.status, STATUS_WAITING | STATUS_PENDING) => {}
                Ok(status) => {
                    failure.get_or_insert(MooncakeError::TransferFailed {
                        task,
                        state: status.status,
                    });
                }
                Err(error) => {
                    failure.get_or_insert(error);
                }
            }
        }
        if free() {
            return match failure {
                Some(error) => Err(error),
                None if timed_out => Err(MooncakeError::Timeout),
                None => Ok(total),
            };
        }
        timed_out |= started.elapsed() >= timeout;
        if started.elapsed() < Duration::from_millis(1) {
            std::thread::yield_now();
        } else {
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}

fn check(operation: &'static str, status: i32) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(MooncakeError::Operation { operation, status })
    }
}

#[cfg(test)]
#[path = "../tests/unit/engine.rs"]
mod tests;
