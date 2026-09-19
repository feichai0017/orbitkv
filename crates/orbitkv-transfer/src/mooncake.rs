//! Thin Rust boundary over the pinned upstream Mooncake Transfer Engine C ABI.
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

use orbitkv_mooncake_provider as native;
use thiserror::Error;

const INVALID_BATCH: u64 = u64::MAX;
const STATUS_WAITING: i32 = 0;
const STATUS_PENDING: i32 = 1;
const STATUS_COMPLETED: i32 = 4;
pub const P2P_METADATA: &str = "P2PHANDSHAKE";
pub const AUTO_MEMORY_LOCATION: &str = "*";
static ENGINE_CREATE_LOCK: Mutex<()> = Mutex::new(());

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransferOp {
    Read,
    Write,
}

impl TransferOp {
    fn code(self) -> i32 {
        match self {
            Self::Read => 0,
            Self::Write => 1,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TransferSlice {
    pub local: NonNull<u8>,
    pub remote_address: u64,
    pub length: usize,
}

unsafe impl Send for TransferSlice {}
unsafe impl Sync for TransferSlice {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Notification {
    pub name: String,
    pub message: String,
}

#[derive(Debug, Error)]
pub enum MooncakeError {
    #[error("Mooncake native runtime unavailable: {0}")]
    NativeRuntime(String),
    #[error("invalid string: {0}")]
    InvalidString(#[from] std::ffi::NulError),
    #[error("Mooncake Transfer Engine creation failed")]
    Create,
    #[error("Mooncake operation {operation} failed with status {status}")]
    Operation {
        operation: &'static str,
        status: i32,
    },
    #[error("Mooncake returned an invalid batch id")]
    InvalidBatch,
    #[error("Mooncake transfer task {task} failed with state {state}")]
    TransferFailed { task: usize, state: i32 },
    #[error("Mooncake transfer batch timed out")]
    Timeout,
    #[error("Mooncake notification buffer has invalid size {0}")]
    InvalidNotificationCount(i32),
    #[error("Mooncake returned a null notification buffer for {0} messages")]
    InvalidNotificationBuffer(i32),
}

pub type Result<T> = std::result::Result<T, MooncakeError>;

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
        let result = submitted.and_then(|()| self.wait_batch(batch, slices.len(), timeout));
        let freed = check("freeBatchID", unsafe {
            native::free_batch(self.native, batch)
        });
        match (result, freed) {
            (Err(error), _) => Err(error),
            (Ok(_), Err(error)) => Err(error),
            (Ok(bytes), Ok(())) => Ok(bytes),
        }
    }

    fn wait_batch(&self, batch: native::BatchId, tasks: usize, timeout: Duration) -> Result<usize> {
        let deadline = Instant::now() + timeout;
        let mut total = 0usize;
        let mut first_failure = None;
        let mut timed_out = false;
        for task in 0..tasks {
            loop {
                let mut status = native::TransferStatus {
                    status: STATUS_WAITING,
                    transferred_bytes: 0,
                };
                check("getTransferStatus", unsafe {
                    native::transfer_status(self.native, batch, task, &mut status)
                })?;
                match status.status {
                    STATUS_COMPLETED => {
                        total = total.saturating_add(status.transferred_bytes as usize);
                        break;
                    }
                    STATUS_WAITING | STATUS_PENDING => {
                        if Instant::now() >= deadline {
                            timed_out = true;
                        }
                        std::thread::yield_now();
                    }
                    _ => {
                        first_failure.get_or_insert(MooncakeError::TransferFailed {
                            task,
                            state: status.status,
                        });
                        break;
                    }
                }
            }
        }
        if let Some(error) = first_failure {
            Err(error)
        } else if timed_out {
            Err(MooncakeError::Timeout)
        } else {
            Ok(total)
        }
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

fn check(operation: &'static str, status: i32) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(MooncakeError::Operation { operation, status })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tcp_loopback_moves_bytes_through_upstream_mooncake() {
        unsafe {
            libc::setenv(c"MC_FORCE_TCP".as_ptr(), c"1".as_ptr(), 1);
        }
        let engine = TransferEngine::new("P2PHANDSHAKE", "127.0.0.1:0", "127.0.0.1", 0, &[])
            .expect("create Mooncake Transfer Engine");
        let segment = engine.local_segment_name().expect("local segment");
        let mut memory = vec![0u8; 8192];
        memory[..4096].fill(0xa5);
        let base = NonNull::new(memory.as_mut_ptr()).expect("memory pointer");
        unsafe {
            engine
                .register_memory(base, memory.len(), "cpu:0")
                .expect("register memory");
        }
        let destination = unsafe { base.byte_add(4096) };
        engine
            .submit_and_notify(
                TransferOp::Write,
                &segment,
                &[TransferSlice {
                    local: base,
                    remote_address: destination.as_ptr() as u64,
                    length: 4096,
                }],
                Duration::from_secs(5),
                &Notification {
                    name: "loopback".to_string(),
                    message: "done".to_string(),
                },
            )
            .expect("loopback write");
        assert_eq!(&memory[..4096], &memory[4096..]);
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let notifications = engine.take_notifications().expect("take notifications");
            if notifications.iter().any(|notification| {
                notification.name == "loopback" && notification.message == "done"
            }) {
                break;
            }
            assert!(Instant::now() < deadline, "notification timed out");
            std::thread::yield_now();
        }
        unsafe {
            engine.unregister_memory(base).expect("unregister memory");
            libc::unsetenv(c"MC_FORCE_TCP".as_ptr());
        }
    }

    #[test]
    fn engine_creation_restores_the_process_nic_filter() {
        let _guard = ENGINE_CREATE_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let original = std::env::var_os("MC_TE_FILTERS");
        unsafe {
            libc::setenv(c"MC_TE_FILTERS".as_ptr(), c"original".as_ptr(), 1);
        }
        let filter =
            ScopedNicFilter::apply(&["temporary".to_string()]).expect("apply temporary NIC filter");
        assert_eq!(
            std::env::var_os("MC_TE_FILTERS").as_deref(),
            Some(std::ffi::OsStr::new("temporary"))
        );
        drop(filter);
        assert_eq!(
            std::env::var_os("MC_TE_FILTERS").as_deref(),
            Some(std::ffi::OsStr::new("original"))
        );
        unsafe {
            match original {
                Some(value) => {
                    let value = CString::new(value.as_bytes()).expect("original environment");
                    libc::setenv(c"MC_TE_FILTERS".as_ptr(), value.as_ptr(), 1);
                }
                None => {
                    libc::unsetenv(c"MC_TE_FILTERS".as_ptr());
                }
            }
        }
    }
}
