use std::io::{Read, Write};
use std::os::fd::{AsRawFd, OwnedFd, RawFd};
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use rustix::net::sockopt::socket_peercred;
use rustix::process::{PidfdFlags, pidfd_open};
use thiserror::Error;

use crate::lifecycle::{LIFECYCLE_HEADER_BYTES, LifecycleCommand, LifecycleHeader};
use crate::{
    BootstrapClient, BootstrapError, CallOptions, Command, CommandCode, LocalClient,
    PublishRequest, QueryBundleRequest, QueryBundleResponse, QueryCodecError,
    RESPONSE_FLAG_REQUEST_CONSUMED, ReleaseRequest, RestoreCommand, RestoreRequest,
    RestoreResponse, RestoreState, StatusCode, TransportError,
};

const RESTORE_NOTIFICATION_POLL_INTERVAL: std::time::Duration =
    std::time::Duration::from_millis(50);

#[derive(Debug, Error)]
pub enum LocalQueryError {
    #[error(transparent)]
    Bootstrap(#[from] BootstrapError),
    #[error(transparent)]
    Transport(#[from] TransportError),
    #[error(transparent)]
    Codec(#[from] QueryCodecError),
    #[error("local query returned {0:?}")]
    Status(StatusCode),
    #[error("local query session is ambiguous after a failed call; reconnect required")]
    SessionRequiresReconnect,
    #[error("restore operation {operation_id} timed out")]
    RestoreTimeout { operation_id: u64 },
    #[error("restore operation {operation_id} failed: {message}")]
    RestoreFailed { operation_id: u64, message: String },
    #[error("cannot observe Cache Manager process death; refusing to start a publish")]
    PublishPeerUnobservable,
    #[error("local lifecycle rejected operation ({code}): {message}")]
    Lifecycle { code: u16, message: String },
}

pub struct LocalQueryClient {
    bootstrap: BootstrapClient,
    client: LocalClient,
    options: CallOptions,
    call_lock: Mutex<()>,
    lifecycle_lock: Mutex<()>,
    poisoned: AtomicBool,
    publish_peer: Option<OwnedFd>,
}

impl LocalQueryClient {
    pub fn connect(
        bootstrap_socket: impl AsRef<Path>,
        options: CallOptions,
    ) -> Result<Self, LocalQueryError> {
        let bootstrap = BootstrapClient::connect(bootstrap_socket)?;
        let publish_peer = socket_peercred(bootstrap.stream())
            .ok()
            .and_then(|credentials| pidfd_open(credentials.pid, PidfdFlags::empty()).ok());
        bootstrap
            .stream()
            .set_read_timeout(Some(options.timeout))
            .map_err(BootstrapError::Io)?;
        bootstrap
            .stream()
            .set_write_timeout(Some(options.timeout))
            .map_err(BootstrapError::Io)?;
        let client = LocalClient::connect(&bootstrap.info().service_name)?;
        Ok(Self {
            bootstrap,
            client,
            options,
            call_lock: Mutex::new(()),
            lifecycle_lock: Mutex::new(()),
            poisoned: AtomicBool::new(false),
            publish_peer,
        })
    }

    pub fn service_name(&self) -> &str {
        &self.bootstrap.info_ref().service_name
    }

    /// Registration and liveness metadata use UDS, independently of the hot descriptor slot.
    pub fn lifecycle(
        &self,
        command: LifecycleCommand,
        payload: &[u8],
    ) -> Result<(), LocalQueryError> {
        self.lifecycle_bytes(command, payload).map(|_| ())
    }

    /// Exchange one bounded lifecycle frame; page operations return binary data.
    pub fn lifecycle_bytes(
        &self,
        command: LifecycleCommand,
        payload: &[u8],
    ) -> Result<Vec<u8>, LocalQueryError> {
        let _guard = self
            .lifecycle_lock
            .lock()
            .map_err(|_| LocalQueryError::SessionRequiresReconnect)?;
        if self.poisoned.load(Ordering::Acquire) {
            return Err(LocalQueryError::SessionRequiresReconnect);
        }
        let exchange = || -> std::io::Result<(u16, Vec<u8>)> {
            let header = LifecycleHeader {
                code: command as u16,
                epoch: self.session_epoch(),
                payload_len: payload.len(),
            }
            .encode()?;
            let mut stream = self.bootstrap.stream();
            let timeout = match command {
                LifecycleCommand::Register
                | LifecycleCommand::Unregister
                | LifecycleCommand::PagePut => self.options.timeout.max(Duration::from_secs(120)),
                LifecycleCommand::PageGet | LifecycleCommand::PageExists => {
                    self.options.timeout.max(Duration::from_secs(45))
                }
                _ => self.options.timeout,
            };
            stream.set_read_timeout(Some(timeout))?;
            stream.write_all(&header)?;
            stream.write_all(payload)?;
            let mut header = [0; LIFECYCLE_HEADER_BYTES];
            stream.read_exact(&mut header)?;
            let header = LifecycleHeader::decode(header)?;
            if header.epoch != self.session_epoch() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "stale lifecycle session",
                ));
            }
            let mut body = vec![0; header.payload_len];
            stream.read_exact(&mut body)?;
            Ok((header.code, body))
        };
        match exchange() {
            Ok((0, body)) => Ok(body),
            Ok((code, message)) => Err(LocalQueryError::Lifecycle {
                code,
                message: String::from_utf8_lossy(&message).into_owned(),
            }),
            Err(error) => {
                self.close();
                // A partial frame must never be reused as a new request.
                Err(BootstrapError::Io(error).into())
            }
        }
    }

    /// End the local session explicitly, without waiting for client destruction.
    pub fn close(&self) {
        self.poisoned.store(true, Ordering::Release);
        let _ = self.bootstrap.stream().shutdown(std::net::Shutdown::Both);
    }

    pub fn session_epoch(&self) -> u64 {
        self.bootstrap.info_ref().session_epoch
    }

    /// Borrowed notification descriptor. It remains valid only while this
    /// client is alive and must not be closed by the caller.
    pub fn notification_fd(&self) -> RawFd {
        self.bootstrap.notification_fd().as_raw_fd()
    }

    pub fn query_bundle(
        &self,
        request_id: u64,
        request: &QueryBundleRequest,
    ) -> Result<QueryBundleResponse, LocalQueryError> {
        let payload = request.encode()?;
        let payload = self.call_descriptor(CommandCode::QueryBundle, request_id, &payload)?;
        match QueryBundleResponse::decode(&payload) {
            Ok(response) => Ok(response),
            Err(error) => {
                self.close();
                Err(error.into())
            }
        }
    }

    pub fn release(&self, request_id: u64, lease: Vec<u8>) -> Result<(), LocalQueryError> {
        let payload = ReleaseRequest { lease }.encode()?;
        let _ = self.call_descriptor(CommandCode::Release, request_id, &payload)?;
        Ok(())
    }

    pub fn publish(
        &self,
        request_id: u64,
        request: &PublishRequest,
    ) -> Result<(), LocalQueryError> {
        let peer = self
            .publish_peer
            .as_ref()
            .ok_or(LocalQueryError::PublishPeerUnobservable)?;
        let payload = request.encode()?;
        let _ = self.call_descriptor_with_peer(CommandCode::Publish, request_id, &payload, peer)?;
        Ok(())
    }

    pub fn restore_submit(
        &self,
        request_id: u64,
        request: &RestoreRequest,
    ) -> Result<u64, LocalQueryError> {
        let payload = RestoreCommand::Submit(request.clone()).encode()?;
        let payload = self.call_descriptor(CommandCode::Restore, request_id, &payload)?;
        let response = RestoreResponse::decode(&payload)?;
        match response.state {
            RestoreState::Pending => Ok(response.operation_id),
            RestoreState::Succeeded => Ok(response.operation_id),
            RestoreState::Failed => Err(LocalQueryError::RestoreFailed {
                operation_id: response.operation_id,
                message: response.message,
            }),
        }
    }

    pub fn restore_poll(
        &self,
        request_id: u64,
        operation_id: u64,
    ) -> Result<RestoreResponse, LocalQueryError> {
        let payload = RestoreCommand::Poll { operation_id }.encode()?;
        let payload = self.call_descriptor(CommandCode::Restore, request_id, &payload)?;
        Ok(RestoreResponse::decode(&payload)?)
    }

    pub fn restore_wait(
        &self,
        request_id: u64,
        operation_id: u64,
        timeout: std::time::Duration,
    ) -> Result<(), LocalQueryError> {
        let deadline = std::time::Instant::now() + timeout;
        let mut poll_request_id = request_id;
        loop {
            let response = self.restore_poll(poll_request_id, operation_id)?;
            match response.state {
                RestoreState::Succeeded => return Ok(()),
                RestoreState::Failed => {
                    return Err(LocalQueryError::RestoreFailed {
                        operation_id,
                        message: response.message,
                    });
                }
                RestoreState::Pending => {}
            }
            poll_request_id = poll_request_id
                .checked_add(1)
                .ok_or(LocalQueryError::SessionRequiresReconnect)?;
            let now = std::time::Instant::now();
            if now >= deadline {
                return Err(LocalQueryError::RestoreTimeout { operation_id });
            }
            let wait = deadline
                .saturating_duration_since(now)
                .min(RESTORE_NOTIFICATION_POLL_INTERVAL);
            let _ = self.bootstrap.wait_for_notification(wait)?;
        }
    }

    fn call_descriptor(
        &self,
        code: CommandCode,
        request_id: u64,
        payload: &[u8],
    ) -> Result<Vec<u8>, LocalQueryError> {
        self.call_descriptor_inner(code, request_id, payload, None)
    }

    fn call_descriptor_with_peer(
        &self,
        code: CommandCode,
        request_id: u64,
        payload: &[u8],
        peer: &OwnedFd,
    ) -> Result<Vec<u8>, LocalQueryError> {
        self.call_descriptor_inner(code, request_id, payload, Some(peer))
    }

    fn call_descriptor_inner(
        &self,
        code: CommandCode,
        request_id: u64,
        payload: &[u8],
        peer: Option<&OwnedFd>,
    ) -> Result<Vec<u8>, LocalQueryError> {
        let _call = self
            .call_lock
            .lock()
            .map_err(|_| BootstrapError::Arena(crate::ArenaError::Poisoned))?;
        if self.poisoned.load(Ordering::Acquire) {
            return Err(LocalQueryError::SessionRequiresReconnect);
        }
        let descriptor = self.bootstrap.write_request(payload)?;
        let info = self.bootstrap.info();
        let command = Command {
            code,
            request_id,
            session_epoch: info.session_epoch,
            descriptor,
            arg0: info.client_token,
            arg1: 0,
        };
        let call = match peer {
            Some(peer) => self
                .client
                .call_until_peer_exit(command, self.options, peer),
            None => self.client.call(command, self.options),
        };
        let response = match call {
            Ok(response) => response,
            Err(error) => {
                self.close();
                if let Some(peer) = peer
                    && !matches!(
                        error,
                        TransportError::PeerExited { .. } | TransportError::Send(_)
                    )
                {
                    LocalClient::wait_for_peer_exit(peer);
                }
                return Err(error.into());
            }
        };
        if response.status != StatusCode::Ok {
            if response.value1 & RESPONSE_FLAG_REQUEST_CONSUMED != 0 {
                self.bootstrap
                    .complete_request(descriptor)
                    .inspect_err(|_| self.close())?;
            } else {
                self.close();
            }
            return Err(LocalQueryError::Status(response.status));
        }
        if response.value1 & RESPONSE_FLAG_REQUEST_CONSUMED == 0 {
            self.close();
            return Err(LocalQueryError::SessionRequiresReconnect);
        }
        let payload = self
            .bootstrap
            .read_response(descriptor, response.descriptor)
            .inspect_err(|_| self.close())?;
        Ok(payload)
    }
}
