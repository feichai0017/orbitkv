use std::os::fd::{AsRawFd, RawFd};
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use thiserror::Error;

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
}

pub struct LocalQueryClient {
    bootstrap: BootstrapClient,
    client: LocalClient,
    options: CallOptions,
    call_lock: Mutex<()>,
    poisoned: AtomicBool,
}

impl LocalQueryClient {
    pub fn connect(
        bootstrap_socket: impl AsRef<Path>,
        options: CallOptions,
    ) -> Result<Self, LocalQueryError> {
        let bootstrap = BootstrapClient::connect(bootstrap_socket)?;
        let client = LocalClient::connect(&bootstrap.info().service_name)?;
        Ok(Self {
            bootstrap,
            client,
            options,
            call_lock: Mutex::new(()),
            poisoned: AtomicBool::new(false),
        })
    }

    pub fn service_name(&self) -> &str {
        &self.bootstrap.info_ref().service_name
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
                self.poisoned.store(true, Ordering::Release);
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
        let payload = request.encode()?;
        let _ = self.call_descriptor(CommandCode::Publish, request_id, &payload)?;
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
        let _call = self
            .call_lock
            .lock()
            .map_err(|_| BootstrapError::Arena(crate::ArenaError::Poisoned))?;
        if self.poisoned.load(Ordering::Acquire) {
            return Err(LocalQueryError::SessionRequiresReconnect);
        }
        let descriptor = self.bootstrap.write_request(payload)?;
        let info = self.bootstrap.info();
        let response = match self.client.call(
            Command {
                code,
                request_id,
                session_epoch: info.session_epoch,
                descriptor,
                arg0: info.client_token,
                arg1: 0,
            },
            self.options,
        ) {
            Ok(response) => response,
            Err(error) => {
                self.poisoned.store(true, Ordering::Release);
                return Err(error.into());
            }
        };
        if response.status != StatusCode::Ok {
            if response.value1 & RESPONSE_FLAG_REQUEST_CONSUMED != 0 {
                self.bootstrap.complete_request(descriptor)?;
            } else {
                self.poisoned.store(true, Ordering::Release);
            }
            return Err(LocalQueryError::Status(response.status));
        }
        if response.value1 & RESPONSE_FLAG_REQUEST_CONSUMED == 0 {
            self.poisoned.store(true, Ordering::Release);
            return Err(LocalQueryError::SessionRequiresReconnect);
        }
        let payload = self
            .bootstrap
            .read_response(descriptor, response.descriptor)?;
        Ok(payload)
    }
}
