use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use thiserror::Error;

use crate::{
    BootstrapClient, BootstrapError, CallOptions, Command, CommandCode, LocalClient,
    PublishRequest, QueryBundleRequest, QueryBundleResponse, QueryCodecError,
    RESPONSE_FLAG_REQUEST_CONSUMED, ReleaseRequest, StatusCode, TransportError,
};

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
