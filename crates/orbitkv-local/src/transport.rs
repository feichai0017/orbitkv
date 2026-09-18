use std::time::{Duration, Instant};

use iceoryx2::prelude::*;
use thiserror::Error;

use crate::{Command, ProtocolError, Response, WireMessage};

type ThreadSafeIpcService = ipc_threadsafe::Service;
type IpcClient =
    iceoryx2::port::client::Client<ThreadSafeIpcService, WireMessage, (), WireMessage, ()>;
type IpcServer =
    iceoryx2::port::server::Server<ThreadSafeIpcService, WireMessage, (), WireMessage, ()>;
type IpcService = iceoryx2::service::port_factory::request_response::PortFactory<
    ThreadSafeIpcService,
    WireMessage,
    (),
    WireMessage,
    (),
>;

#[derive(Clone, Copy, Debug)]
pub struct CallOptions {
    pub timeout: Duration,
    pub spin_iterations: u32,
}

impl Default for CallOptions {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(5),
            spin_iterations: 64,
        }
    }
}

#[derive(Debug, Error)]
pub enum TransportError {
    #[error("invalid service name: {0}")]
    InvalidServiceName(String),
    #[error("failed to create iceoryx2 node: {0}")]
    Node(String),
    #[error("failed to open iceoryx2 service: {0}")]
    Service(String),
    #[error("failed to create iceoryx2 port: {0}")]
    Port(String),
    #[error("failed to start local-control thread: {0}")]
    Thread(String),
    #[error("failed to send local-control message: {0}")]
    Send(String),
    #[error("failed to receive local-control message: {0}")]
    Receive(String),
    #[error("local-control request {request_id} timed out")]
    Timeout { request_id: u64 },
    #[error("response request id mismatch: expected {expected}, got {actual}")]
    RequestMismatch { expected: u64, actual: u64 },
    #[error("response session epoch mismatch: expected {expected}, got {actual}")]
    SessionMismatch { expected: u64, actual: u64 },
    #[error(transparent)]
    Protocol(#[from] ProtocolError),
}

pub struct LocalClient {
    client: IpcClient,
    _service: IpcService,
    _node: iceoryx2::node::Node<ThreadSafeIpcService>,
}

impl LocalClient {
    pub fn connect(service_name: &str) -> Result<Self, TransportError> {
        let node = NodeBuilder::new()
            .create::<ThreadSafeIpcService>()
            .map_err(|error| TransportError::Node(error.to_string()))?;
        let name = service_name.try_into().map_err(
            |error: iceoryx2::service::service_name::ServiceNameError| {
                TransportError::InvalidServiceName(error.to_string())
            },
        )?;
        let service = node
            .service_builder(&name)
            .request_response::<WireMessage, WireMessage>()
            .open()
            .map_err(|error| TransportError::Service(error.to_string()))?;
        let client = service
            .client_builder()
            .create()
            .map_err(|error| TransportError::Port(error.to_string()))?;
        Ok(Self {
            client,
            _service: service,
            _node: node,
        })
    }

    pub fn call(&self, command: Command, options: CallOptions) -> Result<Response, TransportError> {
        let pending = self
            .client
            .send_copy(command.encode())
            .map_err(|error| TransportError::Send(error.to_string()))?;
        let deadline = Instant::now() + options.timeout;
        let mut spins = 0;
        loop {
            if let Some(message) = pending
                .receive()
                .map_err(|error| TransportError::Receive(error.to_string()))?
            {
                let response = Response::decode(*message)?;
                if response.request_id != command.request_id {
                    return Err(TransportError::RequestMismatch {
                        expected: command.request_id,
                        actual: response.request_id,
                    });
                }
                if response.status != crate::StatusCode::StaleSession
                    && response.session_epoch != command.session_epoch
                {
                    return Err(TransportError::SessionMismatch {
                        expected: command.session_epoch,
                        actual: response.session_epoch,
                    });
                }
                return Ok(response);
            }
            if Instant::now() >= deadline {
                return Err(TransportError::Timeout {
                    request_id: command.request_id,
                });
            }
            if spins < options.spin_iterations {
                spins += 1;
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
    }
}

pub struct LocalServer {
    server: IpcServer,
    _service: IpcService,
    _node: iceoryx2::node::Node<ThreadSafeIpcService>,
}

impl LocalServer {
    pub fn bind(service_name: &str) -> Result<Self, TransportError> {
        let node = NodeBuilder::new()
            .create::<ThreadSafeIpcService>()
            .map_err(|error| TransportError::Node(error.to_string()))?;
        let name = service_name.try_into().map_err(
            |error: iceoryx2::service::service_name::ServiceNameError| {
                TransportError::InvalidServiceName(error.to_string())
            },
        )?;
        let service = node
            .service_builder(&name)
            .request_response::<WireMessage, WireMessage>()
            .create()
            .map_err(|error| TransportError::Service(error.to_string()))?;
        let server = service
            .server_builder()
            .create()
            .map_err(|error| TransportError::Port(error.to_string()))?;
        Ok(Self {
            server,
            _service: service,
            _node: node,
        })
    }

    pub fn try_serve(
        &self,
        handler: impl FnOnce(Command) -> Response,
    ) -> Result<bool, TransportError> {
        let Some(request) = self
            .server
            .receive()
            .map_err(|error| TransportError::Receive(error.to_string()))?
        else {
            return Ok(false);
        };
        let command = Command::decode(*request)?;
        request
            .send_copy(handler(command).encode())
            .map_err(|error| TransportError::Send(error.to_string()))?;
        Ok(true)
    }

    /// Reject commands created for an older sidecar session before dispatch.
    pub fn try_serve_for_epoch(
        &self,
        session_epoch: u64,
        handler: impl FnOnce(Command) -> Response,
    ) -> Result<bool, TransportError> {
        self.try_serve(|command| {
            if command.session_epoch == session_epoch {
                handler(command)
            } else {
                Response {
                    status: crate::StatusCode::StaleSession,
                    request_id: command.request_id,
                    session_epoch,
                    descriptor: command.descriptor,
                    value0: 0,
                    value1: 0,
                }
            }
        })
    }
}
