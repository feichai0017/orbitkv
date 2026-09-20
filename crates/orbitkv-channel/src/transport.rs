use std::os::fd::OwnedFd;
use std::time::{Duration, Instant};

use iceoryx2::active_request::ActiveRequest;
use iceoryx2::prelude::*;
use rustix::event::{PollFd, PollFlags, Timespec, poll};
use thiserror::Error;

use crate::{Command, ProtocolError, Response, WireMessage};

type ThreadSafeIpcService = ipc_threadsafe::Service;
type IpcClient =
    iceoryx2::port::client::Client<ThreadSafeIpcService, WireMessage, (), WireMessage, ()>;
type IpcServer =
    iceoryx2::port::server::Server<ThreadSafeIpcService, WireMessage, (), WireMessage, ()>;
type IpcActiveRequest = ActiveRequest<ThreadSafeIpcService, WireMessage, (), WireMessage, ()>;
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
    #[error("Cache Manager process exited before request {request_id} completed")]
    PeerExited { request_id: u64 },
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
        self.call_inner(command, options, None)
    }

    /// A publish cannot time out while the peer still owns its source pages.
    /// The pidfd distinguishes a dead Cache Manager from an IPC stall, so an
    /// ambiguous stall keeps the caller blocked and its pages pinned.
    pub fn call_until_peer_exit(
        &self,
        command: Command,
        options: CallOptions,
        peer: &OwnedFd,
    ) -> Result<Response, TransportError> {
        self.call_inner(command, options, Some(peer))
    }

    fn call_inner(
        &self,
        command: Command,
        options: CallOptions,
        peer: Option<&OwnedFd>,
    ) -> Result<Response, TransportError> {
        let pending = self
            .client
            .send_copy(command.encode())
            .map_err(|error| TransportError::Send(error.to_string()))?;
        let deadline = peer.is_none().then(|| Instant::now() + options.timeout);
        let mut next_peer_check = Instant::now();
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
            if let Some(peer) = peer
                && Instant::now() >= next_peer_check
            {
                if peer_exited(peer)? {
                    return Err(TransportError::PeerExited {
                        request_id: command.request_id,
                    });
                }
                next_peer_check = Instant::now() + Duration::from_millis(10);
            }
            if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                return Err(TransportError::Timeout {
                    request_id: command.request_id,
                });
            }
            if spins < options.spin_iterations {
                spins += 1;
                std::hint::spin_loop();
            } else if peer.is_some() {
                std::thread::sleep(Duration::from_micros(100));
            } else {
                std::thread::yield_now();
            }
        }
    }

    /// A failed receive after submission cannot prove that DMA stopped.
    /// Keep the source pages pinned until the peer process itself is gone.
    pub fn wait_for_peer_exit(peer: &OwnedFd) {
        loop {
            if matches!(peer_exited(peer), Ok(true)) {
                return;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

fn peer_exited(peer: &OwnedFd) -> Result<bool, TransportError> {
    let mut fds = [PollFd::new(peer, PollFlags::IN)];
    let timeout = Timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    poll(&mut fds, Some(&timeout)).map_err(|error| TransportError::Receive(error.to_string()))?;
    Ok(fds[0].revents().contains(PollFlags::IN))
}

pub struct LocalServer {
    server: IpcServer,
    _service: IpcService,
    _node: iceoryx2::node::Node<ThreadSafeIpcService>,
}

/// Owns one iceoryx2 request until its response is sent. The Cache Manager can move
/// this handle to a worker so a long GPU copy does not stop the dispatcher.
pub struct DeferredResponse(IpcActiveRequest);

impl DeferredResponse {
    pub fn send(self, response: Response) -> Result<(), TransportError> {
        self.0
            .send_copy(response.encode())
            .map_err(|error| TransportError::Send(error.to_string()))
    }
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

    /// Reject commands created for an older Cache Manager session before dispatch.
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

    /// The handler may retain the reply handle and send after this call returns.
    /// A stale epoch is always answered before it reaches the handler.
    pub fn try_serve_deferred_for_epoch(
        &self,
        session_epoch: u64,
        handler: impl FnOnce(Command, DeferredResponse) -> Result<(), TransportError>,
    ) -> Result<bool, TransportError> {
        let Some(request) = self
            .server
            .receive()
            .map_err(|error| TransportError::Receive(error.to_string()))?
        else {
            return Ok(false);
        };
        let command = Command::decode(*request)?;
        let reply = DeferredResponse(request);
        if command.session_epoch != session_epoch {
            reply.send(Response {
                status: crate::StatusCode::StaleSession,
                request_id: command.request_id,
                session_epoch,
                descriptor: command.descriptor,
                value0: 0,
                value1: 0,
            })?;
        } else {
            handler(command, reply)?;
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::thread;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    #[test]
    fn deferred_reply_allows_an_unrelated_request_to_finish_first() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let name = format!("orbitkv/test/{}/{nonce}", std::process::id());
        let server = LocalServer::bind(&name).unwrap();
        let first = LocalClient::connect(&name).unwrap();
        let second = LocalClient::connect(&name).unwrap();
        let options = CallOptions {
            timeout: Duration::from_secs(5),
            spin_iterations: 0,
        };
        let publish_options = CallOptions {
            timeout: Duration::from_millis(10),
            spin_iterations: 0,
        };
        let peer = rustix::process::pidfd_open(
            rustix::process::getpid(),
            rustix::process::PidfdFlags::empty(),
        )
        .unwrap();
        let (pending_tx, pending_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let server_thread = thread::spawn(move || {
            let mut pending = None;
            let mut served_second = false;
            let deadline = Instant::now() + Duration::from_secs(5);
            while !served_second && Instant::now() < deadline {
                server
                    .try_serve_deferred_for_epoch(7, |command, reply| {
                        if command.request_id == 1 {
                            pending = Some(reply);
                            pending_tx.send(()).unwrap();
                            Ok(())
                        } else {
                            served_second = true;
                            reply.send(Response::ok(command))
                        }
                    })
                    .unwrap();
                thread::yield_now();
            }
            assert!(served_second);
            release_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            let first_command = Command::ping(1, 7);
            pending.unwrap().send(Response::ok(first_command)).unwrap();
        });
        let first_thread = thread::spawn(move || {
            first.call_until_peer_exit(Command::ping(1, 7), publish_options, &peer)
        });
        pending_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        let response = second.call(Command::ping(2, 7), options).unwrap();
        assert_eq!(response.request_id, 2);
        thread::sleep(Duration::from_millis(30));
        release_tx.send(()).unwrap();
        assert_eq!(first_thread.join().unwrap().unwrap().request_id, 1);
        server_thread.join().unwrap();
    }
}
