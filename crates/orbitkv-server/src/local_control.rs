use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use log::{error, info};
use orbitkv_common::hll::MultiWindowHllTracker;
use orbitkv_core::{EngineError, OrbitKVEngine};
use orbitkv_local::{
    ArenaError, BootstrapError, BootstrapServer, BootstrapSession, Command, CommandCode,
    LocalServer, QueryBundleRequest, QueryBundleResponse, QueryOutcomeCode,
    RESPONSE_FLAG_REQUEST_CONSUMED, Response, StatusCode, TransportError,
};
use thiserror::Error;
use tokio::runtime::Handle;
use tokio::sync::Notify;

use crate::query::{QueryInput, QueryOutcome, execute_query};

const IDLE_POLL_INTERVAL: Duration = Duration::from_micros(50);
const BOOTSTRAP_POLL_INTERVAL: Duration = Duration::from_millis(10);
const LIVENESS_POLL_INTERVAL: Duration = Duration::from_millis(250);

#[derive(Debug, Error)]
pub(crate) enum LocalControlError {
    #[error(transparent)]
    Transport(#[from] TransportError),
    #[error(transparent)]
    Bootstrap(#[from] BootstrapError),
}

/// Dedicated iceoryx2 endpoint owned by one sidecar process.
///
/// QueryBundle uses a generation-checked memfd arena bootstrapped over UDS.
/// Restore, Publish, and Release remain explicit `Invalid` responses.
pub(crate) struct LocalControlEndpoint {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl LocalControlEndpoint {
    #[allow(
        clippy::too_many_arguments,
        reason = "endpoint construction names each transport-owned resource"
    )]
    pub(crate) fn start(
        service_name: String,
        session_epoch: u64,
        bootstrap_socket: PathBuf,
        arena_size: usize,
        slot_size: usize,
        engine: Arc<OrbitKVEngine>,
        runtime: Handle,
        hll_tracker: Arc<std::sync::Mutex<MultiWindowHllTracker>>,
        shutdown: Arc<Notify>,
    ) -> Result<Self, LocalControlError> {
        let server = LocalServer::bind(&service_name)?;
        let bootstrap = BootstrapServer::bind(
            &bootstrap_socket,
            &service_name,
            session_epoch,
            arena_size,
            slot_size,
        )?;
        bootstrap.set_nonblocking(true)?;

        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = Arc::clone(&stop);
        let thread_service = service_name.clone();
        let thread = thread::Builder::new()
            .name("orbitkv-local-control".to_string())
            .spawn(move || {
                info!(
                    "Local control endpoint ready: service={} session_epoch={} bootstrap={}",
                    thread_service,
                    session_epoch,
                    bootstrap_socket.display()
                );
                let mut sessions = HashMap::new();
                let mut next_bootstrap_poll = Instant::now();
                let mut next_liveness_poll = Instant::now();
                while !thread_stop.load(Ordering::Acquire) {
                    let now = Instant::now();
                    if now >= next_bootstrap_poll {
                        accept_pending_sessions(&bootstrap, &mut sessions);
                        next_bootstrap_poll = now + BOOTSTRAP_POLL_INTERVAL;
                    }
                    if now >= next_liveness_poll {
                        sessions.retain(|_, session| match session.is_alive() {
                            Ok(alive) => alive,
                            Err(error) => {
                                error!("Local bootstrap liveness check failed: {error}");
                                false
                            }
                        });
                        next_liveness_poll = now + LIVENESS_POLL_INTERVAL;
                    }

                    let mut request_shutdown = false;
                    match server.try_serve_for_epoch(session_epoch, |command| {
                        dispatch(
                            command,
                            &bootstrap,
                            &mut sessions,
                            &engine,
                            &runtime,
                            &hll_tracker,
                            &mut request_shutdown,
                        )
                    }) {
                        Ok(true) if request_shutdown => {
                            shutdown.notify_waiters();
                            break;
                        }
                        Ok(true) => {}
                        Ok(false) => thread::sleep(IDLE_POLL_INTERVAL),
                        Err(error) => {
                            error!("Local control request failed: {error}");
                            thread::sleep(IDLE_POLL_INTERVAL);
                        }
                    }
                }
                info!("Local control endpoint stopped: service={thread_service}");
            })
            .map_err(|error| TransportError::Thread(error.to_string()))?;

        Ok(Self {
            stop,
            thread: Some(thread),
        })
    }

    pub(crate) fn stop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take()
            && thread.join().is_err()
        {
            error!("Local control thread panicked during shutdown");
        }
    }
}

impl Drop for LocalControlEndpoint {
    fn drop(&mut self) {
        self.stop();
    }
}

fn accept_pending_sessions(
    bootstrap: &BootstrapServer,
    sessions: &mut HashMap<u64, BootstrapSession>,
) {
    loop {
        match bootstrap.try_accept() {
            Ok(Some(session)) => {
                info!(
                    "Local client bootstrapped: pid={} uid={} slot={}",
                    session.credentials().pid,
                    session.credentials().uid,
                    session.slot_index()
                );
                sessions.insert(session.client_token(), session);
            }
            Ok(None) => break,
            Err(error) => {
                error!("Local bootstrap accept failed: {error}");
                break;
            }
        }
    }
}

fn dispatch(
    command: Command,
    bootstrap: &BootstrapServer,
    sessions: &mut HashMap<u64, BootstrapSession>,
    engine: &Arc<OrbitKVEngine>,
    runtime: &Handle,
    hll_tracker: &Arc<std::sync::Mutex<MultiWindowHllTracker>>,
    request_shutdown: &mut bool,
) -> Response {
    let mut response = Response::ok(command);
    response.value1 = 0;
    match command.code {
        CommandCode::Ping => {
            response.value0 = command.arg0.wrapping_add(1);
        }
        CommandCode::Shutdown => {
            *request_shutdown = true;
        }
        CommandCode::QueryBundle => {
            response = dispatch_query(command, bootstrap, sessions, engine, runtime, hll_tracker);
        }
        CommandCode::Restore | CommandCode::Publish | CommandCode::Release => {
            response.status = StatusCode::Invalid;
        }
    }
    response
}

fn dispatch_query(
    command: Command,
    bootstrap: &BootstrapServer,
    sessions: &mut HashMap<u64, BootstrapSession>,
    engine: &Arc<OrbitKVEngine>,
    runtime: &Handle,
    hll_tracker: &Arc<std::sync::Mutex<MultiWindowHllTracker>>,
) -> Response {
    let mut response = Response::ok(command);
    response.value1 = 0;
    let descriptor_slot = match bootstrap.descriptor_slot(command.descriptor.offset) {
        Ok(slot) => slot,
        Err(error) => return error_response(response, local_error_status(&error), &error),
    };
    let session = match sessions.get_mut(&command.arg0) {
        Some(session) => session,
        None => {
            response.status = StatusCode::StaleSession;
            return response;
        }
    };
    if let Err(error) = session.validate_request(command.descriptor, command.arg0, descriptor_slot)
    {
        return error_response(response, bootstrap_error_status(&error), &error);
    }
    let payload = match bootstrap.arena().read(command.descriptor) {
        Ok(payload) => payload,
        Err(error) => return error_response(response, local_error_status(&error), &error),
    };
    if let Err(error) = session.complete_request() {
        return error_response(response, bootstrap_error_status(&error), &error);
    }
    response.value1 |= RESPONSE_FLAG_REQUEST_CONSUMED;
    let request = match QueryBundleRequest::decode(&payload) {
        Ok(request) => request,
        Err(error) => return error_response(response, StatusCode::Invalid, &error),
    };
    let outcome = match runtime.block_on(execute_query(
        engine,
        hll_tracker,
        QueryInput {
            instance_id: request.instance_id,
            block_hashes: request.block_hashes,
            request_id: request.request_id,
            wait_for_full_prefix: request.wait_for_full_prefix,
            group_id: request.group_id,
        },
    )) {
        Ok(outcome) => outcome,
        Err(error) => return error_response(response, engine_error_status(&error), &error),
    };
    let payload = match outcome {
        QueryOutcome::Loading => QueryBundleResponse::loading(),
        QueryOutcome::Ready {
            num_hit_blocks,
            lease,
            hit_positions,
        } => QueryBundleResponse {
            outcome: QueryOutcomeCode::Ready,
            num_hit_blocks,
            lease,
            hit_positions,
        },
    };
    let payload = match payload.encode() {
        Ok(payload) => payload,
        Err(error) => return error_response(response, StatusCode::Internal, &error),
    };
    match bootstrap
        .arena()
        .write_response(command.descriptor, &payload)
    {
        Ok(descriptor) => response.descriptor = descriptor,
        Err(error) => return error_response(response, local_error_status(&error), &error),
    }
    response
}

fn error_response(
    mut response: Response,
    status: StatusCode,
    error: &impl std::fmt::Display,
) -> Response {
    error!("Local control command failed: {error}");
    response.status = status;
    response.value0 = 0;
    response
}

fn local_error_status(error: &ArenaError) -> StatusCode {
    match error {
        ArenaError::StaleGeneration { .. } => StatusCode::StaleGeneration,
        ArenaError::InvalidOffset { .. }
        | ArenaError::PayloadTooLarge { .. }
        | ArenaError::LengthMismatch { .. }
        | ArenaError::ZeroGeneration => StatusCode::Invalid,
        ArenaError::TooSmall { .. }
        | ArenaError::FieldOverflow { .. }
        | ArenaError::System(_)
        | ArenaError::InvalidMagic(_)
        | ArenaError::UnsupportedVersion(_)
        | ArenaError::SessionMismatch { .. }
        | ArenaError::SlotOutOfRange { .. }
        | ArenaError::ConcurrentWrite { .. }
        | ArenaError::Poisoned => StatusCode::Internal,
    }
}

fn bootstrap_error_status(error: &BootstrapError) -> StatusCode {
    match error {
        BootstrapError::UnexpectedGeneration { .. } => StatusCode::StaleGeneration,
        BootstrapError::ClientTokenMismatch | BootstrapError::SlotMismatch { .. } => {
            StatusCode::StaleSession
        }
        _ => StatusCode::Invalid,
    }
}

fn engine_error_status(error: &EngineError) -> StatusCode {
    match error {
        EngineError::InvalidArgument(_)
        | EngineError::InstanceMissing(_)
        | EngineError::WorkerMissing(_, _)
        | EngineError::TopologyMismatch(_) => StatusCode::Invalid,
        EngineError::CudaInit(_) | EngineError::Storage(_) | EngineError::Poisoned(_) => {
            StatusCode::Internal
        }
    }
}
