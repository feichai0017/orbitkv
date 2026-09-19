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
    LocalServer, PublishRequest as LocalPublishRequest, QueryBundleRequest, QueryBundleResponse,
    QueryOutcomeCode, RESPONSE_FLAG_REQUEST_CONSUMED, ReleaseRequest as LocalReleaseRequest,
    Response, RestoreCommand, RestoreResponse, RestoreState, StatusCode, TransportError,
};
use thiserror::Error;
use tokio::runtime::Handle;
use tokio::sync::Notify;

use crate::query::{
    PublishInput, PublishLayerInput, QueryInput, QueryOutcome, RestoreInput, RestoreLeaseInput,
    execute_publish, execute_query, execute_release, execute_restore,
};

const IDLE_POLL_INTERVAL: Duration = Duration::from_micros(50);
const BOOTSTRAP_POLL_INTERVAL: Duration = Duration::from_millis(10);
const LIVENESS_POLL_INTERVAL: Duration = Duration::from_millis(250);
const MAX_RESTORE_OPERATIONS_PER_SESSION: usize = 1024;
const MAX_RESTORE_ERROR_BYTES: usize = 4096;

enum RestoreOperation {
    Pending(tokio::sync::oneshot::Receiver<Result<(), EngineError>>),
    Complete(Result<(), String>),
}

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
/// Restore remains an explicit `Invalid` response.
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
                let mut operations = HashMap::new();
                let mut next_operation_id = 1u64;
                let mut next_bootstrap_poll = Instant::now();
                let mut next_liveness_poll = Instant::now();
                while !thread_stop.load(Ordering::Acquire) {
                    let now = Instant::now();
                    if now >= next_bootstrap_poll {
                        accept_pending_sessions(&bootstrap, &mut sessions);
                        next_bootstrap_poll = now + BOOTSTRAP_POLL_INTERVAL;
                    }
                    if now >= next_liveness_poll {
                        let mut dead_sessions = Vec::new();
                        sessions.retain(|token, session| {
                            let alive = match session.is_alive() {
                                Ok(alive) => alive,
                                Err(error) => {
                                    error!("Local bootstrap liveness check failed: {error}");
                                    false
                                }
                            };
                            if !alive {
                                dead_sessions.push(*token);
                            }
                            alive
                        });
                        operations.retain(|(token, _), _| !dead_sessions.contains(token));
                        next_liveness_poll = now + LIVENESS_POLL_INTERVAL;
                    }
                    advance_restore_operations(&sessions, &mut operations);

                    let mut request_shutdown = false;
                    match server.try_serve_for_epoch(session_epoch, |command| {
                        dispatch(
                            command,
                            &bootstrap,
                            &mut sessions,
                            &engine,
                            &runtime,
                            &hll_tracker,
                            &mut operations,
                            &mut next_operation_id,
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

#[allow(
    clippy::too_many_arguments,
    reason = "dispatch names each sidecar-owned subsystem explicitly"
)]
fn dispatch(
    command: Command,
    bootstrap: &BootstrapServer,
    sessions: &mut HashMap<u64, BootstrapSession>,
    engine: &Arc<OrbitKVEngine>,
    runtime: &Handle,
    hll_tracker: &Arc<std::sync::Mutex<MultiWindowHllTracker>>,
    operations: &mut HashMap<(u64, u64), RestoreOperation>,
    next_operation_id: &mut u64,
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
        CommandCode::Release => {
            response = dispatch_release(command, bootstrap, sessions, engine);
        }
        CommandCode::Publish => {
            response = dispatch_publish(command, bootstrap, sessions, engine, runtime);
        }
        CommandCode::Restore => {
            response = dispatch_restore(
                command,
                bootstrap,
                sessions,
                engine,
                operations,
                next_operation_id,
            );
        }
    }
    response
}

fn advance_restore_operations(
    sessions: &HashMap<u64, BootstrapSession>,
    operations: &mut HashMap<(u64, u64), RestoreOperation>,
) {
    for ((token, _), operation) in operations.iter_mut() {
        let result = match operation {
            RestoreOperation::Pending(receiver) => match receiver.try_recv() {
                Ok(result) => Some(result.map_err(|error| truncate_error(error.to_string()))),
                Err(tokio::sync::oneshot::error::TryRecvError::Empty) => None,
                Err(tokio::sync::oneshot::error::TryRecvError::Closed) => {
                    Some(Err("restore completion channel closed".to_string()))
                }
            },
            RestoreOperation::Complete(_) => None,
        };
        if let Some(result) = result {
            *operation = RestoreOperation::Complete(result);
            if let Some(session) = sessions.get(token)
                && let Err(error) = session.notify()
            {
                error!("Failed to notify local restore completion: {error}");
            }
        }
    }
}

fn truncate_error(mut message: String) -> String {
    if message.len() <= MAX_RESTORE_ERROR_BYTES {
        return message;
    }
    let mut end = MAX_RESTORE_ERROR_BYTES;
    while !message.is_char_boundary(end) {
        end -= 1;
    }
    message.truncate(end);
    message
}

fn dispatch_restore(
    command: Command,
    bootstrap: &BootstrapServer,
    sessions: &mut HashMap<u64, BootstrapSession>,
    engine: &OrbitKVEngine,
    operations: &mut HashMap<(u64, u64), RestoreOperation>,
    next_operation_id: &mut u64,
) -> Response {
    let mut response = Response::ok(command);
    response.value1 = 0;
    let payload = match consume_descriptor(command, bootstrap, sessions, &mut response) {
        Ok(payload) => payload,
        Err(response) => return response,
    };
    let restore = match RestoreCommand::decode(&payload) {
        Ok(restore) => restore,
        Err(error) => return error_response(response, StatusCode::Invalid, &error),
    };
    let restore_response = match restore {
        RestoreCommand::Submit(request) => {
            let active_operations = operations
                .keys()
                .filter(|(token, _)| *token == command.arg0)
                .count();
            if active_operations >= MAX_RESTORE_OPERATIONS_PER_SESSION {
                return error_response(
                    response,
                    StatusCode::Invalid,
                    &"too many unconsumed restore operations",
                );
            }
            let operation_id = *next_operation_id;
            *next_operation_id = match next_operation_id.checked_add(1) {
                Some(next) => next,
                None => {
                    return error_response(
                        response,
                        StatusCode::Internal,
                        &"operation ids exhausted",
                    );
                }
            };
            let loads = request
                .loads
                .into_iter()
                .map(|load| RestoreLeaseInput {
                    lease: load.lease,
                    block_ids_by_group: load.block_ids_by_group,
                })
                .collect();
            let receiver = match execute_restore(
                engine,
                RestoreInput {
                    instance_id: request.instance_id,
                    tp_rank: request.tp_rank,
                    device_id: request.device_id,
                    layer_groups: request.layer_groups,
                    loads,
                },
            ) {
                Ok(receiver) => receiver,
                Err(error) => return error_response(response, engine_error_status(&error), &error),
            };
            operations.insert(
                (command.arg0, operation_id),
                RestoreOperation::Pending(receiver),
            );
            RestoreResponse {
                operation_id,
                state: RestoreState::Pending,
                message: String::new(),
            }
        }
        RestoreCommand::Poll { operation_id } => {
            match operations.get(&(command.arg0, operation_id)) {
                Some(RestoreOperation::Pending(_)) => RestoreResponse {
                    operation_id,
                    state: RestoreState::Pending,
                    message: String::new(),
                },
                Some(RestoreOperation::Complete(Ok(()))) => RestoreResponse {
                    operation_id,
                    state: RestoreState::Succeeded,
                    message: String::new(),
                },
                Some(RestoreOperation::Complete(Err(message))) => {
                    let message = message.clone();
                    RestoreResponse {
                        operation_id,
                        state: RestoreState::Failed,
                        message,
                    }
                }
                None => {
                    return error_response(
                        response,
                        StatusCode::Invalid,
                        &"unknown restore operation",
                    );
                }
            }
        }
    };
    let payload = match restore_response.encode() {
        Ok(payload) => payload,
        Err(error) => return error_response(response, StatusCode::Internal, &error),
    };
    let completed_operation = (restore_response.state != RestoreState::Pending)
        .then_some((command.arg0, restore_response.operation_id));
    match bootstrap
        .arena()
        .write_response(command.descriptor, &payload)
    {
        Ok(descriptor) => {
            response.descriptor = descriptor;
            if let Some(key) = completed_operation {
                operations.remove(&key);
            }
        }
        Err(error) => return error_response(response, local_error_status(&error), &error),
    }
    response
}

fn consume_descriptor(
    command: Command,
    bootstrap: &BootstrapServer,
    sessions: &mut HashMap<u64, BootstrapSession>,
    response: &mut Response,
) -> Result<Vec<u8>, Response> {
    let descriptor_slot = match bootstrap.descriptor_slot(command.descriptor.offset) {
        Ok(slot) => slot,
        Err(error) => {
            return Err(error_response(
                *response,
                local_error_status(&error),
                &error,
            ));
        }
    };
    let session = match sessions.get_mut(&command.arg0) {
        Some(session) => session,
        None => {
            response.status = StatusCode::StaleSession;
            return Err(*response);
        }
    };
    if let Err(error) = session.validate_request(command.descriptor, command.arg0, descriptor_slot)
    {
        return Err(error_response(
            *response,
            bootstrap_error_status(&error),
            &error,
        ));
    }
    let payload = match bootstrap.arena().read(command.descriptor) {
        Ok(payload) => payload,
        Err(error) => {
            return Err(error_response(
                *response,
                local_error_status(&error),
                &error,
            ));
        }
    };
    if let Err(error) = session.complete_request() {
        return Err(error_response(
            *response,
            bootstrap_error_status(&error),
            &error,
        ));
    }
    response.value1 |= RESPONSE_FLAG_REQUEST_CONSUMED;
    Ok(payload)
}

fn dispatch_publish(
    command: Command,
    bootstrap: &BootstrapServer,
    sessions: &mut HashMap<u64, BootstrapSession>,
    engine: &OrbitKVEngine,
    runtime: &Handle,
) -> Response {
    let mut response = Response::ok(command);
    response.value1 = 0;
    let payload = match consume_descriptor(command, bootstrap, sessions, &mut response) {
        Ok(payload) => payload,
        Err(response) => return response,
    };
    let request = match LocalPublishRequest::decode(&payload) {
        Ok(request) => request,
        Err(error) => return error_response(response, StatusCode::Invalid, &error),
    };
    let layers = request
        .layers
        .into_iter()
        .map(|layer| PublishLayerInput {
            layer_name: layer.layer_name,
            block_ids: layer.block_ids,
            block_hashes: layer.block_hashes,
        })
        .collect();
    if let Err(error) = runtime.block_on(execute_publish(
        engine,
        PublishInput {
            instance_id: request.instance_id,
            tp_rank: request.tp_rank,
            pp_rank: request.pp_rank,
            device_id: request.device_id,
            layers,
        },
    )) {
        return error_response(response, engine_error_status(&error), &error);
    }
    match bootstrap.arena().write_response(command.descriptor, &[]) {
        Ok(descriptor) => response.descriptor = descriptor,
        Err(error) => return error_response(response, local_error_status(&error), &error),
    }
    response
}

fn dispatch_release(
    command: Command,
    bootstrap: &BootstrapServer,
    sessions: &mut HashMap<u64, BootstrapSession>,
    engine: &OrbitKVEngine,
) -> Response {
    let mut response = Response::ok(command);
    response.value1 = 0;
    let payload = match consume_descriptor(command, bootstrap, sessions, &mut response) {
        Ok(payload) => payload,
        Err(response) => return response,
    };
    let request = match LocalReleaseRequest::decode(&payload) {
        Ok(request) => request,
        Err(error) => return error_response(response, StatusCode::Invalid, &error),
    };
    if let Err(error) = execute_release(engine, &request.lease) {
        return error_response(response, StatusCode::Invalid, &error);
    }
    match bootstrap.arena().write_response(command.descriptor, &[]) {
        Ok(descriptor) => response.descriptor = descriptor,
        Err(error) => return error_response(response, local_error_status(&error), &error),
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
    let payload = match consume_descriptor(command, bootstrap, sessions, &mut response) {
        Ok(payload) => payload,
        Err(response) => return response,
    };
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
