mod pending;
mod session;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::metric::hll::MultiWindowHllTracker;
use log::{error, info};
use orbitkv_channel::{
    ArenaError, BootstrapError, BootstrapServer, BootstrapSession, Command, CommandCode,
    DeferredResponse, PublishRequest as ChannelPublishRequest, QueryBundleResponse,
    QueryOutcomeCode, RESPONSE_FLAG_REQUEST_CONSUMED, ReleaseRequest as ChannelReleaseRequest,
    Response, RestoreCommand, RestoreResponse, RestoreState, StatusCode, TransportError,
    TransportServer,
};
use orbitkv_core::{EngineError, OrbitKVEngine};
use thiserror::Error;
use tokio::runtime::Handle;
use tokio::sync::Notify;

use crate::cache::operations::{
    PublishInput, PublishLayerInput, QueryOutcome, RestoreInput, RestoreLeaseInput,
    execute_publish, execute_release, execute_restore,
};

const IDLE_POLL_INTERVAL: Duration = Duration::from_micros(50);
const BOOTSTRAP_POLL_INTERVAL: Duration = Duration::from_millis(10);
const LIVENESS_POLL_INTERVAL: Duration = Duration::from_millis(250);
const MAX_RESTORE_OPERATIONS_PER_SESSION: usize = 1024;
const MAX_RESTORE_ERROR_BYTES: usize = 4096;

enum RestoreOperation {
    Pending {
        receiver: tokio::sync::oneshot::Receiver<orbitkv_core::LoadOutcome>,
        started: Instant,
    },
    Complete {
        result: Result<(), String>,
        completed_at: Instant,
    },
}

#[derive(Debug, Error)]
pub(crate) enum ProcessEndpointError {
    #[error(transparent)]
    Transport(#[from] TransportError),
    #[error(transparent)]
    Bootstrap(#[from] BootstrapError),
}

/// Dedicated iceoryx2 endpoint owned by one Cache Manager process.
///
/// QueryBundle uses a generation-checked memfd arena bootstrapped over UDS.
/// Lifecycle metadata uses the authenticated bootstrap UDS.
pub(crate) struct ProcessEndpoint {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl ProcessEndpoint {
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
        lifecycle: crate::cache::lifecycle::LifecycleService,
        read_batch_bytes: u64,
        read_timeout: Option<Duration>,
        read_max_batches: usize,
    ) -> Result<Self, ProcessEndpointError> {
        // A dead Manager's clients can keep its iceoryx2 service alive.
        // Publish a fresh incarnation through the stable bootstrap socket.
        let service_name = format!("{service_name}/{}", uuid::Uuid::new_v4().simple());
        let server = TransportServer::bind(&service_name)?;
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
            .name("orbitkv-channel-control".to_string())
            .spawn(move || {
                info!(
                    "Process channel endpoint ready: service={} session_epoch={} bootstrap={}",
                    thread_service,
                    session_epoch,
                    bootstrap_socket.display()
                );
                let mut sessions = HashMap::new();
                let mut operations = HashMap::new();
                let mut queries = pending::PendingQueries::default();
                queries.read_batch_bytes = read_batch_bytes;
                queries.read_timeout = read_timeout;
                queries.read_max_batches = read_max_batches;
                let mut next_operation_id = 1u64;
                let mut next_bootstrap_poll = Instant::now();
                let mut next_liveness_poll = Instant::now();
                while !thread_stop.load(Ordering::Acquire) {
                    let now = Instant::now();
                    if now >= next_bootstrap_poll {
                        accept_pending_sessions(
                            &bootstrap,
                            &mut sessions,
                            &runtime,
                            &lifecycle,
                            session_epoch,
                            &shutdown,
                        );
                        next_bootstrap_poll = now + BOOTSTRAP_POLL_INTERVAL;
                    }
                    if now >= next_liveness_poll {
                        let mut dead_sessions = Vec::new();
                        sessions.retain(|token, session| {
                            let alive = match session.is_alive() {
                                Ok(alive) => alive,
                                Err(error) => {
                                    error!("Bootstrap liveness check failed: {error}");
                                    false
                                }
                            };
                            if !alive {
                                dead_sessions.push(*token);
                            }
                            alive
                        });
                        operations.retain(|(token, _), _| !dead_sessions.contains(token));
                        queries.retain_sessions(&engine, |token| sessions.contains_key(&token));
                        next_liveness_poll = now + LIVENESS_POLL_INTERVAL;
                    }
                    advance_restore_operations(&sessions, &mut operations, session_epoch);

                    let mut request_shutdown = false;
                    match server.try_serve_deferred_for_epoch(session_epoch, |command, reply| {
                        if command.code == CommandCode::Publish {
                            dispatch_publish(
                                command,
                                &bootstrap,
                                &mut sessions,
                                &engine,
                                &runtime,
                                reply,
                            )
                        } else {
                            reply.send(dispatch(
                                command,
                                &bootstrap,
                                &mut sessions,
                                &engine,
                                &runtime,
                                &hll_tracker,
                                &mut operations,
                                &mut queries,
                                &mut next_operation_id,
                                &mut request_shutdown,
                            ))
                        }
                    }) {
                        Ok(true) if request_shutdown => {
                            shutdown.notify_waiters();
                            break;
                        }
                        Ok(true) => {}
                        Ok(false) => thread::sleep(IDLE_POLL_INTERVAL),
                        Err(error) => {
                            error!("Process channel request failed: {error}");
                            thread::sleep(IDLE_POLL_INTERVAL);
                        }
                    }
                }
                info!("Process channel endpoint stopped: service={thread_service}");
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
            error!("Process channel thread panicked during shutdown");
        }
    }
}

impl Drop for ProcessEndpoint {
    fn drop(&mut self) {
        self.stop();
    }
}

fn accept_pending_sessions(
    bootstrap: &BootstrapServer,
    sessions: &mut HashMap<u64, BootstrapSession>,
    runtime: &Handle,
    lifecycle: &crate::cache::lifecycle::LifecycleService,
    epoch: u64,
    shutdown: &Arc<Notify>,
) {
    loop {
        match bootstrap.try_accept() {
            Ok(Some(session)) => {
                info!(
                    "Inference client bootstrapped: pid={} uid={} slot={}",
                    session.credentials().pid,
                    session.credentials().uid,
                    session.slot_index()
                );
                match session.stream().try_clone() {
                    Ok(stream) => {
                        runtime.spawn(session::serve(
                            stream,
                            epoch,
                            lifecycle.clone(),
                            Arc::clone(shutdown),
                        ));
                    }
                    Err(error) => {
                        error!("Cannot start process-channel lifecycle: {error}");
                        continue;
                    }
                }
                sessions.insert(session.client_token(), session);
            }
            Ok(None) => break,
            Err(error) => {
                error!("Bootstrap accept failed: {error}");
                break;
            }
        }
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "dispatch names each Cache Manager-owned subsystem explicitly"
)]
fn dispatch(
    command: Command,
    bootstrap: &BootstrapServer,
    sessions: &mut HashMap<u64, BootstrapSession>,
    engine: &Arc<OrbitKVEngine>,
    runtime: &Handle,
    hll_tracker: &Arc<std::sync::Mutex<MultiWindowHllTracker>>,
    operations: &mut HashMap<(u64, u64), RestoreOperation>,
    queries: &mut pending::PendingQueries,
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
        CommandCode::QueryBundle | CommandCode::CancelQuery => {
            response = dispatch_query(
                command,
                bootstrap,
                sessions,
                engine,
                runtime,
                hll_tracker,
                queries,
            );
        }
        CommandCode::Release => {
            response = dispatch_release(command, bootstrap, sessions, engine);
        }
        CommandCode::Publish => unreachable!("publish uses deferred response handling"),
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
    epoch: u64,
) {
    for ((token, id), operation) in operations.iter_mut() {
        #[cfg(feature = "test-hooks")]
        if orbitkv_core::test_faults::active("restore") {
            continue;
        }
        let completion = match operation {
            RestoreOperation::Pending { receiver, started } => match receiver.try_recv() {
                Ok(outcome) => Some((
                    outcome
                        .result
                        .map_err(|error| truncate_error(error.to_string())),
                    outcome.completed_at,
                    *started,
                )),
                Err(tokio::sync::oneshot::error::TryRecvError::Empty) => None,
                Err(tokio::sync::oneshot::error::TryRecvError::Closed) => Some((
                    Err("restore completion channel closed".to_string()),
                    Instant::now(),
                    *started,
                )),
            },
            RestoreOperation::Complete { .. } => None,
        };
        if let Some((result, completed_at, started)) = completion {
            crate::metric::timeline::record("restore_complete", || {
                serde_json::json!({
                    "restore_key": format!("manager:{epoch}:{id}"),
                    "elapsed_us": completed_at.saturating_duration_since(started).as_micros() as u64,
                    "success": result.is_ok(),
                })
            });
            *operation = RestoreOperation::Complete {
                result,
                completed_at,
            };
            #[cfg(feature = "test-hooks")]
            if orbitkv_core::test_faults::active("notification") {
                continue;
            }
            if let Some(session) = sessions.get(token)
                && let Err(error) = session.notify()
            {
                error!("Failed to notify local restore completion: {error}");
            }
            crate::metric::timeline::record("restore_notification", || {
                serde_json::json!({
                    "restore_key": format!("manager:{epoch}:{id}"),
                    "elapsed_us": completed_at.elapsed().as_micros() as u64,
                })
            });
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
            let started = Instant::now();
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
                RestoreOperation::Pending { receiver, started },
            );
            RestoreResponse {
                operation_id,
                state: RestoreState::Pending,
                message: String::new(),
            }
        }
        RestoreCommand::Poll { operation_id } => {
            match operations.get(&(command.arg0, operation_id)) {
                Some(RestoreOperation::Pending { .. }) => RestoreResponse {
                    operation_id,
                    state: RestoreState::Pending,
                    message: String::new(),
                },
                Some(RestoreOperation::Complete { result: Ok(()), .. }) => RestoreResponse {
                    operation_id,
                    state: RestoreState::Succeeded,
                    message: String::new(),
                },
                Some(RestoreOperation::Complete {
                    result: Err(message),
                    ..
                }) => {
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
            if let Some(key) = completed_operation
                && let Some(RestoreOperation::Complete { completed_at, .. }) =
                    operations.remove(&key)
            {
                crate::metric::timeline::record("restore_delivered", || {
                    serde_json::json!({
                        "restore_key": format!("manager:{}:{}", command.session_epoch, key.1),
                        "elapsed_us": completed_at.elapsed().as_micros() as u64,
                    })
                });
            }
        }
        Err(error) => return error_response(response, arena_error_status(&error), &error),
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
                arena_error_status(&error),
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
                arena_error_status(&error),
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
    engine: &Arc<OrbitKVEngine>,
    runtime: &Handle,
    reply: DeferredResponse,
) -> Result<(), TransportError> {
    let mut response = Response::ok(command);
    response.value1 = 0;
    let payload = match consume_descriptor(command, bootstrap, sessions, &mut response) {
        Ok(payload) => payload,
        Err(response) => return reply.send(response),
    };
    let request = match ChannelPublishRequest::decode(&payload) {
        Ok(request) => request,
        Err(error) => return reply.send(error_response(response, StatusCode::Invalid, &error)),
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
    match bootstrap.arena().write_response(command.descriptor, &[]) {
        Ok(descriptor) => response.descriptor = descriptor,
        Err(error) => {
            return reply.send(error_response(response, arena_error_status(&error), &error));
        }
    }
    let engine = Arc::clone(engine);
    runtime.spawn(async move {
        #[cfg(feature = "test-hooks")]
        orbitkv_core::test_faults::pause("publish").await;
        let result = execute_publish(
            &engine,
            PublishInput {
                instance_id: request.instance_id,
                tp_rank: request.tp_rank,
                pp_rank: request.pp_rank,
                device_id: request.device_id,
                layers,
            },
        )
        .await;
        if let Err(error) = result {
            response = error_response(response, engine_error_status(&error), &error);
        }
        #[cfg(feature = "test-hooks")]
        if orbitkv_core::test_faults::active("publish_ack") {
            response.value1 = 0;
        }
        if let Err(error) = reply.send(response) {
            error!("Failed to reply to completed publish: {error}");
        }
    });
    Ok(())
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
    let request = match ChannelReleaseRequest::decode(&payload) {
        Ok(request) => request,
        Err(error) => return error_response(response, StatusCode::Invalid, &error),
    };
    if let Err(error) = execute_release(engine, &request.lease) {
        return error_response(response, StatusCode::Invalid, &error);
    }
    match bootstrap.arena().write_response(command.descriptor, &[]) {
        Ok(descriptor) => response.descriptor = descriptor,
        Err(error) => return error_response(response, arena_error_status(&error), &error),
    }
    response
}

#[allow(
    clippy::too_many_arguments,
    reason = "dispatch owns transport and query state"
)]
fn dispatch_query(
    command: Command,
    bootstrap: &BootstrapServer,
    sessions: &mut HashMap<u64, BootstrapSession>,
    engine: &Arc<OrbitKVEngine>,
    runtime: &Handle,
    hll_tracker: &Arc<std::sync::Mutex<MultiWindowHllTracker>>,
    queries: &mut pending::PendingQueries,
) -> Response {
    let mut response = Response::ok(command);
    response.value1 = 0;
    let payload = match consume_descriptor(command, bootstrap, sessions, &mut response) {
        Ok(payload) => payload,
        Err(response) => return response,
    };
    if command.code == CommandCode::CancelQuery {
        let request = match orbitkv_channel::CancelQueryRequest::decode(&payload) {
            Ok(request) => request,
            Err(error) => return error_response(response, StatusCode::Invalid, &error),
        };
        queries.cancel(command.arg0, request.ticket, engine);
        return match bootstrap.arena().write_response(command.descriptor, &[]) {
            Ok(descriptor) => {
                response.descriptor = descriptor;
                response
            }
            Err(error) => error_response(response, arena_error_status(&error), &error),
        };
    }
    let request = match orbitkv_channel::QueryCommand::decode(&payload) {
        Ok(request) => request,
        Err(error) => return error_response(response, StatusCode::Invalid, &error),
    };
    let mut reply = match queries.execute(command.arg0, request, engine, runtime, hll_tracker) {
        Ok(reply) => reply,
        Err(error) => return error_response(response, engine_error_status(&error), &error),
    };
    let outcome = match reply.as_ref().map(|reply| &reply.outcome) {
        Some(Ok(outcome)) => outcome.clone(),
        Some(Err(error)) => return error_response(response, engine_error_status(error), error),
        None => QueryOutcome::Loading,
    };
    let payload = match outcome {
        QueryOutcome::Busy => QueryBundleResponse {
            outcome: QueryOutcomeCode::Busy,
            ..QueryBundleResponse::loading()
        },
        QueryOutcome::Loading => QueryBundleResponse::loading(),
        QueryOutcome::Candidates { hit_positions } => QueryBundleResponse {
            outcome: QueryOutcomeCode::Candidates,
            num_hit_blocks: hit_positions.len() as u64,
            lease: Vec::new(),
            hit_positions,
        },
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
        Ok(descriptor) => {
            response.descriptor = descriptor;
            if let Some(reply) = reply.as_mut() {
                reply.delivered();
            }
        }
        Err(error) => return error_response(response, arena_error_status(&error), &error),
    }
    response
}

fn error_response(
    mut response: Response,
    status: StatusCode,
    error: &impl std::fmt::Display,
) -> Response {
    error!("Process channel command failed: {error}");
    response.status = status;
    response.value0 = 0;
    response
}

fn arena_error_status(error: &ArenaError) -> StatusCode {
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
