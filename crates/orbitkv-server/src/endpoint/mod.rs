mod pending;
mod restore;
mod session;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::metric::hll::MultiWindowHllTracker;
use log::{error, info};
use orbitkv_channel::{
    ArenaError, BootstrapError, BootstrapServer, BootstrapSession, Command, CommandCode,
    CompletionObservationRequest, DeferredResponse, PublishRequest as ChannelPublishRequest,
    QueryBundleResponse, QueryOutcomeCode, RESPONSE_FLAG_REQUEST_CONSUMED,
    ReleaseRequest as ChannelReleaseRequest, Response, RestoreRequest, StatusCode, TransportError,
    TransportServer,
};
use orbitkv_core::{
    CompletionAdmission, CompletionIntent, CompletionObservation, CompletionOutcome,
    CompletionResourceEvidence, CompletionRoute, EngineError, OrbitKVEngine,
};
use thiserror::Error;
use tokio::runtime::Handle;
use tokio::sync::Notify;

use crate::cache::operations::{
    PublishInput, PublishLayerInput, QueryOutcome, RestoreInput, RestoreLeaseInput,
    execute_publish, execute_release, execute_restore,
};

const BOOTSTRAP_POLL_INTERVAL: Duration = Duration::from_millis(10);
const LIVENESS_POLL_INTERVAL: Duration = Duration::from_millis(250);
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
    active_publishes: Arc<ActiveTasks>,
    lifecycle_connections: Arc<ActiveTasks>,
}

#[derive(Default)]
struct ActiveTasks {
    count: AtomicUsize,
    drained: Notify,
}

impl ActiveTasks {
    fn admit(self: &Arc<Self>) -> ActiveTask {
        self.count.fetch_add(1, Ordering::AcqRel);
        ActiveTask(Arc::clone(self))
    }

    async fn drain(&self) {
        loop {
            if self.count.load(Ordering::Acquire) == 0 {
                return;
            }
            let drained = self.drained.notified();
            if self.count.load(Ordering::Acquire) == 0 {
                return;
            }
            drained.await;
        }
    }
}

struct ActiveTask(Arc<ActiveTasks>);

impl Drop for ActiveTask {
    fn drop(&mut self) {
        if self.0.count.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.0.drained.notify_one();
        }
    }
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
        lifecycle_shutdown: Arc<Notify>,
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
        let active_publishes = Arc::new(ActiveTasks::default());
        let thread_active_publishes = Arc::clone(&active_publishes);
        let lifecycle_connections = Arc::new(ActiveTasks::default());
        let thread_lifecycle_connections = Arc::clone(&lifecycle_connections);
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
                let mut grants = HashMap::new();
                let mut queries = pending::PendingQueries::default();
                queries.read_batch_bytes = read_batch_bytes;
                queries.read_timeout = read_timeout;
                queries.read_max_batches = read_max_batches;
                let mut next_bootstrap_poll = Instant::now();
                let mut next_liveness_poll = Instant::now();
                loop {
                    if thread_stop.load(Ordering::Acquire) {
                        if thread_active_publishes.count.load(Ordering::Acquire) == 0 {
                            break;
                        }
                        thread::sleep(Duration::from_millis(1));
                        continue;
                    }
                    let now = Instant::now();
                    if now >= next_bootstrap_poll {
                        accept_pending_sessions(
                            &bootstrap,
                            &mut sessions,
                            &mut grants,
                            &runtime,
                            &lifecycle,
                            session_epoch,
                            &lifecycle_shutdown,
                            &thread_lifecycle_connections,
                        );
                        next_bootstrap_poll = now + BOOTSTRAP_POLL_INTERVAL;
                    }
                    if now >= next_liveness_poll {
                        sessions.retain(|_, session| match session.is_alive() {
                            Ok(alive) => alive,
                            Err(error) => {
                                error!("Bootstrap liveness check failed: {error}");
                                false
                            }
                        });
                        grants.retain(|token, _| sessions.contains_key(token));
                        queries.retain_sessions(&engine, |token| sessions.contains_key(&token));
                        next_liveness_poll = now + LIVENESS_POLL_INTERVAL;
                    }

                    let mut request_shutdown = false;
                    match server.try_serve_deferred_for_epoch(session_epoch, |command, reply| {
                        if command.code == CommandCode::Publish {
                            dispatch_publish(
                                command,
                                &bootstrap,
                                &mut sessions,
                                &engine,
                                &runtime,
                                &thread_active_publishes,
                                reply,
                            )
                        } else {
                            reply.send(dispatch(
                                command,
                                &bootstrap,
                                &mut sessions,
                                &grants,
                                &engine,
                                &runtime,
                                &hll_tracker,
                                &mut queries,
                                &mut request_shutdown,
                            ))
                        }
                    }) {
                        Ok(true) if request_shutdown => {
                            shutdown.notify_waiters();
                            break;
                        }
                        Ok(true) => continue,
                        Ok(false) => {}
                        Err(error) => error!("Process channel request failed: {error}"),
                    }
                    let maintenance = next_bootstrap_poll.min(next_liveness_poll);
                    if let Err(error) = server
                        .wait_for_request(maintenance.saturating_duration_since(Instant::now()))
                    {
                        error!("Process channel request wait failed: {error}");
                        shutdown.notify_waiters();
                        break;
                    }
                }
                info!("Process channel endpoint stopped: service={thread_service}");
            })
            .map_err(|error| TransportError::Thread(error.to_string()))?;

        Ok(Self {
            stop,
            thread: Some(thread),
            active_publishes,
            lifecycle_connections,
        })
    }

    pub(crate) fn stop(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.join_thread();
    }

    fn join_thread(&mut self) {
        if let Some(thread) = self.thread.take()
            && thread.join().is_err()
        {
            error!("Process channel thread panicked during shutdown");
        }
    }

    pub(crate) async fn stop_and_drain_publishes(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.active_publishes.drain().await;
        self.join_thread();
    }

    pub(crate) async fn drain_lifecycle_connections(&self) {
        self.lifecycle_connections.drain().await;
    }
}

impl Drop for ProcessEndpoint {
    fn drop(&mut self) {
        self.stop();
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "session admission names separate shutdown and lifecycle owners"
)]
fn accept_pending_sessions(
    bootstrap: &BootstrapServer,
    sessions: &mut HashMap<u64, BootstrapSession>,
    grants: &mut HashMap<u64, restore::LocalGrants>,
    runtime: &Handle,
    lifecycle: &crate::cache::lifecycle::LifecycleService,
    epoch: u64,
    lifecycle_shutdown: &Arc<Notify>,
    lifecycle_connections: &Arc<ActiveTasks>,
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
                match restore::LocalGrants::start(
                    Arc::clone(session.completions()),
                    runtime,
                    epoch,
                    session.client_token(),
                ) {
                    Ok(owner) => {
                        grants.insert(session.client_token(), owner);
                    }
                    Err(error) => {
                        error!("Cannot start restore grant reaper: {error}");
                        continue;
                    }
                }
                match session.stream().try_clone() {
                    Ok(stream) => {
                        let active = lifecycle_connections.admit();
                        let lifecycle = lifecycle.clone();
                        let shutdown = Arc::clone(lifecycle_shutdown);
                        runtime.spawn(async move {
                            let _active = active;
                            session::serve(stream, epoch, lifecycle, shutdown).await;
                        });
                    }
                    Err(error) => {
                        error!("Cannot start process-channel lifecycle: {error}");
                        grants.remove(&session.client_token());
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
    grants: &HashMap<u64, restore::LocalGrants>,
    engine: &Arc<OrbitKVEngine>,
    runtime: &Handle,
    hll_tracker: &Arc<std::sync::Mutex<MultiWindowHllTracker>>,
    queries: &mut pending::PendingQueries,
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
        CommandCode::ObserveCompletion => {
            response = dispatch_completion_observation(command, bootstrap, sessions, engine);
        }
        CommandCode::Publish => unreachable!("publish uses deferred response handling"),
        CommandCode::Restore => {
            response = dispatch_restore(command, bootstrap, sessions, grants, engine, runtime);
        }
    }
    response
}

fn dispatch_completion_observation(
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
    let request = match CompletionObservationRequest::decode(&payload) {
        Ok(request) => request,
        Err(error) => return error_response(response, StatusCode::Invalid, &error),
    };
    let observation = CompletionObservation {
        instance_id: request.instance_id,
        destination_device_id: request.destination_device_id,
        source_endpoint: request.source_endpoint,
        transfer_generation: request.transfer_generation,
        intent: match request.intent {
            orbitkv_channel::CompletionIntent::HostReady => CompletionIntent::HostReady,
            orbitkv_channel::CompletionIntent::EngineRestore => CompletionIntent::EngineRestore,
            orbitkv_channel::CompletionIntent::SourceRelease => CompletionIntent::SourceRelease,
        },
        route: match request.route {
            orbitkv_channel::CompletionRoute::PrefillToDecodeHandoff => {
                CompletionRoute::PrefillToDecodeHandoff
            }
        },
        representation: request.representation,
        logical_bytes: request.logical_bytes,
        wire_bytes: request.wire_bytes,
        fragment_count: request.fragment_count,
        elapsed: Duration::from_nanos(request.elapsed_ns),
        resources: CompletionResourceEvidence {
            decode_page_bytes: request.decode_page_bytes,
            queue_depth: request.handoff_queue_depth,
            queue_parallelism: request.handoff_queue_parallelism,
            tent_inflight_bytes: request.tent_inflight_bytes,
            tent_bandwidth_bytes_per_second: request.tent_bandwidth_bytes_per_second,
        },
        admission: match request.admission {
            orbitkv_channel::CompletionAdmission::Admitted => CompletionAdmission::Admitted,
            orbitkv_channel::CompletionAdmission::Rejected => CompletionAdmission::Rejected,
        },
        outcome: match request.outcome {
            orbitkv_channel::CompletionOutcome::Completed => CompletionOutcome::Completed,
            orbitkv_channel::CompletionOutcome::Failed => CompletionOutcome::Failed,
            orbitkv_channel::CompletionOutcome::Cancelled => CompletionOutcome::Cancelled,
            orbitkv_channel::CompletionOutcome::TimedOut => CompletionOutcome::TimedOut,
        },
    };
    if let Err(error) = engine.observe_completion(observation) {
        return error_response(response, engine_error_status(&error), &error);
    }
    match bootstrap.arena().write_response(command.descriptor, &[]) {
        Ok(descriptor) => response.descriptor = descriptor,
        Err(error) => return error_response(response, arena_error_status(&error), &error),
    }
    response
}

fn dispatch_restore(
    command: Command,
    bootstrap: &BootstrapServer,
    sessions: &mut HashMap<u64, BootstrapSession>,
    grants: &HashMap<u64, restore::LocalGrants>,
    engine: &OrbitKVEngine,
    runtime: &Handle,
) -> Response {
    let mut response = Response::ok(command);
    response.value1 = 0;
    let payload = match consume_descriptor(command, bootstrap, sessions, &mut response) {
        Ok(payload) => payload,
        Err(response) => return response,
    };
    let completions = Arc::clone(sessions[&command.arg0].completions());
    let operation_id = command.arg1;
    if let Err(error) = completions.claim(operation_id) {
        return error_response(response, StatusCode::Invalid, &error);
    }
    let started = Instant::now();
    let prepared = (|| {
        let request = RestoreRequest::decode(&payload).map_err(|error| error.to_string())?;
        let loads = request
            .loads
            .into_iter()
            .map(|load| RestoreLeaseInput {
                lease: load.lease,
                block_ids_by_group: load.block_ids_by_group,
            })
            .collect();
        execute_restore(
            engine,
            RestoreInput {
                instance_id: request.instance_id,
                tp_rank: request.tp_rank,
                device_id: request.device_id,
                layer_groups: request.layer_groups,
                loads,
            },
        )
        .map_err(|error| error.to_string())
    })();
    match prepared {
        Ok(orbitkv_core::RestoreExecution::Local(grant)) => {
            if let Some(owner) = grants.get(&command.arg0) {
                owner.install(operation_id, grant);
            } else {
                drop(grant);
                let _ = completions.reject(operation_id, "restore session is closed".into());
            }
        }
        Ok(orbitkv_core::RestoreExecution::Managed(receiver)) => {
            runtime.spawn(restore::publish(
                receiver,
                completions,
                command.session_epoch,
                command.arg0,
                operation_id,
                started,
            ));
        }
        Err(error) => {
            if let Err(error) = completions.reject(operation_id, error) {
                error!("Cannot publish rejected restore completion: {error}");
            }
        }
    }
    #[cfg(feature = "test-hooks")]
    if orbitkv_core::test_faults::active("restore_ack") {
        response.value1 = 0;
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
    active_publishes: &Arc<ActiveTasks>,
    reply: DeferredResponse,
) -> Result<(), TransportError> {
    orbitkv_common::timeline::record_diagnostic(
        "publish_manager_receive",
        orbitkv_common::timeline::DiagnosticFields::operation(
            command.request_id,
            command.session_epoch,
            command.arg0,
        ),
    );
    let reply_notification = sessions
        .get(&command.arg0)
        .map(|session| Arc::clone(session.reply_notification_fd()));
    let send_reply = move |response| match reply_notification.as_deref() {
        Some(notification) => reply.send_and_notify(response, notification),
        None => reply.send(response),
    };
    let mut response = Response::ok(command);
    response.value1 = 0;
    let payload = match consume_descriptor(command, bootstrap, sessions, &mut response) {
        Ok(payload) => payload,
        Err(response) => return send_reply(response),
    };
    let request = match ChannelPublishRequest::decode(&payload) {
        Ok(request) => request,
        Err(error) => return send_reply(error_response(response, StatusCode::Invalid, &error)),
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
            return send_reply(error_response(response, arena_error_status(&error), &error));
        }
    }
    let engine = Arc::clone(engine);
    let request_id = command.request_id;
    let session_epoch = command.session_epoch;
    let session_token = command.arg0;
    let diagnostic =
        orbitkv_common::timeline::diagnostic_enabled().then_some(orbitkv_core::PublishDiagnostic {
            request_id,
            session_epoch,
            session_token,
        });
    let active_publish = active_publishes.admit();
    runtime.spawn(async move {
        let _active_publish = active_publish;
        orbitkv_common::timeline::record_diagnostic(
            "publish_manager_process_start",
            orbitkv_common::timeline::DiagnosticFields::operation(
                request_id,
                session_epoch,
                session_token,
            ),
        );
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
                diagnostic,
            },
        )
        .await;
        orbitkv_common::timeline::record_diagnostic(
            "publish_manager_process_complete",
            orbitkv_common::timeline::DiagnosticFields::operation(
                request_id,
                session_epoch,
                session_token,
            )
            .success(result.is_ok()),
        );
        if let Err(error) = result {
            response = error_response(response, engine_error_status(&error), &error);
        }
        #[cfg(feature = "test-hooks")]
        if orbitkv_core::test_faults::active("publish_ack") {
            response.value1 = 0;
        }
        orbitkv_common::timeline::record_diagnostic(
            "publish_manager_response_publish",
            orbitkv_common::timeline::DiagnosticFields::operation(
                request_id,
                session_epoch,
                session_token,
            ),
        );
        let sent = send_reply(response);
        if let Err(error) = sent {
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
    orbitkv_common::timeline::record_diagnostic(
        "query_manager_receive",
        orbitkv_common::timeline::DiagnosticFields::operation(
            command.request_id,
            command.session_epoch,
            command.arg0,
        ),
    );
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
    let diagnostic_outcome = match &outcome {
        QueryOutcome::Busy => "Busy",
        QueryOutcome::Loading => "Loading",
        QueryOutcome::Candidates { .. } => "Candidates",
        QueryOutcome::Ready { .. } => "Ready",
    };
    orbitkv_common::timeline::record_diagnostic(
        "query_manager_process_complete",
        orbitkv_common::timeline::DiagnosticFields::operation(
            command.request_id,
            command.session_epoch,
            command.arg0,
        )
        .outcome(diagnostic_outcome),
    );
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

#[cfg(test)]
#[path = "../../tests/unit/endpoint/mod.rs"]
mod tests;
