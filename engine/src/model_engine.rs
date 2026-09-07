use std::collections::{BTreeMap, VecDeque, btree_map::Entry};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
    mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError},
};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use std::{any::Any, panic::AssertUnwindSafe};

use futures_util::stream;
use orbitkv::{
    CacheSharingPolicy, EngineAppendIntent, EngineCompletionEvidence, EnginePublicationEvidence,
    EngineReleaseEvidence, EngineReleaseOutcome, EngineRequestId, EngineRetirementEvidence,
    HfRetentionOptions, RuntimeSession, compile_hf_runtime_manifest,
    kv_manager::{BackendArenaRegistration, CanonicalKvManager, ManagerConfig, ManagerStats},
};
use orbitkv_executor::{
    AttentionBatch, ExecutorArena, ExecutorPlan,
    model::{CompiledDecoder, DecoderClassStep, DecoderCompileConfig, DecoderConfig, DecoderStep},
};
use orbitkv_server::{
    BatchIntent, Engine, EngineAbortFuture, EngineEvent, EngineEventStream, EngineFuture,
    FinishReason, RequestId, RequestIntent, TokenOutput,
};
use thiserror::Error;
use tokio::sync::{mpsc as async_mpsc, oneshot};

/// Startup and capacity policy for the single-process model engine.
#[derive(Clone, Debug)]
pub struct ModelEngineConfig {
    pub model_directory: PathBuf,
    pub device_index: usize,
    pub page_tokens: u64,
    /// Physical pages for each manifest class, in canonical class-id order.
    pub page_counts: Vec<u32>,
    pub maximum_model_tokens: u64,
    pub maximum_prefill_tokens: usize,
    pub maximum_batch_tokens: usize,
    pub representative_prefill_tokens: usize,
    pub maximum_active_requests: usize,
    pub maximum_queued_requests: usize,
    pub event_buffer_size: usize,
    pub batch_wait_timeout: Duration,
    pub search_graphs: usize,
    pub search_seed: u64,
}

impl ModelEngineConfig {
    fn validate(&self) -> Result<(), ModelEngineError> {
        let maximum_prefill_tokens = u64::try_from(self.maximum_prefill_tokens)
            .map_err(|_| ModelEngineError::InvalidConfig)?;
        if self.page_tokens == 0
            || self.page_tokens > u64::from(u32::MAX)
            || self.page_counts.is_empty()
            || self.page_counts.contains(&0)
            || self.maximum_model_tokens == 0
            || self.maximum_model_tokens > u64::from(u32::MAX)
            || self.maximum_prefill_tokens < 2
            || maximum_prefill_tokens > self.maximum_model_tokens
            || self.maximum_batch_tokens < self.maximum_prefill_tokens
            || !(2..=self.maximum_prefill_tokens).contains(&self.representative_prefill_tokens)
            || self.maximum_active_requests == 0
            || self.maximum_active_requests > self.maximum_batch_tokens
            || self.maximum_queued_requests == 0
            || self.event_buffer_size < 3
            || self.batch_wait_timeout.is_zero()
            || self.batch_wait_timeout > Duration::from_secs(1)
            || self.search_graphs < 2
        {
            return Err(ModelEngineError::InvalidConfig);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum ModelEngineError {
    #[error("invalid model-engine configuration")]
    InvalidConfig,
    #[error("each engine submission must contain exactly one request")]
    BatchSize,
    #[error("the model engine currently accepts only a fresh prompt")]
    ContinuationUnsupported,
    #[error("invalid request intent: {0}")]
    InvalidIntent(String),
    #[error("request exceeds the configured model-token capacity")]
    ModelLength,
    #[error("request is already queued or active")]
    DuplicateRequest,
    #[error("model engine admission queue is full")]
    QueueFull,
    #[error("model engine worker is unavailable")]
    WorkerUnavailable,
    #[error("model engine worker panicked: {0}")]
    WorkerPanicked(String),
    #[error("model engine initialization failed: {0}")]
    Initialization(String),
    #[error("OrbitKV lifecycle failed: {0}")]
    Lifecycle(String),
    #[error("executor failed: {0}")]
    Executor(String),
}

enum WorkerCommand {
    Run {
        request: RequestIntent,
        output: async_mpsc::Sender<Result<EngineEvent, ModelEngineError>>,
        cancelled: Arc<AtomicBool>,
    },
    Stats {
        reply: oneshot::Sender<EngineStats>,
    },
    Shutdown,
}

struct EngineShared {
    commands: SyncSender<WorkerCommand>,
    registry: Arc<Mutex<RequestRegistry>>,
    shutdown: Arc<AtomicBool>,
    worker: Mutex<Option<JoinHandle<()>>>,
    maximum_model_tokens: u64,
    maximum_prefill_tokens: usize,
    maximum_total_requests: usize,
    event_buffer_size: usize,
}

struct RequestRegistry {
    accepting: bool,
    cancellations: BTreeMap<RequestId, Arc<AtomicBool>>,
}

/// Point-in-time manager and continuous-batching scheduler census.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EngineStats {
    pub manager: ManagerStats,
    pub queued_requests: u64,
    pub active_requests: u64,
    pub admitted_requests: u64,
    pub completed_requests: u64,
    pub model_dispatches: u64,
    pub multi_request_dispatches: u64,
    pub mixed_phase_dispatches: u64,
    pub maximum_observed_batch_size: u64,
}

impl Drop for EngineShared {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        if let Ok(mut registry) = self.registry.lock() {
            registry.accepting = false;
            for cancelled in registry.cancellations.values() {
                cancelled.store(true, Ordering::Release);
            }
        }
        let _ = self.commands.try_send(WorkerCommand::Shutdown);
        if let Ok(worker) = self.worker.get_mut()
            && let Some(worker) = worker.take()
        {
            let _ = worker.join();
        }
    }
}

/// Cloneable logical handle to one dedicated model-execution thread.
#[derive(Clone)]
pub struct ModelEngine {
    shared: Arc<EngineShared>,
}

impl ModelEngine {
    /// Loads and compiles one model before making the engine available.
    ///
    /// # Errors
    ///
    /// Returns configuration, checkpoint, compiler, CUDA, or worker startup
    /// failures without exposing a partially initialized engine.
    pub fn start(config: ModelEngineConfig) -> Result<Self, ModelEngineError> {
        config.validate()?;
        let maximum_model_tokens = config.maximum_model_tokens;
        let maximum_prefill_tokens = config.maximum_prefill_tokens;
        let maximum_total_requests = config
            .maximum_active_requests
            .checked_add(config.maximum_queued_requests)
            .ok_or(ModelEngineError::InvalidConfig)?;
        let event_buffer_size = config.event_buffer_size;
        let (commands, receiver) = mpsc::sync_channel(config.maximum_queued_requests);
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        let registry = Arc::new(Mutex::new(RequestRegistry {
            accepting: true,
            cancellations: BTreeMap::new(),
        }));
        let worker_registry = Arc::clone(&registry);
        let shutdown = Arc::new(AtomicBool::new(false));
        let worker_shutdown = Arc::clone(&shutdown);
        let worker = std::thread::Builder::new()
            .name("orbitkv-model-engine".into())
            .spawn(move || match ModelWorker::initialize(&config) {
                Ok(mut worker) => {
                    let _ = ready_tx.send(Ok(()));
                    worker.run(&receiver, &worker_registry, &worker_shutdown);
                }
                Err(error) => {
                    let _ = ready_tx.send(Err(error));
                }
            })
            .map_err(|error| ModelEngineError::Initialization(error.to_string()))?;
        match ready_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                shared: Arc::new(EngineShared {
                    commands,
                    registry,
                    shutdown,
                    worker: Mutex::new(Some(worker)),
                    maximum_model_tokens,
                    maximum_prefill_tokens,
                    maximum_total_requests,
                    event_buffer_size,
                }),
            }),
            Ok(Err(error)) => {
                let _ = worker.join();
                Err(error)
            }
            Err(error) => {
                let _ = worker.join();
                Err(ModelEngineError::Initialization(error.to_string()))
            }
        }
    }

    /// Returns a consistent manager census from the worker thread.
    ///
    /// # Errors
    ///
    /// Returns an error when the worker has stopped.
    pub async fn stats(&self) -> Result<EngineStats, ModelEngineError> {
        let (reply, response) = oneshot::channel();
        match self
            .shared
            .commands
            .try_send(WorkerCommand::Stats { reply })
        {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => return Err(ModelEngineError::QueueFull),
            Err(TrySendError::Disconnected(_)) => {
                return Err(ModelEngineError::WorkerUnavailable);
            }
        }
        response
            .await
            .map_err(|_| ModelEngineError::WorkerUnavailable)
    }
}

impl Engine for ModelEngine {
    type Error = ModelEngineError;

    fn execute(&self, batch: BatchIntent) -> EngineFuture<'_, Self::Error> {
        let shared = Arc::clone(&self.shared);
        Box::pin(async move {
            let request = validate_batch(
                batch,
                shared.maximum_model_tokens,
                shared.maximum_prefill_tokens,
            )?;
            let request_id = request.request_id;
            let cancelled = Arc::new(AtomicBool::new(false));
            {
                let mut registry = shared
                    .registry
                    .lock()
                    .map_err(|_| ModelEngineError::WorkerUnavailable)?;
                if !registry.accepting {
                    return Err(ModelEngineError::WorkerUnavailable);
                }
                if registry.cancellations.len() >= shared.maximum_total_requests {
                    return Err(ModelEngineError::QueueFull);
                }
                match registry.cancellations.entry(request_id) {
                    Entry::Vacant(entry) => {
                        entry.insert(Arc::clone(&cancelled));
                    }
                    Entry::Occupied(_) => return Err(ModelEngineError::DuplicateRequest),
                }
                let (output, receiver) = async_mpsc::channel(shared.event_buffer_size);
                match shared.commands.try_send(WorkerCommand::Run {
                    request,
                    output,
                    cancelled,
                }) {
                    Ok(()) => {}
                    Err(TrySendError::Full(_)) => {
                        registry.cancellations.remove(&request_id);
                        return Err(ModelEngineError::QueueFull);
                    }
                    Err(TrySendError::Disconnected(_)) => {
                        registry.cancellations.remove(&request_id);
                        registry.accepting = false;
                        return Err(ModelEngineError::WorkerUnavailable);
                    }
                }
                drop(registry);
                let events = stream::unfold(receiver, |mut receiver| async move {
                    receiver.recv().await.map(|event| (event, receiver))
                });
                Ok(Box::pin(events) as EngineEventStream<Self::Error>)
            }
        })
    }

    fn abort(&self, request_id: RequestId) -> EngineAbortFuture<'_, Self::Error> {
        let registry = Arc::clone(&self.shared.registry);
        Box::pin(async move {
            let registry = registry
                .lock()
                .map_err(|_| ModelEngineError::WorkerUnavailable)?;
            if let Some(cancelled) = registry.cancellations.get(&request_id) {
                cancelled.store(true, Ordering::Release);
            }
            Ok(())
        })
    }
}

struct ModelWorker {
    session: RuntimeSession,
    plan: ExecutorPlan,
    arenas: Box<[ExecutorArena]>,
    decoder: CompiledDecoder,
    completion_value: u64,
    maximum_active_requests: usize,
    maximum_batch_tokens: usize,
    batch_wait_timeout: Duration,
    counters: SchedulerCounters,
}

#[derive(Default)]
struct SchedulerCounters {
    admitted_requests: u64,
    completed_requests: u64,
    model_dispatches: u64,
    multi_request_dispatches: u64,
    mixed_phase_dispatches: u64,
    maximum_observed_batch_size: u64,
}

struct QueuedRequest {
    request: RequestIntent,
    output: async_mpsc::Sender<Result<EngineEvent, ModelEngineError>>,
    cancelled: Arc<AtomicBool>,
}

struct ActiveRequest {
    request: RequestIntent,
    output: async_mpsc::Sender<Result<EngineEvent, ModelEngineError>>,
    cancelled: Arc<AtomicBool>,
    boundary: u64,
    next_token: Option<u32>,
    generated_tokens: u32,
}

struct DispatchInput {
    active_indices: Vec<usize>,
    request_ids: Vec<EngineRequestId>,
    target_boundaries: Vec<u64>,
    tokens: Vec<u32>,
    positions: Vec<u32>,
}

impl ModelWorker {
    fn initialize(config: &ModelEngineConfig) -> Result<Self, ModelEngineError> {
        let config_bytes = std::fs::read(config.model_directory.join("config.json"))
            .map_err(|error| ModelEngineError::Initialization(error.to_string()))?;
        let decoder_config = DecoderConfig::from_json(&config_bytes)
            .map_err(|error| ModelEngineError::Initialization(error.to_string()))?;
        let manifest = compile_hf_runtime_manifest(
            &config_bytes,
            HfRetentionOptions {
                page_tokens: config.page_tokens,
                kv_dtype_bytes: 2,
            },
        )
        .map_err(|error| ModelEngineError::Initialization(error.to_string()))?;
        let token_plan = manifest
            .attention_state_plan
            .as_ref()
            .ok_or(ModelEngineError::InvalidConfig)?
            .token_manager_plan()
            .map_err(|error| ModelEngineError::Initialization(error.to_string()))?;
        let manager_plan = orbitkv::compile_plan(token_plan)
            .map_err(|error| ModelEngineError::Initialization(error.to_string()))?;
        validate_page_budgets(&manager_plan, config)?;
        let registrations = registrations(&config.page_counts)?;
        let manager = CanonicalKvManager::new(
            &manager_plan,
            ManagerConfig {
                maximum_requests: u32::try_from(config.maximum_active_requests)
                    .map_err(|_| ModelEngineError::InvalidConfig)?,
                maximum_operations: u32::try_from(config.maximum_active_requests)
                    .map_err(|_| ModelEngineError::InvalidConfig)?,
                maximum_prefixes: 1,
                maximum_reclamations: config.page_counts.iter().try_fold(
                    0_u32,
                    |total, pages| {
                        total
                            .checked_add(*pages)
                            .ok_or(ModelEngineError::InvalidConfig)
                    },
                )?,
                maximum_step_tokens: u32::try_from(config.maximum_prefill_tokens)
                    .map_err(|_| ModelEngineError::InvalidConfig)?,
            },
            &registrations,
        )
        .map_err(|error| ModelEngineError::Initialization(error.to_string()))?;
        let session = RuntimeSession::new(manager, CacheSharingPolicy::RequestPrivate);
        let arenas = session
            .arena_stats()
            .iter()
            .copied()
            .zip(registrations)
            .map(|(stats, registration)| ExecutorArena::bind(stats, registration))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| ModelEngineError::Initialization(error.to_string()))?
            .into_boxed_slice();
        let plan = ExecutorPlan::compile(&manifest)
            .map_err(|error| ModelEngineError::Initialization(error.to_string()))?;
        let weights = checkpoint_weights(&config.model_directory)?;
        let decoder = CompiledDecoder::compile_on_device(
            &decoder_config,
            &plan,
            &arenas,
            config.device_index,
            &weights,
            DecoderCompileConfig {
                maximum_query_tokens: config.maximum_batch_tokens,
                representative_prefill_tokens: config.representative_prefill_tokens,
                maximum_batch_size: config.maximum_active_requests,
                maximum_context_pages: config
                    .page_counts
                    .iter()
                    .copied()
                    .max()
                    .and_then(|pages| usize::try_from(pages).ok())
                    .ok_or(ModelEngineError::InvalidConfig)?,
                representative_context_pages: config
                    .representative_prefill_tokens
                    .div_ceil(
                        usize::try_from(config.page_tokens)
                            .map_err(|_| ModelEngineError::InvalidConfig)?,
                    )
                    .max(1),
                search_graphs: config.search_graphs,
                search_seed: config.search_seed,
            },
        )
        .map_err(|error| ModelEngineError::Initialization(error.to_string()))?;
        Ok(Self {
            session,
            plan,
            arenas,
            decoder,
            completion_value: 1,
            maximum_active_requests: config.maximum_active_requests,
            maximum_batch_tokens: config.maximum_batch_tokens,
            batch_wait_timeout: config.batch_wait_timeout,
            counters: SchedulerCounters::default(),
        })
    }

    fn run(
        &mut self,
        receiver: &Receiver<WorkerCommand>,
        registry: &Mutex<RequestRegistry>,
        shutdown_requested: &AtomicBool,
    ) {
        let mut queued = VecDeque::new();
        let mut active = Vec::new();
        let mut shutdown = false;
        loop {
            shutdown |= shutdown_requested.load(Ordering::Acquire);
            if active.is_empty() && queued.is_empty() && !shutdown {
                match receiver.recv() {
                    Ok(command) => {
                        self.handle_command(command, &mut queued, &active, &mut shutdown);
                    }
                    Err(_) => shutdown = true,
                }
                if !shutdown && !queued.is_empty() {
                    self.collect_arrivals(receiver, &mut queued, &active, &mut shutdown);
                }
            }
            self.drain_commands(receiver, &mut queued, &active, &mut shutdown);
            if shutdown {
                for request in &queued {
                    request.cancelled.store(true, Ordering::Release);
                }
                for request in &active {
                    request.cancelled.store(true, Ordering::Release);
                }
            }
            let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
                self.run_cycle(&mut queued, &mut active, registry)
            }))
            .unwrap_or_else(|payload| {
                Err(ModelEngineError::WorkerPanicked(panic_message(&payload)))
            });
            if let Err(error) = result {
                fail_all_requests(&mut queued, &mut active, receiver, registry, &error);
                break;
            }
            if shutdown && queued.is_empty() && active.is_empty() {
                break;
            }
        }
    }

    fn handle_command(
        &self,
        command: WorkerCommand,
        queued: &mut VecDeque<QueuedRequest>,
        active: &[ActiveRequest],
        shutdown: &mut bool,
    ) {
        match command {
            WorkerCommand::Run {
                request,
                output,
                cancelled,
            } => queued.push_back(QueuedRequest {
                request,
                output,
                cancelled,
            }),
            WorkerCommand::Stats { reply } => {
                let _ = reply.send(self.stats(queued.len(), active.len()));
            }
            WorkerCommand::Shutdown => *shutdown = true,
        }
    }

    fn collect_arrivals(
        &self,
        receiver: &Receiver<WorkerCommand>,
        queued: &mut VecDeque<QueuedRequest>,
        active: &[ActiveRequest],
        shutdown: &mut bool,
    ) {
        let deadline = Instant::now() + self.batch_wait_timeout;
        while active.len() + queued.len() < self.maximum_active_requests {
            let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                break;
            };
            match receiver.recv_timeout(remaining) {
                Ok(command) => self.handle_command(command, queued, active, shutdown),
                Err(mpsc::RecvTimeoutError::Timeout) => break,
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    *shutdown = true;
                    break;
                }
            }
            if *shutdown {
                break;
            }
        }
    }

    fn drain_commands(
        &self,
        receiver: &Receiver<WorkerCommand>,
        queued: &mut VecDeque<QueuedRequest>,
        active: &[ActiveRequest],
        shutdown: &mut bool,
    ) {
        loop {
            match receiver.try_recv() {
                Ok(command) => self.handle_command(command, queued, active, shutdown),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    *shutdown = true;
                    break;
                }
            }
        }
    }

    fn run_cycle(
        &mut self,
        queued: &mut VecDeque<QueuedRequest>,
        active: &mut Vec<ActiveRequest>,
        registry: &Mutex<RequestRegistry>,
    ) -> Result<(), ModelEngineError> {
        self.finish_cancelled_queued(queued, registry);
        self.admit(queued, active, registry)?;
        let cancelled = active
            .iter()
            .enumerate()
            .filter(|(_, request)| {
                request.output.is_closed() || request.cancelled.load(Ordering::Acquire)
            })
            .map(|(index, _)| (index, FinishReason::Cancelled))
            .collect::<Vec<_>>();
        self.finish_active(active, cancelled, registry)?;
        if active.is_empty() {
            return Ok(());
        }
        let dispatch = build_dispatch(active, self.maximum_batch_tokens)?;
        if dispatch.active_indices.is_empty() {
            std::thread::sleep(Duration::from_micros(100));
            return Ok(());
        }
        let sampled = self.execute_batch(&dispatch)?;
        self.counters.model_dispatches += 1;
        if dispatch.active_indices.len() > 1 {
            self.counters.multi_request_dispatches += 1;
        }
        let contains_prefill = dispatch
            .active_indices
            .iter()
            .any(|&index| active[index].next_token.is_none());
        let contains_decode = dispatch
            .active_indices
            .iter()
            .any(|&index| active[index].next_token.is_some());
        if contains_prefill && contains_decode {
            self.counters.mixed_phase_dispatches += 1;
        }
        self.counters.maximum_observed_batch_size = self
            .counters
            .maximum_observed_batch_size
            .max(dispatch.active_indices.len() as u64);
        let mut finished = Vec::new();
        for ((&active_index, &target_boundary), token) in dispatch
            .active_indices
            .iter()
            .zip(&dispatch.target_boundaries)
            .zip(sampled)
        {
            let request = &mut active[active_index];
            request.boundary = target_boundary;
            request.next_token = None;
            let reason = if request.cancelled.load(Ordering::Acquire) || request.output.is_closed()
            {
                Some(FinishReason::Cancelled)
            } else if request.request.sampling.stop_token_ids.contains(&token) {
                Some(FinishReason::Stop { token_id: token })
            } else if request
                .output
                .try_send(Ok(EngineEvent::Token(TokenOutput {
                    request_id: request.request.request_id,
                    token_id: token,
                })))
                .is_err()
            {
                Some(FinishReason::Cancelled)
            } else {
                request.generated_tokens += 1;
                if request.generated_tokens == request.request.sampling.max_output_tokens {
                    Some(FinishReason::Length)
                } else {
                    request.next_token = Some(token);
                    None
                }
            };
            if let Some(reason) = reason {
                finished.push((active_index, reason));
            }
        }
        self.finish_active(active, finished, registry)
    }

    fn admit(
        &mut self,
        queued: &mut VecDeque<QueuedRequest>,
        active: &mut Vec<ActiveRequest>,
        registry: &Mutex<RequestRegistry>,
    ) -> Result<(), ModelEngineError> {
        let count = self
            .maximum_active_requests
            .saturating_sub(active.len())
            .min(queued.len());
        if count == 0 {
            return Ok(());
        }
        let mut admitted = Vec::new();
        for _ in 0..count {
            let request = queued.pop_front().expect("admission count was bounded");
            if request.cancelled.load(Ordering::Acquire) || request.output.is_closed() {
                finish_unacquired(request, registry);
                continue;
            }
            admitted.push(request);
        }
        let ids = admitted
            .iter()
            .map(|request| EngineRequestId(request.request.request_id.0))
            .collect::<Vec<_>>();
        if !ids.is_empty()
            && let Err(error) = self.session.acquire_requests(&ids)
        {
            for request in admitted.into_iter().rev() {
                queued.push_front(request);
            }
            return Err(lifecycle_error(error));
        }
        self.counters.admitted_requests += admitted.len() as u64;
        active.extend(admitted.into_iter().map(|queued| {
            if queued
                .output
                .try_send(Ok(EngineEvent::BatchStarted {
                    request_ids: vec![queued.request.request_id].into_boxed_slice(),
                }))
                .is_err()
            {
                queued.cancelled.store(true, Ordering::Release);
            }
            ActiveRequest {
                boundary: queued.request.target_boundary,
                request: queued.request,
                output: queued.output,
                cancelled: queued.cancelled,
                next_token: None,
                generated_tokens: 0,
            }
        }));
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn execute_batch(&mut self, dispatch: &DispatchInput) -> Result<Vec<u32>, ModelEngineError> {
        let intents = dispatch
            .request_ids
            .iter()
            .copied()
            .zip(&dispatch.target_boundaries)
            .map(|(request_id, &target_boundary)| EngineAppendIntent {
                request_id,
                target_boundary,
            })
            .collect::<Vec<_>>();
        let source = self
            .session
            .prepare_append_batch(&intents)
            .map_err(lifecycle_error)?;
        let batch_id = source.batch_id;
        let lowered = (|| {
            let view = self
                .session
                .prepared_execution_view(batch_id)
                .map_err(lifecycle_error)?;
            let attention = self
                .plan
                .attention_batches(&view)
                .map_err(|error| ModelEngineError::Executor(error.to_string()))?;
            let prepared = self
                .plan
                .lower_prepared(source, &self.arenas)
                .map_err(|error| ModelEngineError::Executor(error.to_string()))?;
            Ok::<_, ModelEngineError>((attention, prepared))
        })();
        let (attention, prepared) = match lowered {
            Ok(lowered) => lowered,
            Err(error) => {
                self.abort_prepared(batch_id, &dispatch.request_ids)?;
                return Err(error);
            }
        };
        if prepared
            .steps()
            .iter()
            .flat_map(|step| &step.classes)
            .any(|class| !class.copies.is_empty())
        {
            self.abort_prepared(batch_id, &dispatch.request_ids)?;
            return Err(ModelEngineError::Executor(
                "append copy execution is not implemented by the model coordinator".into(),
            ));
        }
        let write_slots = match decoder_class_write_slots(prepared.steps(), attention.len()) {
            Ok(write_slots) => write_slots,
            Err(error) => {
                self.abort_prepared(batch_id, &dispatch.request_ids)?;
                return Err(error);
            }
        };
        let classes = decoder_class_steps(&write_slots, &attention);
        let executed = self.decoder.execute(DecoderStep {
            tokens: &dispatch.tokens,
            positions: &dispatch.positions,
            classes: &classes,
        });
        let sampled = match executed {
            Ok(result) => match sampled_rows(&result.token_ids, &attention[0].query_indptr) {
                Ok(sampled) if sampled.len() == dispatch.active_indices.len() => sampled,
                Ok(_) => {
                    self.session
                        .quarantine_prepared_execution(batch_id)
                        .map_err(lifecycle_error)?;
                    return Err(ModelEngineError::Executor(
                        "sampled-token row count does not match the dispatched batch".into(),
                    ));
                }
                Err(error) => {
                    self.session
                        .quarantine_prepared_execution(batch_id)
                        .map_err(lifecycle_error)?;
                    return Err(error);
                }
            },
            Err(error) => {
                self.session
                    .quarantine_prepared_execution(batch_id)
                    .map_err(lifecycle_error)?;
                return Err(ModelEngineError::Executor(error.to_string()));
            }
        };
        let evidence = match prepared.execution_evidence_after_success(&self.arenas) {
            Ok(evidence) => evidence,
            Err(error) => {
                let _ = self.session.quarantine_prepared_execution(batch_id);
                return Err(ModelEngineError::Executor(error.to_string()));
            }
        };
        let ticket = match self.session.submit_execution(&evidence) {
            Ok(ticket) => ticket,
            Err(error) => {
                let _ = self.session.quarantine_prepared_execution(batch_id);
                return Err(lifecycle_error(error));
            }
        };
        let publication = match self.session.complete_execution_by_batch(
            ticket.batch_id(),
            EngineCompletionEvidence {
                completion_domain: 1,
                completion_value: self.completion_value,
                confirmed: true,
            },
        ) {
            Ok(publication) => publication,
            Err(error) => {
                let _ = self.session.quarantine_submitted_execution(batch_id);
                return Err(lifecycle_error(error));
            }
        };
        self.completion_value = self
            .completion_value
            .checked_add(1)
            .ok_or_else(|| ModelEngineError::Lifecycle("completion frontier exhausted".into()))?;
        self.session
            .confirm_publication(&EnginePublicationEvidence {
                publication_id: publication.publication_id,
                mirror_cleanup_confirmed: true,
                reclamation_receipts: retirement_evidence(&publication.retirements),
            })
            .map_err(lifecycle_error)?;
        Ok(sampled)
    }

    fn abort_prepared(
        &mut self,
        batch_id: orbitkv::EngineBatchId,
        request_ids: &[EngineRequestId],
    ) -> Result<(), ModelEngineError> {
        let evidence = request_ids
            .iter()
            .copied()
            .map(|request_id| orbitkv::EngineStepAbortEvidence {
                request_id,
                backend_unobserved: true,
            })
            .collect::<Vec<_>>();
        self.session
            .abort_prepared_execution(batch_id, &evidence)
            .map_err(lifecycle_error)
    }

    fn finish_active(
        &mut self,
        active: &mut Vec<ActiveRequest>,
        mut finished: Vec<(usize, FinishReason)>,
        registry: &Mutex<RequestRegistry>,
    ) -> Result<(), ModelEngineError> {
        if finished.is_empty() {
            return Ok(());
        }
        finished.sort_by_key(|(index, _)| *index);
        finished.dedup_by_key(|(index, _)| *index);
        let ids = finished
            .iter()
            .map(|(index, _)| EngineRequestId(active[*index].request.request_id.0))
            .collect::<Vec<_>>();
        let release = self
            .session
            .prepare_release_batch(&ids)
            .map_err(lifecycle_error)?;
        let outcome = self
            .session
            .confirm_release(&EngineReleaseEvidence {
                release_id: release.release_id,
                mirror_cleanup_confirmed: true,
                reclamation_receipts: retirement_evidence(&release.retirements),
            })
            .map_err(lifecycle_error)?;
        if outcome != EngineReleaseOutcome::Completed {
            return Err(ModelEngineError::Lifecycle(
                "request release requires a retry".into(),
            ));
        }
        for (index, reason) in finished.into_iter().rev() {
            let request = active.remove(index);
            let _ = request.output.try_send(Ok(EngineEvent::Finished {
                request_id: request.request.request_id,
                reason,
            }));
            remove_registry(registry, request.request.request_id);
            self.counters.completed_requests += 1;
        }
        Ok(())
    }

    fn finish_cancelled_queued(
        &mut self,
        queued: &mut VecDeque<QueuedRequest>,
        registry: &Mutex<RequestRegistry>,
    ) {
        let mut retained = VecDeque::new();
        while let Some(request) = queued.pop_front() {
            if request.cancelled.load(Ordering::Acquire) || request.output.is_closed() {
                finish_unacquired(request, registry);
                self.counters.completed_requests += 1;
            } else {
                retained.push_back(request);
            }
        }
        *queued = retained;
    }

    fn stats(&self, queued_requests: usize, active_requests: usize) -> EngineStats {
        EngineStats {
            manager: self.session.stats(),
            queued_requests: queued_requests as u64,
            active_requests: active_requests as u64,
            admitted_requests: self.counters.admitted_requests,
            completed_requests: self.counters.completed_requests,
            model_dispatches: self.counters.model_dispatches,
            multi_request_dispatches: self.counters.multi_request_dispatches,
            mixed_phase_dispatches: self.counters.mixed_phase_dispatches,
            maximum_observed_batch_size: self.counters.maximum_observed_batch_size,
        }
    }
}

fn fail_all_requests(
    queued: &mut VecDeque<QueuedRequest>,
    active: &mut Vec<ActiveRequest>,
    receiver: &Receiver<WorkerCommand>,
    registry: &Mutex<RequestRegistry>,
    error: &ModelEngineError,
) {
    for request in queued.drain(..) {
        let _ = request.output.try_send(Err(error.clone()));
    }
    for request in active.drain(..) {
        let _ = request.output.try_send(Err(error.clone()));
    }
    if let Ok(mut registry) = registry.lock() {
        registry.accepting = false;
        registry.cancellations.clear();
        while let Ok(command) = receiver.try_recv() {
            if let WorkerCommand::Run { output, .. } = command {
                let _ = output.try_send(Err(ModelEngineError::WorkerUnavailable));
            }
        }
    }
}

fn finish_unacquired(request: QueuedRequest, registry: &Mutex<RequestRegistry>) {
    let QueuedRequest {
        request,
        output,
        cancelled: _,
    } = request;
    let _ = output.try_send(Ok(EngineEvent::BatchStarted {
        request_ids: vec![request.request_id].into_boxed_slice(),
    }));
    let _ = output.try_send(Ok(EngineEvent::Finished {
        request_id: request.request_id,
        reason: FinishReason::Cancelled,
    }));
    remove_registry(registry, request.request_id);
}

fn build_dispatch(
    active: &[ActiveRequest],
    maximum_batch_tokens: usize,
) -> Result<DispatchInput, ModelEngineError> {
    let mut dispatch = DispatchInput {
        active_indices: Vec::new(),
        request_ids: Vec::new(),
        target_boundaries: Vec::new(),
        tokens: Vec::new(),
        positions: Vec::new(),
    };
    // Decode-first scheduling bounds TPOT under prefill arrivals. Remaining
    // token budget is filled by waiting prefills in stable admission order.
    for pending_prefill in [false, true] {
        for (index, request) in active.iter().enumerate() {
            if request.next_token.is_none() != pending_prefill || request.output.capacity() < 2 {
                continue;
            }
            let query_tokens = request
                .next_token
                .map_or(request.request.input_tokens.len(), |_| 1);
            if dispatch.tokens.len() + query_tokens > maximum_batch_tokens {
                continue;
            }
            let target_boundary = if request.next_token.is_some() {
                request
                    .boundary
                    .checked_add(1)
                    .ok_or(ModelEngineError::ModelLength)?
            } else {
                request.boundary
            };
            dispatch.active_indices.push(index);
            dispatch
                .request_ids
                .push(EngineRequestId(request.request.request_id.0));
            dispatch.target_boundaries.push(target_boundary);
            if let Some(token) = request.next_token {
                dispatch.tokens.push(token);
                dispatch.positions.push(
                    u32::try_from(request.boundary).map_err(|_| ModelEngineError::ModelLength)?,
                );
            } else {
                dispatch.tokens.extend(&request.request.input_tokens);
                let begin = target_boundary - request.request.input_tokens.len() as u64;
                dispatch.positions.extend(
                    (begin..target_boundary)
                        .map(|position| {
                            u32::try_from(position).map_err(|_| ModelEngineError::ModelLength)
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                );
            }
        }
    }
    Ok(dispatch)
}

fn remove_registry(registry: &Mutex<RequestRegistry>, request_id: RequestId) {
    if let Ok(mut registry) = registry.lock() {
        registry.cancellations.remove(&request_id);
    }
}

fn registrations(page_counts: &[u32]) -> Result<Vec<BackendArenaRegistration>, ModelEngineError> {
    page_counts
        .iter()
        .copied()
        .enumerate()
        .map(|(index, page_count)| {
            let identity = u32::try_from(index + 1).map_err(|_| ModelEngineError::InvalidConfig)?;
            Ok(BackendArenaRegistration {
                pool_id: identity,
                class_id: u16::try_from(index).map_err(|_| ModelEngineError::InvalidConfig)?,
                backend_domain: u16::try_from(index + 1)
                    .map_err(|_| ModelEngineError::InvalidConfig)?,
                page_count,
                reserved: 0,
                backend_base_index: 0,
            })
        })
        .collect()
}

fn validate_page_budgets(
    plan: &orbitkv::plan::CompiledKvPlan,
    config: &ModelEngineConfig,
) -> Result<(), ModelEngineError> {
    if plan.classes.len() != config.page_counts.len() {
        return Err(ModelEngineError::InvalidConfig);
    }
    let requests = u64::try_from(config.maximum_active_requests)
        .map_err(|_| ModelEngineError::InvalidConfig)?;
    let maximum_pages = config
        .maximum_model_tokens
        .div_ceil(config.page_tokens)
        .checked_mul(requests)
        .ok_or(ModelEngineError::InvalidConfig)?;
    for (class, &available) in plan.classes.iter().zip(&config.page_counts) {
        let required = class
            .slot_count
            .and_then(|slots| slots.checked_mul(requests))
            .map_or(maximum_pages, |slots| slots.min(maximum_pages));
        if u64::from(available) < required {
            return Err(ModelEngineError::InvalidConfig);
        }
    }
    Ok(())
}

fn validate_batch(
    batch: BatchIntent,
    maximum_model_tokens: u64,
    maximum_prefill_tokens: usize,
) -> Result<RequestIntent, ModelEngineError> {
    if batch.requests.len() != 1 {
        return Err(ModelEngineError::BatchSize);
    }
    let batch = BatchIntent::new(batch.requests)
        .map_err(|error| ModelEngineError::InvalidIntent(error.to_string()))?;
    let request = batch.requests.into_vec().pop().unwrap();
    if request.input_tokens.len() > maximum_prefill_tokens {
        return Err(ModelEngineError::ModelLength);
    }
    if request.target_boundary != u64::try_from(request.input_tokens.len()).unwrap_or(0) {
        return Err(ModelEngineError::ContinuationUnsupported);
    }
    if request
        .target_boundary
        .checked_add(u64::from(request.sampling.max_output_tokens - 1))
        .is_none_or(|boundary| boundary > maximum_model_tokens)
    {
        return Err(ModelEngineError::ModelLength);
    }
    Ok(request)
}

fn checkpoint_weights(directory: &Path) -> Result<Vec<PathBuf>, ModelEngineError> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(directory)
        .map_err(|error| ModelEngineError::Initialization(error.to_string()))?
    {
        let path = entry
            .map_err(|error| ModelEngineError::Initialization(error.to_string()))?
            .path();
        if path
            .extension()
            .is_some_and(|extension| extension == "safetensors")
        {
            files.push(path);
        }
    }
    files.sort();
    if files.is_empty() {
        return Err(ModelEngineError::Initialization(
            "checkpoint has no safetensors weights".into(),
        ));
    }
    Ok(files)
}

fn decoder_class_write_slots(
    steps: &[orbitkv_executor::PreparedStep],
    class_count: usize,
) -> Result<Vec<Vec<u64>>, ModelEngineError> {
    let mut slots = vec![Vec::new(); class_count];
    for step in steps {
        if step.classes.len() != class_count {
            return Err(ModelEngineError::Executor(
                "prepared class count does not match attention metadata".into(),
            ));
        }
        for (class_id, class) in step.classes.iter().enumerate() {
            if usize::from(class.class_id) != class_id {
                return Err(ModelEngineError::Executor(
                    "prepared classes are not in canonical order".into(),
                ));
            }
            slots[class_id].extend(&class.write_slots);
        }
    }
    Ok(slots)
}

fn decoder_class_steps<'a>(
    write_slots: &'a [Vec<u64>],
    attention: &'a [AttentionBatch],
) -> Vec<DecoderClassStep<'a>> {
    write_slots
        .iter()
        .zip(attention)
        .map(|(write_slots, attention)| DecoderClassStep {
            class_id: attention.class_id,
            write_slots,
            attention,
        })
        .collect()
}

fn sampled_rows(token_ids: &[u32], query_indptr: &[i32]) -> Result<Vec<u32>, ModelEngineError> {
    query_indptr
        .windows(2)
        .map(|row| {
            let start = usize::try_from(row[0])
                .ok()
                .filter(|&start| start < token_ids.len())
                .ok_or_else(|| ModelEngineError::Executor("invalid sampled-token rows".into()))?;
            let end = usize::try_from(row[1])
                .ok()
                .filter(|&end| end > start && end <= token_ids.len())
                .ok_or_else(|| ModelEngineError::Executor("invalid sampled-token rows".into()))?;
            Ok(token_ids[end - 1])
        })
        .collect()
}

fn retirement_evidence(
    retirements: &[orbitkv::EngineRetirement],
) -> Box<[EngineRetirementEvidence]> {
    retirements
        .iter()
        .map(|retirement| EngineRetirementEvidence {
            page: retirement.page,
            backend_domain: retirement.backend_domain,
            acknowledged: true,
            backend_index: retirement.backend_index,
        })
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

fn lifecycle_error(error: impl std::fmt::Display) -> ModelEngineError {
    ModelEngineError::Lifecycle(error.to_string())
}

fn panic_message(payload: &Box<dyn Any + Send>) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| {
            payload
                .downcast_ref::<&str>()
                .map(|message| (*message).to_string())
        })
        .unwrap_or_else(|| "unknown panic payload".into())
}

#[cfg(test)]
mod tests;
