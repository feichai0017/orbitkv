use std::collections::{BTreeMap, btree_map::Entry};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
    mpsc,
};
use std::thread::JoinHandle;
use std::{any::Any, panic::AssertUnwindSafe};

use futures_util::stream;
use orbitkv::{
    CacheSharingPolicy, EngineAppendIntent, EngineCompletionEvidence, EnginePublicationEvidence,
    EngineReleaseEvidence, EngineReleaseOutcome, EngineRequestId, EngineRetirementEvidence,
    HfRetentionOptions, RuntimeSession, compile_hf_runtime_manifest,
    kv_manager::{BackendArenaRegistration, CanonicalKvManager, ManagerConfig, ManagerStats},
};
use orbitkv_executor::{
    AttentionBatch, ExecutorArena, ExecutorPlan, PreparedBatch,
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
    pub representative_prefill_tokens: usize,
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
            || !(2..=self.maximum_prefill_tokens).contains(&self.representative_prefill_tokens)
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
    #[error("the serial model engine accepts exactly one request per batch")]
    BatchSize,
    #[error("the serial model engine currently accepts only a fresh prompt")]
    ContinuationUnsupported,
    #[error("invalid request intent: {0}")]
    InvalidIntent(String),
    #[error("request exceeds the configured model-token capacity")]
    ModelLength,
    #[error("request is already queued or active")]
    DuplicateRequest,
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
        output: async_mpsc::UnboundedSender<Result<EngineEvent, ModelEngineError>>,
        cancelled: Arc<AtomicBool>,
    },
    Stats {
        reply: oneshot::Sender<ManagerStats>,
    },
    Shutdown,
}

struct EngineShared {
    commands: mpsc::Sender<WorkerCommand>,
    registry: Arc<Mutex<RequestRegistry>>,
    worker: Mutex<Option<JoinHandle<()>>>,
    maximum_model_tokens: u64,
    maximum_prefill_tokens: usize,
}

struct RequestRegistry {
    accepting: bool,
    cancellations: BTreeMap<RequestId, Arc<AtomicBool>>,
}

impl Drop for EngineShared {
    fn drop(&mut self) {
        if let Ok(mut registry) = self.registry.lock() {
            registry.accepting = false;
            for cancelled in registry.cancellations.values() {
                cancelled.store(true, Ordering::Release);
            }
        }
        let _ = self.commands.send(WorkerCommand::Shutdown);
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
        let (commands, receiver) = mpsc::channel();
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        let registry = Arc::new(Mutex::new(RequestRegistry {
            accepting: true,
            cancellations: BTreeMap::new(),
        }));
        let worker_registry = Arc::clone(&registry);
        let worker = std::thread::Builder::new()
            .name("orbitkv-model-engine".into())
            .spawn(move || match ModelWorker::initialize(&config) {
                Ok(mut worker) => {
                    let _ = ready_tx.send(Ok(()));
                    worker.run(&receiver, &worker_registry);
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
                    worker: Mutex::new(Some(worker)),
                    maximum_model_tokens,
                    maximum_prefill_tokens,
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
    pub async fn stats(&self) -> Result<ManagerStats, ModelEngineError> {
        let (reply, response) = oneshot::channel();
        self.shared
            .commands
            .send(WorkerCommand::Stats { reply })
            .map_err(|_| ModelEngineError::WorkerUnavailable)?;
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
                match registry.cancellations.entry(request_id) {
                    Entry::Vacant(entry) => {
                        entry.insert(Arc::clone(&cancelled));
                    }
                    Entry::Occupied(_) => return Err(ModelEngineError::DuplicateRequest),
                }
                let (output, receiver) = async_mpsc::unbounded_channel();
                if shared
                    .commands
                    .send(WorkerCommand::Run {
                        request,
                        output,
                        cancelled,
                    })
                    .is_err()
                {
                    registry.cancellations.remove(&request_id);
                    registry.accepting = false;
                    return Err(ModelEngineError::WorkerUnavailable);
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
                maximum_requests: 1,
                maximum_operations: 2,
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
                maximum_query_tokens: config.maximum_prefill_tokens,
                representative_prefill_tokens: config.representative_prefill_tokens,
                maximum_batch_size: 1,
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
        })
    }

    fn run(&mut self, receiver: &mpsc::Receiver<WorkerCommand>, registry: &Mutex<RequestRegistry>) {
        while let Ok(command) = receiver.recv() {
            match command {
                WorkerCommand::Run {
                    request,
                    output,
                    cancelled,
                } => {
                    let request_id = request.request_id;
                    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
                        self.run_request(&request, &output, &cancelled)
                    }))
                    .unwrap_or_else(|payload| {
                        Err(ModelEngineError::WorkerPanicked(panic_message(&payload)))
                    });
                    if let Ok(mut registry) = registry.lock() {
                        registry.cancellations.remove(&request_id);
                    }
                    if let Err(error) = result {
                        let _ = output.send(Err(error));
                        fail_pending_requests(receiver, registry);
                        break;
                    }
                }
                WorkerCommand::Stats { reply } => {
                    let _ = reply.send(self.session.stats());
                }
                WorkerCommand::Shutdown => break,
            }
        }
    }

    fn run_request(
        &mut self,
        request: &RequestIntent,
        output: &async_mpsc::UnboundedSender<Result<EngineEvent, ModelEngineError>>,
        cancelled: &AtomicBool,
    ) -> Result<(), ModelEngineError> {
        if output
            .send(Ok(EngineEvent::BatchStarted {
                request_ids: vec![request.request_id].into_boxed_slice(),
            }))
            .is_err()
        {
            return Ok(());
        }
        let request_id = EngineRequestId(request.request_id.0);
        self.session
            .acquire_requests(&[request_id])
            .map_err(lifecycle_error)?;
        if cancelled.load(Ordering::Acquire) {
            self.release(request_id)?;
            let _ = output.send(Ok(EngineEvent::Finished {
                request_id: request.request_id,
                reason: FinishReason::Cancelled,
            }));
            return Ok(());
        }

        let boundary = request.target_boundary;
        let positions = (boundary - u64::try_from(request.input_tokens.len()).unwrap()..boundary)
            .map(|position| u32::try_from(position).map_err(|_| ModelEngineError::ModelLength))
            .collect::<Result<Vec<_>, _>>()?;
        let generated = self.generate(request, request_id, boundary, &positions, output, cancelled);
        let release = self.release(request_id);
        match (generated, release) {
            (Ok(reason), Ok(())) => {
                let _ = output.send(Ok(EngineEvent::Finished {
                    request_id: request.request_id,
                    reason,
                }));
                Ok(())
            }
            (Err(error), _) | (Ok(_), Err(error)) => Err(error),
        }
    }

    fn generate(
        &mut self,
        request: &RequestIntent,
        request_id: EngineRequestId,
        mut boundary: u64,
        positions: &[u32],
        output: &async_mpsc::UnboundedSender<Result<EngineEvent, ModelEngineError>>,
        cancelled: &AtomicBool,
    ) -> Result<FinishReason, ModelEngineError> {
        let mut token =
            self.execute_step(request_id, boundary, &request.input_tokens, positions)?;
        for generated in 0..request.sampling.max_output_tokens {
            if cancelled.load(Ordering::Acquire) || output.is_closed() {
                return Ok(FinishReason::Cancelled);
            }
            if request.sampling.stop_token_ids.contains(&token) {
                return Ok(FinishReason::Stop { token_id: token });
            }
            if output
                .send(Ok(EngineEvent::Token(TokenOutput {
                    request_id: request.request_id,
                    token_id: token,
                })))
                .is_err()
            {
                return Ok(FinishReason::Cancelled);
            }
            if generated + 1 == request.sampling.max_output_tokens {
                return Ok(FinishReason::Length);
            }
            boundary = boundary
                .checked_add(1)
                .ok_or(ModelEngineError::ModelLength)?;
            token = self.execute_step(
                request_id,
                boundary,
                &[token],
                &[u32::try_from(boundary - 1).map_err(|_| ModelEngineError::ModelLength)?],
            )?;
        }
        unreachable!("sampling validation rejects zero output budgets")
    }

    fn execute_step(
        &mut self,
        request_id: EngineRequestId,
        target_boundary: u64,
        tokens: &[u32],
        positions: &[u32],
    ) -> Result<u32, ModelEngineError> {
        let source = self
            .session
            .prepare_append_batch(&[EngineAppendIntent {
                request_id,
                target_boundary,
            }])
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
                self.abort_prepared(batch_id, request_id)?;
                return Err(error);
            }
        };
        if prepared
            .steps()
            .iter()
            .flat_map(|step| &step.classes)
            .any(|class| !class.copies.is_empty())
        {
            self.abort_prepared(batch_id, request_id)?;
            return Err(ModelEngineError::Executor(
                "append copy execution is not implemented by the serial coordinator".into(),
            ));
        }
        let classes = decoder_class_steps(&prepared, &attention);
        let executed = self.decoder.execute(DecoderStep {
            tokens,
            positions,
            classes: &classes,
        });
        let token = match executed {
            Ok(result) => *result
                .token_ids
                .last()
                .ok_or_else(|| ModelEngineError::Executor("decoder returned no token".into()))?,
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
        Ok(token)
    }

    fn abort_prepared(
        &mut self,
        batch_id: orbitkv::EngineBatchId,
        request_id: EngineRequestId,
    ) -> Result<(), ModelEngineError> {
        self.session
            .abort_prepared_execution(
                batch_id,
                &[orbitkv::EngineStepAbortEvidence {
                    request_id,
                    backend_unobserved: true,
                }],
            )
            .map_err(lifecycle_error)
    }

    fn release(&mut self, request_id: EngineRequestId) -> Result<(), ModelEngineError> {
        let release = self
            .session
            .prepare_release_batch(&[request_id])
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
        Ok(())
    }
}

fn fail_pending_requests(
    receiver: &mpsc::Receiver<WorkerCommand>,
    registry: &Mutex<RequestRegistry>,
) {
    if let Ok(mut registry) = registry.lock() {
        registry.accepting = false;
        registry.cancellations.clear();
        while let Ok(command) = receiver.try_recv() {
            if let WorkerCommand::Run { output, .. } = command {
                let _ = output.send(Err(ModelEngineError::WorkerUnavailable));
            }
        }
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
    let maximum_pages = config.maximum_model_tokens.div_ceil(config.page_tokens);
    for (class, &available) in plan.classes.iter().zip(&config.page_counts) {
        let required = class
            .slot_count
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

fn decoder_class_steps<'a>(
    prepared: &'a PreparedBatch,
    attention: &'a [AttentionBatch],
) -> Vec<DecoderClassStep<'a>> {
    prepared.steps()[0]
        .classes
        .iter()
        .zip(attention)
        .map(|(class, attention)| DecoderClassStep {
            class_id: class.class_id,
            write_slots: &class.write_slots,
            attention,
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
mod tests {
    use super::*;
    use orbitkv::{KvClassSpec, KvPlanInput, TokenStorageKind, plan::RetentionKind};
    use orbitkv_server::SamplingIntent;

    fn config(page_counts: Vec<u32>) -> ModelEngineConfig {
        ModelEngineConfig {
            model_directory: PathBuf::from("/unused"),
            device_index: 0,
            page_tokens: 16,
            page_counts,
            maximum_model_tokens: 1_024,
            maximum_prefill_tokens: 512,
            representative_prefill_tokens: 512,
            search_graphs: 2,
            search_seed: 7,
        }
    }

    fn request(
        request_id: u64,
        input_tokens: &[u32],
        target_boundary: u64,
        output_tokens: u32,
    ) -> BatchIntent {
        BatchIntent {
            requests: vec![RequestIntent {
                request_id: RequestId(request_id),
                input_tokens: input_tokens.into(),
                target_boundary,
                sampling: SamplingIntent::greedy(output_tokens, []),
            }]
            .into_boxed_slice(),
        }
    }

    fn hybrid_plan() -> orbitkv::CompiledKvPlan {
        orbitkv::compile_plan(KvPlanInput {
            page_tokens: 16,
            classes: vec![
                KvClassSpec {
                    name: "full".into(),
                    layers: vec![1],
                    retention: RetentionKind::Full,
                    bytes_per_token_per_layer: 128,
                    window_tokens: None,
                    storage: TokenStorageKind::TokenKv,
                    components: Vec::new(),
                },
                KvClassSpec {
                    name: "sliding".into(),
                    layers: vec![0],
                    retention: RetentionKind::Sliding,
                    bytes_per_token_per_layer: 128,
                    window_tokens: Some(512),
                    storage: TokenStorageKind::TokenKv,
                    components: Vec::new(),
                },
            ],
        })
        .unwrap()
    }

    fn engine_with_channel(
        commands: mpsc::Sender<WorkerCommand>,
    ) -> (ModelEngine, Arc<Mutex<RequestRegistry>>) {
        let registry = Arc::new(Mutex::new(RequestRegistry {
            accepting: true,
            cancellations: BTreeMap::new(),
        }));
        let engine = ModelEngine {
            shared: Arc::new(EngineShared {
                commands,
                registry: Arc::clone(&registry),
                worker: Mutex::new(None),
                maximum_model_tokens: 32,
                maximum_prefill_tokens: 16,
            }),
        };
        (engine, registry)
    }

    #[test]
    fn rejects_invalid_capacity_configuration_without_starting_a_worker() {
        let mut invalid = config(vec![64, 33]);
        invalid.page_tokens = 0;
        assert_eq!(invalid.validate(), Err(ModelEngineError::InvalidConfig));
    }

    #[test]
    fn validates_each_compiled_class_page_budget_independently() {
        let plan = hybrid_plan();
        assert_eq!(validate_page_budgets(&plan, &config(vec![64, 33])), Ok(()));
        assert_eq!(
            validate_page_budgets(&plan, &config(vec![64, 32])),
            Err(ModelEngineError::InvalidConfig)
        );
        assert_eq!(
            validate_page_budgets(&plan, &config(vec![64])),
            Err(ModelEngineError::InvalidConfig)
        );
    }

    #[test]
    fn validates_serial_fresh_prompt_and_model_length_contract() {
        let accepted = validate_batch(request(1, &[10, 11], 2, 3), 4, 2).unwrap();
        assert_eq!(accepted.request_id, RequestId(1));

        assert_eq!(
            validate_batch(request(2, &[10], 2, 1), 4, 2),
            Err(ModelEngineError::ContinuationUnsupported)
        );
        assert_eq!(
            validate_batch(request(3, &[10, 11], 2, 4), 4, 2),
            Err(ModelEngineError::ModelLength)
        );
        assert_eq!(
            validate_batch(request(6, &[10, 11, 12], 3, 1), 4, 2),
            Err(ModelEngineError::ModelLength)
        );
        assert_eq!(
            validate_batch(
                BatchIntent {
                    requests: vec![
                        request(4, &[10], 1, 1).requests[0].clone(),
                        request(5, &[11], 1, 1).requests[0].clone(),
                    ]
                    .into_boxed_slice(),
                },
                4,
                2,
            ),
            Err(ModelEngineError::BatchSize)
        );
    }

    #[tokio::test]
    async fn duplicate_submission_preserves_the_active_cancellation_handle() {
        let (commands, _receiver) = mpsc::channel();
        let (engine, registry) = engine_with_channel(commands);
        let active = Arc::new(AtomicBool::new(false));
        registry
            .lock()
            .unwrap()
            .cancellations
            .insert(RequestId(7), Arc::clone(&active));

        let result = engine.execute(request(7, &[10], 1, 1)).await;
        assert!(matches!(result, Err(ModelEngineError::DuplicateRequest)));
        let registry = registry.lock().unwrap();
        assert!(Arc::ptr_eq(
            registry.cancellations.get(&RequestId(7)).unwrap(),
            &active
        ));
    }

    #[tokio::test]
    async fn failed_worker_send_removes_the_request_registry_entry() {
        let (commands, receiver) = mpsc::channel();
        drop(receiver);
        let (engine, registry) = engine_with_channel(commands);

        let result = engine.execute(request(8, &[10], 1, 1)).await;
        assert!(matches!(result, Err(ModelEngineError::WorkerUnavailable)));
        let registry = registry.lock().unwrap();
        assert!(!registry.accepting);
        assert!(registry.cancellations.is_empty());
    }
}
