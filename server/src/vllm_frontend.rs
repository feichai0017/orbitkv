//! Narrow protocol adapter between the vLLM Rust frontend and the local engine.
//!
//! This module deliberately translates only request intent and token events. It
//! does not import vLLM scheduling, KV allocation, or device-runtime ownership.

use std::collections::{BTreeMap, BTreeSet, btree_map::Entry};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use futures_util::StreamExt;
use thiserror::Error;
use vllm_engine_core_client::protocol::output::StopReason;
use vllm_engine_core_client::protocol::output::{EngineCoreFinishReason, EngineCoreOutput};
use vllm_engine_core_client::protocol::request::EngineCoreRequest;
use vllm_engine_core_client::protocol::sampling::EngineCoreSamplingParams;

use crate::{
    BatchIntent, BatchIntentError, Engine, EngineEvent, EngineEventStream, FinishReason, RequestId,
    RequestIntent, SamplingIntent,
};

/// Request fields the local engine cannot currently execute faithfully.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum VllmProtocolError {
    #[error("request id must be nonempty")]
    EmptyRequestId,
    #[error("local request identity space is exhausted")]
    RequestIdExhausted,
    #[error("request id is already active")]
    DuplicateRequestId,
    #[error("tokenized prompt is required")]
    MissingPromptTokens,
    #[error("sampling parameters are required")]
    MissingSampling,
    #[error("unsupported vLLM request field: {0}")]
    UnsupportedField(&'static str),
    #[error("engine event belongs to a different request")]
    EventRequestMismatch,
    #[error("local engine stream ended without a terminal event")]
    MissingTerminalEvent,
    #[error("request registry lock is poisoned")]
    RegistryPoisoned,
}

/// Failure while validating or executing one frontend request.
#[derive(Debug, Error)]
pub enum VllmBridgeError<E> {
    #[error(transparent)]
    Protocol(#[from] VllmProtocolError),
    #[error(transparent)]
    Intent(#[from] BatchIntentError),
    #[error("local engine failed: {0}")]
    Engine(E),
}

#[derive(Default)]
struct RequestRegistry {
    requests: Mutex<BTreeMap<String, RequestId>>,
}

impl RequestRegistry {
    fn insert(&self, frontend: String, internal: RequestId) -> Result<(), VllmProtocolError> {
        let mut requests = self
            .requests
            .lock()
            .map_err(|_| VllmProtocolError::RegistryPoisoned)?;
        match requests.entry(frontend) {
            Entry::Vacant(slot) => {
                slot.insert(internal);
                Ok(())
            }
            Entry::Occupied(_) => Err(VllmProtocolError::DuplicateRequestId),
        }
    }

    fn get(&self, frontend: &str) -> Result<Option<RequestId>, VllmProtocolError> {
        self.requests
            .lock()
            .map_err(|_| VllmProtocolError::RegistryPoisoned)
            .map(|requests| requests.get(frontend).copied())
    }

    fn remove(&self, frontend: &str) -> Result<(), VllmProtocolError> {
        self.requests
            .lock()
            .map_err(|_| VllmProtocolError::RegistryPoisoned)?
            .remove(frontend);
        Ok(())
    }

    fn snapshot(&self) -> Result<Vec<(String, RequestId)>, VllmProtocolError> {
        self.requests
            .lock()
            .map_err(|_| VllmProtocolError::RegistryPoisoned)
            .map(|requests| {
                requests
                    .iter()
                    .map(|(frontend, &internal)| (frontend.clone(), internal))
                    .collect()
            })
    }
}

/// Single-engine adapter used by a frontend transport implementation.
pub struct VllmBridge<E> {
    engine: Arc<E>,
    next_request_id: AtomicU64,
    registry: Arc<RequestRegistry>,
}

impl<E> VllmBridge<E>
where
    E: Engine,
{
    #[must_use]
    pub fn new(engine: Arc<E>) -> Self {
        Self {
            engine,
            next_request_id: AtomicU64::new(1),
            registry: Arc::new(RequestRegistry::default()),
        }
    }

    /// Validates one tokenized vLLM request and starts it on the local engine.
    ///
    /// # Errors
    ///
    /// Rejects every request semantic not represented by [`RequestIntent`],
    /// duplicate active IDs, invalid batches, and local engine failures.
    pub async fn add(
        &self,
        request: EngineCoreRequest,
    ) -> Result<VllmSubmission<E::Error>, VllmBridgeError<E::Error>> {
        let frontend_request_id = request.request_id.clone();
        let (input_tokens, sampling) = lower_request(&request)?;
        let internal_request_id = RequestId(
            self.next_request_id
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
                .map_err(|_| VllmProtocolError::RequestIdExhausted)?,
        );
        let target_boundary =
            u64::try_from(input_tokens.len()).map_err(|_| BatchIntentError::InvalidBoundary)?;
        let batch = BatchIntent::new(
            vec![RequestIntent {
                request_id: internal_request_id,
                input_tokens,
                target_boundary,
                sampling,
            }]
            .into_boxed_slice(),
        )?;
        self.registry
            .insert(frontend_request_id.clone(), internal_request_id)?;
        let stream = match self.engine.execute(batch).await {
            Ok(stream) => stream,
            Err(error) => {
                self.registry.remove(&frontend_request_id)?;
                return Err(VllmBridgeError::Engine(error));
            }
        };
        Ok(VllmSubmission {
            frontend_request_id,
            internal_request_id,
            stream,
            registry: Arc::clone(&self.registry),
            finished: false,
        })
    }

    /// Cancels an active external request ID. Unknown or finished IDs are a
    /// successful no-op, matching the vLLM client contract.
    ///
    /// # Errors
    ///
    /// Propagates registry and local engine failures.
    pub async fn abort(&self, frontend_request_id: &str) -> Result<(), VllmBridgeError<E::Error>> {
        let Some(internal_request_id) = self.registry.get(frontend_request_id)? else {
            return Ok(());
        };
        self.engine
            .abort(internal_request_id)
            .await
            .map_err(VllmBridgeError::Engine)?;
        self.registry.remove(frontend_request_id)?;
        Ok(())
    }

    /// Cancels every request still registered with this frontend instance.
    ///
    /// # Errors
    ///
    /// Attempts every cancellation and returns the first local engine error.
    pub async fn abort_all(&self) -> Result<(), VllmBridgeError<E::Error>> {
        let requests = self.registry.snapshot()?;
        let mut first_error = None;
        for (frontend_request_id, internal_request_id) in requests {
            if let Err(error) = self.engine.abort(internal_request_id).await {
                first_error.get_or_insert(error);
            } else {
                self.registry.remove(&frontend_request_id)?;
            }
        }
        first_error.map_or(Ok(()), |error| Err(VllmBridgeError::Engine(error)))
    }
}

/// One active request stream with stable frontend and local identities.
pub struct VllmSubmission<E> {
    frontend_request_id: String,
    internal_request_id: RequestId,
    stream: EngineEventStream<E>,
    registry: Arc<RequestRegistry>,
    finished: bool,
}

impl<E> VllmSubmission<E> {
    #[must_use]
    pub fn frontend_request_id(&self) -> &str {
        &self.frontend_request_id
    }

    #[must_use]
    pub const fn internal_request_id(&self) -> RequestId {
        self.internal_request_id
    }

    /// Returns the next frontend-visible output, skipping local batch markers.
    ///
    /// # Errors
    ///
    /// Propagates local engine failures and rejects events for another request.
    pub async fn next_output(&mut self) -> Result<Option<EngineCoreOutput>, VllmBridgeError<E>> {
        loop {
            let Some(event) = self.stream.next().await else {
                if self.finished {
                    return Ok(None);
                }
                self.finish()?;
                return Err(VllmProtocolError::MissingTerminalEvent.into());
            };
            let event = match event {
                Ok(event) => event,
                Err(error) => return Err(VllmBridgeError::Engine(error)),
            };
            let translated =
                translate_event(&self.frontend_request_id, self.internal_request_id, event);
            let output = match translated {
                Ok(output) => output,
                Err(error) => return Err(error.into()),
            };
            match output {
                None => {}
                Some(output) => {
                    if output.finished() {
                        self.finish()?;
                    }
                    return Ok(Some(output));
                }
            }
        }
    }

    fn finish(&mut self) -> Result<(), VllmProtocolError> {
        if !self.finished {
            self.registry.remove(&self.frontend_request_id)?;
            self.finished = true;
        }
        Ok(())
    }
}

fn lower_request(
    request: &EngineCoreRequest,
) -> Result<(Box<[u32]>, SamplingIntent), VllmProtocolError> {
    if request.request_id.is_empty() {
        return Err(VllmProtocolError::EmptyRequestId);
    }
    reject_request_fields(request)?;
    let tokens = request
        .prompt_token_ids
        .as_ref()
        .filter(|tokens| !tokens.is_empty())
        .ok_or(VllmProtocolError::MissingPromptTokens)?;
    let sampling = request
        .sampling_params
        .as_ref()
        .ok_or(VllmProtocolError::MissingSampling)?;
    validate_sampling(sampling)?;
    let stop_token_ids = sampling
        .stop_token_ids
        .iter()
        .copied()
        .chain(sampling.eos_token_id)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    Ok((
        tokens.clone().into_boxed_slice(),
        SamplingIntent::greedy(sampling.max_tokens, stop_token_ids),
    ))
}

fn reject_request_fields(request: &EngineCoreRequest) -> Result<(), VllmProtocolError> {
    let unsupported = [
        (request.mm_features.is_some(), "multimodal features"),
        (request.pooling_params.is_some(), "pooling"),
        (request.lora_request.is_some(), "LoRA"),
        (request.cache_salt.is_some(), "cache salt"),
        (
            request.data_parallel_rank.is_some(),
            "data-parallel routing",
        ),
        (request.prompt_embeds.is_some(), "prompt embeddings"),
        (request.prompt_is_token_ids.is_some(), "mixed prompt input"),
        (request.current_wave != 0, "data-parallel wave"),
        (request.priority != 0, "priority scheduling"),
        (request.trace_headers.is_some(), "trace headers"),
        (request.resumable, "resumable requests"),
        (request.reasoning_ended.is_some(), "reasoning state"),
        (
            request.reasoning_parser_kwargs.is_some(),
            "reasoning parser arguments",
        ),
        (request.abort_immediately, "immediate abort"),
        (request.session_id.is_some(), "session identity"),
    ];
    unsupported
        .into_iter()
        .find_map(|(present, field)| present.then_some(field))
        .map_or(Ok(()), |field| {
            Err(VllmProtocolError::UnsupportedField(field))
        })
}

#[allow(clippy::float_cmp)]
fn validate_sampling(params: &EngineCoreSamplingParams) -> Result<(), VllmProtocolError> {
    let unsupported = [
        (params.temperature != 0.0, "non-greedy sampling"),
        (params.top_p != 1.0, "top-p sampling"),
        (params.top_k != 0, "top-k sampling"),
        (params.seed.is_some(), "sampling seed"),
        (params.max_tokens == 0, "zero output budget"),
        (params.min_tokens != 0, "minimum output tokens"),
        (params.thinking_token_budget.is_some(), "thinking budget"),
        (params.logprobs.is_some(), "logprobs"),
        (params.prompt_logprobs.is_some(), "prompt logprobs"),
        (params.min_p != 0.0, "min-p sampling"),
        (params.frequency_penalty != 0.0, "frequency penalty"),
        (params.presence_penalty != 0.0, "presence penalty"),
        (params.repetition_penalty != 1.0, "repetition penalty"),
        (
            params.repetition_detection.is_some(),
            "repetition detection",
        ),
        (params.logit_bias.is_some(), "logit bias"),
        (params.allowed_token_ids.is_some(), "allowed token set"),
        (params.bad_words_token_ids.is_some(), "bad words"),
        (params.structured_outputs.is_some(), "structured outputs"),
        (params.logprob_token_ids.is_some(), "selected logprobs"),
        (
            params.skip_reading_prefix_cache == Some(true),
            "per-request prefix-cache policy",
        ),
        (
            params
                .extra_args
                .as_ref()
                .is_some_and(|args| !args.is_empty()),
            "extension arguments or KV transfer",
        ),
        (params.routed_experts_prompt_start != 0, "routed experts"),
    ];
    unsupported
        .into_iter()
        .find_map(|(present, field)| present.then_some(field))
        .map_or(Ok(()), |field| {
            Err(VllmProtocolError::UnsupportedField(field))
        })
}

fn translate_event(
    frontend_request_id: &str,
    internal_request_id: RequestId,
    event: EngineEvent,
) -> Result<Option<EngineCoreOutput>, VllmProtocolError> {
    let output = match event {
        EngineEvent::BatchStarted { request_ids } => {
            if request_ids.as_ref() != [internal_request_id] {
                return Err(VllmProtocolError::EventRequestMismatch);
            }
            return Ok(None);
        }
        EngineEvent::Token(output) => {
            if output.request_id != internal_request_id {
                return Err(VllmProtocolError::EventRequestMismatch);
            }
            EngineCoreOutput {
                request_id: frontend_request_id.to_string(),
                new_token_ids: vec![output.token_id],
                ..EngineCoreOutput::default()
            }
        }
        EngineEvent::Finished { request_id, reason } => {
            if request_id != internal_request_id {
                return Err(VllmProtocolError::EventRequestMismatch);
            }
            let (new_token_ids, finish_reason, stop_reason) = match reason {
                FinishReason::Stop { token_id } => (
                    vec![token_id],
                    EngineCoreFinishReason::Stop,
                    Some(StopReason::TokenId(token_id)),
                ),
                FinishReason::Length => (Vec::new(), EngineCoreFinishReason::Length, None),
                FinishReason::Cancelled => (Vec::new(), EngineCoreFinishReason::Abort, None),
            };
            EngineCoreOutput {
                request_id: frontend_request_id.to_string(),
                new_token_ids,
                finish_reason: Some(finish_reason),
                stop_reason,
                ..EngineCoreOutput::default()
            }
        }
    };
    Ok(Some(output))
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use futures_util::stream;

    use super::*;
    use crate::{EngineAbortFuture, EngineFuture, TokenOutput};

    #[derive(Default)]
    struct TestEngine {
        batches: Mutex<Vec<BatchIntent>>,
        aborted: Mutex<Vec<RequestId>>,
    }

    impl Engine for TestEngine {
        type Error = Infallible;

        fn execute(&self, batch: BatchIntent) -> EngineFuture<'_, Self::Error> {
            let request_id = batch.requests[0].request_id;
            self.batches.lock().unwrap().push(batch);
            Box::pin(async move {
                Ok(Box::pin(stream::iter([
                    Ok(EngineEvent::BatchStarted {
                        request_ids: vec![request_id].into_boxed_slice(),
                    }),
                    Ok(EngineEvent::Token(TokenOutput {
                        request_id,
                        token_id: 41,
                    })),
                    Ok(EngineEvent::Finished {
                        request_id,
                        reason: FinishReason::Stop { token_id: 2 },
                    }),
                ])) as EngineEventStream<Self::Error>)
            })
        }

        fn abort(&self, request_id: RequestId) -> EngineAbortFuture<'_, Self::Error> {
            self.aborted.lock().unwrap().push(request_id);
            Box::pin(async { Ok(()) })
        }
    }

    fn greedy_request(id: &str) -> EngineCoreRequest {
        let mut sampling = EngineCoreSamplingParams::for_test();
        sampling.temperature = 0.0;
        sampling.max_tokens = 8;
        sampling.eos_token_id = Some(2);
        EngineCoreRequest {
            request_id: id.to_string(),
            prompt_token_ids: Some(vec![10, 11]),
            sampling_params: Some(sampling),
            ..EngineCoreRequest::default()
        }
    }

    #[test]
    fn rejects_unimplemented_sampling_instead_of_silently_ignoring_it() {
        let mut request = greedy_request("external-1");
        request.sampling_params.as_mut().unwrap().temperature = 0.7;
        assert_eq!(
            lower_request(&request),
            Err(VllmProtocolError::UnsupportedField("non-greedy sampling"))
        );
        request.sampling_params.as_mut().unwrap().temperature = 0.0;
        request.sampling_params.as_mut().unwrap().logprobs = Some(1);
        assert_eq!(
            lower_request(&request),
            Err(VllmProtocolError::UnsupportedField("logprobs"))
        );
    }

    #[test]
    fn rejects_non_text_request_features() {
        let mut request = greedy_request("external-1");
        request.pooling_params = Some(vllm_engine_core_client::protocol::OpaqueValue::Nil);
        assert_eq!(
            lower_request(&request),
            Err(VllmProtocolError::UnsupportedField("pooling"))
        );
    }

    #[tokio::test]
    async fn maps_string_identity_and_streams_protocol_outputs() {
        let engine = Arc::new(TestEngine::default());
        let bridge = VllmBridge::new(Arc::clone(&engine));
        let mut submission = bridge.add(greedy_request("external-1")).await.unwrap();

        assert_eq!(submission.frontend_request_id(), "external-1");
        assert_eq!(submission.internal_request_id(), RequestId(1));
        let first = submission.next_output().await.unwrap().unwrap();
        assert_eq!(first.request_id, "external-1");
        assert_eq!(first.new_token_ids, vec![41]);
        assert_eq!(first.finish_reason, None);
        let terminal = submission.next_output().await.unwrap().unwrap();
        assert_eq!(terminal.new_token_ids, vec![2]);
        assert_eq!(terminal.finish_reason, Some(EngineCoreFinishReason::Stop));
        assert_eq!(terminal.stop_reason, Some(StopReason::TokenId(2)));
        assert!(submission.next_output().await.unwrap().is_none());

        let batches = engine.batches.lock().unwrap();
        assert_eq!(batches[0].requests[0].input_tokens.as_ref(), &[10, 11]);
        assert_eq!(batches[0].requests[0].sampling.max_output_tokens, 8);
        assert_eq!(
            batches[0].requests[0].sampling.stop_token_ids.as_ref(),
            &[2]
        );
    }

    #[tokio::test]
    async fn abort_resolves_external_identity_and_unknown_is_noop() {
        let engine = Arc::new(TestEngine::default());
        let bridge = VllmBridge::new(Arc::clone(&engine));
        let submission = bridge.add(greedy_request("external-1")).await.unwrap();
        let internal = submission.internal_request_id();

        bridge.abort("external-1").await.unwrap();
        bridge.abort("unknown").await.unwrap();
        assert_eq!(engine.aborted.lock().unwrap().as_slice(), &[internal]);
    }

    #[tokio::test]
    async fn duplicate_active_frontend_identity_is_rejected() {
        let engine = Arc::new(TestEngine::default());
        let bridge = VllmBridge::new(engine);
        let _first = bridge.add(greedy_request("external-1")).await.unwrap();
        let second = bridge.add(greedy_request("external-1")).await;
        assert!(matches!(
            second,
            Err(VllmBridgeError::Protocol(
                VllmProtocolError::DuplicateRequestId
            ))
        ));
    }
}
