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
