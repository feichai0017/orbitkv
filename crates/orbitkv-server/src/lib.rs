#![forbid(unsafe_code)]

mod engine;
mod intent;
#[cfg(feature = "vllm-frontend")]
mod vllm_frontend;
#[cfg(feature = "vllm-frontend")]
mod vllm_transport;

pub use engine::{
    Engine, EngineAbortFuture, EngineEvent, EngineEventStream, EngineFuture, FinishReason,
    TokenOutput,
};
pub use intent::{BatchIntent, BatchIntentError, RequestId, RequestIntent, SamplingIntent};
#[cfg(feature = "vllm-frontend")]
pub use vllm_frontend::{VllmBridge, VllmBridgeError, VllmProtocolError, VllmSubmission};
#[cfg(feature = "vllm-frontend")]
pub use vllm_transport::{FrontendDtype, HttpFrontendConfig, serve_openai};
