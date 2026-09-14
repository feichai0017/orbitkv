#![forbid(unsafe_code)]

#[cfg(feature = "vllm-frontend")]
mod frontend;
#[cfg(feature = "cuda")]
mod model_engine;
mod protocol;

#[cfg(feature = "vllm-frontend")]
pub use frontend::{
    FrontendDtype, HttpFrontendConfig, VllmBridge, VllmBridgeError, VllmProtocolError,
    VllmSubmission, serve_openai,
};
#[cfg(feature = "cuda")]
pub use model_engine::{
    EngineShutdownReport, EngineStartupReport, EngineStats, ModelEngine, ModelEngineConfig,
    ModelEngineError,
};
pub use protocol::{
    BatchIntent, BatchIntentError, Engine, EngineAbortFuture, EngineEvent, EngineEventStream,
    EngineFuture, FinishReason, RequestId, RequestIntent, SamplingIntent, TokenOutput,
};
