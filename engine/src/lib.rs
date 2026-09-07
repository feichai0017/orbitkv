#![forbid(unsafe_code)]

#[cfg(feature = "cuda")]
mod model_engine;

#[cfg(feature = "cuda")]
pub use model_engine::{ModelEngine, ModelEngineConfig, ModelEngineError};
