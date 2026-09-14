//! Optional cross-layer compiler/runtime diagnostics. Collection does not
//! change artifact identity, search policy, or device synchronization.

pub use luminal_tracing::{StageTraceGuard, install_stage_trace_from_env, stage_trace_layer};
