//! OrbitKV tracing infrastructure.
//!
//! This crate provides a composable Perfetto tracing layer that integrates
//! with the `tracing-subscriber` ecosystem.
//!
//! # Example
//!
//! ```rust,ignore
//! use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
//!
//! let (perfetto, guard) = orbitkv_tracing::perfetto_layer("trace.pftrace");
//!
//! tracing_subscriber::registry()
//!     .with(tracing_subscriber::fmt::layer())  // Console output
//!     .with(orbitkv_tracing::orbitkv_filter()) // Default OrbitKV filters
//!     .with(perfetto)                          // Perfetto trace file
//!     .init();
//!

use tracing_subscriber::filter::{LevelFilter, Targets};

mod perfetto;
pub use perfetto::*;

mod stages;
pub use stages::{StageLayer, StageTraceGuard, install_stage_trace_from_env, stage_trace_layer};

/// Perfetto protobuf schema types for post-processing traces.
#[allow(clippy::all)]
#[rustfmt::skip]
pub mod schema;

pub use prost;

/// Sets some default crate filters: `orbitkv=info, egglog=off, everything_else=error`
pub fn orbitkv_filter() -> Targets {
    Targets::new()
        .with_default(LevelFilter::ERROR)
        .with_target("egglog", LevelFilter::OFF)
        .with_target("orbitkv", LevelFilter::INFO)
}
