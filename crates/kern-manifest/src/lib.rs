//! kern-manifest: the typed execution manifest a model provider ships
//! alongside compiled kernels (`manifest.json + kernels.cubin + weights`),
//! the load-time verifier that refuses anything not provably consistent,
//! and the serving protocol a driver reads off a manifest.
//!
//! The runtime assigns no meaning to any name in the manifest. It schedules
//! opaque kernel dispatches, provisions opaque per-token state bytes, and
//! evaluates a closed set of scalar expressions for launch geometry. All
//! model semantics live on the provider's side of this boundary. What a
//! serving loop needs — which buffer plays which role in a call, which
//! shape of call each program takes — the manifest declares (`fill`,
//! `batch`) and [`protocol::Protocol`] projects; the runtime never reads it.

#![forbid(unsafe_code)]

mod constants;
pub use constants::json_schema;

pub mod protocol;
pub mod types;
pub mod values;
pub mod verify;

pub use protocol::Protocol;
pub use types::Manifest;
pub use verify::{verify, Verified, VerifyErrors};
