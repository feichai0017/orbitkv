//! Local control-plane transport for inference engines and an OrbitKV sidecar.
//!
//! Payload bytes do not travel through this crate. Messages refer to
//! descriptors in separately registered CUDA IPC or shared-memory regions.

#![deny(unsafe_code)]

#[cfg(target_os = "linux")]
#[allow(
    unsafe_code,
    reason = "file-backed mmap creation requires an audited unsafe boundary"
)]
mod arena;
#[cfg(target_os = "linux")]
mod bootstrap;
#[cfg(target_os = "linux")]
mod client;
mod protocol;
mod query;
mod transport;

#[cfg(target_os = "linux")]
pub use arena::{ArenaError, DEFAULT_ARENA_BYTES, DEFAULT_SLOT_CAPACITY, DescriptorArena};
#[cfg(target_os = "linux")]
pub use bootstrap::{
    BootstrapClient, BootstrapError, BootstrapInfo, BootstrapServer, BootstrapSession,
    PeerCredentials,
};
#[cfg(target_os = "linux")]
pub use client::{LocalQueryClient, LocalQueryError};
pub use protocol::{
    ABI_VERSION, Command, CommandCode, DescriptorRef, ProtocolError,
    RESPONSE_FLAG_REQUEST_CONSUMED, Response, StatusCode, WIRE_MESSAGE_BYTES, WireMessage,
};
pub use query::{
    QueryBundleRequest, QueryBundleResponse, QueryCodecError, QueryOutcomeCode, ReleaseRequest,
};
pub use transport::{CallOptions, LocalClient, LocalServer, TransportError};
