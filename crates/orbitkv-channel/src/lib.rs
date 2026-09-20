//! Process channel between inference engines and an OrbitKV Cache Manager.
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
mod cache_protocol;
#[cfg(target_os = "linux")]
mod client;
pub mod lifecycle;
mod protocol;
mod transport;

#[cfg(target_os = "linux")]
pub use arena::{ArenaError, DEFAULT_ARENA_BYTES, DEFAULT_SLOT_CAPACITY, DescriptorArena};
#[cfg(target_os = "linux")]
pub use bootstrap::{
    BootstrapClient, BootstrapError, BootstrapInfo, BootstrapServer, BootstrapSession,
    PeerCredentials,
};
pub use cache_protocol::{
    PublishLayer, PublishRequest, QueryBundleRequest, QueryBundleResponse, QueryCodecError,
    QueryOutcomeCode, ReleaseRequest, RestoreCommand, RestoreLease, RestoreRequest,
    RestoreResponse, RestoreState,
};
#[cfg(target_os = "linux")]
pub use client::{ChannelClient, ChannelError};
pub use protocol::{
    ABI_VERSION, Command, CommandCode, DescriptorRef, ProtocolError,
    RESPONSE_FLAG_REQUEST_CONSUMED, Response, StatusCode, WIRE_MESSAGE_BYTES, WireMessage,
};
pub use transport::{
    CallOptions, DeferredResponse, TransportClient, TransportError, TransportServer,
};
