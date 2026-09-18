//! Local control-plane transport for inference engines and an OrbitKV sidecar.
//!
//! Payload bytes do not travel through this crate. Messages refer to
//! descriptors in separately registered CUDA IPC or shared-memory regions.

#![forbid(unsafe_code)]

mod protocol;
mod transport;

pub use protocol::{
    ABI_VERSION, Command, CommandCode, DescriptorRef, ProtocolError, Response, StatusCode,
    WIRE_MESSAGE_BYTES, WireMessage,
};
pub use transport::{CallOptions, LocalClient, LocalServer, TransportError};
