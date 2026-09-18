mod engine;
mod error;
mod rc_backend;
pub mod rdma_topo;
mod remote;

mod cuda_lib;
mod cuda_sys;
mod cudart_sys;
pub mod v2;

pub use engine::{
    ConnectionStatus, HandshakeMetadata, MemoryRegion, TransferDesc, TransferEngine, TransferOp,
};
pub use error::{Result, TransferError};
pub use remote::{RemoteAddress, RemoteBackendKind, RemoteCompletion, RemoteMover, RemoteSlice};

pub fn init_logging() {
    orbitkv_common::logging::init_stderr("info,orbitkv_transfer=debug");
}
