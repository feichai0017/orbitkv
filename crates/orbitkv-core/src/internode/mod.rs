//! Inter-node communication module for OrbitKV.

pub(crate) mod metaserver_client;
pub(crate) mod p2p_service;

pub use metaserver_client::{MetaServerClient, MetaServerClientConfig};
pub use p2p_service::P2pTransferService;

#[cfg(feature = "mooncake")]
mod discovery;
