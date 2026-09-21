//! Inter-node communication module for OrbitKV.

mod metaserver_client;
pub(crate) mod p2p_service;

pub(crate) use metaserver_client::MetaServerClient;
pub use p2p_service::P2pTransferService;

#[cfg(feature = "mooncake")]
mod discovery;
