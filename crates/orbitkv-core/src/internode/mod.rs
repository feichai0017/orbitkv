//! Inter-node communication module for OrbitKV.

mod catalog_client;
pub(crate) mod p2p_service;

pub(crate) use catalog_client::CatalogClient;
pub use p2p_service::P2pTransferService;

#[cfg(feature = "mooncake")]
mod discovery;
