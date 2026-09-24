//! Framework-neutral contracts shared by OrbitKV engine adapters.
//!
//! These types describe logical model state and local page references without
//! importing vLLM block IDs, SGLang radix nodes, CUDA handles, or transport
//! implementations. Framework adapters translate their native objects into
//! this vocabulary before calling the OrbitKV data plane.

#![forbid(unsafe_code)]

mod bundle;
mod component;
mod discovery;
mod format;
mod inventory;
mod key;
mod page;

pub use bundle::{
    BundleComponent, RecoveryContract, RecoveryError, RecoveryRule, StateBundle, StateRequirement,
};
pub use component::StateComponent;
pub use discovery::{
    BlockCandidates, CacheOwner, DISCOVERY_MAX_BYTES, DISCOVERY_MAX_ENDPOINT_BYTES,
    DISCOVERY_MAX_KEYS, DISCOVERY_MAX_REPLICAS, ReplicaLocation, validate_discovery_query,
};
pub use format::{AttentionRole, Scalar16, StateDType, StateFormat, StateLayout, StorageFormat};
pub use inventory::{
    CATALOG_SHARDS, INVENTORY_BATCH_BYTES, INVENTORY_BATCH_RECORDS, InventoryOperation,
    InventoryRecord, InventoryStatus, catalog_shard,
};
pub use key::{
    ContractError, Digest, StateDescriptor, StateKey, StorageSlot, TokenRange, group_hash,
    storage_namespace,
};
pub use page::{LocalPageRef, RegionId};
