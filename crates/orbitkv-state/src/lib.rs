//! Framework-neutral contracts shared by OrbitKV engine adapters.
//!
//! These types describe logical model state and local page references without
//! importing vLLM block IDs, SGLang radix nodes, CUDA handles, or transport
//! implementations. Framework adapters translate their native objects into
//! this vocabulary before calling the OrbitKV data plane.

#![forbid(unsafe_code)]

mod bundle;
mod component;
mod format;
mod inventory;
mod key;
mod page;

pub use bundle::{BundleComponent, RecoveryContract, StateBundle};
pub use component::StateComponent;
pub use format::{StateDType, StateFormat, StateLayout};
pub use inventory::{
    INVENTORY_BATCH_BYTES, INVENTORY_BATCH_RECORDS, InventoryOperation, InventoryRecord,
    InventoryStatus,
};
pub use key::{
    ContractError, Digest, StateDescriptor, StateKey, StorageSlot, TokenRange, group_hash,
    storage_namespace,
};
pub use page::{LocalPageRef, RegionId};
