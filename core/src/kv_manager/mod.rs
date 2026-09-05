use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::plan::{
    AddressProgram, BlockDomain, ClassLayoutProgram, CompiledKvClass, CompiledKvPlan,
    RetentionKind, RetirementProgram,
};

mod append_transaction;
mod arena;
mod error;
mod external_tier;
mod facade;
mod identity;
mod manager_state;
mod persistent_snapshot;
mod prefix;
mod protocol;
mod reclamation;
mod relocation_transaction;
#[cfg(test)]
mod test_model;
mod token_virtualization;
mod transaction_recovery;
mod transaction_validation;

use arena::{Arena, PageCounts, PagePhase, PageState};
#[cfg(test)]
use facade::validate_class_program;
use manager_state::{
    CensusWork, ClassDelta, ClassTransition, OperationState, PrefixState, PreparedState,
    ReclamationState, RequestState, StepDelta, SubmittedState,
};
use persistent_snapshot::{
    ClassRoot, PersistentRootEntries, RequestSnapshot, RootEntry, RootLayout,
};
#[cfg(test)]
use persistent_snapshot::{HotPathInstrumentation, RootTreeNode, root_instrumentation};
#[cfg(test)]
use test_model::{
    DEVICE_KV_ACCESS_READ, DEVICE_KV_ACCESS_WRITE, DEVICE_KV_NEEDS_BINDING, DeviceKvEntry,
};

pub use error::KvManagerError;
pub(crate) use external_tier::PinnedSnapshotPage;
pub use identity::{
    PageLease, PrefixLease, PrefixSemanticKey, ReclamationLease, RelocationLease, RequestLease,
    SnapshotLease, StepLease, SubmissionLease, ViewVersion,
};
pub use manager_state::CanonicalKvManager;
pub use protocol::{
    ArenaStats, AttachedPrefix, BackendArenaRegistration, BackendBindReceipt, BackendCopyReceipt,
    BackendUnobservedReceipt, BatchCompletionReceipt, CLASS_LOWERING_EPOCH_START,
    CLASS_LOWERING_PACKED, CLASS_LOWERING_RESETTABLE, CancelAttachedRequestItem, ClassLowering,
    CompletionBatch, CopyIntent, DetachedAction, DetachedBinding, DetachedReason, EvictedPrefix,
    ForkedRequest, ManagerConfig, ManagerStats, MaterializedRequestView, PhysicalResidencePolicy,
    PrefixAttachItem, PrefixEvictionBatch, PrefixLookupHint, PrefixPublishItem,
    PrefixPublishRelease, PrepareBatchItem, PreparedStep, PublishedPrefix, PublishedReceipt,
    ReclamationCertificate, ReclamationReceipt, ReleaseBatchCompletion, ReleaseBatchItem,
    ReleaseCompletion, RequestForkItem, RequestView, SnapshotPage, StepCompletion, SubmitBatchItem,
    SubmittedStep, TailAction, TailActionKind, WriteIntent,
};
pub use token_virtualization::{
    ClassTokenDispositionUpdate, CompletedRelocationBatch, CompletedRelocationItem,
    PrepareRelocationItem, PreparedRelocation, RelocationCopyReceipt, RelocationDestination,
    RelocationPageState, RelocationPlan, RelocationPolicy, RelocationUnobservedReceipt,
    SubmittedRelocation, TokenDisposition, TokenDispositionBatchItem, TokenDispositionKind,
    TokenDispositionUpdate, TokenLocation, TokenMove, TokenPlacement, TokenView, TokenViewQuery,
    apply_token_relocation, mark_token_dispositions, plan_token_relocation, validate_token_view,
};
use token_virtualization::{PersistentTokenTable, apply_class_transition};

const CANONICAL_PAGE_TOKENS: u64 = 16;
const FIRST_POOL_EPOCH: u64 = 1;

static NEXT_ENGINE_EPOCH: AtomicU64 = AtomicU64::new(1);

#[cfg(test)]
mod tests;
