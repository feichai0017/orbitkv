use thiserror::Error;

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum KvManagerError {
    #[error("batch must contain at least one item")]
    EmptyBatch,
    #[error("batch contains a duplicate request")]
    DuplicateRequest,
    #[error("batch contains a duplicate step")]
    DuplicateStep,
    #[error("batch contains a duplicate submission")]
    DuplicateSubmission,
    #[error("batch contains a duplicate prefix")]
    DuplicatePrefix,
    #[error("semantic prefix key is already published")]
    DuplicatePrefixKey,
    #[error("batch contains an invalid or non-canonical flat-buffer range")]
    InvalidBatchRange,
    #[error("{0} capacity must be positive")]
    ZeroCapacity(&'static str),
    #[error("{0} arena is exhausted")]
    ArenaExhausted(&'static str),
    #[error("arithmetic overflow: {0}")]
    ArithmeticOverflow(&'static str),
    #[error("engine epoch space is exhausted")]
    EngineEpochExhausted,
    #[error("unsupported manager profile: {0}")]
    UnsupportedProfile(&'static str),
    #[error("manager configuration is invalid")]
    InvalidConfiguration,
    #[error("identity belongs to a different manager")]
    WrongEngine,
    #[error("page belongs to a different manager or pool")]
    WrongPageArena,
    #[error("stale {0} lease")]
    StaleLease(&'static str),
    #[error("request is busy")]
    RequestBusy,
    #[error("request is released or quarantined")]
    RequestUnavailable,
    #[error("request is not recyclable")]
    RequestNotRecyclable,
    #[error("prefix lookup missed")]
    PrefixMiss,
    #[error("prefix lookup hint is stale")]
    PrefixHintStale,
    #[error("prefix must be evicted before recycling")]
    PrefixNotEvicted,
    #[error("prefix attach requires an empty request")]
    AttachRequiresEmptyRequest,
    #[error("prefix boundary must be positive and page-aligned")]
    PrefixBoundaryNotPageAligned,
    #[error("prefix boundary does not match its source snapshot")]
    PrefixBoundaryMismatch,
    #[error("prefix Full/Hybrid root bundle is not exact")]
    PrefixRootMismatch,
    #[error("physical page {0} reference count overflow")]
    ReferenceCountOverflow(u32),
    #[error("target boundary {target} does not advance current boundary {current}")]
    NonMonotonicBoundary { current: u64, target: u64 },
    #[error("step contains {requested} tokens, exceeding maximum {maximum}")]
    StepTooLarge { requested: u64, maximum: u64 },
    #[error("view version space is exhausted")]
    ViewVersionExhausted,
    #[error("physical page capacity is exhausted")]
    PageCapacityExhausted,
    #[error("invalid physical page id {0}")]
    InvalidPage(u32),
    #[error("invalid retention class id {0}")]
    InvalidClass(u16),
    #[error("physical page is stale")]
    StalePage,
    #[error("physical page {0} reader count overflow")]
    ReaderCountOverflow(u32),
    #[error("device view is stale")]
    StaleView,
    #[error("step was already submitted")]
    StepAlreadySubmitted,
    #[error("step was not submitted")]
    StepNotSubmitted,
    #[error("binding receipt does not match the manager-selected write set")]
    BindingReceiptMismatch,
    #[error("binding receipt duplicates a page")]
    DuplicateBindingReceipt,
    #[error("copy receipt does not match the manager-selected copy set")]
    CopyReceiptMismatch,
    #[error("backend copy observation is unknown")]
    CopyObservationUnknown,
    #[error("backend copy is not proven ordered before append writes")]
    CopyOrderingUnknown,
    #[error("batch was quarantined after backend receipt validation failed: {0}")]
    BatchQuarantined(Box<KvManagerError>),
    #[error("device view duplicates a page")]
    DuplicatePage,
    #[error("completion receipt is not confirmed")]
    CompletionNotConfirmed,
    #[error("backend observation is unknown")]
    BackendObservationUnknown,
    #[error("reclamation receipt is not acknowledged")]
    ReclamationNotAcknowledged,
    #[error("reclamation receipt does not match its certificate")]
    ReclamationMismatch,
    #[error("reclamation receipt is duplicated")]
    DuplicateReclamation,
    #[error("reserved field must be zero")]
    ReservedFieldNonZero,
    #[error("internal manager invariant failed: {0}")]
    Invariant(&'static str),
}
