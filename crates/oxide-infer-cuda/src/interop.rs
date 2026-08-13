//! Event-bridged execution over engine-owned CUDA streams and allocations.
//!
//! This is the only library module that calls the raw stream/event driver
//! API. It never wraps an external `CUstream` in `CudaStream`, so Oxide never
//! acquires or destroys the engine's stream. Commands still use the standard
//! checked binding and provider enqueue path on a Oxide-owned non-blocking
//! stream.

use crate::attention::{
    Bf16PagedBatchDecodeAlgorithm, Bf16PagedBatchDecodeArgs, Bf16PagedBatchDecodePlan,
    Bf16SingleDecodeArgs, Bf16SingleDecodePlan, PagedBatchDecodeEnqueueError,
    SingleDecodeEnqueueError,
};
use crate::command::{
    BindingMemorySummary, CheckedBindings, CommandCompletion, CommandCompletionError, CommandError,
    CommandQueue, synchronize_stream_or_abort,
};
use crate::device_status::DeviceStatusProtocolError;
use cuda_core::sys::{CUcontext, CUdeviceptr, CUgreenCtx, CUstream};
use cuda_core::{CudaContext, CudaEvent, CudaStream, DriverError, IntoResult};
use oxide_infer::{ContractError, PagedKvLayout};
use std::any::Any;
use std::fmt::{self, Display, Formatter};
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use thiserror::Error;

const SPECIAL_STREAM_MAX_ADDRESS: usize = 2;

/// A validated, retained borrow of an engine-owned CUDA stream.
pub struct ExternalCudaStream {
    raw: CUstream,
    context: Arc<CudaContext>,
    _lease: Arc<dyn Any + Send + Sync>,
}

// SAFETY: host threads may use CUDA driver stream handles after binding the
// retained context. This value owns no stream destruction capability, and its
// lease is Send + Sync. StreamOrderedEngineAuthority carries submission
// exclusivity rather than shared access to this handle.
unsafe impl Send for ExternalCudaStream {}
// SAFETY: see the Send implementation above. All operations bind the context,
// and the engine authority contract prevents concurrent unordered submission.
unsafe impl Sync for ExternalCudaStream {}

impl ExternalCudaStream {
    /// Validates and retains an engine-owned stream without taking ownership.
    ///
    /// # Safety
    ///
    /// `raw` must be a live, ordinary CUDA stream. Null, `CU_STREAM_LEGACY`,
    /// and `CU_STREAM_PER_THREAD` handles are not accepted. `lease` must keep
    /// that stream alive until its final clone is dropped, and the stream must
    /// not be destroyed while this value exists. Passing an invalid CUDA
    /// handle to the driver's context query is undefined behavior.
    ///
    /// The adapter must represent exclusive submission access in the
    /// [`StreamOrderedEngineAuthority`] coupled to each binding set. No other
    /// thread may enqueue work on `raw` while Oxide owns that authority.
    pub unsafe fn from_raw_parts<L>(
        raw: CUstream,
        context: Arc<CudaContext>,
        lease: Arc<L>,
    ) -> Result<Self, ExternalCudaStreamError>
    where
        L: Any + Send + Sync,
    {
        if raw.addr() <= SPECIAL_STREAM_MAX_ADDRESS {
            return Err(ExternalCudaStreamError::SpecialStreamUnsupported);
        }
        context.bind_to_thread()?;
        // SAFETY: the caller guarantees that `raw` is a live stream. Both
        // output pointers target initialized local storage for this call.
        let (actual_context, green_context) = unsafe { stream_context(raw)? };
        if !green_context.is_null() {
            return Err(ExternalCudaStreamError::GreenContextUnsupported);
        }
        if actual_context != context.cu_ctx() {
            return Err(ExternalCudaStreamError::ContextMismatch {
                expected_device: context.ordinal(),
            });
        }
        let lease: Arc<dyn Any + Send + Sync> = lease;
        Ok(Self {
            raw,
            context,
            _lease: lease,
        })
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }
}

impl fmt::Debug for ExternalCudaStream {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExternalCudaStream")
            .field("raw", &self.raw)
            .field("device", &self.context.ordinal())
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Error)]
pub enum ExternalCudaStreamError {
    #[error("CUDA default, legacy, and per-thread special streams are not supported")]
    SpecialStreamUnsupported,
    #[error("green-context streams are not supported by the primary-context Oxide executor")]
    GreenContextUnsupported,
    #[error("the external stream does not belong to CUDA device {expected_device}'s context")]
    ContextMismatch { expected_device: usize },
    #[error(transparent)]
    Driver(#[from] DriverError),
}

/// Linear engine authority whose safe operations stay on one CUDA stream.
///
/// # Safety
///
/// An implementation must represent the only engine-side capability that can
/// access the spans coupled through [`EngineExternalBindings`]. The value must
/// not be `Clone`, expose a cross-stream access capability, permit immediate or
/// unsynchronized host access, or replace its stream or storage through
/// interior mutability. Every safe device operation and ordered device-to-host
/// transfer, including work started from `Drop`, must use the stable ordinary
/// stream returned by [`Self::submission_stream`].
/// While Oxide owns the value during [`EngineInteropQueue::enqueue`], no other
/// path may submit any work to that stream. This stream-wide exclusion ends
/// only after Oxide enqueues the post-event wait or an error path settles both
/// streams and returns the authority.
///
/// The bound regions must retain independent allocation and context leases.
/// Dropping the authority immediately after a successful handoff must not free
/// those allocations, destroy the stream, invoke a callback that accesses the
/// spans, or otherwise invalidate work retained by Oxide.
pub unsafe trait StreamOrderedEngineAuthority {
    /// Returns the stable ordinary stream governed by this authority.
    fn submission_stream(&self) -> CUstream;
}

/// Cross-stream synchronization used for one engine invocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngineStreamHandoff {
    ExternalEventBridge,
}

/// Operator submitted through the engine interop queue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngineOperator {
    Bf16SingleDecode,
    Bf16PagedBatchDecode,
}

/// Provider algorithm selected by an immutable operator plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngineAlgorithm {
    SingleDecodeDirect,
    PagedBatchDecodeDirect,
    PagedBatchDecodeTokenParallel8,
}

/// Contract dimensions recorded without retaining a plan or allocating.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngineExecutionShape {
    SingleDecode {
        kv_len: usize,
        query_heads: usize,
        kv_heads: usize,
        head_dim: usize,
    },
    PagedBatchDecode {
        batch_size: usize,
        max_num_pages: usize,
        query_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        page_size: usize,
    },
}

/// Exact device addresses in stable checked-binding slot order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngineBufferAddresses {
    SingleDecode([CUdeviceptr; SINGLE_DECODE_BINDING_COUNT]),
    PagedBatchDecode([CUdeviceptr; PAGED_BATCH_DECODE_BINDING_COUNT]),
}

impl EngineBufferAddresses {
    pub const fn as_slice(&self) -> &[CUdeviceptr] {
        match self {
            Self::SingleDecode(addresses) => addresses,
            Self::PagedBatchDecode(addresses) => addresses,
        }
    }
}

/// Evidence emitted by one provider invocation through the engine adapter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngineExecutionTrace {
    memory: BindingMemorySummary,
    buffer_addresses: EngineBufferAddresses,
    operator: EngineOperator,
    algorithm: EngineAlgorithm,
    shape: EngineExecutionShape,
    paged_kv_layout: Option<PagedKvLayout>,
    handoff: EngineStreamHandoff,
    adapter_device_to_device_copies: usize,
}

impl EngineExecutionTrace {
    pub const fn provider(&self) -> &'static str {
        "oxide-infer-cuda"
    }

    pub const fn operator(&self) -> EngineOperator {
        self.operator
    }

    pub const fn algorithm(&self) -> EngineAlgorithm {
        self.algorithm
    }

    pub const fn shape(&self) -> EngineExecutionShape {
        self.shape
    }

    pub const fn paged_kv_layout(&self) -> Option<PagedKvLayout> {
        self.paged_kv_layout
    }

    pub const fn memory(&self) -> BindingMemorySummary {
        self.memory
    }

    /// Returns exact device addresses in checked-binding slot order.
    pub fn buffer_addresses(&self) -> &[CUdeviceptr] {
        self.buffer_addresses.as_slice()
    }

    pub const fn stream_handoff(&self) -> EngineStreamHandoff {
        self.handoff
    }

    pub const fn adapter_device_to_device_copies(&self) -> usize {
        self.adapter_device_to_device_copies
    }

    /// Returns whether all operator buffers were external and this adapter
    /// issued no device-to-device copies. It does not describe copies issued
    /// elsewhere by the engine or provider implementation.
    pub const fn is_adapter_zero_copy(&self) -> bool {
        self.memory.all_external() && self.adapter_device_to_device_copies == 0
    }
}

/// External bindings coupled to the engine authority that governs them.
///
/// The engine adapter supplies `A`. It should contain every tensor,
/// storage, and stream-submission guard required to prevent access to the
/// bound ranges while Oxide establishes the event handoff.
pub struct EngineExternalBindings<A> {
    bindings: CheckedBindings,
    authority: A,
}

impl<A: StreamOrderedEngineAuthority> EngineExternalBindings<A> {
    /// Couples external bindings to the adapter's linear authority bundle.
    ///
    /// # Safety
    ///
    /// `authority` must govern the exact ordered device spans and access modes
    /// in `bindings`. No other engine path may access those spans. The unsafe
    /// trait implementation must satisfy the stream and lifetime rules on
    /// [`StreamOrderedEngineAuthority`].
    pub unsafe fn assume_engine_authority(
        bindings: CheckedBindings,
        authority: A,
    ) -> Result<Self, EngineExternalBindingsError<A>> {
        let memory = bindings.memory_summary();
        if !memory.all_external() || memory.total() != bindings.len() {
            return Err(EngineExternalBindingsError {
                cause: EngineExternalBindingsCause::NotAllExternal {
                    slots: bindings.len(),
                    live: bindings.live_regions(),
                    external_regions: memory.external_regions(),
                    device_buffers: memory.device_buffers(),
                },
                bindings: Box::new(bindings),
                authority: Box::new(authority),
            });
        }
        Ok(Self {
            bindings,
            authority,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum EngineExternalBindingsCause {
    #[error(
        "engine interop requires every binding slot to contain an external region; got {live} live regions across {slots} slots ({external_regions} external, {device_buffers} Oxide-owned)"
    )]
    NotAllExternal {
        slots: usize,
        live: usize,
        external_regions: usize,
        device_buffers: usize,
    },
}

/// A rejected coupling that retains the binding until authority is recovered.
pub struct EngineExternalBindingsError<A> {
    cause: EngineExternalBindingsCause,
    bindings: Box<CheckedBindings>,
    authority: Box<A>,
}

impl<A> EngineExternalBindingsError<A> {
    pub const fn cause(&self) -> EngineExternalBindingsCause {
        self.cause
    }

    /// Drops the rejected binding capability before returning engine authority.
    pub fn into_authority(self) -> A {
        let Self {
            bindings,
            authority,
            ..
        } = self;
        drop(bindings);
        *authority
    }
}

impl<A> fmt::Debug for EngineExternalBindingsError<A> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EngineExternalBindingsError")
            .field("cause", &self.cause)
            .finish_non_exhaustive()
    }
}

impl<A> Display for EngineExternalBindingsError<A> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.cause, formatter)
    }
}

impl<A> std::error::Error for EngineExternalBindingsError<A> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

/// A reusable Oxide queue ordered against one retained engine stream.
pub struct EngineInteropQueue {
    shared: Arc<EngineInteropShared>,
    queue: CommandQueue,
}

const SINGLE_DECODE_BINDING_COUNT: usize = 5;
const PAGED_BATCH_DECODE_BINDING_COUNT: usize = 9;

struct EngineInteropShared {
    external: Arc<ExternalCudaStream>,
    oxide_stream: Arc<CudaStream>,
    free_slots: Mutex<Vec<EngineHandoffSlot>>,
    free_bindings: Mutex<Vec<CheckedBindings>>,
    poisoned: AtomicBool,
}

struct EngineHandoffSlot {
    pre_event: CudaEvent,
    post_event: CudaEvent,
}

/// One immutable plan and its checked argument handles.
#[derive(Clone, Copy)]
pub enum EngineCommand<'a> {
    Bf16SingleDecode {
        plan: &'a Bf16SingleDecodePlan,
        args: Bf16SingleDecodeArgs,
    },
    Bf16PagedBatchDecode {
        plan: &'a Bf16PagedBatchDecodePlan,
        args: Bf16PagedBatchDecodeArgs,
    },
}

impl EngineCommand<'_> {
    const fn operator(self) -> EngineOperator {
        match self {
            Self::Bf16SingleDecode { .. } => EngineOperator::Bf16SingleDecode,
            Self::Bf16PagedBatchDecode { .. } => EngineOperator::Bf16PagedBatchDecode,
        }
    }

    const fn binding_count(self) -> usize {
        match self {
            Self::Bf16SingleDecode { .. } => SINGLE_DECODE_BINDING_COUNT,
            Self::Bf16PagedBatchDecode { .. } => PAGED_BATCH_DECODE_BINDING_COUNT,
        }
    }

    const fn required_commands(self) -> usize {
        match self {
            Self::Bf16SingleDecode { .. } => 1,
            Self::Bf16PagedBatchDecode { .. } => 3,
        }
    }

    fn trace(
        self,
        memory: BindingMemorySummary,
        bindings: &CheckedBindings,
    ) -> Option<EngineExecutionTrace> {
        let (addresses, algorithm, shape, paged_kv_layout) = match self {
            Self::Bf16SingleDecode { plan, .. } => {
                let spec = plan.spec();
                (
                    EngineBufferAddresses::SingleDecode(bindings.exact_device_addresses()?),
                    EngineAlgorithm::SingleDecodeDirect,
                    EngineExecutionShape::SingleDecode {
                        kv_len: spec.kv_len(),
                        query_heads: spec.num_query_heads(),
                        kv_heads: spec.num_kv_heads(),
                        head_dim: spec.head_dim(),
                    },
                    None,
                )
            }
            Self::Bf16PagedBatchDecode { plan, .. } => {
                let spec = plan.spec();
                let algorithm = match plan.algorithm() {
                    Bf16PagedBatchDecodeAlgorithm::Direct => {
                        EngineAlgorithm::PagedBatchDecodeDirect
                    }
                    Bf16PagedBatchDecodeAlgorithm::TokenParallel8 => {
                        EngineAlgorithm::PagedBatchDecodeTokenParallel8
                    }
                };
                (
                    EngineBufferAddresses::PagedBatchDecode(bindings.exact_device_addresses()?),
                    algorithm,
                    EngineExecutionShape::PagedBatchDecode {
                        batch_size: spec.batch_size(),
                        max_num_pages: spec.max_num_pages(),
                        query_heads: spec.num_query_heads(),
                        kv_heads: spec.num_kv_heads(),
                        head_dim: spec.head_dim(),
                        page_size: spec.page_size(),
                    },
                    Some(spec.kv_layout()),
                )
            }
        };
        Some(EngineExecutionTrace {
            memory,
            buffer_addresses: addresses,
            operator: self.operator(),
            algorithm,
            shape,
            paged_kv_layout,
            handoff: EngineStreamHandoff::ExternalEventBridge,
            adapter_device_to_device_copies: 0,
        })
    }

    fn enqueue_into(
        self,
        scope: &mut crate::command::CommandScope<'_>,
    ) -> Result<(), EngineProviderError> {
        match self {
            Self::Bf16SingleDecode { plan, args } => {
                plan.enqueue_into(scope, args).map_err(Into::into)
            }
            Self::Bf16PagedBatchDecode { plan, args } => {
                plan.enqueue_into(scope, args).map_err(Into::into)
            }
        }
    }
}

impl EngineInteropQueue {
    /// Creates a Oxide-owned non-blocking execution stream for `external`.
    pub fn new(
        external: ExternalCudaStream,
        max_commands: usize,
        max_in_flight: usize,
    ) -> Result<Self, EngineInteropBuildError> {
        let oxide_stream = external.context.new_stream()?;
        let queue = CommandQueue::new(oxide_stream.clone(), max_commands, max_in_flight)?;
        let mut free_slots = Vec::with_capacity(max_in_flight);
        for _ in 0..max_in_flight {
            free_slots.push(EngineHandoffSlot {
                pre_event: external.context.new_event(None)?,
                post_event: external.context.new_event(None)?,
            });
        }
        Ok(Self {
            shared: Arc::new(EngineInteropShared {
                external: Arc::new(external),
                oxide_stream,
                free_slots: Mutex::new(free_slots),
                free_bindings: Mutex::new(Vec::with_capacity(max_in_flight)),
                poisoned: AtomicBool::new(false),
            }),
            queue,
        })
    }

    /// Returns checked binding storage for this exact Oxide execution queue.
    /// Successfully settled storage is reused when its capacity matches.
    pub fn bindings(&self, capacity: usize) -> Result<CheckedBindings, CommandError> {
        let mut free_bindings = lock_or_recover(&self.shared.free_bindings);
        if let Some(index) = free_bindings
            .iter()
            .position(|bindings| bindings.capacity() == capacity)
        {
            return Ok(free_bindings.swap_remove(index));
        }
        drop(free_bindings);
        self.queue.bindings(capacity)
    }

    /// Enqueues one checked operator between two event handoffs.
    ///
    /// The pre-event orders all prior engine work before Oxide. The post-event
    /// orders future engine work after Oxide. The provider launches directly
    /// against the bound device pointers and the adapter performs no copy.
    ///
    /// `external_bindings` transfers the engine's stream and storage authority
    /// for the handoff. On success, the post-event wait is already enqueued;
    /// [`EngineSubmission::into_parts`] returns the stream-ordered authority
    /// without waiting for the command. Failures before bridge work starts
    /// return coupled authority and bindings immediately. Later failures settle
    /// both streams before returning any authority.
    pub fn enqueue<A: StreamOrderedEngineAuthority>(
        &mut self,
        command: EngineCommand<'_>,
        external_bindings: EngineExternalBindings<A>,
    ) -> Result<EngineSubmission<A>, EngineEnqueueError<A>> {
        let EngineExternalBindings {
            bindings,
            authority,
        } = external_bindings;
        if self.shared.poisoned.load(Ordering::Acquire) {
            return Err(EngineEnqueueError::recovered(
                EngineEnqueueCause::QueuePoisoned,
                authority,
                bindings,
            ));
        }

        let memory = bindings.memory_summary();
        if !memory.all_external() || memory.total() != bindings.len() {
            let cause = EngineEnqueueCause::BindingsNotAllExternal {
                slots: bindings.len(),
                live: bindings.live_regions(),
                external_regions: memory.external_regions(),
                device_buffers: memory.device_buffers(),
            };
            return Err(EngineEnqueueError::recovered(cause, authority, bindings));
        }
        if authority.submission_stream() != self.shared.external.raw {
            return Err(EngineEnqueueError::recovered(
                EngineEnqueueCause::AuthorityStreamMismatch,
                authority,
                bindings,
            ));
        }
        let Some(trace) = command.trace(memory, &bindings) else {
            let cause = EngineEnqueueCause::BindingShape {
                operator: command.operator(),
                expected: command.binding_count(),
                live: bindings.live_regions(),
                slots: bindings.len(),
            };
            return Err(EngineEnqueueError::recovered(cause, authority, bindings));
        };
        if self.queue.max_commands() < command.required_commands() {
            return Err(EngineEnqueueError::recovered(
                EngineEnqueueCause::Command(CommandError::CommandCapacityExceeded {
                    capacity: self.queue.max_commands(),
                }),
                authority,
                bindings,
            ));
        }

        let Some(slot) = lock_or_recover(&self.shared.free_slots).pop() else {
            return Err(EngineEnqueueError::recovered(
                EngineEnqueueCause::InFlightCapacityExceeded,
                authority,
                bindings,
            ));
        };
        let mut scope = match self.queue.begin(bindings) {
            Ok(scope) => scope,
            Err(error) => {
                lock_or_recover(&self.shared.free_slots).push(slot);
                let (error, bindings) = error.into_parts();
                return Err(EngineEnqueueError::recovered(
                    EngineEnqueueCause::Command(error),
                    authority,
                    bindings,
                ));
            }
        };
        let oxide_stream = Arc::clone(&self.shared.oxide_stream);
        let external_raw = self.shared.external.raw;

        // SAFETY: ExternalCudaStream validated and retains `external_raw`.
        // `pre_event` belongs to the same primary context.
        if let Err(error) = unsafe { record_event_on_raw_stream(&slot.pre_event, external_raw) } {
            self.shared.poisoned.store(true, Ordering::Release);
            return Err(settle_started_failure(
                Arc::clone(&self.shared),
                slot,
                scope.finish(),
                EngineEnqueuePrimaryFailure::Bridge(error),
                authority,
            ));
        }
        if let Err(error) = oxide_stream.wait(&slot.pre_event) {
            self.shared.poisoned.store(true, Ordering::Release);
            return Err(settle_started_failure(
                Arc::clone(&self.shared),
                slot,
                scope.finish(),
                EngineEnqueuePrimaryFailure::Bridge(error),
                authority,
            ));
        }
        if let Err(error) = command.enqueue_into(&mut scope) {
            return Err(settle_started_failure(
                Arc::clone(&self.shared),
                slot,
                scope.finish(),
                EngineEnqueuePrimaryFailure::Provider(error),
                authority,
            ));
        }

        scope.finalize_device_status();
        if let Err(error) = slot.post_event.record(&oxide_stream) {
            self.shared.poisoned.store(true, Ordering::Release);
            return Err(settle_started_failure(
                Arc::clone(&self.shared),
                slot,
                scope.finish(),
                EngineEnqueuePrimaryFailure::Bridge(error),
                authority,
            ));
        }
        // SAFETY: ExternalCudaStream retains the live raw stream and the event
        // slot is retained by the returned completion. This enqueues only a
        // wait and transfers no stream ownership.
        if let Err(error) = unsafe { wait_raw_stream_on_event(external_raw, &slot.post_event) } {
            self.shared.poisoned.store(true, Ordering::Release);
            return Err(settle_started_failure(
                Arc::clone(&self.shared),
                slot,
                scope.finish(),
                EngineEnqueuePrimaryFailure::Bridge(error),
                authority,
            ));
        }

        let completion = scope.finish();
        Ok(EngineSubmission {
            completion: EngineCommandCompletion {
                command: Some(completion),
                trace,
                slot: Some(slot),
                shared: Arc::clone(&self.shared),
                complete: false,
            },
            authority,
        })
    }
}

#[derive(Debug, Error)]
pub enum EngineInteropBuildError {
    #[error(transparent)]
    Command(#[from] CommandError),
    #[error(transparent)]
    Driver(#[from] DriverError),
}

#[derive(Debug, Error)]
pub enum EngineProviderError {
    #[error(transparent)]
    SingleDecode(#[from] SingleDecodeEnqueueError),
    #[error(transparent)]
    PagedBatchDecode(#[from] PagedBatchDecodeEnqueueError),
}

#[derive(Debug, Error)]
pub enum EngineEnqueuePrimaryFailure {
    #[error("provider enqueue failed: {0}")]
    Provider(EngineProviderError),
    #[error("stream bridge failed: {0}")]
    Bridge(DriverError),
}

#[derive(Debug, Error)]
pub enum EngineEnqueueCause {
    #[error(
        "engine interop requires every binding slot to contain an external region; got {live} live regions across {slots} slots ({external_regions} external, {device_buffers} Oxide-owned)"
    )]
    BindingsNotAllExternal {
        slots: usize,
        live: usize,
        external_regions: usize,
        device_buffers: usize,
    },
    #[error("engine authority governs a different CUDA stream")]
    AuthorityStreamMismatch,
    #[error(
        "{operator:?} interop requires exactly {expected} live binding slots, got {live} live across {slots} slots"
    )]
    BindingShape {
        operator: EngineOperator,
        expected: usize,
        live: usize,
        slots: usize,
    },
    #[error("the engine interop queue is poisoned after an earlier bridge failure")]
    QueuePoisoned,
    #[error("engine interop in-flight capacity is exhausted")]
    InFlightCapacityExceeded,
    #[error(transparent)]
    Command(CommandError),
    #[error("{primary}")]
    Started {
        primary: EngineEnqueuePrimaryFailure,
    },
    #[error("{primary}; command settlement also reported {completion}")]
    StartedAndCompletion {
        primary: EngineEnqueuePrimaryFailure,
        completion: Box<EngineCommandFailure>,
    },
}

/// Linear recovery state after an enqueue failure.
pub enum EngineEnqueueRecovery<A> {
    /// The original binding capability remains coupled to engine authority.
    Coupled(EngineExternalBindings<A>),
    /// Oxide could not recover the binding capability; only engine authority remains.
    AuthorityOnly(A),
}

impl<A: StreamOrderedEngineAuthority> EngineEnqueueRecovery<A> {
    pub const fn is_coupled(&self) -> bool {
        matches!(self, Self::Coupled(_))
    }

    pub fn into_coupled(self) -> Result<EngineExternalBindings<A>, A> {
        match self {
            Self::Coupled(bindings) => Ok(bindings),
            Self::AuthorityOnly(authority) => Err(authority),
        }
    }

    /// Drops any recovered binding capability before returning authority.
    pub fn into_authority(self) -> A {
        match self {
            Self::Coupled(EngineExternalBindings {
                bindings,
                authority,
            }) => {
                drop(bindings);
                authority
            }
            Self::AuthorityOnly(authority) => authority,
        }
    }
}

/// A settled enqueue failure with one linear recovery value.
pub struct EngineEnqueueError<A> {
    cause: EngineEnqueueCause,
    recovery: Box<EngineEnqueueRecovery<A>>,
}

impl<A: StreamOrderedEngineAuthority> EngineEnqueueError<A> {
    fn recovered(cause: EngineEnqueueCause, authority: A, bindings: CheckedBindings) -> Self {
        Self {
            cause,
            recovery: Box::new(EngineEnqueueRecovery::Coupled(EngineExternalBindings {
                bindings,
                authority,
            })),
        }
    }

    fn unrecoverable(cause: EngineEnqueueCause, authority: A) -> Self {
        Self {
            cause,
            recovery: Box::new(EngineEnqueueRecovery::AuthorityOnly(authority)),
        }
    }

    pub const fn cause(&self) -> &EngineEnqueueCause {
        &self.cause
    }

    pub fn recovery_is_coupled(&self) -> bool {
        self.recovery.is_coupled()
    }

    /// Returns one recovery value after both bridge streams are quiescent.
    pub fn into_parts(self) -> (EngineEnqueueCause, EngineEnqueueRecovery<A>) {
        (self.cause, *self.recovery)
    }
}

impl<A: StreamOrderedEngineAuthority> fmt::Debug for EngineEnqueueError<A> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EngineEnqueueError")
            .field("cause", &self.cause)
            .field("bindings_recovered", &self.recovery.is_coupled())
            .finish()
    }
}

impl<A: StreamOrderedEngineAuthority> Display for EngineEnqueueError<A> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.cause, formatter)
    }
}

impl<A: StreamOrderedEngineAuthority> std::error::Error for EngineEnqueueError<A> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

/// A completed event handoff split into engine authority and Oxide work.
#[must_use = "split the submission to recover engine authority and retain the Oxide completion"]
pub struct EngineSubmission<A> {
    completion: EngineCommandCompletion,
    authority: A,
}

impl<A> EngineSubmission<A> {
    /// Returns authority after the queued post-event wait without a host wait.
    pub fn into_parts(self) -> (EngineCommandCompletion, A) {
        (self.completion, self.authority)
    }
}

/// One in-flight provider command and its external-stream handoff evidence.
#[must_use = "dropping the completion waits before releasing buffer and stream leases"]
pub struct EngineCommandCompletion {
    command: Option<CommandCompletion>,
    trace: EngineExecutionTrace,
    slot: Option<EngineHandoffSlot>,
    shared: Arc<EngineInteropShared>,
    complete: bool,
}

impl EngineCommandCompletion {
    pub const fn trace(&self) -> &EngineExecutionTrace {
        &self.trace
    }

    pub const fn submitted(&self) -> usize {
        self.command
            .as_ref()
            .expect("live engine completion has a command")
            .submitted()
    }

    pub fn is_complete(&mut self) -> Result<bool, CommandError> {
        self.command
            .as_mut()
            .expect("live engine completion has a command")
            .is_complete()
    }

    /// Waits for execution and returns trace evidence only.
    pub fn wait(mut self) -> Result<EngineExecutionTrace, EngineCommandCompletionError> {
        let command = self
            .command
            .take()
            .expect("live engine completion has a command");
        let result = match sanitize_completion(command.wait()) {
            Ok(bindings) => {
                recycle_bindings(&self.shared, bindings);
                Ok(self.trace.clone())
            }
            Err(cause) => Err(EngineCommandCompletionError {
                cause: Box::new(cause),
                trace: Box::new(self.trace.clone()),
            }),
        };
        self.return_slot();
        self.complete = true;
        result
    }

    fn return_slot(&mut self) {
        let slot = self
            .slot
            .take()
            .expect("live engine completion has a handoff slot");
        lock_or_recover(&self.shared.free_slots).push(slot);
    }
}

#[derive(Debug, Error)]
pub enum EngineCommandFailure {
    #[error("command execution failed: {0}")]
    Execution(CommandError),
    #[error("device rejected command metadata: {0}")]
    DeviceRejected(ContractError),
    #[error("device status protocol failed: {0}")]
    StatusProtocol(DeviceStatusProtocolError),
}

#[derive(Debug, Error)]
#[error("engine command failed: {cause}")]
pub struct EngineCommandCompletionError {
    cause: Box<EngineCommandFailure>,
    trace: Box<EngineExecutionTrace>,
}

impl EngineCommandCompletionError {
    pub const fn trace(&self) -> &EngineExecutionTrace {
        &self.trace
    }

    pub const fn cause(&self) -> &EngineCommandFailure {
        &self.cause
    }
}

impl Drop for EngineCommandCompletion {
    fn drop(&mut self) {
        if self.complete {
            return;
        }
        drop(self.command.take());
        self.return_slot();
        self.complete = true;
    }
}

fn sanitize_completion(
    result: Result<CheckedBindings, CommandCompletionError>,
) -> Result<CheckedBindings, EngineCommandFailure> {
    match result {
        Ok(bindings) => Ok(bindings),
        Err(CommandCompletionError::Execution(error)) => {
            Err(EngineCommandFailure::Execution(error))
        }
        Err(CommandCompletionError::DeviceRejected(rejection)) => {
            let (error, bindings) = rejection.into_parts();
            drop(bindings);
            Err(EngineCommandFailure::DeviceRejected(error))
        }
        Err(CommandCompletionError::StatusProtocol(error)) => {
            Err(EngineCommandFailure::StatusProtocol(error))
        }
    }
}

fn recycle_bindings(shared: &EngineInteropShared, mut bindings: CheckedBindings) {
    if bindings.prepare_for_reuse().is_err() {
        shared.poisoned.store(true, Ordering::Release);
        return;
    }
    lock_or_recover(&shared.free_bindings).push(bindings);
}

fn settle_started_failure<A: StreamOrderedEngineAuthority>(
    shared: Arc<EngineInteropShared>,
    slot: EngineHandoffSlot,
    completion: CommandCompletion,
    primary: EngineEnqueuePrimaryFailure,
    authority: A,
) -> EngineEnqueueError<A> {
    let result = completion.wait();
    settle_bridge_streams(&shared);
    lock_or_recover(&shared.free_slots).push(slot);
    match result {
        Ok(bindings) => EngineEnqueueError::recovered(
            EngineEnqueueCause::Started { primary },
            authority,
            bindings,
        ),
        Err(CommandCompletionError::DeviceRejected(rejection)) => {
            let (error, bindings) = rejection.into_parts();
            EngineEnqueueError::recovered(
                EngineEnqueueCause::StartedAndCompletion {
                    primary,
                    completion: Box::new(EngineCommandFailure::DeviceRejected(error)),
                },
                authority,
                bindings,
            )
        }
        Err(CommandCompletionError::Execution(error)) => EngineEnqueueError::unrecoverable(
            EngineEnqueueCause::StartedAndCompletion {
                primary,
                completion: Box::new(EngineCommandFailure::Execution(error)),
            },
            authority,
        ),
        Err(CommandCompletionError::StatusProtocol(error)) => EngineEnqueueError::unrecoverable(
            EngineEnqueueCause::StartedAndCompletion {
                primary,
                completion: Box::new(EngineCommandFailure::StatusProtocol(error)),
            },
            authority,
        ),
    }
}

fn settle_bridge_streams(shared: &EngineInteropShared) {
    let external_error = synchronize_external_stream_or_abort(&shared.external);
    let oxide_error = synchronize_stream_or_abort(&shared.oxide_stream);
    if external_error.is_some() || oxide_error.is_some() {
        shared.poisoned.store(true, Ordering::Release);
    }
    if let Some(error) = external_error {
        shared.external.context.record_err::<()>(Err(error));
    }
    if let Some(error) = oxide_error {
        shared.oxide_stream.context().record_err::<()>(Err(error));
    }
}

fn lock_or_recover<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn synchronize_external_stream_or_abort(stream: &ExternalCudaStream) -> Option<DriverError> {
    let stream_result = stream.context.bind_to_thread().and_then(|()| {
        // SAFETY: ExternalCudaStream retains this validated raw handle.
        unsafe { cuda_core::sys::cuStreamSynchronize(stream.raw).result() }
    });
    match stream_result {
        Ok(()) => None,
        Err(stream_error) => match stream.context.synchronize() {
            Ok(()) => Some(stream_error),
            Err(context_error) => abort_after_external_sync_failure(stream_error, context_error),
        },
    }
}

fn abort_after_external_sync_failure(stream_error: DriverError, context_error: DriverError) -> ! {
    eprintln!(
        "oxide-infer-cuda cannot confirm external CUDA quiescence after stream and context \
         synchronization failed; aborting to preserve external allocation safety: \
         stream={stream_error}; context={context_error}"
    );
    std::process::abort()
}

/// Returns the regular and green contexts associated with a live raw stream.
///
/// # Safety
///
/// `stream` must be a valid CUDA stream handle. NVIDIA documents invalid
/// handles to `cuStreamGetCtx_v2` as undefined behavior.
unsafe fn stream_context(stream: CUstream) -> Result<(CUcontext, CUgreenCtx), DriverError> {
    let mut context = MaybeUninit::uninit();
    let mut green_context = MaybeUninit::uninit();
    // SAFETY: both outputs are valid and the caller guarantees the stream.
    unsafe {
        cuda_core::sys::cuStreamGetCtx_v2(stream, context.as_mut_ptr(), green_context.as_mut_ptr())
            .result()?;
        Ok((context.assume_init(), green_context.assume_init()))
    }
}

/// Records `event` on an externally owned raw stream without adopting it.
///
/// # Safety
///
/// `stream` must remain live and must belong to `event.context()`.
unsafe fn record_event_on_raw_stream(
    event: &CudaEvent,
    stream: CUstream,
) -> Result<(), DriverError> {
    event.context().bind_to_thread()?;
    // SAFETY: the caller supplies a live same-context stream and event.
    unsafe { cuda_core::sys::cuEventRecord(event.cu_event(), stream).result() }
}

/// Enqueues a wait on an externally owned raw stream without adopting it.
///
/// # Safety
///
/// `stream` must remain live through this call. `event` must be a live event
/// visible to that stream's context.
unsafe fn wait_raw_stream_on_event(stream: CUstream, event: &CudaEvent) -> Result<(), DriverError> {
    event.context().bind_to_thread()?;
    // SAFETY: the caller supplies a live raw stream and retained event.
    unsafe {
        cuda_core::sys::cuStreamWaitEvent(
            stream,
            event.cu_event(),
            cuda_core::sys::CUevent_wait_flags_enum_CU_EVENT_WAIT_DEFAULT,
        )
        .result()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send<T: Send>() {}

    #[test]
    fn engine_completion_is_send() {
        assert_send::<EngineCommandCompletion>();
    }

    #[test]
    fn zero_copy_trace_requires_only_external_bindings() {
        let trace = EngineExecutionTrace {
            memory: BindingMemorySummary::from_counts(0, 5),
            buffer_addresses: EngineBufferAddresses::SingleDecode([1, 2, 3, 4, 5]),
            operator: EngineOperator::Bf16SingleDecode,
            algorithm: EngineAlgorithm::SingleDecodeDirect,
            shape: EngineExecutionShape::SingleDecode {
                kv_len: 16,
                query_heads: 8,
                kv_heads: 2,
                head_dim: 128,
            },
            paged_kv_layout: None,
            handoff: EngineStreamHandoff::ExternalEventBridge,
            adapter_device_to_device_copies: 0,
        };
        assert!(trace.is_adapter_zero_copy());
        assert_eq!(trace.provider(), "oxide-infer-cuda");
        assert_eq!(trace.operator(), EngineOperator::Bf16SingleDecode);
    }
}
