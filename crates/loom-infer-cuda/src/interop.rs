//! Event-bridged execution over engine-owned CUDA streams and allocations.
//!
//! This is the only library module that calls the raw stream/event driver
//! API. It never wraps an external `CUstream` in `CudaStream`, so Loom never
//! acquires or destroys the engine's stream. Commands still use the standard
//! checked binding and provider enqueue path on a Loom-owned non-blocking
//! stream.

use crate::attention::{Bf16SingleDecodeArgs, Bf16SingleDecodePlan, SingleDecodeEnqueueError};
use crate::command::{
    BindingMemorySummary, CheckedBindings, CommandCompletion, CommandCompletionError, CommandError,
    CommandQueue, EngineBindingFingerprint, synchronize_stream_or_abort,
};
use cuda_core::sys::{CUcontext, CUdeviceptr, CUgreenCtx, CUstream};
use cuda_core::{CudaContext, CudaEvent, CudaStream, DriverError, IntoResult};
use loom_infer::ContractError;
use std::any::Any;
use std::fmt::{self, Display, Formatter};
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::sync::Arc;
use thiserror::Error;

const SPECIAL_STREAM_MAX_ADDRESS: usize = 2;

/// A validated, retained borrow of an engine-owned CUDA stream.
pub struct ExternalCudaStream {
    raw: CUstream,
    context: Arc<CudaContext>,
    _lease: Arc<dyn Any + Send + Sync>,
}

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
    /// authority passed through [`EngineExternalBindings`]. For each
    /// [`EngineInteropQueue::enqueue_bf16_single_decode`] call, no other thread
    /// may enqueue work on `raw` from the pre-event record until Loom enqueues
    /// the post-event wait.
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
    #[error("green-context streams are not supported by the primary-context Loom executor")]
    GreenContextUnsupported,
    #[error("the external stream does not belong to CUDA device {expected_device}'s context")]
    ContextMismatch { expected_device: usize },
    #[error(transparent)]
    Driver(#[from] DriverError),
}

/// Cross-stream synchronization used for one engine invocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngineStreamHandoff {
    ExternalEventBridge,
}

/// Kernel selected by the bounded engine interop slice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngineSingleDecodeAlgorithm {
    Direct,
}

/// Evidence emitted by one provider invocation through the engine adapter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngineExecutionTrace {
    memory: BindingMemorySummary,
    buffer_addresses: [CUdeviceptr; SINGLE_DECODE_BINDING_COUNT],
    algorithm: EngineSingleDecodeAlgorithm,
    handoff: EngineStreamHandoff,
    adapter_device_to_device_copies: usize,
}

impl EngineExecutionTrace {
    pub const fn provider(&self) -> &'static str {
        "loom-infer-cuda"
    }

    pub const fn operator(&self) -> &'static str {
        "bf16_single_decode"
    }

    pub const fn algorithm(&self) -> EngineSingleDecodeAlgorithm {
        self.algorithm
    }

    pub const fn memory(&self) -> BindingMemorySummary {
        self.memory
    }

    /// Returns the exact device addresses forwarded to checked bindings.
    pub fn buffer_addresses(&self) -> &[CUdeviceptr] {
        &self.buffer_addresses
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
/// `A` is supplied by the engine adapter. It should contain every tensor,
/// storage, and stream-submission guard required to prevent access to the
/// bound ranges while Loom establishes the event handoff.
pub struct EngineExternalBindings<A> {
    bindings: CheckedBindings,
    authority: A,
    fingerprint: Arc<EngineBindingFingerprint>,
}

impl<A> EngineExternalBindings<A> {
    /// Couples external bindings to the adapter's linear authority bundle.
    ///
    /// # Safety
    ///
    /// `authority` must govern the exact ordered device spans and access modes
    /// in `bindings`. While this value is owned by Loom, no engine path may use
    /// a writable span or enqueue work through the guarded external stream.
    /// `A` must be a linear capability: its safe API cannot clone, replace, or
    /// extract the guarded stream or storage authority. Shared access to `A`
    /// may submit only through the same external stream and must keep every
    /// allocation alive until that submitted work settles.
    /// Dropping `A`, or recovering it before Loom completion, must only restore
    /// engine operations ordered after the post-event wait. It must not expose
    /// host or cross-stream access that bypasses the handoff.
    /// The internal fingerprint detects a substituted or changed binding set;
    /// it does not prove that `authority` governs those allocations.
    pub unsafe fn assume_engine_authority(
        bindings: CheckedBindings,
        authority: A,
    ) -> Result<Self, EngineExternalBindingsError<A>> {
        let memory = bindings.memory_summary();
        let Some(fingerprint) = bindings.engine_fingerprint() else {
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
        };
        Ok(Self {
            bindings,
            authority,
            fingerprint: Arc::new(fingerprint),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum EngineExternalBindingsCause {
    #[error(
        "engine interop requires every binding slot to contain an external region; got {live} live regions across {slots} slots ({external_regions} external, {device_buffers} Loom-owned)"
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

/// A reusable Loom queue ordered against one retained engine stream.
pub struct EngineInteropQueue {
    external: ExternalCudaStream,
    loom_stream: Arc<CudaStream>,
    pre_event: CudaEvent,
    post_event: CudaEvent,
    queue: CommandQueue,
    poisoned: bool,
}

const SINGLE_DECODE_BINDING_COUNT: usize = 5;

impl EngineInteropQueue {
    /// Creates a Loom-owned non-blocking execution stream for `external`.
    pub fn new(
        external: ExternalCudaStream,
        max_commands: usize,
    ) -> Result<Self, EngineInteropBuildError> {
        let loom_stream = external.context.new_stream()?;
        let pre_event = external.context.new_event(None)?;
        let post_event = external.context.new_event(None)?;
        let queue = CommandQueue::new(loom_stream.clone(), max_commands, 1)?;
        Ok(Self {
            external,
            loom_stream,
            pre_event,
            post_event,
            queue,
            poisoned: false,
        })
    }

    /// Creates checked binding storage for this exact Loom execution queue.
    pub fn bindings(&self, capacity: usize) -> Result<CheckedBindings, CommandError> {
        self.queue.bindings(capacity)
    }

    /// Enqueues standard single-decode work between two event handoffs.
    ///
    /// The pre-event orders all prior engine work before Loom. The post-event
    /// orders future engine work after Loom. The provider launches directly
    /// against the bound device pointers and the adapter performs no copy.
    ///
    /// `external_bindings` transfers the engine's stream and storage authority
    /// for the handoff. On success, the post-event wait is already enqueued;
    /// [`EngineSubmission::into_parts`] can return authority before the kernel
    /// completes. Every error path conservatively settles both streams before
    /// returning authority. Bindings are returned in the error whenever
    /// command settlement can recover them safely. No post-wait is promised
    /// after an error.
    pub fn enqueue_bf16_single_decode<'queue, A>(
        &'queue mut self,
        plan: &Bf16SingleDecodePlan,
        external_bindings: EngineExternalBindings<A>,
        args: Bf16SingleDecodeArgs,
    ) -> Result<EngineSubmission<'queue, A>, EngineSingleDecodeEnqueueError<A>> {
        let EngineExternalBindings {
            bindings,
            authority,
            fingerprint,
        } = external_bindings;
        let loom_stream = self.loom_stream.clone();
        if self.poisoned {
            settle_bridge_streams(&self.external, &loom_stream, &mut self.poisoned);
            return Err(EngineSingleDecodeEnqueueError::recovered(
                EngineSingleDecodeEnqueueCause::QueuePoisoned,
                authority,
                bindings,
                fingerprint,
            ));
        }

        let memory = bindings.memory_summary();
        if !memory.all_external() || memory.total() != bindings.len() {
            let cause = EngineSingleDecodeEnqueueCause::BindingsNotAllExternal {
                slots: bindings.len(),
                live: bindings.live_regions(),
                external_regions: memory.external_regions(),
                device_buffers: memory.device_buffers(),
            };
            settle_bridge_streams(&self.external, &loom_stream, &mut self.poisoned);
            drop(bindings);
            return Err(EngineSingleDecodeEnqueueError::unrecoverable(
                cause, authority,
            ));
        }
        if !bindings.matches_engine_fingerprint(&fingerprint) {
            settle_bridge_streams(&self.external, &loom_stream, &mut self.poisoned);
            drop(bindings);
            return Err(EngineSingleDecodeEnqueueError::unrecoverable(
                EngineSingleDecodeEnqueueCause::BindingFingerprintMismatch,
                authority,
            ));
        }

        let actual_bindings = bindings.live_regions();
        let Some(buffer_addresses) = bindings.exact_device_addresses() else {
            let cause = EngineSingleDecodeEnqueueCause::BindingShape {
                expected: SINGLE_DECODE_BINDING_COUNT,
                live: actual_bindings,
                slots: bindings.len(),
            };
            settle_bridge_streams(&self.external, &loom_stream, &mut self.poisoned);
            return Err(EngineSingleDecodeEnqueueError::recovered(
                cause,
                authority,
                bindings,
                fingerprint,
            ));
        };
        let trace = EngineExecutionTrace {
            memory,
            buffer_addresses,
            algorithm: EngineSingleDecodeAlgorithm::Direct,
            handoff: EngineStreamHandoff::ExternalEventBridge,
            adapter_device_to_device_copies: 0,
        };
        let external_raw = self.external.raw;

        // SAFETY: ExternalCudaStream validated and retains `external_raw`.
        // `pre_event` belongs to the same primary context.
        if let Err(error) = unsafe { record_event_on_raw_stream(&self.pre_event, external_raw) } {
            self.poisoned = true;
            settle_bridge_streams(&self.external, &loom_stream, &mut self.poisoned);
            return Err(EngineSingleDecodeEnqueueError::recovered(
                EngineSingleDecodeEnqueueCause::Bridge(error),
                authority,
                bindings,
                fingerprint,
            ));
        }
        if let Err(error) = loom_stream.wait(&self.pre_event) {
            self.poisoned = true;
            settle_bridge_streams(&self.external, &loom_stream, &mut self.poisoned);
            return Err(EngineSingleDecodeEnqueueError::recovered(
                EngineSingleDecodeEnqueueCause::Bridge(error),
                authority,
                bindings,
                fingerprint,
            ));
        }

        let mut scope = match self.queue.begin(bindings) {
            Ok(scope) => scope,
            Err(error) => {
                let (error, bindings) = error.into_parts();
                settle_bridge_streams(&self.external, &loom_stream, &mut self.poisoned);
                return Err(EngineSingleDecodeEnqueueError::recovered(
                    EngineSingleDecodeEnqueueCause::Command(error),
                    authority,
                    bindings,
                    fingerprint,
                ));
            }
        };
        if let Err(error) = plan.enqueue_into(&mut scope, args) {
            let completion = scope.finish();
            return Err(settle_provider_failure(
                &self.external,
                &loom_stream,
                &mut self.poisoned,
                completion,
                error,
                authority,
                fingerprint,
            ));
        }

        // This bounded operator has no deferred status copies. Recording the
        // post event before the standard finish call keeps one final command
        // fence while placing that fence after the post event.
        if let Err(error) = self.post_event.record(&loom_stream) {
            self.poisoned = true;
            let completion = scope.finish();
            return Err(settle_bridge_failure(
                &self.external,
                &loom_stream,
                &mut self.poisoned,
                completion,
                error,
                authority,
                fingerprint,
            ));
        }
        let completion = scope.finish();
        // SAFETY: ExternalCudaStream retains the live raw stream and the event
        // is preallocated in this queue. This enqueues only a wait.
        if let Err(error) = unsafe { wait_raw_stream_on_event(external_raw, &self.post_event) } {
            self.poisoned = true;
            return Err(settle_bridge_failure(
                &self.external,
                &loom_stream,
                &mut self.poisoned,
                completion,
                error,
                authority,
                fingerprint,
            ));
        }

        let authority_fingerprint = Arc::clone(&fingerprint);
        Ok(EngineSubmission {
            authority: EngineReturnedAuthority {
                authority,
                fingerprint: authority_fingerprint,
            },
            completion: EngineCommandCompletion {
                command: completion,
                trace,
                fingerprint,
                queue_borrow: PhantomData,
            },
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
pub enum EngineSingleDecodeEnqueueCause {
    #[error(
        "single-decode interop requires every binding slot to contain an external region; got {live} live regions across {slots} slots ({external_regions} external, {device_buffers} Loom-owned)"
    )]
    BindingsNotAllExternal {
        slots: usize,
        live: usize,
        external_regions: usize,
        device_buffers: usize,
    },
    #[error("the external binding set changed after engine authority was attached")]
    BindingFingerprintMismatch,
    #[error(
        "single-decode interop requires exactly {expected} live binding slots, got {live} live across {slots} slots"
    )]
    BindingShape {
        expected: usize,
        live: usize,
        slots: usize,
    },
    #[error("the engine interop queue is poisoned after an earlier bridge failure")]
    QueuePoisoned,
    #[error(transparent)]
    Command(CommandError),
    #[error(transparent)]
    Provider(SingleDecodeEnqueueError),
    #[error(transparent)]
    Bridge(DriverError),
    #[error(
        "provider enqueue failed with {provider}; settling the command also failed: {completion}"
    )]
    ProviderAndCompletion {
        provider: SingleDecodeEnqueueError,
        completion: Box<CommandCompletionError>,
    },
    #[error(
        "provider enqueue failed with {provider}; the device also rejected the command: {device}"
    )]
    ProviderAndDeviceRejection {
        provider: SingleDecodeEnqueueError,
        device: ContractError,
    },
    #[error(
        "stream bridge failed with {bridge}; settling the submitted command also failed: {completion}"
    )]
    BridgeAndCompletion {
        bridge: DriverError,
        completion: Box<CommandCompletionError>,
    },
    #[error("stream bridge failed with {bridge}; the device also rejected the command: {device}")]
    BridgeAndDeviceRejection {
        bridge: DriverError,
        device: ContractError,
    },
}

/// Linear recovery state after an enqueue failure.
pub enum EngineEnqueueRecovery<A> {
    /// The original binding capability remains coupled to engine authority.
    Coupled(EngineExternalBindings<A>),
    /// Loom could not recover the binding capability; only engine authority remains.
    AuthorityOnly(A),
}

impl<A> EngineEnqueueRecovery<A> {
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
                ..
            }) => {
                drop(bindings);
                authority
            }
            Self::AuthorityOnly(authority) => authority,
        }
    }
}

/// A settled enqueue failure with one linear recovery value.
pub struct EngineSingleDecodeEnqueueError<A> {
    cause: EngineSingleDecodeEnqueueCause,
    recovery: Box<EngineEnqueueRecovery<A>>,
}

impl<A> EngineSingleDecodeEnqueueError<A> {
    fn recovered(
        cause: EngineSingleDecodeEnqueueCause,
        authority: A,
        bindings: CheckedBindings,
        fingerprint: Arc<EngineBindingFingerprint>,
    ) -> Self {
        Self {
            cause,
            recovery: Box::new(EngineEnqueueRecovery::Coupled(EngineExternalBindings {
                bindings,
                authority,
                fingerprint,
            })),
        }
    }

    fn unrecoverable(cause: EngineSingleDecodeEnqueueCause, authority: A) -> Self {
        Self {
            cause,
            recovery: Box::new(EngineEnqueueRecovery::AuthorityOnly(authority)),
        }
    }

    pub const fn cause(&self) -> &EngineSingleDecodeEnqueueCause {
        &self.cause
    }

    pub fn recovery_is_coupled(&self) -> bool {
        self.recovery.is_coupled()
    }

    /// Returns one recovery value after both bridge streams are quiescent.
    pub fn into_parts(self) -> (EngineSingleDecodeEnqueueCause, EngineEnqueueRecovery<A>) {
        (self.cause, *self.recovery)
    }
}

impl<A> fmt::Debug for EngineSingleDecodeEnqueueError<A> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EngineSingleDecodeEnqueueError")
            .field("cause", &self.cause)
            .field("bindings_recovered", &self.recovery.is_coupled())
            .finish()
    }
}

impl<A> Display for EngineSingleDecodeEnqueueError<A> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.cause, formatter)
    }
}

impl<A> std::error::Error for EngineSingleDecodeEnqueueError<A> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

/// A completed event handoff split into returned engine authority and Loom work.
///
/// The external stream already waits on Loom's post event. Calling
/// [`Self::into_parts`] therefore returns an opaque authority token without
/// waiting for the kernel.
/// The completion keeps checked bindings and their external allocation leases
/// until it is settled.
#[must_use = "split the submission to recover engine authority and retain the Loom completion"]
pub struct EngineSubmission<'queue, A> {
    completion: EngineCommandCompletion<'queue>,
    authority: EngineReturnedAuthority<A>,
}

impl<'queue, A> EngineSubmission<'queue, A> {
    pub fn into_parts(self) -> (EngineCommandCompletion<'queue>, EngineReturnedAuthority<A>) {
        (self.completion, self.authority)
    }
}

/// Engine authority ordered after Loom while its binding capability stays private.
///
/// Shared access exposes the adapter's stream-scoped operations, but the
/// authority value itself cannot be replaced while its submission identity is
/// needed for rejoin.
pub struct EngineReturnedAuthority<A> {
    authority: A,
    fingerprint: Arc<EngineBindingFingerprint>,
}

impl<A> EngineReturnedAuthority<A> {
    pub const fn authority(&self) -> &A {
        &self.authority
    }
}

/// One in-flight provider command and its external-stream handoff evidence.
#[must_use = "dropping the completion waits before releasing buffer and stream leases"]
pub struct EngineCommandCompletion<'queue> {
    command: CommandCompletion,
    trace: EngineExecutionTrace,
    fingerprint: Arc<EngineBindingFingerprint>,
    queue_borrow: PhantomData<&'queue mut EngineInteropQueue>,
}

impl EngineCommandCompletion<'_> {
    pub const fn trace(&self) -> &EngineExecutionTrace {
        &self.trace
    }

    pub const fn submitted(&self) -> usize {
        self.command.submitted()
    }

    /// Waits for Loom execution and returns opaque settled bindings.
    pub fn wait(self) -> Result<EngineCommandOutcome, EngineCommandCompletionError> {
        let trace = self.trace;
        let fingerprint = self.fingerprint;
        match self.command.wait() {
            Ok(bindings) => Ok(EngineCommandOutcome {
                bindings,
                trace,
                fingerprint,
            }),
            Err(source) => Err(EngineCommandCompletionError { source, trace }),
        }
    }
}

/// Settled bindings that remain opaque until authority is rejoined or released.
pub struct EngineCommandOutcome {
    bindings: CheckedBindings,
    trace: EngineExecutionTrace,
    fingerprint: Arc<EngineBindingFingerprint>,
}

impl EngineCommandOutcome {
    pub const fn trace(&self) -> &EngineExecutionTrace {
        &self.trace
    }

    /// Re-couples the exact authority returned by this submission for reuse.
    pub fn rejoin<A>(
        self,
        authority: EngineReturnedAuthority<A>,
    ) -> Result<EngineExternalBindings<A>, EngineAuthorityRejoinError<A>> {
        if !Arc::ptr_eq(&self.fingerprint, &authority.fingerprint) {
            return Err(EngineAuthorityRejoinError::new(
                EngineAuthorityRejoinCause::SubmissionMismatch,
                self,
                authority,
            ));
        }
        if !self.bindings.matches_engine_fingerprint(&self.fingerprint) {
            return Err(EngineAuthorityRejoinError::new(
                EngineAuthorityRejoinCause::BindingFingerprintMismatch,
                self,
                authority,
            ));
        }
        Ok(EngineExternalBindings {
            bindings: self.bindings,
            authority: authority.authority,
            fingerprint: self.fingerprint,
        })
    }

    /// Drops Loom's binding capability and returns the matching engine authority.
    pub fn release<A>(
        self,
        authority: EngineReturnedAuthority<A>,
    ) -> Result<(A, EngineExecutionTrace), EngineAuthorityRejoinError<A>> {
        if !Arc::ptr_eq(&self.fingerprint, &authority.fingerprint) {
            return Err(EngineAuthorityRejoinError::new(
                EngineAuthorityRejoinCause::SubmissionMismatch,
                self,
                authority,
            ));
        }
        if !self.bindings.matches_engine_fingerprint(&self.fingerprint) {
            return Err(EngineAuthorityRejoinError::new(
                EngineAuthorityRejoinCause::BindingFingerprintMismatch,
                self,
                authority,
            ));
        }
        let Self {
            bindings, trace, ..
        } = self;
        drop(bindings);
        Ok((authority.authority, trace))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum EngineAuthorityRejoinCause {
    #[error("engine authority belongs to a different Loom submission")]
    SubmissionMismatch,
    #[error("settled bindings no longer match their engine handoff fingerprint")]
    BindingFingerprintMismatch,
}

/// A rejected rejoin with both opaque linear capabilities preserved.
pub struct EngineAuthorityRejoinError<A> {
    cause: EngineAuthorityRejoinCause,
    outcome: Box<EngineCommandOutcome>,
    authority: Box<EngineReturnedAuthority<A>>,
}

impl<A> EngineAuthorityRejoinError<A> {
    fn new(
        cause: EngineAuthorityRejoinCause,
        outcome: EngineCommandOutcome,
        authority: EngineReturnedAuthority<A>,
    ) -> Self {
        Self {
            cause,
            outcome: Box::new(outcome),
            authority: Box::new(authority),
        }
    }

    pub const fn cause(&self) -> EngineAuthorityRejoinCause {
        self.cause
    }

    pub fn into_parts(
        self,
    ) -> (
        EngineAuthorityRejoinCause,
        EngineCommandOutcome,
        EngineReturnedAuthority<A>,
    ) {
        (self.cause, *self.outcome, *self.authority)
    }
}

impl<A> fmt::Debug for EngineAuthorityRejoinError<A> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EngineAuthorityRejoinError")
            .field("cause", &self.cause)
            .finish_non_exhaustive()
    }
}

impl<A> Display for EngineAuthorityRejoinError<A> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.cause, formatter)
    }
}

impl<A> std::error::Error for EngineAuthorityRejoinError<A> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

#[derive(Debug, Error)]
#[error("engine command failed: {source}")]
pub struct EngineCommandCompletionError {
    source: CommandCompletionError,
    trace: EngineExecutionTrace,
}

impl EngineCommandCompletionError {
    pub const fn trace(&self) -> &EngineExecutionTrace {
        &self.trace
    }

    pub const fn source_error(&self) -> &CommandCompletionError {
        &self.source
    }
}

fn settle_bridge_failure<A>(
    external: &ExternalCudaStream,
    loom_stream: &CudaStream,
    poisoned: &mut bool,
    completion: CommandCompletion,
    bridge: DriverError,
    authority: A,
    fingerprint: Arc<EngineBindingFingerprint>,
) -> EngineSingleDecodeEnqueueError<A> {
    let result = completion.wait();
    settle_bridge_streams(external, loom_stream, poisoned);
    match result {
        Ok(bindings) => EngineSingleDecodeEnqueueError::recovered(
            EngineSingleDecodeEnqueueCause::Bridge(bridge),
            authority,
            bindings,
            fingerprint,
        ),
        Err(CommandCompletionError::DeviceRejected(rejection)) => {
            let (device, bindings) = rejection.into_parts();
            EngineSingleDecodeEnqueueError::recovered(
                EngineSingleDecodeEnqueueCause::BridgeAndDeviceRejection { bridge, device },
                authority,
                bindings,
                fingerprint,
            )
        }
        Err(completion) => EngineSingleDecodeEnqueueError::unrecoverable(
            EngineSingleDecodeEnqueueCause::BridgeAndCompletion {
                bridge,
                completion: Box::new(completion),
            },
            authority,
        ),
    }
}

fn settle_provider_failure<A>(
    external: &ExternalCudaStream,
    loom_stream: &CudaStream,
    poisoned: &mut bool,
    completion: CommandCompletion,
    provider: SingleDecodeEnqueueError,
    authority: A,
    fingerprint: Arc<EngineBindingFingerprint>,
) -> EngineSingleDecodeEnqueueError<A> {
    let result = completion.wait();
    settle_bridge_streams(external, loom_stream, poisoned);
    match result {
        Ok(bindings) => EngineSingleDecodeEnqueueError::recovered(
            EngineSingleDecodeEnqueueCause::Provider(provider),
            authority,
            bindings,
            fingerprint,
        ),
        Err(CommandCompletionError::DeviceRejected(rejection)) => {
            let (device, bindings) = rejection.into_parts();
            EngineSingleDecodeEnqueueError::recovered(
                EngineSingleDecodeEnqueueCause::ProviderAndDeviceRejection { provider, device },
                authority,
                bindings,
                fingerprint,
            )
        }
        Err(completion) => EngineSingleDecodeEnqueueError::unrecoverable(
            EngineSingleDecodeEnqueueCause::ProviderAndCompletion {
                provider,
                completion: Box::new(completion),
            },
            authority,
        ),
    }
}

fn settle_bridge_streams(
    external: &ExternalCudaStream,
    loom_stream: &CudaStream,
    poisoned: &mut bool,
) {
    let external_error = synchronize_external_stream_or_abort(external);
    let loom_error = synchronize_stream_or_abort(loom_stream);
    if external_error.is_some() || loom_error.is_some() {
        *poisoned = true;
    }
    if let Some(error) = external_error {
        external.context.record_err::<()>(Err(error));
    }
    if let Some(error) = loom_error {
        loom_stream.context().record_err::<()>(Err(error));
    }
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
        "loom-infer-cuda cannot confirm external CUDA quiescence after stream and context \
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

    #[test]
    fn zero_copy_trace_requires_only_external_bindings() {
        let trace = EngineExecutionTrace {
            memory: BindingMemorySummary::from_counts(0, 5),
            buffer_addresses: [1, 2, 3, 4, 5],
            algorithm: EngineSingleDecodeAlgorithm::Direct,
            handoff: EngineStreamHandoff::ExternalEventBridge,
            adapter_device_to_device_copies: 0,
        };
        assert!(trace.is_adapter_zero_copy());
        assert_eq!(trace.provider(), "loom-infer-cuda");
        assert_eq!(trace.operator(), "bf16_single_decode");
    }
}
