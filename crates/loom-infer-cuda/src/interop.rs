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
    CommandQueue, synchronize_stream_or_abort,
};
use cuda_core::sys::{CUcontext, CUdeviceptr, CUgreenCtx, CUstream};
use cuda_core::{CudaContext, CudaEvent, CudaStream, DriverError, IntoResult};
use loom_infer::ContractError;
use std::any::Any;
use std::fmt::{self, Display, Formatter};
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
    /// The caller must serialize engine submissions with this adapter. For
    /// each [`EngineInteropQueue::enqueue_bf16_single_decode`] call, no other
    /// thread may enqueue work on `raw` from the adapter's pre-event record
    /// until the method has enqueued the post-event wait. The method returns
    /// only after that critical section ends.
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
        let queue = CommandQueue::new(loom_stream.clone(), max_commands)?;
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
    /// The unsafe construction contract of [`ExternalCudaStream`] requires
    /// exclusive external-stream submission authority for this method's
    /// duration. On success, the post-event wait is already enqueued before
    /// this method returns. Every error path conservatively settles both
    /// streams before releasing a lease. Bindings are returned in the error
    /// whenever command settlement can recover them safely. No post-wait is
    /// promised after an error.
    pub fn enqueue_bf16_single_decode<'queue>(
        &'queue mut self,
        plan: &Bf16SingleDecodePlan,
        bindings: CheckedBindings,
        args: Bf16SingleDecodeArgs,
    ) -> Result<EngineCommandCompletion<'queue>, EngineSingleDecodeEnqueueError> {
        let loom_stream = self.loom_stream.clone();
        if self.poisoned {
            settle_bridge_streams(&self.external, &loom_stream, &mut self.poisoned);
            return Err(EngineSingleDecodeEnqueueError::recovered(
                EngineSingleDecodeEnqueueCause::QueuePoisoned,
                bindings,
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
            return Err(EngineSingleDecodeEnqueueError::recovered(cause, bindings));
        };
        let trace = EngineExecutionTrace {
            memory: bindings.memory_summary(),
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
                bindings,
            ));
        }
        if let Err(error) = loom_stream.wait(&self.pre_event) {
            self.poisoned = true;
            settle_bridge_streams(&self.external, &loom_stream, &mut self.poisoned);
            return Err(EngineSingleDecodeEnqueueError::recovered(
                EngineSingleDecodeEnqueueCause::Bridge(error),
                bindings,
            ));
        }

        let mut scope = match self.queue.begin_recover(bindings) {
            Ok(scope) => scope,
            Err((error, bindings)) => {
                settle_bridge_streams(&self.external, &loom_stream, &mut self.poisoned);
                return Err(EngineSingleDecodeEnqueueError::recovered(
                    EngineSingleDecodeEnqueueCause::Command(error),
                    *bindings,
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
            ));
        }

        Ok(EngineCommandCompletion {
            command: completion,
            trace,
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

/// A settled enqueue failure with recovered bindings when safe.
pub struct EngineSingleDecodeEnqueueError {
    cause: EngineSingleDecodeEnqueueCause,
    bindings: Option<Box<CheckedBindings>>,
}

impl EngineSingleDecodeEnqueueError {
    fn recovered(cause: EngineSingleDecodeEnqueueCause, bindings: CheckedBindings) -> Self {
        Self {
            cause,
            bindings: Some(Box::new(bindings)),
        }
    }

    fn unrecoverable(cause: EngineSingleDecodeEnqueueCause) -> Self {
        Self {
            cause,
            bindings: None,
        }
    }

    pub const fn cause(&self) -> &EngineSingleDecodeEnqueueCause {
        &self.cause
    }

    /// Returns bindings only after both bridge streams are quiescent.
    pub fn recovered_bindings(&self) -> Option<&CheckedBindings> {
        self.bindings.as_deref()
    }

    pub fn into_parts(self) -> (EngineSingleDecodeEnqueueCause, Option<CheckedBindings>) {
        (self.cause, self.bindings.map(|bindings| *bindings))
    }
}

impl fmt::Debug for EngineSingleDecodeEnqueueError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EngineSingleDecodeEnqueueError")
            .field("cause", &self.cause)
            .field("bindings_recovered", &self.bindings.is_some())
            .finish()
    }
}

impl Display for EngineSingleDecodeEnqueueError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.cause, formatter)
    }
}

impl std::error::Error for EngineSingleDecodeEnqueueError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

/// One in-flight provider command and its external-stream handoff evidence.
#[must_use = "dropping the completion waits before releasing buffer and stream leases"]
pub struct EngineCommandCompletion<'queue> {
    command: CommandCompletion<'queue>,
    trace: EngineExecutionTrace,
}

impl EngineCommandCompletion<'_> {
    pub const fn trace(&self) -> &EngineExecutionTrace {
        &self.trace
    }

    pub const fn submitted(&self) -> usize {
        self.command.submitted()
    }

    /// Waits for Loom execution and returns the reusable external bindings.
    pub fn wait(self) -> Result<EngineCommandOutcome, EngineCommandCompletionError> {
        let trace = self.trace;
        match self.command.wait() {
            Ok(bindings) => Ok(EngineCommandOutcome { bindings, trace }),
            Err(source) => Err(EngineCommandCompletionError { source, trace }),
        }
    }
}

/// Completed external bindings plus the provider trace for their invocation.
pub struct EngineCommandOutcome {
    bindings: CheckedBindings,
    trace: EngineExecutionTrace,
}

impl EngineCommandOutcome {
    pub const fn trace(&self) -> &EngineExecutionTrace {
        &self.trace
    }

    pub fn into_bindings(self) -> CheckedBindings {
        self.bindings
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

    pub fn into_source(self) -> CommandCompletionError {
        self.source
    }
}

fn settle_bridge_failure(
    external: &ExternalCudaStream,
    loom_stream: &CudaStream,
    poisoned: &mut bool,
    completion: CommandCompletion<'_>,
    bridge: DriverError,
) -> EngineSingleDecodeEnqueueError {
    let result = completion.wait();
    settle_bridge_streams(external, loom_stream, poisoned);
    match result {
        Ok(bindings) => EngineSingleDecodeEnqueueError::recovered(
            EngineSingleDecodeEnqueueCause::Bridge(bridge),
            bindings,
        ),
        Err(CommandCompletionError::DeviceRejected(rejection)) => {
            let (device, bindings) = rejection.into_parts();
            EngineSingleDecodeEnqueueError::recovered(
                EngineSingleDecodeEnqueueCause::BridgeAndDeviceRejection { bridge, device },
                bindings,
            )
        }
        Err(completion) => EngineSingleDecodeEnqueueError::unrecoverable(
            EngineSingleDecodeEnqueueCause::BridgeAndCompletion {
                bridge,
                completion: Box::new(completion),
            },
        ),
    }
}

fn settle_provider_failure(
    external: &ExternalCudaStream,
    loom_stream: &CudaStream,
    poisoned: &mut bool,
    completion: CommandCompletion<'_>,
    provider: SingleDecodeEnqueueError,
) -> EngineSingleDecodeEnqueueError {
    let result = completion.wait();
    settle_bridge_streams(external, loom_stream, poisoned);
    match result {
        Ok(bindings) => EngineSingleDecodeEnqueueError::recovered(
            EngineSingleDecodeEnqueueCause::Provider(provider),
            bindings,
        ),
        Err(CommandCompletionError::DeviceRejected(rejection)) => {
            let (device, bindings) = rejection.into_parts();
            EngineSingleDecodeEnqueueError::recovered(
                EngineSingleDecodeEnqueueCause::ProviderAndDeviceRejection { provider, device },
                bindings,
            )
        }
        Err(completion) => EngineSingleDecodeEnqueueError::unrecoverable(
            EngineSingleDecodeEnqueueCause::ProviderAndCompletion {
                provider,
                completion: Box::new(completion),
            },
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
