//! Stream-ordered command admission, retention, and capture transfer.

use super::{
    CheckedBindings, CommandAdmissionError, CommandCompletion, CommandError, ExternalCommandError,
    SubmissionError, synchronize_stream_or_abort,
};
use crate::device_status::{DeviceStatusDecoder, STATUS_PACKET_WORDS};
use crate::memory::enqueue_status_packet_copy;
use cuda_core::{CudaEvent, CudaFunction, CudaStream, DriverError};
use oxide_infer::ContractError;
use std::any::Any;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// A reusable submission queue for one exact CUDA stream.
///
/// The queue preallocates one completion event per allowed in-flight scope.
/// Each finished scope owns its event until settlement, so later scopes may be
/// submitted immediately without re-recording an in-flight fence.
pub struct CommandQueue {
    pub(super) id: u64,
    pub(super) shared: Arc<QueueShared>,
    pub(super) max_commands: usize,
    pub(super) max_in_flight: usize,
}

pub(crate) struct QueueShared {
    pub(super) stream: Arc<CudaStream>,
    poisoned: AtomicBool,
    free_slots: Mutex<Vec<CompletionSlot>>,
    unobserved_rejections: Mutex<VecDeque<ContractError>>,
}

pub(crate) struct CompletionSlot {
    pub(super) event: CudaEvent,
    pub(super) retained_resources: Vec<RetainedResource>,
}

impl QueueShared {
    pub(super) fn is_poisoned(&self) -> bool {
        self.poisoned.load(Ordering::Acquire)
    }

    pub(super) fn poison(&self) {
        self.poisoned.store(true, Ordering::Release);
    }

    pub(super) fn take_slot(&self) -> Option<CompletionSlot> {
        lock_or_recover(&self.free_slots).pop()
    }

    pub(super) fn return_slot(&self, slot: CompletionSlot) {
        lock_or_recover(&self.free_slots).push(slot);
    }

    pub(super) fn record_unobserved_rejection(&self, error: ContractError) {
        let mut rejections = lock_or_recover(&self.unobserved_rejections);
        if rejections.len() == rejections.capacity() {
            self.poison();
            return;
        }
        rejections.push_back(error);
    }

    fn take_unobserved_rejection(&self) -> Option<ContractError> {
        lock_or_recover(&self.unobserved_rejections).pop_front()
    }

    fn first_unobserved_rejection(&self) -> Option<ContractError> {
        lock_or_recover(&self.unobserved_rejections)
            .front()
            .copied()
    }
}

impl CommandQueue {
    /// Creates a queue for `stream` with explicit per-scope and in-flight bounds.
    pub fn new(
        stream: Arc<CudaStream>,
        max_commands: usize,
        max_in_flight: usize,
    ) -> Result<Self, CommandError> {
        if max_commands == 0 {
            return Err(CommandError::ZeroCommandCapacity);
        }
        if max_in_flight == 0 {
            return Err(CommandError::ZeroInFlightCapacity);
        }

        let id = fresh_id()?;
        let mut free_slots = Vec::with_capacity(max_in_flight);
        for _ in 0..max_in_flight {
            free_slots.push(CompletionSlot {
                event: stream.context().new_event(None)?,
                retained_resources: Vec::with_capacity(max_commands),
            });
        }
        Ok(Self {
            id,
            shared: Arc::new(QueueShared {
                stream,
                poisoned: AtomicBool::new(false),
                free_slots: Mutex::new(free_slots),
                unobserved_rejections: Mutex::new(VecDeque::with_capacity(max_in_flight)),
            }),
            max_commands,
            max_in_flight,
        })
    }

    /// Returns the exact stream used by every scope from this queue.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.shared.stream
    }

    pub const fn max_commands(&self) -> usize {
        self.max_commands
    }

    pub const fn max_in_flight(&self) -> usize {
        self.max_in_flight
    }

    /// Creates reusable checked binding storage outside the enqueue path.
    pub fn bindings(&self, capacity: usize) -> Result<CheckedBindings, CommandError> {
        if capacity == 0 {
            return Err(CommandError::ZeroBindingCapacity);
        }

        Ok(CheckedBindings {
            queue_id: self.id,
            set_id: fresh_id()?,
            stream: self.shared.stream.clone(),
            leases: Vec::with_capacity(capacity),
            capacity,
            status: super::status::DeviceStatusState::new(
                self.shared.stream.context(),
                self.max_commands,
            )?,
        })
    }

    /// Begins one stream-ordered command scope.
    ///
    /// Admission never waits for CUDA completion. On failure,
    /// [`CommandAdmissionError`] preserves the complete binding capability for
    /// retry or explicit release.
    pub fn begin<'queue>(
        &'queue mut self,
        bindings: CheckedBindings,
    ) -> Result<CommandScope<'queue>, CommandAdmissionError> {
        if self.shared.is_poisoned() {
            return Err(CommandAdmissionError::new(
                CommandError::QueuePoisoned,
                bindings,
            ));
        }
        if bindings.queue_id != self.id
            || bindings.stream.cu_stream() != self.shared.stream.cu_stream()
            || bindings.stream.context().cu_ctx() != self.shared.stream.context().cu_ctx()
        {
            return Err(CommandAdmissionError::new(
                CommandError::BindingsQueueMismatch,
                bindings,
            ));
        }
        if let Some(error) = self.shared.take_unobserved_rejection() {
            return Err(CommandAdmissionError::new(
                CommandError::UnobservedDeviceRejection(error),
                bindings,
            ));
        }
        if !bindings.status.is_empty() {
            self.shared.poison();
            return Err(CommandAdmissionError::new(
                CommandError::QueuePoisoned,
                bindings,
            ));
        }

        let Some(slot) = self.shared.take_slot() else {
            return Err(CommandAdmissionError::new(
                CommandError::InFlightCapacityExceeded {
                    capacity: self.max_in_flight,
                },
                bindings,
            ));
        };

        let scope_id = match fresh_id() {
            Ok(scope_id) => scope_id,
            Err(error) => {
                self.shared.return_slot(slot);
                return Err(CommandAdmissionError::new(error, bindings));
            }
        };

        Ok(CommandScope {
            queue: Some(self),
            slot: Some(slot),
            bindings: Some(bindings),
            capture_resources: None,
            scope_id,
            submitted: 0,
            status_copies_submitted: 0,
            submission_error: None,
            finished: false,
        })
    }

    pub(crate) fn begin_capture<'queue>(
        &'queue mut self,
        bindings: CheckedBindings,
    ) -> Result<CommandScope<'queue>, CommandError> {
        if self.shared.is_poisoned() {
            return Err(CommandError::QueuePoisoned);
        }
        if bindings.queue_id != self.id
            || bindings.stream.cu_stream() != self.shared.stream.cu_stream()
            || bindings.stream.context().cu_ctx() != self.shared.stream.context().cu_ctx()
        {
            return Err(CommandError::BindingsQueueMismatch);
        }
        if let Some(error) = self.shared.take_unobserved_rejection() {
            return Err(CommandError::UnobservedDeviceRejection(error));
        }
        if !bindings.status.is_empty() {
            self.shared.poison();
            return Err(CommandError::QueuePoisoned);
        }
        let max_commands = self.max_commands;
        Ok(CommandScope {
            queue: Some(self),
            slot: None,
            bindings: Some(bindings),
            capture_resources: Some(Vec::with_capacity(max_commands)),
            scope_id: fresh_id()?,
            submitted: 0,
            status_copies_submitted: 0,
            submission_error: None,
            finished: false,
        })
    }
}

/// A stream-ordered sequence of commands with one final completion fence.
pub struct CommandScope<'queue> {
    pub(super) queue: Option<&'queue mut CommandQueue>,
    pub(super) slot: Option<CompletionSlot>,
    pub(super) bindings: Option<CheckedBindings>,
    pub(super) capture_resources: Option<Vec<RetainedResource>>,
    pub(super) scope_id: u64,
    pub(super) submitted: usize,
    pub(super) status_copies_submitted: usize,
    pub(super) submission_error: Option<SubmissionError>,
    pub(super) finished: bool,
}

impl<'queue> CommandScope<'queue> {
    /// Records one final fence and transfers all bindings to the completion.
    pub fn finish(mut self) -> CommandCompletion {
        self.enqueue_device_status_readbacks();
        let queue = self.queue.take().expect("live command scope has a queue");
        let slot = self
            .slot
            .take()
            .expect("eager command scope has a completion slot");
        let bindings = self
            .bindings
            .take()
            .expect("live command scope has bindings");
        let record_error = if self.submitted == 0 || self.submission_error.is_some() {
            None
        } else {
            slot.event.record(&queue.shared.stream).err()
        };
        if record_error.is_some() {
            queue.shared.poison();
        }

        self.finished = true;
        CommandCompletion::new(
            queue.shared.clone(),
            slot,
            bindings,
            self.submitted,
            self.submission_error,
            record_error,
        )
    }

    pub(crate) fn prepare_command(&self) -> Result<CommandPermit, CommandError> {
        self.require_command_capacity(1)?;
        let queue = self.queue.as_ref().expect("live command scope has a queue");
        Ok(CommandPermit {
            queue_id: queue.id,
            scope_id: self.scope_id,
            submission_index: self.submitted,
        })
    }

    pub(crate) fn require_command_capacity(
        &self,
        additional_commands: usize,
    ) -> Result<(), CommandError> {
        let queue = self.queue.as_ref().expect("live command scope has a queue");
        if queue.shared.is_poisoned() {
            return Err(CommandError::QueuePoisoned);
        }
        if let Some(error) = queue.shared.first_unobserved_rejection() {
            return Err(CommandError::UnobservedDeviceRejection(error));
        }
        if self.submission_error.is_some() {
            return Err(CommandError::ScopePoisoned);
        }
        let reserved_status_copies = self
            .bindings
            .as_ref()
            .expect("live command scope has bindings")
            .status
            .len()
            .saturating_sub(self.status_copies_submitted);
        let required = self
            .submitted
            .saturating_add(reserved_status_copies)
            .saturating_add(additional_commands);
        if required > queue.max_commands {
            Err(CommandError::CommandCapacityExceeded {
                capacity: queue.max_commands,
            })
        } else {
            Ok(())
        }
    }

    pub(crate) const fn submitted_commands(&self) -> usize {
        self.submitted
    }

    pub(crate) const fn scope_id(&self) -> u64 {
        self.scope_id
    }

    pub(crate) fn reserve_device_status(
        &mut self,
        source: super::Read<i32>,
        decoder: DeviceStatusDecoder,
    ) -> Result<DeviceStatusReservation, CommandError> {
        self.require_command_capacity(1)?;
        let source = self.resolve_device_status_source(source)?;
        let bindings = self
            .bindings
            .as_mut()
            .expect("live command scope has bindings");
        let index = bindings.status.reserve(source, decoder)?;
        Ok(DeviceStatusReservation {
            scope_id: self.scope_id,
            index,
        })
    }

    pub(crate) fn cancel_device_status(&mut self, reservation: DeviceStatusReservation) {
        assert_eq!(
            reservation.scope_id, self.scope_id,
            "device status reservation belongs to another command scope"
        );
        assert!(
            self.status_copies_submitted <= reservation.index,
            "a submitted device status cannot be cancelled"
        );
        self.bindings
            .as_mut()
            .expect("live command scope has bindings")
            .status
            .cancel_last(reservation.index);
    }

    pub(crate) fn finalize_device_status(&mut self) {
        self.enqueue_device_status_readbacks();
    }

    pub(crate) fn capture_error(&self) -> Option<CommandError> {
        self.submission_error.map(Into::into)
    }

    pub(crate) fn finish_capture(mut self) -> CapturedCommandSet {
        assert!(
            self.submission_error.is_none()
                && self.submitted > 0
                && self.status_copies_submitted
                    == self
                        .bindings
                        .as_ref()
                        .expect("live command scope has bindings")
                        .status
                        .len(),
            "only a non-empty healthy command scope may become a captured graph"
        );
        let queue = self.queue.take().expect("live command scope has a queue");
        let bindings = self
            .bindings
            .take()
            .expect("live command scope has bindings");
        assert!(
            self.slot.is_none(),
            "capture scope must not own an eager completion slot"
        );
        let resources = self
            .capture_resources
            .take()
            .expect("capture scope has capture resource storage");
        self.finished = true;
        CapturedCommandSet {
            stream: queue.shared.stream.clone(),
            bindings,
            resources,
            submitted: self.submitted,
        }
    }

    pub(crate) fn record_cuda_submission(&mut self, permit: CommandPermit, function: CudaFunction) {
        self.record_submission(
            permit,
            RetainedResource::Kernel {
                _function: function,
            },
        );
    }

    pub(crate) fn record_failed_cuda_submission(
        &mut self,
        permit: CommandPermit,
        function: CudaFunction,
        error: DriverError,
    ) {
        self.record_submission(
            permit,
            RetainedResource::Kernel {
                _function: function,
            },
        );
        self.queue
            .as_ref()
            .expect("live command scope has a queue")
            .shared
            .poison();
        self.submission_error = Some(SubmissionError::Driver(error));
    }

    pub(crate) fn record_external_submission<T>(&mut self, permit: CommandPermit, resource: Arc<T>)
    where
        T: Any + Send + Sync,
    {
        self.record_submission(
            permit,
            RetainedResource::External {
                _resource: resource,
            },
        );
    }

    pub(crate) fn record_failed_external_submission<T>(
        &mut self,
        permit: CommandPermit,
        resource: Arc<T>,
        error: ExternalCommandError,
    ) where
        T: Any + Send + Sync,
    {
        self.record_submission(
            permit,
            RetainedResource::External {
                _resource: resource,
            },
        );
        self.queue
            .as_ref()
            .expect("live command scope has a queue")
            .shared
            .poison();
        self.submission_error = Some(SubmissionError::External(error));
    }

    pub(crate) fn record_preflight_driver_failure(&mut self, error: DriverError) {
        self.queue
            .as_ref()
            .expect("live command scope has a queue")
            .shared
            .poison();
        self.submission_error = Some(SubmissionError::Driver(error));
    }

    fn record_submission(&mut self, permit: CommandPermit, resource: RetainedResource) {
        let queue = self.queue.as_ref().expect("live command scope has a queue");
        if permit.queue_id != queue.id
            || permit.scope_id != self.scope_id
            || permit.submission_index != self.submitted
            || self.retained_resources().len() != self.submitted
            || self.submitted >= queue.max_commands
        {
            abort_after_bookkeeping_invariant();
        }
        self.retained_resources_mut().push(resource);
        self.submitted += 1;
    }

    fn enqueue_device_status_readbacks(&mut self) {
        let status_count = self
            .bindings
            .as_ref()
            .expect("live command scope has bindings")
            .status
            .len();
        if self.submission_error.is_some() {
            return;
        }
        while self.status_copies_submitted < status_count {
            let index = self.status_copies_submitted;
            let permit = CommandPermit {
                queue_id: self
                    .queue
                    .as_ref()
                    .expect("live command scope has a queue")
                    .id,
                scope_id: self.scope_id,
                submission_index: self.submitted,
            };
            let result = {
                let queue = self.queue.as_ref().expect("live command scope has a queue");
                let bindings = self
                    .bindings
                    .as_mut()
                    .expect("live command scope has bindings");
                let pending = bindings.status.pending(index);
                enqueue_status_packet_copy(
                    &queue.shared.stream,
                    pending.source(),
                    bindings.status.host_mut(),
                    index * STATUS_PACKET_WORDS,
                    STATUS_PACKET_WORDS,
                )
            };
            self.record_submission(permit, RetainedResource::StatusCopy);
            self.status_copies_submitted += 1;
            if let Err(error) = result {
                self.queue
                    .as_ref()
                    .expect("live command scope has a queue")
                    .shared
                    .poison();
                self.submission_error = Some(SubmissionError::Driver(error));
                break;
            }
        }
    }
}

impl Drop for CommandScope<'_> {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        let Some(queue) = self.queue.as_ref() else {
            return;
        };
        let shared = queue.shared.clone();

        if self.submitted > 0 {
            let synchronize_error = synchronize_stream_or_abort(&shared.stream);
            if self.submission_error.is_some() || synchronize_error.is_some() {
                shared.poison();
            }
            let submission_driver_error = self
                .submission_error
                .and_then(SubmissionError::driver_error);
            self.retained_resources_mut().clear();
            if let Some(error) = submission_driver_error {
                shared.stream.context().record_err::<()>(Err(error));
            }
            if let Some(error) = synchronize_error {
                shared.stream.context().record_err::<()>(Err(error));
            }
        } else {
            self.retained_resources_mut().clear();
            if let Some(error) = self.submission_error {
                shared.poison();
                if let Some(error) = error.driver_error() {
                    shared.stream.context().record_err::<()>(Err(error));
                }
            }
        }
        if let Some(slot) = self.slot.take() {
            shared.return_slot(slot);
        }
    }
}

impl CommandScope<'_> {
    fn retained_resources(&self) -> &Vec<RetainedResource> {
        match self.slot.as_ref() {
            Some(slot) => &slot.retained_resources,
            None => self
                .capture_resources
                .as_ref()
                .expect("capture scope has capture resource storage"),
        }
    }

    fn retained_resources_mut(&mut self) -> &mut Vec<RetainedResource> {
        match self.slot.as_mut() {
            Some(slot) => &mut slot.retained_resources,
            None => self
                .capture_resources
                .as_mut()
                .expect("capture scope has capture resource storage"),
        }
    }
}

/// A single-use proof that one command fits in the preallocated queue.
pub(crate) struct CommandPermit {
    queue_id: u64,
    scope_id: u64,
    submission_index: usize,
}

pub(crate) struct DeviceStatusReservation {
    scope_id: u64,
    index: usize,
}

pub(crate) enum RetainedResource {
    Kernel {
        _function: CudaFunction,
    },
    External {
        _resource: Arc<dyn Any + Send + Sync>,
    },
    StatusCopy,
}

pub(crate) struct CapturedCommandSet {
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) bindings: CheckedBindings,
    pub(crate) resources: Vec<RetainedResource>,
    pub(crate) submitted: usize,
}

pub(super) fn fresh_id() -> Result<u64, CommandError> {
    NEXT_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .map_err(|_| CommandError::IdentifierSpaceExhausted)
}

fn abort_after_bookkeeping_invariant() -> ! {
    eprintln!(
        "oxide-infer-cuda detected an internal command-accounting violation after GPU submission; \
         aborting to preserve resource safety"
    );
    std::process::abort()
}

fn lock_or_recover<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
