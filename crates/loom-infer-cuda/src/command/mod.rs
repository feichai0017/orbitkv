//! Checked CUDA resource bindings and stream-ordered command submission.

#![forbid(unsafe_code)]

mod binding;
mod completion;
mod resolve;
mod status;
mod submission;

pub use crate::device_status::DeviceStatusProtocolError;
pub use binding::{
    BindError, BindingElement, BindingMemorySummary, CheckedBindings, ErasedLease, Read, ReadWrite,
    TakeDeviceBufferError, Write,
};
pub use completion::CommandCompletion;
pub(crate) use completion::synchronize_stream_or_abort;
pub use completion::{CommandCompletionError, DeviceRejection};
use cuda_core::DriverError;
use loom_infer::ContractError;
pub(crate) use resolve::ResolvedRrww;
pub(crate) use submission::{
    CapturedCommandSet, CommandPermit, CompletionSlot, DeviceStatusReservation, QueueShared,
    RetainedResource,
};
pub use submission::{CommandQueue, CommandScope};
use thiserror::Error;

/// A failed scope admission with the caller's binding capability preserved.
pub struct CommandAdmissionError {
    cause: CommandError,
    bindings: Box<CheckedBindings>,
}

impl CommandAdmissionError {
    pub(crate) fn new(cause: CommandError, bindings: CheckedBindings) -> Self {
        Self {
            cause,
            bindings: Box::new(bindings),
        }
    }

    pub const fn cause(&self) -> &CommandError {
        &self.cause
    }

    pub fn into_parts(self) -> (CommandError, CheckedBindings) {
        (self.cause, *self.bindings)
    }
}

impl std::fmt::Debug for CommandAdmissionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CommandAdmissionError")
            .field("cause", &self.cause)
            .finish_non_exhaustive()
    }
}

impl std::fmt::Display for CommandAdmissionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.cause, formatter)
    }
}

impl std::error::Error for CommandAdmissionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
#[error("{provider} command submission failed with status {status}")]
pub struct ExternalCommandError {
    provider: &'static str,
    status: i32,
}

impl ExternalCommandError {
    pub const fn new(provider: &'static str, status: i32) -> Self {
        Self { provider, status }
    }

    pub const fn provider(self) -> &'static str {
        self.provider
    }

    pub const fn status(self) -> i32 {
        self.status
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SubmissionError {
    Driver(DriverError),
    External(ExternalCommandError),
}

impl SubmissionError {
    pub(super) const fn driver_error(self) -> Option<DriverError> {
        match self {
            Self::Driver(error) => Some(error),
            Self::External(_) => None,
        }
    }
}

impl From<SubmissionError> for CommandError {
    fn from(error: SubmissionError) -> Self {
        match error {
            SubmissionError::Driver(error) => Self::Driver(error),
            SubmissionError::External(error) => Self::External(error),
        }
    }
}

#[derive(Debug, Error)]
pub enum CommandError {
    #[error("command queues require capacity for at least one command")]
    ZeroCommandCapacity,
    #[error("command queues require capacity for at least one in-flight scope")]
    ZeroInFlightCapacity,
    #[error("checked bindings require capacity for at least one resource")]
    ZeroBindingCapacity,
    #[error("command queue identifier space is exhausted")]
    IdentifierSpaceExhausted,
    #[error("the checked bindings were created by a different command queue")]
    BindingsQueueMismatch,
    #[error("the command queue is poisoned by an earlier completion failure")]
    QueuePoisoned,
    /// The next admission consumes and reports exactly one queued rejection.
    /// A scope admitted earlier only peeks this error and leaves it for that
    /// acknowledgement path.
    #[error("an earlier dropped completion reported a device rejection: {0}")]
    UnobservedDeviceRejection(ContractError),
    #[error("the command scope is poisoned by an earlier submission failure")]
    ScopePoisoned,
    #[error("checked binding capacity {capacity} is exhausted")]
    BindingCapacityExceeded { capacity: usize },
    #[error(
        "device region belongs to CUDA device {region_device}, but the queue stream belongs to device {stream_device}"
    )]
    RegionContextMismatch {
        region_device: usize,
        stream_device: usize,
    },
    #[error(
        "device region overlaps binding slot {existing_slot}; any overlap involving a writer is rejected"
    )]
    OverlappingDeviceRegions { existing_slot: usize },
    #[error("the resource handle belongs to a different checked binding set")]
    BindingSetMismatch,
    #[error("binding slot {slot} is out of range for {bindings} bindings")]
    BindingSlotOutOfRange { slot: usize, bindings: usize },
    #[error("binding slot {slot} is read-only")]
    BindingIsReadOnly { slot: usize },
    #[error("binding slot {slot} has already been removed")]
    BindingSlotVacant { slot: usize },
    #[error("binding slot {slot} has a different element type than its resource handle")]
    BindingTypeMismatch { slot: usize },
    #[error("one command cannot use the same binding slot for multiple operands")]
    DuplicateBindingSlot,
    #[error("command scope capacity {capacity} is exhausted")]
    CommandCapacityExceeded { capacity: usize },
    #[error("command queue in-flight capacity {capacity} is exhausted")]
    InFlightCapacityExceeded { capacity: usize },
    #[error("device status storage capacity overflowed")]
    DeviceStatusCapacityOverflow,
    #[error("device status capacity {capacity} is exhausted")]
    DeviceStatusCapacityExceeded { capacity: usize },
    #[error("one device status workspace cannot be validated twice in the same command scope")]
    DuplicateDeviceStatusSource,
    #[error("device status source must contain at least one complete packet")]
    DeviceStatusSourceTooShort,
    #[error(transparent)]
    External(#[from] ExternalCommandError),
    #[error(transparent)]
    Driver(#[from] DriverError),
}
