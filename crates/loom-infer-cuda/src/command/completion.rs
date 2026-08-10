//! Completion fences and conservative CUDA quiescence handling.

use super::{CheckedBindings, CommandError, CompletionSlot, QueueShared, SubmissionError};
use crate::device_status::DeviceStatusProtocolError;
use cuda_core::{CudaStream, DriverError};
use loom_infer::ContractError;
use std::fmt::{self, Display, Formatter};
use std::sync::Arc;
use thiserror::Error;

/// The final fence and all bindings retained by a completed command scope.
///
/// Forgetting this value is memory-safe but leaks its slot, retained bindings,
/// and one shared queue reference. The queue permanently loses that capacity.
/// It never waits or allocates to replace the slot, and it cannot observe or
/// propagate errors from forgotten work.
#[must_use = "dropping the completion waits before releasing CUDA resources"]
pub struct CommandCompletion {
    shared: Arc<QueueShared>,
    slot: Option<CompletionSlot>,
    bindings: Option<CheckedBindings>,
    submitted: usize,
    submission_error: Option<SubmissionError>,
    record_error: Option<DriverError>,
    poll_error: Option<DriverError>,
    complete: bool,
}

impl CommandCompletion {
    pub(super) fn new(
        shared: Arc<QueueShared>,
        slot: CompletionSlot,
        bindings: CheckedBindings,
        submitted: usize,
        submission_error: Option<SubmissionError>,
        record_error: Option<DriverError>,
    ) -> Self {
        Self {
            shared,
            slot: Some(slot),
            bindings: Some(bindings),
            submitted,
            submission_error,
            record_error,
            poll_error: None,
            complete: false,
        }
    }

    /// Returns whether all submitted commands have completed without blocking.
    pub fn is_complete(&mut self) -> Result<bool, CommandError> {
        if let Some(error) = self.submission_error {
            return Err(error.into());
        }
        if let Some(error) = self.record_error {
            return Err(error.into());
        }
        if let Some(error) = self.poll_error {
            return Err(error.into());
        }
        if self.submitted == 0 {
            return Ok(true);
        }
        let slot = self.slot.as_ref().expect("live completion has a slot");
        match slot.event.query() {
            Ok(complete) => Ok(complete),
            Err(error) => {
                self.poll_error = Some(error);
                self.shared.poison();
                Err(error.into())
            }
        }
    }

    /// Waits once and returns the reusable checked bindings.
    pub fn wait(mut self) -> Result<CheckedBindings, CommandCompletionError> {
        match self.settle() {
            None => {
                let status = self
                    .bindings
                    .as_ref()
                    .expect("live completion has bindings")
                    .status
                    .decode();
                match status {
                    Ok(None) => {
                        let mut bindings =
                            self.bindings.take().expect("live completion has bindings");
                        bindings.status.clear();
                        self.complete_and_return_slot();
                        Ok(bindings)
                    }
                    Ok(Some(error)) => {
                        let mut bindings =
                            self.bindings.take().expect("live completion has bindings");
                        bindings.status.clear();
                        self.complete_and_return_slot();
                        Err(CommandCompletionError::DeviceRejected(
                            DeviceRejection::new(error, bindings),
                        ))
                    }
                    Err(error) => {
                        self.shared.poison();
                        self.bindings.take();
                        self.complete_and_return_slot();
                        Err(CommandCompletionError::StatusProtocol(error))
                    }
                }
            }
            Some(failure) => {
                self.shared.poison();
                record_settlement_errors(&self.shared, failure);
                self.bindings.take();
                self.complete_and_return_slot();
                Err(CommandCompletionError::Execution(failure.command_error()))
            }
        }
    }

    /// Number of commands covered by this one completion fence.
    pub const fn submitted(&self) -> usize {
        self.submitted
    }

    fn settle(&self) -> Option<SettlementFailure> {
        if self.submitted == 0 {
            return self.submission_error.map(|reported| SettlementFailure {
                reported,
                synchronize_error: None,
            });
        }
        if let Some(submission_error) = self.submission_error {
            return Some(SettlementFailure {
                reported: submission_error,
                synchronize_error: synchronize_stream_or_abort(&self.shared.stream),
            });
        }
        if let Some(record_error) = self.record_error {
            return Some(SettlementFailure {
                reported: SubmissionError::Driver(record_error),
                synchronize_error: synchronize_stream_or_abort(&self.shared.stream),
            });
        }
        if let Some(poll_error) = self.poll_error {
            return Some(SettlementFailure {
                reported: SubmissionError::Driver(poll_error),
                synchronize_error: synchronize_stream_or_abort(&self.shared.stream),
            });
        }
        let slot = self.slot.as_ref().expect("live completion has a slot");
        match slot.event.synchronize() {
            Ok(()) => None,
            Err(event_error) => Some(SettlementFailure {
                reported: SubmissionError::Driver(event_error),
                synchronize_error: synchronize_stream_or_abort(&self.shared.stream),
            }),
        }
    }

    fn complete_and_return_slot(&mut self) {
        let mut slot = self.slot.take().expect("live completion has a slot");
        slot.retained_resources.clear();
        self.shared.return_slot(slot);
        self.complete = true;
    }
}

#[derive(Debug, Error)]
pub enum CommandCompletionError {
    #[error(transparent)]
    Execution(CommandError),
    #[error(transparent)]
    DeviceRejected(DeviceRejection),
    #[error(transparent)]
    StatusProtocol(DeviceStatusProtocolError),
}

impl CommandCompletionError {
    pub const fn execution_error(&self) -> Option<&CommandError> {
        match self {
            Self::Execution(error) => Some(error),
            Self::DeviceRejected(_) | Self::StatusProtocol(_) => None,
        }
    }

    pub const fn device_rejection(&self) -> Option<&DeviceRejection> {
        match self {
            Self::DeviceRejected(rejection) => Some(rejection),
            Self::Execution(_) | Self::StatusProtocol(_) => None,
        }
    }

    pub fn into_device_rejection(self) -> Option<DeviceRejection> {
        match self {
            Self::DeviceRejected(rejection) => Some(rejection),
            Self::Execution(_) | Self::StatusProtocol(_) => None,
        }
    }
}

/// A device-side contract rejection with all completed bindings recovered.
pub struct DeviceRejection {
    error: ContractError,
    bindings: Box<CheckedBindings>,
}

impl DeviceRejection {
    pub(crate) fn new(error: ContractError, bindings: CheckedBindings) -> Self {
        Self {
            error,
            bindings: Box::new(bindings),
        }
    }

    pub const fn error(&self) -> ContractError {
        self.error
    }

    pub fn into_parts(self) -> (ContractError, CheckedBindings) {
        (self.error, *self.bindings)
    }
}

impl fmt::Debug for DeviceRejection {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DeviceRejection")
            .field("error", &self.error)
            .finish_non_exhaustive()
    }
}

impl Display for DeviceRejection {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "device rejected command metadata: {}",
            self.error
        )
    }
}

impl std::error::Error for DeviceRejection {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

impl Drop for CommandCompletion {
    fn drop(&mut self) {
        if self.complete {
            return;
        }
        let result = self.settle();
        if let Some(failure) = result {
            self.shared.poison();
            record_settlement_errors(&self.shared, failure);
        } else {
            match self
                .bindings
                .as_ref()
                .expect("live completion has bindings")
                .status
                .decode()
            {
                Ok(Some(error)) => self.shared.record_unobserved_rejection(error),
                Ok(None) => {}
                Err(_) => self.shared.poison(),
            }
        }
        self.bindings.take();
        self.complete_and_return_slot();
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SettlementFailure {
    pub(super) reported: SubmissionError,
    synchronize_error: Option<DriverError>,
}

impl SettlementFailure {
    pub(super) fn command_error(self) -> CommandError {
        self.reported.into()
    }
}

fn record_settlement_errors(shared: &QueueShared, failure: SettlementFailure) {
    if let Some(error) = failure.synchronize_error {
        shared.stream.context().record_err::<()>(Err(error));
    }
    if let Some(error) = failure.reported.driver_error() {
        shared.stream.context().record_err::<()>(Err(error));
    }
}

pub(crate) fn synchronize_stream_or_abort(stream: &CudaStream) -> Option<DriverError> {
    match stream.synchronize() {
        Ok(()) => None,
        Err(stream_error) => match stream.context().synchronize() {
            Ok(()) => Some(stream_error),
            Err(context_error) => abort_after_sync_failure(stream_error, context_error),
        },
    }
}

fn abort_after_sync_failure(stream_error: DriverError, context_error: DriverError) -> ! {
    eprintln!(
        "loom-infer-cuda cannot confirm CUDA quiescence after stream and context synchronization \
         failed; aborting to preserve resource safety: stream={stream_error}; \
         context={context_error}"
    );
    std::process::abort()
}
