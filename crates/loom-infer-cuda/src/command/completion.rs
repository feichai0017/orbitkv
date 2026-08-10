//! Completion fences and conservative CUDA quiescence handling.

use super::{CheckedBindings, CommandError, CommandQueue, SubmissionError};
use crate::device_status::DeviceStatusProtocolError;
use cuda_core::{CudaStream, DriverError};
use loom_infer::ContractError;
use std::fmt::{self, Display, Formatter};
use thiserror::Error;

/// The final fence and all bindings retained by a completed command scope.
#[must_use = "dropping the completion waits before releasing CUDA resources"]
pub struct CommandCompletion<'queue> {
    queue: Option<&'queue mut CommandQueue>,
    bindings: Option<CheckedBindings>,
    submitted: usize,
    submission_error: Option<SubmissionError>,
    record_error: Option<DriverError>,
    poll_error: Option<DriverError>,
    complete: bool,
}

impl<'queue> CommandCompletion<'queue> {
    pub(super) fn new(
        queue: &'queue mut CommandQueue,
        bindings: CheckedBindings,
        submitted: usize,
        submission_error: Option<SubmissionError>,
        record_error: Option<DriverError>,
    ) -> Self {
        Self {
            queue: Some(queue),
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
        let queue = self.queue.as_ref().expect("live completion has a queue");
        match queue.completion_event.query() {
            Ok(complete) => Ok(complete),
            Err(error) => {
                self.poll_error = Some(error);
                Err(error.into())
            }
        }
    }

    /// Waits once and returns the reusable checked bindings.
    pub fn wait(mut self) -> Result<CheckedBindings, CommandCompletionError> {
        match self.settle() {
            None => {
                self.complete = true;
                let queue = self.queue.as_mut().expect("live completion has a queue");
                queue.retained_resources.clear();
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
                        Ok(bindings)
                    }
                    Ok(Some(error)) => {
                        let mut bindings =
                            self.bindings.take().expect("live completion has bindings");
                        bindings.status.clear();
                        Err(CommandCompletionError::DeviceRejected(
                            DeviceRejection::new(error, bindings),
                        ))
                    }
                    Err(error) => {
                        queue.poisoned = true;
                        self.bindings.take();
                        Err(CommandCompletionError::StatusProtocol(error))
                    }
                }
            }
            Some(failure) => {
                self.complete = true;
                let queue = self.queue.as_mut().expect("live completion has a queue");
                queue.poisoned = true;
                queue.retained_resources.clear();
                record_settlement_errors(queue, failure);
                self.bindings.take();
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
        let queue = self.queue.as_ref().expect("live completion has a queue");
        if let Some(submission_error) = self.submission_error {
            return Some(SettlementFailure {
                reported: submission_error,
                synchronize_error: synchronize_stream_or_abort(&queue.stream),
            });
        }
        if let Some(record_error) = self.record_error {
            return Some(SettlementFailure {
                reported: SubmissionError::Driver(record_error),
                synchronize_error: synchronize_stream_or_abort(&queue.stream),
            });
        }
        if let Some(poll_error) = self.poll_error {
            return Some(SettlementFailure {
                reported: SubmissionError::Driver(poll_error),
                synchronize_error: synchronize_stream_or_abort(&queue.stream),
            });
        }
        match queue.completion_event.synchronize() {
            Ok(()) => None,
            Err(event_error) => Some(SettlementFailure {
                reported: SubmissionError::Driver(event_error),
                synchronize_error: synchronize_stream_or_abort(&queue.stream),
            }),
        }
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

impl Drop for CommandCompletion<'_> {
    fn drop(&mut self) {
        if self.complete {
            return;
        }
        let result = self.settle();
        let queue = self.queue.as_mut().expect("live completion has a queue");
        if let Some(failure) = result {
            queue.poisoned = true;
            queue.retained_resources.clear();
            record_settlement_errors(queue, failure);
        } else {
            queue.retained_resources.clear();
        }
        self.complete = true;
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

fn record_settlement_errors(queue: &CommandQueue, failure: SettlementFailure) {
    if let Some(error) = failure.synchronize_error {
        queue.stream.context().record_err::<()>(Err(error));
    }
    if let Some(error) = failure.reported.driver_error() {
        queue.stream.context().record_err::<()>(Err(error));
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
