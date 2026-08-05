//! Stream-ordered command admission, retention, and capture transfer.

use super::{
    CheckedBindings, CommandCompletion, CommandError, ExternalCommandError, SubmissionError,
    synchronize_stream_or_abort,
};
use cuda_core::{CudaEvent, CudaFunction, CudaStream, DriverError};
use std::any::Any;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// A reusable submission queue for one exact CUDA stream.
///
/// The queue preallocates its completion event and resource-retention storage.
/// Rust's mutable borrow rules prevent a second scope from re-recording the
/// event while an earlier completion is still alive.
pub struct CommandQueue {
    pub(super) id: u64,
    pub(super) stream: Arc<CudaStream>,
    pub(super) completion_event: CudaEvent,
    pub(super) retained_resources: Vec<RetainedResource>,
    pub(super) max_commands: usize,
    pub(super) poisoned: bool,
}

impl CommandQueue {
    /// Creates a queue for `stream` with storage for at most `max_commands`
    /// commands per scope.
    pub fn new(stream: Arc<CudaStream>, max_commands: usize) -> Result<Self, CommandError> {
        if max_commands == 0 {
            return Err(CommandError::ZeroCommandCapacity);
        }

        let id = fresh_id()?;
        let completion_event = stream.context().new_event(None)?;
        Ok(Self {
            id,
            stream,
            completion_event,
            retained_resources: Vec::with_capacity(max_commands),
            max_commands,
            poisoned: false,
        })
    }

    /// Returns the exact stream used by every scope from this queue.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub const fn max_commands(&self) -> usize {
        self.max_commands
    }

    /// Creates reusable checked binding storage outside the enqueue path.
    pub fn bindings(&self, capacity: usize) -> Result<CheckedBindings, CommandError> {
        if capacity == 0 {
            return Err(CommandError::ZeroBindingCapacity);
        }

        Ok(CheckedBindings {
            queue_id: self.id,
            set_id: fresh_id()?,
            stream: self.stream.clone(),
            leases: Vec::with_capacity(capacity),
            capacity,
        })
    }

    /// Begins one stream-ordered command scope.
    pub fn begin<'queue>(
        &'queue mut self,
        bindings: CheckedBindings,
    ) -> Result<CommandScope<'queue>, CommandError> {
        if self.poisoned {
            return Err(CommandError::QueuePoisoned);
        }
        if bindings.queue_id != self.id
            || bindings.stream.cu_stream() != self.stream.cu_stream()
            || bindings.stream.context().cu_ctx() != self.stream.context().cu_ctx()
        {
            return Err(CommandError::BindingsQueueMismatch);
        }
        if !self.retained_resources.is_empty() {
            self.poisoned = true;
            return Err(CommandError::QueuePoisoned);
        }

        Ok(CommandScope {
            queue: Some(self),
            bindings: Some(bindings),
            scope_id: fresh_id()?,
            submitted: 0,
            submission_error: None,
            finished: false,
        })
    }
}

impl Drop for CommandQueue {
    fn drop(&mut self) {
        if self.retained_resources.is_empty() {
            return;
        }

        let synchronize_error = synchronize_stream_or_abort(&self.stream);
        self.retained_resources.clear();
        if let Some(error) = synchronize_error {
            eprintln!(
                "loom-infer-cuda command queue synchronized after a stream error during drop: \
                 {error}"
            );
        }
    }
}

/// A stream-ordered sequence of commands with one final completion fence.
pub struct CommandScope<'queue> {
    pub(super) queue: Option<&'queue mut CommandQueue>,
    pub(super) bindings: Option<CheckedBindings>,
    pub(super) scope_id: u64,
    pub(super) submitted: usize,
    pub(super) submission_error: Option<SubmissionError>,
    pub(super) finished: bool,
}

impl<'queue> CommandScope<'queue> {
    /// Records one final fence and transfers all bindings to the completion.
    pub fn finish(mut self) -> CommandCompletion<'queue> {
        let queue = self.queue.take().expect("live command scope has a queue");
        let bindings = self
            .bindings
            .take()
            .expect("live command scope has bindings");
        let record_error = if self.submitted == 0 || self.submission_error.is_some() {
            None
        } else {
            queue.completion_event.record(&queue.stream).err()
        };

        self.finished = true;
        CommandCompletion::new(
            queue,
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
        if self.submission_error.is_some() {
            return Err(CommandError::ScopePoisoned);
        }
        let queue = self.queue.as_ref().expect("live command scope has a queue");
        let required = self.submitted.saturating_add(additional_commands);
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

    pub(crate) fn capture_error(&self) -> Option<CommandError> {
        self.submission_error.map(Into::into)
    }

    pub(crate) fn finish_capture(mut self) -> CapturedCommandSet {
        assert!(
            self.submission_error.is_none() && self.submitted > 0,
            "only a non-empty healthy command scope may become a captured graph"
        );
        let queue = self.queue.take().expect("live command scope has a queue");
        let bindings = self
            .bindings
            .take()
            .expect("live command scope has bindings");
        let resources = std::mem::replace(
            &mut queue.retained_resources,
            Vec::with_capacity(queue.max_commands),
        );
        self.finished = true;
        CapturedCommandSet {
            stream: queue.stream.clone(),
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
            .as_mut()
            .expect("live command scope has a queue")
            .poisoned = true;
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
            .as_mut()
            .expect("live command scope has a queue")
            .poisoned = true;
        self.submission_error = Some(SubmissionError::External(error));
    }

    pub(crate) fn record_preflight_driver_failure(&mut self, error: DriverError) {
        self.queue
            .as_mut()
            .expect("live command scope has a queue")
            .poisoned = true;
        self.submission_error = Some(SubmissionError::Driver(error));
    }

    fn record_submission(&mut self, permit: CommandPermit, resource: RetainedResource) {
        let queue = self.queue.as_mut().expect("live command scope has a queue");
        if permit.queue_id != queue.id
            || permit.scope_id != self.scope_id
            || permit.submission_index != self.submitted
            || queue.retained_resources.len() != self.submitted
            || self.submitted >= queue.max_commands
        {
            abort_after_bookkeeping_invariant();
        }
        queue.retained_resources.push(resource);
        self.submitted += 1;
    }
}

impl Drop for CommandScope<'_> {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        let Some(queue) = self.queue.as_mut() else {
            return;
        };

        if self.submitted > 0 {
            let synchronize_error = synchronize_stream_or_abort(&queue.stream);
            if self.submission_error.is_some() || synchronize_error.is_some() {
                queue.poisoned = true;
            }
            let submission_driver_error = self
                .submission_error
                .and_then(SubmissionError::driver_error);
            queue.retained_resources.clear();
            if let Some(error) = submission_driver_error {
                queue.stream.context().record_err::<()>(Err(error));
            }
            if let Some(error) = synchronize_error {
                queue.stream.context().record_err::<()>(Err(error));
            }
        } else {
            queue.retained_resources.clear();
            if let Some(error) = self.submission_error {
                queue.poisoned = true;
                if let Some(error) = error.driver_error() {
                    queue.stream.context().record_err::<()>(Err(error));
                }
            }
        }
    }
}

/// A single-use proof that one command fits in the preallocated queue.
pub(crate) struct CommandPermit {
    queue_id: u64,
    scope_id: u64,
    submission_index: usize,
}

pub(crate) enum RetainedResource {
    Kernel {
        _function: CudaFunction,
    },
    External {
        _resource: Arc<dyn Any + Send + Sync>,
    },
}

pub(crate) struct CapturedCommandSet {
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) bindings: CheckedBindings,
    pub(crate) resources: Vec<RetainedResource>,
    pub(crate) submitted: usize,
}

fn fresh_id() -> Result<u64, CommandError> {
    NEXT_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .map_err(|_| CommandError::IdentifierSpaceExhausted)
}

fn abort_after_bookkeeping_invariant() -> ! {
    eprintln!(
        "loom-infer-cuda detected an internal command-accounting violation after GPU submission; \
         aborting to preserve resource safety"
    );
    std::process::abort()
}
