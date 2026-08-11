//! Fixed-address CUDA Graph capture and replay.
//!
//! The first graph contract deliberately excludes rebinding, node updates,
//! cross-stream launch, concurrent replay, and default-stream capture. A graph
//! owns its checked buffer bindings and every submitted kernel or vendor plan
//! until the executable graph is destroyed.

use crate::command::{
    CapturedCommandSet, CheckedBindings, CommandError, CommandQueue, CommandScope, DeviceRejection,
    RetainedResource, synchronize_stream_or_abort,
};
use crate::device_status::DeviceStatusProtocolError;
use crate::driver::bind_context_for_cleanup;
use cuda_core::{CudaContext, CudaEvent, CudaStream, DriverError, IntoResult, sys};
use oxide_infer::ContractError;
use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::rc::Rc;
use std::sync::Arc;
use thiserror::Error;

/// A capture-only queue with a private CUDA stream.
///
/// The stream handle is intentionally not exposed. This prevents safe caller
/// code from inserting untracked CUDA work into a capture while the closure is
/// recording checked Oxide commands.
pub struct GraphQueue {
    queue: CommandQueue,
    not_send_sync: PhantomData<Rc<()>>,
}

impl GraphQueue {
    /// Creates a graph queue whose stream is private to Oxide.
    pub fn new(context: &Arc<CudaContext>, max_commands: usize) -> Result<Self, CommandError> {
        let stream = context.new_stream().map_err(CommandError::from)?;
        Ok(Self {
            queue: CommandQueue::new(stream, max_commands, 1)?,
            not_send_sync: PhantomData,
        })
    }

    /// Creates fixed binding storage for this exact graph queue.
    pub fn bindings(&self, capacity: usize) -> Result<CheckedBindings, CommandError> {
        self.queue.bindings(capacity)
    }

    /// Consumes this one-shot queue and captures one non-empty checked command
    /// sequence.
    ///
    /// The first contract establishes input readiness with one context
    /// synchronization before capture. Consuming the queue prevents safe code
    /// from starting another capture on a stream retained by an earlier graph.
    /// Replay remains asynchronous.
    pub fn capture<E, F>(
        mut self,
        bindings: CheckedBindings,
        record: F,
    ) -> Result<CapturedGraph, CaptureError<E>>
    where
        F: FnOnce(&mut CommandScope<'_>) -> Result<(), E>,
    {
        if let Err(error) = self.queue.stream().context().synchronize() {
            eprintln!(
                "oxide-infer-cuda cannot establish CUDA Graph input readiness: {error}; \
                 aborting before releasing caller-supplied allocations"
            );
            std::process::abort();
        }
        self.queue.capture_checked(bindings, record)
    }
}

impl CommandQueue {
    /// Captures one non-empty command sequence with fixed buffer bindings.
    ///
    /// Planning, allocation, tuning, and host synchronization must happen
    /// outside `record`. The closure may enqueue only commands accepted by the
    /// ordinary checked [`CommandScope`] lifecycle.
    fn capture_checked<E, F>(
        &mut self,
        bindings: CheckedBindings,
        record: F,
    ) -> Result<CapturedGraph, CaptureError<E>>
    where
        F: FnOnce(&mut CommandScope<'_>) -> Result<(), E>,
    {
        let stream = self.stream().clone();
        if stream.cu_stream().is_null() {
            return Err(CaptureError::Graph(GraphError::DefaultStreamUnsupported));
        }

        let mut commands = self
            .begin_capture(bindings)
            .map_err(CaptureError::Command)?;
        begin_capture(&stream).map_err(CaptureError::Graph)?;
        let guard = CaptureGuard::new(stream);

        let record_result = record(&mut commands);
        if record_result.is_ok()
            && commands.capture_error().is_none()
            && commands.submitted_commands() > 0
        {
            commands.finalize_device_status();
        }
        let graph_result = guard.finish();

        if let Err(error) = record_result {
            match graph_result {
                Ok(graph) => drop(graph),
                Err(graph_error) => return Err(CaptureError::Graph(graph_error)),
            }
            return Err(CaptureError::Record(error));
        }

        let graph = graph_result.map_err(CaptureError::Graph)?;
        if let Some(error) = commands.capture_error() {
            drop(graph);
            return Err(CaptureError::Command(error));
        }
        if commands.submitted_commands() == 0 {
            drop(graph);
            return Err(CaptureError::Graph(GraphError::EmptyCapture));
        }

        let CapturedCommandSet {
            stream,
            bindings,
            resources,
            submitted,
        } = commands.finish_capture();
        Ok(CapturedGraph {
            graph: Some(graph),
            stream,
            bindings: Some(bindings),
            resources,
            commands: submitted,
            not_send_sync: PhantomData,
        })
    }
}

/// An error produced while the caller records a graph.
#[derive(Debug)]
pub enum CaptureError<E> {
    /// The checked command lifecycle rejected the scope or a submission.
    Command(CommandError),
    /// The caller's recording closure returned an error.
    Record(E),
    /// CUDA Graph capture or ownership failed.
    Graph(GraphError),
}

impl<E: Display> Display for CaptureError<E> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Command(error) => Display::fmt(error, formatter),
            Self::Record(error) => write!(formatter, "graph recording failed: {error}"),
            Self::Graph(error) => Display::fmt(error, formatter),
        }
    }
}

impl<E: fmt::Debug + Display> Error for CaptureError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Command(error) => Some(error),
            Self::Record(_) => None,
            Self::Graph(error) => Some(error),
        }
    }
}

/// A captured CUDA graph with fixed device addresses and retained resources.
#[must_use = "instantiate or drop the captured graph"]
pub struct CapturedGraph {
    graph: Option<RawGraph>,
    stream: Arc<CudaStream>,
    bindings: Option<CheckedBindings>,
    resources: Vec<RetainedResource>,
    commands: usize,
    not_send_sync: PhantomData<Rc<()>>,
}

impl CapturedGraph {
    /// Returns the number of commands captured into the graph.
    pub const fn commands(&self) -> usize {
        self.commands
    }

    /// Instantiates one executable graph while preserving fixed bindings.
    pub fn instantiate(mut self) -> Result<GraphExec, GraphError> {
        let graph = self.graph.take().expect("captured graph owns a raw graph");
        let completion_event = self
            .stream
            .context()
            .new_event(None)
            .map_err(|error| GraphError::driver("completion event creation", error))?;
        let exec = RawGraphExec::instantiate(&graph)?;
        Ok(GraphExec {
            exec: Some(exec),
            graph: Some(graph),
            stream: self.stream.clone(),
            completion_event,
            bindings: self.bindings.take(),
            resources: std::mem::take(&mut self.resources),
            commands: self.commands,
            launches: 0,
            in_flight: false,
            record_error: None,
            poll_error: None,
            poisoned: false,
            unobserved_rejection: None,
            not_send_sync: PhantomData,
        })
    }
}

impl Drop for CapturedGraph {
    fn drop(&mut self) {
        drop(self.graph.take());
        self.resources.clear();
        self.bindings.take();
    }
}

/// One executable fixed-address graph.
///
/// Mutable launch access prevents concurrent replay. This first version is
/// intentionally neither `Send` nor `Sync`.
pub struct GraphExec {
    exec: Option<RawGraphExec>,
    graph: Option<RawGraph>,
    stream: Arc<CudaStream>,
    completion_event: CudaEvent,
    bindings: Option<CheckedBindings>,
    resources: Vec<RetainedResource>,
    commands: usize,
    launches: u64,
    in_flight: bool,
    record_error: Option<DriverError>,
    poll_error: Option<DriverError>,
    poisoned: bool,
    unobserved_rejection: Option<ContractError>,
    not_send_sync: PhantomData<Rc<()>>,
}

impl GraphExec {
    /// Number of commands in every replay.
    pub const fn commands(&self) -> usize {
        self.commands
    }

    /// Number of graph launches accepted by the CUDA driver.
    pub const fn launches(&self) -> u64 {
        self.launches
    }

    /// Launches the graph on the exact stream used during capture.
    pub fn launch(&mut self) -> Result<GraphLaunchCompletion<'_>, GraphError> {
        if self.poisoned {
            return Err(GraphError::ExecutionPoisoned);
        }
        if let Some(error) = self.unobserved_rejection.take() {
            return Err(GraphError::DeviceRejected(error));
        }
        if self.in_flight {
            return Err(GraphError::ReplayInFlight);
        }
        let launch_index = self
            .launches
            .checked_add(1)
            .ok_or(GraphError::LaunchCountExhausted)?;
        let exec = self.exec.as_ref().expect("live graph exec owns a raw exec");
        self.stream
            .context()
            .bind_to_thread()
            .map_err(|error| GraphError::driver("launch context binding", error))?;
        // Treat every raw launch attempt as possibly submitted. CUDA may
        // report an earlier asynchronous error from this call, so resources
        // cannot be released until stream or context quiescence is proven.
        self.in_flight = true;
        self.record_error = None;
        self.poll_error = None;
        let launch_result = unsafe {
            // SAFETY: `exec` is live, belongs to the stream context, and fixed
            // bindings plus retained resources remain owned by `self`.
            sys::cuGraphLaunch(exec.handle, self.stream.cu_stream()).result()
        };
        if let Err(error) = launch_result {
            let failure = GraphSettlementFailure {
                reported: error,
                synchronize_error: synchronize_stream_or_abort(&self.stream),
            };
            self.in_flight = false;
            self.poisoned = true;
            record_graph_settlement_errors(self, failure);
            return Err(failure.into_error("launch"));
        }
        self.launches = launch_index;
        self.record_error = self.completion_event.record(&self.stream).err();
        Ok(GraphLaunchCompletion {
            exec: Some(self),
            launch_index,
            settled: false,
        })
    }

    /// Measures one replay on the private capture stream with caller-owned
    /// timing events.
    ///
    /// Capture, instantiation, and event creation remain outside the measured
    /// interval. The interval includes the graph nodes and the completion
    /// event that every safe [`GraphExec::launch`] records after them.
    pub fn measure_launch_ms(
        &mut self,
        start: &CudaEvent,
        end: &CudaEvent,
    ) -> Result<f32, GraphError> {
        if !Arc::ptr_eq(start.context(), self.stream.context())
            || !Arc::ptr_eq(end.context(), self.stream.context())
        {
            return Err(GraphError::TimingEventContextMismatch);
        }
        start
            .record(&self.stream)
            .map_err(|error| GraphError::driver("timing start event record", error))?;
        let completion = self.launch()?;
        if let Err(error) = end.record(&completion.exec.as_ref().expect("live completion").stream) {
            drop(completion);
            return Err(GraphError::driver("timing end event record", error));
        }
        completion.wait()?;
        start
            .elapsed_ms(end)
            .map_err(|error| GraphError::driver("timing event elapsed query", error))
    }

    /// Destroys graph handles and returns the original checked bindings.
    pub fn into_bindings(mut self) -> Result<CheckedBindings, GraphBindingsError> {
        if let Some(failure) = self.settle_in_flight() {
            self.poisoned = true;
            record_graph_settlement_errors(&self, failure);
            return Err(GraphBindingsError::Execution(
                failure.into_error("completion"),
            ));
        }
        let status = self.decode_device_status();
        drop(self.exec.take());
        drop(self.graph.take());
        self.resources.clear();
        let mut bindings = self
            .bindings
            .take()
            .expect("live graph exec owns checked bindings");
        bindings.status.clear();
        match status {
            Ok(None) => Ok(bindings),
            Ok(Some(error)) => Err(GraphBindingsError::DeviceRejected(DeviceRejection::new(
                error, bindings,
            ))),
            Err(error) => Err(GraphBindingsError::StatusProtocol(error)),
        }
    }

    fn decode_device_status(&self) -> Result<Option<ContractError>, DeviceStatusProtocolError> {
        self.bindings
            .as_ref()
            .expect("live graph exec owns checked bindings")
            .status
            .decode()
    }

    fn settle_in_flight(&mut self) -> Option<GraphSettlementFailure> {
        if !self.in_flight {
            return None;
        }
        let failure = if let Some(error) = self.record_error {
            Some(GraphSettlementFailure {
                reported: error,
                synchronize_error: synchronize_stream_or_abort(&self.stream),
            })
        } else if let Some(error) = self.poll_error {
            Some(GraphSettlementFailure {
                reported: error,
                synchronize_error: synchronize_stream_or_abort(&self.stream),
            })
        } else {
            match self.completion_event.synchronize() {
                Ok(()) => None,
                Err(error) => Some(GraphSettlementFailure {
                    reported: error,
                    synchronize_error: synchronize_stream_or_abort(&self.stream),
                }),
            }
        };
        self.in_flight = false;
        self.record_error = None;
        self.poll_error = None;
        failure
    }
}

impl Drop for GraphExec {
    fn drop(&mut self) {
        if let Some(failure) = self.settle_in_flight() {
            self.poisoned = true;
            record_graph_settlement_errors(self, failure);
        }
        drop(self.exec.take());
        drop(self.graph.take());
        self.resources.clear();
        self.bindings.take();
    }
}

/// Completion fence for one graph replay.
#[must_use = "wait for or drop the graph completion before replaying again"]
pub struct GraphLaunchCompletion<'exec> {
    exec: Option<&'exec mut GraphExec>,
    launch_index: u64,
    settled: bool,
}

impl GraphLaunchCompletion<'_> {
    /// Returns the one-based replay index covered by this completion.
    pub const fn launch_index(&self) -> u64 {
        self.launch_index
    }

    /// Queries completion without blocking.
    pub fn is_complete(&mut self) -> Result<bool, GraphError> {
        let exec = self
            .exec
            .as_mut()
            .expect("live completion has a graph exec");
        if let Some(error) = exec.record_error {
            return Err(GraphError::driver("completion event record", error));
        }
        if let Some(error) = exec.poll_error {
            return Err(GraphError::driver("completion event query", error));
        }
        match exec.completion_event.query() {
            Ok(complete) => Ok(complete),
            Err(error) => {
                exec.poll_error = Some(error);
                Err(GraphError::driver("completion event query", error))
            }
        }
    }

    /// Waits for this replay to finish.
    pub fn wait(mut self) -> Result<(), GraphError> {
        match self.settle() {
            None => {
                self.settled = true;
                let exec = self
                    .exec
                    .as_mut()
                    .expect("live completion has a graph exec");
                match exec.decode_device_status() {
                    Ok(None) => Ok(()),
                    Ok(Some(error)) => Err(GraphError::DeviceRejected(error)),
                    Err(error) => {
                        exec.poisoned = true;
                        Err(GraphError::StatusProtocol(error))
                    }
                }
            }
            Some(failure) => {
                self.settled = true;
                let exec = self
                    .exec
                    .as_mut()
                    .expect("live completion has a graph exec");
                exec.poisoned = true;
                record_graph_settlement_errors(exec, failure);
                Err(failure.into_error("completion"))
            }
        }
    }

    fn settle(&mut self) -> Option<GraphSettlementFailure> {
        let exec = self
            .exec
            .as_mut()
            .expect("live completion has a graph exec");
        exec.settle_in_flight()
    }
}

impl Drop for GraphLaunchCompletion<'_> {
    fn drop(&mut self) {
        if self.settled {
            return;
        }
        if let Some(failure) = self.settle() {
            let exec = self
                .exec
                .as_mut()
                .expect("live completion has a graph exec");
            exec.poisoned = true;
            record_graph_settlement_errors(exec, failure);
        } else {
            let exec = self
                .exec
                .as_mut()
                .expect("live completion has a graph exec");
            match exec.decode_device_status() {
                Ok(Some(error)) => exec.unobserved_rejection = Some(error),
                Ok(None) => {}
                Err(_) => exec.poisoned = true,
            }
        }
        self.settled = true;
    }
}

#[derive(Clone, Copy)]
struct GraphSettlementFailure {
    reported: DriverError,
    synchronize_error: Option<DriverError>,
}

impl GraphSettlementFailure {
    const fn into_error(self, operation: &'static str) -> GraphError {
        match self.synchronize_error {
            Some(synchronization) => GraphError::Settlement {
                operation,
                source: self.reported,
                synchronization,
            },
            None => GraphError::Driver {
                operation,
                source: self.reported,
            },
        }
    }
}

fn record_graph_settlement_errors(exec: &GraphExec, failure: GraphSettlementFailure) {
    if let Some(error) = failure.synchronize_error {
        exec.stream.context().record_err::<()>(Err(error));
    }
    // cuda-oxide keeps the latest sticky error. Record the primary failure
    // last so fallback synchronization diagnostics do not overwrite it.
    exec.stream
        .context()
        .record_err::<()>(Err(failure.reported));
}

struct CaptureGuard {
    stream: Arc<CudaStream>,
    active: bool,
    not_send_sync: PhantomData<Rc<()>>,
}

impl CaptureGuard {
    fn new(stream: Arc<CudaStream>) -> Self {
        Self {
            stream,
            active: true,
            not_send_sync: PhantomData,
        }
    }

    fn finish(mut self) -> Result<RawGraph, GraphError> {
        let result = end_capture(&self.stream);
        self.active = false;
        result
    }
}

impl Drop for CaptureGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        match end_capture(&self.stream) {
            Ok(graph) => drop(graph),
            Err(GraphError::Driver { source, .. }) => {
                self.stream.context().record_err::<()>(Err(source));
            }
            Err(error) => {
                eprintln!("oxide-infer-cuda failed to discard CUDA Graph capture: {error}");
            }
        }
        self.active = false;
    }
}

struct RawGraph {
    handle: sys::CUgraph,
    context: Arc<CudaContext>,
}

impl Drop for RawGraph {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        match bind_context_for_cleanup(&self.context) {
            Ok(()) => {
                // SAFETY: this wrapper uniquely owns the graph handle and no
                // graph exec can outlive it in the explicit owner hierarchy.
                if let Err(error) = unsafe { sys::cuGraphDestroy(self.handle).result() } {
                    abort_after_graph_cleanup_failure("graph", error);
                }
            }
            Err(error) => abort_after_graph_cleanup_failure("graph context binding", error),
        }
        self.handle = std::ptr::null_mut();
    }
}

struct RawGraphExec {
    handle: sys::CUgraphExec,
    context: Arc<CudaContext>,
}

impl RawGraphExec {
    fn instantiate(graph: &RawGraph) -> Result<Self, GraphError> {
        graph
            .context
            .bind_to_thread()
            .map_err(|error| GraphError::driver("instantiate context binding", error))?;
        let mut handle = MaybeUninit::uninit();
        // SAFETY: `graph` is live, flags are zero, and CUDA initializes
        // `handle` only on success.
        unsafe {
            sys::cuGraphInstantiateWithFlags(handle.as_mut_ptr(), graph.handle, 0)
                .result()
                .map_err(|error| GraphError::driver("instantiate", error))?;
            let handle = handle.assume_init();
            if handle.is_null() {
                return Err(GraphError::NullExecutableHandle);
            }
            Ok(Self {
                handle,
                context: graph.context.clone(),
            })
        }
    }
}

impl Drop for RawGraphExec {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        match bind_context_for_cleanup(&self.context) {
            Ok(()) => {
                // SAFETY: this wrapper uniquely owns the executable handle.
                if let Err(error) = unsafe { sys::cuGraphExecDestroy(self.handle).result() } {
                    abort_after_graph_cleanup_failure("executable graph", error);
                }
            }
            Err(error) => {
                abort_after_graph_cleanup_failure("executable graph context binding", error)
            }
        }
        self.handle = std::ptr::null_mut();
    }
}

fn begin_capture(stream: &CudaStream) -> Result<(), GraphError> {
    stream
        .context()
        .bind_to_thread()
        .map_err(|error| GraphError::driver("capture context binding", error))?;
    // SAFETY: the non-default stream is live and owned by the current context.
    unsafe {
        sys::cuStreamBeginCapture_v2(
            stream.cu_stream(),
            sys::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
        )
        .result()
        .map_err(|error| GraphError::driver("capture begin", error))
    }
}

fn end_capture(stream: &CudaStream) -> Result<RawGraph, GraphError> {
    if let Err(error) = bind_context_for_cleanup(stream.context()) {
        eprintln!(
            "oxide-infer-cuda cannot bind the CUDA context to end an active graph capture: \
             {error}; aborting to preserve buffer and stream safety"
        );
        std::process::abort();
    }
    let mut handle = MaybeUninit::uninit();
    // SAFETY: this function is called exactly once for a successfully started
    // capture on the same thread and stream.
    let result =
        unsafe { sys::cuStreamEndCapture(stream.cu_stream(), handle.as_mut_ptr()).result() };
    match result {
        Ok(()) => {
            // SAFETY: CUDA initialized the output handle on success.
            let handle = unsafe { handle.assume_init() };
            if handle.is_null() {
                Err(GraphError::CaptureInvalidated)
            } else {
                Ok(RawGraph {
                    handle,
                    context: stream.context().clone(),
                })
            }
        }
        Err(error) => {
            ensure_capture_ended_or_abort(stream, error);
            Err(GraphError::driver("capture end", error))
        }
    }
}

fn ensure_capture_ended_or_abort(stream: &CudaStream, end_error: DriverError) {
    let mut status = MaybeUninit::uninit();
    // SAFETY: `status` is valid output storage and `stream` remains live.
    let query =
        unsafe { sys::cuStreamIsCapturing(stream.cu_stream(), status.as_mut_ptr()).result() };
    if query.is_ok() {
        // SAFETY: CUDA initialized `status` on success.
        let status = unsafe { status.assume_init() };
        if status == sys::CUstreamCaptureStatus_enum_CU_STREAM_CAPTURE_STATUS_NONE {
            return;
        }
    }
    eprintln!(
        "oxide-infer-cuda cannot prove CUDA Graph capture ended after {end_error}; aborting to preserve buffer and stream safety"
    );
    std::process::abort()
}

fn abort_after_graph_cleanup_failure(resource: &str, error: DriverError) -> ! {
    eprintln!(
        "oxide-infer-cuda failed to destroy a CUDA {resource}: {error}; aborting before releasing \
         fixed bindings or retained resources"
    );
    std::process::abort()
}

#[derive(Debug, Error)]
pub enum GraphError {
    #[error("CUDA Graph capture requires a non-default stream")]
    DefaultStreamUnsupported,
    #[error("CUDA Graph capture recorded no commands")]
    EmptyCapture,
    #[error("CUDA Graph capture was invalidated and returned no graph")]
    CaptureInvalidated,
    #[error("CUDA Graph instantiation returned a null executable handle")]
    NullExecutableHandle,
    #[error("CUDA Graph execution is poisoned by an earlier replay failure")]
    ExecutionPoisoned,
    #[error("a CUDA Graph replay is already in flight")]
    ReplayInFlight,
    #[error("device rejected CUDA Graph metadata: {0}")]
    DeviceRejected(ContractError),
    #[error(transparent)]
    StatusProtocol(DeviceStatusProtocolError),
    #[error("CUDA Graph launch count is exhausted")]
    LaunchCountExhausted,
    #[error("CUDA Graph timing events belong to a different CUDA context")]
    TimingEventContextMismatch,
    #[error("CUDA Graph {operation} failed: {source}")]
    Driver {
        operation: &'static str,
        #[source]
        source: DriverError,
    },
    #[error(
        "CUDA Graph {operation} failed: {source}; fallback stream synchronization also reported: \
         {synchronization}"
    )]
    Settlement {
        operation: &'static str,
        #[source]
        source: DriverError,
        synchronization: DriverError,
    },
}

#[derive(Debug, Error)]
pub enum GraphBindingsError {
    #[error(transparent)]
    Execution(GraphError),
    #[error(transparent)]
    DeviceRejected(DeviceRejection),
    #[error(transparent)]
    StatusProtocol(DeviceStatusProtocolError),
}

impl GraphError {
    const fn driver(operation: &'static str, source: DriverError) -> Self {
        Self::Driver { operation, source }
    }
}
