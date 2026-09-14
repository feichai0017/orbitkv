use std::{
    fs::File,
    path::{Path, PathBuf},
};

use tracing_appender::non_blocking::{NonBlocking, WorkerGuard};
use tracing_perfetto::PerfettoLayer as PerfettoLayerInner;

/// Creates a Perfetto tracing layer with default configuration.
///
/// Returns a tuple of `(layer, guard)`. The layer implements
/// [`Layer<S>`](tracing_subscriber::Layer) and can be composed with other layers
/// using [`tracing_subscriber::registry()`].
///
/// The guard must be kept alive and [`stop()`](PerfettoGuard::stop) should be called
/// before dropping to ensure all events are flushed.
pub fn perfetto_layer(path: impl AsRef<Path>) -> (PerfettoLayer, PerfettoGuard) {
    let path = path.as_ref().to_path_buf();
    let file = File::create(&path).expect("Failed to create Perfetto trace file");
    let (writer, writer_guard) = tracing_appender::non_blocking(file);

    let layer = PerfettoLayerInner::new(writer).with_debug_annotations(true);

    let guard = PerfettoGuard {
        _writer_guard: Some(writer_guard),
        path,
    };

    (layer, guard)
}

/// The concrete Perfetto layer type.
pub type PerfettoLayer = PerfettoLayerInner<NonBlocking>;

/// Guard that manages the lifecycle of a Perfetto tracing session.
///
/// This guard must be kept alive for the duration of tracing. Call [`stop`](Self::stop)
/// before dropping to ensure all trace events are flushed to disk.
///
/// The [`path`](Self::path) field provides access to the trace file path, which is
/// useful for post-processing (e.g., merging GPU timing data).
pub struct PerfettoGuard {
    _writer_guard: Option<WorkerGuard>,
    /// Path to the Perfetto trace file.
    pub path: PathBuf,
}

impl PerfettoGuard {
    /// Stop the tracing session and flush all pending events.
    ///
    /// This should be called before dropping the guard to ensure all trace
    /// data is written to disk.
    pub fn stop(&mut self) {
        // Drop the writer guard, which flushes pending writes
        self._writer_guard.take();
    }
}
