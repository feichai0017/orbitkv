use crate::transfer::finish_gpu_transfer;
use std::sync::{Arc, mpsc as std_mpsc};
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaStream};
use log::{debug, error, info, warn};
use logforth::diagnostic::ThreadLocalDiagnostic;
use parking_lot::Mutex;
use tokio::sync::{OnceCell, OwnedSemaphorePermit, Semaphore, mpsc, oneshot};

use crate::EngineError;
use crate::block::{RawBlock, SealedBlock};
use crate::memory::numa::{NumaNode, pin_thread_to_numa_node};
use crate::metrics::core_metrics;
use crate::transfer::layout::{BlockCopies, KVCacheLayout};
use crate::transfer::{CopyDesc, KernelBackend, MemcpyBackend, TransferBackend, TransferMode};

pub(crate) mod ssd;
use ssd::GpuWrite;

/// A task to restore KV blocks from leased sources to GPU layers
pub(crate) struct LoadTask {
    pub layers: Vec<LayerTransferData>,
    pub completion: oneshot::Sender<LoadOutcome>,
    pub reservations: Vec<crate::QueryReservation>,
}

/// Terminal GPU transfer evidence, timestamped before notifying the dispatcher.
#[derive(Debug)]
pub struct LoadOutcome {
    pub result: Result<(), EngineError>,
    pub completed_at: Instant,
}

/// One layer in a transfer task (either direction): GPU layout plus the
/// blocks to move.
pub(crate) struct LayerTransferData {
    pub layer_name: String,
    pub layout: KVCacheLayout,
    pub blocks: Vec<TransferBlock>,
}

/// Owned payload or leased source for a GPU transfer.
pub(crate) enum TransferPayload {
    /// Save path: uniquely owned, moved through the GPU worker and returned.
    Owned(RawBlock),
    Ssd {
        source: Arc<crate::SsdReadLease>,
        slot_id: usize,
        offset: usize,
    },
    /// Load path: slot borrowed from a cached sealed block.
    Cached {
        sealed: Arc<SealedBlock>,
        slot_id: usize,
        /// Byte offset into the slot's `RawBlock` where this layer begins.
        /// Page-first reads every layer from one page slot at distinct
        /// offsets; layer-first is always 0 (the slot is the layer).
        offset: usize,
    },
}

impl TransferPayload {
    pub(crate) fn raw(&self) -> &RawBlock {
        match self {
            Self::Ssd { .. } => unreachable!("SSD sources do not own host memory"),
            Self::Owned(block) => block,
            Self::Cached {
                sealed, slot_id, ..
            } => sealed
                .get_slot(*slot_id)
                .expect("cached slot_id validated at construction"),
        }
    }

    /// Byte offset of this layer within its host slot (0 unless page-first).
    fn host_offset(&self) -> usize {
        match self {
            Self::Ssd { offset, .. } => *offset,
            Self::Owned(_) => 0,
            Self::Cached { offset, .. } => *offset,
        }
    }
}

/// One block in a transfer: the GPU slot index and its host-side image.
/// Loads copy `block` -> GPU slot, saves copy GPU slot -> `block`.
pub(crate) struct TransferBlock {
    pub block_idx: usize,
    pub block: TransferPayload,
}

/// A task to save KV blocks from GPU to CPU for multiple layers.
/// Caller pre-allocates the host blocks, worker does the GPU->CPU copy.
/// All layers are copied on the same CUDA stream with a single synchronization.
pub(crate) struct SaveTask {
    pub layers: Vec<LayerTransferData>,
    pub reply: oneshot::Sender<Result<Vec<LayerTransferData>, EngineError>>,
    pub ssd_writes: Vec<GpuWrite>,
    pub ssd_admission: Option<OwnedSemaphorePermit>,
    #[cfg(feature = "tracing")]
    pub trace_ctx: Option<::fastrace::prelude::SpanContext>,
}

enum WorkerCommand {
    Load(LoadTask),
    Save(SaveTask),
    Drain(oneshot::Sender<Result<(), String>>),
}

/// Independent memory read/write lanes and one lazily allocated GPU storage lane.
pub(crate) struct GpuWorkerPool {
    device_id: i32,
    numa_node: NumaNode,
    transfer_mode: TransferMode,
    ssd_tx: Mutex<Option<mpsc::UnboundedSender<WorkerCommand>>>,
    ssd_write_admission: Arc<Semaphore>,
    load_tx: mpsc::UnboundedSender<WorkerCommand>,
    save_tx: mpsc::UnboundedSender<WorkerCommand>,
    closed: Mutex<bool>,
    drained: OnceCell<Result<(), String>>,
}

impl GpuWorkerPool {
    pub(crate) fn new(
        device_id: i32,
        numa_node: NumaNode,
        transfer_mode: TransferMode,
    ) -> Result<Self, EngineError> {
        Ok(Self {
            device_id,
            numa_node,
            transfer_mode,
            load_tx: spawn_worker(device_id, numa_node, transfer_mode, "load")?,
            save_tx: spawn_worker(device_id, numa_node, transfer_mode, "save")?,
            ssd_tx: Mutex::new(None),
            ssd_write_admission: Arc::new(Semaphore::new(ssd::MAX_WRITES)),
            closed: Mutex::new(false),
            drained: OnceCell::new(),
        })
    }

    fn submit(&self, command: WorkerCommand, disk: bool) -> Result<(), EngineError> {
        let closed = self.closed.lock();
        if *closed {
            return Err(EngineError::Storage("GPU worker is draining".into()));
        }
        if disk {
            let mut sender = self.ssd_tx.lock();
            if sender.is_none() {
                *sender = Some(spawn_worker(
                    self.device_id,
                    self.numa_node,
                    self.transfer_mode,
                    "ssd",
                )?);
            }
            sender
                .as_ref()
                .expect("SSD worker initialized")
                .send(command)
        } else {
            match command {
                WorkerCommand::Load(_) => self.load_tx.send(command),
                _ => self.save_tx.send(command),
            }
        }
        .map_err(|_| {
            EngineError::Storage(format!(
                "GPU worker channel closed for device {}",
                self.device_id
            ))
        })
    }

    pub(crate) fn submit_load(&self, task: LoadTask) -> Result<(), EngineError> {
        let disk = task.layers.iter().any(|layer| {
            layer
                .blocks
                .iter()
                .any(|block| matches!(block.block, TransferPayload::Ssd { .. }))
        });
        self.submit(WorkerCommand::Load(task), disk)
    }

    pub(crate) async fn batch_save(
        &self,
        layers: Vec<LayerTransferData>,
        mut ssd_writes: Vec<GpuWrite>,
    ) -> Result<Vec<LayerTransferData>, EngineError> {
        let (reply, receiver) = oneshot::channel();
        let ssd_admission = if ssd_writes.is_empty() {
            None
        } else {
            match Arc::clone(&self.ssd_write_admission).try_acquire_owned() {
                Ok(permit) => Some(permit),
                Err(_) => {
                    // Roll back unsubmitted GPU extents, preserving the normal
                    // D2H publication and bounded io_uring writeback path.
                    ssd_writes.clear();
                    core_metrics().ssd_gpu_write_fallbacks.add(1, &[]);
                    None
                }
            }
        };
        let disk = ssd_admission.is_some();
        self.submit(
            WorkerCommand::Save(SaveTask {
                layers,
                reply,
                ssd_writes,
                ssd_admission,
                #[cfg(feature = "tracing")]
                trace_ctx: ::fastrace::prelude::SpanContext::current_local_parent(),
            }),
            disk,
        )?;
        receiver
            .await
            .map_err(|_| EngineError::Storage("GPU save worker disappeared".into()))?
    }

    /// Reject new work; every lane releases staging and imported pointers before acknowledgement.
    pub(crate) async fn drain(&self) -> Result<(), EngineError> {
        self.drained
            .get_or_init(|| async {
                let mut acknowledgements = Vec::new();
                {
                    let mut closed = self.closed.lock();
                    *closed = true;
                    let ssd = self.ssd_tx.lock();
                    for sender in [Some(&self.load_tx), Some(&self.save_tx), ssd.as_ref()]
                        .into_iter()
                        .flatten()
                    {
                        let (reply, receiver) = oneshot::channel();
                        sender
                            .send(WorkerCommand::Drain(reply))
                            .map_err(|_| "GPU worker unavailable while draining".to_owned())?;
                        acknowledgements.push(receiver);
                    }
                }
                for receiver in acknowledgements {
                    receiver
                        .await
                        .map_err(|_| "GPU worker exited before draining".to_owned())??;
                }
                Ok(())
            })
            .await
            .clone()
            .map_err(EngineError::Storage)
    }
}

fn spawn_worker(
    device_id: i32,
    numa: NumaNode,
    mode: TransferMode,
    name: &str,
) -> Result<mpsc::UnboundedSender<WorkerCommand>, EngineError> {
    let (sender, receiver) = mpsc::unbounded_channel();
    let (ready_tx, ready_rx) = std_mpsc::channel();
    let storage = name == "ssd";
    std::thread::Builder::new()
        .name(format!("gpu{device_id}-{name}"))
        .spawn(move || {
            if numa.is_valid()
                && let Err(error) = pin_thread_to_numa_node(numa)
            {
                warn!("Failed to pin GPU worker: {error}");
            }
            match init_worker(device_id, mode) {
                Ok(runtime) => {
                    let _ = ready_tx.send(Ok(()));
                    if storage {
                        ssd::run(receiver, runtime);
                    } else {
                        worker_loop(device_id, receiver, runtime);
                    }
                }
                Err(error) => {
                    let _ = ready_tx.send(Err(error));
                }
            }
        })
        .map_err(|e| EngineError::CudaInit(e.to_string()))?;
    ready_rx
        .recv()
        .map_err(|e| EngineError::CudaInit(e.to_string()))??;
    Ok(sender)
}

struct WorkerRuntime {
    stream: Arc<CudaStream>,
    backend: Box<dyn TransferBackend>,
}

fn build_backend(
    mode: TransferMode,
    ctx: &std::sync::Arc<CudaContext>,
) -> Result<Box<dyn TransferBackend>, EngineError> {
    match mode {
        TransferMode::Direct => Ok(Box::new(MemcpyBackend)),
        TransferMode::Kernel => {
            let kernel = KernelBackend::new(ctx)
                .map_err(|e| EngineError::CudaInit(format!("kernel backend init failed: {e}")))?;
            Ok(Box::new(kernel))
        }
    }
}

fn init_worker(device_id: i32, transfer_mode: TransferMode) -> Result<WorkerRuntime, EngineError> {
    // Initialize CUDA context for this thread
    let ctx = CudaContext::new(device_id as usize)
        .map_err(|e| EngineError::CudaInit(format!("Failed to create CUDA context: {e:?}")))?;
    let stream = ctx
        .new_stream()
        .map_err(|e| EngineError::CudaInit(format!("Failed to create CUDA stream: {e:?}")))?;

    // Set thread-local diagnostic info
    ThreadLocalDiagnostic::insert("device_id", device_id.to_string());

    let backend = build_backend(transfer_mode, &ctx)?;

    info!(
        "GPU worker initialized: device={} backend={}",
        device_id,
        backend.name()
    );

    Ok(WorkerRuntime { stream, backend })
}

fn worker_loop(
    device_id: i32,
    mut receiver: mpsc::UnboundedReceiver<WorkerCommand>,
    runtime: WorkerRuntime,
) {
    while let Some(command) = receiver.blocking_recv() {
        match command {
            WorkerCommand::Drain(reply) => {
                let result = runtime
                    .stream
                    .synchronize()
                    .map_err(|e| format!("GPU drain failed: {e}"));
                let _ = reply.send(result);
                break;
            }
            WorkerCommand::Load(task) => {
                let started = Instant::now();
                let result = (|| {
                    let (copies, bytes) = build_copy_descs(&task.layers)?;
                    finish_gpu_transfer(
                        &runtime.stream,
                        runtime.backend.h2d(&copies, &runtime.stream),
                    )?;
                    Ok(bytes)
                })();
                let bytes = result.as_ref().copied().unwrap_or(0);
                finish_load(task, result.map(|_| ()), started, bytes);
            }
            WorkerCommand::Save(SaveTask {
                layers,
                reply,
                ssd_writes: _,
                ssd_admission: _,
                #[cfg(feature = "tracing")]
                trace_ctx,
            }) => {
                let result = process_save_task(
                    &layers,
                    &runtime.stream,
                    runtime.backend.as_ref(),
                    #[cfg(feature = "tracing")]
                    trace_ctx,
                )
                .map(|()| layers);
                let _ = reply.send(result);
            }
        }
    }
    info!("GPU worker shutting down: device={device_id}");
}

/// Build one `CopyDesc` per GPU segment of every block across all layers,
/// pairing device ranges from the layout with the host segments of each
/// block's `RawBlock`. Direction-agnostic: load and save submit the same
/// descriptors to `h2d`/`d2h` respectively.
///
/// Returns `(copies, total_bytes)`.
fn build_copy_descs(layers: &[LayerTransferData]) -> Result<(Vec<CopyDesc>, usize), EngineError> {
    let mut copies: Vec<CopyDesc> = Vec::new();
    let mut total_bytes = 0usize;

    for (layer_index, layer) in layers.iter().enumerate() {
        let layer_name = &layer.layer_name;

        for block in &layer.blocks {
            if matches!(block.block, TransferPayload::Ssd { .. }) {
                continue;
            }
            let block_copies = layer
                .layout
                .block_copies(block.block_idx)
                .map_err(|e| EngineError::Storage(format!("layer {layer_name}: {e}")))?;

            // Page-first reads/writes every layer from one slot at its byte
            // offset; layer-first leaves this 0 (slot == layer).
            let host_offset = block.block.host_offset();

            match block_copies {
                BlockCopies::Split { k, v } => {
                    let raw = block.block.raw();
                    let k_ptr = raw.segment_mapped_ptr(0).unwrap().add(host_offset);
                    // SAFETY: For a contiguous host block (segment 1 absent), the
                    // allocation is 2 * segment size, so k + k.bytes is in bounds.
                    let v_ptr = raw
                        .segment_mapped_ptr(1)
                        .map(|p| p.add(host_offset))
                        .unwrap_or_else(|| k_ptr.add(k.bytes));

                    copies.push(CopyDesc {
                        device: k.addr,
                        host: k_ptr.host().as_ptr(),
                        host_device: k_ptr.device().as_ptr() as u64,
                        size: k.bytes,
                        device_allocation: layer_index,
                        host_allocation: raw.segment_allocation_id(0).unwrap(),
                    });
                    copies.push(CopyDesc {
                        device: v.addr,
                        host: v_ptr.host().as_ptr(),
                        host_device: v_ptr.device().as_ptr() as u64,
                        size: v.bytes,
                        device_allocation: layer_index,
                        host_allocation: raw
                            .segment_allocation_id(1)
                            .unwrap_or_else(|| raw.segment_allocation_id(0).unwrap()),
                    });
                    total_bytes += k.bytes + v.bytes;
                }
                BlockCopies::Contiguous(c) => {
                    let ptr = block
                        .block
                        .raw()
                        .segment_mapped_ptr(0)
                        .unwrap()
                        .add(host_offset);
                    copies.push(CopyDesc {
                        device: c.addr,
                        host: ptr.host().as_ptr(),
                        host_device: ptr.device().as_ptr() as u64,
                        size: c.bytes,
                        device_allocation: layer_index,
                        host_allocation: block.block.raw().segment_allocation_id(0).unwrap(),
                    });
                    total_bytes += c.bytes;
                }
            }
        }
    }

    Ok((copies, total_bytes))
}

/// Publish completion only after the worker establishes that all GPU access has ended.
fn finish_load(task: LoadTask, result: Result<(), EngineError>, started: Instant, bytes: usize) {
    if result.is_ok() {
        for layer in &task.layers {
            for block in &layer.blocks {
                if let TransferPayload::Cached { sealed, .. } = &block.block {
                    sealed.mark_warmup_restored();
                }
            }
        }
        if bytes != 0 {
            core_metrics().load_bytes.add(bytes as u64, &[]);
            core_metrics()
                .load_duration_seconds
                .record(started.elapsed().as_secs_f64(), &[]);
        }
    } else if let Err(error) = &result {
        error!("GPU restore failed: {error}");
        core_metrics().load_failures.add(1, &[]);
    }
    drop(task.layers);
    drop(task.reservations);
    let _ = task.completion.send(LoadOutcome {
        result,
        completed_at: Instant::now(),
    });
}

/// Process a save task: copy blocks from GPU to CPU pinned memory. All layers
/// and segments are collected into one descriptor batch, handed to a single
/// backend, then synchronized once.
fn process_save_task(
    layers: &[LayerTransferData],
    stream: &Arc<CudaStream>,
    backend: &dyn TransferBackend,
    #[cfg(feature = "tracing")] trace_ctx: Option<::fastrace::prelude::SpanContext>,
) -> Result<(), EngineError> {
    trace_child!("gpu.save_task", trace_ctx);
    let start = std::time::Instant::now();
    let total_blocks: usize = layers.iter().map(|l| l.blocks.len()).sum();

    let (copies, total_bytes) = build_copy_descs(layers)?;

    let submitted = backend.d2h(&copies, stream);
    finish_gpu_transfer(stream, submitted)?;

    let elapsed = start.elapsed();
    let bandwidth_gbps = if elapsed.as_secs_f64() > 0.0 {
        (total_bytes as f64 / 1e9) / elapsed.as_secs_f64()
    } else {
        0.0
    };

    debug!(
        "Save task completed: layers={} blocks={} copies={} bytes={} elapsed_ms={:.2} bandwidth_gbps={:.2} backend={}",
        layers.len(),
        total_blocks,
        copies.len(),
        total_bytes,
        elapsed.as_secs_f64() * 1000.0,
        bandwidth_gbps,
        backend.name()
    );

    Ok(())
}

#[cfg(test)]
#[path = "../../../tests/unit/transfer/worker/mod.rs"]
mod drain_tests;
