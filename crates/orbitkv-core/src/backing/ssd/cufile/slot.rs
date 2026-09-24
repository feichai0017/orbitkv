use std::cell::UnsafeCell;
use std::ffi::c_void;
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaEvent, CudaStream, result, sys};

use super::{Cufile, CufileFile, IoBatch, STAGING_BYTES};
use crate::cost::{CostKey, CostPath, Observation, Outcome, Representation};
use crate::metrics::core_metrics;
use crate::transfer::finish_gpu_transfer;

struct Arguments {
    size: usize,
    file_offset: i64,
    buffer_offset: i64,
    transferred: isize,
}

struct Pending {
    file: Arc<CufileFile>,
    batch: IoBatch,
    write: bool,
    scattering: bool,
    started: Instant,
    observation: Observation,
}

/// A reusable stream, registered buffer and stable host arguments. The worker
/// separately retains the task's extent leases and engine pages until poll completes.
pub(crate) struct GpuSlot {
    driver: Arc<Cufile>,
    stream: Arc<CudaStream>,
    event: CudaEvent,
    pointer: u64,
    buffer_registered: bool,
    stream_registered: bool,
    arguments: Box<UnsafeCell<Arguments>>,
    pending: Option<Pending>,
}

impl GpuSlot {
    pub(crate) fn new(context: &Arc<CudaContext>) -> Result<Self, String> {
        let driver = Cufile::get(false)?;
        let stream = context.new_stream().map_err(|e| e.to_string())?;
        let event = context.new_event(None).map_err(|e| e.to_string())?;
        // SAFETY: the calling thread is bound by new_stream/new_event. This
        // allocation remains owned through stream drain and deregistration.
        let pointer = unsafe { result::malloc_sync(STAGING_BYTES) }.map_err(|e| e.to_string())?;
        core_metrics()
            .ssd_gpu_staging_bytes
            .add(STAGING_BYTES as i64, &[]);
        let mut slot = Self {
            driver,
            stream,
            event,
            pointer,
            buffer_registered: false,
            stream_registered: false,
            arguments: Box::new(UnsafeCell::new(Arguments {
                size: 0,
                file_offset: 0,
                buffer_offset: 0,
                transferred: isize::MIN,
            })),
            pending: None,
        };
        // SAFETY: both registrations are owned by slot, including partial init failure.
        unsafe { (slot.driver.register_buffer)(pointer as *const c_void, STAGING_BYTES, 0) }
            .check("cuFileBufRegister")?;
        slot.buffer_registered = true;
        // FIXED_BUF_OFFSET | FIXED_FILE_OFFSET | FIXED_FILE_SIZE. Arguments
        // still remain unchanged and address-stable until the completion event.
        unsafe { (slot.driver.register_stream)(slot.stream.cu_stream(), 0x7) }
            .check("cuFileStreamRegister")?;
        slot.stream_registered = true;
        Ok(slot)
    }

    pub(crate) fn submit(
        &mut self,
        file: Arc<CufileFile>,
        batch: IoBatch,
        write: bool,
    ) -> Result<(), String> {
        assert!(
            self.pending.is_none(),
            "GPU storage slot is still owned by a transfer"
        );
        batch.validate()?;
        // SAFETY: there is no previous in-flight access to this stable storage.
        unsafe {
            *self.arguments.get() = Arguments {
                size: batch.bytes,
                file_offset: batch.file_offset as i64,
                buffer_offset: 0,
                transferred: isize::MIN,
            }
        };
        let mut observation = Observation::new(
            CostKey::new(
                if write {
                    CostPath::SsdCufileWrite
                } else {
                    CostPath::SsdCufileRead
                },
                file.cost_resource,
                Representation::Unknown,
                batch.bytes as u64,
                batch.copies.len(),
            ),
            // GPU ranges may contain encoded payload, padding or duplicate
            // scatter consumers. Only the restore owner knows logical bytes.
            None,
        );
        observation.admitted();
        observation.submitted();
        self.pending = Some(Pending {
            file,
            batch,
            write,
            scattering: false,
            started: Instant::now(),
            observation,
        });
        core_metrics().ssd_cufile_inflight_batches.add(1, &[]);
        let submitted = (|| {
            let pending = self.pending.as_ref().expect("pending transfer inserted");
            if write {
                // SAFETY: the validated plan fits this slot and the worker holds
                // all source GPU pages until this stream completes.
                unsafe {
                    result::memset_d8_async(
                        self.pointer,
                        0,
                        pending.batch.bytes,
                        self.stream.cu_stream(),
                    )
                }
                .map_err(|e| e.to_string())?;
                for copy in &pending.batch.copies {
                    unsafe {
                        result::memcpy_dtod_async(
                            self.pointer + copy.file_offset - pending.batch.file_offset,
                            copy.device,
                            copy.bytes,
                            self.stream.cu_stream(),
                        )
                    }
                    .map_err(|e| e.to_string())?;
                }
            }
            let args = self.arguments.get();
            let operation = if write {
                self.driver.write
            } else {
                self.driver.read
            };
            // SAFETY: cuFile retains these pointers; the slot owns their storage,
            // file and buffer until its event completes or Drop drains the stream.
            unsafe {
                operation(
                    pending.file.handle.as_ptr(),
                    self.pointer as *mut c_void,
                    &raw mut (*args).size,
                    &raw mut (*args).file_offset,
                    &raw mut (*args).buffer_offset,
                    &raw mut (*args).transferred,
                    self.stream.cu_stream(),
                )
            }
            .check(if write {
                "cuFileWriteAsync"
            } else {
                "cuFileReadAsync"
            })?;
            self.event.record(&self.stream).map_err(|e| e.to_string())
        })();
        if let Err(error) = submitted {
            // A failed submission can still have queued GPU work. Never return
            // ownership of its arguments, file, staging or pages before drain.
            let _ = finish_gpu_transfer(&self.stream, Ok(()));
            self.finish(Err(error.clone()), Outcome::Failed);
            return Err(error);
        }
        Ok(())
    }

    pub(crate) fn poll(&mut self) -> Option<Result<(), String>> {
        let pending = self.pending.as_ref()?;
        // Unlike is_complete(), distinguish NOT_READY from a CUDA failure.
        match unsafe { result::event::query(self.event.cu_event()) } {
            Ok(()) => {}
            Err(error) if error.0 == sys::CUresult::CUDA_ERROR_NOT_READY => return None,
            Err(error) => {
                let _ = finish_gpu_transfer(&self.stream, Ok(()));
                let error = error.to_string();
                self.finish(Err(error.clone()), Outcome::Failed);
                return Some(Err(error));
            }
        }
        #[cfg(feature = "test-hooks")]
        if crate::test_faults::active(if pending.write {
            "cufile_write_completion"
        } else {
            "cufile_read_completion"
        }) {
            return None;
        }
        if !pending.scattering {
            // SAFETY: the event follows cuFile on this stream. It proves the
            // asynchronous writer has finished accessing the result storage.
            let bytes = unsafe { (*self.arguments.get()).transferred };
            #[cfg(feature = "test-hooks")]
            let bytes = if pending.write && crate::test_faults::active("cufile_write_error") {
                -1
            } else {
                bytes
            };
            if bytes != pending.batch.bytes as isize {
                let error = format!(
                    "cuFile{}Async completed {bytes}, expected {} at {}",
                    if pending.write { "Write" } else { "Read" },
                    pending.batch.bytes,
                    pending.batch.file_offset
                );
                self.finish(Err(error.clone()), Outcome::Failed);
                return Some(Err(error));
            }
            if !pending.write {
                let submitted = (|| {
                    for copy in &pending.batch.copies {
                        // SAFETY: source I/O succeeded and the worker still owns
                        // every destination page. A partial scatter also drains.
                        unsafe {
                            result::memcpy_dtod_async(
                                copy.device,
                                self.pointer + copy.file_offset - pending.batch.file_offset,
                                copy.bytes,
                                self.stream.cu_stream(),
                            )
                        }
                        .map_err(|e| e.to_string())?;
                    }
                    self.event.record(&self.stream).map_err(|e| e.to_string())
                })();
                if let Err(error) = submitted {
                    let _ = finish_gpu_transfer(&self.stream, Ok(()));
                    self.finish(Err(error.clone()), Outcome::Failed);
                    return Some(Err(error));
                }
                self.pending
                    .as_mut()
                    .expect("pending transfer exists")
                    .scattering = true;
                return None;
            }
        }
        self.finish(Ok(()), Outcome::Completed);
        Some(Ok(()))
    }

    fn finish(&mut self, result: Result<(), String>, outcome: Outcome) {
        let pending = self.pending.take().expect("pending transfer exists");
        // Every caller has observed the event or drained the existing stream.
        // This is host-observed gather/I/O/scatter time, not native-GDS proof.
        let transferred = unsafe { (*self.arguments.get()).transferred };
        pending
            .observation
            .finish(outcome, (transferred >= 0).then_some(transferred as u64));
        let metrics = core_metrics();
        let (seconds, bytes, failures) = if pending.write {
            (
                &metrics.ssd_cufile_write_seconds,
                &metrics.ssd_cufile_write_bytes,
                &metrics.ssd_cufile_write_failures,
            )
        } else {
            (
                &metrics.ssd_cufile_read_seconds,
                &metrics.ssd_cufile_read_bytes,
                &metrics.ssd_cufile_read_failures,
            )
        };
        seconds.record(pending.started.elapsed().as_secs_f64(), &[]);
        match result {
            Ok(()) => bytes.add(pending.batch.bytes as u64, &[]),
            Err(error) => {
                failures.add(1, &[]);
                pending.file.gpu_io.failed(&error);
            }
        }
        metrics.ssd_cufile_inflight_batches.add(-1, &[]);
    }
}

impl Drop for GpuSlot {
    fn drop(&mut self) {
        let _ = finish_gpu_transfer(&self.stream, Ok(()));
        if self.pending.is_some() {
            self.finish(
                Err("GPU storage slot drained during teardown".into()),
                Outcome::Cancelled,
            );
        }
        // SAFETY: all I/O, gather/scatter and argument access has drained.
        let cleanup = (|| {
            if self.stream_registered {
                unsafe { (self.driver.deregister_stream)(self.stream.cu_stream()) }
                    .check("cuFileStreamDeregister")?;
            }
            if self.buffer_registered {
                unsafe { (self.driver.deregister_buffer)(self.pointer as *const c_void) }
                    .check("cuFileBufDeregister")?;
            }
            unsafe { result::free_sync(self.pointer) }.map_err(|e| e.to_string())
        })();
        if let Err(error) = cleanup {
            log::error!("Cannot release GPU storage slot: {error}");
            std::process::abort();
        }
        core_metrics()
            .ssd_gpu_staging_bytes
            .add(-(STAGING_BYTES as i64), &[]);
    }
}
