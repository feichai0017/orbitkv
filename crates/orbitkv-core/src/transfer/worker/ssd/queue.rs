use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use cudarc::driver::{CudaEvent, result, sys};
use tokio::sync::{mpsc, oneshot};

use crate::EngineError;
use crate::backing::ssd::GpuWriteLease;
use crate::backing::ssd::cufile::{CufileFile, GpuSlot, IoBatch, STAGING_SLOTS};
use crate::transfer::finish_gpu_transfer;

use super::super::{
    LoadTask, SaveTask, WorkerCommand, WorkerRuntime, build_copy_descs, finish_load,
};

pub(in crate::transfer::worker) const MAX_WRITES: usize = 8;
const READ_BURST: usize = 4;

enum Task {
    Load(LoadTask),
    Save(SaveTask),
}

struct Work {
    file: Arc<CufileFile>,
    batch: IoBatch,
    write: Option<usize>,
}

struct Write {
    lease: Option<GpuWriteLease>,
    remaining: usize,
}

#[derive(Clone, Copy)]
struct BatchOwner {
    job: u64,
    write: Option<usize>,
}

struct Job {
    id: u64,
    task: Task,
    work: VecDeque<Work>,
    writes: Vec<Write>,
    inflight: usize,
    error: Option<String>,
    prepared: bool,
    host_completion: Option<CudaEvent>,
    started: Instant,
    bytes: usize,
}

impl Job {
    fn new(id: u64, command: WorkerCommand) -> Self {
        let task = match command {
            WorkerCommand::Load(task) => Task::Load(task),
            WorkerCommand::Save(task) => Task::Save(task),
            WorkerCommand::Drain(_) => unreachable!("drain is handled by the queue"),
        };
        let mut job = Self {
            id,
            task,
            work: VecDeque::new(),
            writes: Vec::new(),
            inflight: 0,
            error: None,
            prepared: false,
            host_completion: None,
            started: Instant::now(),
            bytes: 0,
        };
        let planned = (|| {
            match &mut job.task {
                Task::Load(task) => {
                    let plans = super::plan(&task.layers)?;
                    for (file, batches) in plans {
                        for batch in batches {
                            job.bytes += batch.copies.iter().map(|copy| copy.bytes).sum::<usize>();
                            job.work.push_back(Work {
                                file: Arc::clone(&file),
                                batch,
                                write: None,
                            });
                        }
                    }
                }
                Task::Save(task) => {
                    for (index, write) in
                        std::mem::take(&mut task.ssd_writes).into_iter().enumerate()
                    {
                        let file = Arc::clone(write.lease.file());
                        job.writes.push(Write {
                            lease: Some(write.lease),
                            remaining: write.batches.len(),
                        });
                        for batch in write.batches {
                            job.work.push_back(Work {
                                file: Arc::clone(&file),
                                batch,
                                write: Some(index),
                            });
                        }
                    }
                }
            }
            Ok::<(), EngineError>(())
        })();
        if let Err(error) = planned {
            job.fail(error.to_string());
        }
        job
    }

    fn prepare(&mut self, runtime: &WorkerRuntime) -> Result<(), EngineError> {
        if self.prepared {
            return Ok(());
        }
        let layers = match &self.task {
            Task::Load(task) => &task.layers,
            Task::Save(task) => &task.layers,
        };
        let (copies, bytes) = build_copy_descs(layers)?;
        if !copies.is_empty() {
            let event = runtime
                .stream
                .context()
                .new_event(None)
                .map_err(|error| EngineError::Storage(error.to_string()))?;
            let submitted = if self.is_write() {
                runtime.backend.d2h(&copies, &runtime.stream)
            } else {
                runtime.backend.h2d(&copies, &runtime.stream)
            };
            if let Err(error) = submitted.and_then(|()| {
                event
                    .record(&runtime.stream)
                    .map_err(|error| error.to_string())
            }) {
                // Partial submission still owns the job's host and GPU pages.
                return finish_gpu_transfer(&runtime.stream, Err(error));
            }
            self.host_completion = Some(event);
        }
        if !self.is_write() {
            self.bytes += bytes;
        }
        self.prepared = true;
        Ok(())
    }

    fn poll_host(&mut self, runtime: &WorkerRuntime) {
        let Some(event) = &self.host_completion else {
            return;
        };
        // This event covers only this job. A later copy on the shared stream
        // must not delay its completion or block cuFile submission/polling.
        match unsafe { result::event::query(event.cu_event()) } {
            Ok(()) => {
                #[cfg(feature = "test-hooks")]
                if crate::test_faults::active("ssd_host_completion") {
                    return;
                }
                self.host_completion = None;
            }
            Err(error) if error.0 == sys::CUresult::CUDA_ERROR_NOT_READY => {}
            Err(error) => {
                let _ = finish_gpu_transfer(&runtime.stream, Ok(()));
                self.host_completion = None;
                self.fail(error.to_string());
            }
        }
    }

    fn is_write(&self) -> bool {
        matches!(self.task, Task::Save(_))
    }

    fn fail(&mut self, error: String) {
        self.error.get_or_insert(error);
        self.work.clear();
    }

    fn cancel_abandoned(&mut self) {
        let closed = match &self.task {
            Task::Load(task) => task.completion.is_closed(),
            Task::Save(task) => task.reply.is_closed(),
        };
        if closed {
            self.fail("GPU transfer consumer closed".into());
        }
    }

    fn complete(&mut self, write: Option<usize>, result: Result<(), String>) {
        self.inflight -= 1;
        if let Some(index) = write {
            let write = &mut self.writes[index];
            write.remaining -= 1;
            if write.remaining == 0 && result.is_ok() {
                write
                    .lease
                    .take()
                    .expect("unpublished extent owned")
                    .commit();
            }
        }
        if let Err(error) = result {
            self.fail(error);
        }
    }

    fn finish(self) {
        assert_eq!(self.inflight, 0, "submitted I/O still owns this job");
        assert!(
            self.host_completion.is_none(),
            "host copy still owns this job"
        );
        // Drop unpublished extents before telling the caller it can reuse pages.
        drop(self.writes);
        let result = self
            .error
            .map_or(Ok(()), |error| Err(EngineError::Storage(error)));
        match self.task {
            Task::Load(task) => finish_load(task, result, self.started, self.bytes),
            Task::Save(task) => {
                drop(task.ssd_admission);
                let _ = task.reply.send(result.map(|()| task.layers));
            }
        }
    }
}

/// Two in-flight batches; at most one write leaves capacity for demand reads.
/// Read bursts and round-robin jobs bound starvation without preempting DMA.
pub(in crate::transfer::worker) fn run(
    mut receiver: mpsc::UnboundedReceiver<WorkerCommand>,
    runtime: WorkerRuntime,
) {
    let mut jobs: VecDeque<Job> = VecDeque::new();
    let mut slots: Vec<(GpuSlot, Option<BatchOwner>)> = Vec::new();
    let mut drain: Option<oneshot::Sender<Result<(), String>>> = None;
    let mut closed = false;
    let mut next_id = 0;
    let mut read_burst = 0;
    let mut reset_slots = false;
    loop {
        // Bound CPU admission work per turn so producers cannot starve completions.
        for _ in 0..32 {
            if closed {
                break;
            }
            let command = if jobs.is_empty() {
                receiver.blocking_recv()
            } else {
                match receiver.try_recv() {
                    Ok(command) => Some(command),
                    Err(mpsc::error::TryRecvError::Empty) => break,
                    Err(mpsc::error::TryRecvError::Disconnected) => None,
                }
            };
            match command {
                Some(WorkerCommand::Drain(reply)) => {
                    drain = Some(reply);
                    closed = true;
                }
                Some(command) => {
                    jobs.push_back(Job::new(next_id, command));
                    next_id += 1;
                }
                None => closed = true,
            }
        }
        for job in &mut jobs {
            job.poll_host(&runtime);
            job.cancel_abandoned();
        }
        for (slot, owner) in &mut slots {
            if let Some(result) = slot.poll() {
                reset_slots |= result.is_err();
                let owner = owner.take().expect("submitted slot has an owner");
                jobs.iter_mut()
                    .find(|job| job.id == owner.job)
                    .expect("submitted job retained")
                    .complete(owner.write, result);
            }
        }
        // Errors can leave a stream unusable. Keep every other slot alive until
        // its submitted work completes, then recreate the registered resources.
        if reset_slots && slots.iter().all(|(_, owner)| owner.is_none()) {
            slots.clear();
            reset_slots = false;
        }
        let mut index = 0;
        while index < jobs.len() {
            if jobs[index].inflight == 0
                && jobs[index].work.is_empty()
                && jobs[index].host_completion.is_none()
            {
                jobs.remove(index).expect("completed job exists").finish();
            } else {
                index += 1;
            }
        }
        if jobs.is_empty() {
            if closed {
                drop(slots);
                if let Some(reply) = drain {
                    let _ = reply.send(Ok(()));
                }
                break;
            }
            continue;
        }
        if slots.is_empty() {
            let created = (0..STAGING_SLOTS)
                .map(|_| GpuSlot::new(runtime.stream.context()).map(|slot| (slot, None)))
                .collect::<Result<Vec<_>, _>>();
            match created {
                Ok(created) => slots = created,
                Err(error) => {
                    for job in &mut jobs {
                        for work in &job.work {
                            work.file.gpu_io.failed(&error);
                        }
                        job.fail(error.clone());
                    }
                    continue;
                }
            }
        }
        for index in 0..slots.len() {
            if reset_slots {
                break;
            }
            if slots[index].1.is_some() {
                continue;
            }
            let writing = slots
                .iter()
                .any(|(_, owner)| owner.is_some_and(|owner| owner.write.is_some()));
            let eligible = |job: &Job| {
                if job.work.is_empty() {
                    return false;
                }
                #[cfg(feature = "test-hooks")]
                if crate::test_faults::active(if job.is_write() {
                    "cufile_write"
                } else {
                    "cufile"
                }) {
                    return false;
                }
                true
            };
            let read = jobs.iter().position(|job| !job.is_write() && eligible(job));
            let write = if writing {
                None
            } else {
                jobs.iter().position(|job| job.is_write() && eligible(job))
            };
            let selected = match (read, write) {
                (Some(_), Some(write)) if read_burst >= READ_BURST => Some((write, false)),
                (Some(read), _) => Some((read, true)),
                (_, Some(write)) => Some((write, false)),
                _ => None,
            };
            let Some((position, reading)) = selected else {
                continue;
            };
            read_burst = if reading {
                (read_burst + 1).min(READ_BURST)
            } else {
                0
            };
            let mut job = jobs.remove(position).expect("selected job exists");
            if let Err(error) = job.prepare(&runtime) {
                job.fail(error.to_string());
                jobs.push_back(job);
                continue;
            }
            let work = job.work.pop_front().expect("eligible job has work");
            let owner = BatchOwner {
                job: job.id,
                write: work.write,
            };
            match slots[index].0.submit(work.file, work.batch, !reading) {
                Ok(()) => {
                    job.inflight += 1;
                    slots[index].1 = Some(owner);
                }
                Err(error) => {
                    job.fail(error);
                    reset_slots = true;
                }
            }
            jobs.push_back(job);
        }
        std::thread::sleep(Duration::from_micros(50));
    }
}
