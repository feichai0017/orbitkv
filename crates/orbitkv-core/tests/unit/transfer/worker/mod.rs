use super::*;

fn empty_restore_plan(device_id: i32) -> crate::planning::restore::RestorePlan {
    crate::planning::restore::RestorePlan::new(
        device_id,
        std::iter::empty::<(usize, &crate::RestoreSource)>(),
    )
    .unwrap()
}

#[test]
fn gpu_ssd_write_admission_is_shared_per_device_and_recreated_after_release() {
    let first = device_ssd_write_admission(i32::MAX);
    let same = device_ssd_write_admission(i32::MAX);
    let other = device_ssd_write_admission(i32::MAX - 1);
    assert!(Arc::ptr_eq(&first, &same));
    assert!(!Arc::ptr_eq(&first, &other));
    let permits = (0..ssd::MAX_WRITES)
        .map(|_| Arc::clone(&first).try_acquire_owned().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(same.available_permits(), 0);
    assert!(Arc::clone(&same).try_acquire_owned().is_err());
    drop(permits);
    assert_eq!(same.available_permits(), ssd::MAX_WRITES);
    drop(first);
    drop(same);
    let replacement = device_ssd_write_admission(i32::MAX);
    assert_eq!(replacement.available_permits(), ssd::MAX_WRITES);
}

#[test]
fn cufile_worker_ownership_is_shared_per_device() {
    let first = device_cufile_worker_admission(i32::MAX - 10);
    let same = device_cufile_worker_admission(i32::MAX - 10);
    let other = device_cufile_worker_admission(i32::MAX - 11);
    assert!(Arc::ptr_eq(&first, &same));
    assert!(!Arc::ptr_eq(&first, &other));
    let owner = Arc::clone(&first).try_acquire_owned().unwrap();
    assert!(Arc::clone(&same).try_acquire_owned().is_err());
    assert!(Arc::clone(&other).try_acquire_owned().is_ok());
    drop(owner);
    assert!(Arc::clone(&same).try_acquire_owned().is_ok());
}

#[test]
fn save_task_owns_shared_admission_until_terminal_drop() {
    let admission = device_ssd_write_admission(i32::MAX - 2);
    let permit = Arc::clone(&admission).try_acquire_owned().unwrap();
    let (reply, _receiver) = oneshot::channel();
    let command = WorkerCommand::Save(
        SaveTask {
            layers: Vec::new(),
            reply,
            ssd_writes: Vec::new(),
            codec_groups: Vec::new(),
            storage: None,
            numa: NumaNode::UNKNOWN,
            ssd_admission: Some(permit),
            #[cfg(feature = "tracing")]
            trace_ctx: None,
        },
        Observation::disabled(),
    );
    assert_eq!(admission.available_permits(), ssd::MAX_WRITES - 1);
    drop(command);
    assert_eq!(admission.available_permits(), ssd::MAX_WRITES);
}

#[tokio::test]
async fn shared_admission_saturation_falls_back_before_worker_submission() {
    for blocked_cufile_owner in [false, true] {
        let write_admission = Arc::new(Semaphore::new(ssd::MAX_WRITES));
        let cufile_admission = Arc::new(Semaphore::new(1));
        let write_permits = if blocked_cufile_owner {
            Vec::new()
        } else {
            (0..ssd::MAX_WRITES)
                .map(|_| Arc::clone(&write_admission).try_acquire_owned().unwrap())
                .collect::<Vec<_>>()
        };
        let cufile_permit = blocked_cufile_owner
            .then(|| Arc::clone(&cufile_admission).try_acquire_owned().unwrap());
        let (load_tx, _load_rx) = mpsc::unbounded_channel();
        let (save_tx, mut save_rx) = mpsc::unbounded_channel();
        let pool = GpuWorkerPool {
            device_id: i32::MAX - 3,
            numa_node: NumaNode::UNKNOWN,
            transfer_mode: TransferMode::Direct,
            ssd_tx: Mutex::new(None),
            ssd_host_tx: Mutex::new(None),
            codec_write_tx: Mutex::new(None),
            ssd_write_admission: Arc::clone(&write_admission),
            cufile_worker_admission: Arc::clone(&cufile_admission),
            cufile_worker_owner: Mutex::new(None),
            load_tx,
            save_tx,
            closed: Mutex::new(false),
            drained: OnceCell::new(),
        };
        let mut saving = Box::pin(pool.batch_save(
            Vec::new(),
            Vec::new(),
            vec![SaveGroup {
                key: crate::block::StateKey::new("shared-admission".into(), vec![1]),
                blocks: Vec::new(),
            }],
            None,
        ));
        assert!(futures::poll!(saving.as_mut()).is_pending());
        let Some(WorkerCommand::Save(task, _)) = save_rx.recv().await else {
            panic!("saturated GPU write must use the ordinary save lane")
        };
        assert!(task.codec_groups.is_empty());
        assert!(task.ssd_writes.is_empty());
        assert!(task.ssd_admission.is_none());
        assert!(task.reply.send(Ok(task.layers)).is_ok());
        saving.await.unwrap();
        drop(pool);
        drop(write_permits);
        drop(cufile_permit);
        assert_eq!(write_admission.available_permits(), ssd::MAX_WRITES);
        assert_eq!(cufile_admission.available_permits(), 1);
    }
}

#[test]
fn overlapping_restore_targets_are_rejected_before_worker_or_codec_dispatch() {
    let (load_tx, mut load_rx) = mpsc::unbounded_channel();
    let (save_tx, _save_rx) = mpsc::unbounded_channel();
    let pool = GpuWorkerPool {
        device_id: 0,
        numa_node: NumaNode::UNKNOWN,
        transfer_mode: TransferMode::Direct,
        ssd_tx: Mutex::new(None),
        ssd_host_tx: Mutex::new(None),
        codec_write_tx: Mutex::new(None),
        ssd_write_admission: Arc::new(Semaphore::new(ssd::MAX_WRITES)),
        cufile_worker_admission: Arc::new(Semaphore::new(1)),
        cufile_worker_owner: Mutex::new(None),
        load_tx,
        save_tx,
        closed: Mutex::new(false),
        drained: OnceCell::new(),
    };
    let mut layout = KVCacheLayout::new(0x10000, 257 * 4096, 257, 4096, 0, 1).unwrap();
    layout.storage_format = orbitkv_state::StorageFormat::Fp8FromBf16;
    for codec_budget in [4096, 64 * 1024 * 1024] {
        for indices in [vec![0, 0], (0..257).chain([0]).collect()] {
            let (completion, _) = oneshot::channel();
            let task = LoadTask {
                plan: empty_restore_plan(0),
                layers: vec![LayerTransferData {
                    layer_name: "attention".into(),
                    layout: layout.clone(),
                    blocks: indices
                        .into_iter()
                        .map(|block_idx| TransferBlock {
                            block_idx,
                            // Rejection must precede source access or GPU work.
                            block: TransferPayload::Pending,
                        })
                        .collect(),
                }],
                completion,
                reservations: vec![],
                codec_budget,
            };
            let error = pool.submit_load(task).unwrap_err();
            assert!(error.to_string().contains("overlap"));
            assert!(matches!(
                load_rx.try_recv(),
                Err(mpsc::error::TryRecvError::Empty)
            ));
        }
    }
}

#[test]
fn restore_plan_must_target_the_worker_device() {
    let (load_tx, mut load_rx) = mpsc::unbounded_channel();
    let (save_tx, _save_rx) = mpsc::unbounded_channel();
    let pool = GpuWorkerPool {
        device_id: 0,
        numa_node: NumaNode::UNKNOWN,
        transfer_mode: TransferMode::Direct,
        ssd_tx: Mutex::new(None),
        ssd_host_tx: Mutex::new(None),
        codec_write_tx: Mutex::new(None),
        ssd_write_admission: Arc::new(Semaphore::new(ssd::MAX_WRITES)),
        cufile_worker_admission: Arc::new(Semaphore::new(1)),
        cufile_worker_owner: Mutex::new(None),
        load_tx,
        save_tx,
        closed: Mutex::new(false),
        drained: OnceCell::new(),
    };
    let (completion, _) = oneshot::channel();
    let error = pool
        .submit_load(LoadTask {
            plan: empty_restore_plan(1),
            layers: Vec::new(),
            completion,
            reservations: Vec::new(),
            codec_budget: 0,
        })
        .unwrap_err();
    assert!(error.to_string().contains("targets device 1"));
    assert!(matches!(
        load_rx.try_recv(),
        Err(mpsc::error::TryRecvError::Empty)
    ));
}

#[tokio::test]
async fn drain_rejects_new_transfers_and_waits_for_all_workers() {
    let (load_tx, mut load_rx) = mpsc::unbounded_channel();
    let (save_tx, mut save_rx) = mpsc::unbounded_channel();
    let (ssd_tx, mut ssd_rx) = mpsc::unbounded_channel();
    let (ssd_host_tx, mut ssd_host_rx) = mpsc::unbounded_channel();
    let (codec_write_tx, mut codec_write_rx) = mpsc::unbounded_channel();
    let cufile_admission = Arc::new(Semaphore::new(1));
    let cufile_owner = Arc::clone(&cufile_admission).try_acquire_owned().unwrap();
    let pool = Arc::new(GpuWorkerPool {
        device_id: 0,
        numa_node: NumaNode::UNKNOWN,
        transfer_mode: TransferMode::Direct,
        ssd_tx: Mutex::new(Some(ssd_tx)),
        ssd_host_tx: Mutex::new(Some(ssd_host_tx)),
        codec_write_tx: Mutex::new(Some(codec_write_tx)),
        ssd_write_admission: Arc::new(Semaphore::new(ssd::MAX_WRITES)),
        cufile_worker_admission: Arc::clone(&cufile_admission),
        cufile_worker_owner: Mutex::new(Some(cufile_owner)),
        load_tx,
        save_tx,
        closed: Mutex::new(false),
        drained: OnceCell::new(),
    });
    let (reply, _result) = oneshot::channel();
    pool.submit_load(LoadTask {
        plan: empty_restore_plan(0),
        layers: vec![],
        completion: reply,
        reservations: vec![],
        codec_budget: 64 * 1024 * 1024,
    })
    .unwrap();
    let draining = Arc::clone(&pool);
    let waiter = tokio::spawn(async move { draining.drain().await });
    assert!(matches!(
        load_rx.recv().await,
        Some(WorkerCommand::Load(..))
    ));
    let Some(WorkerCommand::Drain(load_ack)) = load_rx.recv().await else {
        panic!("missing load barrier")
    };
    let Some(WorkerCommand::Drain(save_ack)) = save_rx.recv().await else {
        panic!("missing save barrier")
    };
    let Some(WorkerCommand::Drain(ssd_ack)) = ssd_rx.recv().await else {
        panic!("missing SSD barrier")
    };
    let Some(WorkerCommand::Drain(codec_write_ack)) = codec_write_rx.recv().await else {
        panic!("missing encoded writeback barrier")
    };
    let Some(WorkerCommand::Drain(ssd_host_ack)) = ssd_host_rx.recv().await else {
        panic!("missing SSD host barrier")
    };
    let (reply, _) = oneshot::channel();
    assert!(
        pool.submit_load(LoadTask {
            plan: empty_restore_plan(0),
            layers: vec![],
            completion: reply,
            reservations: vec![],
            codec_budget: 64 * 1024 * 1024,
        })
        .is_err()
    );
    assert!(pool.batch_save(vec![], vec![], vec![], None).await.is_err());
    load_ack.send(Ok(())).unwrap();
    tokio::task::yield_now().await;
    assert!(
        !waiter.is_finished(),
        "load completion alone must not release mappings"
    );
    save_ack.send(Ok(())).unwrap();
    tokio::task::yield_now().await;
    assert!(
        !waiter.is_finished(),
        "SSD completion must precede mapping release"
    );
    ssd_ack.send(Ok(())).unwrap();
    tokio::task::yield_now().await;
    assert!(!waiter.is_finished());
    codec_write_ack.send(Ok(())).unwrap();
    tokio::task::yield_now().await;
    assert!(
        !waiter.is_finished(),
        "host reads must drain before mapping release"
    );
    ssd_host_ack.send(Ok(())).unwrap();
    waiter.await.unwrap().unwrap();
    assert_eq!(cufile_admission.available_permits(), 1);
    pool.drain().await.unwrap();
}

#[test]
fn transfer_cost_shape_uses_logical_ranges_and_actual_encoding() {
    use std::num::NonZeroU64;

    use crate::block::Segment;
    use crate::codec::EncodedSegment;
    use crate::memory::pool::PinnedAllocator;
    use orbitkv_state::StorageFormat;

    let pool = PinnedAllocator::new_global(4096, 1, false, false, None);
    let block = |bytes: usize| {
        let allocation = pool
            .allocate(NonZeroU64::new(bytes as u64).unwrap(), NumaNode::UNKNOWN)
            .unwrap();
        RawBlock::single_segment(Segment::new(allocation.as_non_null(), bytes, allocation))
    };
    let mut encoded = block(512);
    encoded.storage_format = StorageFormat::Ans;
    encoded.encoding = Some(vec![EncodedSegment {
        version: 1,
        format: StorageFormat::Ans,
        logical_bytes: 600,
        stored_bytes: 100,
        checksum: 0,
    }]);
    let layers = vec![LayerTransferData {
        layer_name: "split".into(),
        layout: KVCacheLayout::new(0x10000, 4096, 2, 300, 2048, 2)
            .unwrap()
            .with_ssd_padding(512),
        blocks: vec![
            TransferBlock {
                block_idx: 0,
                block: TransferPayload::Owned(encoded),
            },
            TransferBlock {
                block_idx: 1,
                block: TransferPayload::Owned(block(1024)),
            },
        ],
    }];
    assert_eq!(transfer_shape(&layers), (1200, 4));
    let (key, bytes) = transfer_key(&layers, 2, TransferMode::Direct, false, false);
    assert_eq!(bytes, 1200);
    assert_eq!(
        key,
        CostEstimateKey::new(
            CostObservationKind::GpuDecode,
            ExecutionResource::Gpu(2),
            Representation::Mixed,
            1200,
            4
        )
    );
    let (key, _) = transfer_key(&layers, 2, TransferMode::Direct, false, true);
    assert_eq!(
        key,
        CostEstimateKey::new(
            CostObservationKind::GpuSsdLoad,
            ExecutionResource::Gpu(2),
            Representation::Mixed,
            1200,
            4
        )
    );
}

#[test]
fn failed_gpu_work_is_not_reclassified_by_consumer_loss() {
    for (completed, closed, expected) in [
        (true, false, Outcome::Completed),
        (true, true, Outcome::Cancelled),
        (false, false, Outcome::Failed),
        (false, true, Outcome::Failed),
    ] {
        assert_eq!(terminal_outcome(completed, closed), expected);
    }
}

#[test]
fn raw_copy_candidates_distinguish_dma_coalescing_and_direction() {
    let mut host = [0u8; 16];
    let contiguous: Vec<_> = (0..4)
        .map(|index| CopyDesc {
            device: 0x1000 + (index * 4) as u64,
            host: host.as_mut_ptr().wrapping_add(index * 4),
            host_device: 0x2000 + (index * 4) as u64,
            size: 4,
            device_allocation: 1,
            host_allocation: 2,
        })
        .collect();
    let mut fragmented = contiguous.clone();
    fragmented.swap(1, 2);
    let mut allocations = contiguous.clone();
    for (index, copy) in allocations.iter_mut().enumerate() {
        copy.host_allocation = index;
    }
    for write in [false, true] {
        let (merged_keys, bytes) = raw_copy_keys(&contiguous, 3, write);
        assert_eq!(bytes, 16);
        let paths = if write {
            [
                CostObservationKind::GpuSaveDirect,
                CostObservationKind::GpuSaveKernel,
            ]
        } else {
            [
                CostObservationKind::GpuLoadDirect,
                CostObservationKind::GpuLoadKernel,
            ]
        };
        for (key, path) in merged_keys.iter().zip(paths) {
            assert_eq!(
                *key,
                CostEstimateKey::new(path, ExecutionResource::Gpu(3), Representation::Raw, 16, 4)
                    .with_dma_ranges(1)
            );
        }
        for copies in [&fragmented, &allocations] {
            let (keys, bytes) = raw_copy_keys(copies, 3, write);
            assert_eq!(bytes, 16);
            for ((key, merged_key), path) in keys.iter().zip(merged_keys).zip(paths) {
                assert_ne!(*key, merged_key);
                assert_eq!(
                    *key,
                    CostEstimateKey::new(
                        path,
                        ExecutionResource::Gpu(3),
                        Representation::Raw,
                        16,
                        4
                    )
                    .with_dma_ranges(4)
                );
            }
        }
        assert_ne!(merged_keys, raw_copy_keys(&contiguous, 3, !write).0);
    }
}
