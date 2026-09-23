use super::*;

#[test]
fn validate_direct_io_rejects_unaligned_offset() {
    let iovecs = vec![(SSD_ALIGNMENT, SSD_ALIGNMENT)];
    let err = UringIoEngine::validate_direct_io(iovecs, 1).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
}

#[test]
fn validate_direct_io_rejects_unaligned_buffer() {
    let iovecs = vec![(SSD_ALIGNMENT + 1, SSD_ALIGNMENT)];
    let err = UringIoEngine::validate_direct_io(iovecs, 0).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
}

#[test]
fn validate_direct_io_rejects_unaligned_length() {
    let iovecs = vec![(SSD_ALIGNMENT, SSD_ALIGNMENT - 1)];
    let err = UringIoEngine::validate_direct_io(iovecs, 0).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
}

#[test]
fn a_full_write_queue_does_not_block_single_file_reads() {
    use std::sync::Arc;
    use std::time::Duration;

    let (write_tx, write_rx) = mpsc::sync_channel(1);
    let (read_tx, read_rx) = mpsc::sync_channel(1);
    let engine = Arc::new(UringIoEngine {
        fds: vec![0], // No kernel worker: the test completes queued I/O itself.
        txs: vec![write_tx, read_tx],
        write_shards: 1,
        next_read: AtomicUsize::new(0),
        handles: Vec::new(),
    });
    #[repr(align(512))]
    struct Buffer([u8; SSD_ALIGNMENT]);
    let source = Buffer([1; SSD_ALIGNMENT]);
    let write = engine
        .writev_at_async(0, vec![(source.0.as_ptr(), SSD_ALIGNMENT)], 0)
        .unwrap();
    let (submitted_tx, submitted_rx) = mpsc::channel();
    let reader = std::thread::spawn(move || {
        let mut target = Buffer([0; SSD_ALIGNMENT]);
        let read = engine
            .readv_at_async(0, vec![(target.0.as_mut_ptr(), SSD_ALIGNMENT)], 0)
            .unwrap();
        submitted_tx.send(()).unwrap();
        // Keep the target alive until our simulated kernel completion.
        read.blocking_recv().unwrap().unwrap()
    });

    let independent = submitted_rx.recv_timeout(Duration::from_secs(1)).is_ok();
    write_rx
        .recv()
        .unwrap()
        .complete
        .send(Ok(SSD_ALIGNMENT))
        .unwrap();
    assert_eq!(write.blocking_recv().unwrap().unwrap(), SSD_ALIGNMENT);
    if !independent {
        // Unblock even a regressed dispatcher before failing the assertion.
        submitted_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    }
    read_rx
        .try_recv()
        .or_else(|_| write_rx.try_recv())
        .unwrap()
        .complete
        .send(Ok(SSD_ALIGNMENT))
        .unwrap();
    assert_eq!(reader.join().unwrap(), SSD_ALIGNMENT);
    assert!(independent, "a read waited for space in the write queue");
}
