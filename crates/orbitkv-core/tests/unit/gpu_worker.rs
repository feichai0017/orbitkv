use super::*;

#[test]
fn failed_submission_waits_for_already_enqueued_stream_work() {
    struct Gate {
        entered: std_mpsc::Sender<()>,
        release: std_mpsc::Receiver<()>,
    }
    unsafe extern "C" fn hold_stream(data: *mut std::ffi::c_void) {
        // SAFETY: the successful launch transfers this Box to one callback.
        let gate = unsafe { Box::from_raw(data.cast::<Gate>()) };
        let _ = gate.entered.send(());
        let _ = gate.release.recv_timeout(std::time::Duration::from_secs(5));
    }

    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let (entered_tx, entered_rx) = std_mpsc::channel();
    let (release_tx, release_rx) = std_mpsc::channel();
    let gate = Box::into_raw(Box::new(Gate {
        entered: entered_tx,
        release: release_rx,
    }));
    // SAFETY: stream and callback data stay alive through synchronization;
    // the callback does not call CUDA or unwind across the C ABI.
    let launched = unsafe {
        cudarc::driver::result::stream::launch_host_function(
            stream.cu_stream(),
            hold_stream,
            gate.cast(),
        )
    };
    if let Err(error) = launched {
        // SAFETY: a failed launch did not take ownership of the callback.
        unsafe { drop(Box::from_raw(gate)) };
        panic!("host callback launch failed: {error}");
    }
    let (finished_tx, finished_rx) = std_mpsc::channel();
    let worker = std::thread::spawn(move || {
        let result = finish_gpu_transfer(&stream, Err("partial submission".to_string()));
        let _ = finished_tx.send(result);
    });
    entered_rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .unwrap();
    let premature = finished_rx.recv_timeout(std::time::Duration::from_millis(50));
    let _ = release_tx.send(());
    worker.join().unwrap();
    assert!(matches!(
        premature,
        Err(std_mpsc::RecvTimeoutError::Timeout)
    ));
    assert!(
        finished_rx
            .recv()
            .unwrap()
            .unwrap_err()
            .to_string()
            .contains("partial submission")
    );
}

#[tokio::test]
async fn drain_rejects_new_transfers_and_waits_for_both_workers() {
    let (load_tx, mut load_rx) = mpsc::unbounded_channel();
    let (save_tx, mut save_rx) = mpsc::unbounded_channel();
    let pool = Arc::new(GpuWorkerPool {
        device_id: 0,
        load_tx,
        save_tx,
        closed: Mutex::new(false),
        drained: OnceCell::new(),
    });
    let (reply, _result) = oneshot::channel();
    pool.submit_load(LoadTask {
        layers: vec![],
        completion: LoadCompletion::Channel(reply),
        reservations: vec![],
    })
    .unwrap();
    let draining = Arc::clone(&pool);
    let waiter = tokio::spawn(async move { draining.drain().await });
    assert!(matches!(
        load_rx.recv().await,
        Some(WorkerCommand::Transfer(_))
    ));
    let Some(WorkerCommand::Drain(load_ack)) = load_rx.recv().await else {
        panic!("missing load barrier")
    };
    let Some(WorkerCommand::Drain(save_ack)) = save_rx.recv().await else {
        panic!("missing save barrier")
    };
    let (reply, _) = oneshot::channel();
    assert!(
        pool.submit_load(LoadTask {
            layers: vec![],
            completion: LoadCompletion::Channel(reply),
            reservations: vec![],
        })
        .is_err()
    );
    assert!(pool.batch_save(vec![]).await.is_err());
    load_ack.send(Ok(())).unwrap();
    tokio::task::yield_now().await;
    assert!(
        !waiter.is_finished(),
        "load completion alone must not release mappings"
    );
    save_ack.send(Ok(())).unwrap();
    waiter.await.unwrap().unwrap();
    pool.drain().await.unwrap();
}
