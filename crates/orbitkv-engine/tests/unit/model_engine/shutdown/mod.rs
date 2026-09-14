use super::*;
use crate::model_engine::{RequestId, tests::engine_with_channel};
use std::sync::{Arc, atomic::AtomicBool, mpsc};

#[test]
fn shutdown_closes_admission_and_wakes_a_full_command_queue() {
    let (commands, receiver) = mpsc::sync_channel(1);
    let (reply, _response) = tokio::sync::oneshot::channel();
    commands.send(WorkerCommand::Stats { reply }).unwrap();
    let (engine, registry) = engine_with_channel(commands);
    let cancelled = Arc::new(AtomicBool::new(false));
    registry
        .lock()
        .unwrap()
        .cancellations
        .insert(RequestId(3), Arc::clone(&cancelled));
    let shutdown = Arc::clone(&engine.shared.shutdown);
    let (release, released) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        // The shutdown command cannot enter the full queue. The already
        // queued command and atomic flag must suffice to stop the worker.
        released.recv().unwrap();
        assert!(matches!(
            receiver.recv().unwrap(),
            WorkerCommand::Stats { .. }
        ));
        assert!(shutdown.load(Ordering::Acquire));
        Ok(EngineShutdownReport::default())
    });
    *engine.shared.worker.lock().unwrap() = Some(worker);
    engine.shared.request_shutdown().unwrap();
    assert!(!registry.lock().unwrap().accepting);
    assert!(cancelled.load(Ordering::Acquire));
    release.send(()).unwrap();
    assert!(engine.shutdown().unwrap().is_drained());
    assert_eq!(engine.shutdown(), Err(ModelEngineError::WorkerUnavailable));
}

#[test]
fn shutdown_returns_worker_execution_failure() {
    let (commands, _receiver) = mpsc::sync_channel(1);
    let (engine, _) = engine_with_channel(commands);
    let error = ModelEngineError::Executor("device execution failed".into());
    let returned = error.clone();
    *engine.shared.worker.lock().unwrap() = Some(std::thread::spawn(move || Err(returned)));
    assert_eq!(engine.shutdown(), Err(error));
}

#[test]
fn shutdown_reports_worker_panic() {
    let (commands, _receiver) = mpsc::sync_channel(1);
    let (engine, _) = engine_with_channel(commands);
    *engine.shared.worker.lock().unwrap() = Some(std::thread::spawn(|| panic!("worker fault")));
    assert_eq!(
        engine.shutdown(),
        Err(ModelEngineError::WorkerPanicked("worker fault".into()))
    );
}

#[test]
fn drain_requires_retired_token_and_fixed_state_ownership() {
    let pool = orbitkv::StateCheckpointPool::new(1, 1, 1, 128, 2).unwrap();
    let mut report = EngineShutdownReport {
        fixed_states: vec![(0, pool.stats())].into_boxed_slice(),
        ..Default::default()
    };
    assert!(report.is_drained());
    report.stats.manager.total_reader_pins = 1;
    assert!(!report.is_drained());
    report.stats.manager.total_reader_pins = 0;
    report.stats.manager.quarantined_pages = 1;
    assert!(!report.is_drained());
    report.stats.manager.quarantined_pages = 0;
    report.fixed_states[0].1.pending_transitions = 1;
    assert!(!report.is_drained());
    report.fixed_states[0].1.pending_transitions = 0;
    report.fixed_states[0].1.free_slots -= 1;
    assert!(!report.is_drained());
}
