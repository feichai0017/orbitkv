use std::sync::Arc;
use std::thread;

use rustix::event::{EventfdFlags, eventfd};

use super::*;

fn records() -> Arc<RestoreCompletions> {
    Arc::new(
        RestoreCompletions::create(17, 29, eventfd(0, EventfdFlags::NONBLOCK).unwrap()).unwrap(),
    )
}

#[test]
fn only_consumed_terminal_records_are_reusable_and_old_generations_stay_closed() {
    let records = records();
    let ids: Vec<_> = (0..RESTORE_COMPLETION_SLOTS)
        .map(|_| records.reserve().unwrap())
        .collect();
    for id in &ids {
        assert_eq!(records.poll(*id).unwrap().state, RestoreState::Pending);
    }
    assert!(matches!(records.reserve(), Err(CompletionError::Full)));
    records.claim(ids[0]).unwrap();
    records.complete(ids[0], Ok(())).unwrap();
    assert!(matches!(records.reserve(), Err(CompletionError::Full)));
    assert_eq!(records.poll(ids[0]).unwrap().state, RestoreState::Succeeded);
    let replacement = records.reserve().unwrap();
    assert_eq!(
        replacement as usize % RESTORE_COMPLETION_SLOTS,
        ids[0] as usize % RESTORE_COMPLETION_SLOTS
    );
    assert!(matches!(
        records.poll(ids[0]),
        Err(CompletionError::Stale(_))
    ));
    assert!(matches!(
        records.complete(ids[0], Err("late".into())),
        Err(CompletionError::Stale(_))
    ));
    assert_eq!(
        records.poll(replacement).unwrap().state,
        RestoreState::Pending
    );
}

#[test]
fn mappings_validate_session_identity_and_preserve_bounded_utf8_errors() {
    let records = records();
    let open = |epoch, token| {
        RestoreCompletions::open(
            records.file().try_clone().unwrap().into(),
            records.notification_fd().try_clone().unwrap(),
            epoch,
            token,
        )
    };
    assert!(matches!(open(18, 29), Err(CompletionError::InvalidMapping)));
    assert!(matches!(open(17, 30), Err(CompletionError::InvalidMapping)));
    let reader = open(17, 29).unwrap();
    let id = records.reserve().unwrap();
    records.claim(id).unwrap();
    let message = "错".repeat(2000);
    records.complete(id, Err(message)).unwrap();
    let response = reader.poll(id).unwrap();
    assert_eq!(response.state, RestoreState::Failed);
    assert_eq!(response.message, "错".repeat(RESTORE_ERROR_BYTES / 3));
    assert!(matches!(records.poll(id), Err(CompletionError::Stale(_))));
}

#[test]
fn concurrent_completion_and_consumption_do_not_ack_pending_or_consume_twice() {
    let records = records();
    let id = records.reserve().unwrap();
    records.claim(id).unwrap();
    let gate = std::sync::Barrier::new(3);
    thread::scope(|scope| {
        let readers: Vec<_> = (0..2)
            .map(|_| {
                scope.spawn(|| {
                    gate.wait();
                    loop {
                        match records.poll(id) {
                            Ok(response) if response.state == RestoreState::Pending => {
                                thread::yield_now();
                            }
                            Ok(response) => {
                                assert_eq!(response.message, "error-after-drain");
                                return true;
                            }
                            Err(CompletionError::Stale(_)) => return false,
                            Err(error) => panic!("{error}"),
                        }
                    }
                })
            })
            .collect();
        assert_eq!(records.poll(id).unwrap().state, RestoreState::Pending);
        gate.wait();
        records
            .complete(id, Err("error-after-drain".into()))
            .unwrap();
        assert_eq!(
            readers
                .into_iter()
                .map(|reader| reader.join().unwrap())
                .filter(|won| *won)
                .count(),
            1
        );
    });
}

#[test]
fn concurrent_reuse_never_exposes_mixed_error_bytes() {
    let records = records();
    let current = AtomicU64::new(0);
    let finished = std::sync::atomic::AtomicBool::new(false);
    let consumed = AtomicU64::new(0);
    thread::scope(|scope| {
        for _ in 0..2 {
            let records = &records;
            let current = &current;
            let finished = &finished;
            let consumed = &consumed;
            scope.spawn(move || {
                while !finished.load(Ordering::Acquire) {
                    let id = current.load(Ordering::Acquire);
                    if id == 0 {
                        thread::yield_now();
                        continue;
                    }
                    match records.poll(id) {
                        Ok(response) if response.state == RestoreState::Pending => {
                            thread::yield_now();
                        }
                        Ok(response) => {
                            let generation = (id - 1) / RESTORE_COMPLETION_SLOTS as u64;
                            let expected = if generation.is_multiple_of(2) {
                                "错".repeat(1365)
                            } else {
                                "é".repeat(2048)
                            };
                            assert_eq!(response.message, expected);
                            consumed.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(CompletionError::Stale(_)) => thread::yield_now(),
                        Err(error) => panic!("concurrent reuse returned {error}"),
                    }
                }
            });
        }
        for generation in 0..128u64 {
            records.word(NEXT_OPERATION_OFFSET).store(
                1 + generation * RESTORE_COMPLETION_SLOTS as u64,
                Ordering::Relaxed,
            );
            let id = records.reserve().unwrap();
            records.claim(id).unwrap();
            current.store(id, Ordering::Release);
            let message = if generation.is_multiple_of(2) {
                "错".repeat(1365)
            } else {
                "é".repeat(2048)
            };
            records.complete(id, Err(message)).unwrap();
            let (record, _) = RestoreCompletions::offsets(id).unwrap();
            while records.word(record).load(Ordering::Acquire) != (id << STATE_BITS) | ACKNOWLEDGED
            {
                thread::yield_now();
            }
        }
        finished.store(true, Ordering::Release);
    });
    assert_eq!(consumed.load(Ordering::Relaxed), 128);
}

#[test]
fn cancellation_can_only_win_before_claim_and_publication_is_once() {
    let records = records();
    let cancelled = records.reserve().unwrap();
    assert!(records.cancel(cancelled).unwrap());
    assert!(matches!(
        records.claim(cancelled),
        Err(CompletionError::Stale(_))
    ));
    assert!(matches!(
        records.complete(cancelled, Ok(())),
        Err(CompletionError::Stale(_))
    ));
    assert!(matches!(
        records.cancel(cancelled),
        Err(CompletionError::Stale(_))
    ));

    let submitted = records.reserve().unwrap();
    assert!(matches!(
        records.complete(submitted, Ok(())),
        Err(CompletionError::Stale(_))
    ));
    records.claim(submitted).unwrap();
    assert!(matches!(
        records.claim(submitted),
        Err(CompletionError::Stale(_))
    ));
    assert!(!records.cancel(submitted).unwrap());
    assert_eq!(
        records.poll(submitted).unwrap().state,
        RestoreState::Pending
    );
    records.complete(submitted, Ok(())).unwrap();
    assert!(matches!(
        records.complete(submitted, Err("second result".into())),
        Err(CompletionError::Stale(_))
    ));
    assert!(!records.cancel(submitted).unwrap());
    assert_eq!(
        records.poll(submitted).unwrap().state,
        RestoreState::Succeeded
    );
    assert!(matches!(
        records.claim(submitted),
        Err(CompletionError::Stale(_))
    ));
}

#[test]
fn independent_mappings_share_ids_and_claim_races_cancellation() {
    let records = records();
    let manager = RestoreCompletions::open(
        records.file().try_clone().unwrap().into(),
        records.notification_fd().try_clone().unwrap(),
        17,
        29,
    )
    .unwrap();
    let first = records.reserve().unwrap();
    let second = manager.reserve().unwrap();
    assert_ne!(first, second);
    assert!(manager.cancel(first).unwrap());
    assert!(records.cancel(second).unwrap());

    let gate = std::sync::Barrier::new(2);
    let operation = AtomicU64::new(0);
    let was_cancelled = std::sync::atomic::AtomicBool::new(false);
    thread::scope(|scope| {
        let cancelled = scope.spawn(|| {
            for _ in 0..128 {
                gate.wait();
                let id = operation.load(Ordering::Acquire);
                let cancelled = records.cancel(id).unwrap();
                was_cancelled.store(cancelled, Ordering::Release);
                gate.wait();
            }
        });
        for _ in 0..128 {
            let id = manager.reserve().unwrap();
            operation.store(id, Ordering::Release);
            gate.wait();
            let claimed = manager.claim(id).is_ok();
            gate.wait();
            assert_ne!(claimed, was_cancelled.load(Ordering::Acquire));
            if claimed {
                assert!(!records.cancel(id).unwrap());
                manager.complete(id, Ok(())).unwrap();
                assert_eq!(records.poll(id).unwrap().state, RestoreState::Succeeded);
            } else {
                assert!(matches!(manager.claim(id), Err(CompletionError::Stale(_))));
                assert!(matches!(
                    manager.complete(id, Ok(())),
                    Err(CompletionError::Stale(_))
                ));
            }
        }
        cancelled.join().unwrap();
    });
}

#[test]
fn operation_identity_exhaustion_never_wraps_into_an_old_generation() {
    let records = records();
    records
        .word(NEXT_OPERATION_OFFSET)
        .store(MAX_OPERATION_ID, Ordering::Relaxed);
    let last = records.reserve().unwrap();
    assert_eq!(last, MAX_OPERATION_ID);
    records.claim(last).unwrap();
    records.complete(last, Ok(())).unwrap();
    assert_eq!(records.poll(last).unwrap().state, RestoreState::Succeeded);
    assert!(matches!(records.reserve(), Err(CompletionError::Exhausted)));
    assert!(matches!(records.reserve(), Err(CompletionError::Exhausted)));
    assert!(matches!(
        records.claim(last),
        Err(CompletionError::Stale(_))
    ));
    for invalid in [0, MAX_OPERATION_ID + 1, u64::MAX] {
        assert!(matches!(
            records.poll(invalid),
            Err(CompletionError::Stale(_))
        ));
        assert!(matches!(
            records.claim(invalid),
            Err(CompletionError::Stale(_))
        ));
        assert!(matches!(
            records.cancel(invalid),
            Err(CompletionError::Stale(_))
        ));
    }
}
