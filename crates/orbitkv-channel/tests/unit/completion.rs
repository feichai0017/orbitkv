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
    let mut next = 1;
    let ids: Vec<_> = (0..RESTORE_COMPLETION_SLOTS)
        .map(|_| records.reserve(&mut next).unwrap())
        .collect();
    for id in &ids {
        assert_eq!(records.poll(*id).unwrap().state, RestoreState::Pending);
    }
    assert!(matches!(
        records.reserve(&mut next),
        Err(CompletionError::Full)
    ));
    records.complete(ids[0], Ok(())).unwrap();
    assert!(matches!(
        records.reserve(&mut next),
        Err(CompletionError::Full)
    ));
    assert_eq!(records.poll(ids[0]).unwrap().state, RestoreState::Succeeded);
    let replacement = records.reserve(&mut next).unwrap();
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
    let id = records.reserve(&mut 1).unwrap();
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
    let id = records.reserve(&mut 1).unwrap();
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
            let mut next = 1 + generation * RESTORE_COMPLETION_SLOTS as u64;
            let id = records.reserve(&mut next).unwrap();
            current.store(id, Ordering::Release);
            let message = if generation.is_multiple_of(2) {
                "错".repeat(1365)
            } else {
                "é".repeat(2048)
            };
            records.complete(id, Err(message)).unwrap();
            let (record, _) = RestoreCompletions::offsets(id).unwrap();
            while records.word(record).load(Ordering::Acquire) != (id << 2) | ACKNOWLEDGED {
                thread::yield_now();
            }
        }
        finished.store(true, Ordering::Release);
    });
    assert_eq!(consumed.load(Ordering::Relaxed), 128);
}

#[test]
fn rollback_is_only_for_unsubmitted_pending_reservations_and_publication_is_once() {
    let records = records();
    let mut next = 1;
    let abandoned = records.reserve(&mut next).unwrap();
    records.abandon(abandoned).unwrap();
    assert!(matches!(
        records.complete(abandoned, Ok(())),
        Err(CompletionError::Stale(_))
    ));
    let submitted = records.reserve(&mut next).unwrap();
    records.complete(submitted, Ok(())).unwrap();
    assert!(matches!(
        records.complete(submitted, Err("second result".into())),
        Err(CompletionError::Stale(_))
    ));
    assert!(matches!(
        records.abandon(submitted),
        Err(CompletionError::Stale(_))
    ));
    assert_eq!(
        records.poll(submitted).unwrap().state,
        RestoreState::Succeeded
    );
}
