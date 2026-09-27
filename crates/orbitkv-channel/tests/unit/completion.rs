use std::sync::Arc;
use std::thread;

use rustix::event::{EventfdFlags, eventfd};

use super::*;

fn records() -> Arc<RestoreCompletions> {
    Arc::new(
        RestoreCompletions::create(17, 29, eventfd(0, EventfdFlags::NONBLOCK).unwrap()).unwrap(),
    )
}

fn independent(records: &RestoreCompletions) -> RestoreCompletions {
    RestoreCompletions::open(
        records.file().try_clone().unwrap().into(),
        records.notification_fd().try_clone().unwrap(),
        records.manager_notification_fd().try_clone().unwrap(),
        17,
        29,
    )
    .unwrap()
}

#[test]
fn only_reaped_and_acknowledged_records_recycle() {
    let records = records();
    let engine = independent(&records);
    let ids: Vec<_> = (0..RESTORE_COMPLETION_SLOTS)
        .map(|_| engine.reserve().unwrap())
        .collect();
    assert!(matches!(records.reserve(), Err(CompletionError::Full)));
    let id = ids[0];
    records.claim(id).unwrap();
    assert!(records.publish_local(id, b"checked source plan").unwrap());
    assert_eq!(engine.poll(id).unwrap().state, RestoreState::Pending);
    assert_eq!(
        engine.claim_local(id).unwrap().unwrap(),
        b"checked source plan"
    );
    assert!(!records.revoke(id).unwrap());
    assert!(matches!(records.reserve(), Err(CompletionError::Full)));
    engine.drained(id, Ok(())).unwrap();
    assert_eq!(engine.poll(id).unwrap().state, RestoreState::Pending);
    assert_eq!(
        records.manager_updates().unwrap(),
        vec![(id, GrantState::Drained)]
    );
    assert!(records.drain_succeeded(id).unwrap());
    records.release_plan(id).unwrap();
    records.reap(id).unwrap();
    assert!(matches!(records.reserve(), Err(CompletionError::Full)));
    assert_eq!(engine.poll(id).unwrap().state, RestoreState::Succeeded);
    let replacement = engine.reserve().unwrap();
    assert_eq!(
        replacement as usize % RESTORE_COMPLETION_SLOTS,
        id as usize % RESTORE_COMPLETION_SLOTS
    );
    assert!(matches!(records.claim(id), Err(CompletionError::Stale(_))));
    assert!(matches!(
        engine.drained(id, Ok(())),
        Err(CompletionError::Stale(_))
    ));
    assert_eq!(records.state(replacement).unwrap(), GrantState::Reserved);
}

#[test]
fn plan_bank_is_bounded_and_consumption_does_not_release_sources() {
    let records = records();
    let engine = independent(&records);
    let first = engine.reserve().unwrap();
    records.claim(first).unwrap();
    let large = vec![0x5a; RESTORE_PLAN_BYTES];
    assert!(records.publish_local(first, &large).unwrap());
    let second = engine.reserve().unwrap();
    records.claim(second).unwrap();
    assert!(matches!(
        records.publish_local(second, b"next"),
        Err(CompletionError::PlanFull)
    ));
    assert!(records.release_plan(first).is_err());
    assert_eq!(engine.claim_local(first).unwrap().unwrap(), large);
    assert_eq!(
        records.manager_updates().unwrap(),
        vec![(first, GrantState::Active)]
    );
    records.release_plan(first).unwrap();
    assert!(records.publish_local(second, b"next").unwrap());
    assert_eq!(records.state(first).unwrap(), GrantState::Active);
    assert!(records.reap(first).is_err());
    assert_eq!(engine.claim_local(second).unwrap().unwrap(), b"next");
    engine
        .drained(second, Err("partial enqueue drained".into()))
        .unwrap();
    engine.drained(first, Ok(())).unwrap();
    for id in [first, second] {
        records.reap(id).unwrap();
    }
    assert_eq!(
        engine.poll(second).unwrap().message,
        "partial enqueue drained"
    );
    assert_eq!(engine.poll(first).unwrap().state, RestoreState::Succeeded);
}

#[test]
fn cancellation_has_a_separate_preparation_drain() {
    let records = records();
    let before = records.reserve().unwrap();
    assert!(records.cancel(before).unwrap());
    assert!(records.claim(before).is_err());
    let during = records.reserve().unwrap();
    records.claim(during).unwrap();
    assert!(!records.cancel(during).unwrap());
    assert_eq!(records.poll(during).unwrap().state, RestoreState::Pending);
    assert!(!records.publish_local(during, b"not published").unwrap());
    records.finish_cancelled(during).unwrap();
    assert_eq!(records.poll(during).unwrap().state, RestoreState::Failed);
    // A managed worker can already have submitted before route publication.
    let managed = records.reserve().unwrap();
    records.claim(managed).unwrap();
    assert!(!records.cancel(managed).unwrap());
    records.start_managed(managed).unwrap();
    assert_eq!(records.poll(managed).unwrap().state, RestoreState::Pending);
    records.complete(managed, Ok(())).unwrap();
    assert_eq!(
        records.poll(managed).unwrap().state,
        RestoreState::Succeeded
    );
}

#[test]
fn claim_and_revoke_have_exactly_one_winner_across_mappings() {
    let records = records();
    let engine = independent(&records);
    for _ in 0..128 {
        let id = engine.reserve().unwrap();
        records.claim(id).unwrap();
        records.publish_local(id, b"plan").unwrap();
        let gate = std::sync::Barrier::new(2);
        thread::scope(|scope| {
            let claim = scope.spawn(|| {
                gate.wait();
                engine.claim_local(id)
            });
            gate.wait();
            let revoked = records.revoke(id).unwrap();
            match claim.join().unwrap() {
                Ok(Some(plan)) => {
                    assert!(!revoked);
                    assert_eq!(plan, b"plan");
                    engine.drained(id, Ok(())).unwrap();
                }
                Ok(None) | Err(CompletionError::Stale(_)) => assert!(revoked),
                other => panic!("unexpected claim: {other:?}"),
            }
            assert_eq!(records.drain_succeeded(id).unwrap(), !revoked);
            records.reap(id).unwrap();
            assert_eq!(
                engine.poll(id).unwrap().state,
                if revoked {
                    RestoreState::Failed
                } else {
                    RestoreState::Succeeded
                }
            );
        });
    }
}

#[test]
fn bounded_dirty_bits_preserve_all_operations_without_queue_overflow() {
    let records = records();
    let engine = independent(&records);
    let ids: Vec<_> = (0..RESTORE_COMPLETION_SLOTS)
        .map(|_| engine.reserve().unwrap())
        .collect();
    for id in &ids {
        records.claim(*id).unwrap();
        records.publish_local(*id, b"plan").unwrap();
        engine.claim_local(*id).unwrap().unwrap();
        engine.drained(*id, Ok(())).unwrap();
    }
    let updates = records.manager_updates().unwrap();
    assert_eq!(updates.len(), RESTORE_COMPLETION_SLOTS);
    assert!(
        updates
            .iter()
            .all(|(_, state)| *state == GrantState::Drained)
    );
    assert!(records.manager_updates().unwrap().is_empty());
    for (id, _) in updates {
        records.reap(id).unwrap();
        engine.poll(id).unwrap();
    }
    assert!(records.plans.lock().unwrap().is_empty());
}

#[test]
fn stale_readers_cannot_mix_recycled_error_generations() {
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
                                "错".repeat(RESTORE_ERROR_BYTES / 3)
                            } else {
                                "é".repeat(RESTORE_ERROR_BYTES / 2)
                            };
                            assert_eq!(response.message, expected);
                            consumed.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(CompletionError::Stale(_)) => thread::yield_now(),
                        Err(error) => panic!("{error}"),
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
            records.start_managed(id).unwrap();
            current.store(id, Ordering::Release);
            let message = if generation.is_multiple_of(2) {
                "错".repeat(2000)
            } else {
                "é".repeat(2000)
            };
            records.complete(id, Err(message)).unwrap();
            while records.state(id).unwrap() != GrantState::Acknowledged {
                thread::yield_now();
            }
        }
        finished.store(true, Ordering::Release);
    });
    assert_eq!(consumed.load(Ordering::Relaxed), 128);
}

#[test]
fn mapping_identity_and_operation_exhaustion_are_checked() {
    let records = records();
    assert!(matches!(
        RestoreCompletions::open(
            records.file().try_clone().unwrap().into(),
            records.notification_fd().try_clone().unwrap(),
            records.manager_notification_fd().try_clone().unwrap(),
            18,
            29
        ),
        Err(CompletionError::InvalidMapping)
    ));
    records
        .word(NEXT_OPERATION_OFFSET)
        .store(MAX_OPERATION_ID, Ordering::Relaxed);
    let last = records.reserve().unwrap();
    assert_eq!(last, MAX_OPERATION_ID);
    records.claim(last).unwrap();
    records.start_managed(last).unwrap();
    records.complete(last, Ok(())).unwrap();
    assert_eq!(records.poll(last).unwrap().state, RestoreState::Succeeded);
    assert!(matches!(records.reserve(), Err(CompletionError::Exhausted)));
    for invalid in [0, MAX_OPERATION_ID + 1, u64::MAX] {
        assert!(records.poll(invalid).is_err());
        assert!(records.claim(invalid).is_err());
        assert!(records.cancel(invalid).is_err());
    }
}

#[test]
fn admission_and_preparation_cancellation_race_without_losing_the_operation() {
    let records = records();
    let engine = independent(&records);
    let first = engine.reserve().unwrap();
    let second = records.reserve().unwrap();
    assert_ne!(first, second);
    assert!(engine.cancel(first).unwrap());
    assert!(records.cancel(second).unwrap());
    let gate = std::sync::Barrier::new(2);
    let operation = AtomicU64::new(0);
    let safe_cancel = std::sync::atomic::AtomicBool::new(false);
    thread::scope(|scope| {
        let cancel = scope.spawn(|| {
            for _ in 0..128 {
                gate.wait();
                safe_cancel.store(
                    engine.cancel(operation.load(Ordering::Acquire)).unwrap(),
                    Ordering::Release,
                );
                gate.wait();
            }
        });
        for _ in 0..128 {
            let id = engine.reserve().unwrap();
            operation.store(id, Ordering::Release);
            gate.wait();
            let admitted = records.claim(id).is_ok();
            gate.wait();
            assert_ne!(admitted, safe_cancel.load(Ordering::Acquire));
            assert!(records.claim(id).is_err(), "admission must be exactly once");
            if admitted {
                assert_eq!(records.state(id).unwrap(), GrantState::CancelRequested);
                assert_eq!(engine.poll(id).unwrap().state, RestoreState::Pending);
                records.finish_cancelled(id).unwrap();
                assert_eq!(engine.poll(id).unwrap().state, RestoreState::Failed);
            } else {
                assert_eq!(records.state(id).unwrap(), GrantState::Acknowledged);
            }
        }
        cancel.join().unwrap();
    });
}
