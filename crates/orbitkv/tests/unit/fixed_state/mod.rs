use super::*;

fn receipt(intent: StateCopyIntent) -> StateCopyReceipt {
    StateCopyReceipt {
        transition: intent.transition,
        source: intent.source.unwrap_or_default(),
        destination: intent.destination,
        byte_count: intent.byte_count,
        source_present: u8::from(intent.source.is_some()),
        observed: 1,
        written: 1,
        reserved8: 0,
        reserved32: 0,
    }
}

#[test]
fn initial_and_replacement_state_are_generation_checked() {
    let mut pool = StateCheckpointPool::new(1, 2, 3, 4096, 2).unwrap();
    let initial = pool.prepare(7, None).unwrap();
    pool.submit(receipt(initial)).unwrap();
    let first = pool
        .complete(
            initial.transition,
            StateCompletionReceipt {
                engine_epoch: 1,
                completion_domain: 4,
                completion_value: 1,
                confirmed: true,
            },
        )
        .unwrap();
    assert_eq!(pool.current(7), Some(first.slot));
    assert!(first.retirement.is_none());

    let replacement = pool.prepare(7, Some(first.slot)).unwrap();
    pool.submit(receipt(replacement)).unwrap();
    let second = pool
        .complete(
            replacement.transition,
            StateCompletionReceipt {
                engine_epoch: 1,
                completion_domain: 4,
                completion_value: 2,
                confirmed: true,
            },
        )
        .unwrap();
    let retirement = second.retirement.unwrap();
    assert_ne!(first.slot, second.slot);
    assert_eq!(
        pool.prepare(8, None),
        Err(StateCheckpointError::PoolExhausted)
    );
    pool.acknowledge(retirement).unwrap();
    let reused = pool.prepare(8, None).unwrap();
    assert_eq!(reused.destination.slot_id, first.slot.slot_id);
    assert!(reused.destination.generation > first.slot.generation);
}

#[test]
fn owner_release_requires_completion_and_ack_before_reuse() {
    let mut pool = StateCheckpointPool::new(1, 2, 3, 1024, 2).unwrap();
    let prepared = pool.prepare(7, None).unwrap();
    pool.submit(receipt(prepared)).unwrap();
    let published = pool
        .complete(
            prepared.transition,
            StateCompletionReceipt {
                engine_epoch: 1,
                completion_domain: 4,
                completion_value: 1,
                confirmed: true,
            },
        )
        .unwrap();
    assert_eq!(
        pool.retire_owner(
            7,
            published.slot,
            StateCompletionReceipt {
                engine_epoch: 1,
                completion_domain: 4,
                completion_value: 2,
                confirmed: false,
            },
        ),
        Err(StateCheckpointError::CompletionNotConfirmed)
    );
    let certificate = pool
        .retire_owner(
            7,
            published.slot,
            StateCompletionReceipt {
                engine_epoch: 1,
                completion_domain: 4,
                completion_value: 2,
                confirmed: true,
            },
        )
        .unwrap();
    assert_eq!(pool.current(7), None);
    let other = pool.prepare(8, None).unwrap();
    assert_ne!(other.destination.slot_id, published.slot.slot_id);
    pool.abort(other.transition, true).unwrap();
    pool.acknowledge(certificate).unwrap();
    let reused = pool.prepare(9, None).unwrap();
    assert_eq!(reused.destination.slot_id, published.slot.slot_id);
    assert!(reused.destination.generation > published.slot.generation);
}

#[test]
fn abort_is_recoverable_but_observed_mismatch_quarantines() {
    let mut pool = StateCheckpointPool::new(1, 2, 3, 16, 2).unwrap();
    let prepared = pool.prepare(7, None).unwrap();
    pool.abort(prepared.transition, true).unwrap();
    let retried = pool.prepare(7, None).unwrap();
    let mut malformed = receipt(retried);
    malformed.byte_count += 1;
    assert_eq!(
        pool.submit(malformed),
        Err(StateCheckpointError::CopyReceiptMismatch)
    );
    assert_eq!(
        pool.prepare(7, None),
        Err(StateCheckpointError::OwnerQuarantined)
    );
}

#[test]
fn abort_without_unobserved_proof_quarantines_the_entire_batch() {
    let mut pool = StateCheckpointPool::new(1, 2, 3, 16, 2).unwrap();
    let prepared = pool.prepare_batch(&[(7, None), (8, None)]).unwrap();
    assert_eq!(
        pool.abort_batch(&[
            (prepared[0].transition, true),
            (prepared[1].transition, false),
        ]),
        Err(StateCheckpointError::CopyObservationUnknown)
    );
    let stats = pool.stats();
    assert_eq!(stats.free_slots, 0);
    assert_eq!(stats.reserved_slots, 0);
    assert_eq!(stats.quarantined_slots, 2);
    assert_eq!(stats.pending_transitions, 0);
    assert_eq!(
        pool.prepare(7, None),
        Err(StateCheckpointError::OwnerQuarantined)
    );
    assert_eq!(
        pool.prepare(8, None),
        Err(StateCheckpointError::OwnerQuarantined)
    );
}

#[test]
fn batch_preflight_is_atomic_and_census_is_exact() {
    let mut pool = StateCheckpointPool::new(1, 2, 3, 16, 2).unwrap();
    let before = pool.stats();
    assert_eq!(
        pool.prepare_batch(&[(7, None), (7, None)]),
        Err(StateCheckpointError::OwnerBusy)
    );
    assert_eq!(pool.stats(), before);

    let prepared = pool.prepare_batch(&[(7, None), (8, None)]).unwrap();
    assert_eq!(pool.stats().reserved_slots, 2);
    let mut receipts = prepared.iter().copied().map(receipt).collect::<Vec<_>>();
    receipts[1].byte_count += 1;
    assert_eq!(
        pool.submit_batch(&receipts),
        Err(StateCheckpointError::CopyReceiptMismatch)
    );
    assert_eq!(pool.stats().reserved_slots, 0);
    assert_eq!(pool.stats().quarantined_slots, 2);
    assert_eq!(pool.stats().pending_transitions, 0);
    assert_eq!(
        pool.prepare(7, None),
        Err(StateCheckpointError::OwnerQuarantined)
    );
}
