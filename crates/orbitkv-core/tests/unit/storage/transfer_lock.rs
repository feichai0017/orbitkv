use super::*;

impl TransferLockManager {
    fn active_session_count(&self) -> usize {
        self.inner.lock().len()
    }

    fn total_locked_blocks(&self) -> usize {
        self.inner.lock().values().map(|s| s.blocks.len()).sum()
    }
}

fn make_test_block() -> (StateKey, Arc<SealedBlock>) {
    let key = StateKey::new("ns".into(), vec![1, 2, 3]);
    let block = Arc::new(SealedBlock::from_slots(Vec::new()));
    (key, block)
}

#[test]
fn lock_and_release() {
    let mgr = TransferLockManager::new(Duration::from_secs(30));
    let (key, block) = make_test_block();

    let session_id = mgr.lock_blocks("node-a", vec![(key.clone(), block.clone())]);
    assert_eq!(mgr.active_session_count(), 1);
    assert_eq!(mgr.total_locked_blocks(), 1);

    let released = mgr.release(&session_id);
    assert_eq!(released, 1);
    assert_eq!(mgr.active_session_count(), 0);
    assert_eq!(mgr.total_locked_blocks(), 0);
}

#[test]
fn release_unknown_session_returns_zero() {
    let mgr = TransferLockManager::new(Duration::from_secs(30));
    assert_eq!(mgr.release("nonexistent"), 0);
}

#[test]
fn gc_expired_sessions() {
    let mgr = TransferLockManager::new(Duration::from_millis(10));
    let (key, block) = make_test_block();
    let _session_id = mgr.lock_blocks("node-a", vec![(key, block)]);

    // Not expired yet
    assert_eq!(mgr.gc_expired(), 0);
    assert_eq!(mgr.active_session_count(), 1);

    // Wait for expiration
    std::thread::sleep(Duration::from_millis(20));
    assert_eq!(mgr.gc_expired(), 1);
    assert_eq!(mgr.active_session_count(), 0);
}

#[test]
fn multiple_concurrent_sessions() {
    let mgr = TransferLockManager::new(Duration::from_secs(30));
    let (key1, block1) = make_test_block();
    let key2 = StateKey::new("ns".into(), vec![4, 5, 6]);
    let block2 = Arc::new(SealedBlock::from_slots(Vec::new()));

    let s1 = mgr.lock_blocks("node-a", vec![(key1.clone(), block1.clone())]);
    let s2 = mgr.lock_blocks("node-b", vec![(key1, block1), (key2, block2)]);

    assert_eq!(mgr.active_session_count(), 2);
    assert_eq!(mgr.total_locked_blocks(), 3);

    mgr.release(&s1);
    assert_eq!(mgr.active_session_count(), 1);
    assert_eq!(mgr.total_locked_blocks(), 2);

    mgr.release(&s2);
    assert_eq!(mgr.active_session_count(), 0);
}

#[test]
fn arc_keeps_memory_alive() {
    let mgr = TransferLockManager::new(Duration::from_secs(30));
    let key = StateKey::new("ns".into(), vec![1]);
    let block = Arc::new(SealedBlock::from_slots(Vec::new()));

    // Lock holds an Arc clone
    let session_id = mgr.lock_blocks("node-a", vec![(key, block.clone())]);

    // Block has 2 strong refs: our local `block` + the lock's copy
    assert_eq!(Arc::strong_count(&block), 2);

    mgr.release(&session_id);
    // After release, only our local ref remains
    assert_eq!(Arc::strong_count(&block), 1);
}

#[test]
fn lock_empty_blocks_list() {
    let mgr = TransferLockManager::new(Duration::from_secs(30));

    let session_id = mgr.lock_blocks("node-a", vec![]);
    assert_eq!(mgr.active_session_count(), 1);
    assert_eq!(mgr.total_locked_blocks(), 0);

    let released = mgr.release(&session_id);
    assert_eq!(released, 0);
    assert_eq!(mgr.active_session_count(), 0);
}

#[test]
fn lock_with_zero_timeout_expires_immediately() {
    let mgr = TransferLockManager::new(Duration::from_secs(0));
    let (key, block) = make_test_block();

    let _session_id = mgr.lock_blocks("node-a", vec![(key, block)]);
    assert_eq!(mgr.active_session_count(), 1);

    // With zero timeout, any elapsed time > 0 means expired
    std::thread::sleep(Duration::from_millis(1));
    assert_eq!(mgr.gc_expired(), 1);
    assert_eq!(mgr.active_session_count(), 0);
    assert_eq!(mgr.total_locked_blocks(), 0);
}

#[test]
fn gc_when_no_sessions_exist() {
    let mgr = TransferLockManager::new(Duration::from_secs(30));

    // GC on empty manager returns 0 and does not panic
    assert_eq!(mgr.gc_expired(), 0);
    assert_eq!(mgr.active_session_count(), 0);
}

#[test]
fn large_number_of_concurrent_sessions() {
    let mgr = TransferLockManager::new(Duration::from_secs(300));
    let mut session_ids = Vec::new();

    let blocks_per_session = 5;
    let num_sessions = 100;

    for i in 0..num_sessions {
        let blocks: Vec<(StateKey, Arc<SealedBlock>)> = (0..blocks_per_session)
            .map(|j| {
                let key = StateKey::new("ns".into(), vec![i as u8, j as u8]);
                let block = Arc::new(SealedBlock::from_slots(Vec::new()));
                (key, block)
            })
            .collect();
        let requester = format!("node-{}", i);
        let sid = mgr.lock_blocks(&requester, blocks);
        session_ids.push(sid);
    }

    assert_eq!(mgr.active_session_count(), num_sessions);
    assert_eq!(mgr.total_locked_blocks(), num_sessions * blocks_per_session);

    // Release half
    for sid in &session_ids[..num_sessions / 2] {
        mgr.release(sid);
    }
    assert_eq!(mgr.active_session_count(), num_sessions / 2);
    assert_eq!(
        mgr.total_locked_blocks(),
        (num_sessions / 2) * blocks_per_session
    );

    // Release the rest
    for sid in &session_ids[num_sessions / 2..] {
        mgr.release(sid);
    }
    assert_eq!(mgr.active_session_count(), 0);
    assert_eq!(mgr.total_locked_blocks(), 0);
}

#[test]
fn release_same_session_twice_returns_zero_second_time() {
    let mgr = TransferLockManager::new(Duration::from_secs(30));
    let (key, block) = make_test_block();

    let session_id = mgr.lock_blocks("node-a", vec![(key, block)]);
    assert_eq!(mgr.release(&session_id), 1);
    // Second release is a no-op
    assert_eq!(mgr.release(&session_id), 0);
    assert_eq!(mgr.active_session_count(), 0);
}

#[test]
fn gc_only_removes_expired_sessions() {
    let mgr = TransferLockManager::new(Duration::from_millis(10));
    let (key1, block1) = make_test_block();

    // Session 1: will expire
    let _s1 = mgr.lock_blocks("node-a", vec![(key1, block1)]);

    std::thread::sleep(Duration::from_millis(20));

    // Session 2: fresh, should NOT expire
    let key2 = StateKey::new("ns".into(), vec![7, 8, 9]);
    let block2 = Arc::new(SealedBlock::from_slots(Vec::new()));
    let s2 = mgr.lock_blocks("node-b", vec![(key2, block2)]);

    assert_eq!(mgr.active_session_count(), 2);
    assert_eq!(mgr.gc_expired(), 1);
    assert_eq!(mgr.active_session_count(), 1);

    // The surviving session is s2
    assert_eq!(mgr.release(&s2), 1);
    assert_eq!(mgr.active_session_count(), 0);
}
