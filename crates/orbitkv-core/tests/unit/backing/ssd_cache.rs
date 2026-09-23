use super::*;

fn make_key(n: u8) -> StateKey {
    StateKey::new("test".to_string(), vec![n])
}

impl SsdRingBuffer {
    fn test_entry(&self, shard_id: usize, begin: u64, len: u64) -> SsdIndexEntry {
        SsdIndexEntry {
            shard_id,
            begin,
            len,
            file_offset: begin % self.shards[shard_id].capacity.max(1),
            slots: vec![],
        }
    }

    /// Insert a Committed entry for testing. Returns the key.
    fn insert_committed(&mut self, n: u8, begin: u64, len: u64) -> StateKey {
        let key = make_key(n);
        let entry = self.test_entry(0, begin, len);
        self.entries
            .insert(key.clone(), SsdEntryState::Committed(entry));
        self.shards[0].order.push_back(key.clone());
        key
    }

    /// Insert a Writing entry for testing. Returns the key.
    fn insert_writing(&mut self, n: u8, begin: u64, len: u64) -> StateKey {
        let key = make_key(n);
        let entry = self.test_entry(0, begin, len);
        self.entries
            .insert(key.clone(), SsdEntryState::Writing(entry));
        self.shards[0].order.push_back(key.clone());
        key
    }
}

// ========================================================================
// allocate_contiguous tests
// ========================================================================

#[test]
fn test_allocate_contiguous_basic() {
    let mut ring = SsdRingBuffer::new(1000);

    let (begin, offset) = ring.allocate_contiguous(0, 100).unwrap();
    assert_eq!(
        (begin, offset, ring.shards[0].head, ring.shards[0].tail),
        (0, 0, 100, 0)
    );

    let (begin, offset) = ring.allocate_contiguous(0, 200).unwrap();
    assert_eq!(
        (begin, offset, ring.shards[0].head, ring.shards[0].tail),
        (100, 100, 300, 0)
    );
}

#[test]
fn test_allocate_contiguous_wrap_around() {
    let mut ring = SsdRingBuffer::new(1000);
    ring.allocate_contiguous(0, 900).unwrap();

    // 200 bytes doesn't fit in remaining 100, skips to wrap point
    let (begin, offset) = ring.allocate_contiguous(0, 200).unwrap();
    assert_eq!(begin, 1000); // skipped to wrap point
    assert_eq!(offset, 0); // wraps to file start
    assert_eq!(ring.shards[0].tail, 200); // head(1200) - capacity(1000)
}

#[test]
fn test_allocate_contiguous_prunes_expired() {
    let mut ring = SsdRingBuffer::new(1000);
    let key = ring.insert_committed(1, 0, 100);

    ring.allocate_contiguous(0, 600).unwrap();
    ring.allocate_contiguous(0, 600).unwrap(); // head=1200, tail=200

    assert!(!ring.entries.contains_key(&key));
    assert!(ring.shards[0].order.is_empty());
}

// ========================================================================
// is_offset_valid tests
// ========================================================================

#[test]
fn test_is_offset_valid() {
    let mut ring = SsdRingBuffer::new(1000);
    assert!(ring.is_offset_valid(&ring.test_entry(0, 0, 10)));

    ring.shards[0].tail = 50;
    assert!(!ring.is_offset_valid(&ring.test_entry(0, 49, 10)));
    assert!(ring.is_offset_valid(&ring.test_entry(0, 50, 10)));
}

// ========================================================================
// commit tests
// ========================================================================

#[test]
fn test_commit_writing_to_committed() {
    let mut ring = SsdRingBuffer::new(1000);
    let key = ring.insert_writing(1, 100, 50);

    assert!(ring.commit(&key, true));
    assert!(matches!(
        ring.entries.get(&key),
        Some(SsdEntryState::Committed(_))
    ));
}

#[test]
fn test_commit_failure_removes_entry() {
    let mut ring = SsdRingBuffer::new(1000);
    let key = ring.insert_writing(1, 100, 50);

    assert!(!ring.commit(&key, false));
    assert!(!ring.entries.contains_key(&key));
    assert_eq!(ring.shards[0].order.len(), 1); // order cleaned by advance_tail later
}

#[test]
fn test_commit_expired_entry() {
    let mut ring = SsdRingBuffer::new(1000);
    let key = ring.insert_writing(1, 100, 50);
    ring.shards[0].tail = 200; // expire it

    assert!(!ring.commit(&key, true));
    assert!(!ring.entries.contains_key(&key));
}

#[test]
fn test_commit_edge_cases() {
    let mut ring = SsdRingBuffer::new(1000);

    // Missing key
    assert!(!ring.commit(&make_key(99), true));

    // Already committed (idempotent)
    let key = ring.insert_committed(1, 100, 50);
    assert!(ring.commit(&key, true));
}

// ========================================================================
// get tests
// ========================================================================

#[test]
fn test_get_committed_entries() {
    let mut ring = SsdRingBuffer::new(1000);
    let k_writing = ring.insert_writing(1, 100, 50);
    let k_committed = ring.insert_committed(2, 200, 50);

    // Writing: not readable
    assert!(ring.get(&k_writing).is_none());

    // Committed: readable
    let entry = ring.get(&k_committed).unwrap();
    assert_eq!((entry.begin, entry.len), (200, 50));

    // Expired: not readable
    ring.shards[0].tail = 250;
    assert!(ring.get(&k_committed).is_none());
}

// ========================================================================
// advance_tail tests
// ========================================================================

#[test]
fn test_advance_tail_prunes_expired() {
    let mut ring = SsdRingBuffer::new(1000);
    ring.insert_committed(0, 0, 50);
    ring.insert_committed(1, 100, 50);
    ring.insert_committed(2, 200, 50);

    ring.advance_tail(0, 150);

    assert_eq!(ring.entries.len(), 1);
    assert!(ring.get(&make_key(2)).is_some());
}

#[test]
fn test_advance_tail_cleans_ghost_entries() {
    let mut ring = SsdRingBuffer::new(1000);
    // Ghost: in order but not in entries (aborted write)
    ring.shards[0].order.push_back(make_key(1));
    ring.insert_committed(2, 200, 50);

    ring.advance_tail(0, 100);

    assert_eq!(ring.shards[0].order.len(), 1);
    assert_eq!(ring.shards[0].order.front(), Some(&make_key(2)));
}

#[test]
fn test_advance_tail_preserves_valid_writing() {
    let mut ring = SsdRingBuffer::new(1000);
    ring.insert_writing(1, 100, 50);

    ring.advance_tail(0, 50); // tail < begin, should preserve

    assert_eq!(ring.entries.len(), 1);
}

// ========================================================================
// Duplicate key filtering
// ========================================================================

#[test]
fn test_duplicate_key_filtered() {
    let mut ring = SsdRingBuffer::new(1000);
    let key = ring.insert_writing(1, 100, 50);

    let filtered: Vec<_> = vec![key]
        .into_iter()
        .filter(|k| !ring.entries.contains_key(k))
        .collect();

    assert!(filtered.is_empty());
}
