use super::*;
use smallvec::smallvec;

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

// ========================================================================
// group_slots_by_numa tests
// ========================================================================

fn make_slot(numa_node: NumaNode, size: u64) -> SlotMeta {
    SlotMeta::new(smallvec![size], numa_node)
}

fn make_prefetch_request(n: u8, slots: Vec<SlotMeta>) -> PrefetchRequest {
    let total_size: u64 = slots.iter().map(|s| s.total_size()).sum();
    PrefetchRequest {
        key: make_key(n),
        entry: SsdIndexEntry {
            shard_id: 0,
            begin: 0,
            len: total_size,
            file_offset: 0,
            slots,
        },
    }
}

#[test]
fn test_group_slots_non_numa_single_group() {
    // When !is_numa, all slots collapse into None regardless of numa_node
    let requests = vec![
        make_prefetch_request(
            1,
            vec![make_slot(NumaNode(0), 100), make_slot(NumaNode(1), 100)],
        ),
        make_prefetch_request(2, vec![make_slot(NumaNode(0), 200)]),
    ];

    let groups = group_slots_by_numa(false, &requests);
    assert_eq!(groups.len(), 1);
    assert!(groups.contains_key(&None));
    assert_eq!(groups[&None].len(), 3); // all 3 slots in one group
}

#[test]
fn test_group_slots_numa_tp8_split() {
    // TP8: 4 slots on NUMA0, 4 on NUMA1, 2 blocks
    let slots = |n0, n1| {
        vec![
            make_slot(NumaNode(0), 64),
            make_slot(NumaNode(0), 64),
            make_slot(NumaNode(0), 64),
            make_slot(NumaNode(0), 64),
            make_slot(NumaNode(n0), 64),
            make_slot(NumaNode(n0), 64),
            make_slot(NumaNode(n1), 64),
            make_slot(NumaNode(n1), 64),
        ]
    };
    let requests = vec![
        make_prefetch_request(1, slots(1, 1)),
        make_prefetch_request(2, slots(1, 1)),
    ];

    let groups = group_slots_by_numa(true, &requests);
    assert_eq!(groups.len(), 2); // NUMA0 and NUMA1

    let numa0 = &groups[&Some(NumaNode(0))];
    let numa1 = &groups[&Some(NumaNode(1))];

    // 4 slots/block * 2 blocks = 8 per NUMA
    assert_eq!(numa0.len(), 8);
    assert_eq!(numa1.len(), 8);

    // Total size: 8 * 64 = 512 per group
    let total_0: u64 = numa0.iter().map(|r| r.size).sum();
    let total_1: u64 = numa1.iter().map(|r| r.size).sum();
    assert_eq!(total_0, 512);
    assert_eq!(total_1, 512);

    // Verify slots reference correct slot indices
    assert!(numa0.iter().all(|r| r.slot_idx < 4));
    assert!(numa1.iter().all(|r| r.slot_idx >= 4));
}

#[test]
fn test_group_slots_unknown_maps_to_none() {
    let requests = vec![make_prefetch_request(
        1,
        vec![
            make_slot(NumaNode(0), 100),
            make_slot(NumaNode::UNKNOWN, 100),
        ],
    )];

    let groups = group_slots_by_numa(true, &requests);
    assert_eq!(groups.len(), 2);
    assert_eq!(groups[&Some(NumaNode(0))].len(), 1);
    assert_eq!(groups[&None].len(), 1); // UNKNOWN → None
}

#[test]
fn test_k3_style_prefetch_uses_bounded_numa_chunks() {
    const MIB: u64 = 1024 * 1024;
    let slots = || {
        (0..8)
            .map(|slot| make_slot(NumaNode(slot / 4), 16 * MIB))
            .collect()
    };
    let requests: Vec<_> = (1..=56)
        .map(|block| make_prefetch_request(block, slots()))
        .collect();

    let groups = group_slots_by_numa(true, &requests);
    for numa in [NumaNode(0), NumaNode(1)] {
        let refs = &groups[&Some(numa)];
        let whole_batch_bytes: u64 = refs.iter().map(|slot| slot.size).sum();
        let chunks = chunk_slot_refs(refs, SSD_PREFETCH_CHUNK_BYTES).unwrap();

        assert_eq!(whole_batch_bytes, 3584 * MIB);
        assert_eq!(chunks.len(), 14);
        assert_eq!(
            chunks.iter().map(|chunk| chunk.size).max(),
            Some(SSD_PREFETCH_CHUNK_BYTES)
        );
        assert_eq!(
            chunks.iter().map(|chunk| chunk.slots.len()).sum::<usize>(),
            refs.len()
        );
    }
}

#[test]
fn test_oversized_prefetch_slot_gets_dedicated_chunk() {
    let refs = vec![
        SlotRef {
            block_idx: 0,
            slot_idx: 0,
            size: 300,
        },
        SlotRef {
            block_idx: 0,
            slot_idx: 1,
            size: 100,
        },
    ];

    let chunks = chunk_slot_refs(&refs, 256).unwrap();
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].size, 300);
    assert_eq!(chunks[0].slots[0].offset, 0);
    assert_eq!(chunks[1].size, 100);
    assert_eq!(chunks[1].slots[0].offset, 0);
}

#[test]
fn test_non_divisible_prefetch_chunks_allocate_only_slot_bytes() {
    let refs = [200, 100, 100]
        .into_iter()
        .enumerate()
        .map(|(slot_idx, size)| SlotRef {
            block_idx: 0,
            slot_idx,
            size,
        })
        .collect::<Vec<_>>();

    let chunks = chunk_slot_refs(&refs, 256).unwrap();
    let allocation_sizes = chunks.iter().map(|chunk| chunk.size).collect::<Vec<_>>();

    assert_eq!(allocation_sizes, vec![200, 200]);
    assert_eq!(allocation_sizes.iter().sum::<u64>(), 400);
}
