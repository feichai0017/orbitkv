use super::*;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

fn test_allocate_fn(calls: Arc<AtomicUsize>) -> AllocateFn {
    let allocator = Arc::new(crate::pinned_pool::PinnedAllocator::new_global(
        32 * 1024 * 1024,
        1,
        false,
        false,
        None,
    ));
    Arc::new(move |size, _numa| {
        calls.fetch_add(1, Ordering::Relaxed);
        allocator.allocate(NonZeroU64::new(size)?, NumaNode::UNKNOWN)
    })
}

fn remaining(bytes: u64) -> HashMap<NumaNode, u64> {
    HashMap::from([(NumaNode(0), bytes)])
}

#[test]
fn chunked_slabs_bump_within_chunk_then_refill() {
    let calls = Arc::new(AtomicUsize::new(0));
    let allocate_fn = test_allocate_fn(Arc::clone(&calls));
    let mut slabs = ChunkedSlabs::new(&allocate_fn, 1024, remaining(1536));

    let (p1, a1) = slabs.alloc_segment(NumaNode(0), 512, "K").expect("first");
    let (p2, _a2) = slabs.alloc_segment(NumaNode(0), 512, "V").expect("second");
    assert_eq!(p2.as_ptr() as usize - p1.as_ptr() as usize, 512);
    assert_eq!(calls.load(Ordering::Relaxed), 1);

    // Third segment exceeds the current chunk: a fresh chunk is allocated
    // while earlier segments stay valid through their own chunk Arc.
    let (_p3, a3) = slabs.alloc_segment(NumaNode(0), 512, "K").expect("third");
    assert_eq!(calls.load(Ordering::Relaxed), 2);
    assert_eq!(slabs.chunk_count, 2);
    assert!(!Arc::ptr_eq(&a1, &a3));
}

#[test]
fn chunked_slabs_oversized_segment_gets_dedicated_chunk() {
    let calls = Arc::new(AtomicUsize::new(0));
    let allocate_fn = test_allocate_fn(Arc::clone(&calls));
    let mut slabs = ChunkedSlabs::new(&allocate_fn, 1024, remaining(4096));

    slabs
        .alloc_segment(NumaNode(0), 4096, "K")
        .expect("oversized segment");
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(slabs.chunk_count, 1);
}

#[test]
fn chunked_slabs_allocation_failure_is_an_error() {
    let allocate_fn: AllocateFn = Arc::new(|_, _| None);
    let mut slabs = ChunkedSlabs::new(&allocate_fn, 1024, remaining(512));

    let err = match slabs.alloc_segment(NumaNode(0), 512, "K") {
        Ok(_) => panic!("allocation should fail"),
        Err(err) => err,
    };
    assert!(err.contains("failed to allocate fetch chunk"));
}

#[test]
fn chunked_slabs_chunk_clamped_to_batch_remaining() {
    // A small fetch must not request the whole chunk_bytes cap — that
    // fails outright on pools smaller than the cap (jz p2p IT regression).
    let sizes = Arc::new(Mutex::new(Vec::new()));
    let recorded = Arc::clone(&sizes);
    let inner = test_allocate_fn(Arc::new(AtomicUsize::new(0)));
    let allocate_fn: AllocateFn = Arc::new(move |size, numa| {
        recorded.lock().unwrap().push(size);
        inner(size, numa)
    });
    let mut slabs = ChunkedSlabs::new(&allocate_fn, 256 << 20, remaining(4096));

    slabs.alloc_segment(NumaNode(0), 1024, "K").expect("first");
    slabs.alloc_segment(NumaNode(0), 3072, "V").expect("second");

    // One chunk sized to the batch total, not to the 256 MiB cap.
    assert_eq!(*sizes.lock().unwrap(), vec![4096]);
    assert_eq!(slabs.chunk_count, 1);
}
