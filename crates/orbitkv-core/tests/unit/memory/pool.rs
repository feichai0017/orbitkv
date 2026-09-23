use super::*;

#[test]
fn numa_largest_free_for_node_is_not_global_min() {
    let mut pools = HashMap::new();
    pools.insert(
        0,
        ShardedPinnedPool::new(
            4096,
            1,
            false,
            false,
            NonZeroU64::new(512),
            NumaNode::UNKNOWN,
        ),
    );
    pools.insert(
        1,
        ShardedPinnedPool::new(
            4096,
            1,
            false,
            false,
            NonZeroU64::new(512),
            NumaNode::UNKNOWN,
        ),
    );
    let allocator = PinnedAllocator::Numa(NumaAwarePinnedPools { pools });

    let pinned = match &allocator {
        PinnedAllocator::Numa(pools) => pools.pools.get(&0).unwrap(),
        _ => unreachable!(),
    };
    let _held = pinned.allocate(NonZeroU64::new(3584).unwrap()).unwrap();

    let global_min = match &allocator {
        PinnedAllocator::Numa(pools) => pools
            .pools
            .values()
            .map(|p| p.largest_free_allocation())
            .min()
            .unwrap_or(0),
        _ => unreachable!(),
    };
    let node1_largest = allocator.largest_free_allocation_for_node(NumaNode(1));

    assert_eq!(global_min, 512);
    assert_eq!(node1_largest, 4096);
}

#[test]
fn numa_largest_free_for_unknown_node_is_zero() {
    let mut pools = HashMap::new();
    pools.insert(
        0,
        ShardedPinnedPool::new(
            4096,
            1,
            false,
            false,
            NonZeroU64::new(512),
            NumaNode::UNKNOWN,
        ),
    );
    let allocator = PinnedAllocator::Numa(NumaAwarePinnedPools { pools });

    assert_eq!(
        allocator.largest_free_allocation_for_node(NumaNode::UNKNOWN),
        0
    );
}
