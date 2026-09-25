use super::*;

#[test]
#[ignore = "requires CUDA and Mooncake TCP; run after all native builds"]
fn registration_holds_the_pool_until_transport_drop() {
    let pool = Arc::new(PinnedAllocator::new_global(1 << 20, 1, false, false, None));
    let memory = Arc::downgrade(&pool);
    let transport = MooncakeTransport::new(&[], pool.clone(), "127.0.0.1:0").unwrap();
    drop(pool);
    assert!(
        memory.upgrade().is_some(),
        "registered memory must outlive its original owner"
    );
    drop(transport);
    assert!(
        memory.upgrade().is_none(),
        "unregistered memory must be reclaimable"
    );
}
