use super::*;

#[test]
fn tcp_loopback_moves_bytes_through_upstream_mooncake() {
    unsafe {
        libc::setenv(c"MC_FORCE_TCP".as_ptr(), c"1".as_ptr(), 1);
    }
    let engine = TransferEngine::new("P2PHANDSHAKE", "127.0.0.1:0", "127.0.0.1", 0, &[])
        .expect("create Mooncake Transfer Engine");
    let segment = engine.local_segment_name().expect("local segment");
    let mut memory = vec![0u8; 8192];
    memory[..4096].fill(0xa5);
    let base = NonNull::new(memory.as_mut_ptr()).expect("memory pointer");
    unsafe {
        engine
            .register_memory(base, memory.len(), "cpu:0")
            .expect("register memory");
    }
    let destination = unsafe { base.byte_add(4096) };
    engine
        .submit_and_notify(
            TransferOp::Write,
            &segment,
            &[TransferSlice {
                local: base,
                remote_address: destination.as_ptr() as u64,
                length: 4096,
            }],
            Duration::from_secs(5),
            &Notification {
                name: "loopback".to_string(),
                message: "done".to_string(),
            },
        )
        .expect("loopback write");
    assert_eq!(&memory[..4096], &memory[4096..]);
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let notifications = engine.take_notifications().expect("take notifications");
        if notifications
            .iter()
            .any(|notification| notification.name == "loopback" && notification.message == "done")
        {
            break;
        }
        assert!(Instant::now() < deadline, "notification timed out");
        std::thread::yield_now();
    }
    unsafe {
        engine.unregister_memory(base).expect("unregister memory");
        libc::unsetenv(c"MC_FORCE_TCP".as_ptr());
    }
}

#[test]
fn engine_creation_restores_the_process_nic_filter() {
    let _guard = ENGINE_CREATE_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let original = std::env::var_os("MC_TE_FILTERS");
    unsafe {
        libc::setenv(c"MC_TE_FILTERS".as_ptr(), c"original".as_ptr(), 1);
    }
    let filter =
        ScopedNicFilter::apply(&["temporary".to_string()]).expect("apply temporary NIC filter");
    assert_eq!(
        std::env::var_os("MC_TE_FILTERS").as_deref(),
        Some(std::ffi::OsStr::new("temporary"))
    );
    drop(filter);
    assert_eq!(
        std::env::var_os("MC_TE_FILTERS").as_deref(),
        Some(std::ffi::OsStr::new("original"))
    );
    unsafe {
        match original {
            Some(value) => {
                let value = CString::new(value.as_bytes()).expect("original environment");
                libc::setenv(c"MC_TE_FILTERS".as_ptr(), value.as_ptr(), 1);
            }
            None => {
                libc::unsetenv(c"MC_TE_FILTERS".as_ptr());
            }
        }
    }
}
