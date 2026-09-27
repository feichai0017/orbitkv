use super::*;

#[test]
fn fixed_tent_strings_are_bounded_even_without_a_terminator() {
    assert_eq!(fixed_c_string(&[b'a' as c_char, 0, b'b' as c_char]), "a");
    assert_eq!(fixed_c_string(&[b'a' as c_char, b'b' as c_char]), "ab");
}

#[test]
fn tcp_loopback_moves_bytes_through_upstream_mooncake() {
    unsafe {
        libc::setenv(c"MC_FORCE_TCP".as_ptr(), c"1".as_ptr(), 1);
    }
    let engine = Arc::new(
        TransferEngine::new("P2PHANDSHAKE", "127.0.0.1:0", "127.0.0.1", 0, &[])
            .expect("create Mooncake TENT"),
    );
    let segment = engine.local_segment_name().expect("local segment");
    let mut memory = vec![0u8; 8192];
    memory[..4096].fill(0xa5);
    let base = NonNull::new(memory.as_mut_ptr()).expect("memory pointer");
    let registration = unsafe {
        engine
            .register_memory_owned(base, memory.len(), "cpu:0")
            .expect("register memory")
    };
    assert_eq!(registration.address(), base);
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
    let _ = engine.nic_load_stats().expect("query TENT NIC load stats");
    registration.unregister().expect("unregister memory");
    let registration = unsafe {
        engine
            .register_memory_owned(base, memory.len(), "cpu:0")
            .expect("register memory again")
    };
    drop(engine);
    drop(registration);
    unsafe {
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

#[test]
fn uncertain_transfer_states_drain_all_tasks_before_returning() {
    for mode in ["deadline", "status-error", "partial-submit"] {
        let rounds = std::cell::Cell::new(0);
        let cancellations = std::cell::Cell::new(0);
        let submitted = if mode == "partial-submit" {
            Err(MooncakeError::Operation {
                operation: "tent_submit",
                status: -1,
            })
        } else {
            Ok(())
        };
        let result = drain_batch(
            2,
            if mode == "deadline" {
                Duration::ZERO
            } else {
                Duration::from_secs(5)
            },
            submitted,
            |task| {
                if task == 0 {
                    return Ok(native::TransferStatus {
                        status: STATUS_COMPLETED,
                        transferred_bytes: 11,
                    });
                }
                rounds.set(rounds.get() + 1);
                if rounds.get() < 4 {
                    if mode == "status-error" {
                        return Err(MooncakeError::Operation {
                            operation: "getTransferStatus",
                            status: -1,
                        });
                    }
                    return Ok(native::TransferStatus {
                        status: STATUS_PENDING,
                        transferred_bytes: 0,
                    });
                }
                Ok(native::TransferStatus {
                    status: STATUS_COMPLETED,
                    transferred_bytes: 13,
                })
            },
            |_| {
                cancellations.set(cancellations.get() + 1);
                Ok(())
            },
            || rounds.get() == 4,
        );
        assert!(result.is_err(), "{mode}");
        assert_eq!(
            rounds.get(),
            4,
            "{mode} must not return before native free succeeds"
        );
        assert_eq!(cancellations.get(), 1, "{mode}");
    }

    for terminal in [
        STATUS_CANCELED,
        STATUS_FAILED,
        STATUS_INVALID,
        STATUS_TIMEOUT,
    ] {
        let freed = std::cell::Cell::new(false);
        let result = drain_batch(
            1,
            Duration::from_secs(5),
            Ok(()),
            |_| {
                Ok(native::TransferStatus {
                    status: terminal,
                    transferred_bytes: 0,
                })
            },
            |_| panic!("terminal native status needs no cancellation"),
            || {
                freed.set(true);
                true
            },
        );
        assert!(result.is_err(), "terminal={terminal}");
        assert!(freed.get(), "terminal={terminal}");
    }
}

#[test]
fn batch_drain_counts_completed_bytes_once_and_handles_rejected_submission() {
    let rounds = std::cell::Cell::new(0);
    let result = drain_batch(
        2,
        Duration::from_secs(5),
        Ok(()),
        |task| {
            if task == 1 {
                rounds.set(rounds.get() + 1);
            }
            Ok(native::TransferStatus {
                status: if task == 0 || rounds.get() == 3 {
                    STATUS_COMPLETED
                } else {
                    STATUS_PENDING
                },
                transferred_bytes: 11,
            })
        },
        |_| panic!("successful batch must not be cancelled"),
        || rounds.get() == 3,
    );
    assert_eq!(result.unwrap(), 22);
    let result = drain_batch(
        1,
        Duration::ZERO,
        Err(MooncakeError::Operation {
            operation: "tent_submit",
            status: -1,
        }),
        |_| {
            Ok(native::TransferStatus {
                status: STATUS_CANCELED,
                transferred_bytes: 0,
            })
        },
        |_| Ok(()),
        || true,
    );
    assert!(matches!(
        result,
        Err(MooncakeError::Operation {
            operation: "tent_submit",
            ..
        })
    ));
}
