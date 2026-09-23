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

#[test]
fn uncertain_transfer_states_drain_all_tasks_before_returning() {
    for mode in [
        "deadline",
        "status-error",
        "native-timeout",
        "partial-submit",
    ] {
        let rounds = std::cell::Cell::new(0);
        let submitted = if mode == "partial-submit" {
            Err(MooncakeError::Operation {
                operation: "submitTransfer",
                status: -1,
            })
        } else {
            Ok(())
        };
        let result = drain_batch(
            2,
            Duration::ZERO,
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
                        status: if mode == "native-timeout" {
                            6
                        } else {
                            STATUS_PENDING
                        },
                        transferred_bytes: 0,
                    });
                }
                Ok(native::TransferStatus {
                    status: STATUS_COMPLETED,
                    transferred_bytes: 13,
                })
            },
            || rounds.get() == 4,
        );
        assert!(result.is_err(), "{mode}");
        assert_eq!(
            rounds.get(),
            4,
            "{mode} must not return before native free succeeds"
        );
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
        || rounds.get() == 3,
    );
    assert_eq!(result.unwrap(), 22);
    let result = drain_batch(
        1,
        Duration::ZERO,
        Err(MooncakeError::Operation {
            operation: "submitTransfer",
            status: -1,
        }),
        |_| {
            Err(MooncakeError::Operation {
                operation: "getTransferStatus",
                status: -2,
            })
        },
        || true,
    );
    assert!(matches!(
        result,
        Err(MooncakeError::Operation {
            operation: "submitTransfer",
            ..
        })
    ));
}
