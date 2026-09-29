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
    let notification_generation = engine
        .open_notification_scope("loopback")
        .expect("open notification scope");
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
    assert_eq!(
        engine
            .wait_for_notification(
                "loopback",
                notification_generation,
                &[("done".to_string(), 1)],
                Duration::from_secs(5),
            )
            .expect("wait for notification"),
        Some("done".to_string())
    );
    engine.close_notification_scope("loopback", notification_generation);
    let cancelled_generation = engine
        .open_notification_scope("cancelled")
        .expect("open cancelled notification scope");
    let waiter_engine = Arc::clone(&engine);
    let waiter = std::thread::spawn(move || {
        waiter_engine.wait_for_notification(
            "cancelled",
            cancelled_generation,
            &[("done".to_string(), 1)],
            Duration::from_secs(5),
        )
    });
    std::thread::sleep(Duration::from_millis(10));
    engine.close_notification_scope("cancelled", cancelled_generation);
    assert_eq!(waiter.join().expect("join waiter").unwrap(), None);
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
fn timeout_cancellation_waits_for_all_terminal_tasks_and_batch_reclamation() {
    for native_timeout in [false, true] {
        let polls = [std::cell::Cell::new(0), std::cell::Cell::new(0)];
        let cancelled = [std::cell::Cell::new(false), std::cell::Cell::new(false)];
        let terminal = [std::cell::Cell::new(false), std::cell::Cell::new(false)];
        let frees = std::cell::Cell::new(0);
        let result = drain_batch(
            2,
            if native_timeout {
                Duration::MAX
            } else {
                Duration::ZERO
            },
            Ok(()),
            |task| {
                polls[task].set(polls[task].get() + 1);
                let status = if native_timeout && task == 0 {
                    STATUS_TIMEOUT
                } else if cancelled[task].get() && polls[task].get() >= 4 {
                    STATUS_CANCELED
                } else {
                    STATUS_PENDING
                };
                terminal[task].set(status != STATUS_PENDING);
                Ok(native::TransferStatus {
                    status,
                    transferred_bytes: 0,
                })
            },
            |task| {
                assert!(!cancelled[task].replace(true), "cancel a task at most once");
                Ok(())
            },
            || {
                assert!(terminal.iter().all(std::cell::Cell::get));
                frees.set(frees.get() + 1);
                frees.get() == 2
            },
        );
        assert!(matches!(result, Err(MooncakeError::Timeout)));
        assert_eq!(polls[0].get(), if native_timeout { 1 } else { 4 });
        assert_eq!(polls[1].get(), 4);
        assert_eq!(cancelled[0].get(), !native_timeout);
        assert!(cancelled[1].get());
        assert_eq!(
            frees.get(),
            2,
            "terminal status alone cannot reclaim a batch"
        );
    }
}

#[test]
fn timeout_cancellation_preserves_submission_polling_and_cancellation_errors() {
    for operation in ["tent_submit", "getTransferStatus", "tent_cancel_task"] {
        let error = || MooncakeError::Operation {
            operation,
            status: -23,
        };
        let polls = std::cell::Cell::new(0);
        let cancellations = std::cell::Cell::new(0);
        let freed = std::cell::Cell::new(false);
        let result = drain_batch(
            1,
            Duration::ZERO,
            if operation == "tent_submit" {
                Err(error())
            } else {
                Ok(())
            },
            |_| {
                polls.set(polls.get() + 1);
                if polls.get() == 1 && operation == "getTransferStatus" {
                    return Err(error());
                }
                Ok(native::TransferStatus {
                    status: if polls.get() == 1 {
                        STATUS_PENDING
                    } else {
                        STATUS_CANCELED
                    },
                    transferred_bytes: 0,
                })
            },
            |_| {
                cancellations.set(cancellations.get() + 1);
                if operation == "tent_cancel_task" {
                    Err(error())
                } else {
                    Ok(())
                }
            },
            || {
                assert_eq!(polls.get(), 2, "retain the owner until cancellation drains");
                freed.set(true);
                true
            },
        );
        assert!(matches!(
            result,
            Err(MooncakeError::Operation { operation: cause, status: -23 }) if cause == operation
        ));
        assert_eq!(cancellations.get(), 1);
        assert!(freed.get());
    }
}

#[test]
fn unsolicited_cancellation_remains_failure_even_when_another_timeout_was_observed() {
    for (native_timeout, timeout) in [
        (false, Duration::MAX),
        (false, Duration::ZERO),
        (true, Duration::MAX),
    ] {
        let canceled_task = usize::from(native_timeout);
        let result = drain_batch(
            canceled_task + 1,
            timeout,
            Ok(()),
            |task| {
                Ok(native::TransferStatus {
                    status: if native_timeout && task == 0 {
                        STATUS_TIMEOUT
                    } else {
                        STATUS_CANCELED
                    },
                    transferred_bytes: 0,
                })
            },
            |_| panic!("an unsolicited terminal state must not request cancellation"),
            || true,
        );
        assert!(matches!(
            result,
            Err(MooncakeError::TransferFailed { task, state: STATUS_CANCELED }) if task == canceled_task
        ));
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
