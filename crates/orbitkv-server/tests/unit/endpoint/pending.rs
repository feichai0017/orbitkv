use super::*;
use orbitkv_core::StorageConfig;

fn request(operation_id: u64, revision: u64) -> QueryBundleRequest {
    QueryBundleRequest {
        ticket: QueryTicket {
            operation_id,
            revision,
        },
        instance_id: "model".into(),
        request_id: "same-request".into(),
        block_hashes: vec![vec![revision as u8]],
        group_id: 0,
        wait_for_full_prefix: false,
        warmup: false,
        discover: false,
        materialize: false,
    }
}

fn engine() -> Arc<OrbitKVEngine> {
    Arc::new(OrbitKVEngine::new_with_config(1 << 20, false, StorageConfig::default()).unwrap())
}

fn tracker() -> Arc<Mutex<MultiWindowHllTracker>> {
    Arc::new(Mutex::new(MultiWindowHllTracker::new(
        vec![("test".into(), Duration::from_secs(60))],
        4,
    )))
}

#[test]
fn revisions_and_session_teardown_drain_old_work_without_restarting_it() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let engine = engine();
    let hll = tracker();
    let mut queries = PendingQueries::default();
    let capacity = Arc::clone(&queries.capacity);
    let held = Arc::clone(&capacity)
        .try_acquire_many_owned((MAX_ACTIVE_QUERIES - 2) as u32)
        .unwrap();
    let mut completions = Vec::new();
    for token in [1, 2] {
        queries.sessions.entry(token).or_default().last_operation = 1;
        queries.insert(token, request(1, 1));
        let (complete, wait) = oneshot::channel();
        completions.push(complete);
        let result_engine = Arc::clone(&engine);
        let permit = Arc::clone(&capacity).try_acquire_owned().unwrap();
        assert!(
            queries
                .start_query(
                    (token, 1),
                    async move {
                        wait.await.unwrap();
                        QueryReply {
                            outcome: Ok(QueryOutcome::Ready {
                                num_hit_blocks: 0,
                                lease: vec![],
                                hit_positions: vec![],
                            }),
                            engine: result_engine,
                            delivered: false,
                            _permits: vec![permit],
                        }
                    },
                    runtime.handle()
                )
                .is_none()
        );
    }
    let old = request(1, 1).ticket;
    let new = request(1, 2).ticket;
    assert!(
        queries
            .execute(
                1,
                QueryCommand::Submit(request(1, 2)),
                &engine,
                runtime.handle(),
                &hll
            )
            .unwrap()
            .is_none()
    );
    queries.cancel(1, old, &engine);
    assert_eq!(queries.pending[&(1, 1)].request.ticket, new);
    assert_eq!(queries.pending[&(2, 1)].request.ticket, old);
    for (session, ticket) in [(1, old), (9, new)] {
        assert!(
            queries
                .execute(
                    session,
                    QueryCommand::Poll(ticket),
                    &engine,
                    runtime.handle(),
                    &hll
                )
                .is_err()
        );
    }
    assert_eq!(
        capacity.available_permits(),
        0,
        "superseding work does not finish its I/O"
    );
    queries.cancel(1, new, &engine);
    assert!(
        queries
            .execute(
                1,
                QueryCommand::Submit(request(1, 2)),
                &engine,
                runtime.handle(),
                &hll
            )
            .is_err()
    );
    queries.retain_sessions(&engine, |_| false);
    for completion in completions {
        completion.send(()).unwrap();
    }
    runtime.block_on(async {
        let _done = tokio::time::timeout(Duration::from_secs(2), capacity.acquire_many(2))
            .await
            .unwrap()
            .unwrap();
    });
    assert_eq!(capacity.available_permits(), 2);
    assert!(queries.pending.is_empty());
    drop(held);
}

#[test]
fn expiration_and_capacity_pressure_never_turn_a_poll_into_submission() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let engine = engine();
    let hll = tracker();
    let mut queries = PendingQueries::default();
    queries.sessions.entry(1).or_default().last_operation = MAX_PENDING_PER_SESSION as u64;
    for id in 1..=MAX_PENDING_PER_SESSION as u64 {
        queries.insert(1, request(id, 1));
    }
    let next = request(MAX_PENDING_PER_SESSION as u64 + 1, 1);
    let reply = queries
        .execute(
            1,
            QueryCommand::Submit(next.clone()),
            &engine,
            runtime.handle(),
            &hll,
        )
        .unwrap()
        .unwrap();
    assert!(matches!(reply.outcome, Ok(QueryOutcome::Busy)));
    assert!(
        queries
            .execute(
                1,
                QueryCommand::Poll(next.ticket),
                &engine,
                runtime.handle(),
                &hll
            )
            .is_err()
    );
    queries.pending.get_mut(&(1, 1)).unwrap().expires = Instant::now();
    queries.retain_sessions(&engine, |_| true);
    let error = queries
        .execute(
            1,
            QueryCommand::Poll(request(1, 1).ticket),
            &engine,
            runtime.handle(),
            &hll,
        )
        .err()
        .unwrap();
    assert!(error.to_string().contains("timed out"));
    assert!(
        queries
            .execute(
                1,
                QueryCommand::Submit(request(1, 2)),
                &engine,
                runtime.handle(),
                &hll
            )
            .is_err()
    );
}

#[test]
fn warmups_skip_full_capacity_and_reap_unpolled_completions() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let engine = engine();
    let hll = tracker();
    let mut queries = PendingQueries::default();
    let held = Arc::clone(&queries.warming)
        .try_acquire_many_owned(MAX_ACTIVE_WARMUPS as u32)
        .unwrap();
    let mut warmup = request(1, 1);
    warmup.warmup = true;
    let reply = queries
        .execute(
            1,
            QueryCommand::Submit(warmup.clone()),
            &engine,
            runtime.handle(),
            &hll,
        )
        .unwrap()
        .unwrap();
    assert!(matches!(reply.outcome, Ok(QueryOutcome::Busy)));
    drop(reply);
    assert!(queries.pending.is_empty());
    assert_eq!(queries.capacity.available_permits(), MAX_ACTIVE_QUERIES);
    drop(held);

    let permit = Arc::clone(&queries.warming).try_acquire_owned().unwrap();
    warmup.ticket.operation_id = 2;
    queries.insert(1, warmup);
    let (sender, receiver) = oneshot::channel();
    queries.pending.get_mut(&(1, 2)).unwrap().receiver = Some(receiver);
    let _sent = sender.send(QueryReply {
        outcome: Ok(QueryOutcome::Ready {
            num_hit_blocks: 0,
            lease: vec![],
            hit_positions: vec![],
        }),
        engine: Arc::clone(&engine),
        delivered: false,
        _permits: vec![permit],
    });
    queries.retain_sessions(&engine, |_| true);
    assert!(queries.pending.is_empty());
    assert_eq!(queries.warming.available_permits(), MAX_ACTIVE_WARMUPS);
}
