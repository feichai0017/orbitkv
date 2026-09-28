//! Query lease protocol tests.
//!
//! Verifies the scheduler->worker contract: query_prefetch returns an opaque
//! lease that owns ready blocks, and load consumes one lease share per worker.

mod common;

use common::*;

/// vLLM worker must not load after the scheduler releases the query lease.
#[tokio::test]
async fn load_requires_query_prefetch() {
    let env = TestEnvBuilder::new("test-load-needs-query", "test-ns")
        .layer("layer_0", 4, 1024)
        .build();
    let hashes = env.hashes(0);

    env.save_and_wait(&hashes).await;

    // Query and immediately release — no lease held.
    assert_eq!(env.count_hits_then_release(&hashes).await, 4);

    let released = env.assert_all_hit_lease(&hashes).await;
    env.release(&released);
    env.expect_load_error(released, hashes.len(), "query lease is unknown or expired");
}

/// One scheduler query lease is consumed once per registered world-size worker.
#[tokio::test]
async fn query_then_load_consumes_reservation_budget() {
    if !has_cuda_devices(2) {
        eprintln!("skipping query_then_load_consumes_reservation_budget: needs >= 2 CUDA devices");
        return;
    }

    let env = TestEnvBuilder::new("test-query-lease", "test-ns")
        .layer("layer_0", 4, 1024)
        .world_size(2)
        .build();
    let hashes = env.hashes(22);

    env.save_and_wait(&hashes).await;
    let lease = env.assert_all_hit_lease(&hashes).await;

    env.data().zero_gpu();
    env.load_to_gpu(lease, hashes.len()).await;
    env.data().assert_gpu_matches_expected();

    env.data().zero_gpu();
    env.load_to_gpu(lease, hashes.len()).await;
    env.data().assert_gpu_matches_expected();

    let block_ids: Vec<Option<usize>> = (0..hashes.len()).map(Some).collect();
    let layer_names: Vec<&str> = env.layers.iter().map(|l| l.name.as_str()).collect();
    let layer_groups = vec![layer_names];
    let err = env
        .engine
        .restore(
            &env.instance_id,
            0,
            0,
            &layer_groups,
            &[(lease, vec![block_ids])],
        )
        .expect_err("third load should fail");
    assert!(
        err.to_string()
            .contains("query lease is unknown or expired")
    );
}

/// Preparation rejects an entire malformed batch without consuming earlier
/// leases or touching destination pages. A corrected batch can reuse both leases.
#[tokio::test]
async fn rejected_restore_batch_preserves_leases_and_gpu_pages() {
    use orbitkv_core::QueryLeaseId;

    let env = TestEnvBuilder::new("restore-batch-validation", "restore-batch-ns")
        .layer("layer_0", 4, 1024)
        .build();
    let hashes = env.hashes(77);
    env.save_and_wait(&hashes).await;
    env.data().zero_gpu();
    let first = env.assert_all_hit_lease(&hashes[..2]).await;
    let second = env.assert_all_hit_lease(&hashes[2..]).await;
    let first_targets = vec![vec![Some(0), Some(1)]];
    let second_targets = vec![vec![Some(2), Some(3)]];
    for (name, rank, groups, loads, expected) in [
        (
            "unknown later lease",
            0,
            vec![vec!["layer_0"]],
            vec![
                (first, first_targets.clone()),
                (QueryLeaseId::fresh(), second_targets.clone()),
            ],
            "query lease is unknown or expired",
        ),
        (
            "later source shape",
            0,
            vec![vec!["layer_0"]],
            vec![
                (first, first_targets.clone()),
                (second, vec![vec![Some(2)]]),
            ],
            "destination block count",
        ),
        (
            "later group shape",
            0,
            vec![vec!["layer_0"]],
            vec![(first, first_targets.clone()), (second, vec![])],
            "load group count",
        ),
        (
            "later destination bounds",
            0,
            vec![vec!["layer_0"]],
            vec![
                (first, first_targets.clone()),
                (second, vec![vec![Some(2), Some(4)]]),
            ],
            "out of range",
        ),
        (
            "duplicate lease",
            0,
            vec![vec!["layer_0"]],
            vec![
                (first, first_targets.clone()),
                (first, second_targets.clone()),
            ],
            "duplicate query lease",
        ),
        (
            "duplicate layer",
            0,
            vec![vec!["layer_0", "layer_0"]],
            vec![
                (first, first_targets.clone()),
                (second, second_targets.clone()),
            ],
            "layer names must be unique",
        ),
        (
            "wrong device rank",
            1,
            vec![vec!["layer_0"]],
            vec![
                (first, first_targets.clone()),
                (second, second_targets.clone()),
            ],
            "represents tp_rank",
        ),
    ] {
        let error = env
            .engine
            .restore(&env.instance_id, rank, 0, &groups, &loads)
            .expect_err(name);
        assert!(error.to_string().contains(expected), "{name}: {error}");
        env.data().assert_gpu_matches(&[0; 4096]);
    }
    let completion = env
        .engine
        .restore(
            &env.instance_id,
            0,
            0,
            &[vec!["layer_0"]],
            &[(first, first_targets), (second, second_targets)],
        )
        .expect("corrected batch uses both original leases");
    env.restore_outcome(completion)
        .await
        .result
        .expect("restore failed");
    env.data().assert_gpu_matches_expected();
}
