use std::sync::Arc;
use std::time::{Duration, Instant};

use etcd_client::{Client, GetOptions};
use orbitkv_catalog::GlobalIndex;
use orbitkv_core::ResidencyInventory;
use orbitkv_state::{
    INVENTORY_BATCH_BYTES, InventoryRecord, ReplicaMedium, ReplicaMetadata, ReplicaRepresentation,
    StateKey,
};
use prost::Message;

use super::*;
use crate::cluster::Cluster;
use crate::cluster::tests::etcd::{Etcd, view};
use crate::cluster::tests::gate::TcpGate;

const SOURCES: usize = 16;
const KEYS_PER_SOURCE: usize = 2048;
const ACTIVE_KEYS_PER_SOURCE: usize = 1536;
const WINDOW_SHIFT: usize = 128;
const CHANGES_PER_SOURCE_ROUND: usize = WINDOW_SHIFT * 2;
const ROUNDS: usize = 60;
const ROUND_INTERVAL: Duration = Duration::from_secs(1);
const INDEX_BYTES: usize = 16 * 1024 * 1024;
const ETCD_QUOTA_BYTES: i64 = 1024 * 1024 * 1024;
const HARD_RUNTIME: Duration = Duration::from_secs(180);
const MAX_PUBLICATION_P95_MS: f64 = 100.0;
const MAX_PUBLICATION_MS: f64 = 1000.0;
const MAX_WATCH_P95_MS: f64 = 100.0;
const MAX_WATCH_MS: f64 = 1000.0;
const MAX_REBUILD_MS: f64 = 10_000.0;
const MIN_CHANGES_PER_SECOND: f64 = 1500.0;
const MAX_TEST_CPU_CORES: f64 = 8.0;
const MAX_ETCD_CPU_CORES: f64 = 4.0;
const MAX_HWM_GROWTH_KIB: u64 = 512 * 1024;
const MAX_ETCD_GROWTH_BYTES: i64 = 768 * 1024 * 1024;
const REBUILD_ROUNDS: [usize; 4] = [15, 30, 45, 60];

struct Source {
    publisher: Publisher,
    lease: i64,
    owner: orbitkv_state::CacheOwner,
    keys: Vec<StateKey>,
    sequences: Vec<u64>,
    present: Vec<bool>,
    sequence: u64,
    window_start: usize,
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "requires ETCD_BIN and ORBITKV_METADATA_ARTIFACT_DIR; 60-second real-etcd capacity qualification"]
async fn sustained_churn_bounds_batches_and_rebuilds_exactly() {
    let output = artifact_root();
    let contract = workload_contract();
    std::fs::write(
        output.join("workload-contract.json"),
        serde_json::to_vec_pretty(&contract).unwrap(),
    )
    .unwrap();

    let quota = ETCD_QUOTA_BYTES.to_string();
    let server = Etcd::start_with_args(1, &["--quota-backend-bytes", &quota]).await;
    let mut etcd_client = Client::connect(&server.endpoints, None).await.unwrap();
    let empty_status = etcd_client.status().await.unwrap();
    let test_before_load = process_sample(std::iter::once(std::process::id()));
    let etcd_before_load = process_sample(server.pids());

    let stable_view = view(60000);
    let stable_index = Arc::new(GlobalIndex::new(Arc::clone(&stable_view), INDEX_BYTES));
    let stable_cluster = Cluster::join(
        &server.endpoints,
        "sustained-capacity",
        "stable-reader",
        120,
        stable_view,
        Arc::new(ResidencyInventory::new(4096)),
        Arc::clone(&stable_index),
    )
    .await
    .unwrap();
    wait_available(&stable_index).await;

    let cluster = cluster_id(etcd_client.status().await.unwrap().header()).unwrap();
    let prefix = "/orbitkv/v2/sustained-capacity/";
    let mut sources = Vec::with_capacity(SOURCES);
    let mut publication_batch_records_peak = 0;
    let mut publication_batch_bytes_peak = 0;
    for source_id in 0..SOURCES {
        let source_view = view(60100 + source_id as u16);
        let granted_at = Instant::now();
        let grant = etcd_client.lease_grant(3600, None).await.unwrap();
        assert!(source_view.renew(granted_at, Duration::from_secs(3600)));
        let member = crate::cluster::registration::register(
            &mut etcd_client,
            prefix,
            &format!("source-{source_id}"),
            source_view.owner(),
            grant.id(),
            cluster,
        )
        .await
        .unwrap();
        let owner = member.owner.clone();
        let mut publisher = Publisher {
            client: etcd_client.clone(),
            prefix: prefix.into(),
            member,
            view: source_view,
            cluster,
            progress: Progress::default(),
            previous: None,
        };
        let keys = (0..KEYS_PER_SOURCE)
            .map(|key_id| capacity_key(source_id, key_id))
            .collect::<Vec<_>>();
        let mut sequences = vec![0; KEYS_PER_SOURCE];
        let mut present = vec![false; KEYS_PER_SOURCE];
        let mut sequence = 0;
        let mut records = Vec::with_capacity(ACTIVE_KEYS_PER_SOURCE);
        for key_id in 0..ACTIVE_KEYS_PER_SOURCE {
            sequence += 1;
            sequences[key_id] = sequence;
            present[key_id] = true;
            records.push(capacity_record(&keys[key_id], sequence, key_id, true));
        }
        for batch in records.chunks(MAX_BATCH_RECORDS) {
            observe_batch_bound(
                prefix,
                owner.incarnation,
                batch,
                &mut publication_batch_records_peak,
                &mut publication_batch_bytes_peak,
            );
            publisher
                .records(batch, batch.last().unwrap().sequence, false)
                .await
                .unwrap();
        }
        let revision = publisher.commit(Vec::new(), sequence, true).await.unwrap();
        wait_revision(&stable_index, revision).await;
        sources.push(Source {
            publisher,
            lease: grant.id(),
            owner,
            keys,
            sequences,
            present,
            sequence,
            window_start: 0,
        });
    }
    assert_exact(&stable_index, &sources);
    let loaded_status = etcd_client.status().await.unwrap();
    let test_before_churn = process_sample(std::iter::once(std::process::id()));
    let etcd_before_churn = process_sample(server.pids());

    let mut publication_ms = Vec::with_capacity(SOURCES * ROUNDS * 6);
    let mut watch_lag_ms = Vec::with_capacity(SOURCES * ROUNDS * 6);
    let mut snapshot_rebuild_ms = Vec::with_capacity(REBUILD_ROUNDS.len());
    let mut rebuild_convergence_ms = Vec::with_capacity(REBUILD_ROUNDS.len());
    let mut max_index_bytes = stable_index.bytes();
    let mut final_revision = 0;
    let workload_started = Instant::now();

    tokio::time::timeout(HARD_RUNTIME, async {
        for round in 1..=ROUNDS {
            tokio::time::sleep_until((workload_started + ROUND_INTERVAL * round as u32).into())
                .await;

            let rebuild = if REBUILD_ROUNDS.contains(&round) {
                Some(start_rebuild_reader(&server, round).await)
            } else {
                None
            };

            for source in &mut sources {
                let old_start = source.window_start;
                let mut changes = Vec::with_capacity(CHANGES_PER_SOURCE_ROUND);
                for offset in 0..WINDOW_SHIFT {
                    let key_id = (old_start + offset) % KEYS_PER_SOURCE;
                    assert!(source.present[key_id]);
                    source.sequence += 1;
                    source.sequences[key_id] = source.sequence;
                    source.present[key_id] = false;
                    changes.push(capacity_record(
                        &source.keys[key_id],
                        source.sequence,
                        key_id,
                        false,
                    ));
                }
                for offset in 0..WINDOW_SHIFT {
                    let key_id = (old_start + ACTIVE_KEYS_PER_SOURCE + offset) % KEYS_PER_SOURCE;
                    assert!(!source.present[key_id]);
                    source.sequence += 1;
                    source.sequences[key_id] = source.sequence;
                    source.present[key_id] = true;
                    changes.push(capacity_record(
                        &source.keys[key_id],
                        source.sequence,
                        key_id,
                        true,
                    ));
                }
                source.window_start = (old_start + WINDOW_SHIFT) % KEYS_PER_SOURCE;

                for batch in changes.chunks(MAX_BATCH_RECORDS) {
                    observe_batch_bound(
                        prefix,
                        source.owner.incarnation,
                        batch,
                        &mut publication_batch_records_peak,
                        &mut publication_batch_bytes_peak,
                    );
                    let sequence = batch.last().unwrap().sequence;
                    let started = Instant::now();
                    let revision = source
                        .publisher
                        .records(batch, sequence, true)
                        .await
                        .unwrap();
                    publication_ms.push(started.elapsed().as_secs_f64() * 1000.0);
                    let watch_started = Instant::now();
                    wait_revision(&stable_index, revision).await;
                    watch_lag_ms.push(watch_started.elapsed().as_secs_f64() * 1000.0);
                    final_revision = final_revision.max(revision);
                }
            }
            assert_exact(&stable_index, &sources);

            if let Some(active) = rebuild {
                active.gate.heal(Duration::ZERO);
                let first_complete = active.first_complete.await.unwrap();
                wait_revision(&active.index, final_revision).await;
                assert_exact(&active.index, &sources);
                snapshot_rebuild_ms.push(first_complete.as_secs_f64() * 1000.0);
                rebuild_convergence_ms.push(active.started.elapsed().as_secs_f64() * 1000.0);
                max_index_bytes = max_index_bytes.max(active.index.bytes());
                active.cluster.shutdown().await;
                active.gate.shutdown().await;
            }
            max_index_bytes = max_index_bytes.max(stable_index.bytes());
        }
    })
    .await
    .expect("sustained metadata workload exceeded its 180-second hard bound");

    assert_exact(&stable_index, &sources);
    let block_count = etcd_client
        .get(
            format!("{prefix}blocks/"),
            Some(GetOptions::new().with_prefix().with_count_only()),
        )
        .await
        .unwrap()
        .count();
    let expected_records = (SOURCES * ACTIVE_KEYS_PER_SOURCE) as i64;
    assert_eq!(
        block_count, expected_records,
        "unexpected raw etcd block set"
    );

    tokio::time::sleep(Duration::from_millis(500)).await;
    let final_status = etcd_client.status().await.unwrap();
    let test_after = process_sample(std::iter::once(std::process::id()));
    let etcd_after = process_sample(server.pids());
    let elapsed = workload_started.elapsed();
    let ticks_per_second = clock_ticks_per_second();
    let test_cpu_cores = cpu_cores(test_before_churn, test_after, ticks_per_second, elapsed);
    let etcd_cpu_cores = cpu_cores(etcd_before_churn, etcd_after, ticks_per_second, elapsed);
    let publication = DurationSummary::new(&publication_ms);
    let watch = DurationSummary::new(&watch_lag_ms);
    let rebuild = DurationSummary::new(&snapshot_rebuild_ms);
    let rebuild_convergence = DurationSummary::new(&rebuild_convergence_ms);
    let changes = SOURCES * CHANGES_PER_SOURCE_ROUND * ROUNDS;
    let changes_per_second = changes as f64 / elapsed.as_secs_f64();
    let test_hwm_growth = test_after.hwm_kib.saturating_sub(test_before_churn.hwm_kib);
    let etcd_hwm_growth = etcd_after.hwm_kib.saturating_sub(etcd_before_churn.hwm_kib);
    let etcd_growth = final_status.db_size() - empty_status.db_size();
    let accepted = changes_per_second >= MIN_CHANGES_PER_SECOND
        && publication.p95 <= MAX_PUBLICATION_P95_MS
        && publication.max <= MAX_PUBLICATION_MS
        && watch.p95 <= MAX_WATCH_P95_MS
        && watch.max <= MAX_WATCH_MS
        && rebuild.max <= MAX_REBUILD_MS
        && rebuild_convergence.max <= MAX_REBUILD_MS
        && publication_batch_records_peak <= MAX_BATCH_RECORDS
        && publication_batch_bytes_peak <= INVENTORY_BATCH_BYTES + MAX_BATCH_RECORDS * 128
        && max_index_bytes <= INDEX_BYTES
        && test_cpu_cores <= MAX_TEST_CPU_CORES
        && etcd_cpu_cores <= MAX_ETCD_CPU_CORES
        && test_hwm_growth <= MAX_HWM_GROWTH_KIB
        && etcd_hwm_growth <= MAX_HWM_GROWTH_KIB
        && etcd_growth <= MAX_ETCD_GROWTH_BYTES;

    let report = serde_json::json!({
        "contract": contract,
        "accepted": accepted,
        "workload_elapsed_seconds": elapsed.as_secs_f64(),
        "changes": changes,
        "changes_per_second": changes_per_second,
        "final_revision": final_revision,
        "raw_etcd_block_records": block_count,
        "publication_ms": publication,
        "watch_lag_ms": watch,
        "snapshot_rebuild_ms": rebuild,
        "rebuild_to_round_convergence_ms": rebuild_convergence,
        "publication_batch_records_peak": publication_batch_records_peak,
        "publication_batch_bytes_peak": publication_batch_bytes_peak,
        "index_bytes_peak": max_index_bytes,
        "index_bytes_final": stable_index.bytes(),
        "etcd_db_bytes_empty": empty_status.db_size(),
        "etcd_db_bytes_after_initial_load": loaded_status.db_size(),
        "etcd_db_bytes_final": final_status.db_size(),
        "etcd_db_growth_bytes": etcd_growth,
        "etcd_db_in_use_bytes_empty": empty_status.raft_used_db_size(),
        "etcd_db_in_use_bytes_final": final_status.raft_used_db_size(),
        "clock_ticks_per_second": ticks_per_second,
        "test_process_before_load": test_before_load,
        "test_process_before_churn": test_before_churn,
        "test_process_after": test_after,
        "test_cpu_average_cores": test_cpu_cores,
        "test_hwm_growth_kib": test_hwm_growth,
        "etcd_process_before_load": etcd_before_load,
        "etcd_process_before_churn": etcd_before_churn,
        "etcd_process_after": etcd_after,
        "etcd_cpu_average_cores": etcd_cpu_cores,
        "etcd_hwm_growth_kib": etcd_hwm_growth,
    });
    std::fs::write(
        output.join("sustained-capacity.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();

    for source in &sources {
        let _ = etcd_client.lease_revoke(source.lease).await;
    }
    stable_cluster.shutdown().await;
    assert!(
        accepted,
        "sustained metadata capacity thresholds failed: {report}"
    );
}

struct ActiveRebuild {
    gate: TcpGate,
    cluster: Cluster,
    index: Arc<GlobalIndex>,
    first_complete: tokio::task::JoinHandle<Duration>,
    started: Instant,
}

async fn start_rebuild_reader(server: &Etcd, round: usize) -> ActiveRebuild {
    let gate = TcpGate::start(&server.endpoints[0]).await;
    gate.set_downstream_delay(Duration::from_millis(2));
    let reader_view = view(62000 + round as u16);
    let index = Arc::new(GlobalIndex::new(Arc::clone(&reader_view), INDEX_BYTES));
    let started = Instant::now();
    let endpoints = [gate.endpoint.clone()];
    let cluster = Cluster::join(
        &endpoints,
        "sustained-capacity",
        &format!("rebuild-{round}"),
        120,
        reader_view,
        Arc::new(ResidencyInventory::new(4096)),
        Arc::clone(&index),
    )
    .await
    .unwrap();
    assert!(
        !index.status().available,
        "delayed reader completed before the concurrent eviction window"
    );
    let observed = Arc::clone(&index);
    let first_complete = tokio::spawn(async move {
        wait_available(&observed).await;
        started.elapsed()
    });
    ActiveRebuild {
        gate,
        cluster,
        index,
        first_complete,
        started,
    }
}

fn capacity_key(source: usize, key: usize) -> StateKey {
    let mut hash = Vec::with_capacity(16);
    hash.extend_from_slice(&(source as u64).to_le_bytes());
    hash.extend_from_slice(&(key as u64).to_le_bytes());
    StateKey::new(format!("s2-6-source-{source}"), hash)
}

fn capacity_record(key: &StateKey, sequence: u64, key_id: usize, present: bool) -> InventoryRecord {
    InventoryRecord {
        key: key.clone(),
        sequence,
        present,
        metadata: Some(metadata(key_id)),
    }
}

fn medium(key: usize) -> ReplicaMedium {
    if key.is_multiple_of(2) {
        ReplicaMedium::Dram
    } else {
        ReplicaMedium::Ssd
    }
}

fn metadata(key: usize) -> ReplicaMetadata {
    ReplicaMetadata {
        medium: medium(key),
        representation: ReplicaRepresentation::Raw,
        stored_bytes: Some(4096),
    }
}

fn observe_batch_bound(
    prefix: &str,
    owner: uuid::Uuid,
    records: &[InventoryRecord],
    record_peak: &mut usize,
    byte_peak: &mut usize,
) {
    let bytes = records
        .iter()
        .map(|record| {
            record_key(prefix, owner, record).unwrap().len()
                + orbitkv_proto::proto::engine::InventoryRecord::from(record.clone()).encoded_len()
        })
        .sum();
    *record_peak = (*record_peak).max(records.len());
    *byte_peak = (*byte_peak).max(bytes);
    assert!(records.len() <= MAX_BATCH_RECORDS);
    assert!(bytes <= INVENTORY_BATCH_BYTES + MAX_BATCH_RECORDS * 128);
}

fn assert_exact(index: &GlobalIndex, sources: &[Source]) {
    assert!(
        index.status().available,
        "exact check requires complete coverage"
    );
    for source in sources {
        let rows = index.lookup(&source.keys);
        for (key_id, row) in rows.iter().enumerate() {
            assert_eq!(row.key, source.keys[key_id]);
            if !source.present[key_id] {
                assert!(row.replicas.is_empty(), "deleted key remained visible");
                continue;
            }
            assert_eq!(
                row.replicas.len(),
                1,
                "retained key has the wrong owner count"
            );
            let replica = &row.replicas[0];
            assert_eq!(replica.owner, source.owner);
            assert_eq!(replica.sequence, source.sequences[key_id]);
            assert_eq!(replica.metadata, metadata(key_id));
        }
    }
}

async fn wait_available(index: &GlobalIndex) {
    let deadline = Instant::now() + Duration::from_secs(20);
    while !index.status().available {
        assert!(
            Instant::now() < deadline,
            "complete index did not become available"
        );
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
}

async fn wait_revision(index: &GlobalIndex, revision: i64) {
    let deadline = Instant::now() + Duration::from_secs(20);
    while !index.revision().is_some_and(|applied| applied >= revision) {
        assert!(
            Instant::now() < deadline,
            "metadata revision did not converge"
        );
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
}

fn artifact_root() -> std::path::PathBuf {
    let output = std::path::PathBuf::from(
        std::env::var_os("ORBITKV_METADATA_ARTIFACT_DIR")
            .expect("capacity evidence requires ORBITKV_METADATA_ARTIFACT_DIR"),
    );
    std::fs::create_dir_all(&output).unwrap();
    output.canonicalize().unwrap()
}

fn workload_contract() -> serde_json::Value {
    serde_json::json!({
        "sources": SOURCES,
        "keys_per_source": KEYS_PER_SOURCE,
        "active_keys_per_source": ACTIVE_KEYS_PER_SOURCE,
        "window_shift_per_round": WINDOW_SHIFT,
        "changes_per_source_round": CHANGES_PER_SOURCE_ROUND,
        "rounds": ROUNDS,
        "scheduled_duration_seconds": ROUND_INTERVAL.as_secs() * ROUNDS as u64,
        "rebuild_rounds": REBUILD_ROUNDS,
        "index_bytes_per_reader": INDEX_BYTES,
        "publication_batch_records": MAX_BATCH_RECORDS,
        "publication_batch_bytes": INVENTORY_BATCH_BYTES + MAX_BATCH_RECORDS * 128,
        "etcd_quota_bytes": ETCD_QUOTA_BYTES,
        "hard_runtime_seconds": HARD_RUNTIME.as_secs(),
        "thresholds": {
            "minimum_changes_per_second": MIN_CHANGES_PER_SECOND,
            "publication_p95_ms": MAX_PUBLICATION_P95_MS,
            "publication_max_ms": MAX_PUBLICATION_MS,
            "watch_p95_ms": MAX_WATCH_P95_MS,
            "watch_max_ms": MAX_WATCH_MS,
            "rebuild_max_ms": MAX_REBUILD_MS,
            "test_cpu_average_cores": MAX_TEST_CPU_CORES,
            "etcd_cpu_average_cores": MAX_ETCD_CPU_CORES,
            "process_hwm_growth_kib": MAX_HWM_GROWTH_KIB,
            "etcd_db_growth_bytes": MAX_ETCD_GROWTH_BYTES,
        },
    })
}

#[derive(serde::Serialize)]
struct DurationSummary {
    samples: usize,
    p50: f64,
    p95: f64,
    max: f64,
}

impl DurationSummary {
    fn new(values: &[f64]) -> Self {
        assert!(!values.is_empty());
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        let percentile = |value: f64| sorted[((sorted.len() - 1) as f64 * value).round() as usize];
        Self {
            samples: sorted.len(),
            p50: percentile(0.50),
            p95: percentile(0.95),
            max: *sorted.last().unwrap(),
        }
    }
}

#[derive(Clone, Copy, serde::Serialize)]
struct ProcessSample {
    processes: usize,
    cpu_ticks: u64,
    rss_kib: u64,
    hwm_kib: u64,
}

fn process_sample(pids: impl IntoIterator<Item = u32>) -> ProcessSample {
    let mut sample = ProcessSample {
        processes: 0,
        cpu_ticks: 0,
        rss_kib: 0,
        hwm_kib: 0,
    };
    for pid in pids {
        let Ok(stat) = std::fs::read_to_string(format!("/proc/{pid}/stat")) else {
            continue;
        };
        let Some((_, fields)) = stat.rsplit_once(") ") else {
            continue;
        };
        let fields = fields.split_ascii_whitespace().collect::<Vec<_>>();
        let Ok(utime) = fields[11].parse::<u64>() else {
            continue;
        };
        let Ok(stime) = fields[12].parse::<u64>() else {
            continue;
        };
        let status = std::fs::read_to_string(format!("/proc/{pid}/status")).unwrap_or_default();
        sample.processes += 1;
        sample.cpu_ticks += utime + stime;
        sample.rss_kib += status_kib(&status, "VmRSS:");
        sample.hwm_kib += status_kib(&status, "VmHWM:");
    }
    sample
}

fn status_kib(status: &str, name: &str) -> u64 {
    status
        .lines()
        .find_map(|line| line.strip_prefix(name))
        .and_then(|value| value.split_ascii_whitespace().next())
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}

fn cpu_cores(before: ProcessSample, after: ProcessSample, ticks: u64, elapsed: Duration) -> f64 {
    after.cpu_ticks.saturating_sub(before.cpu_ticks) as f64 / ticks as f64 / elapsed.as_secs_f64()
}

fn clock_ticks_per_second() -> u64 {
    let output = std::process::Command::new("getconf")
        .arg("CLK_TCK")
        .output()
        .expect("run getconf CLK_TCK");
    assert!(output.status.success(), "getconf CLK_TCK failed");
    std::str::from_utf8(&output.stdout)
        .unwrap()
        .trim()
        .parse()
        .unwrap()
}
