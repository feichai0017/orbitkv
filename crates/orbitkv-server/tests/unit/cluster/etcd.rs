use super::*;
use std::process::{Child, Command, Stdio};

pub(crate) struct Etcd {
    processes: Vec<Child>,
    pub(crate) endpoints: Vec<String>,
    pub(crate) directory: std::path::PathBuf,
    _temporary: Option<tempfile::TempDir>,
}

impl Etcd {
    pub(crate) async fn start(count: usize) -> Self {
        Self::start_with_args(count, &[]).await
    }

    pub(crate) async fn start_with_args(count: usize, args: &[&str]) -> Self {
        static LOGGING: std::sync::Once = std::sync::Once::new();
        LOGGING.call_once(|| orbitkv_common::logging::init_stderr("warn"));
        let binary = std::env::var("ETCD_BIN").expect("set ETCD_BIN to an etcd 3.x binary");
        let (directory, temporary) = match std::env::var_os("ORBITKV_METADATA_ARTIFACT_DIR") {
            Some(root) => {
                std::fs::create_dir_all(&root).unwrap();
                let root = std::fs::canonicalize(root).unwrap();
                if let Some(checkout) = checkout_root() {
                    assert!(
                        !root.starts_with(checkout),
                        "metadata evidence must be outside checkout"
                    );
                }
                (
                    tempfile::Builder::new()
                        .prefix("etcd-")
                        .tempdir_in(root)
                        .unwrap()
                        .keep(),
                    None,
                )
            }
            None => {
                let temporary = tempfile::tempdir().unwrap();
                (temporary.path().to_path_buf(), Some(temporary))
            }
        };
        let mut reservations = Vec::new();
        let mut endpoints = Vec::new();
        let mut peers = Vec::new();
        for _ in 0..count {
            for urls in [&mut endpoints, &mut peers] {
                let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
                urls.push(format!("http://{}", listener.local_addr().unwrap()));
                reservations.push(listener);
            }
        }
        let initial = peers
            .iter()
            .enumerate()
            .map(|(i, p)| format!("test{i}={p}"))
            .collect::<Vec<_>>()
            .join(",");
        drop(reservations);
        let mut processes = Vec::new();
        for i in 0..count {
            let log = std::fs::File::create(directory.join(format!("etcd-{i}.log"))).unwrap();
            processes.push(
                Command::new(&binary)
                    .args(["--name", &format!("test{i}"), "--data-dir"])
                    .arg(directory.join(format!("data-{i}")))
                    .args([
                        "--listen-client-urls",
                        &endpoints[i],
                        "--advertise-client-urls",
                        &endpoints[i],
                        "--listen-peer-urls",
                        &peers[i],
                        "--initial-advertise-peer-urls",
                        &peers[i],
                        "--initial-cluster",
                        &initial,
                        "--heartbeat-interval",
                        "50",
                        "--election-timeout",
                        "500",
                        "--log-level",
                        "error",
                    ])
                    .args(args)
                    .stdout(Stdio::null())
                    .stderr(log)
                    .spawn()
                    .unwrap(),
            );
        }
        let mut server = Self {
            processes,
            endpoints,
            directory,
            _temporary: temporary,
        };
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            assert!(
                server
                    .processes
                    .iter_mut()
                    .all(|p| p.try_wait().unwrap().is_none()),
                "etcd exited"
            );
            if let Ok(mut client) = rpc(Client::connect(&server.endpoints, None)).await
                && rpc(client.get("startup-probe", None)).await.is_ok()
            {
                return server;
            }
            assert!(Instant::now() < deadline, "etcd did not elect a leader");
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    pub(crate) async fn leader(&self) -> usize {
        for (i, endpoint) in self.endpoints.iter().enumerate() {
            if let Ok(mut client) = Client::connect([endpoint], None).await
                && let Ok(status) = rpc(client.status()).await
                && status.header().unwrap().member_id() == status.leader()
            {
                return i;
            }
        }
        panic!("no leader");
    }

    pub(crate) fn kill(&mut self, node: usize) {
        let _ = self.processes[node].kill();
        let _ = self.processes[node].wait();
    }
}

fn checkout_root() -> Option<std::path::PathBuf> {
    let compiled = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    if let Ok(root) = compiled.canonicalize() {
        return Some(root);
    }
    let output = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let root = std::str::from_utf8(&output.stdout).ok()?.trim();
    std::fs::canonicalize(root).ok()
}

impl Drop for Etcd {
    fn drop(&mut self) {
        for process in &mut self.processes {
            let _ = process.kill();
            let _ = process.wait();
        }
    }
}

pub(crate) fn view(port: u16) -> Arc<MembershipView> {
    Arc::new(MembershipView::new(CacheOwner {
        endpoint: format!("127.0.0.1:{port}"),
        incarnation: uuid::Uuid::new_v4(),
    }))
}

pub(crate) async fn wait_for(condition: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(20);
    while !condition() {
        assert!(Instant::now() < deadline, "metadata did not converge");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

pub(crate) async fn join(
    server: &Etcd,
    cluster: &str,
    node: &str,
    view: Arc<MembershipView>,
    ttl: i64,
) -> (Cluster, Arc<GlobalIndex>, Arc<ResidencyInventory>) {
    let index = Arc::new(GlobalIndex::new(view.clone(), 1 << 20));
    let inventory = Arc::new(ResidencyInventory::new(16 << 10));
    let member = Cluster::join(
        &server.endpoints,
        cluster,
        node,
        ttl,
        view,
        inventory.clone(),
        index.clone(),
    )
    .await
    .unwrap_or_else(|error| panic!("join {node}: {error}"));
    (member, index, inventory)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires ETCD_BIN; starts a real isolated etcd process"]
async fn snapshots_compaction_and_restarts_preserve_member_incarnations() {
    let server = Etcd::start(1).await;
    let a = view(51001);
    let b = view(51002);
    let (member_a, index_a, inventory_a) = join(&server, "recovery", "a", a.clone(), 60).await;
    let (member_b, index_b, _) = join(&server, "recovery", "b", b.clone(), 60).await;
    wait_for(|| {
        a.permits(b.owner())
            && b.permits(a.owner())
            && index_a.revision().is_some()
            && index_b.revision().is_some()
    })
    .await;
    let revision = inventory_a.flush().await.unwrap();
    wait_for(|| index_b.revision().is_some_and(|r| r >= revision)).await;
    assert!(
        Cluster::join(
            &server.endpoints,
            "recovery",
            "a",
            60,
            view(51003),
            Arc::new(ResidencyInventory::new(4096)),
            Arc::new(GlobalIndex::new(a.clone(), 4096))
        )
        .await
        .is_err()
    );
    let mut client = Client::connect(&server.endpoints, None).await.unwrap();
    let cluster = cluster_id(client.status().await.unwrap().header()).unwrap();
    let scratch = GlobalIndex::new(a.clone(), 1 << 20);
    let own = client
        .get("/orbitkv/v2/recovery/members/a", None)
        .await
        .unwrap();
    let mut registration: Member = serde_json::from_slice(own.kvs()[0].value()).unwrap();
    registration.lease = own.kvs()[0].lease();
    let (mut members, mut old) = watch::bootstrap(
        &mut client,
        "/orbitkv/v2/recovery/",
        cluster,
        &registration,
        &a,
        &scratch,
    )
    .await
    .unwrap();
    client
        .put("outside-cluster-compaction", "0", None)
        .await
        .unwrap();
    let later = client
        .put("outside-cluster-compaction", "1", None)
        .await
        .unwrap()
        .header()
        .unwrap()
        .revision();
    client.compact(later, None).await.unwrap();
    let result = tokio::time::timeout(
        Duration::from_secs(10),
        watch::follow(
            &mut client,
            "/orbitkv/v2/recovery/",
            cluster,
            &registration,
            &a,
            &scratch,
            &mut members,
            &mut old,
        ),
    )
    .await
    .expect("compacted Watch did not close");
    assert!(matches!(result, Err(watch::FollowError::Rebuild(_))));
    scratch.reset();
    watch::bootstrap(
        &mut client,
        "/orbitkv/v2/recovery/",
        cluster,
        &registration,
        &a,
        &scratch,
    )
    .await
    .unwrap();
    member_b.shutdown().await;
    wait_for(|| !a.permits(b.owner())).await;
    let replacement = view(51002);
    let (new_b, _, _) = join(&server, "recovery", "b", replacement.clone(), 60).await;
    wait_for(|| a.permits(replacement.owner())).await;
    assert!(!a.permits(b.owner()));
    assert_eq!(
        client
            .get("/orbitkv/v2/recovery/epochs/b", None)
            .await
            .unwrap()
            .kvs()[0]
            .value(),
        b"2"
    );
    new_b.shutdown().await;
    member_a.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires ETCD_BIN; starts a real three-member etcd cluster"]
async fn leader_loss_recovers_and_quorum_loss_fences_remote_admission() {
    let mut server = Etcd::start(3).await;
    let a = view(52001);
    let (member_a, index_a, _) = join(&server, "quorum", "a", a.clone(), 30).await;
    wait_for(|| a.permits(a.owner()) && index_a.revision().is_some()).await;
    let leader = server.leader().await;
    server.kill(leader);
    let b = view(52002);
    let (member_b, _, _) = join(&server, "quorum", "b", b.clone(), 30).await;
    wait_for(|| a.permits(b.owner()) && b.permits(a.owner())).await;
    server.kill((leader + 1) % 3);
    wait_for(|| !a.registration_valid() && !b.registration_valid()).await;
    assert!(!a.permits(b.owner()));
    member_b.shutdown().await;
    member_a.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires ETCD_BIN; registration reply reconciliation and competing identity rejection"]
async fn registration_reconciles_its_committed_reply_without_advancing_epoch_or_stealing_a_node() {
    let server = Etcd::start(1).await;
    let mut client = Client::connect(&server.endpoints, None).await.unwrap();
    let grant = client.lease_grant(60, None).await.unwrap();
    let cluster = cluster_id(grant.header()).unwrap();
    let owner = view(57001);
    let prefix = "/orbitkv/v2/register-retry/";
    watch::install_format(&mut client, prefix, cluster)
        .await
        .unwrap();
    let first =
        registration::register(&mut client, prefix, "a", owner.owner(), grant.id(), cluster)
            .await
            .unwrap();
    // Repeat the same intent as after a lost response; the committed identity is authoritative.
    let repeated =
        registration::register(&mut client, prefix, "a", owner.owner(), grant.id(), cluster)
            .await
            .unwrap();
    assert_eq!(first, repeated);
    assert_eq!(first.epoch, 1);
    assert_eq!(
        client
            .get(format!("{prefix}epochs/a"), None)
            .await
            .unwrap()
            .kvs()[0]
            .value(),
        b"1"
    );
    let other = view(57001);
    assert!(matches!(
        registration::register(&mut client, prefix, "a", other.owner(), grant.id(), cluster).await,
        Err(BootstrapError::Rejected(_))
    ));
    let other_lease = client.lease_grant(60, None).await.unwrap();
    assert!(matches!(
        registration::register(
            &mut client,
            prefix,
            "a",
            owner.owner(),
            other_lease.id(),
            cluster
        )
        .await,
        Err(BootstrapError::Rejected(_))
    ));
    client.lease_revoke(grant.id()).await.unwrap();
    let replacement = registration::register(
        &mut client,
        prefix,
        "a",
        other.owner(),
        other_lease.id(),
        cluster,
    )
    .await
    .unwrap();
    assert_eq!(replacement.epoch, 2);
    assert_ne!(first.owner.incarnation, replacement.owner.incarnation);
}
