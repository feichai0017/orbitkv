use std::process::{Child, Command, Stdio};

use super::*;
use etcd_client::{PutOptions, Txn, TxnOp};

struct Etcd {
    process: Child,
    endpoint: String,
    _directory: tempfile::TempDir,
}

impl Etcd {
    async fn start() -> Self {
        let binary = std::env::var("ETCD_BIN").expect("set ETCD_BIN to an etcd 3.x binary");
        let directory = tempfile::tempdir().unwrap();
        let client_port = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let peer_port = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", client_port.local_addr().unwrap());
        let peer = format!("http://{}", peer_port.local_addr().unwrap());
        drop((client_port, peer_port));
        let log = std::fs::File::create(directory.path().join("etcd.log")).unwrap();
        let process = Command::new(binary)
            .args(["--name", "test", "--data-dir"])
            .arg(directory.path().join("data"))
            .args([
                "--listen-client-urls",
                &endpoint,
                "--advertise-client-urls",
                &endpoint,
                "--listen-peer-urls",
                &peer,
                "--initial-advertise-peer-urls",
                &peer,
                "--initial-cluster",
                &format!("test={peer}"),
                "--heartbeat-interval",
                "50",
                "--election-timeout",
                "500",
                "--log-level",
                "error",
            ])
            .stdout(Stdio::null())
            .stderr(log)
            .spawn()
            .unwrap();
        let mut server = Self {
            process,
            endpoint,
            _directory: directory,
        };
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            assert!(
                server.process.try_wait().unwrap().is_none(),
                "etcd exited during startup"
            );
            if let Ok(mut client) = rpc(Client::connect([&server.endpoint], None)).await
                && rpc(client.status()).await.is_ok()
            {
                return server;
            }
            assert!(Instant::now() < deadline, "etcd startup timed out");
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    fn signal(&self, signal: &str) {
        assert!(
            Command::new("kill")
                .args([signal, &self.process.id().to_string()])
                .status()
                .unwrap()
                .success()
        );
    }
}

impl Drop for Etcd {
    fn drop(&mut self) {
        let _ = self.process.kill();
        let _ = self.process.wait();
    }
}

fn view(port: u16) -> Arc<MembershipView> {
    Arc::new(MembershipView::new(CacheOwner {
        endpoint: format!("127.0.0.1:{port}"),
        incarnation: uuid::Uuid::new_v4(),
    }))
}

async fn wait_for(condition: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(20);
    while !condition() {
        assert!(Instant::now() < deadline, "membership did not converge");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires ETCD_BIN; starts a real isolated etcd process"]
async fn registration_watch_compaction_and_restart_preserve_incarnations() {
    let server = Etcd::start().await;
    let endpoints = [server.endpoint.clone()];
    let a = view(51001);
    let b = view(51002);
    let member_a = Membership::join(&endpoints, "test", "a", 12, a.clone())
        .await
        .unwrap();
    let member_b = Membership::join(&endpoints, "test", "b", 12, b.clone())
        .await
        .unwrap();
    wait_for(|| a.permits(b.owner()) && b.permits(a.owner())).await;
    assert!(
        Membership::join(&endpoints, "test", "a", 12, view(51003))
            .await
            .err()
            .unwrap()
            .contains("live registration")
    );
    assert!(a.permits(a.owner()));

    let mut client = Client::connect(&endpoints, None).await.unwrap();
    let status = client.status().await.unwrap();
    let cluster = cluster_id(status.header()).unwrap();
    let prefix = "/orbitkv/v1/test/members/";
    let (members, revision) = watch::snapshot(&mut client, prefix, cluster).await.unwrap();
    assert_eq!(members["a"].epoch, 1);
    assert_eq!(members["b"].epoch, 1);
    client.put("test-compaction", "1", None).await.unwrap();
    let later = client
        .put("test-compaction", "2", None)
        .await
        .unwrap()
        .header()
        .unwrap()
        .revision();
    client.compact(later, None).await.unwrap();
    let error = tokio::time::timeout(
        Duration::from_secs(5),
        watch::follow(
            &mut client,
            prefix,
            cluster,
            &members["a"].clone(),
            &a,
            members.clone(),
            revision,
        ),
    )
    .await
    .unwrap()
    .unwrap_err();
    assert!(error.contains("compacted"), "{error}");
    let (repaired, _) = watch::snapshot(&mut client, prefix, cluster).await.unwrap();
    assert_eq!(repaired.len(), 2);

    member_b.shutdown().await;
    wait_for(|| !a.permits(b.owner())).await;
    assert!(!b.permits(b.owner()));
    let restarted = view(51002);
    let replacement = Membership::join(&endpoints, "test", "b", 12, restarted.clone())
        .await
        .unwrap();
    wait_for(|| a.permits(restarted.owner())).await;
    assert!(!a.permits(b.owner()));
    let (members, _) = watch::snapshot(&mut client, prefix, cluster).await.unwrap();
    assert_eq!(members["b"].epoch, 2);

    // Revocation/deletion cannot be repaired by keepalive as a new registration.
    client.lease_revoke(replacement.lease).await.unwrap();
    wait_for(|| !a.permits(restarted.owner()) && !restarted.permits(restarted.owner())).await;
    assert!(!restarted.renew(Instant::now(), Duration::from_secs(30)));
    replacement.shutdown().await;
    let crashed = view(51004);
    let crash_registration = Membership::join(&endpoints, "test", "crashed", 12, crashed.clone())
        .await
        .unwrap();
    wait_for(|| a.permits(crashed.owner())).await;
    drop(crash_registration);
    assert!(!crashed.permits(crashed.owner()));
    wait_for(|| !a.permits(crashed.owner())).await;

    // Matching endpoint/UUID alone is insufficient if the registration's
    // epoch or attached lease changed underneath the running Manager.
    for change in ["epoch", "lease", "recreate"] {
        let candidate = view(51005);
        let registration = Membership::join(&endpoints, "test", change, 12, candidate.clone())
            .await
            .unwrap();
        wait_for(|| candidate.permits(candidate.owner())).await;
        let key = format!("{prefix}{change}");
        let bytes = client.get(key.clone(), None).await.unwrap().kvs()[0]
            .value()
            .to_vec();
        let mut record: Member = serde_json::from_slice(&bytes).unwrap();
        let replacement_lease = if change == "lease" {
            client.lease_grant(12, None).await.unwrap().id()
        } else {
            registration.lease
        };
        if change == "epoch" {
            record.epoch += 1;
        }
        if change == "recreate" {
            client.delete(key.clone(), None).await.unwrap();
        }
        client
            .put(
                key,
                serde_json::to_vec(&record).unwrap(),
                Some(PutOptions::new().with_lease(replacement_lease)),
            )
            .await
            .unwrap();
        wait_for(|| !candidate.permits(candidate.owner())).await;
        assert!(!candidate.renew(Instant::now(), Duration::from_secs(30)));
        registration.shutdown().await;
        if change == "lease" {
            client.lease_revoke(replacement_lease).await.unwrap();
        }
    }
    member_a.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires ETCD_BIN; pauses a real coordinator to exercise lease expiry"]
async fn coordinator_outage_fences_remote_admission_without_reviving_the_runtime() {
    let server = Etcd::start().await;
    let endpoints = [server.endpoint.clone()];
    let owner = view(52001);
    let membership = Membership::join(&endpoints, "outage", "a", 12, owner.clone())
        .await
        .unwrap();
    wait_for(|| owner.permits(owner.owner())).await;
    server.signal("-STOP");
    tokio::time::sleep(Duration::from_secs(7)).await;
    assert!(!owner.permits(owner.owner()));
    server.signal("-CONT");
    tokio::time::sleep(Duration::from_secs(4)).await;
    assert!(
        !owner.permits(owner.owner()),
        "late renewal must not re-enable this incarnation"
    );
    assert!(!owner.renew(Instant::now(), Duration::from_secs(30)));
    membership.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires ETCD_BIN; verifies bounded snapshots against real member records"]
async fn oversized_membership_snapshot_is_rejected_without_partial_publication() {
    let server = Etcd::start().await;
    let mut client = Client::connect([&server.endpoint], None).await.unwrap();
    let grant = client.lease_grant(120, None).await.unwrap();
    let cluster = cluster_id(grant.header()).unwrap();
    let owner = view(53001);
    let prefix = "/orbitkv/v1/budget/members/";
    for start in (0..=MAX_MEMBERS).step_by(100) {
        let operations = (start..(start + 100).min(MAX_MEMBERS + 1))
            .map(|i| {
                let member = Member {
                    node_id: format!("node-{i:05}"),
                    epoch: 1,
                    owner: owner.owner().clone(),
                    lease: grant.id(),
                };
                TxnOp::put(
                    format!("{prefix}{}", member.node_id),
                    serde_json::to_vec(&member).unwrap(),
                    Some(PutOptions::new().with_lease(grant.id())),
                )
            })
            .collect::<Vec<_>>();
        client.txn(Txn::new().and_then(operations)).await.unwrap();
    }
    assert!(
        watch::snapshot(&mut client, prefix, cluster)
            .await
            .unwrap_err()
            .contains("budget")
    );
}
