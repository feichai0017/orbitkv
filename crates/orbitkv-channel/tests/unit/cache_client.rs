use super::*;
use crate::{
    BootstrapServer, BootstrapSession, Command, CommandCode, RESPONSE_FLAG_REQUEST_CONSUMED,
    Response, RestoreCommand, TransportServer,
};
use std::sync::mpsc;
use std::thread;

fn hashes(values: &[&[u8]]) -> BlockHashes {
    BlockHashes::new(values.iter().map(|value| value.to_vec()).collect())
}

fn key(instance: &str, group: u32) -> QueryKey {
    QueryKey {
        instance: instance.into(),
        request: "r".into(),
        group,
    }
}

#[test]
fn polls_reuse_hash_storage_and_changed_demand_revises_one_ticket() {
    let mut queries = Queries::default();
    let key = key("model", 0);
    let batch = hashes(&[b"one", b"two"]);
    let first = queries.prepare(&key, &batch, false).unwrap();
    let QueryCommand::Submit(first) = first else {
        panic!("expected submit")
    };
    let storage = queries.pending[&key].hashes.as_slice()[0].as_ptr();
    for _ in 0..10 {
        assert_eq!(
            queries.prepare(&key, &batch, false).unwrap(),
            QueryCommand::Poll(first.ticket)
        );
        assert_eq!(queries.pending[&key].hashes.as_slice()[0].as_ptr(), storage);
    }
    for (bytes, wait, revision) in [
        (vec![b"changed".as_slice()], false, 2),
        (vec![b"changed".as_slice()], true, 3),
    ] {
        let QueryCommand::Submit(changed) = queries.prepare(&key, &hashes(&bytes), wait).unwrap()
        else {
            panic!("expected revision")
        };
        assert_eq!(
            changed.ticket,
            QueryTicket {
                operation_id: 1,
                revision
            }
        );
    }
    queries.pending.get_mut(&key).unwrap().ticket.revision = u64::MAX;
    assert!(queries.prepare(&key, &hashes(&[b"new"]), false).is_err());
}

#[test]
fn terminal_and_busy_queries_retire_and_instances_and_groups_do_not_alias() {
    let mut queries = Queries::default();
    for (index, key) in [key("a", 0), key("a", 1), key("b", 0)].iter().enumerate() {
        let QueryCommand::Submit(request) =
            queries.prepare(key, &hashes(&[b"hash"]), false).unwrap()
        else {
            panic!("expected submit")
        };
        assert_eq!(request.ticket.operation_id, index as u64 + 1);
    }
    for outcome in [QueryOutcomeCode::Ready, QueryOutcomeCode::Busy] {
        queries.complete(&key("a", 0), outcome);
        assert!(matches!(
            queries
                .prepare(&key("a", 0), &hashes(&[b"hash"]), false)
                .unwrap(),
            QueryCommand::Submit(_)
        ));
    }
    queries.next_operation = u64::MAX;
    assert!(queries.ticket().is_err());
    assert!(next_id(&AtomicU64::new(u64::MAX)).is_err());
}

#[test]
fn hash_views_share_storage_but_do_not_hide_changed_content_or_bounds() {
    let batch = hashes(&[b"first", b"second", b"third"]);
    let view = batch.slice(1..3).unwrap();
    assert!(Arc::ptr_eq(&batch.hashes, &view.hashes));
    assert_eq!(view.as_slice()[0].as_ptr(), batch.as_slice()[1].as_ptr());
    let prefix = view.slice(0..1).unwrap();
    assert_eq!(prefix, hashes(&[b"second"]));
    assert_ne!(prefix, batch.slice(0..1).unwrap());
    assert!(batch.slice(0..4).is_none());
    assert!(batch.slice(Range { start: 1, end: 0 }).is_none());
    assert!(view.slice(2..2).unwrap().as_slice().is_empty());
    let mut queries = Queries::default();
    queries.prepare(&key("m", 0), &view, false).unwrap();
    // Different allocations with equal values retain their operation too.
    assert!(matches!(
        queries
            .prepare(&key("m", 0), &hashes(&[b"second", b"third"]), false)
            .unwrap(),
        QueryCommand::Poll(_)
    ));
    assert!(matches!(
        queries.prepare(&key("m", 0), &prefix, false).unwrap(),
        QueryCommand::Submit(_)
    ));
}

/// Real bootstrap/descriptor framing with controlled completion timing.
struct Peer {
    socket: PathBuf,
    stop: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
    _dir: tempfile::TempDir,
}

impl Peer {
    fn new(
        mut reply: impl FnMut(Command, &[u8], &BootstrapSession) -> Option<Vec<u8>> + Send + 'static,
    ) -> Self {
        static NAMES: AtomicU64 = AtomicU64::new(1);
        let dir = tempfile::tempdir().unwrap();
        let socket = dir.path().join("cache.sock");
        let name = format!(
            "orbitkv/test/client/{}/{}",
            std::process::id(),
            NAMES.fetch_add(1, Ordering::Relaxed)
        );
        let bootstrap = BootstrapServer::bind(&socket, &name, 71, 64 * 1024, 4096).unwrap();
        bootstrap.set_nonblocking(true).unwrap();
        let transport = TransportServer::bind(&name).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = Arc::clone(&stop);
        let thread = thread::spawn(move || {
            let mut sessions = HashMap::new();
            let mut pending = Vec::new();
            while !stopping.load(Ordering::Acquire) {
                while let Some(session) = bootstrap.try_accept().unwrap() {
                    sessions.insert(session.client_token(), session);
                }
                transport
                    .try_serve_deferred_for_epoch(71, |command, response| {
                        let session = sessions.get_mut(&command.arg0).unwrap();
                        let slot = bootstrap
                            .descriptor_slot(command.descriptor.offset)
                            .unwrap();
                        session
                            .validate_request(command.descriptor, command.arg0, slot)
                            .unwrap();
                        let payload = bootstrap.arena().read(command.descriptor).unwrap();
                        pending.push((command, payload, response));
                        Ok(())
                    })
                    .unwrap();
                let mut index = 0;
                while index < pending.len() {
                    let (command, payload, _) = &pending[index];
                    let session = sessions.get_mut(&command.arg0).unwrap();
                    if let Some(bytes) = reply(*command, payload, session) {
                        let (command, _, response) = pending.swap_remove(index);
                        let mut result = Response::ok(command);
                        result.descriptor = bootstrap
                            .arena()
                            .write_response(command.descriptor, &bytes)
                            .unwrap();
                        result.value1 = RESPONSE_FLAG_REQUEST_CONSUMED;
                        session.complete_request().unwrap();
                        response.send(result).unwrap();
                    } else {
                        index += 1;
                    }
                }
                thread::sleep(Duration::from_micros(100));
            }
        });
        Self {
            socket,
            stop,
            thread: Some(thread),
            _dir: dir,
        }
    }

    fn client(&self) -> CacheClient {
        CacheClient::connect(&self.socket, CallOptions::default()).unwrap()
    }
}

impl Drop for Peer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.thread.take().unwrap().join().unwrap();
    }
}

fn loading() -> Vec<u8> {
    QueryBundleResponse {
        outcome: QueryOutcomeCode::Loading,
        num_hit_blocks: 0,
        lease: vec![],
        hit_positions: vec![],
    }
    .encode()
    .unwrap()
}

#[test]
fn warming_is_bounded_expires_and_is_cancelled_before_demand() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&events);
    let peer = Peer::new(move |command, payload, _| {
        captured.lock().unwrap().push(command.code);
        Some(match command.code {
            CommandCode::QueryBundle => loading(),
            CommandCode::CancelQuery => {
                CancelQueryRequest::decode(payload).unwrap();
                vec![]
            }
            _ => panic!("unexpected command"),
        })
    });
    let client = peer.client();
    assert!(!client.warm_prefix("m", &hashes(&[]), "empty").unwrap());
    for index in 0..MAX_WARMUPS {
        assert!(
            client
                .warm_prefix("m", &hashes(&[b"hash"]), &index.to_string())
                .unwrap()
        );
    }
    assert!(!client.warm_prefix("m", &hashes(&[b"hash"]), "0").unwrap());
    assert!(
        !client
            .warm_prefix("m", &hashes(&[b"hash"]), "overflow")
            .unwrap()
    );
    client
        .query_prefetch("m", &hashes(&[b"changed"]), "0", false, 0)
        .unwrap();
    assert_eq!(
        &events.lock().unwrap()[MAX_WARMUPS..],
        &[CommandCode::CancelQuery, CommandCode::QueryBundle]
    );
    client.cancel_query("m", "0", 0).unwrap();
    client.cancel_query("m", "0", 0).unwrap();
    for (_, submitted) in client.queries.lock().unwrap().warmups.values_mut() {
        *submitted = Instant::now() - WARMUP_TTL;
    }
    assert!(client.warm_prefix("m", &hashes(&[b"new"]), "new").unwrap());
    assert_eq!(client.queries.lock().unwrap().warmups.len(), 1);
    client.close();
    assert!(client.queries.lock().unwrap().warmups.is_empty());
    assert!(
        client
            .query_prefetch("m", &hashes(&[b"h"]), "closed", false, 0)
            .is_err()
    );
}

#[test]
fn rejected_revision_retires_the_previous_interest_without_reusing_its_ticket() {
    let cancels = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&cancels);
    let peer = Peer::new(move |command, payload, _| {
        Some(match command.code {
            CommandCode::QueryBundle => loading(),
            CommandCode::CancelQuery => {
                captured
                    .lock()
                    .unwrap()
                    .push(CancelQueryRequest::decode(payload).unwrap().ticket);
                vec![]
            }
            _ => panic!("unexpected command"),
        })
    });
    let client = peer.client();
    client
        .query_prefetch("m", &hashes(&[b"small"]), "r", false, 0)
        .unwrap();
    let oversized = BlockHashes::new(vec![vec![0; 32]; 1024]);
    assert!(
        client
            .query_prefetch("m", &oversized, "r", false, 0)
            .is_err()
    );
    assert!(client.queries.lock().unwrap().pending.is_empty());
    assert_eq!(
        *cancels.lock().unwrap(),
        vec![
            QueryTicket {
                operation_id: 1,
                revision: 2
            },
            QueryTicket {
                operation_id: 1,
                revision: 1
            },
        ]
    );
    client
        .query_prefetch("m", &hashes(&[b"small"]), "r", false, 0)
        .unwrap();
    assert_eq!(
        client.queries.lock().unwrap().pending[&key("m", 0)]
            .ticket
            .operation_id,
        2
    );
}

#[test]
fn restore_deadline_and_lost_notification_preserve_ownership_and_reject_other_clients() {
    let completed = Arc::new(AtomicBool::new(false));
    let done = Arc::clone(&completed);
    let peer = Peer::new(move |command, payload, _| {
        assert_eq!(
            command.code,
            CommandCode::Restore,
            "a timeout must never release pages"
        );
        RestoreCommand::decode(payload).unwrap();
        Some(
            RestoreResponse {
                operation_id: 9,
                state: if done.load(Ordering::Acquire) {
                    RestoreState::Succeeded
                } else {
                    RestoreState::Pending
                },
                message: String::new(),
            }
            .encode()
            .unwrap(),
        )
    });
    let client = peer.client();
    let other = peer.client();
    let handle = client
        .start_restore(&RestoreRequest {
            instance_id: "m".into(),
            tp_rank: 0,
            device_id: 0,
            layer_groups: vec![],
            loads: vec![],
        })
        .unwrap();
    assert!(other.poll_restore(handle).is_err());
    assert!(matches!(
        client.wait_restore(handle, Duration::from_millis(2)),
        Err(ChannelError::RestoreTimeout { .. })
    ));
    thread::scope(|scope| {
        scope.spawn(|| {
            thread::sleep(Duration::from_millis(10));
            completed.store(true, Ordering::Release);
        });
        assert_eq!(
            client
                .wait_restore(handle, Duration::from_secs(1))
                .unwrap()
                .state,
            RestoreState::Succeeded
        );
    });
}

#[test]
fn blocked_publish_does_not_serialize_query_or_restore() {
    let finish = Arc::new(AtomicBool::new(false));
    let publishing = Arc::clone(&finish);
    let (started_tx, started_rx) = mpsc::channel();
    let mut started = false;
    let peer = Peer::new(move |command, _, _| {
        if command.code == CommandCode::Publish {
            if !started {
                started = true;
                started_tx.send(()).unwrap();
            }
            return publishing.load(Ordering::Acquire).then(Vec::new);
        }
        Some(loading())
    });
    let client = peer.client();
    thread::scope(|scope| {
        let publish = scope.spawn(|| {
            client.publish(&PublishRequest {
                instance_id: "m".into(),
                tp_rank: 0,
                pp_rank: 0,
                device_id: 0,
                layers: vec![],
            })
        });
        started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        let result = client.query_prefetch("m", &hashes(&[b"h"]), "r", false, 0);
        finish.store(true, Ordering::Release);
        assert_eq!(result.unwrap().outcome, QueryOutcomeCode::Loading);
        publish.join().unwrap().unwrap();
    });
    client.close();
    assert!(
        client
            .publish(&PublishRequest {
                instance_id: "m".into(),
                tp_rank: 0,
                pp_rank: 0,
                device_id: 0,
                layers: vec![]
            })
            .is_err()
    );
}
