use std::sync::mpsc;
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use super::*;

#[test]
fn deferred_publishers_allow_an_unrelated_request_to_finish_first() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let name = format!("orbitkv/test/{}/{nonce}", std::process::id());
    let server = TransportServer::bind(&name).unwrap();
    let publishers: Vec<_> = (0..8)
        .map(|_| TransportClient::connect(&name).unwrap())
        .collect();
    let second = TransportClient::connect(&name).unwrap();
    let options = CallOptions {
        timeout: Duration::from_secs(5),
        spin_iterations: 0,
    };
    let publish_options = CallOptions {
        timeout: Duration::from_millis(10),
        spin_iterations: 0,
    };
    let peer = rustix::process::pidfd_open(
        rustix::process::getpid(),
        rustix::process::PidfdFlags::empty(),
    )
    .unwrap();
    let (pending_tx, pending_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let server_thread = thread::spawn(move || {
        let mut pending = Vec::new();
        let mut served_second = false;
        let deadline = Instant::now() + Duration::from_secs(5);
        while !served_second && Instant::now() < deadline {
            server
                .try_serve_deferred_for_epoch(7, |command, reply| {
                    if command.request_id <= 8 {
                        pending.push((command, reply));
                        if pending.len() == 8 {
                            pending_tx.send(()).unwrap();
                        }
                        Ok(())
                    } else {
                        served_second = true;
                        reply.send(Response::ok(command))
                    }
                })
                .unwrap();
            thread::yield_now();
        }
        assert!(served_second);
        release_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        for (command, reply) in pending {
            reply.send(Response::ok(command)).unwrap();
        }
    });
    let peer = std::sync::Arc::new(peer);
    let threads: Vec<_> = publishers
        .into_iter()
        .enumerate()
        .map(|(index, client)| {
            let peer = std::sync::Arc::clone(&peer);
            thread::spawn(move || {
                client.call_until_peer_exit(
                    Command::ping(index as u64 + 1, 7),
                    publish_options,
                    &peer,
                )
            })
        })
        .collect();
    pending_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    let response = second.call(Command::ping(9, 7), options).unwrap();
    assert_eq!(response.request_id, 9);
    thread::sleep(Duration::from_millis(30));
    release_tx.send(()).unwrap();
    for (index, thread) in threads.into_iter().enumerate() {
        assert_eq!(thread.join().unwrap().unwrap().request_id, index as u64 + 1);
    }
    server_thread.join().unwrap();
}
