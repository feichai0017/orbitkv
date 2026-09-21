use std::sync::mpsc;
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use super::*;

#[test]
fn deferred_reply_allows_an_unrelated_request_to_finish_first() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let name = format!("orbitkv/test/{}/{nonce}", std::process::id());
    let server = TransportServer::bind(&name).unwrap();
    let first = TransportClient::connect(&name).unwrap();
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
        let mut pending = None;
        let mut served_second = false;
        let deadline = Instant::now() + Duration::from_secs(5);
        while !served_second && Instant::now() < deadline {
            server
                .try_serve_deferred_for_epoch(7, |command, reply| {
                    if command.request_id == 1 {
                        pending = Some(reply);
                        pending_tx.send(()).unwrap();
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
        let first_command = Command::ping(1, 7);
        pending.unwrap().send(Response::ok(first_command)).unwrap();
    });
    let first_thread = thread::spawn(move || {
        first.call_until_peer_exit(Command::ping(1, 7), publish_options, &peer)
    });
    pending_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    let response = second.call(Command::ping(2, 7), options).unwrap();
    assert_eq!(response.request_id, 2);
    thread::sleep(Duration::from_millis(30));
    release_tx.send(()).unwrap();
    assert_eq!(first_thread.join().unwrap().unwrap().request_id, 1);
    server_thread.join().unwrap();
}
