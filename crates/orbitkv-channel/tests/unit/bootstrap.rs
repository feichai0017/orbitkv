use std::sync::Arc;
use std::thread;

use super::*;

fn connect_pair(server: &BootstrapServer, path: &Path) -> (BootstrapClient, BootstrapSession) {
    thread::scope(|scope| {
        let client = scope.spawn(|| BootstrapClient::connect(path).unwrap());
        let session = server.accept().unwrap();
        (client.join().unwrap(), session)
    })
}

#[test]
fn bootstrap_passes_arena_and_notification_descriptors() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("bootstrap.sock");
    let server = Arc::new(
        BootstrapServer::bind(&path, "orbitkv/test/bootstrap", 29, 16 * 1024, 1024).unwrap(),
    );
    let server_thread = {
        let server = Arc::clone(&server);
        thread::spawn(move || {
            let session = server.accept().unwrap();
            assert_eq!(session.credentials().uid, geteuid().as_raw());
            assert!(session.credentials().pid > 0);
            session.notify().unwrap();
            session
        })
    };

    let client = BootstrapClient::connect(&path).unwrap();
    assert_eq!(client.info().session_epoch, 29);
    assert_eq!(client.info().service_name, "orbitkv/test/bootstrap");
    assert_ne!(client.info().client_token, 0);
    assert!(
        client
            .wait_for_notification(Duration::from_secs(1))
            .unwrap()
    );
    let descriptor = client.write_request(b"query").unwrap();
    assert_eq!(
        server.descriptor_slot(descriptor.offset).unwrap(),
        client.info().slot_index
    );
    assert_eq!(server.arena().read(descriptor).unwrap(), b"query");
    let response = server.arena().write_response(descriptor, b"ready").unwrap();
    assert_eq!(
        client.read_response(descriptor, response).unwrap(),
        b"ready"
    );
    assert_eq!(response.generation, descriptor.generation + 1);

    drop(client);
    server_thread.join().unwrap();
}

#[test]
fn reused_slot_gets_a_new_generation_and_old_request_is_rejected() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("bootstrap.sock");
    let server = BootstrapServer::bind(&path, "orbitkv/test/reuse", 31, 16 * 1024, 4096).unwrap();

    let (first_client, mut first_session) = connect_pair(&server, &path);
    let first = first_client.write_request(b"first").unwrap();
    let first_slot = server.descriptor_slot(first.offset).unwrap();
    first_session
        .validate_request(first, first_client.info().client_token, first_slot)
        .unwrap();
    first_session.complete_request().unwrap();
    drop(first_session);
    drop(first_client);

    let (second_client, mut second_session) = connect_pair(&server, &path);
    assert_eq!(second_client.info().slot_index, first_slot);
    assert!(second_client.info().initial_generation > first.generation);
    assert!(!second_client.info().initial_generation.is_multiple_of(2));
    assert!(matches!(
        second_session.validate_request(first, second_client.info().client_token, first_slot),
        Err(BootstrapError::UnexpectedGeneration { .. })
    ));
}

#[test]
fn failed_bind_does_not_remove_an_existing_socket() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("bootstrap.sock");
    let first = BootstrapServer::bind(&path, "orbitkv/test/first", 41, 16 * 1024, 1024).unwrap();
    assert!(BootstrapServer::bind(&path, "orbitkv/test/second", 43, 16 * 1024, 1024).is_err());
    assert!(path.exists());
    drop(first);
    assert!(!path.exists());
}

#[test]
fn bind_reclaims_a_stale_socket_after_owner_exit() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("bootstrap.sock");
    drop(UnixListener::bind(&path).unwrap());
    assert!(path.exists());

    let server = BootstrapServer::bind(&path, "orbitkv/test/restart", 53, 16 * 1024, 1024).unwrap();
    assert!(path.exists());
    drop(server);
    assert!(!path.exists());
}

#[test]
fn bind_never_removes_a_non_socket_path() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("bootstrap.sock");
    fs::write(&path, b"keep").unwrap();

    assert!(
        BootstrapServer::bind(&path, "orbitkv/test/regular-file", 59, 16 * 1024, 1024).is_err()
    );
    assert_eq!(fs::read(&path).unwrap(), b"keep");
}

#[test]
fn session_rejects_wrong_identity_and_replayed_generation() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("bootstrap.sock");
    let server =
        BootstrapServer::bind(&path, "orbitkv/test/identity", 47, 16 * 1024, 1024).unwrap();
    let (client, mut session) = connect_pair(&server, &path);
    let descriptor = client.write_request(b"query").unwrap();
    let slot = server.descriptor_slot(descriptor.offset).unwrap();

    assert!(matches!(
        session.validate_request(descriptor, client.info().client_token + 1, slot),
        Err(BootstrapError::ClientTokenMismatch)
    ));
    assert!(matches!(
        session.validate_request(descriptor, client.info().client_token, slot + 1),
        Err(BootstrapError::SlotMismatch { .. })
    ));
    session
        .validate_request(descriptor, client.info().client_token, slot)
        .unwrap();
    session.complete_request().unwrap();
    assert!(matches!(
        session.validate_request(descriptor, client.info().client_token, slot),
        Err(BootstrapError::UnexpectedGeneration { .. })
    ));
}
