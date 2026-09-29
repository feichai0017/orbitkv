use super::*;

#[test]
fn unknown_watch_transport_status_resumes_while_compaction_rebuilds() {
    let reset = etcd_client::Error::GRpcStatus(tonic::Status::unknown(
        "h2 protocol error: stream closed because of a broken pipe",
    ));
    assert!(matches!(
        FollowError::from(reset),
        FollowError::Disconnected(_)
    ));

    let unknown =
        etcd_client::Error::GRpcStatus(tonic::Status::unknown("metadata application failed"));
    assert!(matches!(
        FollowError::from(unknown),
        FollowError::Rebuild(_)
    ));

    let compacted = etcd_client::Error::GRpcStatus(tonic::Status::out_of_range(
        "required revision has been compacted",
    ));
    assert!(matches!(
        FollowError::from(compacted),
        FollowError::Rebuild(_)
    ));
}
