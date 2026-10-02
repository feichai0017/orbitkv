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

#[tokio::test]
async fn watch_rpc_keeps_semantic_unknown_distinct_from_transport_unknown() {
    let semantic = follow_rpc(async {
        Err::<(), _>(etcd_client::Error::GRpcStatus(tonic::Status::unknown(
            "metadata application failed",
        )))
    })
    .await;
    assert!(matches!(semantic, Err(FollowError::Rebuild(_))));

    let transport = follow_rpc(async {
        Err::<(), _>(etcd_client::Error::GRpcStatus(tonic::Status::unknown(
            "h2 protocol error: connection reset",
        )))
    })
    .await;
    assert!(matches!(transport, Err(FollowError::Disconnected(_))));
}
