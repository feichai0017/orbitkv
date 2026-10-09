use super::*;
use orbitkv_proto::proto::engine::{SessionRequest, ShardQueryTarget};

fn configuration() -> ConfigureShardQueriesRequest {
    let instance = "dense-instance".to_owned();
    ConfigureShardQueriesRequest {
        local: Some(SessionRequest {
            instance_id: instance.clone(),
            namespace: "node-0".into(),
            tp_size: 1,
            world_size: 1,
        }),
        shards: (0..2)
            .map(|node| {
                let endpoint = format!("http://127.0.0.1:{}", 50100 + node);
                let namespace = format!("node-{node}");
                ShardQueryTarget {
                    endpoint: endpoint.clone(),
                    namespace: namespace.clone(),
                    target: Some(RegisteredQueryTarget {
                        version: 1,
                        endpoint,
                        instance_id: instance.clone(),
                        namespace,
                        storage_namespace: format!("store-{node}"),
                        tp_size: 1,
                        world_size: 1,
                        manager_incarnation: Uuid::new_v4().as_bytes().to_vec(),
                        registration_generation: Uuid::new_v4().as_bytes().to_vec(),
                        capability: Uuid::new_v4().as_bytes().to_vec(),
                    }),
                }
            })
            .collect(),
    }
}
#[test]
fn declared_targets_reject_mismatch_duplication_and_routable_peers() {
    validate_configuration(&configuration()).unwrap();
    for case in 0..10 {
        let mut config = configuration();
        match case {
            0 => config.shards.clear(),
            1 => config.shards[1] = config.shards[0].clone(),
            2 => config.shards[1].target.as_mut().unwrap().instance_id = "other".into(),
            3 => config.shards[1].target.as_mut().unwrap().tp_size = 2,
            4 => config.shards[1].namespace = "undeclared".into(),
            5 => config.shards[1].target.as_mut().unwrap().capability = vec![0; 16],
            6 => config.shards[1].endpoint = "http://10.1.1.1:50000".into(),
            7 => config.local.as_mut().unwrap().world_size = 2,
            8 => {
                config.shards[1]
                    .target
                    .as_mut()
                    .unwrap()
                    .manager_incarnation = config.shards[0]
                    .target
                    .as_ref()
                    .unwrap()
                    .manager_incarnation
                    .clone();
            }
            _ => config.shards[1].endpoint = "http://localhost:50101".into(),
        }
        assert!(validate_configuration(&config).is_err(), "case {case}");
    }
}
#[tokio::test]
async fn closed_admission_does_not_reopen_after_capacity_drain() {
    let owner = ShardQueries::new(Handle::current());
    let permit = owner.capacity.clone().acquire_owned().await.unwrap();
    let draining = owner.clone();
    let task = tokio::spawn(async move { draining.stop_and_drain().await });
    tokio::task::yield_now().await;
    assert!(!task.is_finished());
    assert!(owner.book.lock().closed);
    drop(permit);
    task.await.unwrap().unwrap();
    assert!(
        owner
            .execute(
                1,
                QueryCommand::Poll(QueryTicket {
                    operation_id: 1,
                    revision: 1
                })
            )
            .is_err()
    );
    assert_eq!(owner.capacity.available_permits(), CAPACITY);
}

use orbitkv_proto::proto::engine::cache_query_control_server::{
    CacheQueryControl, CacheQueryControlServer,
};
use orbitkv_proto::proto::engine::{QueryControlEmpty, QueryControlResponse};
use tonic::{Request, Response};

#[derive(Default)]
struct SourceState {
    replies: HashMap<u64, Vec<u8>>,
    actions: Vec<String>,
    open: usize,
    closed: usize,
    claims: usize,
}
#[derive(Clone)]
struct Source {
    prefix: u64,
    state: Arc<Mutex<SourceState>>,
    incarnation: Vec<u8>,
    lost_claim: bool,
    loading: Arc<AtomicBool>,
}
#[tonic::async_trait]
impl CacheQueryControl for Source {
    async fn open_interest(
        &self,
        _: Request<OpenQueryInterestRequest>,
    ) -> Result<Response<QueryInterest>, Status> {
        self.state.lock().open += 1;
        Ok(Response::new(QueryInterest {
            id: Uuid::new_v4().as_bytes().to_vec(),
            manager_incarnation: self.incarnation.clone(),
        }))
    }
    async fn execute(
        &self,
        request: Request<QueryControlExecuteRequest>,
    ) -> Result<Response<QueryControlResponse>, Status> {
        let command = QueryCommand::decode(&request.into_inner().command).unwrap();
        let mut state = self.state.lock();
        let operation = match command {
            QueryCommand::Submit(query) => {
                state.actions.push(format!(
                    "submit:{}:{}",
                    query.ticket.operation_id,
                    query.block_hashes.len()
                ));
                let hit = self.prefix.min(query.block_hashes.len() as u64);
                state.replies.insert(
                    query.ticket.operation_id,
                    QueryBundleResponse {
                        outcome: QueryOutcomeCode::Ready,
                        num_hit_blocks: hit,
                        lease: if hit > 0 {
                            Uuid::new_v4().as_bytes().to_vec()
                        } else {
                            Vec::new()
                        },
                        hit_positions: Vec::new(),
                    }
                    .encode()
                    .unwrap(),
                );
                query.ticket.operation_id
            }
            QueryCommand::Poll(ticket) => ticket.operation_id,
            _ => return Err(Status::invalid_argument("claim command")),
        };
        Ok(Response::new(QueryControlResponse {
            payload: if self.loading.load(Ordering::Acquire) {
                QueryBundleResponse::loading().encode().unwrap()
            } else {
                state.replies[&operation].clone()
            },
        }))
    }
    async fn claim(
        &self,
        request: Request<QueryControlClaimRequest>,
    ) -> Result<Response<QueryControlResponse>, Status> {
        let request = request.into_inner();
        let mut state = self.state.lock();
        state.claims += 1;
        let payload = state.replies[&request.operation_id].clone();
        assert_eq!(
            QueryBundleResponse::decode(&payload).unwrap().lease,
            request.lease
        );
        if self.lost_claim && state.claims == 1 {
            return Err(Status::unavailable("response lost after claim"));
        }
        Ok(Response::new(QueryControlResponse { payload }))
    }
    async fn cancel(
        &self,
        request: Request<QueryControlCancelRequest>,
    ) -> Result<Response<QueryControlEmpty>, Status> {
        let request = request.into_inner();
        let mut state = self.state.lock();
        state
            .actions
            .push(format!("cancel:{}", request.operation_id));
        state.replies.remove(&request.operation_id);
        Ok(Response::new(QueryControlEmpty {}))
    }
    async fn close(
        &self,
        _: Request<QueryInterest>,
    ) -> Result<Response<QueryControlEmpty>, Status> {
        self.state.lock().closed += 1;
        Ok(Response::new(QueryControlEmpty {}))
    }
}
fn query(operation_id: u64, revision: u64, blocks: usize) -> QueryCommand {
    QueryCommand::Submit(QueryBundleRequest {
        ticket: QueryTicket {
            operation_id,
            revision,
        },
        instance_id: "dense-instance".into(),
        request_id: "request".into(),
        block_hashes: vec![vec![1; 32]; blocks],
        group_id: 0,
        wait_for_full_prefix: false,
        warmup: false,
        discover: false,
        materialize: false,
        prepare: false,
        demand: None,
    })
}
async fn source_target(
    prefix: u64,
    lost_claim: bool,
    loading: bool,
) -> (
    Target,
    Source,
    tokio::sync::oneshot::Sender<()>,
    tokio::task::JoinHandle<()>,
) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = format!("http://{}", listener.local_addr().unwrap());
    let source = Source {
        prefix,
        state: Arc::default(),
        incarnation: Uuid::new_v4().as_bytes().to_vec(),
        lost_claim,
        loading: Arc::new(AtomicBool::new(loading)),
    };
    let mut wire = configuration().shards.remove(0).target.unwrap();
    wire.endpoint = address.clone();
    wire.manager_incarnation = source.incarnation.clone();
    let client = CacheQueryControlClient::new(
        Endpoint::from_shared(address)
            .unwrap()
            .timeout(RPC_TIMEOUT)
            .connect_lazy(),
    );
    let (stop, receiver) = tokio::sync::oneshot::channel();
    let service = source.clone();
    let task = tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(CacheQueryControlServer::new(service))
            .serve_with_incoming_shutdown(
                tokio_stream::wrappers::TcpListenerStream::new(listener),
                async {
                    let _ = receiver.await;
                },
            )
            .await
            .unwrap();
    });
    (Target { wire, client }, source, stop, task)
}
fn install(owner: &ShardQueries, token: u64, targets: Vec<Target>) {
    owner.book.lock().sessions.insert(
        token,
        Session {
            instance: "dense-instance".into(),
            configuration: Vec::new(),
            targets,
            retired: 0,
            operations: HashMap::new(),
        },
    );
}
async fn ready(owner: &ShardQueries, ticket: QueryTicket) -> ShardQueryResponse {
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let response = owner.execute(1, QueryCommand::Poll(ticket)).unwrap();
            if response.outcome != QueryOutcomeCode::Loading {
                return response;
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    })
    .await
    .unwrap()
}
#[tokio::test]
async fn common_prefix_cancels_long_reply_before_reacquiring_and_replays_exact_claim() {
    let (first, a, stop_a, task_a) = source_target(3, true, false).await;
    let (second, b, stop_b, task_b) = source_target(2, false, false).await;
    let owner = ShardQueries::new(Handle::current());
    install(&owner, 1, vec![first.clone(), second.clone()]);
    install(&owner, 3, vec![first, second]);
    assert_eq!(
        owner.execute(1, query(1, 1, 4)).unwrap().outcome,
        QueryOutcomeCode::Loading
    );
    let response = ready(
        &owner,
        QueryTicket {
            operation_id: 1,
            revision: 1,
        },
    )
    .await;
    assert_eq!(response.num_hit_blocks, 2);
    assert_eq!(
        a.state.lock().actions,
        ["submit:1:4", "cancel:1", "submit:2:2"]
    );
    assert_eq!(a.state.lock().claims, 2);
    assert_eq!(b.state.lock().claims, 1);
    assert_eq!(owner.capacity.available_permits(), CAPACITY - 1);
    owner.release(3, &response.control_id).unwrap();
    assert_eq!(owner.capacity.available_permits(), CAPACITY - 1);
    owner.release(1, &response.control_id).unwrap();
    owner.release(1, &response.control_id).unwrap();
    assert!(owner.execute(1, query(1, 2, 4)).is_err());
    owner.stop_and_drain().await.unwrap();
    {
        let state = a.state.lock();
        assert_eq!(state.open, state.closed);
    }
    {
        let state = b.state.lock();
        assert_eq!(state.open, state.closed);
    }
    stop_a.send(()).unwrap();
    stop_b.send(()).unwrap();
    task_a.await.unwrap();
    task_b.await.unwrap();
}
#[tokio::test]
async fn replacement_revision_and_cancellation_drain_every_known_interest() {
    let (target, source, stop, task) = source_target(3, false, true).await;
    let owner = ShardQueries::new(Handle::current());
    install(&owner, 1, vec![target]);
    owner.execute(1, query(1, 1, 4)).unwrap();
    tokio::time::timeout(Duration::from_secs(5), async {
        while source.state.lock().open == 0 {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .unwrap();
    owner.execute(1, query(1, 2, 1)).unwrap();
    source.loading.store(false, Ordering::Release);
    let response = ready(
        &owner,
        QueryTicket {
            operation_id: 1,
            revision: 2,
        },
    )
    .await;
    assert_eq!(response.num_hit_blocks, 1);
    assert!(owner.execute(1, query(1, 1, 4)).is_err());
    let mut changed = query(1, 2, 2);
    assert!(owner.execute(1, changed.clone()).is_err());
    owner.cancel(
        1,
        QueryTicket {
            operation_id: 1,
            revision: 2,
        },
    );
    changed = query(1, 3, 1);
    assert!(owner.execute(1, changed).is_err());
    owner.stop_and_drain().await.unwrap();
    {
        let state = source.state.lock();
        assert_eq!(state.open, state.closed);
    }
    assert_eq!(owner.capacity.available_permits(), CAPACITY);
    stop.send(()).unwrap();
    task.await.unwrap();
}
