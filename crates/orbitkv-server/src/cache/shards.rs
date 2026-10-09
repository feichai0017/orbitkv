//! One bounded native owner for a registered dense TP query and its source holds.
use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use futures::future::join_all;
use orbitkv_channel::{
    QueryBundleRequest, QueryBundleResponse, QueryCommand, QueryOutcomeCode, QueryTicket,
    ShardQueryResponse,
};
use orbitkv_proto::proto::engine::cache_query_control_client::CacheQueryControlClient;
use orbitkv_proto::proto::engine::{
    ConfigureShardQueriesRequest, OpenQueryInterestRequest, QueryControlCancelRequest,
    QueryControlClaimRequest, QueryControlExecuteRequest, QueryInterest, RegisteredQueryTarget,
};
use parking_lot::Mutex;
use prost::Message;
use sha2::{Digest, Sha256};
use tokio::runtime::Handle;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tonic::transport::{Channel, Endpoint};
use tonic::{Code, Status};
use uuid::Uuid;

use super::query_control::QueryControlService;

const CAPACITY: usize = 1024;
const PER_SESSION: usize = 128;
const RPC_TIMEOUT: Duration = Duration::from_secs(2);
const QUERY_TIMEOUT: Duration = Duration::from_secs(30);
const HOLD_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Clone)]
struct Target {
    wire: RegisteredQueryTarget,
    client: CacheQueryControlClient<Channel>,
}
struct NodeInterest {
    target: Target,
    interest: Option<QueryInterest>,
    ticket: QueryTicket,
    reply: Option<QueryBundleResponse>,
}
struct Transaction {
    nodes: Vec<NodeInterest>,
    permit: Option<OwnedSemaphorePermit>,
    runtime: Handle,
}
impl Transaction {
    async fn close(&mut self) {
        join_all(self.nodes.iter_mut().map(NodeInterest::close)).await;
        self.permit.take();
    }
}
impl Drop for Transaction {
    fn drop(&mut self) {
        let mut nodes = std::mem::take(&mut self.nodes);
        let permit = self.permit.take();
        if nodes.iter().any(|node| node.interest.is_some()) {
            self.runtime.spawn(async move {
                join_all(nodes.iter_mut().map(NodeInterest::close)).await;
                drop(permit);
            });
        }
    }
}
struct Ready {
    response: ShardQueryResponse,
    _transaction: Option<Transaction>,
}
struct Operation {
    ticket: QueryTicket,
    digest: [u8; 32],
    canceled: Arc<AtomicBool>,
    result: Arc<Mutex<Option<Result<Ready, Status>>>>,
    expires: Instant,
}
impl Drop for Operation {
    fn drop(&mut self) {
        self.canceled.store(true, Ordering::Release);
    }
}
struct Session {
    instance: String,
    configuration: Vec<u8>,
    targets: Vec<Target>,
    retired: u64,
    operations: HashMap<u64, Operation>,
}
#[derive(Default)]
struct Book {
    closed: bool,
    submitted: u64,
    sessions: HashMap<u64, Session>,
}
#[derive(Clone)]
pub(crate) struct ShardQueries {
    book: Arc<Mutex<Book>>,
    capacity: Arc<Semaphore>,
    runtime: Handle,
    incarnation: Uuid,
}

fn valid_uuid(bytes: &[u8]) -> bool {
    Uuid::from_slice(bytes).is_ok_and(|id| !id.is_nil())
}
fn validate_configuration(request: &ConfigureShardQueriesRequest) -> Result<(), Status> {
    if request.encoded_len() > 64 * 1024
        || request.shards.is_empty()
        || request.shards.len() > ShardQueryResponse::MAX_SHARDS
    {
        return Err(Status::invalid_argument(
            "bounded shard configuration required",
        ));
    }
    let local = request
        .local
        .as_ref()
        .ok_or_else(|| Status::invalid_argument("missing local registration"))?;
    if local.instance_id.is_empty() || local.tp_size == 0 || local.tp_size != local.world_size {
        return Err(Status::invalid_argument(
            "dense TP-only local registration required",
        ));
    }
    let mut endpoints = HashSet::new();
    let mut incarnations = HashSet::new();
    for shard in &request.shards {
        let target = shard
            .target
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("missing shard target"))?;
        let address: SocketAddr = shard
            .endpoint
            .strip_prefix("http://")
            .ok_or_else(|| Status::invalid_argument("loopback HTTP shard endpoint required"))?
            .parse()
            .map_err(|_| Status::invalid_argument("literal shard socket address required"))?;
        if !address.ip().is_loopback()
            || address.port() == 0
            || shard.endpoint != target.endpoint
            || shard.namespace != target.namespace
            || target.version != 1
            || target.instance_id != local.instance_id
            || target.tp_size != local.tp_size
            || target.world_size != local.world_size
            || target.storage_namespace.is_empty()
            || !valid_uuid(&target.capability)
            || !valid_uuid(&target.manager_incarnation)
            || !valid_uuid(&target.registration_generation)
            || !endpoints.insert(address)
            || !incarnations.insert(target.manager_incarnation.clone())
        {
            return Err(Status::invalid_argument(
                "mismatched, duplicate or unsupported shard registration",
            ));
        }
    }
    if request.shards[0].namespace != local.namespace {
        return Err(Status::invalid_argument("local shard namespace differs"));
    }
    Ok(())
}
impl ShardQueries {
    pub(crate) fn new(runtime: Handle) -> Self {
        let book = Arc::new(Mutex::new(Book::default()));
        let capacity = Arc::new(Semaphore::new(CAPACITY));
        let weak = Arc::downgrade(&capacity);
        opentelemetry::global::meter("orbitkv-server")
            .u64_observable_gauge("orbitkv_shard_query_active")
            .with_callback(move |observer| {
                if let Some(capacity) = weak.upgrade() {
                    observer.observe((CAPACITY - capacity.available_permits()) as u64, &[]);
                }
            })
            .build();
        let weak = Arc::downgrade(&book);
        opentelemetry::global::meter("orbitkv-server").u64_observable_gauge("orbitkv_shard_query_holds").with_callback(move |observer| {
            if let Some(book) = weak.upgrade() {
                let count = book.lock().sessions.values().flat_map(|session| session.operations.values()).filter(|op| op.result.lock().as_ref().is_some_and(|result| matches!(result, Ok(ready) if !ready.response.control_id.is_empty()))).count();
                observer.observe(count as u64, &[]);
            }
        }).build();
        let weak = Arc::downgrade(&book);
        opentelemetry::global::meter("orbitkv-server")
            .u64_observable_counter("orbitkv_shard_query_submissions_total")
            .with_callback(move |observer| {
                if let Some(book) = weak.upgrade() {
                    observer.observe(book.lock().submitted, &[]);
                }
            })
            .build();
        Self {
            book,
            capacity,
            runtime,
            incarnation: Uuid::new_v4(),
        }
    }
    pub(crate) fn configure(
        &self,
        token: u64,
        request: ConfigureShardQueriesRequest,
        local_control: &QueryControlService,
    ) -> Result<(), Status> {
        validate_configuration(&request)?;
        let local = request
            .local
            .clone()
            .ok_or_else(|| Status::invalid_argument("missing local registration"))?;
        if request.shards[0].target.as_ref() != Some(&local_control.export(local.clone())?) {
            return Err(Status::permission_denied(
                "first target is not this Manager registration",
            ));
        }
        let encoded = request.encode_to_vec();
        let mut book = self.book.lock();
        if book.closed {
            return Err(Status::unavailable("shard queries are stopping"));
        }
        if let Some(session) = book.sessions.get(&token) {
            return if session.configuration == encoded {
                Ok(())
            } else {
                Err(Status::failed_precondition(
                    "shard configuration is immutable for this connection",
                ))
            };
        }
        if book.sessions.len() >= CAPACITY {
            return Err(Status::resource_exhausted("shard session limit"));
        }
        let mut targets = Vec::with_capacity(request.shards.len());
        let _runtime = self.runtime.enter();
        for shard in request.shards {
            let endpoint = Endpoint::from_shared(shard.endpoint)
                .map_err(|error| Status::invalid_argument(error.to_string()))?
                .connect_timeout(RPC_TIMEOUT)
                .timeout(RPC_TIMEOUT);
            targets.push(Target {
                wire: shard
                    .target
                    .ok_or_else(|| Status::invalid_argument("missing target"))?,
                client: CacheQueryControlClient::new(endpoint.connect_lazy())
                    .max_decoding_message_size(128 * 1024)
                    .max_encoding_message_size(128 * 1024),
            });
        }
        book.sessions.insert(
            token,
            Session {
                instance: local.instance_id,
                configuration: encoded,
                targets,
                retired: 0,
                operations: HashMap::new(),
            },
        );
        Ok(())
    }
    pub(crate) fn execute(
        &self,
        token: u64,
        command: QueryCommand,
    ) -> Result<ShardQueryResponse, Status> {
        let encoded = command
            .encode()
            .map_err(|error| Status::invalid_argument(error.to_string()))?;
        if encoded.len() > 64 * 1024 {
            return Err(Status::resource_exhausted("shard query size limit"));
        }
        let mut book = self.book.lock();
        if book.closed {
            return Err(Status::unavailable("shard queries are stopping"));
        }
        let session = book
            .sessions
            .get_mut(&token)
            .ok_or_else(|| Status::failed_precondition("worker handshake is not configured"))?;
        let mut admitted = false;
        let ticket = match &command {
            QueryCommand::Submit(request) => request.ticket,
            QueryCommand::Poll(ticket) => *ticket,
            QueryCommand::Claim { .. } => {
                return Err(Status::invalid_argument(
                    "shard preparation claim is unsupported",
                ));
            }
        };
        if let QueryCommand::Submit(request) = &command {
            validate_query(request, &session.instance)?;
            let digest: [u8; 32] = Sha256::digest(&encoded).into();
            if let Some(previous) = session.operations.get(&ticket.operation_id) {
                if previous.ticket.revision > ticket.revision {
                    return Err(Status::failed_precondition("retired shard revision"));
                }
                if previous.ticket.revision == ticket.revision && previous.digest != digest {
                    return Err(Status::invalid_argument("query changed without revision"));
                }
            } else if ticket.operation_id <= session.retired {
                return Err(Status::failed_precondition("retired shard operation"));
            }
            if session
                .operations
                .get(&ticket.operation_id)
                .is_some_and(|previous| previous.ticket != ticket)
            {
                session.operations.remove(&ticket.operation_id);
            }
            if !session.operations.contains_key(&ticket.operation_id) {
                if session.operations.len() >= PER_SESSION {
                    return Ok(busy());
                }
                let permit = match Arc::clone(&self.capacity).try_acquire_owned() {
                    Ok(permit) => permit,
                    Err(_) => return Ok(busy()),
                };
                let canceled = Arc::new(AtomicBool::new(false));
                let result = Arc::new(Mutex::new(None));
                let transaction = Transaction {
                    nodes: session
                        .targets
                        .iter()
                        .cloned()
                        .map(|target| NodeInterest {
                            target,
                            interest: None,
                            ticket: QueryTicket {
                                operation_id: 1,
                                revision: 1,
                            },
                            reply: None,
                        })
                        .collect(),
                    permit: Some(permit),
                    runtime: self.runtime.clone(),
                };
                session.operations.insert(
                    ticket.operation_id,
                    Operation {
                        ticket,
                        digest,
                        canceled: Arc::clone(&canceled),
                        result: Arc::clone(&result),
                        expires: Instant::now() + HOLD_TIMEOUT,
                    },
                );
                admitted = true;
                let request = request.clone();
                let incarnation = self.incarnation;
                self.runtime.spawn(async move {
                    let ready = coordinate(transaction, request, &canceled, incarnation).await;
                    if !canceled.load(Ordering::Acquire) {
                        *result.lock() = Some(ready);
                    }
                });
            }
        }
        let operation = session
            .operations
            .get(&ticket.operation_id)
            .filter(|operation| operation.ticket == ticket)
            .ok_or_else(|| Status::not_found("unknown shard query revision"))?;
        let result = operation.result.lock();
        let response = match result.as_ref() {
            None => {
                drop(result);
                if admitted {
                    book.submitted = book.submitted.saturating_add(1);
                }
                return Ok(ShardQueryResponse::loading());
            }
            Some(Err(error)) => return Err(error.clone()),
            Some(Ok(ready)) => ready.response.clone(),
        };
        drop(result);
        if response.control_id.is_empty() {
            session.operations.remove(&ticket.operation_id);
            session.retired = session.retired.max(ticket.operation_id);
        }
        if admitted {
            book.submitted = book.submitted.saturating_add(1);
        }
        Ok(response)
    }
    pub(crate) fn cancel(&self, token: u64, ticket: QueryTicket) {
        if let Some(session) = self.book.lock().sessions.get_mut(&token) {
            if session
                .operations
                .get(&ticket.operation_id)
                .is_some_and(|operation| operation.ticket.revision <= ticket.revision)
            {
                session.operations.remove(&ticket.operation_id);
            }
            session.retired = session.retired.max(ticket.operation_id);
        }
    }
    pub(crate) fn release(&self, token: u64, control: &[u8]) -> Result<(), Status> {
        if !valid_uuid(control) {
            return Err(Status::invalid_argument("invalid shard control id"));
        }
        let mut book = self.book.lock();
        let session = book
            .sessions
            .get_mut(&token)
            .ok_or_else(|| Status::failed_precondition("shard session is closed"))?;
        let operation = session.operations.iter().find_map(|(id, operation)| {
            operation
                .result
                .lock()
                .as_ref()
                .and_then(|result| result.as_ref().ok())
                .filter(|ready| ready.response.control_id == control)
                .map(|_| *id)
        });
        if let Some(id) = operation {
            session.operations.remove(&id);
            session.retired = session.retired.max(id);
        }
        Ok(())
    }
    pub(crate) fn close_session(&self, token: u64) {
        self.book.lock().sessions.remove(&token);
    }
    pub(crate) fn retain_sessions(&self, live: impl Fn(u64) -> bool) {
        let mut book = self.book.lock();
        book.sessions.retain(|token, _| live(*token));
        let now = Instant::now();
        for session in book.sessions.values_mut() {
            let mut retired = session.retired;
            session.operations.retain(|id, op| {
                if op.expires <= now {
                    retired = retired.max(*id);
                    false
                } else {
                    true
                }
            });
            session.retired = retired;
        }
    }
    pub(crate) async fn stop_and_drain(&self) -> Result<(), Status> {
        {
            let mut book = self.book.lock();
            book.closed = true;
            book.sessions.clear();
        }
        let permits = Arc::clone(&self.capacity)
            .acquire_many_owned(CAPACITY as u32)
            .await
            .map_err(|_| Status::internal("shard capacity closed during drain"))?;
        drop(permits);
        Ok(())
    }
}
fn validate_query(query: &QueryBundleRequest, instance: &str) -> Result<(), Status> {
    if query.instance_id != instance
        || query.group_id != 0
        || query.warmup
        || query.prepare
        || query.discover
        || query.materialize
        || query.wait_for_full_prefix
        || query.demand.is_some()
        || query.block_hashes.len() > 4096
        || query.request_id.len() > 256
        || query
            .block_hashes
            .iter()
            .any(|hash| hash.is_empty() || hash.len() > 128)
    {
        return Err(Status::invalid_argument(
            "bounded dense prefix query required",
        ));
    }
    Ok(())
}
fn busy() -> ShardQueryResponse {
    ShardQueryResponse {
        outcome: QueryOutcomeCode::Busy,
        num_hit_blocks: 0,
        leases: Vec::new(),
        control_id: Vec::new(),
    }
}
fn check(canceled: &AtomicBool, deadline: Instant) -> Result<(), Status> {
    if canceled.load(Ordering::Acquire) {
        return Err(Status::cancelled("shard query canceled"));
    }
    if Instant::now() >= deadline {
        return Err(Status::deadline_exceeded("shard query deadline"));
    }
    Ok(())
}
impl NodeInterest {
    async fn close(&mut self) {
        if let Some(interest) = self.interest.take()
            && let Err(error) = self.target.client.close(interest).await
        {
            log::warn!(
                "Query interest close failed; source expiry retains the remaining bound: {error}"
            );
        }
    }
    async fn open_and_query(
        &mut self,
        request: &QueryBundleRequest,
        canceled: &AtomicBool,
        deadline: Instant,
        incarnation: Uuid,
    ) -> Result<(), Status> {
        check(canceled, deadline)?;
        let interest = self
            .target
            .client
            .open_interest(OpenQueryInterestRequest {
                target: Some(self.target.wire.clone()),
                coordinator_incarnation: incarnation.as_bytes().to_vec(),
            })
            .await?
            .into_inner();
        if !valid_uuid(&interest.id)
            || interest.manager_incarnation != self.target.wire.manager_incarnation
        {
            return Err(Status::data_loss("source returned mismatched interest"));
        }
        self.interest = Some(interest);
        self.query(request, canceled, deadline).await
    }
    async fn query(
        &mut self,
        request: &QueryBundleRequest,
        canceled: &AtomicBool,
        deadline: Instant,
    ) -> Result<(), Status> {
        let mut request = request.clone();
        request.ticket = self.ticket;
        let mut command = QueryCommand::Submit(request.clone());
        loop {
            check(canceled, deadline)?;
            let response = self
                .target
                .client
                .execute(QueryControlExecuteRequest {
                    interest: self.interest.clone(),
                    command: command
                        .encode()
                        .map_err(|error| Status::invalid_argument(error.to_string()))?,
                })
                .await?
                .into_inner();
            let reply = QueryBundleResponse::decode(&response.payload)
                .map_err(|error| Status::data_loss(error.to_string()))?;
            match reply.outcome {
                QueryOutcomeCode::Loading => {
                    command = QueryCommand::Poll(self.ticket);
                    tokio::time::sleep(Duration::from_millis(2)).await;
                }
                QueryOutcomeCode::Busy => {
                    return Err(Status::resource_exhausted("source query busy"));
                }
                QueryOutcomeCode::Ready
                    if reply.num_hit_blocks <= request.block_hashes.len() as u64
                        && reply.hit_positions.is_empty()
                        && ((reply.num_hit_blocks == 0 && reply.lease.is_empty())
                            || (reply.num_hit_blocks > 0 && valid_uuid(&reply.lease))) =>
                {
                    self.reply = Some(reply);
                    return Ok(());
                }
                _ => return Err(Status::data_loss("invalid dense source response")),
            }
        }
    }
    async fn shorten(
        &mut self,
        request: &QueryBundleRequest,
        common: u64,
        canceled: &AtomicBool,
        deadline: Instant,
    ) -> Result<(), Status> {
        if self
            .reply
            .as_ref()
            .is_some_and(|reply| reply.num_hit_blocks == common)
        {
            return Ok(());
        }
        check(canceled, deadline)?;
        self.target
            .client
            .cancel(QueryControlCancelRequest {
                interest: self.interest.clone(),
                operation_id: self.ticket.operation_id,
                revision: self.ticket.revision,
            })
            .await?;
        self.ticket.operation_id = 2;
        self.reply = None;
        let mut request = request.clone();
        request.block_hashes.truncate(common as usize);
        self.query(&request, canceled, deadline).await?;
        if !self
            .reply
            .as_ref()
            .is_some_and(|reply| reply.num_hit_blocks == common)
        {
            return Err(Status::failed_precondition(
                "source prefix changed during common-prefix acquisition",
            ));
        }
        Ok(())
    }
    async fn claim(&mut self, canceled: &AtomicBool, deadline: Instant) -> Result<(), Status> {
        check(canceled, deadline)?;
        let reply = self
            .reply
            .as_ref()
            .ok_or_else(|| Status::internal("missing selected reply"))?;
        let request = QueryControlClaimRequest {
            interest: self.interest.clone(),
            operation_id: self.ticket.operation_id,
            revision: self.ticket.revision,
            lease: reply.lease.clone(),
        };
        let response = match self.target.client.claim(request.clone()).await {
            Ok(response) => response,
            Err(error)
                if matches!(
                    error.code(),
                    Code::Unavailable | Code::DeadlineExceeded | Code::Unknown
                ) =>
            {
                check(canceled, deadline)?;
                self.target.client.claim(request).await?
            }
            Err(error) => return Err(error),
        };
        let claimed = QueryBundleResponse::decode(&response.into_inner().payload)
            .map_err(|error| Status::data_loss(error.to_string()))?;
        if claimed != *reply {
            return Err(Status::data_loss("claimed source reply differs"));
        }
        Ok(())
    }
}
async fn coordinate(
    mut transaction: Transaction,
    request: QueryBundleRequest,
    canceled: &AtomicBool,
    incarnation: Uuid,
) -> Result<Ready, Status> {
    let deadline = Instant::now() + QUERY_TIMEOUT;
    let result: Result<ShardQueryResponse, Status> = async {
        let initial = join_all(
            transaction
                .nodes
                .iter_mut()
                .map(|node| node.open_and_query(&request, canceled, deadline, incarnation)),
        )
        .await;
        for result in initial {
            result?;
        }
        let common = transaction
            .nodes
            .iter()
            .filter_map(|node| node.reply.as_ref().map(|reply| reply.num_hit_blocks))
            .min()
            .ok_or_else(|| Status::internal("empty shard query"))?;
        if common == 0 {
            return Ok(ShardQueryResponse {
                outcome: QueryOutcomeCode::Ready,
                num_hit_blocks: 0,
                leases: vec![Vec::new(); transaction.nodes.len()],
                control_id: Vec::new(),
            });
        }
        let shortened = join_all(
            transaction
                .nodes
                .iter_mut()
                .map(|node| node.shorten(&request, common, canceled, deadline)),
        )
        .await;
        for result in shortened {
            result?;
        }
        let claimed = join_all(
            transaction
                .nodes
                .iter_mut()
                .map(|node| node.claim(canceled, deadline)),
        )
        .await;
        for result in claimed {
            result?;
        }
        check(canceled, deadline)?;
        Ok(ShardQueryResponse {
            outcome: QueryOutcomeCode::Ready,
            num_hit_blocks: common,
            leases: transaction
                .nodes
                .iter()
                .filter_map(|node| node.reply.as_ref().map(|reply| reply.lease.clone()))
                .collect(),
            control_id: Uuid::new_v4().as_bytes().to_vec(),
        })
    }
    .await;
    match result {
        Ok(response) if !response.control_id.is_empty() => Ok(Ready {
            response,
            _transaction: Some(transaction),
        }),
        result => {
            transaction.close().await;
            match result {
                Ok(response) => Ok(Ready {
                    response,
                    _transaction: None,
                }),
                Err(error) if error.code() == Code::ResourceExhausted => Ok(Ready {
                    response: busy(),
                    _transaction: None,
                }),
                Err(error) => Err(error),
            }
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/cache/shards.rs"]
mod tests;
