//! Bounded network query interests in the same admission owner as local IPC.
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use orbitkv_channel::{QueryBundleResponse, QueryCommand, QueryOutcomeCode, QueryTicket};
use orbitkv_core::OrbitKVEngine;
use orbitkv_proto::proto::engine::cache_query_control_server::CacheQueryControl;
use orbitkv_proto::proto::engine::{
    OpenQueryInterestRequest, QueryControlCancelRequest, QueryControlClaimRequest,
    QueryControlEmpty, QueryControlExecuteRequest, QueryControlResponse, QueryInterest,
    RegisteredQueryTarget, SessionRequest,
};
use parking_lot::Mutex as BookMutex;
use prost::Message;
use sha2::{Digest, Sha256};
use tokio::runtime::Handle;
use tonic::{Request, Response, Status};
use uuid::Uuid;

use super::operations::QueryOutcome;
use super::pending::{PendingQueries, QueryReply};
use crate::metric::hll::MultiWindowHllTracker;

const MAX_TARGETS: usize = 1024;
const MAX_INTERESTS: usize = 1024;
const MAX_INTERESTS_PER_TARGET: usize = 128;
const MAX_OPERATIONS: usize = 128;
const MAX_COMMAND_BYTES: usize = 64 * 1024;
const INTEREST_TIMEOUT: Duration = Duration::from_secs(60);

struct RetainedReply {
    reply: QueryReply,
    payload: Vec<u8>,
    expires: Instant,
}
enum Submission {
    Active([u8; 32]),
    Canceled,
}
struct Interest {
    target: RegisteredQueryTarget,
    token: u64,
    expires: Instant,
    submitted: HashMap<(u64, u64), Submission>,
    replies: HashMap<(u64, u64), RetainedReply>,
}
#[derive(Default)]
struct ControlBook {
    targets: HashMap<Uuid, RegisteredQueryTarget>,
    interests: HashMap<Uuid, Interest>,
    next_token: u64,
    closed: bool,
}

#[derive(Clone)]
pub(crate) struct QueryControlService {
    pub(crate) queries: Arc<BookMutex<PendingQueries>>,
    engine: Arc<OrbitKVEngine>,
    hll: Arc<Mutex<MultiWindowHllTracker>>,
    runtime: Handle,
    endpoint: String,
    incarnation: Uuid,
    book: Arc<BookMutex<ControlBook>>,
}

fn uuid(bytes: &[u8], name: &str) -> Result<Uuid, Status> {
    let id = Uuid::from_slice(bytes).map_err(|_| Status::invalid_argument(name))?;
    if id.is_nil() {
        return Err(Status::invalid_argument(name));
    }
    Ok(id)
}
fn encode_outcome(reply: &QueryReply) -> Result<Vec<u8>, Status> {
    let outcome = reply
        .outcome
        .as_ref()
        .map_err(|error| Status::failed_precondition(error.to_string()))?;
    let response = match outcome {
        QueryOutcome::Ready {
            num_hit_blocks,
            lease,
            hit_positions,
        } => QueryBundleResponse {
            outcome: QueryOutcomeCode::Ready,
            num_hit_blocks: *num_hit_blocks,
            lease: lease.clone(),
            hit_positions: hit_positions.clone(),
        },
        QueryOutcome::Busy => QueryBundleResponse {
            outcome: QueryOutcomeCode::Busy,
            num_hit_blocks: 0,
            lease: Vec::new(),
            hit_positions: Vec::new(),
        },
        _ => return Err(Status::internal("unsupported registered query outcome")),
    };
    response
        .encode()
        .map_err(|error| Status::internal(error.to_string()))
}
impl Interest {
    fn cancel(&mut self, ticket: QueryTicket) -> Result<(), Status> {
        let key = (ticket.operation_id, ticket.revision);
        if !self.submitted.contains_key(&key) && self.submitted.len() >= MAX_OPERATIONS {
            return Err(Status::resource_exhausted("query operation limit"));
        }
        self.submitted.insert(key, Submission::Canceled);
        self.replies.remove(&key);
        Ok(())
    }

    fn admit(&mut self, command: &QueryCommand, encoded: &[u8]) -> Result<QueryTicket, Status> {
        match command {
            QueryCommand::Submit(query) => {
                if query.instance_id != self.target.instance_id
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
                        "query control accepts only bounded dense prefixes",
                    ));
                }
                let key = (query.ticket.operation_id, query.ticket.revision);
                let digest: [u8; 32] = Sha256::digest(encoded).into();
                match self.submitted.get(&key) {
                    Some(Submission::Canceled) => {
                        return Err(Status::failed_precondition("query operation canceled"));
                    }
                    Some(Submission::Active(previous)) if *previous != digest => {
                        return Err(Status::invalid_argument(
                            "query changed without new revision",
                        ));
                    }
                    Some(_) => {}
                    None => {
                        if self.submitted.iter().any(|((operation, _), state)| {
                            *operation == query.ticket.operation_id
                                && matches!(state, Submission::Canceled)
                        }) {
                            return Err(Status::failed_precondition("query operation canceled"));
                        }
                        if self.submitted.len() >= MAX_OPERATIONS {
                            return Err(Status::resource_exhausted("query operation limit"));
                        }
                        if self
                            .replies
                            .keys()
                            .any(|(operation, _)| *operation == query.ticket.operation_id)
                        {
                            return Err(Status::failed_precondition(
                                "finished operation requires a new id",
                            ));
                        }
                        self.submitted.insert(key, Submission::Active(digest));
                    }
                }
                Ok(query.ticket)
            }
            QueryCommand::Poll(ticket) => {
                if matches!(
                    self.submitted.get(&(ticket.operation_id, ticket.revision)),
                    Some(Submission::Canceled)
                ) {
                    return Err(Status::failed_precondition("query operation canceled"));
                }
                Ok(*ticket)
            }
            QueryCommand::Claim { .. } => Err(Status::invalid_argument(
                "preparation claim is not network delivery",
            )),
        }
    }
}

impl QueryControlService {
    pub(crate) fn new(
        endpoint: String,
        engine: Arc<OrbitKVEngine>,
        hll: Arc<Mutex<MultiWindowHllTracker>>,
        runtime: Handle,
    ) -> Self {
        Self {
            queries: Arc::new(BookMutex::new(PendingQueries::default())),
            engine,
            hll,
            runtime,
            endpoint,
            incarnation: Uuid::new_v4(),
            book: Arc::default(),
        }
    }

    fn registration_matches(&self, target: &RegisteredQueryTarget) -> bool {
        self.engine
            .has_query_registration(&target.instance_id, &target.registration_generation)
    }

    pub(crate) fn export(&self, request: SessionRequest) -> Result<RegisteredQueryTarget, Status> {
        let registration = self
            .engine
            .query_registration(&request.instance_id)
            .map_err(|error| Status::failed_precondition(error.to_string()))?;
        if request.namespace != registration.namespace
            || request.tp_size as usize != registration.tp_size
            || request.world_size as usize != registration.world_size
        {
            return Err(Status::failed_precondition(
                "query target registration differs",
            ));
        }
        let mut book = self.book.lock();
        self.retire(&mut book);
        if book.closed {
            return Err(Status::unavailable("query control is closed"));
        }
        if let Some(target) = book.targets.values().find(|target| {
            target.instance_id == request.instance_id
                && target.registration_generation == registration.generation
        }) {
            return Ok(target.clone());
        }
        if book.targets.len() >= MAX_TARGETS {
            return Err(Status::resource_exhausted("registered query target limit"));
        }
        let capability = Uuid::new_v4();
        let target = RegisteredQueryTarget {
            version: 1,
            endpoint: self.endpoint.clone(),
            instance_id: request.instance_id,
            namespace: registration.namespace,
            storage_namespace: registration.storage_namespace,
            tp_size: u32::try_from(registration.tp_size)
                .map_err(|_| Status::internal("tp size"))?,
            world_size: u32::try_from(registration.world_size)
                .map_err(|_| Status::internal("world size"))?,
            manager_incarnation: self.incarnation.as_bytes().to_vec(),
            registration_generation: registration.generation.to_vec(),
            capability: capability.as_bytes().to_vec(),
        };
        if target.encoded_len() > orbitkv_channel::lifecycle::MAX_QUERY_TARGET_PAYLOAD {
            return Err(Status::resource_exhausted("query target metadata limit"));
        }
        book.targets.insert(capability, target.clone());
        Ok(target)
    }

    fn retire(&self, book: &mut ControlBook) {
        book.targets
            .retain(|_, target| self.registration_matches(target));
        let now = Instant::now();
        let mut queries = self.queries.lock();
        book.interests.retain(|_, interest| {
            let live = !book.closed
                && interest.expires > now
                && self.registration_matches(&interest.target);
            if !live {
                queries.close_session(interest.token, &self.engine);
            } else {
                interest.replies.retain(|(operation, revision), reply| {
                    if reply.expires <= now {
                        interest
                            .submitted
                            .insert((*operation, *revision), Submission::Canceled);
                        queries.cancel(
                            interest.token,
                            QueryTicket {
                                operation_id: *operation,
                                revision: *revision,
                            },
                            &self.engine,
                        );
                        false
                    } else {
                        true
                    }
                });
            }
            live
        });
    }

    pub(crate) fn start_reaper(&self) {
        let control = self.clone();
        self.runtime.spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(250));
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                interval.tick().await;
                let mut book = control.book.lock();
                control.retire(&mut book);
                if book.closed {
                    break;
                }
            }
        });
    }

    pub(crate) fn stop(&self) {
        let mut book = self.book.lock();
        book.closed = true;
        self.retire(&mut book);
        book.targets.clear();
    }

    fn interest<'a>(
        &self,
        book: &'a mut ControlBook,
        wire: Option<QueryInterest>,
    ) -> Result<&'a mut Interest, Status> {
        let wire = wire.ok_or_else(|| Status::invalid_argument("missing interest"))?;
        if wire.manager_incarnation != self.incarnation.as_bytes() {
            return Err(Status::failed_precondition("retired Manager incarnation"));
        }
        let id = uuid(&wire.id, "invalid interest id")?;
        if book.closed
            || book.interests.get(&id).is_some_and(|interest| {
                interest.expires <= Instant::now() || !self.registration_matches(&interest.target)
            })
        {
            if let Some(interest) = book.interests.remove(&id) {
                self.queries
                    .lock()
                    .close_session(interest.token, &self.engine);
            }
            return Err(Status::failed_precondition(
                "retired query interest registration",
            ));
        }
        let interest = book
            .interests
            .get_mut(&id)
            .ok_or_else(|| Status::not_found("unknown or retired query interest"))?;
        interest.expires = Instant::now() + INTEREST_TIMEOUT;
        Ok(interest)
    }
}

#[tonic::async_trait]
impl CacheQueryControl for QueryControlService {
    async fn open_interest(
        &self,
        request: Request<OpenQueryInterestRequest>,
    ) -> Result<Response<QueryInterest>, Status> {
        let request = request.into_inner();
        uuid(
            &request.coordinator_incarnation,
            "invalid coordinator incarnation",
        )?;
        let target = request
            .target
            .ok_or_else(|| Status::invalid_argument("missing target"))?;
        let capability = uuid(&target.capability, "invalid capability")?;
        let mut book = self.book.lock();
        self.retire(&mut book);
        if book.closed {
            return Err(Status::unavailable("query control is closed"));
        }
        if book.targets.get(&capability) != Some(&target) {
            return Err(Status::permission_denied(
                "retired or mismatched query capability",
            ));
        }
        if book.interests.len() >= MAX_INTERESTS
            || book
                .interests
                .values()
                .filter(|interest| interest.target.capability == target.capability)
                .count()
                >= MAX_INTERESTS_PER_TARGET
        {
            return Err(Status::resource_exhausted("query interest limit"));
        }
        // Bootstrap session tokens are odd; remote interests use only even tokens.
        book.next_token = book
            .next_token
            .checked_add(2)
            .ok_or_else(|| Status::resource_exhausted("query token space exhausted"))?;
        let token = book.next_token;
        let id = Uuid::new_v4();
        book.interests.insert(
            id,
            Interest {
                target,
                token,
                expires: Instant::now() + INTEREST_TIMEOUT,
                submitted: HashMap::new(),
                replies: HashMap::new(),
            },
        );
        Ok(Response::new(QueryInterest {
            id: id.as_bytes().to_vec(),
            manager_incarnation: self.incarnation.as_bytes().to_vec(),
        }))
    }

    async fn execute(
        &self,
        request: Request<QueryControlExecuteRequest>,
    ) -> Result<Response<QueryControlResponse>, Status> {
        let request = request.into_inner();
        if request.command.len() > MAX_COMMAND_BYTES {
            return Err(Status::resource_exhausted("query command limit"));
        }
        let command = QueryCommand::decode(&request.command)
            .map_err(|error| Status::invalid_argument(error.to_string()))?;
        let mut book = self.book.lock();
        let interest = self.interest(&mut book, request.interest)?;
        let ticket = interest.admit(&command, &request.command)?;
        let key = (ticket.operation_id, ticket.revision);
        if interest
            .replies
            .get(&key)
            .is_some_and(|reply| reply.expires <= Instant::now())
        {
            interest.cancel(ticket)?;
            self.queries
                .lock()
                .cancel(interest.token, ticket, &self.engine);
            return Err(Status::failed_precondition("query result expired"));
        }
        if let Some(reply) = interest.replies.get(&key) {
            return Ok(Response::new(QueryControlResponse {
                payload: reply.payload.clone(),
            }));
        }
        if !interest.submitted.contains_key(&key) {
            return Err(Status::not_found("unknown query revision"));
        }
        let reply = self
            .queries
            .lock()
            .execute(
                interest.token,
                command,
                &self.engine,
                &self.runtime,
                &self.hll,
            )
            .map_err(|error| Status::failed_precondition(error.to_string()))?;
        let payload = match reply {
            None => QueryBundleResponse::loading()
                .encode()
                .map_err(|error| Status::internal(error.to_string()))?,
            Some(reply) => {
                // A tonic response enqueue does not deliver a cache lease.
                let payload = encode_outcome(&reply)?;
                interest.replies.insert(
                    key,
                    RetainedReply {
                        reply,
                        payload: payload.clone(),
                        expires: Instant::now() + INTEREST_TIMEOUT,
                    },
                );
                payload
            }
        };
        Ok(Response::new(QueryControlResponse { payload }))
    }

    async fn claim(
        &self,
        request: Request<QueryControlClaimRequest>,
    ) -> Result<Response<QueryControlResponse>, Status> {
        let request = request.into_inner();
        let mut book = self.book.lock();
        let interest = self.interest(&mut book, request.interest)?;
        let key = (request.operation_id, request.revision);
        if interest
            .replies
            .get(&key)
            .is_some_and(|reply| reply.expires <= Instant::now())
        {
            let ticket = QueryTicket {
                operation_id: request.operation_id,
                revision: request.revision,
            };
            interest.cancel(ticket)?;
            self.queries
                .lock()
                .cancel(interest.token, ticket, &self.engine);
            return Err(Status::failed_precondition("query result expired"));
        }
        let reply = interest
            .replies
            .get_mut(&key)
            .ok_or_else(|| Status::not_found("query reply not retained"))?;
        match &reply.reply.outcome {
            Ok(QueryOutcome::Ready { lease, .. }) if *lease == request.lease => {}
            _ => {
                return Err(Status::failed_precondition(
                    "claim requires the exact ready lease",
                ));
            }
        }
        reply.reply.delivered();
        Ok(Response::new(QueryControlResponse {
            payload: reply.payload.clone(),
        }))
    }

    async fn cancel(
        &self,
        request: Request<QueryControlCancelRequest>,
    ) -> Result<Response<QueryControlEmpty>, Status> {
        let request = request.into_inner();
        let ticket = QueryTicket {
            operation_id: request.operation_id,
            revision: request.revision,
        };
        if ticket.operation_id == 0 || ticket.revision == 0 {
            return Err(Status::invalid_argument("invalid query ticket"));
        }
        let mut book = self.book.lock();
        let interest = self.interest(&mut book, request.interest)?;
        interest.cancel(ticket)?;
        self.queries
            .lock()
            .cancel(interest.token, ticket, &self.engine);
        Ok(Response::new(QueryControlEmpty {}))
    }

    async fn close(
        &self,
        request: Request<QueryInterest>,
    ) -> Result<Response<QueryControlEmpty>, Status> {
        let wire = request.into_inner();
        if wire.manager_incarnation != self.incarnation.as_bytes() {
            return Err(Status::failed_precondition("retired Manager incarnation"));
        }
        let id = uuid(&wire.id, "invalid interest id")?;
        let mut book = self.book.lock();
        if let Some(interest) = book.interests.remove(&id) {
            self.queries
                .lock()
                .close_session(interest.token, &self.engine);
        }
        Ok(Response::new(QueryControlEmpty {}))
    }
}

#[cfg(test)]
#[path = "../../tests/unit/cache/query_control.rs"]
mod tests;
