//! Own versioned query operations without blocking the process dispatcher.
use std::collections::HashMap;
use std::future::{Future, poll_fn};
use std::sync::{Arc, Mutex};
use std::task::Poll;
use std::time::{Duration, Instant};

use crate::metric::hll::MultiWindowHllTracker;
use orbitkv_channel::{QueryBundleRequest, QueryCommand, QueryTicket};
use orbitkv_core::{
    EngineError, OrbitKVEngine, QueryAdmission, QueryLeaseId, QueryMode, QueryOwner,
};
use tokio::runtime::Handle;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, oneshot};

use crate::cache::operations::{
    QueryInput, QueryOutcome, execute_query, execute_release, record_prefix_reuse,
};
use crate::cache::read::ReadControl;

const MAX_PENDING_PER_SESSION: usize = 128;
const MAX_ACTIVE_QUERIES: usize = 1024;
const MAX_WARMUPS_PER_SESSION: usize = 16;
const MAX_ACTIVE_WARMUPS: usize = 128;
const QUERY_TIMEOUT: Duration = Duration::from_secs(60);
const WARMUP_TIMEOUT: Duration = Duration::from_secs(5);
const PREPARATION_TTL: Duration = Duration::from_secs(1);
const MAX_PREPARATIONS_PER_SESSION: usize = 4;
type QueryKey = (u64, u64);

pub(crate) struct QueryReply {
    pub(crate) outcome: Result<QueryOutcome, EngineError>,
    engine: Arc<OrbitKVEngine>,
    delivered: bool,
    _permits: Vec<OwnedSemaphorePermit>,
}
impl QueryReply {
    pub(crate) fn delivered(&mut self) {
        if let Ok(QueryOutcome::Ready { lease, .. }) = &self.outcome
            && let Ok(lease) = QueryLeaseId::from_bytes(lease)
        {
            self.engine.claim_query(&lease);
        }
        self.delivered = true;
    }
}
impl Drop for QueryReply {
    fn drop(&mut self) {
        if !self.delivered
            && let Ok(QueryOutcome::Ready { lease, .. }) = &self.outcome
            && !lease.is_empty()
        {
            let _ = execute_release(&self.engine, lease);
        }
    }
}

struct PendingQuery {
    request: QueryBundleRequest,
    receiver: Option<oneshot::Receiver<QueryReply>>,
    expires: Instant,
    control: Arc<ReadControl>,
    count_claim: bool,
}
impl Drop for PendingQuery {
    fn drop(&mut self) {
        self.control.cancel();
    }
}
struct Session {
    last_operation: u64,
    capacity: Arc<Semaphore>,
    warming: Arc<Semaphore>,
    preparation: Arc<Semaphore>,
}
impl Default for Session {
    fn default() -> Self {
        Self {
            last_operation: 0,
            capacity: Arc::new(Semaphore::new(MAX_PENDING_PER_SESSION)),
            warming: Arc::new(Semaphore::new(MAX_WARMUPS_PER_SESSION)),
            preparation: Arc::new(Semaphore::new(MAX_PREPARATIONS_PER_SESSION)),
        }
    }
}
pub(crate) struct PendingQueries {
    pending: HashMap<QueryKey, PendingQuery>,
    sessions: HashMap<u64, Session>,
    capacity: Arc<Semaphore>,
    warming: Arc<Semaphore>,
    pub(crate) read_batch_bytes: u64,
    pub(crate) read_timeout: Option<Duration>,
    pub(crate) read_max_batches: usize,
}
impl Default for PendingQueries {
    fn default() -> Self {
        Self {
            pending: HashMap::new(),
            sessions: HashMap::new(),
            capacity: Arc::new(Semaphore::new(MAX_ACTIVE_QUERIES)),
            warming: Arc::new(Semaphore::new(MAX_ACTIVE_WARMUPS)),
            read_batch_bytes: 0,
            read_timeout: None,
            read_max_batches: usize::MAX,
        }
    }
}
fn owner(session: u64, ticket: QueryTicket) -> QueryOwner {
    QueryOwner {
        session,
        operation: ticket.operation_id,
        revision: ticket.revision,
    }
}
fn invalid(message: &str) -> EngineError {
    EngineError::InvalidArgument(message.into())
}
impl PendingQueries {
    pub(crate) fn retain_sessions(&mut self, engine: &OrbitKVEngine, live: impl Fn(u64) -> bool) {
        let now = Instant::now();
        self.pending.retain(|(token, _), task| {
            if task.request.prepare && task.expires <= now {
                return false;
            }
            if task.request.warmup
                && (task.expires <= now
                    || task
                        .receiver
                        .as_ref()
                        .is_some_and(|receiver| !receiver.is_empty()))
            {
                return false;
            }
            if task.expires <= now {
                task.control.cancel();
                task.receiver = None;
            }
            live(*token)
        });
        self.sessions.retain(|token, _| {
            if live(*token) {
                true
            } else {
                engine.release_query_session(*token);
                false
            }
        });
    }

    pub(crate) fn cancel(&mut self, token: u64, ticket: QueryTicket, engine: &OrbitKVEngine) {
        let key = (token, ticket.operation_id);
        if self
            .pending
            .get(&key)
            .is_some_and(|task| task.request.ticket == ticket)
            && let Some(task) = self.pending.remove(&key)
        {
            crate::metric::timeline::record("query_cancel", || {
                serde_json::json!({
                    "request_id": task.request.request_id,
                    "operation_id": ticket.operation_id, "revision": ticket.revision,
                })
            });
        }
        engine.cancel_query(owner(token, ticket));
    }

    pub(crate) fn execute(
        &mut self,
        token: u64,
        command: QueryCommand,
        engine: &Arc<OrbitKVEngine>,
        runtime: &Handle,
        hll: &Arc<Mutex<MultiWindowHllTracker>>,
    ) -> Result<Option<QueryReply>, EngineError> {
        let ticket = match command {
            QueryCommand::Poll(ticket) => ticket,
            QueryCommand::Claim {
                ticket,
                count_lookup,
            } => {
                let key = (token, ticket.operation_id);
                if self
                    .pending
                    .get(&key)
                    .is_some_and(|task| task.expires <= Instant::now())
                {
                    self.pending.remove(&key);
                }
                let Some(task) = self.pending.get_mut(&(token, ticket.operation_id)) else {
                    return Ok(Some(QueryReply {
                        outcome: Ok(QueryOutcome::Busy),
                        engine: Arc::clone(engine),
                        delivered: false,
                        _permits: Vec::new(),
                    }));
                };
                if task.request.ticket != ticket || !task.request.prepare {
                    return Err(invalid("claim requires the current preparation revision"));
                }
                if count_lookup && task.request.group_id != 0 {
                    return Err(invalid(
                        "only an attention-prefix claim counts a logical lookup",
                    ));
                }
                task.count_claim = count_lookup;
                task.request.prepare = false;
                task.expires = Instant::now() + QUERY_TIMEOUT;
                ticket
            }
            QueryCommand::Submit(request) => {
                if request.prepare
                    && (!request.materialize
                        || request.warmup
                        || request.discover
                        || request.wait_for_full_prefix)
                {
                    return Err(invalid(
                        "preparation requires a selected, non-waiting recovery read",
                    ));
                }
                if request.materialize
                    && (request.discover || request.warmup || request.wait_for_full_prefix)
                {
                    return Err(invalid(
                        "recovery reads cannot discover, warm or wait for publication",
                    ));
                }
                if request.discover && (request.warmup || request.wait_for_full_prefix) {
                    return Err(invalid("discovery cannot warm or wait for publication"));
                }
                if request.warmup && (request.group_id != 0 || request.wait_for_full_prefix) {
                    return Err(invalid("warmup requires a non-waiting attention prefix"));
                }
                let ticket = request.ticket;
                let key = (token, ticket.operation_id);
                if let Some(pending) = self.pending.get(&key) {
                    if ticket.revision < pending.request.ticket.revision {
                        return Err(invalid("stale query revision"));
                    }
                    if ticket.revision == pending.request.ticket.revision {
                        if request != pending.request {
                            return Err(invalid("query parameters changed without a new revision"));
                        }
                    } else {
                        if request.instance_id != pending.request.instance_id
                            || request.request_id != pending.request.request_id
                            || request.group_id != pending.request.group_id
                        {
                            return Err(invalid(
                                "query operation cannot change its owner or storage group",
                            ));
                        }
                        self.cancel(token, pending.request.ticket, engine);
                        self.insert(token, request);
                    }
                } else {
                    let session = self.sessions.entry(token).or_default();
                    if ticket.operation_id <= session.last_operation {
                        return Err(invalid("query operation is already retired"));
                    }
                    if self.pending.len() >= MAX_ACTIVE_QUERIES
                        || self
                            .pending
                            .keys()
                            .filter(|(owner, _)| *owner == token)
                            .count()
                            >= MAX_PENDING_PER_SESSION
                    {
                        return Ok(Some(QueryReply {
                            outcome: Ok(QueryOutcome::Busy),
                            engine: Arc::clone(engine),
                            delivered: false,
                            _permits: Vec::new(),
                        }));
                    }
                    session.last_operation = ticket.operation_id;
                    self.insert(token, request);
                }
                ticket
            }
        };
        let key = (token, ticket.operation_id);
        let task = self
            .pending
            .get_mut(&key)
            .ok_or_else(|| invalid("query operation is unknown or retired"))?;
        if task.request.ticket != ticket {
            return Err(invalid("stale query revision"));
        }
        if task.expires <= Instant::now() {
            self.pending.remove(&key);
            return Err(EngineError::Storage("cache query timed out".into()));
        }
        if !task.request.wait_for_full_prefix && !task.request.discover && task.control.expired() {
            crate::metric::timeline::record("read_deadline", || {
                serde_json::json!({
                    "request_id": task.request.request_id,
                    "operation_id": ticket.operation_id, "revision": ticket.revision,
                })
            });
            // Retire caller interest immediately. The spawned query keeps all
            // submitted buffers and permits until its current batch completes.
            self.pending.remove(&key);
            return Ok(Some(QueryReply {
                outcome: Ok(QueryOutcome::Ready {
                    num_hit_blocks: 0,
                    lease: Vec::new(),
                    hit_positions: Vec::new(),
                }),
                engine: Arc::clone(engine),
                delivered: false,
                _permits: Vec::new(),
            }));
        }
        if let Some(receiver) = task.receiver.as_mut() {
            return match receiver.try_recv() {
                Ok(reply) => {
                    if task.count_claim
                        && task.control.max_batches == usize::MAX
                        && let Ok(QueryOutcome::Ready { num_hit_blocks, .. }) = &reply.outcome
                    {
                        record_prefix_reuse(
                            engine,
                            hll,
                            &task.request.instance_id,
                            &task.request.block_hashes,
                            *num_hit_blocks as usize,
                        );
                    }
                    self.pending.remove(&key);
                    Ok(Some(reply))
                }
                Err(oneshot::error::TryRecvError::Empty) => Ok(None),
                Err(oneshot::error::TryRecvError::Closed) => {
                    self.pending.remove(&key);
                    Err(EngineError::Storage("cache query task closed".into()))
                }
            };
        }
        let request = &task.request;
        let warmup = request.warmup;
        let preparing = request.prepare;
        let speculative = warmup || preparing;
        let session = self
            .sessions
            .get(&token)
            .ok_or_else(|| invalid("unknown query session"))?;
        let mut permits = Vec::with_capacity(4);
        let mut capacities = vec![Arc::clone(&session.capacity), Arc::clone(&self.capacity)];
        if speculative {
            capacities.extend([Arc::clone(&session.warming), Arc::clone(&self.warming)]);
        }
        if preparing {
            capacities.push(Arc::clone(&session.preparation));
        }
        for capacity in capacities {
            let Ok(permit) = capacity.try_acquire_owned() else {
                if speculative {
                    self.pending.remove(&key);
                    return Ok(Some(QueryReply {
                        outcome: Ok(QueryOutcome::Busy),
                        engine: Arc::clone(engine),
                        delivered: false,
                        _permits: permits,
                    }));
                }
                return Ok(None);
            };
            permits.push(permit);
        }
        let reservation = if request.discover {
            None
        } else {
            let admission = engine.reserve_query(
                &request.instance_id,
                request.group_id,
                request.block_hashes.len(),
                if preparing {
                    QueryMode::Prepare
                } else if warmup {
                    QueryMode::Warmup
                } else {
                    QueryMode::Demand
                },
            );
            let reservation = match admission {
                Ok(QueryAdmission::Busy) if !speculative => return Ok(None),
                Ok(QueryAdmission::Admitted(reservation)) => reservation,
                result => {
                    self.pending.remove(&key);
                    return Ok(Some(QueryReply {
                        outcome: match result {
                            Err(error) => Err(error),
                            Ok(QueryAdmission::Busy | QueryAdmission::TooLarge) if speculative => {
                                Ok(QueryOutcome::Busy)
                            }
                            Ok(QueryAdmission::TooLarge) => Ok(QueryOutcome::Ready {
                                num_hit_blocks: 0,
                                lease: Vec::new(),
                                hit_positions: Vec::new(),
                            }),
                            _ => unreachable!(),
                        },
                        engine: Arc::clone(engine),
                        delivered: false,
                        _permits: permits,
                    }));
                }
            };
            Some(reservation)
        };
        let request = request.clone();
        let input = QueryInput {
            instance_id: request.instance_id,
            block_hashes: request.block_hashes,
            request_id: request.request_id,
            wait_for_full_prefix: request.wait_for_full_prefix,
            group_id: request.group_id,
            warmup: request.warmup,
            discover: request.discover,
            materialize: request.materialize,
            prepare: request.prepare,
            control: Arc::clone(&task.control),
        };
        let engine = Arc::clone(engine);
        let hll = Arc::clone(hll);
        let query = async move {
            let outcome =
                execute_query(&engine, &hll, input, reservation, owner(token, ticket)).await;
            QueryReply {
                outcome,
                engine,
                delivered: false,
                _permits: permits,
            }
        };
        Ok(self.start_query(key, query, runtime))
    }

    fn insert(&mut self, token: u64, request: QueryBundleRequest) {
        let timeout = if request.prepare {
            PREPARATION_TTL
        } else if request.warmup {
            WARMUP_TIMEOUT
        } else {
            QUERY_TIMEOUT
        };
        let expires = Instant::now() + timeout;
        let deadline = if request.wait_for_full_prefix {
            expires
        } else {
            self.read_timeout.map_or(expires, |timeout| {
                expires.min(Instant::now() + timeout.min(QUERY_TIMEOUT))
            })
        };
        let batch_bytes = if request.prepare {
            if self.read_batch_bytes == 0 {
                32 << 20
            } else {
                self.read_batch_bytes.min(32 << 20)
            }
        } else {
            self.read_batch_bytes
        };
        self.pending.insert(
            (token, request.ticket.operation_id),
            PendingQuery {
                request,
                receiver: None,
                expires,
                control: Arc::new(ReadControl::new(
                    deadline,
                    batch_bytes,
                    self.read_max_batches,
                )),
                count_claim: false,
            },
        );
    }

    fn start_query(
        &mut self,
        key: QueryKey,
        query: impl Future<Output = QueryReply> + Send + 'static,
        runtime: &Handle,
    ) -> Option<QueryReply> {
        let mut query = Box::pin(query);
        match runtime.block_on(poll_fn(|cx| Poll::Ready(query.as_mut().poll(cx)))) {
            Poll::Ready(reply) => {
                if self.pending[&key].request.prepare {
                    let (sender, receiver) = oneshot::channel();
                    let _ = sender.send(reply);
                    self.pending
                        .get_mut(&key)
                        .expect("registered query")
                        .receiver = Some(receiver);
                    return None;
                }
                self.pending.remove(&key);
                Some(reply)
            }
            Poll::Pending => {
                let (sender, receiver) = oneshot::channel();
                self.pending
                    .get_mut(&key)
                    .expect("registered query")
                    .receiver = Some(receiver);
                runtime.spawn(async move {
                    // Cancellation drops interest, never a submitted I/O future.
                    let _ = sender.send(query.await);
                });
                None
            }
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/endpoint/pending.rs"]
mod tests;
