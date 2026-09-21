//! Own versioned query operations without blocking the process dispatcher.
use std::collections::HashMap;
use std::future::{Future, poll_fn};
use std::sync::{Arc, Mutex};
use std::task::Poll;
use std::time::{Duration, Instant};

use orbitkv_channel::{QueryBundleRequest, QueryCommand, QueryTicket};
use orbitkv_common::hll::MultiWindowHllTracker;
use orbitkv_core::{EngineError, OrbitKVEngine, QueryAdmission, QueryOwner};
use tokio::runtime::Handle;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, oneshot};

use crate::cache::operations::{QueryInput, QueryOutcome, execute_query, execute_release};

const MAX_PENDING_PER_SESSION: usize = 128;
const MAX_ACTIVE_QUERIES: usize = 1024;
const QUERY_TIMEOUT: Duration = Duration::from_secs(60);
type QueryKey = (u64, u64);

pub(crate) struct QueryReply {
    pub(crate) outcome: Result<QueryOutcome, EngineError>,
    engine: Arc<OrbitKVEngine>,
    delivered: bool,
    _permits: Vec<OwnedSemaphorePermit>,
}
impl QueryReply {
    pub(crate) fn delivered(&mut self) {
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
}
struct Session {
    last_operation: u64,
    capacity: Arc<Semaphore>,
}
impl Default for Session {
    fn default() -> Self {
        Self {
            last_operation: 0,
            capacity: Arc::new(Semaphore::new(MAX_PENDING_PER_SESSION)),
        }
    }
}
pub(crate) struct PendingQueries {
    pending: HashMap<QueryKey, PendingQuery>,
    sessions: HashMap<u64, Session>,
    capacity: Arc<Semaphore>,
}
impl Default for PendingQueries {
    fn default() -> Self {
        Self {
            pending: HashMap::new(),
            sessions: HashMap::new(),
            capacity: Arc::new(Semaphore::new(MAX_ACTIVE_QUERIES)),
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
            if task.expires <= now {
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
        {
            self.pending.remove(&key);
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
            QueryCommand::Submit(request) => {
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
        if let Some(receiver) = task.receiver.as_mut() {
            return match receiver.try_recv() {
                Ok(reply) => {
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
        let request = task.request.clone();
        let session = self
            .sessions
            .get(&token)
            .ok_or_else(|| invalid("unknown query session"))?;
        let mut permits = Vec::with_capacity(2);
        for capacity in [Arc::clone(&session.capacity), Arc::clone(&self.capacity)] {
            let Ok(permit) = capacity.try_acquire_owned() else {
                return Ok(None);
            };
            permits.push(permit);
        }
        let admission = engine.reserve_query(
            &request.instance_id,
            request.group_id,
            request.block_hashes.len(),
        );
        let reservation = match admission {
            Ok(QueryAdmission::Busy) => return Ok(None),
            Ok(QueryAdmission::Admitted(reservation)) => reservation,
            result => {
                self.pending.remove(&key);
                return Ok(Some(QueryReply {
                    outcome: match result {
                        Err(error) => Err(error),
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
        let input = QueryInput {
            instance_id: request.instance_id,
            block_hashes: request.block_hashes,
            request_id: request.request_id,
            wait_for_full_prefix: request.wait_for_full_prefix,
            group_id: request.group_id,
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
        self.pending.insert(
            (token, request.ticket.operation_id),
            PendingQuery {
                request,
                receiver: None,
                expires: Instant::now() + QUERY_TIMEOUT,
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
mod tests {
    use super::*;
    use orbitkv_core::StorageConfig;

    fn request(operation_id: u64, revision: u64) -> QueryBundleRequest {
        QueryBundleRequest {
            ticket: QueryTicket {
                operation_id,
                revision,
            },
            instance_id: "model".into(),
            request_id: "same-request".into(),
            block_hashes: vec![vec![revision as u8]],
            group_id: 0,
            wait_for_full_prefix: false,
        }
    }

    fn engine() -> Arc<OrbitKVEngine> {
        Arc::new(OrbitKVEngine::new_with_config(1 << 20, false, StorageConfig::default()).unwrap())
    }

    fn tracker() -> Arc<Mutex<MultiWindowHllTracker>> {
        Arc::new(Mutex::new(MultiWindowHllTracker::new(
            vec![("test".into(), Duration::from_secs(60))],
            4,
        )))
    }

    #[test]
    fn revisions_and_session_teardown_drain_old_work_without_restarting_it() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let engine = engine();
        let hll = tracker();
        let mut queries = PendingQueries::default();
        let capacity = Arc::clone(&queries.capacity);
        let held = Arc::clone(&capacity)
            .try_acquire_many_owned((MAX_ACTIVE_QUERIES - 2) as u32)
            .unwrap();
        let mut completions = Vec::new();
        for token in [1, 2] {
            queries.sessions.entry(token).or_default().last_operation = 1;
            queries.insert(token, request(1, 1));
            let (complete, wait) = oneshot::channel();
            completions.push(complete);
            let result_engine = Arc::clone(&engine);
            let permit = Arc::clone(&capacity).try_acquire_owned().unwrap();
            assert!(
                queries
                    .start_query(
                        (token, 1),
                        async move {
                            wait.await.unwrap();
                            QueryReply {
                                outcome: Ok(QueryOutcome::Ready {
                                    num_hit_blocks: 0,
                                    lease: vec![],
                                    hit_positions: vec![],
                                }),
                                engine: result_engine,
                                delivered: false,
                                _permits: vec![permit],
                            }
                        },
                        runtime.handle()
                    )
                    .is_none()
            );
        }
        let old = request(1, 1).ticket;
        let new = request(1, 2).ticket;
        assert!(
            queries
                .execute(
                    1,
                    QueryCommand::Submit(request(1, 2)),
                    &engine,
                    runtime.handle(),
                    &hll
                )
                .unwrap()
                .is_none()
        );
        queries.cancel(1, old, &engine);
        assert_eq!(queries.pending[&(1, 1)].request.ticket, new);
        assert_eq!(queries.pending[&(2, 1)].request.ticket, old);
        for (session, ticket) in [(1, old), (9, new)] {
            assert!(
                queries
                    .execute(
                        session,
                        QueryCommand::Poll(ticket),
                        &engine,
                        runtime.handle(),
                        &hll
                    )
                    .is_err()
            );
        }
        assert_eq!(
            capacity.available_permits(),
            0,
            "superseding work does not finish its I/O"
        );
        queries.cancel(1, new, &engine);
        assert!(
            queries
                .execute(
                    1,
                    QueryCommand::Submit(request(1, 2)),
                    &engine,
                    runtime.handle(),
                    &hll
                )
                .is_err()
        );
        queries.retain_sessions(&engine, |_| false);
        for completion in completions {
            completion.send(()).unwrap();
        }
        runtime.block_on(async {
            let _done = tokio::time::timeout(Duration::from_secs(2), capacity.acquire_many(2))
                .await
                .unwrap()
                .unwrap();
        });
        assert_eq!(capacity.available_permits(), 2);
        assert!(queries.pending.is_empty());
        drop(held);
    }

    #[test]
    fn expiration_and_capacity_pressure_never_turn_a_poll_into_submission() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let engine = engine();
        let hll = tracker();
        let mut queries = PendingQueries::default();
        queries.sessions.entry(1).or_default().last_operation = MAX_PENDING_PER_SESSION as u64;
        for id in 1..=MAX_PENDING_PER_SESSION as u64 {
            queries.insert(1, request(id, 1));
        }
        let next = request(MAX_PENDING_PER_SESSION as u64 + 1, 1);
        let reply = queries
            .execute(
                1,
                QueryCommand::Submit(next.clone()),
                &engine,
                runtime.handle(),
                &hll,
            )
            .unwrap()
            .unwrap();
        assert!(matches!(reply.outcome, Ok(QueryOutcome::Busy)));
        assert!(
            queries
                .execute(
                    1,
                    QueryCommand::Poll(next.ticket),
                    &engine,
                    runtime.handle(),
                    &hll
                )
                .is_err()
        );
        queries.pending.get_mut(&(1, 1)).unwrap().expires = Instant::now();
        queries.retain_sessions(&engine, |_| true);
        let error = queries
            .execute(
                1,
                QueryCommand::Poll(request(1, 1).ticket),
                &engine,
                runtime.handle(),
                &hll,
            )
            .err()
            .unwrap();
        assert!(error.to_string().contains("timed out"));
        assert!(
            queries
                .execute(
                    1,
                    QueryCommand::Submit(request(1, 2)),
                    &engine,
                    runtime.handle(),
                    &hll
                )
                .is_err()
        );
    }
}
