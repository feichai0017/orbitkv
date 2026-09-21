//! Polling cache queries without blocking the shared iceoryx2 dispatcher.
use std::collections::HashMap;
use std::future::{Future, poll_fn};
use std::sync::{Arc, Mutex};
use std::task::Poll;
use std::time::{Duration, Instant};

use orbitkv_channel::QueryBundleRequest;
use orbitkv_common::hll::MultiWindowHllTracker;
use orbitkv_core::{EngineError, OrbitKVEngine};
use tokio::runtime::Handle;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, oneshot};

use crate::cache::operations::{QueryInput, QueryOutcome, execute_query, execute_release};

const MAX_PENDING_PER_SESSION: usize = 128;
const MAX_ACTIVE_QUERIES: usize = 1024;
const QUERY_TIMEOUT: Duration = Duration::from_secs(60);
type QueryKey = (u64, String, String, u32);

/// An undelivered result owns its lease, including while queued in a channel.
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
pub(crate) struct PendingQueries {
    pending: HashMap<QueryKey, PendingQuery>,
    sessions: HashMap<u64, Arc<Semaphore>>,
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
impl PendingQueries {
    pub(crate) fn retain_sessions(&mut self, live: impl Fn(u64) -> bool) {
        let now = Instant::now();
        self.pending.retain(|(token, ..), task| {
            if task.expires <= now {
                // Keep a bounded tombstone so a late poll cannot restart I/O.
                task.receiver = None;
            }
            live(*token)
        });
        self.sessions.retain(|token, _| live(*token));
    }

    pub(crate) fn cancel(&mut self, token: u64, instance: &str, request: &str, group: u32) {
        self.pending
            .remove(&(token, instance.into(), request.into(), group));
    }

    pub(crate) fn poll(
        &mut self,
        token: u64,
        request: QueryBundleRequest,
        engine: &Arc<OrbitKVEngine>,
        runtime: &Handle,
        hll: &Arc<Mutex<MultiWindowHllTracker>>,
    ) -> Result<Option<QueryReply>, EngineError> {
        let key = (
            token,
            request.instance_id.clone(),
            request.request_id.clone(),
            request.group_id,
        );
        if let Some(pending) = self.pending.get_mut(&key) {
            if pending.request != request {
                return Err(EngineError::InvalidArgument(
                    "query parameters changed while pending".into(),
                ));
            }
            if pending.expires <= Instant::now() || pending.receiver.is_none() {
                self.pending.remove(&key);
                return Err(EngineError::Storage("cache query timed out".into()));
            }
            match pending.receiver.as_mut().expect("checked above").try_recv() {
                Ok(reply) => {
                    self.pending.remove(&key);
                    return Ok(Some(reply));
                }
                Err(oneshot::error::TryRecvError::Empty) => return Ok(None),
                Err(oneshot::error::TryRecvError::Closed) => {
                    self.pending.remove(&key);
                    return Err(EngineError::Storage("cache query task closed".into()));
                }
            }
        }
        if self
            .pending
            .keys()
            .filter(|(owner, ..)| *owner == token)
            .count()
            >= MAX_PENDING_PER_SESSION
        {
            return Ok(None);
        }
        // Cancellation releases result ownership, but not capacity occupied by
        // an in-flight read. The worker drains safely before returning permits.
        let session = self
            .sessions
            .entry(token)
            .or_insert_with(|| Arc::new(Semaphore::new(MAX_PENDING_PER_SESSION)));
        let mut permits = Vec::with_capacity(2);
        for capacity in [Arc::clone(session), Arc::clone(&self.capacity)] {
            let Ok(permit) = capacity.try_acquire_owned() else {
                // Queue pressure is retryable. No work or result is retained
                // until admitted; the adapter's waiting deadline still applies.
                return Ok(None);
            };
            permits.push(permit);
        }
        let input = QueryInput {
            instance_id: request.instance_id.clone(),
            block_hashes: request.block_hashes.clone(),
            request_id: request.request_id.clone(),
            wait_for_full_prefix: request.wait_for_full_prefix,
            group_id: request.group_id,
        };
        let engine = Arc::clone(engine);
        let hll = Arc::clone(hll);
        let query = async move {
            let outcome = execute_query(&engine, &hll, input).await;
            QueryReply {
                outcome,
                engine,
                delivered: false,
                _permits: permits,
            }
        };
        Ok(self.start_query(key, request, query, runtime))
    }

    fn start_query(
        &mut self,
        key: QueryKey,
        request: QueryBundleRequest,
        query: impl Future<Output = QueryReply> + Send + 'static,
        runtime: &Handle,
    ) -> Option<QueryReply> {
        let mut query = Box::pin(query);
        // Resident hits and validation errors usually finish in this first poll.
        // Any actual wait continues on Tokio, freeing the shared dispatcher.
        match runtime.block_on(poll_fn(|cx| Poll::Ready(query.as_mut().poll(cx)))) {
            Poll::Ready(reply) => Some(reply),
            Poll::Pending => {
                let (sender, receiver) = oneshot::channel();
                self.pending.insert(
                    key,
                    PendingQuery {
                        request,
                        receiver: Some(receiver),
                        expires: Instant::now() + QUERY_TIMEOUT,
                    },
                );
                runtime.spawn(async move {
                    // A disconnected client drops the result and releases its lease.
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

    #[test]
    fn cancellation_and_disconnect_drain_work_without_reusing_results() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let engine = Arc::new(
            OrbitKVEngine::new_with_config(1 << 20, false, StorageConfig::default()).unwrap(),
        );
        let capacity = Arc::new(Semaphore::new(4));
        let mut queries = PendingQueries::default();
        let mut completions = Vec::new();
        // Same caller request ID across session, instance, and group boundaries.
        for (token, instance, group) in [(1, "a", 0), (2, "a", 0), (1, "b", 0), (1, "a", 1)] {
            let request = QueryBundleRequest {
                instance_id: instance.into(),
                request_id: "same".into(),
                block_hashes: vec![vec![group as u8]],
                group_id: group,
                wait_for_full_prefix: false,
            };
            let (complete, wait) = oneshot::channel::<()>();
            completions.push(complete);
            let permit = Arc::clone(&capacity).try_acquire_owned().unwrap();
            let engine = Arc::clone(&engine);
            assert!(
                queries
                    .start_query(
                        (token, instance.into(), "same".into(), group),
                        request,
                        async move {
                            wait.await.unwrap();
                            QueryReply {
                                outcome: Ok(QueryOutcome::Ready {
                                    num_hit_blocks: 0,
                                    lease: Vec::new(),
                                    hit_positions: Vec::new(),
                                }),
                                engine,
                                delivered: false,
                                _permits: vec![permit],
                            }
                        },
                        runtime.handle()
                    )
                    .is_none()
            );
        }
        queries.cancel(1, "a", "same", 0);
        queries.cancel(1, "a", "same", 0); // Idempotent.
        assert_eq!(queries.pending.len(), 3);
        assert!(
            queries
                .pending
                .contains_key(&(2, "a".into(), "same".into(), 0))
        );
        assert_eq!(
            capacity.available_permits(),
            0,
            "cancel must not claim I/O completed"
        );
        queries.retain_sessions(|_| false);
        assert!(queries.pending.is_empty());
        for complete in completions {
            complete.send(()).unwrap();
        }
        runtime.block_on(async {
            let _permit = tokio::time::timeout(Duration::from_secs(2), capacity.acquire_many(4))
                .await
                .unwrap()
                .unwrap();
        });
        assert_eq!(
            capacity.available_permits(),
            4,
            "late replies must release resources without polling"
        );
    }

    #[test]
    fn expired_result_is_not_restarted_by_a_late_poll() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let engine = Arc::new(
            OrbitKVEngine::new_with_config(1 << 20, false, StorageConfig::default()).unwrap(),
        );
        let hll = Arc::new(Mutex::new(MultiWindowHllTracker::new(
            vec![("test".into(), Duration::from_secs(60))],
            4,
        )));
        let request = QueryBundleRequest {
            instance_id: "a".into(),
            request_id: "expired".into(),
            block_hashes: vec![],
            group_id: 0,
            wait_for_full_prefix: false,
        };
        let (_sender, receiver) = oneshot::channel();
        let mut queries = PendingQueries::default();
        queries.pending.insert(
            (1, "a".into(), "expired".into(), 0),
            PendingQuery {
                request: request.clone(),
                receiver: Some(receiver),
                expires: Instant::now(),
            },
        );
        queries.retain_sessions(|_| true);
        let error = queries
            .poll(1, request, &engine, runtime.handle(), &hll)
            .err()
            .unwrap();
        assert!(error.to_string().contains("timed out"));
        assert!(queries.pending.is_empty());
    }

    #[test]
    fn waiting_query_yields_and_preserves_its_identity_until_completed() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let engine = Arc::new(
            OrbitKVEngine::new_with_config(1 << 20, false, StorageConfig::default()).unwrap(),
        );
        let hll = Arc::new(Mutex::new(MultiWindowHllTracker::new(
            vec![("test".into(), Duration::from_secs(60))],
            4,
        )));
        let request = QueryBundleRequest {
            instance_id: "inst".into(),
            request_id: "slow".into(),
            block_hashes: vec![],
            wait_for_full_prefix: true,
            group_id: 0,
        };
        let mut queries = PendingQueries::default();
        let (resume, gate) = oneshot::channel::<()>();
        let result_engine = Arc::clone(&engine);
        assert!(
            queries
                .start_query(
                    (1, "inst".into(), "slow".into(), 0),
                    request.clone(),
                    async move {
                        gate.await.unwrap();
                        QueryReply {
                            outcome: Ok(QueryOutcome::Loading),
                            engine: result_engine,
                            delivered: false,
                            _permits: Vec::new(),
                        }
                    },
                    runtime.handle()
                )
                .is_none()
        );
        assert!(
            queries
                .poll(1, request.clone(), &engine, runtime.handle(), &hll)
                .unwrap()
                .is_none()
        );
        let mut changed = request.clone();
        changed.block_hashes.push(vec![1]);
        assert!(
            queries
                .poll(1, changed, &engine, runtime.handle(), &hll)
                .is_err()
        );
        // An unrelated query must execute immediately while the first waits.
        let mut other = request.clone();
        other.request_id = "other".into();
        let reply = queries
            .poll(1, other, &engine, runtime.handle(), &hll)
            .unwrap()
            .unwrap();
        assert!(matches!(
            reply.outcome,
            Err(EngineError::InstanceMissing(_))
        ));
        drop(reply);
        let capacity = Arc::clone(&queries.capacity)
            .try_acquire_many_owned(MAX_ACTIVE_QUERIES as u32)
            .unwrap();
        let mut waiting_for_capacity = request.clone();
        waiting_for_capacity.request_id = "capacity".into();
        assert!(
            queries
                .poll(
                    1,
                    waiting_for_capacity.clone(),
                    &engine,
                    runtime.handle(),
                    &hll
                )
                .unwrap()
                .is_none()
        );
        assert_eq!(
            queries.pending.len(),
            1,
            "backpressure must not retain extra work"
        );
        drop(capacity);
        assert!(
            queries
                .poll(1, waiting_for_capacity, &engine, runtime.handle(), &hll)
                .unwrap()
                .is_some()
        );
        resume.send(()).unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if let Some(reply) = queries
                .poll(1, request.clone(), &engine, runtime.handle(), &hll)
                .unwrap()
            {
                assert!(matches!(reply.outcome, Ok(QueryOutcome::Loading)));
                break;
            }
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(queries.pending.is_empty());
    }
}
