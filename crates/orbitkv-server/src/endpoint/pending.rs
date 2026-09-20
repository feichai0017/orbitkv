//! Polling cache queries without blocking the shared iceoryx2 dispatcher.
use std::collections::HashMap;
use std::future::{Future, poll_fn};
use std::sync::{Arc, Mutex};
use std::task::Poll;
use std::time::{Duration, Instant};

use orbitkv_common::hll::MultiWindowHllTracker;
use orbitkv_core::{EngineError, OrbitKVEngine};
use orbitkv_local::QueryBundleRequest;
use tokio::runtime::Handle;
use tokio::sync::oneshot;

use crate::cache::operations::{QueryInput, QueryOutcome, execute_query, execute_release};

const MAX_PENDING_PER_SESSION: usize = 128;
const QUERY_TIMEOUT: Duration = Duration::from_secs(60);
type QueryKey = (u64, String, String, u32);

/// An undelivered result owns its lease, including while queued in a channel.
pub(crate) struct QueryReply {
    pub(crate) outcome: Result<QueryOutcome, EngineError>,
    engine: Arc<OrbitKVEngine>,
    delivered: bool,
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
    receiver: oneshot::Receiver<QueryReply>,
    expires: Instant,
}
#[derive(Default)]
pub(crate) struct PendingQueries {
    pending: HashMap<QueryKey, PendingQuery>,
}
impl PendingQueries {
    pub(crate) fn retain_sessions(&mut self, live: impl Fn(u64) -> bool) {
        let now = Instant::now();
        self.pending
            .retain(|(token, ..), task| live(*token) && task.expires > now);
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
            match pending.receiver.try_recv() {
                Ok(reply) => {
                    self.pending.remove(&key);
                    return Ok(Some(reply));
                }
                Err(oneshot::error::TryRecvError::Empty) => return Ok(None),
                Err(oneshot::error::TryRecvError::Closed) => {
                    self.pending.remove(&key);
                    return Err(EngineError::Storage("local query task closed".into()));
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
            return Err(EngineError::InvalidArgument(
                "too many pending local queries".into(),
            ));
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
            let outcome = tokio::time::timeout(QUERY_TIMEOUT, execute_query(&engine, &hll, input))
                .await
                .unwrap_or_else(|_| Err(EngineError::Storage("local query timed out".into())));
            QueryReply {
                outcome,
                engine,
                delivered: false,
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
                        receiver,
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
