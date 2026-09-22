//! Request ownership shared by all framework adapters.

use std::collections::HashMap;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::{
    CallOptions, CancelQueryRequest, ChannelClient, ChannelError, PublishRequest,
    QueryBundleRequest, QueryBundleResponse, QueryCommand, QueryOutcomeCode, QueryTicket,
    RestoreRequest, RestoreResponse, RestoreState,
};

const MAX_WARMUPS: usize = 16;
const WARMUP_TTL: Duration = Duration::from_secs(5);
const COMPLETION_POLL: Duration = Duration::from_millis(50);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct QueryKey {
    instance: String,
    request: String,
    group: u32,
}

struct PendingQuery {
    ticket: QueryTicket,
    hashes: BlockHashes,
    wait_for_full_prefix: bool,
}

/// Immutable query hashes with shared, allocation-free prefix views.
/// Reusing a batch lets pending queries compare identity in constant time.
#[derive(Clone, Debug)]
pub struct BlockHashes {
    hashes: Arc<[Vec<u8>]>,
    range: Range<usize>,
}

impl BlockHashes {
    pub fn new(hashes: Vec<Vec<u8>>) -> Self {
        let range = 0..hashes.len();
        Self {
            hashes: hashes.into(),
            range,
        }
    }

    pub fn as_slice(&self) -> &[Vec<u8>] {
        &self.hashes[self.range.clone()]
    }

    pub fn slice(&self, range: Range<usize>) -> Option<Self> {
        if range.start > range.end || range.end > self.range.len() {
            return None;
        }
        Some(Self {
            hashes: Arc::clone(&self.hashes),
            range: self.range.start + range.start..self.range.start + range.end,
        })
    }
}

impl PartialEq for BlockHashes {
    fn eq(&self, other: &Self) -> bool {
        (Arc::ptr_eq(&self.hashes, &other.hashes) && self.range == other.range)
            || self.as_slice() == other.as_slice()
    }
}
impl Eq for BlockHashes {}

#[derive(Default)]
struct Queries {
    next_operation: u64,
    pending: HashMap<QueryKey, PendingQuery>,
    warmups: HashMap<QueryKey, (QueryTicket, Instant)>,
}

impl Queries {
    fn ticket(&mut self) -> Result<QueryTicket, ChannelError> {
        self.next_operation = self
            .next_operation
            .checked_add(1)
            .ok_or(ChannelError::SessionRequiresReconnect)?;
        Ok(QueryTicket {
            operation_id: self.next_operation,
            revision: 1,
        })
    }

    fn prepare(
        &mut self,
        key: &QueryKey,
        hashes: &BlockHashes,
        wait: bool,
    ) -> Result<QueryCommand, ChannelError> {
        if let Some(query) = self.pending.get_mut(key) {
            if query.hashes == *hashes && query.wait_for_full_prefix == wait {
                return Ok(QueryCommand::Poll(query.ticket));
            }
            query.ticket.revision = query
                .ticket
                .revision
                .checked_add(1)
                .ok_or(ChannelError::SessionRequiresReconnect)?;
            query.hashes = hashes.clone();
            query.wait_for_full_prefix = wait;
        } else {
            let ticket = self.ticket()?;
            self.pending.insert(
                key.clone(),
                PendingQuery {
                    ticket,
                    hashes: hashes.clone(),
                    wait_for_full_prefix: wait,
                },
            );
        }
        let query = &self.pending[key];
        Ok(QueryCommand::Submit(QueryBundleRequest {
            ticket: query.ticket,
            instance_id: key.instance.clone(),
            request_id: key.request.clone(),
            block_hashes: query.hashes.as_slice().to_vec(),
            group_id: key.group,
            wait_for_full_prefix: wait,
            warmup: false,
        }))
    }

    fn complete(&mut self, key: &QueryKey, outcome: QueryOutcomeCode) {
        if outcome != QueryOutcomeCode::Loading {
            self.pending.remove(key);
        }
    }
}

/// A GPU restore remains owned after a wait deadline. Only a terminal reply
/// or confirmed Manager death permits the engine to reuse its destinations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RestoreHandle {
    pub operation_id: u64,
    pub session_epoch: u64,
    owner: u64,
}

/// Owns query revisions, warming interests, and independent publish/restore
/// sessions. It does not allocate engine GPU pages or determine cache locality.
pub struct CacheClient {
    channel: ChannelClient,
    socket: PathBuf,
    options: CallOptions,
    publisher: Mutex<Option<Arc<ChannelClient>>>,
    closed: AtomicBool,
    requests: AtomicU64,
    queries: Mutex<Queries>,
    last_completion_poll: Mutex<Instant>,
    owner: u64,
}

impl CacheClient {
    pub fn connect(socket: impl AsRef<Path>, options: CallOptions) -> Result<Self, ChannelError> {
        static OWNERS: AtomicU64 = AtomicU64::new(1);
        let owner = next_id(&OWNERS)?;
        Ok(Self {
            channel: ChannelClient::connect(&socket, options)?,
            socket: socket.as_ref().to_path_buf(),
            options,
            publisher: Mutex::new(None),
            closed: AtomicBool::new(false),
            requests: AtomicU64::new(1),
            queries: Mutex::new(Queries::default()),
            last_completion_poll: Mutex::new(Instant::now()),
            owner,
        })
    }

    pub fn channel(&self) -> &ChannelClient {
        &self.channel
    }
    pub fn socket(&self) -> &Path {
        &self.socket
    }

    pub fn close(&self) {
        self.closed.store(true, Ordering::Release);
        self.channel.close();
        if let Ok(publisher) = self.publisher.lock()
            && let Some(publisher) = publisher.as_ref()
        {
            publisher.close();
        }
        if let Ok(mut queries) = self.queries.lock() {
            queries.pending.clear();
            queries.warmups.clear();
        }
    }

    pub fn query_prefetch(
        &self,
        instance: &str,
        hashes: &BlockHashes,
        request: &str,
        wait: bool,
        group: u32,
    ) -> Result<QueryBundleResponse, ChannelError> {
        let key = QueryKey {
            instance: instance.into(),
            request: request.into(),
            group,
        };
        let mut queries = self
            .queries
            .lock()
            .map_err(|_| ChannelError::SessionRequiresReconnect)?;
        if let Some((ticket, _)) = queries.warmups.remove(&key) {
            self.cancel(ticket)?;
        }
        let previous = queries.pending.get(&key).map(|query| query.ticket);
        let command = queries.prepare(&key, hashes, wait)?;
        let response = self
            .channel
            .query_bundle(next_id(&self.requests)?, &command);
        match response {
            Ok(response) => {
                queries.complete(&key, response.outcome);
                Ok(response)
            }
            Err(error) => {
                // Failure before submission can leave the prior revision alive;
                // failure after submission can leave the replacement alive.
                if let Some(query) = queries.pending.remove(&key) {
                    let _ = self.cancel(query.ticket);
                    if let Some(previous) = previous.filter(|ticket| *ticket != query.ticket) {
                        let _ = self.cancel(previous);
                    }
                }
                Err(error)
            }
        }
    }

    pub fn warm_prefix(
        &self,
        instance: &str,
        hashes: &BlockHashes,
        request: &str,
    ) -> Result<bool, ChannelError> {
        if hashes.as_slice().is_empty() {
            return Ok(false);
        }
        let key = QueryKey {
            instance: instance.into(),
            request: request.into(),
            group: 0,
        };
        let mut queries = self
            .queries
            .lock()
            .map_err(|_| ChannelError::SessionRequiresReconnect)?;
        let now = Instant::now();
        let expired: Vec<_> = queries
            .warmups
            .iter()
            .filter(|(_, (_, submitted))| now.duration_since(*submitted) >= WARMUP_TTL)
            .map(|(key, _)| key.clone())
            .collect();
        for key in expired {
            if let Some((ticket, _)) = queries.warmups.remove(&key) {
                self.cancel(ticket)?;
            }
        }
        if queries.warmups.contains_key(&key)
            || queries.pending.contains_key(&key)
            || queries.warmups.len() >= MAX_WARMUPS
        {
            return Ok(false);
        }
        let ticket = queries.ticket()?;
        let response = self.channel.query_bundle(
            next_id(&self.requests)?,
            &QueryCommand::Submit(QueryBundleRequest {
                ticket,
                instance_id: instance.into(),
                request_id: request.into(),
                block_hashes: hashes.as_slice().to_vec(),
                group_id: 0,
                wait_for_full_prefix: false,
                warmup: true,
            }),
        )?;
        match response.outcome {
            QueryOutcomeCode::Loading => {
                queries.warmups.insert(key, (ticket, now));
                Ok(true)
            }
            QueryOutcomeCode::Busy => Ok(false),
            QueryOutcomeCode::Ready
                if response.num_hit_blocks == 0 && response.lease.is_empty() =>
            {
                Ok(true)
            }
            QueryOutcomeCode::Ready => {
                self.channel.close();
                Err(ChannelError::SessionRequiresReconnect)
            }
        }
    }

    pub fn cancel_query(
        &self,
        instance: &str,
        request: &str,
        group: u32,
    ) -> Result<(), ChannelError> {
        let key = QueryKey {
            instance: instance.into(),
            request: request.into(),
            group,
        };
        let mut queries = self
            .queries
            .lock()
            .map_err(|_| ChannelError::SessionRequiresReconnect)?;
        if let Some((ticket, _)) = queries.warmups.remove(&key) {
            self.cancel(ticket)?;
        }
        if let Some(query) = queries.pending.remove(&key) {
            self.cancel(query.ticket)?;
        }
        Ok(())
    }

    fn cancel(&self, ticket: QueryTicket) -> Result<(), ChannelError> {
        self.channel
            .cancel_query(next_id(&self.requests)?, &CancelQueryRequest { ticket })
    }

    pub fn release(&self, lease: Vec<u8>) -> Result<(), ChannelError> {
        self.channel.release(next_id(&self.requests)?, lease)
    }

    pub fn publish(&self, request: &PublishRequest) -> Result<(), ChannelError> {
        // A long D2H publish must not hold the query/restore descriptor slot.
        let publisher = {
            let mut publisher = self
                .publisher
                .lock()
                .map_err(|_| ChannelError::SessionRequiresReconnect)?;
            if self.closed.load(Ordering::Acquire) {
                return Err(ChannelError::SessionRequiresReconnect);
            }
            if publisher.is_none() {
                *publisher = Some(Arc::new(ChannelClient::connect(
                    &self.socket,
                    self.options,
                )?));
            }
            Arc::clone(
                publisher
                    .as_ref()
                    .ok_or(ChannelError::SessionRequiresReconnect)?,
            )
        };
        publisher.publish(next_id(&self.requests)?, request)
    }

    pub fn start_restore(&self, request: &RestoreRequest) -> Result<RestoreHandle, ChannelError> {
        let operation_id = self
            .channel
            .restore_submit(next_id(&self.requests)?, request)?;
        Ok(RestoreHandle {
            operation_id,
            session_epoch: self.channel.session_epoch(),
            owner: self.owner,
        })
    }

    pub fn poll_restore(&self, handle: RestoreHandle) -> Result<RestoreResponse, ChannelError> {
        if handle.owner != self.owner || handle.session_epoch != self.channel.session_epoch() {
            return Err(ChannelError::SessionRequiresReconnect);
        }
        self.channel
            .restore_poll(next_id(&self.requests)?, handle.operation_id)
    }

    pub fn restore_completions_ready(&self, timeout: Duration) -> Result<bool, ChannelError> {
        let mut last = self
            .last_completion_poll
            .lock()
            .map_err(|_| ChannelError::SessionRequiresReconnect)?;
        let remaining = COMPLETION_POLL.saturating_sub(last.elapsed());
        if remaining.is_zero()
            || self.channel.wait_for_notification(timeout.min(remaining))?
            || last.elapsed() >= COMPLETION_POLL
        {
            *last = Instant::now();
            return Ok(true);
        }
        Ok(false)
    }

    pub fn wait_restore(
        &self,
        handle: RestoreHandle,
        timeout: Duration,
    ) -> Result<RestoreResponse, ChannelError> {
        let deadline = Instant::now()
            .checked_add(timeout)
            .ok_or(ChannelError::RestoreTimeout {
                operation_id: handle.operation_id,
            })?;
        loop {
            let response = self.poll_restore(handle)?;
            if response.state != RestoreState::Pending {
                return Ok(response);
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(ChannelError::RestoreTimeout {
                    operation_id: handle.operation_id,
                });
            }
            self.restore_completions_ready(remaining)?;
        }
    }
}

fn next_id(counter: &AtomicU64) -> Result<u64, ChannelError> {
    counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .map_err(|_| ChannelError::SessionRequiresReconnect)
}

#[cfg(test)]
#[path = "../tests/unit/cache_client.rs"]
mod tests;
