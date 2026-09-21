//! Liveness session registry.
//!
//! Tracks an active inference session per instance_id. Each install produces a
//! monotonically increasing token; a new session for the same instance_id
//! supersedes the previous token, so a stale session's cleanup hook becomes
//! a no-op.

use dashmap::DashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SessionTopology {
    pub namespace: String,
    pub tp_size: u32,
    pub world_size: u32,
}

struct SessionEntry {
    token: u64,
    topology: SessionTopology,
}

#[derive(Default)]
pub(crate) struct SessionRegistry {
    sessions: DashMap<String, SessionEntry>,
    next_token: AtomicU64,
}

impl SessionRegistry {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Install a new session for `instance_id`, returning the token that
    /// identifies it. Overwrites any existing token (the old session's
    /// cleanup will observe `is_current == false` and skip).
    pub(crate) fn install(
        &self,
        instance_id: String,
        namespace: String,
        tp_size: u32,
        world_size: u32,
    ) -> u64 {
        let token = self.next_token.fetch_add(1, Ordering::Relaxed) + 1;
        self.sessions.insert(
            instance_id,
            SessionEntry {
                token,
                topology: SessionTopology {
                    namespace,
                    tp_size,
                    world_size,
                },
            },
        );
        token
    }

    pub(crate) fn topology(&self, instance_id: &str) -> Option<SessionTopology> {
        self.sessions
            .get(instance_id)
            .map(|entry| entry.topology.clone())
    }

    /// CAS-remove: only removes if `token` is still the current one.
    /// Returns true if this caller owns the cleanup.
    pub(crate) fn take(&self, instance_id: &str, token: u64) -> bool {
        let mut owned = false;
        self.sessions.remove_if(instance_id, |_, entry| {
            if entry.token == token {
                owned = true;
                true
            } else {
                false
            }
        });
        owned
    }
}

#[cfg(test)]
#[path = "../../tests/unit/cache/session.rs"]
mod tests;
