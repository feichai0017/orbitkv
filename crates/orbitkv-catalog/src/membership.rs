use std::collections::HashMap;
use std::time::{Duration, Instant};

use orbitkv_state::CacheOwner;
use parking_lot::RwLock;

/// Cached membership evidence and this runtime's conservative registration deadline.
/// The control-plane adapter updates it; cache operations perform no coordinator I/O.
pub struct MembershipView {
    owner: CacheOwner,
    state: RwLock<View>,
}

#[derive(Default)]
struct View {
    members: HashMap<uuid::Uuid, CacheOwner>,
    ready: bool,
    valid_until: Option<Instant>,
    fenced: bool,
}

impl MembershipView {
    pub fn new(owner: CacheOwner) -> Self {
        Self {
            owner,
            state: RwLock::new(View::default()),
        }
    }

    pub fn owner(&self) -> &CacheOwner {
        &self.owner
    }

    pub fn resolve(&self, incarnation: uuid::Uuid) -> Option<CacheOwner> {
        let state = self.state.read();
        if state.fenced
            || !state.ready
            || state
                .valid_until
                .is_none_or(|until| Instant::now() >= until)
        {
            return None;
        }
        state.members.get(&incarnation).cloned()
    }

    /// A delayed acknowledgement cannot revive an expired runtime. Half the
    /// returned TTL is reserved for response delay and lease-clock uncertainty.
    pub fn renew(&self, sent_at: Instant, ttl: Duration) -> bool {
        let now = Instant::now();
        let mut state = self.state.write();
        let deadline = sent_at.checked_add(ttl / 2);
        if state.fenced
            || state.valid_until.is_some_and(|until| now >= until)
            || deadline.is_none_or(|until| now >= until)
        {
            state.fenced = true;
            return false;
        }
        state.valid_until = deadline;
        true
    }

    pub fn replace_members(&self, members: impl IntoIterator<Item = (String, CacheOwner)>) {
        let mut state = self.state.write();
        state.members = members
            .into_iter()
            .map(|(_, owner)| (owner.incarnation, owner))
            .collect();
        state.ready = true;
        if state.members.get(&self.owner.incarnation) != Some(&self.owner) {
            state.fenced = true;
        }
    }

    /// Watch reconnect/repair suspends discovery until a complete snapshot arrives.
    pub fn invalidate_snapshot(&self) {
        let mut state = self.state.write();
        state.ready = false;
        state.members.clear();
    }

    pub fn fence(&self) {
        self.state.write().fenced = true;
    }

    pub fn registration_valid(&self) -> bool {
        let state = self.state.read();
        !state.fenced
            && state
                .valid_until
                .is_some_and(|until| Instant::now() < until)
    }

    pub fn permits(&self, owner: &CacheOwner) -> bool {
        let state = self.state.read();
        !state.fenced
            && state.ready
            && state
                .valid_until
                .is_some_and(|until| Instant::now() < until)
            && state.members.get(&owner.incarnation) == Some(owner)
    }
}

#[cfg(test)]
#[path = "../tests/unit/membership.rs"]
mod tests;
