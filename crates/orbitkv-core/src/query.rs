//! Byte reservations held from query admission through the last GPU consumer.

use std::collections::HashMap;
use std::sync::Arc;

use opentelemetry::KeyValue;
use parking_lot::Mutex;

use crate::metrics::core_metrics;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QueryOwner {
    pub session: u64,
    pub operation: u64,
    pub revision: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Preparing,
    Ready,
    Restoring,
}

impl Phase {
    fn label(self) -> &'static str {
        match self {
            Self::Preparing => "preparing",
            Self::Ready => "ready",
            Self::Restoring => "restoring",
        }
    }
}

#[derive(Default)]
struct Usage {
    total: u64,
    instances: HashMap<String, u64>,
}

pub(crate) struct QueryBudget {
    global: u64,
    per_instance: u64,
    usage: Mutex<Usage>,
}

pub enum QueryAdmission {
    Admitted(QueryReservation),
    Busy,
    TooLarge,
}

/// Accounting is conservative: two owners of the same pages each reserve bytes.
/// The pinned allocator separately measures physical memory shared by owners.
#[derive(Clone)]
pub struct QueryReservation(Arc<Reservation>);

struct Reservation {
    budget: Arc<QueryBudget>,
    pub(crate) instance: String,
    pub(crate) namespace: String,
    state: Mutex<(u64, Phase)>,
}

impl QueryBudget {
    pub(crate) fn new(global: u64, per_instance: u64) -> Result<Arc<Self>, String> {
        if global == 0 || per_instance == 0 || per_instance > global || global > i64::MAX as u64 {
            return Err("query byte limits must satisfy 0 < instance <= global <= i64::MAX".into());
        }
        Ok(Arc::new(Self {
            global,
            per_instance,
            usage: Mutex::new(Usage::default()),
        }))
    }

    pub(crate) fn reserve(
        self: &Arc<Self>,
        instance: &str,
        namespace: &str,
        bytes: u64,
    ) -> QueryAdmission {
        if bytes > self.per_instance {
            core_metrics().query_budget_bypasses.add(1, &[]);
            return QueryAdmission::TooLarge;
        }
        let mut usage = self.usage.lock();
        let instance_used = usage.instances.get(instance).copied().unwrap_or(0);
        if bytes > self.global - usage.total || bytes > self.per_instance - instance_used {
            core_metrics().query_budget_waits.add(1, &[]);
            return QueryAdmission::Busy;
        }
        usage.total += bytes;
        *usage.instances.entry(instance.into()).or_default() += bytes;
        account(bytes as i64, Phase::Preparing);
        QueryAdmission::Admitted(QueryReservation(Arc::new(Reservation {
            budget: Arc::clone(self),
            instance: instance.into(),
            namespace: namespace.into(),
            state: Mutex::new((bytes, Phase::Preparing)),
        })))
    }
}

fn account(bytes: i64, phase: Phase) {
    core_metrics()
        .query_reserved_bytes
        .add(bytes, &[KeyValue::new("phase", phase.label())]);
}

impl QueryReservation {
    pub(crate) fn instance(&self) -> &str {
        &self.0.instance
    }

    pub(crate) fn namespace(&self) -> &str {
        &self.0.namespace
    }

    pub(crate) fn ready(&self, bytes: u64) -> Result<(), String> {
        let mut state = self.0.state.lock();
        if state.1 != Phase::Preparing || bytes > state.0 {
            return Err("query result exceeds its registered layout byte reservation".into());
        }
        let released = state.0 - bytes;
        self.0.release_bytes(released);
        account(-(state.0 as i64), state.1);
        account(bytes as i64, Phase::Ready);
        *state = (bytes, Phase::Ready);
        Ok(())
    }

    pub(crate) fn restoring(&self) {
        let mut state = self.0.state.lock();
        if state.1 == Phase::Ready {
            account(-(state.0 as i64), state.1);
            state.1 = Phase::Restoring;
            account(state.0 as i64, state.1);
        }
    }
}

impl Reservation {
    fn release_bytes(&self, bytes: u64) {
        let mut usage = self.budget.usage.lock();
        usage.total -= bytes;
        if let Some(instance) = usage.instances.get_mut(&self.instance) {
            *instance -= bytes;
            if *instance == 0 {
                usage.instances.remove(&self.instance);
            }
        }
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        let (bytes, phase) = *self.state.get_mut();
        self.release_bytes(bytes);
        account(-(bytes as i64), phase);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reserve(budget: &Arc<QueryBudget>, instance: &str, bytes: u64) -> QueryReservation {
        match budget.reserve(instance, "state", bytes) {
            QueryAdmission::Admitted(reservation) => reservation,
            _ => panic!("expected admission"),
        }
    }

    #[test]
    fn bytes_remain_charged_until_all_consumers_finish() {
        let budget = QueryBudget::new(100, 80).unwrap();
        let first = reserve(&budget, "a", 70);
        assert!(matches!(
            budget.reserve("a", "state", 11),
            QueryAdmission::Busy
        ));
        let second = reserve(&budget, "b", 30);
        assert!(matches!(
            budget.reserve("c", "state", 1),
            QueryAdmission::Busy
        ));
        assert!(matches!(
            budget.reserve("b", "state", 81),
            QueryAdmission::TooLarge
        ));
        assert!(first.ready(71).is_err());
        first.ready(40).unwrap();
        let gpu = first.clone();
        gpu.restoring();
        drop(first);
        assert_eq!(budget.usage.lock().total, 70);
        drop(gpu);
        assert_eq!(budget.usage.lock().total, 30);
        drop(second);
        assert_eq!(budget.usage.lock().total, 0);
        assert!(budget.usage.lock().instances.is_empty());
    }
}
