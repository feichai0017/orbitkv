use std::collections::HashMap;
use std::sync::LazyLock;
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use super::ExecutionResource;
use crate::CompletionResourceEvidence;

const CAPACITY: usize = 128;
const MAX_AGE: Duration = Duration::from_secs(2);

#[derive(Clone, Copy)]
pub(crate) struct FreshResourceEvidence {
    pub(crate) resources: CompletionResourceEvidence,
    observed_at: Instant,
}

static EVIDENCE: LazyLock<Mutex<HashMap<ExecutionResource, FreshResourceEvidence>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

pub(crate) fn record(
    resource: ExecutionResource,
    resources: CompletionResourceEvidence,
    age: Duration,
) {
    let now = Instant::now();
    let observed_at = now.checked_sub(age).unwrap_or(now);
    let mut evidence = EVIDENCE.lock();
    if !evidence.contains_key(&resource)
        && evidence.len() == CAPACITY
        && let Some(oldest) = evidence
            .iter()
            .min_by_key(|(_, evidence)| evidence.observed_at)
            .map(|(resource, _)| *resource)
    {
        evidence.remove(&oldest);
    }
    evidence.insert(
        resource,
        FreshResourceEvidence {
            resources,
            observed_at,
        },
    );
}

pub(crate) fn current(resource: ExecutionResource, now: Instant) -> Option<FreshResourceEvidence> {
    EVIDENCE.lock().get(&resource).copied().filter(|evidence| {
        evidence.observed_at <= now && now.duration_since(evidence.observed_at) <= MAX_AGE
    })
}

#[cfg(test)]
#[path = "../../tests/unit/cost/admission.rs"]
mod tests;
