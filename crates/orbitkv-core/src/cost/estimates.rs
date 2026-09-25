use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::LazyLock;
use std::time::Instant;

use super::{ALPHA, CAPACITY, CostKey, MAX_AGE, MIN_SAMPLES};

pub(super) static ESTIMATES: LazyLock<Mutex<Estimates>> =
    LazyLock::new(|| Mutex::new(Estimates::default()));

#[derive(Clone, Copy)]
pub(super) struct Estimate {
    pub(super) count: u64,
    pub(super) seconds: f64,
    pub(super) absolute_error: f64,
    pub(super) updated: Instant,
}

impl Estimate {
    fn reliable(self, now: Instant) -> bool {
        self.count >= MIN_SAMPLES && now.saturating_duration_since(self.updated) <= MAX_AGE
    }
}

#[derive(Default)]
pub(super) struct Estimates {
    entries: HashMap<CostKey, Estimate>,
}

impl Estimates {
    pub(super) fn predict(&self, key: CostKey, now: Instant) -> Option<Estimate> {
        self.entries.get(&key).copied().filter(|e| e.reliable(now))
    }

    pub(super) fn observe(&mut self, key: CostKey, seconds: f64, now: Instant) -> bool {
        let mut evicted = false;
        if !self.entries.contains_key(&key)
            && self.entries.len() == CAPACITY
            && let Some(oldest) = self
                .entries
                .iter()
                .min_by_key(|(_, e)| e.updated)
                .map(|(k, _)| *k)
        {
            self.entries.remove(&oldest);
            evicted = true;
        }
        let entry = self.entries.entry(key).or_insert(Estimate {
            count: 0,
            seconds,
            absolute_error: 0.0,
            updated: now,
        });
        // An idle path needs new evidence; old counts must not make it warm.
        if now.saturating_duration_since(entry.updated) > MAX_AGE {
            entry.count = 0;
        }
        if entry.count == 0 {
            entry.seconds = seconds;
            entry.absolute_error = 0.0;
        } else {
            entry.absolute_error +=
                ALPHA * ((seconds - entry.seconds).abs() - entry.absolute_error);
            entry.seconds += ALPHA * (seconds - entry.seconds);
        }
        entry.count = entry.count.saturating_add(1);
        entry.updated = now;
        evicted
    }
}

#[cfg(test)]
#[path = "../../tests/unit/cost/estimates.rs"]
mod tests;
