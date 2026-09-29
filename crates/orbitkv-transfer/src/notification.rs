use std::collections::HashMap;

use crate::types::Notification;

#[derive(Default)]
pub(crate) struct NotificationMailbox {
    next_generation: u64,
    active: HashMap<String, u64>,
    counts: HashMap<String, HashMap<String, usize>>,
}

pub(crate) enum NotificationMatch {
    Pending,
    Closed,
    Matched(String),
}

impl NotificationMailbox {
    pub(crate) fn open(&mut self, name: &str) -> u64 {
        self.next_generation = self.next_generation.wrapping_add(1).max(1);
        self.active.insert(name.to_string(), self.next_generation);
        self.counts.remove(name);
        self.next_generation
    }

    pub(crate) fn close(&mut self, name: &str, generation: u64) {
        if self.active.get(name) != Some(&generation) {
            return;
        }
        self.active.remove(name);
        self.counts.remove(name);
    }

    pub(crate) fn record(&mut self, notifications: Vec<Notification>) {
        for notification in notifications {
            if self.active.contains_key(&notification.name) {
                *self
                    .counts
                    .entry(notification.name)
                    .or_default()
                    .entry(notification.message)
                    .or_default() += 1;
            }
        }
    }

    pub(crate) fn status(
        &self,
        name: &str,
        generation: u64,
        expectations: &[(String, usize)],
    ) -> NotificationMatch {
        if self.active.get(name) != Some(&generation) {
            return NotificationMatch::Closed;
        }
        let counts = self.counts.get(name);
        expectations
            .iter()
            .find(|(message, count)| {
                counts
                    .and_then(|counts| counts.get(message))
                    .copied()
                    .unwrap_or_default()
                    >= *count
            })
            .map(|(message, _)| NotificationMatch::Matched(message.clone()))
            .unwrap_or(NotificationMatch::Pending)
    }
}

#[cfg(test)]
#[path = "../tests/unit/notification.rs"]
mod tests;
