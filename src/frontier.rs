use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct CompletionDomainId(pub String);

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct ExecutionSubmissionId(pub u64);

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ExecutionTicket {
    pub submission_id: ExecutionSubmissionId,
    pub completion_domain: CompletionDomainId,
    pub domain_sequence: u64,
    pub request_ids: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CompletionDomainSnapshot {
    pub completion_domain: CompletionDomainId,
    pub completed_through: u64,
    pub completed_out_of_order: Vec<u64>,
    pub pending_submissions: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CompletionSequenceRange {
    pub start: u64,
    pub end_exclusive: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RequestCompletionWitness {
    pub completion_domain: CompletionDomainId,
    pub required_ranges: Vec<CompletionSequenceRange>,
    pub completed_through: u64,
    pub completed_out_of_order: Vec<u64>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct ExecutionFrontierStats {
    pub completion_domains: u64,
    pub pending_submissions: u64,
    pub tracked_requests: u64,
}

#[derive(Clone, Debug)]
struct CompletionDomain {
    next_sequence: u64,
    completed_through: u64,
    completed_out_of_order: BTreeSet<u64>,
    pending: BTreeSet<ExecutionSubmissionId>,
}

#[derive(Clone, Debug)]
pub struct ExecutionFrontier {
    next_submission_id: u64,
    domains: BTreeMap<CompletionDomainId, CompletionDomain>,
    pending: BTreeMap<ExecutionSubmissionId, ExecutionTicket>,
    request_submissions: BTreeMap<String, BTreeSet<ExecutionSubmissionId>>,
    request_domain_sequences: BTreeMap<String, BTreeMap<CompletionDomainId, BTreeSet<u64>>>,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum ExecutionFrontierError {
    #[error("completion domain must not be empty")]
    EmptyCompletionDomain,
    #[error("execution submission must reference at least one request")]
    EmptyRequests,
    #[error("execution submission contains an empty request id")]
    EmptyRequestId,
    #[error("execution submission contains duplicate request {0:?}")]
    DuplicateRequest(String),
    #[error("execution submission generation exhausted")]
    SubmissionGenerationExhausted,
    #[error("completion-domain sequence exhausted for {0:?}")]
    DomainSequenceExhausted(CompletionDomainId),
    #[error("unknown execution submission {0:?}")]
    UnknownSubmission(ExecutionSubmissionId),
    #[error("unknown completion domain {0:?}")]
    UnknownCompletionDomain(CompletionDomainId),
    #[error("execution submission {submission:?} belongs to {actual:?}, not {expected:?}")]
    CompletionDomainMismatch {
        submission: ExecutionSubmissionId,
        expected: CompletionDomainId,
        actual: CompletionDomainId,
    },
    #[error("execution completion contains duplicate submission {0:?}")]
    DuplicateCompletion(ExecutionSubmissionId),
    #[error("request {0:?} still has pending execution submissions")]
    RequestStillPending(String),
    #[error("execution frontier lost request {request:?} for submission {submission:?}")]
    InconsistentRequestIndex {
        request: String,
        submission: ExecutionSubmissionId,
    },
    #[error("execution completion witness is incomplete for request {0:?}")]
    IncompleteWitness(String),
}

impl Default for ExecutionFrontier {
    fn default() -> Self {
        Self {
            next_submission_id: 1,
            domains: BTreeMap::new(),
            pending: BTreeMap::new(),
            request_submissions: BTreeMap::new(),
            request_domain_sequences: BTreeMap::new(),
        }
    }
}

impl ExecutionFrontier {
    /// Registers one immutable execution submission in a completion domain.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid identities or exhausted counters.
    pub fn register(
        &mut self,
        completion_domain: CompletionDomainId,
        mut request_ids: Vec<String>,
    ) -> Result<ExecutionTicket, ExecutionFrontierError> {
        if completion_domain.0.is_empty() {
            return Err(ExecutionFrontierError::EmptyCompletionDomain);
        }
        if request_ids.is_empty() {
            return Err(ExecutionFrontierError::EmptyRequests);
        }
        request_ids.sort();
        for request_id in &request_ids {
            if request_id.is_empty() {
                return Err(ExecutionFrontierError::EmptyRequestId);
            }
        }
        if let Some(pair) = request_ids.windows(2).find(|pair| pair[0] == pair[1]) {
            return Err(ExecutionFrontierError::DuplicateRequest(pair[0].clone()));
        }

        let submission_id = ExecutionSubmissionId(self.next_submission_id);
        let next_submission_id = self
            .next_submission_id
            .checked_add(1)
            .ok_or(ExecutionFrontierError::SubmissionGenerationExhausted)?;
        let domain = self
            .domains
            .entry(completion_domain.clone())
            .or_insert_with(|| CompletionDomain {
                next_sequence: 1,
                completed_through: 0,
                completed_out_of_order: BTreeSet::new(),
                pending: BTreeSet::new(),
            });
        let domain_sequence = domain.next_sequence;
        let next_sequence = domain_sequence.checked_add(1).ok_or_else(|| {
            ExecutionFrontierError::DomainSequenceExhausted(completion_domain.clone())
        })?;
        let ticket = ExecutionTicket {
            submission_id,
            completion_domain,
            domain_sequence,
            request_ids,
        };

        self.next_submission_id = next_submission_id;
        domain.next_sequence = next_sequence;
        domain.pending.insert(submission_id);
        for request_id in &ticket.request_ids {
            self.request_submissions
                .entry(request_id.clone())
                .or_default()
                .insert(submission_id);
            self.request_domain_sequences
                .entry(request_id.clone())
                .or_default()
                .entry(ticket.completion_domain.clone())
                .or_default()
                .insert(ticket.domain_sequence);
        }
        self.pending.insert(submission_id, ticket.clone());
        Ok(ticket)
    }

    /// Atomically completes submissions after validating their domain.
    ///
    /// Completion may arrive out of order. Each domain advances only through
    /// its largest contiguous completed sequence.
    ///
    /// # Errors
    ///
    /// Returns an error without changing state when any submission is unknown,
    /// duplicated, or belongs to another domain.
    pub fn complete(
        &mut self,
        completion_domain: &CompletionDomainId,
        submission_ids: &[ExecutionSubmissionId],
    ) -> Result<(), ExecutionFrontierError> {
        let unique = submission_ids.iter().copied().collect::<BTreeSet<_>>();
        if unique.len() != submission_ids.len() {
            let duplicate = submission_ids
                .iter()
                .copied()
                .find(|id| submission_ids.iter().filter(|other| *other == id).count() > 1)
                .unwrap_or(ExecutionSubmissionId(0));
            return Err(ExecutionFrontierError::DuplicateCompletion(duplicate));
        }
        if !self.domains.contains_key(completion_domain) {
            return Err(ExecutionFrontierError::UnknownCompletionDomain(
                completion_domain.clone(),
            ));
        }
        let mut tickets = Vec::with_capacity(submission_ids.len());
        for submission_id in submission_ids {
            let ticket = self
                .pending
                .get(submission_id)
                .cloned()
                .ok_or(ExecutionFrontierError::UnknownSubmission(*submission_id))?;
            if ticket.completion_domain != *completion_domain {
                return Err(ExecutionFrontierError::CompletionDomainMismatch {
                    submission: *submission_id,
                    expected: completion_domain.clone(),
                    actual: ticket.completion_domain,
                });
            }
            tickets.push(ticket);
        }
        for ticket in &tickets {
            for request_id in &ticket.request_ids {
                if !self
                    .request_submissions
                    .get(request_id)
                    .is_some_and(|submissions| submissions.contains(&ticket.submission_id))
                {
                    return Err(ExecutionFrontierError::InconsistentRequestIndex {
                        request: request_id.clone(),
                        submission: ticket.submission_id,
                    });
                }
            }
        }

        let domain = self.domains.get_mut(completion_domain).ok_or_else(|| {
            ExecutionFrontierError::UnknownCompletionDomain(completion_domain.clone())
        })?;
        for ticket in tickets {
            self.pending.remove(&ticket.submission_id);
            domain.pending.remove(&ticket.submission_id);
            record_completion(domain, ticket.domain_sequence);
            for request_id in ticket.request_ids {
                let remove_request =
                    if let Some(submissions) = self.request_submissions.get_mut(&request_id) {
                        submissions.remove(&ticket.submission_id);
                        submissions.is_empty()
                    } else {
                        false
                    };
                if remove_request {
                    self.request_submissions.remove(&request_id);
                }
            }
        }
        Ok(())
    }

    #[must_use]
    pub fn pending_for_request(&self, request_id: &str) -> u64 {
        self.request_submissions
            .get(request_id)
            .map_or(0, |submissions| submissions.len() as u64)
    }

    #[must_use]
    pub fn request_domains(&self, request_id: &str) -> Vec<CompletionDomainSnapshot> {
        let Some(domains) = self.request_domain_sequences.get(request_id) else {
            return Vec::new();
        };
        domains
            .keys()
            .filter_map(|domain| self.snapshot(domain))
            .collect()
    }

    #[must_use]
    pub fn request_completion_witnesses(&self, request_id: &str) -> Vec<RequestCompletionWitness> {
        let Some(domains) = self.request_domain_sequences.get(request_id) else {
            return Vec::new();
        };
        domains
            .iter()
            .filter_map(|(completion_domain, required_sequences)| {
                let snapshot = self.snapshot(completion_domain)?;
                Some(RequestCompletionWitness {
                    completion_domain: completion_domain.clone(),
                    required_ranges: compress_sequences(required_sequences),
                    completed_through: snapshot.completed_through,
                    completed_out_of_order: snapshot.completed_out_of_order,
                })
            })
            .collect()
    }

    /// Forgets completed request history after verifying that no execution
    /// submission still references it.
    ///
    /// # Errors
    ///
    /// Returns an error while the request still has pending submissions.
    pub fn release_request(&mut self, request_id: &str) -> Result<(), ExecutionFrontierError> {
        if self.pending_for_request(request_id) > 0 {
            return Err(ExecutionFrontierError::RequestStillPending(
                request_id.to_owned(),
            ));
        }
        self.request_domain_sequences.remove(request_id);
        Ok(())
    }

    #[must_use]
    pub fn snapshot(
        &self,
        completion_domain: &CompletionDomainId,
    ) -> Option<CompletionDomainSnapshot> {
        self.domains
            .get(completion_domain)
            .map(|domain| CompletionDomainSnapshot {
                completion_domain: completion_domain.clone(),
                completed_through: domain.completed_through,
                completed_out_of_order: domain.completed_out_of_order.iter().copied().collect(),
                pending_submissions: domain.pending.len() as u64,
            })
    }

    #[must_use]
    pub fn snapshots(&self) -> Vec<CompletionDomainSnapshot> {
        self.domains
            .keys()
            .filter_map(|domain| self.snapshot(domain))
            .collect()
    }

    #[must_use]
    pub fn stats(&self) -> ExecutionFrontierStats {
        ExecutionFrontierStats {
            completion_domains: self.domains.len() as u64,
            pending_submissions: self.pending.len() as u64,
            tracked_requests: self.request_domain_sequences.len() as u64,
        }
    }
}

impl RequestCompletionWitness {
    #[must_use]
    pub fn proves_complete(&self) -> bool {
        let out_of_order = self
            .completed_out_of_order
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        self.required_ranges.iter().all(|range| {
            if range.start >= range.end_exclusive {
                return false;
            }
            let first_uncovered = range.start.max(self.completed_through.saturating_add(1));
            if first_uncovered >= range.end_exclusive {
                return true;
            }
            let required = range.end_exclusive - first_uncovered;
            u64::try_from(
                out_of_order
                    .range(first_uncovered..range.end_exclusive)
                    .count(),
            )
            .is_ok_and(|completed| completed == required)
        })
    }
}

fn record_completion(domain: &mut CompletionDomain, sequence: u64) {
    if sequence == domain.completed_through + 1 {
        domain.completed_through = sequence;
        while domain
            .completed_out_of_order
            .remove(&(domain.completed_through + 1))
        {
            domain.completed_through += 1;
        }
    } else if sequence > domain.completed_through + 1 {
        domain.completed_out_of_order.insert(sequence);
    }
}

fn compress_sequences(sequences: &BTreeSet<u64>) -> Vec<CompletionSequenceRange> {
    let mut ranges = Vec::new();
    let mut iter = sequences.iter().copied();
    let Some(mut start) = iter.next() else {
        return ranges;
    };
    let mut end_exclusive = start + 1;
    for sequence in iter {
        if sequence == end_exclusive {
            end_exclusive += 1;
        } else {
            ranges.push(CompletionSequenceRange {
                start,
                end_exclusive,
            });
            start = sequence;
            end_exclusive = sequence + 1;
        }
    }
    ranges.push(CompletionSequenceRange {
        start,
        end_exclusive,
    });
    ranges
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn domains_advance_independently_with_out_of_order_completion() {
        let mut frontier = ExecutionFrontier::default();
        let stream_a = CompletionDomainId("cuda:0:forward".into());
        let stream_b = CompletionDomainId("cuda:0:copy".into());
        let first = frontier
            .register(stream_a.clone(), vec!["r0".into()])
            .unwrap();
        let second = frontier
            .register(stream_a.clone(), vec!["r0".into(), "r1".into()])
            .unwrap();
        let copy = frontier
            .register(stream_b.clone(), vec!["r1".into()])
            .unwrap();

        frontier
            .complete(&stream_a, &[second.submission_id])
            .unwrap();
        assert_eq!(frontier.snapshot(&stream_a).unwrap().completed_through, 0);
        assert_eq!(frontier.pending_for_request("r0"), 1);
        assert_eq!(frontier.pending_for_request("r1"), 1);

        frontier.complete(&stream_b, &[copy.submission_id]).unwrap();
        assert_eq!(frontier.snapshot(&stream_b).unwrap().completed_through, 1);
        assert_eq!(frontier.pending_for_request("r1"), 0);

        frontier
            .complete(&stream_a, &[first.submission_id])
            .unwrap();
        assert_eq!(frontier.snapshot(&stream_a).unwrap().completed_through, 2);
        assert_eq!(frontier.pending_for_request("r0"), 0);
    }

    #[test]
    fn completion_preflight_is_atomic() {
        let mut frontier = ExecutionFrontier::default();
        let domain = CompletionDomainId("cuda:0:forward".into());
        let ticket = frontier
            .register(domain.clone(), vec!["r0".into()])
            .unwrap();
        assert_eq!(
            frontier.complete(&domain, &[ticket.submission_id, ExecutionSubmissionId(999)]),
            Err(ExecutionFrontierError::UnknownSubmission(
                ExecutionSubmissionId(999)
            ))
        );
        assert_eq!(frontier.pending_for_request("r0"), 1);
        assert_eq!(frontier.stats().pending_submissions, 1);
    }

    #[test]
    fn sparse_request_witness_accepts_only_explicit_completion() {
        let witness = RequestCompletionWitness {
            completion_domain: CompletionDomainId("cuda:0:forward".into()),
            required_ranges: vec![
                CompletionSequenceRange {
                    start: 2,
                    end_exclusive: 3,
                },
                CompletionSequenceRange {
                    start: 5,
                    end_exclusive: 7,
                },
            ],
            completed_through: 2,
            completed_out_of_order: vec![5, 6],
        };
        assert!(witness.proves_complete());

        let mut incomplete = witness;
        incomplete.completed_out_of_order.pop();
        assert!(!incomplete.proves_complete());
    }
}
