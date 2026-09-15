//! Validate the entire teacher-forced submission/lifetime plan before touching CUDA.

use super::*;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Query {
    pub case: String,
    pub tokens: usize,
}

pub(super) struct PlannedQuery {
    pub case: usize,
    pub start: usize,
    pub end: usize,
    pub output_step: Option<usize>,
    pub finished: bool,
}

impl Case {
    pub(super) fn input_len(&self) -> usize {
        self.prompt_token_ids.len() + self.continuation_token_ids.len() - 1
    }

    pub(super) fn inputs(&self, start: usize, end: usize) -> Vec<u32> {
        self.prompt_token_ids
            .iter()
            .chain(&self.continuation_token_ids)
            .copied()
            .skip(start)
            .take(end - start)
            .collect()
    }
}

impl Probe {
    pub(super) fn plan_batches(&self) -> Result<Vec<Vec<PlannedQuery>>, &'static str> {
        let capacity = self.compile.maximum_query_tokens;
        if capacity == 0 || self.compile.maximum_batch_size == 0 {
            return Err("empty executable capacity");
        }
        let batches = self.batches.clone().unwrap_or_else(|| {
            self.cases
                .iter()
                .flat_map(|case| {
                    let mut remaining = case.prompt_token_ids.len();
                    let mut batches = Vec::new();
                    while remaining > 0 {
                        let tokens = remaining.min(capacity);
                        batches.push(vec![Query {
                            case: case.id.clone(),
                            tokens,
                        }]);
                        remaining -= tokens;
                    }
                    batches.extend((1..case.continuation_token_ids.len()).map(|_| {
                        vec![Query {
                            case: case.id.clone(),
                            tokens: 1,
                        }]
                    }));
                    batches
                })
                .collect()
        });
        let indices = self
            .cases
            .iter()
            .enumerate()
            .map(|(index, case)| (case.id.as_str(), index))
            .collect::<HashMap<_, _>>();
        let mut cursors = vec![0_usize; self.cases.len()];
        let mut active = 0_usize;
        let mut plan = Vec::new();
        for batch in batches {
            if batch.is_empty() || batch.len() > self.compile.maximum_batch_size {
                return Err("submission request count exceeds capacity");
            }
            let mut seen = HashSet::new();
            let mut tokens = 0_usize;
            let mut finished = 0;
            let mut queries = Vec::new();
            for query in batch {
                let &index = indices.get(query.case.as_str()).ok_or("unknown case")?;
                if !seen.insert(index) || query.tokens == 0 {
                    return Err("duplicate case or empty query");
                }
                tokens = tokens
                    .checked_add(query.tokens)
                    .ok_or("query count overflow")?;
                if tokens > capacity {
                    return Err("submission token count exceeds capacity");
                }
                let case = &self.cases[index];
                let start = cursors[index];
                let end = start.checked_add(query.tokens).ok_or("position overflow")?;
                let prompt = case.prompt_token_ids.len();
                if end > case.input_len()
                    || (start < prompt && end > prompt)
                    || (start >= prompt && query.tokens != 1)
                {
                    return Err("query crosses a teacher-forced generation boundary");
                }
                active += usize::from(start == 0);
                let done = end == case.input_len();
                finished += usize::from(done);
                queries.push(PlannedQuery {
                    case: index,
                    start,
                    end,
                    output_step: end.checked_sub(prompt),
                    finished: done,
                });
                cursors[index] = end;
            }
            // New owners are acquired before this submission completes, so
            // finishing peers cannot lend them slots within the same batch.
            if active > self.compile.maximum_batch_size {
                return Err("live requests exceed state-slot capacity");
            }
            active -= finished;
            plan.push(queries);
        }
        if active != 0
            || cursors
                .iter()
                .zip(&self.cases)
                .any(|(&n, case)| n != case.input_len())
        {
            return Err("submission plan leaves cases incomplete");
        }
        Ok(plan)
    }
}

#[cfg(test)]
#[path = "batches_tests.rs"]
mod tests;
