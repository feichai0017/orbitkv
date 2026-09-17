//! The kern scheduler: one tray, many sequences.
//!
//! Implements the pegainfer frontend's [`Scheduler`] contract — `submit`,
//! `step`, `metrics` — over a kern manifest driven by a [`Tray`] (one
//! runtime per GPU, in lockstep; `tray.rs` is the design). What the
//! manifest must speak is its [`Protocol`]: fills naming which buffer
//! carries the tokens, positions, slots and lengths of a call and which
//! output hands tokens back, and a `batch` on every program a step may
//! run — `groups` sequences of `rows` rows each. The scheduler picks a
//! forward by shape and never by name: a decode step and a speculative
//! round are the same motion — stage `rows` rows per sequence at its
//! position, run the forward that accepts `(b, rows)`, read the tokens it
//! hands each sequence, `count` of them when it says so — and differ only
//! in the rows the operator asked for (`--rows`, a shape the manifest
//! declares; the widest by default). A prompt goes through the forward
//! that takes one sequence of as many rows as fed, in chunks; when that
//! forward hands a token back (a hybrid GDN model, whose chunked kernels
//! must see every prompt token) every prompt token goes through it and
//! the first generated token comes from it, otherwise the last prompt
//! token is the first step's input. Without such a forward the prompt
//! goes through one-row steps, a token at a time — a row feeding its
//! prompt is a row like any other, whose outputs are dropped until the
//! last prompt token is in — or, when the manifest takes a run (a
//! `span`, see `tray.rs`), in every tray group the oldest sequence still
//! feeding its prompt feeds a run of up to `--chunk` of its tokens as
//! consecutive rows of the step, the run's last row handing its token
//! back; the runs of a step are one length, the shortest any of them can
//! feed, since the span var is one value for the tray. That is what a
//! decode-only tray manifest (K3 today) gets: correct, and slow without
//! the run.
//!
//! Policy, deliberately simple:
//! - prefill first: each step admits every waiting request it can seat,
//!   in order, and prefills each whole (one sequence, `chunk` tokens per
//!   call, launch by launch: no graph per chunk length), then runs one
//!   step over every running sequence;
//! - a request leases every KV page its worst case (`prompt + max_tokens`,
//!   plus `rows - 1` for the last step's rows past the end) needs at
//!   admission (`Tray::lease`, on the rank with the fewest rows, then
//!   pages),
//!   so a step never runs out of pages and nothing is ever preempted; the
//!   row drops with the sequence;
//! - batches are padded up to a bucket size — the same bucket on every
//!   rank of the tray, from the most loaded one — and each bucket's
//!   forward is CUDA-graph-captured once; padding rows write into a page
//!   the tray leases for them and nobody reads;
//! - greedy only: the manifest's `argmax` is the sampler. Non-greedy
//!   sampling params are logged once and served greedily;
//! - a speculative round is a forward of several rows per sequence whose
//!   `tokens` output is `[groups, rows]` and whose `count` output says how
//!   many of a sequence's the device accepted: the rows past the count
//!   land past the sequence's position and the next step overwrites them,
//!   the paged state's free rollback. Whether a round beats a plain step
//!   at a given batch size is the operator's call (`--rows`), not the
//!   scheduler's;
//! - prefix reuse: a finished sequence's KV lives on as snapshots
//!   (`Tray::checkpoint` / `retire`, indexed by their tokens in a `Prefix`
//!   tree over the whole tray), and a new prompt starts from the
//!   longest snapshot holding a proper prefix of it (`Tray::lease_from`;
//!   prefill covers the rest). A paged-only manifest checkpoints every
//!   whole page as a sequence fills it — free, a shared page — so any
//!   earlier prompt or output is reusable at page granularity; a manifest
//!   with a recurrent state checkpoints only where a request ends (the
//!   finished sequence's state slots become the snapshot's, nothing is
//!   copied), so only a prompt that continues an earlier request's whole
//!   context hits; a speculative round that accepted past `max_tokens` or
//!   the stop leaves a state no next turn continues, which is not kept.
//!   A `Busy` lease makes room and retries until it fits or
//!   nothing is left: the least recently hit snapshot is parked into the
//!   host tier (`--host-gib`: pinned DRAM per rank, `Tray::park`, all of
//!   its members' pieces or none; the coldest parked ones are dropped when
//!   it is full) or, without one, dropped. A prompt hitting a parked
//!   snapshot wakes it (`Tray::wake`) straight into its own row: one
//!   lease sized for the request, the prefix's pages copied in over the
//!   transfer streams, so room is decided once (waking the prefix and then
//!   leasing from it would ask twice, and the second ask could park what
//!   the first woke, without end). The request waits in `waking` until
//!   `Tray::awake` hands the row back and starts from it; no step queues
//!   behind them. The parked snapshot stays parked; the request's own
//!   finish indexes the longer context resident.

use std::collections::VecDeque;
use std::time::Instant;

use anyhow::{bail, Result};
use kern_manifest::protocol::{Forward, Rows};
use kern_manifest::Protocol;
use kern_pool::{Denied, Evicted, Found, Prefix, Tier};
use kern_runtime::Error;
use pegainfer_frontend::engine::{
    FinishReason, QueuedRequest, RejectReason, RequestId, RequestLedger, Scheduler, SchedulerMetrics,
    SpecDecodeCounters, MAX_SPEC_TOKENS,
};
use tracing::{debug, info, warn};

use crate::logline;
use crate::tray::{Cell, Rising, Row, Sleeping, Snapshot, Tray};

/// Decode batch buckets; a batch is padded up to the smallest one that
/// fits and each bucket owns one captured graph. The steps past 256 are
/// a tray's: a full run and a row or two on each other rank, padded by
/// a few rows rather than a block (cutting the run instead costs a whole
/// step whenever a prompt's remainder spills over).
const BUCKETS: [usize; 23] =
    [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 264, 272, 288, 304, 320, 384, 512, 640, 768, 1024];

fn bucket(n: usize) -> usize {
    BUCKETS.iter().copied().find(|&b| b >= n).unwrap_or(n)
}

/// Rows per rank for a step of `k` rows: the bucket, capped at `cap`
/// (`--max-seqs`) for a step of sequences; a run of more rows than that
/// keeps the ladder, since the same run has to land on the same bucket
/// whatever else is in the step (cuBLAS picks its kernel by m).
fn rows_per_rank(k: usize, cap: usize) -> usize {
    match k > cap {
        true => bucket(k),
        false => bucket(k).min(cap),
    }
}

/// A step's runs: in each tray group of `t` of the `n` ranks the oldest
/// sequence still feeding its prompt feeds one, and every run is as long
/// as the shortest of them can feed, within `cap`, since the span var is
/// one value for the tray. `seqs[i] = (rank, tokens it can feed)` in age
/// order; a rank holds `limit` rows, and a run's rows replace its one
/// (a group without a run leads with as many padding rows). `(c, the
/// runners)`: `(1, [])` when nothing is a run.
fn runs(seqs: &[(usize, usize)], n: usize, t: usize, limit: usize, cap: usize) -> (usize, Vec<usize>) {
    let runners: Vec<usize> = (0..n / t).filter_map(|g| seqs.iter().position(|&(q, a)| q / t == g && a > 1)).collect();
    let on = |q: usize| seqs.iter().filter(|&&(r, _)| r == q).count();
    let owns = |q: usize| runners.iter().any(|&i| seqs[i].0 == q);
    let leads = |q: usize| q.is_multiple_of(t) && !(q..q + t).any(owns);
    let room = (0..n).filter_map(|q| match (owns(q), leads(q)) {
        (true, _) => Some((limit + 1).saturating_sub(on(q))),
        (_, true) => Some(limit.saturating_sub(on(q))),
        _ => None,
    });
    let c = runners.iter().map(|&i| seqs[i].1).chain(room).chain(std::iter::once(cap)).min().unwrap_or(1);
    match c > 1 && !runners.is_empty() {
        true => (c, runners),
        false => (1, Vec::new()),
    }
}

pub struct Policy {
    /// Prefill chunk (tokens per `prefill` call): the manifest's `tokens`
    /// bound, or a smaller one.
    pub chunk: Option<usize>,
    /// Cap on concurrently running sequences per rank (≤ the manifest's
    /// `seqs` bound).
    pub max_seqs: usize,
    /// Token ids that end a request unless it asked `ignore_eos`.
    pub stop_tokens: Vec<u32>,
    /// Rows per sequence of a step, a shape the manifest declares; the
    /// widest by default.
    pub rows: Option<u64>,
    /// Pinned host memory per rank for parked snapshots (0: none).
    pub host_bytes: u64,
}

/// How this process drives the manifest: the forwards a step and a
/// prompt go through and the batch they allow. Pure over the protocol and
/// the tray's size, so a synthetic manifest exercises every rejection.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Plan {
    /// Rows per sequence of a step.
    rows: u64,
    /// The forward that takes one sequence of `rows` rows; a batch of `b`
    /// runs the one the protocol picks for `(b, rows)`.
    step: Forward,
    /// The forward a prompt chunk goes through, `None` when the prompt
    /// goes through steps.
    chunk: Option<Forward>,
    /// Most rows one sequence may feed as a run in a one-row step, when
    /// the manifest takes one at every batch size up to `max_seqs`.
    span: Option<usize>,
    /// Concurrent sequences per rank.
    max_seqs: usize,
}

impl Plan {
    /// `page` is the tray's page in tokens (the pad page a step's rows
    /// must fit), `seqs_max` its sequences per rank, `want_rows` and
    /// `want_seqs` the operator's asks.
    fn check(p: &Protocol, page: usize, seqs_max: usize, want_rows: Option<u64>, want_seqs: usize) -> Result<Plan> {
        let shapes = p.row_shapes();
        let Some(rows) = want_rows.or(shapes.last().copied()) else {
            bail!("no program takes a fixed number of rows per sequence: nothing to run a step with")
        };
        let Some(step) = p.forward(1, Rows::Const(rows)).cloned() else {
            bail!("no program takes one sequence of {rows} rows; the manifest declares rows {shapes:?}")
        };
        if rows as usize > page {
            bail!("{rows} rows per sequence per step exceed the {page}-token pad page");
        }
        if rows as usize - 1 > MAX_SPEC_TOKENS {
            bail!("{} rows past the first per step, the frontend's metrics hold {MAX_SPEC_TOKENS}", rows - 1);
        }
        let chunk = p.chunk().cloned();
        if chunk.is_none() && rows > 1 {
            bail!(
                "no program takes a prompt chunk, so prompts go through steps, which takes one-row steps, not {rows}"
            );
        }
        let max_seqs = want_seqs
            .clamp(1, seqs_max)
            .min(p.rows.max as usize / rows as usize)
            .min(p.max_groups(Rows::Const(rows)) as usize)
            .max(1);
        let span =
            p.span.as_ref().filter(|_| rows == 1 && p.spanned(max_seqs as u64).is_some()).map(|s| s.max as usize);
        Ok(Plan { rows, step, chunk, span, max_seqs })
    }

    /// Rows a step writes past the token it feeds: what a lease holds past
    /// `prompt + max_tokens` so the last step's rows have slots.
    fn headroom(&self) -> usize {
        self.rows as usize - 1
    }

    /// The acceptance counters, for a plan whose steps take several rows.
    fn counters(&self) -> Option<SpecDecodeCounters> {
        (self.rows > 1).then(|| SpecDecodeCounters { num_spec_tokens: self.rows - 1, ..Default::default() })
    }
}

struct Seq {
    id: RequestId,
    /// Tokens already in the KV state.
    pos: usize,
    /// The token the next decode step feeds at `pos`.
    next: u32,
    /// Prompt tokens still to feed after `next`, one per step, when the
    /// prompt goes through decode; the outputs of those steps are dropped.
    pending: VecDeque<u32>,
    generated: usize,
    /// Tokens the client got: `generated` less the stop token and whatever
    /// a speculative round accepted past `max_tokens` or the stop.
    emitted: usize,
    max_tokens: usize,
    ignore_eos: bool,
    /// Its KV pages and state slots across the tray; returned when the
    /// sequence drops.
    row: Row,
    prompt_len: usize,
    /// The tokens in the state, `pos` of them; the prefix tree keys this
    /// sequence's snapshots by them.
    history: Vec<i64>,
    /// Tokens already checkpointed, a whole number of pages.
    checkpointed: usize,
    admitted: Instant,
}

impl Seq {
    /// `fed` went through a program and is in the state now.
    fn advance(&mut self, fed: impl IntoIterator<Item = u32>) {
        for t in fed {
            self.pos += 1;
            self.history.push(t as i64);
        }
    }

    /// Account `toks` as generated, in order: each is emitted until a stop
    /// token (itself not emitted, pegainfer convention; it still counts
    /// against `max_tokens` like vLLM's) or `max_tokens`. Finishes the
    /// request in the ledger when it is done, otherwise the last token is
    /// the next step's input. Returns how many tokens were emitted and
    /// whether the sequence finished.
    fn emit(&mut self, toks: &[u32], stop: &[u32], ledger: &mut RequestLedger) -> (u64, bool) {
        let mut out = Vec::with_capacity(toks.len());
        let mut reason = None;
        for &tok in toks {
            self.generated += 1;
            if !self.ignore_eos && stop.contains(&tok) {
                reason = Some(FinishReason::Stop);
                break;
            }
            out.push(tok);
            if self.generated >= self.max_tokens {
                reason = Some(FinishReason::Length);
                break;
            }
        }
        self.emitted += out.len();
        if !out.is_empty() {
            ledger.push_tokens(self.id, &out, &[]);
        }
        let done = match reason {
            Some(r) => {
                debug!(
                    request = %self.id,
                    reason = ?r,
                    prompt = self.prompt_len,
                    generated = self.generated,
                    elapsed_s = logline::secs(self.admitted.elapsed()),
                    "finished"
                );
                ledger.finish(self.id, r);
                true
            }
            None => {
                if let Some(&t) = toks.last() {
                    self.next = t;
                }
                false
            }
        };
        (out.len() as u64, done)
    }
}

pub struct KernScheduler {
    tray: Tray,
    policy: Policy,
    /// Tokens per `prefill` call, within the manifest's bound.
    chunk: usize,
    plan: Plan,
    /// Draft acceptance, for a plan whose steps take several rows.
    counters: Option<SpecDecodeCounters>,
    waiting: VecDeque<QueuedRequest>,
    /// Requests whose rows are still on the way in from the host tier.
    waking: VecDeque<(QueuedRequest, Rising)>,
    running: Vec<Seq>,
    /// Snapshots of finished prefixes, for the next prompt that shares
    /// one (see the module doc).
    prefix: Prefix<Snapshot, Sleeping>,
    /// Checkpoint at every page (a paged-only manifest) rather than only
    /// where a request ends (one with a recurrent state).
    every_page: bool,
    warned_sampling: bool,
    stats: Stats,
}

/// Rolling counters for the periodic log line, reset when it prints.
struct Stats {
    since: Instant,
    /// Steps and their time.
    steps: u64,
    step_ns: u128,
    /// Tokens emitted to the ledger.
    tokens: u64,
    /// Prompt tokens fed: through the chunk forward (timed) or through steps.
    prefill_tokens: u64,
    prefill_ns: u128,
    /// Prompt tokens found in a snapshot, and snapshots evicted for room.
    prefix_hit_tokens: u64,
    evictions: u64,
    /// Requests whose prompt was found in a resident snapshot, and in one
    /// woken from the host, with the tokens found.
    resident_hits: u64,
    resident_hit_tokens: u64,
    host_hits: u64,
    host_hit_tokens: u64,
    /// Snapshots parked to the host, dropped from it for room, woken
    /// from it, and the tokens woken.
    parks: u64,
    host_evictions: u64,
    wakes: u64,
    wake_tokens: u64,
    /// The acceptance counters at the window's start, so the window's
    /// acceptance is reported rather than the process's.
    spec_at: (u64, u64, u64),
}

impl Stats {
    fn new(counters: &Option<SpecDecodeCounters>) -> Stats {
        let spec_at =
            counters.as_ref().map_or((0, 0, 0), |c| (c.num_drafts, c.num_draft_tokens, c.num_accepted_tokens));
        Stats {
            since: Instant::now(),
            steps: 0,
            step_ns: 0,
            tokens: 0,
            prefill_tokens: 0,
            prefill_ns: 0,
            prefix_hit_tokens: 0,
            evictions: 0,
            resident_hits: 0,
            resident_hit_tokens: 0,
            host_hits: 0,
            host_hit_tokens: 0,
            parks: 0,
            host_evictions: 0,
            wakes: 0,
            wake_tokens: 0,
            spec_at,
        }
    }
}

/// Public facts the frontend wants at launch.
pub struct Facts {
    pub total_blocks: usize,
    pub block_size: usize,
    /// Longest request (prompt + completion) one sequence can hold.
    pub max_request_tokens: usize,
}

/// What a lease attempt handed out.
enum Got {
    Row(Row),
    /// A row in flight from the host tier.
    Rising(Rising),
}

impl KernScheduler {
    /// Wrap a loaded tray (weights bound, peers connected, pads leased):
    /// settle how this process drives the manifest, within its bounds.
    pub fn new(tray: Tray, policy: Policy) -> Result<KernScheduler> {
        let plan = Plan::check(tray.protocol(), tray.page(), tray.seqs_max(), policy.rows, policy.max_seqs)?;
        let max = tray.protocol().rows.max as usize;
        let chunk = match policy.chunk {
            None => max,
            Some(c) if (1..=max).contains(&c) => c,
            Some(c) => bail!("--chunk {c}: the manifest's `tokens` bound is {max}; a chunk can only be smaller"),
        };
        let policy = Policy { max_seqs: plan.max_seqs, ..policy };
        let counters = plan.counters();
        let stats = Stats::new(&counters);
        let every_page = !tray.has_seq_state();
        let prefix = Prefix::new(tray.page());
        let s = KernScheduler {
            tray,
            policy,
            chunk,
            plan,
            counters,
            waiting: VecDeque::new(),
            waking: VecDeque::new(),
            running: Vec::new(),
            prefix,
            every_page,
            warned_sampling: false,
            stats,
        };
        s.log_ready();
        Ok(s)
    }

    /// What the frontend advertises at launch.
    pub fn facts(&self) -> Facts {
        // What admit() leases is `prompt + max_tokens + headroom`;
        // advertise the request-shaped remainder so the frontend clamps
        // `max_tokens` to something admissible instead of the scheduler
        // bouncing it (the wire turns a scheduler reject into a 500).
        Facts {
            total_blocks: self.tray.pages_total(),
            block_size: self.tray.page(),
            max_request_tokens: self.tray.max_seq_tokens() - self.headroom(),
        }
    }

    fn headroom(&self) -> usize {
        self.plan.headroom()
    }

    fn log_ready(&self) {
        let (tray, policy, plan, facts) = (&self.tray, &self.policy, &self.plan, self.facts());
        let (slots, _) = tray.seq_slots();
        // Pages: what requests can lease (one more per rank holds the
        // padding rows). Graphs are captured per bucket on first use
        // unless `eager`.
        info!(
            ranks = tray.len(),
            tray = tray.group_size(),
            pages = facts.total_blocks,
            page = facts.block_size,
            seq_slots = (slots > 0).then_some(slots),
            max_seqs_per_rank = policy.max_seqs,
            max_request_tokens = facts.max_request_tokens,
            chunk = self.chunk,
            buckets = ?BUCKETS.iter().filter(|&&b| b <= policy.max_seqs).collect::<Vec<_>>(),
            rows = plan.rows,
            step = %plan.step.name,
            span = plan.span,
            steps = ?tray.protocol().forwards.iter().filter(|f| f.rows == plan.step.rows).map(|f| (f.name.as_str(), f.groups)).collect::<Vec<_>>(),
            prefill = match &plan.chunk {
                Some(f) if f.emits.is_some() => format!("`{}`, emits the first token", f.name),
                Some(f) => format!("`{}`, state only", f.name),
                None => "through steps".to_string(),
            },
            checkpoints = if self.every_page { "every page" } else { "at request end" },
            host_gib_per_rank = (policy.host_bytes > 0).then_some(policy.host_bytes >> 30),
            "scheduler ready"
        );
    }

    /// Rows each rank has (running, and woken ones on the way in).
    fn rows_per_rank(&self) -> Vec<usize> {
        let mut rows = vec![0usize; self.tray.len()];
        for s in &self.running {
            rows[s.row.owner().index()] += 1;
        }
        for (_, r) in &self.waking {
            rows[r.owner().index()] += 1;
        }
        rows
    }

    /// Admit waiting requests in order and prefill each (single sequence,
    /// chunked); one that cannot be seated stops the scan.
    fn admit(&mut self, ledger: &mut RequestLedger) -> Result<()> {
        // Wakes land in order: the first still in flight stops the scan;
        // a landed one starts from its row ahead of the line.
        while let Some((q, r)) = self.waking.pop_front() {
            match self.tray.awake(r)? {
                Ok(row) => {
                    self.stats.wakes += 1;
                    self.stats.wake_tokens += row.prefix() as u64;
                    if ledger.is_aborted(q.id) {
                        ledger.retire(q.id);
                        continue;
                    }
                    self.admit_one(q, row, true, ledger)?;
                }
                Err(r) => {
                    self.waking.push_front((q, r));
                    break;
                }
            }
        }
        while let Some(q) = self.waiting.front() {
            let id = q.id;
            if ledger.is_aborted(id) {
                ledger.retire(id);
                self.waiting.pop_front();
                continue;
            }
            let cap = self.policy.max_seqs;
            let rows = self.rows_per_rank();
            if rows.iter().all(|&r| r >= cap) {
                break;
            }
            let prompt = q.request.prompt_tokens.len();
            let max_tokens = q.request.max_tokens;
            let worst = prompt + max_tokens + self.headroom();
            let limit = self.tray.max_seq_tokens();
            if prompt == 0 || worst > limit {
                ledger.reject(id, RejectReason::ContextLength { prompt_tokens: prompt, max_tokens, limit });
                self.waiting.pop_front();
                continue;
            }
            let ids: Vec<i64> = q.request.prompt_tokens.iter().map(|&t| t as i64).collect();
            // A hit continues on the snapshot's owner, which must have a row free.
            let owner = self.prefix.lookup(&ids).map(|h| match h.found {
                Found::Resident(snap) => snap.owner(),
                Found::Parked(p) => p.owner(),
            });
            if owner.is_some_and(|o| rows[o.index()] >= cap) {
                break;
            }
            // Room made for a retry may have parked or dropped the
            // snapshot hit above: look it up again where it is now.
            let got = loop {
                let tray = &mut self.tray;
                let attempt = match self.prefix.lookup(&ids) {
                    Some(h) => match h.found {
                        Found::Resident(snap) => tray.lease_from(&snap, h.len, worst).map(Got::Row),
                        Found::Parked(p) => tray.wake(&p, h.len, worst).map(Got::Rising),
                    },
                    None => tray.lease(worst, |r| Some(rows[r.index()]).filter(|&n| n < cap)).map(Got::Row),
                };
                match attempt {
                    Err(Error::Denied(Denied::Busy)) if self.make_room()? => {}
                    r => break r,
                }
            };
            let row = match got {
                Ok(Got::Row(row)) => row,
                Ok(Got::Rising(r)) => {
                    // Its copies are in flight; it starts once they land.
                    let q = self.waiting.pop_front().unwrap();
                    self.waking.push_back((q, r));
                    continue;
                }
                Err(Error::Denied(Denied::Busy)) => break, // wait for pages / a slot
                Err(Error::Denied(Denied::Remapping)) => break, // pages or a slot are on their way
                Err(Error::Denied(Denied::ExceedsRow { limit })) => {
                    ledger.reject(id, RejectReason::ContextLength { prompt_tokens: prompt, max_tokens, limit });
                    self.waiting.pop_front();
                    continue;
                }
                Err(Error::Denied(Denied::ExceedsPool)) => {
                    ledger.reject(id, RejectReason::KvBudget { prompt_tokens: prompt, worst_case_tokens: worst });
                    self.waiting.pop_front();
                    continue;
                }
                Err(e) => return Err(e.into()),
            };
            let q = self.waiting.pop_front().unwrap();
            self.admit_one(q, row, false, ledger)?;
        }
        Ok(())
    }

    /// Start `q` running in `row` (past the row's prefix, `woken` from
    /// the host tier or resident): through the chunk forward when the
    /// manifest has one, else with the prompt queued for the steps.
    fn admit_one(&mut self, q: QueuedRequest, row: Row, woken: bool, ledger: &mut RequestLedger) -> Result<()> {
        let id = q.id;
        let prompt = q.request.prompt_tokens.len();
        let max_tokens = q.request.max_tokens;
        let ids: Vec<i64> = q.request.prompt_tokens.iter().map(|&t| t as i64).collect();
        if !q.request.params.is_greedy() && !self.warned_sampling {
            warn!(
                request = %id,
                temperature = q.request.params.temperature,
                top_p = q.request.params.top_p,
                top_k = q.request.params.top_k,
                "non-greedy sampling params; this engine samples greedily (argmax in the manifest); further requests are not warned about"
            );
            self.warned_sampling = true;
        }
        ledger.admit(id);
        let t0 = Instant::now();
        let start = row.prefix();
        // The prompt tokens this request did not pay for: its `usage` says so
        // (`prompt_tokens_details.cached_tokens`), zero included, so a caller
        // can tell a miss from an engine that does not report hits.
        ledger.set_cached_tokens(id, start);
        // With a chunk forward every prompt token goes through it when it
        // hands the first generated token back itself; otherwise
        // everything but the last, which is the first step's input. A
        // snapshot hit skips its tokens (never the last one). Without one
        // the prompt past the hit is fed a token per step.
        let (n_pre, first, pending) = match self.plan.chunk.clone() {
            Some(f) => {
                let n_pre = if f.emits.is_some() { prompt } else { prompt - 1 };
                let first = self.prefill(&f, &row, &ids[..n_pre], start)?;
                self.stats.prefill_ns += t0.elapsed().as_nanos();
                self.stats.prefill_tokens += (n_pre - start) as u64;
                (n_pre, first, VecDeque::new())
            }
            None => (start, None, q.request.prompt_tokens[start + 1..].iter().copied().collect()),
        };
        self.stats.prefix_hit_tokens += start as u64;
        if woken {
            self.stats.host_hits += 1;
            self.stats.host_hit_tokens += start as u64;
        } else if start > 0 {
            self.stats.resident_hits += 1;
            self.stats.resident_hit_tokens += start as u64;
        }
        debug!(
            request = %id,
            prompt,
            max_tokens,
            row = ?row,
            prefix_hit = start,
            woken,
            prefill_ms = logline::ms(t0.elapsed()),
            "admitted"
        );
        let page = self.tray.page();
        let mut seq = Seq {
            id,
            pos: n_pre,
            next: q.request.prompt_tokens[n_pre.min(prompt - 1)],
            pending,
            generated: 0,
            emitted: 0,
            max_tokens,
            ignore_eos: q.request.params.ignore_eos,
            row,
            prompt_len: prompt,
            history: ids[..n_pre].to_vec(),
            checkpointed: start / page * page,
            admitted: t0,
        };
        self.checkpoint(&mut seq)?;
        if let Some(tok) = first {
            let (emitted, done) = seq.emit(&[tok], &self.policy.stop_tokens, ledger);
            self.stats.tokens += emitted;
            if done {
                self.finish(seq);
                return Ok(());
            }
        }
        self.running.push(seq);
        Ok(())
    }

    /// Room for a `Busy` lease: the coldest resident snapshot goes to
    /// the host tier when there is one (the coldest parked ones dropped
    /// until it fits), else is dropped. `false` when nothing is resident.
    fn make_room(&mut self) -> Result<bool> {
        let tray = &mut self.tray;
        let park = (self.policy.host_bytes > 0).then_some(|snap| tray.park(snap));
        let dropped = match self.prefix.evict(park)? {
            Some(Evicted::Parked { key, dropped }) => {
                debug!(tokens = key.len(), dropped, "parked");
                self.stats.parks += 1;
                dropped
            }
            Some(Evicted::Dropped { key, dropped }) => {
                debug!(tokens = key.len(), dropped, "dropped");
                self.stats.evictions += 1;
                dropped
            }
            None => return Ok(false),
        };
        self.stats.host_evictions += dropped as u64;
        Ok(true)
    }

    /// Checkpoint every whole page of `s` not yet checkpointed, when the
    /// manifest checkpoints every page.
    fn checkpoint(&mut self, s: &mut Seq) -> Result<()> {
        if !self.every_page {
            return Ok(());
        }
        let unit = self.tray.page();
        while s.checkpointed + unit <= s.pos {
            let len = s.checkpointed + unit;
            let snap = self.tray.checkpoint(&mut s.row, len)?;
            self.prefix.insert(&s.history[..len], snap);
            s.checkpointed = len;
        }
        Ok(())
    }

    /// A sequence is done: with a recurrent state, its whole context
    /// becomes a snapshot (the slots move, nothing is copied); without
    /// one, every whole page already is. A state holding tokens the client
    /// never got (a speculative round accepted past `max_tokens` or the
    /// stop) is a key no next turn can send: it returns to the pool.
    fn finish(&mut self, s: Seq) {
        if self.every_page || s.pos == 0 {
            return;
        }
        // The last token is normally the next input, not in the state
        // yet; the stop token itself may be in it, and stays: a chat's
        // next turn ends the answer with it.
        let visible = s.prompt_len + s.emitted;
        let past = s.history.get(visible..).unwrap_or_default();
        if s.pos > visible + 1 || past.iter().any(|&t| !self.policy.stop_tokens.contains(&(t as u32))) {
            debug!(request = %s.id, tokens = s.pos, visible, "not kept");
            return;
        }
        let snap = self.tray.retire(s.row, s.pos);
        self.prefix.insert(&s.history, snap);
    }

    /// Drop aborted sequences before a step so they neither pad nor compute.
    fn drop_aborted(&mut self, ledger: &mut RequestLedger) {
        let running = std::mem::take(&mut self.running);
        for s in running {
            if ledger.is_aborted(s.id) {
                ledger.retire(s.id);
                self.finish(s);
            } else {
                self.running.push(s);
            }
        }
    }

    /// Chunked single-sequence prefill of `ids[start..]` at positions
    /// `start..` of `row` (`start` is the row's prefix) through `f`; the
    /// first generated token when `f` hands it back.
    fn prefill(&mut self, f: &Forward, row: &Row, ids: &[i64], start: usize) -> Result<Option<u32>> {
        let chunk = self.chunk;
        let mut first = None;
        let mut pos = start;
        while pos < ids.len() {
            let c = (ids.len() - pos).min(chunk);
            let cells = [Cell { row, ids: ids[pos..pos + c].to_vec(), pos }];
            let mut st = self.tray.stage(&cells, c, |_| 1)?;
            st.run(f)?;
            pos += c;
            if pos == ids.len() {
                first = st.emitted(f)?[0].first().map(|&t| t as u32);
            }
        }
        Ok(first)
    }

    /// One step over every running sequence: `rows` rows per sequence at
    /// its position, the forward the protocol picks for the batch, and
    /// what it hands each sequence back. Under a span each tray group's
    /// oldest sequence still feeding its prompt feeds a run of up to
    /// `--chunk` tokens within the rows bound ([`runs`]); only a run's
    /// last output matters, as with a prefill chunk.
    fn step(&mut self, ledger: &mut RequestLedger) -> Result<()> {
        self.drop_aborted(ledger);
        if self.running.is_empty() {
            return Ok(());
        }
        let rows = self.plan.rows;
        let t0 = Instant::now();
        let (c, runners) = match self.plan.span {
            Some(mx) => {
                let seqs: Vec<(usize, usize)> =
                    self.running.iter().map(|s| (s.row.owner().index(), 1 + s.pending.len())).collect();
                let (n, t, limit) = (self.tray.len(), self.tray.group_size(), self.tray.seqs_max());
                runs(&seqs, n, t, limit, mx.min(self.chunk))
            }
            None => (1, Vec::new()),
        };
        let max_seqs = self.policy.max_seqs;
        // What each sequence feeds: its next token `rows` times, or a
        // runner's next and the `c - 1` prompt tokens after it.
        let fed: Vec<Vec<u32>> = self
            .running
            .iter_mut()
            .enumerate()
            .map(|(i, s)| match runners.contains(&i) {
                true => std::iter::once(s.next).chain(s.pending.drain(..c - 1)).collect(),
                false => vec![s.next; rows as usize],
            })
            .collect();
        let cells: Vec<Cell> = self
            .running
            .iter()
            .zip(&fed)
            .map(|(s, ids)| Cell { row: &s.row, ids: ids.iter().map(|&t| t as i64).collect(), pos: s.pos })
            .collect();
        let mut st = self.tray.stage(&cells, rows as usize, |k| rows_per_rank(k, max_seqs))?;
        let f = st.forward(rows).expect("planned: every bucket up to max_seqs has a forward");
        st.run(&f)?;
        let out = st.emitted(&f)?;
        drop(st);
        self.stats.step_ns += t0.elapsed().as_nanos();
        self.stats.steps += 1;

        let running = std::mem::take(&mut self.running);
        for (i, mut s) in running.into_iter().enumerate() {
            let toks: Vec<u32> = out[i].iter().map(|&t| t as u32).collect();
            // The tokens in the state now: the run's, or the one fed.
            let fed = if runners.contains(&i) { fed[i].clone() } else { vec![s.next] };
            if let Some(c) = &mut self.counters {
                // The first token is the next input's answer; the rest are
                // accepted drafts.
                let accepted = toks.len().saturating_sub(1);
                for p in &mut c.num_accepted_tokens_per_pos[..accepted] {
                    *p += 1;
                }
                c.num_drafts += 1;
                c.num_draft_tokens += rows - 1;
                c.num_accepted_tokens += accepted as u64;
            }
            // A row still feeding its prompt: this step's output is
            // dropped, the next prompt token is the next input.
            let done = match s.pending.pop_front() {
                Some(t) => {
                    self.stats.prefill_tokens += fed.len() as u64;
                    s.advance(fed);
                    s.next = t;
                    false
                }
                None => {
                    // The tokens taken are in the state now: what was fed
                    // and every accepted one after it.
                    self.stats.prefill_tokens += fed.len() as u64 - 1;
                    let taken = toks.len().max(1);
                    s.advance(fed.into_iter().chain(toks[..taken - 1].iter().copied()));
                    let (n, done) = s.emit(&toks, &self.policy.stop_tokens, ledger);
                    self.stats.tokens += n;
                    done
                }
            };
            if done {
                self.finish(s);
            } else {
                self.checkpoint(&mut s)?;
                self.running.push(s);
            }
        }
        Ok(())
    }

    /// One line per 5 s window in which anything happened; a window that
    /// only idled is dropped (and restarted, so the next line's rates are
    /// over its own window). `accepted` / `accept_pct` are the window's.
    fn log_stats(&mut self) {
        let st = &self.stats;
        let dt = st.since.elapsed();
        if dt.as_secs() < 5 {
            return;
        }
        if st.tokens > 0 || st.prefill_tokens > 0 {
            let round = |x: f64, d: f64| (x * d).round() / d;
            let host = self.tray.host_tier();
            let (slots, slots_used) = self.tray.seq_slots();
            let (drafts, draft_tokens, accepted) = self.counters.as_ref().map_or((0, 0, 0), |c| {
                (c.num_drafts - st.spec_at.0, c.num_draft_tokens - st.spec_at.1, c.num_accepted_tokens - st.spec_at.2)
            });
            info!(
                running = self.running.len(),
                rows_per_rank = ?self.rows_per_rank(),
                waiting = self.waiting.len() + self.waking.len(),
                kv_pct = round(self.tray.pages_used() as f64 * 100.0 / self.tray.pages_total().max(1) as f64, 10.0),
                steps = st.steps,
                step_ms = round(st.step_ns as f64 / 1e6 / st.steps.max(1) as f64, 100.0),
                tok_s = round(st.tokens as f64 / dt.as_secs_f64(), 1.0),
                prefill_tokens = st.prefill_tokens,
                prefill_tok_s =
                    (st.prefill_ns > 0).then(|| round(st.prefill_tokens as f64 / (st.prefill_ns as f64 / 1e9), 1.0)),
                prefix_hit_tokens = st.prefix_hit_tokens,
                resident_hits = st.resident_hits,
                resident_hit_tokens = st.resident_hit_tokens,
                host_hits = host.map(|_| st.host_hits),
                host_hit_tokens = host.map(|_| st.host_hit_tokens),
                checkpoints = self.prefix.entries(),
                evictions = st.evictions,
                parked = host.map(|_| self.prefix.count(Tier::Parked)),
                host_gib = host.map(|(u, _)| round(u as f64 / (1u64 << 30) as f64, 10.0)),
                parks = host.map(|_| st.parks),
                host_evictions = host.map(|_| st.host_evictions),
                wakes = host.map(|_| st.wakes),
                wake_tokens = host.map(|_| st.wake_tokens),
                slots_used = self.tray.has_seq_state().then_some(slots_used),
                slots = self.tray.has_seq_state().then_some(slots),
                remaps = self.tray.remaps(),
                accepted = self.counters.as_ref().map(|_| round((accepted + drafts) as f64 / drafts.max(1) as f64, 100.0)),
                accept_pct =
                    self.counters.as_ref().map(|_| round(accepted as f64 * 100.0 / draft_tokens.max(1) as f64, 1.0)),
                "stats"
            );
        }
        self.stats = Stats::new(&self.counters);
    }
}

impl Scheduler for KernScheduler {
    fn submit(&mut self, request: QueuedRequest) {
        self.waiting.push_back(request);
    }

    fn step(&mut self, ledger: &mut RequestLedger) -> Result<()> {
        self.admit(ledger)?;
        self.step(ledger)?;
        self.log_stats();
        Ok(())
    }

    fn metrics(&self) -> SchedulerMetrics {
        SchedulerMetrics {
            kv_used_blocks: self.tray.pages_used() as u64,
            kv_total_blocks: self.tray.pages_total() as u64,
            num_running_reqs: self.running.len() as u64,
            num_waiting_reqs: self.waiting.len() as u64,
            spec_decode: self.counters,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kern_manifest::types::Manifest;
    use kern_manifest::verify;

    #[test]
    fn rows_per_rank_keeps_the_ladder_for_a_run() {
        assert_eq!((rows_per_rank(5, 6), rows_per_rank(6, 6), rows_per_rank(16, 16)), (6, 6, 16));
        assert_eq!(
            (rows_per_rank(17, 16), rows_per_rank(160, 16), rows_per_rank(223, 16), rows_per_rank(256, 16)),
            (24, 192, 256, 256)
        );
    }

    #[test]
    fn every_group_runs_its_oldest_prompt_at_one_length() {
        // Four ranks alone (t=1): the oldest prompt of each rank runs,
        // all as long as the shortest can feed; a rank with none leads
        // with padding, so its rows bound the length too.
        let seqs = [(0, 1), (1, 9), (0, 5), (2, 1), (1, 7), (3, 3)];
        assert_eq!(runs(&seqs, 4, 1, 16, 256), (3, vec![2, 1, 5]));
        assert_eq!(runs(&seqs, 4, 1, 16, 2), (2, vec![2, 1, 5]));
        // Room: a runner's rows replace its one (rank 0 holds 2, so 15
        // more fit), a leading rank's add to its own (rank 2 holds 1).
        assert_eq!(runs(&[(0, 100), (0, 1), (2, 1)], 4, 1, 16, 256), (15, vec![0]));
        assert_eq!(runs(&[(0, 100), (2, 1)], 3, 1, 4, 256), (3, vec![0]));
        // One group of four: one run, the oldest; its peers' rows are not
        // in the way and no group leads.
        assert_eq!(runs(&[(3, 1), (1, 9), (0, 5)], 4, 4, 16, 256), (9, vec![1]));
        // Nothing to run, or no room for a run of two.
        assert_eq!(runs(&[(0, 1), (1, 1)], 2, 1, 16, 256), (1, vec![]));
        assert_eq!(runs(&[(0, 9), (1, 1)], 2, 1, 1, 256), (1, vec![]));
        assert_eq!(runs(&[], 2, 1, 16, 256), (1, vec![]));
    }

    /// The plain contract of kern-manifest's own tests: a prefill chunk
    /// that only writes state, a one-row step for one sequence and one
    /// for four.
    fn plain() -> Manifest {
        Manifest::from_json(include_str!("../../kern-manifest/tests/fixtures/plain.json")).unwrap()
    }

    /// The same plus a speculative round: 4 rows per sequence for up to 2
    /// sequences, handing back 4 tokens and a count.
    fn speculative() -> Manifest {
        let mut v: serde_json::Value = serde_json::from_str(&serde_json::to_string(&plain()).unwrap()).unwrap();
        v["buffers"]["verify_tokens"] =
            serde_json::json!({"kind": "output", "dtype": "i64", "shape": ["seqs", 4], "fill": "tokens"});
        v["buffers"]["nacc"] =
            serde_json::json!({"kind": "output", "dtype": "i32", "shape": ["seqs"], "fill": "count"});
        v["ops"]["accept"] = serde_json::json!({"params": ["out buffer<i64>", "out buffer<i32>"], "impl": {"launches": [{"entry": "extern:x"}]}});
        v["programs"]["round"] = serde_json::json!({"batch": {"groups": 2, "rows": 4}, "calls": [
            {"op": "accept", "args": [{"buf": "verify_tokens"}, {"buf": "nacc"}]}]});
        Manifest::from_json(&v.to_string()).unwrap()
    }

    /// The plain contract plus a run: `decode_span` takes 4 sequences of
    /// one row, one of them up to 6 rows.
    fn spanned() -> Manifest {
        let mut v: serde_json::Value = serde_json::from_str(&serde_json::to_string(&plain()).unwrap()).unwrap();
        v["vars"]["span"] = serde_json::json!({"max": 6});
        v["buffers"]["span_at"] = serde_json::json!({"kind": "input", "dtype": "i32", "shape": [1], "fill": "span_at"});
        v["ops"]["read_span"] =
            serde_json::json!({"params": ["in buffer<i32>"], "impl": {"launches": [{"entry": "extern:x"}]}});
        let mut calls = v["programs"]["decode"]["calls"].clone();
        calls.as_array_mut().unwrap().push(serde_json::json!({"op": "read_span", "args": [{"buf": "span_at"}]}));
        v["programs"]["decode_span"] =
            serde_json::json!({"batch": {"groups": 4, "rows": 1, "span": "span"}, "calls": calls});
        Manifest::from_json(&v.to_string()).unwrap()
    }

    fn check(m: &Manifest, page: usize, rows: Option<u64>, seqs: usize) -> Result<Plan> {
        Plan::check(&Protocol::check(&verify(m.clone())?)?, page, 4, rows, seqs)
    }

    fn rejects(m: &Manifest, page: usize, rows: Option<u64>, what: &str) {
        let Err(e) = check(m, page, rows, 4) else { panic!("accepted, expected `{what}`") };
        let e = format!("{e:#}");
        assert!(e.contains(what), "{e}");
    }

    #[test]
    fn plain_plan() {
        let p = check(&plain(), 16, None, 4).unwrap();
        assert_eq!(
            (p.rows, p.step.name.as_str(), p.chunk.as_ref().map(|f| f.name.as_str())),
            (1, "decode", Some("prefill"))
        );
        assert_eq!((p.headroom(), p.counters().is_none(), p.chunk.as_ref().unwrap().emits), (0, true, None));
        // The operator's ask, within the sequences bound.
        assert_eq!(
            (check(&plain(), 16, None, 0).unwrap().max_seqs, check(&plain(), 16, None, 100).unwrap().max_seqs),
            (1, 4)
        );
    }

    #[test]
    fn a_prompt_goes_through_steps_without_a_chunk_forward() {
        let mut m = plain();
        m.programs.remove("prefill");
        m.ops.remove("write");
        let p = check(&m, 16, None, 4).unwrap();
        assert_eq!((p.chunk, p.rows, p.span), (None, 1, None));
    }

    #[test]
    fn a_run_when_the_manifest_takes_one_at_every_batch_size() {
        assert_eq!(check(&spanned(), 16, None, 4).unwrap().span, Some(6));
        // Not past the span program's sequences bound, nor in a several-row step.
        let mut m = spanned();
        m.programs.get_mut("decode_span").unwrap().batch.as_mut().unwrap().groups = 2;
        assert_eq!((check(&m, 16, None, 4).unwrap().span, check(&m, 16, None, 2).unwrap().span), (None, Some(6)));
        let mut m = spanned();
        m.programs.get_mut("decode_span").unwrap().batch.as_mut().unwrap().groups = 2;
        assert_eq!(check(&m, 16, None, 2).unwrap().span, Some(6));
        assert_eq!(check(&plain(), 16, None, 4).unwrap().span, None);
    }

    #[test]
    fn the_widest_rows_by_default() {
        let p = check(&speculative(), 16, None, 4).unwrap();
        assert_eq!((p.rows, p.step.name.as_str(), p.headroom()), (4, "round", 3));
        assert_eq!(p.counters().map(|c| c.num_spec_tokens), Some(3));
        // Four rows per sequence fit twice in 8, and the round takes two sequences.
        assert_eq!(p.max_seqs, 2);
        // The plain step on the same manifest, on request.
        let p = check(&speculative(), 16, Some(1), 4).unwrap();
        assert_eq!((p.rows, p.step.name.as_str(), p.max_seqs), (1, "decode", 4));
    }

    #[test]
    fn plan_rejections() {
        rejects(&plain(), 16, Some(4), "no program takes one sequence of 4 rows; the manifest declares rows [1]");
        rejects(&speculative(), 2, Some(4), "exceed the 2-token pad page");
        let mut m = speculative();
        m.programs.remove("prefill");
        m.ops.remove("write");
        rejects(&m, 16, Some(4), "prompts go through steps, which takes one-row steps, not 4");
        // Only a chunk forward, one that hands a token back: nothing to step with.
        let mut m = plain();
        let head = m.programs["decode"].calls[0].clone();
        m.programs.get_mut("prefill").unwrap().calls.push(head);
        m.programs.remove("decode");
        m.programs.remove("decode_batch");
        rejects(&m, 16, None, "no program takes a fixed number of rows");
    }
}
