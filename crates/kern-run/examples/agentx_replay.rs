//! Replay the AgentX (Claude Code) traces through the checkpoint table:
//! page numbers only, no GPU, no model. The roadmap's K1 gate.
//!
//!     agentx_replay --traces <cc-traces-weka-062126>/traces.jsonl   # HF semianalysisai/cc-traces-weka-062126 \
//!         [--unit 64] [--kv-bytes 65536] [--state <bytes per slot>] [--budget-gib 250 | --capacity <tokens>]
//!         [--slots 130] [--concurrency 32] [--sessions N] [--host-gib 0]
//!
//! Every request of a session is a sequence: leased at its timestamp for
//! `in + out` tokens, prefilled from the longest checkpoint holding a
//! prefix of its input, dropped `api_time` later. A paged-only model
//! (default) leaves a checkpoint at every page boundary of the prompt as it
//! is admitted and of the output as it finishes; `--state` models a
//! recurrent state too — a checkpoint costs a sequence slot, so only the
//! finished request is retired into one. Tokens are synthesized from the
//! trace's 64-token block hashes (a request's output tokens are what the
//! next request of the session carries at those positions, when one
//! extends it); positions no block hash covers get tokens nothing else
//! shares. Sessions start `--concurrency` at a time, in file order, each
//! when the one that many before it ends. A `Busy` lease evicts
//! checkpoints until it fits, or waits for the next sequence to finish.
//! Pages and slots come out of one chunk budget (`--budget-gib`, or
//! `--capacity` tokens of pages plus `--slots` slots) as the runtime's
//! do: `--slots` is only what the pool starts with, a `Remapping` denial
//! is landed on the spot and the lease asked for again. With `--host-gib`
//! a `Busy` lease parks the coldest resident checkpoint into a host tier
//! of that size instead of dropping it (dropping the coldest parked ones
//! when the tier is full), and a hit on a parked one wakes it.
//!
//! Reported: hit rate (prefix tokens found / input tokens), the hits and
//! their tokens by tier (resident, or woken from the host), extend
//! percentiles, checkpoints kept, evictions, requests that had to wait,
//! remaps, the most slots the pool grew to, parks and wakes.

use std::collections::{BTreeMap, VecDeque};
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::sync::Arc;

use kern_manifest::types::Manifest;
use kern_pool::{Denied, Evicted, Found, Host, Lease, Pool, Prefix, Tier};
use serde::Deserialize;

#[derive(Deserialize)]
struct Session {
    requests: Vec<Record>,
}

#[derive(Deserialize)]
struct Record {
    t: f64,
    #[serde(rename = "type")]
    kind: String,
    #[serde(rename = "in", default)]
    input: u64,
    #[serde(default)]
    out: u64,
    #[serde(default)]
    hash_ids: Vec<u32>,
    api_time: Option<f64>,
    #[serde(default)]
    requests: Vec<Record>,
}

/// One model request: `t` seconds into its session, `input` prompt tokens
/// whose 64-token blocks hash to `blocks`, `out` generated, live `api`
/// seconds.
struct Req {
    session: usize,
    t: f64,
    input: usize,
    out: usize,
    blocks: Vec<u32>,
    api: f64,
    subagent: bool,
}

const BLOCK: usize = 64;

fn flatten(session: usize, s: Session) -> Vec<Req> {
    let mut out = Vec::new();
    for r in s.requests {
        if r.kind == "subagent" {
            for q in r.requests {
                out.push(Req {
                    session,
                    t: q.t,
                    input: q.input as usize,
                    out: q.out as usize,
                    blocks: q.hash_ids,
                    api: q.api_time.unwrap_or(1.0),
                    subagent: true,
                });
            }
        } else {
            out.push(Req {
                session,
                t: r.t,
                input: r.input as usize,
                out: r.out as usize,
                blocks: r.hash_ids,
                api: r.api_time.unwrap_or(1.0),
                subagent: false,
            });
        }
    }
    out.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap());
    out
}

/// The token at position `p` of a request whose blocks are `blocks`:
/// from the block hash when one covers it, else unique to (`session`,
/// `req`, `p`).
fn token(session: usize, req: usize, blocks: &[u32], p: usize) -> i64 {
    match blocks.get(p / BLOCK) {
        Some(&h) => ((session as i64) << 40) | ((h as i64) << 7) | (p % BLOCK) as i64,
        None => (1 << 62) | ((session as i64) << 40) | ((req as i64) << 20) | p as i64,
    }
}

/// The tokens of request `i` of `reqs` (sorted by time within a session):
/// its prompt, then its output as the next request extending it carries
/// it, else unique tokens.
fn tokens(reqs: &[Req], i: usize) -> Vec<i64> {
    let r = &reqs[i];
    let mut out: Vec<i64> = (0..r.input).map(|p| token(r.session, i, &r.blocks, p)).collect();
    let successor = reqs[i + 1..].iter().take(16).find(|q| {
        q.session == r.session && q.blocks.len() > r.blocks.len() && q.blocks[..r.blocks.len()] == r.blocks[..]
    });
    for p in r.input..r.input + r.out {
        out.push(match successor {
            Some(q) if p / BLOCK < q.blocks.len() => token(q.session, i, &q.blocks, p),
            _ => token(r.session, i, &[], p),
        });
    }
    out
}

fn manifest(unit: usize, kv_bytes: u64, state: u64, slots: usize, row: usize) -> Manifest {
    let state_json = if state > 0 { format!(r#", "rec": {{"bytes_per_seq": {state}}}"#) } else { String::new() };
    let line_json = if state > 0 {
        r#", "line_index": {"kind": "input", "dtype": "i32", "shape": [1, "seqs"], "domain": {"index_into": "rec", "stride": 1}}"#
    } else {
        ""
    };
    Manifest::from_json(&format!(
        r#"{{
        "schema_version": 5, "model": "replay", "vars": {{"tokens": {{"max": 1}}, "seqs": {{"max": {seqs}}}}},
        "states": {{"kv": {{"bytes_per_token": {kv_bytes}}}{state_json}}},
        "buffers": {{
            "block_table": {{"kind": "input", "dtype": "i32", "shape": ["seqs", {row}], "domain": {{"index_into": "kv", "stride": {unit}}}}}{line_json}
        }},
        "modules": {{}}, "ops": {{}}, "programs": {{}}
    }}"#,
        seqs = slots.saturating_sub(2).max(1)
    ))
    .expect("synthetic manifest")
}

fn pct(xs: &mut [usize], p: f64) -> usize {
    if xs.is_empty() {
        return 0;
    }
    xs.sort_unstable();
    xs[((xs.len() as f64 * p) as usize).min(xs.len() - 1)]
}

struct Live {
    lease: Lease,
    req: usize,
    hit: usize,
    /// The prompt and the output both: the output's tokens are known up
    /// front here.
    tokens: Vec<i64>,
}

#[derive(Default)]
struct Tally {
    input: u64,
    hit: u64,
    extend: Vec<usize>,
    /// Requests whose prefix was found resident, and on the host (woken).
    resident_hits: u64,
    resident_hit_tokens: u64,
    host_hits: u64,
    host_hit_tokens: u64,
    checkpoints: u64,
    evictions: u64,
    waited: u64,
    rejected: u64,
    max_live: usize,
    remaps: u64,
    max_slots: usize,
    parks: u64,
    wakes: u64,
    wake_tokens: u64,
    host_evictions: u64,
    host_peak: u64,
}

/// Make room for a `Busy` lease: park the coldest resident checkpoint
/// when there is a host tier (dropping the coldest parked ones until it
/// fits), else drop it. `false` when nothing is resident.
fn make_room(prefix: &mut Prefix, host: Option<&Arc<Host>>, tally: &mut Tally) -> bool {
    // The plan's copies would run on the runtime; here the bytes are
    // imaginary and the checkpoint goes straight back.
    let park = host.map(|h| move |cp| Ok::<_, ()>(h.park(&cp).map(|(p, _)| p).map_err(|_| cp)));
    let dropped = match prefix.evict(park).unwrap() {
        Some(Evicted::Parked { dropped, .. }) => {
            tally.parks += 1;
            tally.host_peak = tally.host_peak.max(host.map_or(0, |h| h.used()));
            dropped
        }
        Some(Evicted::Dropped { dropped, .. }) => {
            tally.evictions += 1;
            dropped
        }
        None => return false,
    };
    tally.host_evictions += dropped as u64;
    true
}

/// The runtime's chunk rule: 2 MiB granularity, half the smallest object,
/// at most 64 MiB.
fn chunk_size(page_bytes: u64, state: u64) -> u64 {
    let g = 2u64 << 20;
    let smallest = if state > 0 { page_bytes.min(state) } else { page_bytes };
    (smallest / 2 / g).clamp(1, 32) * g
}

fn main() {
    let mut traces = PathBuf::new();
    let mut unit = 64usize;
    let mut kv_bytes = 65536u64;
    let mut state = 0u64;
    let mut capacity: Option<u64> = None;
    let mut budget_gib = 250u64;
    let mut slots = 130usize;
    let mut concurrency = 32usize;
    let mut sessions_cap = usize::MAX;
    let mut host_gib = 0u64;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut v = || args.next().expect("value");
        match a.as_str() {
            "--traces" => traces = PathBuf::from(v()),
            "--unit" => unit = v().parse().unwrap(),
            "--kv-bytes" => kv_bytes = v().parse().unwrap(),
            "--state" => state = v().parse().unwrap(),
            "--capacity" => capacity = Some(v().parse().unwrap()),
            "--budget-gib" => budget_gib = v().parse().unwrap(),
            "--slots" => slots = v().parse().unwrap(),
            "--concurrency" => concurrency = v().parse().unwrap(),
            "--sessions" => sessions_cap = v().parse().unwrap(),
            "--host-gib" => host_gib = v().parse().unwrap(),
            _ => panic!("unknown arg {a}"),
        }
    }
    let file = std::fs::File::open(&traces).unwrap_or_else(|e| panic!("{}: {e}", traces.display()));
    let mut reqs: Vec<Req> = Vec::new();
    let mut spans: Vec<(f64, f64)> = Vec::new();
    for (i, line) in BufReader::new(file).lines().enumerate() {
        if i >= sessions_cap {
            break;
        }
        let s: Session = serde_json::from_str(&line.unwrap()).expect("trace line");
        let flat = flatten(i, s);
        let end = flat.iter().map(|r| r.t + r.api).fold(0.0, f64::max);
        spans.push((flat.first().map_or(0.0, |r| r.t), end));
        reqs.extend(flat);
    }
    let sessions = spans.len();
    // Session k starts when session k - concurrency ends.
    let mut offset = vec![0.0f64; sessions];
    for k in 0..sessions {
        if k >= concurrency {
            let prev = k - concurrency;
            offset[k] = offset[prev] + spans[prev].1;
        }
    }
    let row = (1usize << 20) / unit + 64;
    let m = manifest(unit, kv_bytes, state, slots, row);
    let page_bytes = kv_bytes * unit as u64;
    let chunk = chunk_size(page_bytes, state);
    let chunks = match capacity {
        Some(tokens) => (tokens / unit as u64 * page_bytes).div_ceil(chunk) + (slots as u64 * state).div_ceil(chunk),
        None => (budget_gib << 30) / chunk,
    };
    let first_slots = if state > 0 { slots } else { 0 };
    let pool =
        Arc::new(Pool::new(&m, chunk, chunks as u32, first_slots, None).expect("the budget holds the first slots").0);
    let host = (host_gib > 0).then(|| Arc::new(Host::new(host_gib << 30, 1 << 16, page_bytes, state)));
    let mut prefix = Prefix::new(unit);
    let mut tally = Tally::default();
    let mut order: Vec<usize> = (0..reqs.len()).collect();
    order.sort_by(|&a, &b| {
        let ta = offset[reqs[a].session] + reqs[a].t;
        let tb = offset[reqs[b].session] + reqs[b].t;
        ta.partial_cmp(&tb).unwrap().then(a.cmp(&b))
    });
    // Finish events keyed by (time bits, live index) so the order is total.
    let mut finishing: BTreeMap<(u64, usize), Live> = BTreeMap::new();
    let mut waiting: VecDeque<usize> = VecDeque::new();
    let mut live_ids = 0usize;

    let finish = |live: Live, prefix: &mut Prefix, tally: &mut Tally| {
        let r = &reqs[live.req];
        let total = r.input + r.out;
        if state > 0 {
            if total >= 1 {
                let cp = pool.retire(live.lease, total);
                prefix.insert(&live.tokens[..total], cp);
                tally.checkpoints += 1;
            }
        } else {
            let mut lease = live.lease;
            let first = r.input.div_ceil(unit).max(live.hit / unit + 1);
            for k in first..=total / unit {
                if let Ok((cp, _)) = pool.checkpoint(&mut lease, k * unit) {
                    prefix.insert(&live.tokens[..k * unit], cp);
                    tally.checkpoints += 1;
                }
            }
        }
    };

    let mut i = 0usize;
    let mut now = 0.0f64;
    while i < order.len() || !finishing.is_empty() || !waiting.is_empty() {
        let next_admit = order.get(i).map(|&r| offset[reqs[r].session] + reqs[r].t);
        let next_finish = finishing.keys().next().map(|&(bits, _)| f64::from_bits(bits));
        let admit_now = match (next_admit, next_finish) {
            (Some(a), Some(f)) => a <= f && waiting.is_empty(),
            (Some(_), None) => true,
            (None, _) => false,
        };
        if admit_now || (!waiting.is_empty() && next_finish.is_none()) {
            let ri = if let Some(w) = waiting.pop_front() {
                w
            } else {
                i += 1;
                order[i - 1]
            };
            let r = &reqs[ri];
            now = now.max(offset[r.session] + r.t);
            let toks = tokens(&reqs, ri);
            let worst = r.input + r.out;
            let mut woken = false;
            let lease = loop {
                // Room made for a retry may have parked or dropped the
                // checkpoint hit: look it up again where it is now.
                let attempt = match prefix.lookup(&toks[..r.input]) {
                    Some(h) => match h.found {
                        Found::Resident(cp) => pool.restore(&cp, h.len, worst).map(|(l, _)| l),
                        // Woken straight into the lease; the parked
                        // checkpoint stays on the host.
                        Found::Parked(p) => {
                            host.as_ref().expect("a parked hit").restore(&p, &pool, h.len, worst).map(|(l, _)| {
                                woken = true;
                                tally.wakes += 1;
                                tally.wake_tokens += h.len as u64;
                                l
                            })
                        }
                    },
                    None => pool.lease(worst),
                };
                match attempt {
                    Ok(l) => break Some(l),
                    Err(Denied::Busy) => {
                        if !make_room(&mut prefix, host.as_ref(), &mut tally) {
                            break None;
                        }
                    }
                    Err(Denied::Remapping) => {
                        let plan = pool.take_pending().expect("a remap was planned");
                        pool.complete(plan);
                        tally.remaps += 1;
                        tally.max_slots = tally.max_slots.max(pool.slots());
                    }
                    Err(_) => {
                        tally.rejected += 1;
                        break None;
                    }
                }
            };
            let Some(mut lease) = lease else {
                if finishing.is_empty() {
                    tally.rejected += 1; // nothing to wait for
                } else {
                    tally.waited += 1;
                    waiting.push_front(ri);
                    // Fall through to the next finish.
                    let ((bits, _), live) = finishing.pop_first().unwrap();
                    now = f64::from_bits(bits);
                    finish(live, &mut prefix, &mut tally);
                }
                continue;
            };
            let hit_len = lease.prefix();
            tally.input += r.input as u64;
            tally.hit += hit_len as u64;
            tally.extend.push(r.input - hit_len);
            if woken {
                tally.host_hits += 1;
                tally.host_hit_tokens += hit_len as u64;
            } else if hit_len > 0 {
                tally.resident_hits += 1;
                tally.resident_hit_tokens += hit_len as u64;
            }
            if state == 0 {
                for k in (hit_len / unit + 1)..=r.input / unit {
                    if let Ok((cp, _)) = pool.checkpoint(&mut lease, k * unit) {
                        prefix.insert(&toks[..k * unit], cp);
                        tally.checkpoints += 1;
                    }
                }
            }
            let end = now + r.api;
            let id = live_ids;
            live_ids += 1;
            finishing.insert((end.to_bits(), id), Live { lease, req: ri, hit: hit_len, tokens: toks });
            tally.max_live = tally.max_live.max(finishing.len());
        } else {
            let ((bits, _), live) = finishing.pop_first().unwrap();
            now = f64::from_bits(bits);
            finish(live, &mut prefix, &mut tally);
        }
    }
    let main_reqs = reqs.iter().filter(|r| !r.subagent).count();
    println!(
        "{} sessions, {} requests ({main_reqs} main), unit {unit} ({} MiB/page), {}, budget {:.1} GiB in {chunks} chunks of {} MiB: {} pages ({} tokens) and {} slots at the end, {} slots at most, {} remaps, concurrency {concurrency}",
        sessions,
        reqs.len(),
        page_bytes >> 20,
        if state > 0 { format!("recurrent state {} MiB/slot (checkpoint at request end)", state >> 20) } else { "paged only (checkpoint every page)".into() },
        (chunks * chunk) as f64 / (1u64 << 30) as f64,
        chunk >> 20,
        pool.total(),
        pool.total() as u64 * unit as u64,
        pool.slots(),
        tally.max_slots.max(pool.slots()),
        tally.remaps,
    );
    if host.is_some() {
        println!(
            "host tier {host_gib} GiB: {} parks, {} wakes ({} tokens), {} parked dropped for room, peak {:.1} GiB, {} parked at the end",
            tally.parks,
            tally.wakes,
            tally.wake_tokens,
            tally.host_evictions,
            tally.host_peak as f64 / (1u64 << 30) as f64,
            prefix.count(Tier::Parked),
        );
    }
    println!(
        "hit {:.1}% of {} input tokens: {} requests hit resident ({} tokens, {:.1}%), {} hit the host ({} tokens, {:.1}%); extend p50 {} p90 {} p99 {}; {} checkpoints made, {} kept, {} evicted; {} requests waited, {} rejected; max live {}",
        100.0 * tally.hit as f64 / tally.input.max(1) as f64,
        tally.input,
        tally.resident_hits,
        tally.resident_hit_tokens,
        100.0 * tally.resident_hit_tokens as f64 / tally.input.max(1) as f64,
        tally.host_hits,
        tally.host_hit_tokens,
        100.0 * tally.host_hit_tokens as f64 / tally.input.max(1) as f64,
        pct(&mut tally.extend, 0.5),
        pct(&mut tally.extend, 0.9),
        pct(&mut tally.extend, 0.99),
        tally.checkpoints,
        prefix.entries(),
        tally.evictions,
        tally.waited,
        tally.rejected,
        tally.max_live,
    );
}
