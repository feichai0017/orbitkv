//! Kimi-K3 decode gate: replay pegainfer's golden fixture through kern's
//! `decode` program, one sequence per rank, and compare every step's argmax.
//!
//!   cargo run --release -p kern-run --example k3_golden -- \
//!       --manifest examples/k3-4l-ep4.json --weights /data/kern-k3/4l \
//!       --fixture <pegainfer>/pegainfer-k3/tests/fixtures/k3_4l_greedy.json \
//!       --gpus 0,1,2,3 [--graph] [--iters 50] [--margin-abs 0.125]
//!       [--world 8 --rank-base 4 --rendezvous tray04:7400] [--seqs 2 [--mixed --seed 1]]
//!
//! One process per tray: `--world` is the EP size, `--rank-base` the global
//! rank of this process's first GPU, and `--rendezvous` the rank-0 process's
//! address, where fabric handles are swapped over TCP (the rank-0 process
//! listens; every process, itself included, connects and sends its ranks'
//! handles, then receives the whole world's).
//!
//! The fixture feeds `prompt + greedy continuation` one token at a time from
//! position 0 (pure decode, no prefill) and records the reference's argmax
//! and top-5 logits after each. A step whose reference margin is inside the
//! measured noise floor (2 bf16 ULP at the logit's magnitude) is excused when
//! the sampled token is one of the reference's top 5 — pegainfer's own rule
//! (tests/golden_decode.rs). Every rank runs the same sequence, so an EP
//! world must also agree with itself token for token.
//!
//! `--free N` appends N greedy steps past the fixture (row 0's own argmax fed
//! back) and prints them, to read the continuation back through a tokenizer.
//! `K3_GOLDEN_DUMP=<dir>` (with `K3_GOLDEN_DUMP_BUFS=a,b,...`) writes every
//! row's states and the named buffers after each scripted step, so two runs
//! that must agree (a span against per-token steps, say) can be diffed
//! layer by layer and their logits read for the top-2 margin before a
//! differing token is called a bug.
//! `--seqs N` runs N sequences per rank as one batch (`tokens` == `seqs` ==
//! N). Plain: every row feeds the fixture and every row must match row 0
//! token for token. `--mixed`: row 0 feeds the fixture, rows 1.. feed a
//! seeded random prompt that is first run alone (B = 1); the batch rows
//! must reproduce that solo run — rows do not leak into each other.
//! `--fork S` branches two children off row 0 after step S
//! (`Runtime::fork`: pages shared, the page in progress and the KDA state
//! copied). The twin keeps feeding the fixture and must match row 0 token
//! for token inside the same batch; the stray feeds random tokens from
//! there, so row 0 staying the fixture's proves the children write their
//! own pages. The stray is also held to a from-scratch run of its tokens
//! (reported, not judged: the rows around it differ, and that is a
//! near-tie's difference). Every reference batch forks the same way, so
//! rows meet the same batch sizes in both runs (the K2 gate).
//!
//! A manifest with a `tp` group runs tray batches (docs/multi-gpu.md "最终
//! 形态"): `tp` consecutive local ranks hold one batch of `tp * seqs` rows in
//! the "own rows first" layout, every rank stages the whole batch's tokens
//! (its peers' feeds are computable, their free-running tokens are in its
//! own replicated output), and the copies of the batch the ranks compute
//! must agree token for token. `--distinct` then also varies the random
//! rows across the ranks of a group.
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Barrier, Mutex};

use kern_manifest::types::Dim;
use kern_pool::Lease;
use kern_runtime::{PeerHandle, Runtime, Topology};

const NOISE_FLOOR_ULP: f32 = 2.0;

struct Golden {
    feed: Vec<i64>,
    argmax: Vec<i64>,
    top5: Vec<Vec<i64>>,
    top5_logits: Vec<Vec<f32>>,
    num_layers: usize,
    /// An absolute top-1/top-2 margin that counts as a coin flip, for
    /// fixtures whose `top5_logits` are logprobs (tools/k3_oracle_dump.py);
    /// `None` uses the bf16-ULP rule on logits.
    noise_abs: Option<f32>,
}

impl Golden {
    fn load(path: &Path) -> Golden {
        let j: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).expect("fixture")).expect("fixture json");
        let steps = j["steps"].as_array().expect("steps");
        let ints =
            |v: &serde_json::Value| v.as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect::<Vec<_>>();
        Golden {
            feed: steps.iter().map(|s| s["feed"].as_i64().unwrap()).collect(),
            argmax: steps.iter().map(|s| s["argmax"].as_i64().unwrap()).collect(),
            top5: steps.iter().map(|s| ints(&s["top5_ids"])).collect(),
            top5_logits: steps
                .iter()
                .map(|s| s["top5_logits"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect())
                .collect(),
            num_layers: j["num_layers"].as_u64().expect("num_layers") as usize,
            noise_abs: None,
        }
    }

    fn margin_ulp(&self, step: usize) -> f32 {
        let top = self.top5_logits[step][0];
        let ulp = f32::from_bits((top.abs().to_bits() & 0x7f80_0000).max(1)) / 128.0;
        (top - self.top5_logits[step][1]) / ulp
    }

    /// Exact match, or a coin flip the reference itself decided inside the
    /// noise floor with our pick among its top 5.
    fn coin_flip(&self, step: usize) -> bool {
        match self.noise_abs {
            Some(m) => self.top5_logits[step][0] - self.top5_logits[step][1] <= m,
            None => self.margin_ulp(step) <= NOISE_FLOOR_ULP,
        }
    }

    fn accept(&self, step: usize, got: i64) -> (bool, bool) {
        let exact = got == self.argmax[step];
        let excused = !exact && self.coin_flip(step) && self.top5[step].contains(&got);
        (exact, excused)
    }
}

fn stage_cubins(cubins: &Path, kernels: &Path) {
    std::fs::create_dir_all(kernels).unwrap();
    for entry in std::fs::read_dir(cubins).expect("cubins dir") {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "cubin") {
            let bytes = std::fs::read(&path).unwrap();
            let sha = hex::encode(<sha2::Sha256 as sha2::Digest>::digest(&bytes));
            let stem = path.file_stem().unwrap().to_string_lossy();
            std::fs::write(kernels.join(format!("{stem}-{}.cubin", &sha[..12])), &bytes).unwrap();
        }
    }
}

/// The weight blobs one rank needs: the shared dense files plus its expert
/// shard, all memory-mapped.
/// The rank's weight files: the bookends, the dense layers (a `--tp` manifest
/// loads tools/shard_k3_tp.py's `dense-tp{R}/r{me}` slice), its experts.
fn weight_files(weights: &Path, layers: usize, ranks: usize, rank: usize, tp: usize, me: usize) -> Vec<PathBuf> {
    let mut files = vec![weights.join("dense/bookends.safetensors")];
    let dense = if tp > 1 { format!("dense-tp{tp}/r{me}") } else { "dense".to_string() };
    for i in 0..layers {
        files.push(weights.join(format!("{dense}/l{i}.safetensors")));
    }
    for i in 1..layers {
        files.push(weights.join(format!("experts/ep{ranks}-r{rank}-l{i}.safetensors")));
    }
    files
}

struct Outcome {
    tokens: Vec<i64>,
    /// greedy continuation past the fixture (`--free N`), row 0
    free: Vec<i64>,
    /// Every rank of the tray group's rows as this rank computed them, by
    /// group index and row: the group's replicated output must agree.
    table: Vec<Vec<Vec<i64>>>,
    checked: usize,
    exact: usize,
    excused: usize,
    failures: Vec<String>,
    step_ms: Option<f64>,
    /// the `decode_span` step's replay time (`--span`), same batch as the last step
    span_ms: Option<f64>,
}

type Handles = BTreeMap<String, PeerHandle>;

fn write_handles(w: &mut impl Write, ranks: &[(u64, Handles)]) -> std::io::Result<()> {
    w.write_all(&(ranks.len() as u32).to_le_bytes())?;
    for (rank, map) in ranks {
        w.write_all(&rank.to_le_bytes())?;
        w.write_all(&(map.len() as u32).to_le_bytes())?;
        for (name, h) in map {
            w.write_all(&(name.len() as u16).to_le_bytes())?;
            w.write_all(name.as_bytes())?;
            let wire = h.to_bytes().ok_or_else(|| std::io::Error::other(format!("`{name}`: not a fabric handle")))?;
            w.write_all(&wire)?;
        }
    }
    w.flush()
}

fn read_handles(r: &mut impl Read) -> std::io::Result<Vec<(u64, Handles)>> {
    fn u32_of(r: &mut impl Read) -> std::io::Result<u32> {
        let mut b = [0u8; 4];
        r.read_exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }
    let n = u32_of(r)?;
    let mut out = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let mut b = [0u8; 8];
        r.read_exact(&mut b)?;
        let rank = u64::from_le_bytes(b);
        let entries = u32_of(r)?;
        let mut map = BTreeMap::new();
        for _ in 0..entries {
            let mut l = [0u8; 2];
            r.read_exact(&mut l)?;
            let mut name = vec![0u8; u16::from_le_bytes(l) as usize];
            r.read_exact(&mut name)?;
            let mut h = [0u8; PeerHandle::BYTES];
            r.read_exact(&mut h)?;
            map.insert(String::from_utf8(name).unwrap(), PeerHandle::from_bytes(&h).unwrap());
        }
        out.push((rank, map));
    }
    Ok(out)
}

/// The rank-0 process: collect every rank's handles from `world` ranks'
/// worth of connections, then send the whole table back on each.
fn serve_rendezvous(listener: TcpListener, world: usize) -> std::io::Result<()> {
    let mut streams = Vec::new();
    let mut table: Vec<(u64, Handles)> = Vec::new();
    while table.len() < world {
        let (mut s, _) = listener.accept()?;
        table.extend(read_handles(&mut s)?);
        streams.push(s);
    }
    table.sort_by_key(|(r, _)| *r);
    for s in &mut streams {
        write_handles(s, &table)?;
    }
    Ok(())
}

fn exchange(addr: &str, mine: &[(u64, Handles)], world: usize) -> anyhow::Result<Vec<Handles>> {
    let mut s = None;
    for _ in 0..600 {
        match TcpStream::connect(addr) {
            Ok(c) => {
                s = Some(c);
                break;
            }
            Err(_) => std::thread::sleep(std::time::Duration::from_millis(500)),
        }
    }
    let mut s = s.ok_or_else(|| anyhow::anyhow!("rendezvous {addr}: no listener after 5 minutes"))?;
    write_handles(&mut s, mine)?;
    let table = read_handles(&mut s)?;
    anyhow::ensure!(table.len() == world, "rendezvous returned {} ranks, world is {world}", table.len());
    for (i, (r, _)) in table.iter().enumerate() {
        anyhow::ensure!(*r == i as u64, "rendezvous table has rank {r} at index {i}");
    }
    Ok(table.into_iter().map(|(_, m)| m).collect())
}

/// The per-rank batch: one lease per row, staged as `[seqs]`-shaped inputs.
/// One rank's view of a tray batch: `leases[q][row]` holds group rank q's
/// row. Every rank leases every row of the tray batch, since the
/// head-sharded KDA state of a row lives on all ranks (docs/multi-gpu.md
/// "最终形态"); the pages behind a peer's lease are its MLA KV, which only
/// the owner writes and reads. Alone (`tp` = 1) the group is this rank.
struct Batch {
    leases: Vec<Vec<Lease>>,
    me: usize,
    seqs_max: usize,
    rows_max: usize,
    /// Each row's position (tokens fed so far); a span moves one row by
    /// many, so rows drift apart.
    pos: Vec<usize>,
}

impl Batch {
    fn new(rt: &mut Runtime, tp: usize, me: usize, rows: usize, tokens_per_row: usize) -> anyhow::Result<Batch> {
        let seqs_max = rt.manifest.vars["seqs"].max as usize;
        let rows_max = rt.manifest.vars["rows"].max as usize;
        anyhow::ensure!(rows <= seqs_max, "{rows} rows, manifest seqs bound is {seqs_max}");
        let leases = (0..tp)
            .map(|_| (0..rows).map(|_| rt.lease(tokens_per_row)).collect::<Result<Vec<_>, _>>())
            .collect::<Result<Vec<_>, _>>()?;
        let b = Batch { leases, me, seqs_max, rows_max, pos: vec![0; rows] };
        b.stage_tables(rt, &(0..rows).collect::<Vec<_>>())?;
        // K3's KDA kernels take the span's first row from an input; the
        // span (when there is one) is row 0's tokens, at the front.
        if rt.manifest.buffers.contains_key("span_at") {
            rt.write_input("span_at", &0i32.to_le_bytes())?;
        }
        Ok(b)
    }

    fn own(&self) -> &[Lease] {
        &self.leases[self.me]
    }

    /// Page-table rows: batch row i = own lease `cells[i]` (a span repeats
    /// its row's lease), the rest (never dereferenced, but domain-checked)
    /// repeat the last. Line-table columns are the tray batch's rows, own
    /// rows first, then rank (me + d)'s at block d.
    fn stage_tables(&self, rt: &mut Runtime, cells: &[usize]) -> anyhow::Result<()> {
        let tp = self.leases.len();
        let b = cells.len();
        anyhow::ensure!(b <= self.seqs_max, "{b} batch rows, manifest seqs bound is {}", self.seqs_max);
        let tables: Vec<String> = rt.page_tables().map(str::to_string).collect();
        for name in tables {
            let mut table = Vec::new();
            for i in 0..self.seqs_max {
                self.own()[cells[i.min(b - 1)]].extend_row(&name, &mut table)?;
            }
            rt.write_input(&name, &le_bytes_i32(&table))?;
        }
        let lines: Vec<String> = rt.seq_tables().map(str::to_string).collect();
        for name in lines {
            let l0 = &self.own()[0];
            let w = l0.seq_width(&name)?;
            anyhow::ensure!(w == 1, "`{name}`: wide line tables are not staged here");
            let mut table = Vec::new();
            for r in 0..l0.seq_lines(&name)? {
                for i in 0..self.rows_max {
                    let (d, j) = (i / b, i % b);
                    table.push(if d < tp { self.leases[(self.me + d) % tp][cells[j]].seq_line(&name, r)? } else { 0 });
                }
            }
            rt.write_input(&name, &le_bytes_i32(&table))?;
        }
        Ok(())
    }

    /// A child of row `row` from the batch's position, on every rank of the
    /// group: one more row each, the tables restaged.
    fn fork(&mut self, rt: &mut Runtime, row: usize, tokens_per_row: usize) -> anyhow::Result<()> {
        anyhow::ensure!(self.own().len() < self.seqs_max, "no row for a fork: {} rows", self.seqs_max);
        for q in 0..self.leases.len() {
            let child = rt.fork(&mut self.leases[q][row], self.pos[row], tokens_per_row)?;
            self.leases[q].push(child);
        }
        self.pos.push(self.pos[row]);
        self.stage_tables(rt, &(0..self.own().len()).collect::<Vec<_>>())
    }

    /// Stage one step: this rank's rows as batch rows 0.., row r
    /// contributing `toks[r]` (one token, or a span of them: consecutive
    /// positions of that row, its tables repeated), then rank (me + d)'s
    /// rows at block d from `peer(q, row)` (docs/multi-gpu.md "own rows
    /// first"). Returns the vars and the program it selects, and moves the
    /// rows' positions.
    fn stage(
        &mut self,
        rt: &mut Runtime,
        toks: &[Vec<i64>],
        peer: &dyn Fn(usize, usize) -> i64,
    ) -> anyhow::Result<(BTreeMap<String, u64>, &'static str)> {
        let (tp, me) = (self.leases.len(), self.me);
        let b = self.own().len();
        assert_eq!(toks.len(), b);
        let span = toks[0].len();
        anyhow::ensure!(toks[1..].iter().all(|t| t.len() <= 1), "only row 0 may carry a span");
        anyhow::ensure!(span == 1 || tp == 1, "a span in a tray batch is not staged here");
        // The tables follow the batch rows every step: a span row's lease
        // fills `span` rows this step and one the next; a row with nothing
        // to feed (done while others catch up) is not in the batch.
        let cells: Vec<usize> = (0..b).flat_map(|r| std::iter::repeat_n(r, toks[r].len())).collect();
        let n = cells.len();
        self.stage_tables(rt, &cells)?;
        let mut e = BTreeMap::from([
            ("tokens".to_string(), n as u64),
            ("seqs".to_string(), n as u64),
            ("rows".to_string(), (tp * n) as u64),
        ]);
        if span > 1 {
            e.insert("span".to_string(), span as u64);
        }
        let mut all: Vec<i64> = toks.concat();
        for d in 1..tp {
            let q = (me + d) % tp;
            all.extend((0..b).map(|j| peer(q, j)));
        }
        rt.write_input_at("token_ids", &le_bytes_i64(&all), &e)?;
        // The tray's blocks, equal here: rank (me + d)'s rows at block d.
        if rt.manifest.buffers.contains_key("tp_blocks") {
            let blocks: Vec<i32> = (0..=tp).map(|d| (d * n) as i32).collect();
            rt.write_input("tp_blocks", &le_bytes_i32(&blocks))?;
        }
        let mut slots = Vec::with_capacity(n);
        let mut lens = Vec::with_capacity(n);
        for (r, t) in toks.iter().enumerate() {
            for j in 0..t.len() {
                slots.push(self.own()[r].slot(self.pos[r] + j));
                lens.push((self.pos[r] + j + 1) as i32);
            }
        }
        rt.write_input_at("slot_mapping", &le_bytes_i64(&slots), &e)?;
        rt.write_input_at("seq_lens", &le_bytes_i32(&lens), &e)?;
        for (r, t) in toks.iter().enumerate() {
            self.pos[r] += t.len();
        }
        Ok((e, if span > 1 { "decode_span" } else { "decode" }))
    }
}

fn le_bytes_i64(v: &[i64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn le_bytes_i32(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Run a batch: `feeds[q]` is rank `q`'s rows (one token list per row, all
/// the same length) for every rank of this rank's tray group, `me` this
/// rank's index in it; every rank runs the same schedule, so each can stage
/// its peers' tokens from the feeds and, past them, from the tray batch's
/// own replicated output. Returns every rank's sampled tokens by group index
/// and row (`steps + free` of them) and, with `iters > 0`, the replay time.
/// `fork` = (step, stray feeds): after that many steps two children of row
/// 0 join as the last rows on every rank, the twin feeding row 0's tokens
/// and the stray `stray[q][step..]`; their tokens before the fork are empty.
/// `span` > 1: row 0 feeds its scripted tokens `span` at a time through
/// `decode_span` steps (a prompt as chunks; the predictions for a chunk
/// come out together), then generates one by one; the other rows step one
/// token at a time throughout, so row 0 runs ahead. The timings, with
/// `iters` > 0, are the last decode step's replay and the last span
/// step's.
#[allow(clippy::too_many_arguments)]
fn run_batch(
    rt: &mut Runtime,
    feeds: &[Vec<Vec<i64>>],
    me: usize,
    tokens_per_row: usize,
    graph: bool,
    iters: usize,
    free: usize,
    fork: Option<(usize, &[Vec<i64>])>,
    span: usize,
    span_from: usize,
) -> anyhow::Result<(Vec<Vec<Vec<i64>>>, Option<f64>, Option<f64>)> {
    let tp = feeds.len();
    let rows = feeds[me].len();
    let steps = feeds[me][0].len();
    anyhow::ensure!(span <= steps, "a span of {span} needs {span} fixture tokens, the fixture has {steps}");
    anyhow::ensure!(span <= 1 || fork.is_none(), "--span and --fork together are not run here");
    let mut batch = Batch::new(rt, tp, me, rows, tokens_per_row)?;
    let mut out: Vec<Vec<Vec<i64>>> = vec![vec![Vec::with_capacity(steps + free); rows]; tp];
    let mut vars = BTreeMap::new();
    let mut span_vars = None;
    // After the scripted feed, `free` more steps run each row on its own
    // argmax (a greedy continuation, for reading the text back). A row's
    // k-th token is its feed's, then its own (k-1)-th output.
    let mut step = 0;
    while out[me].iter().any(|o| o.len() < steps + free) {
        if fork.is_some_and(|(at, _)| at == step) {
            for _ in 0..2 {
                batch.fork(rt, 0, tokens_per_row)?;
                for o in out.iter_mut() {
                    o.push(Vec::with_capacity(steps + free - step));
                }
            }
        }
        let feed = |q: usize, r: usize, k: usize| match fork {
            Some((_, stray)) if r == rows + 1 => stray[q][k],
            Some(_) if r == rows => feeds[q][0][k],
            _ => feeds[q][r][k],
        };
        let token = |q: usize, r: usize, k: usize| if k < steps { feed(q, r, k) } else { out[q][r][k - 1] };
        let counts: Vec<usize> = (0..out[me].len())
            .map(|r| match out[me][r].len() {
                fed if fed >= steps + free => 0,
                fed if r == 0 && fed < steps && fed >= span_from => span.max(1).min(steps - fed),
                _ => 1,
            })
            .collect();
        let count = |r: usize| counts[r];
        let toks: Vec<Vec<i64>> =
            (0..out[me].len()).map(|r| (0..count(r)).map(|j| token(me, r, out[me][r].len() + j)).collect()).collect();
        let peer = |q: usize, r: usize| token(q, r, out[q][r].len());
        let (e, program) = batch.stage(rt, &toks, &peer)?;
        if graph {
            if !rt.is_captured(program, &e) {
                rt.capture(program, &e)?;
            }
            rt.run_captured(program, &e)?;
        } else {
            rt.run(program, &e)?;
        }
        if program == "decode_span" {
            span_vars = Some(e.clone());
        } else {
            vars = e;
        }
        let bytes = rt.read_output("next_token")?;
        let b = out[me].len();
        let n: usize = (0..b).map(count).sum();
        for (q, o) in out.iter_mut().enumerate() {
            let block = (q + tp - me) % tp;
            let mut at = block * n;
            for (r, row) in o.iter_mut().enumerate() {
                for _ in 0..count(r) {
                    row.push(i64::from_le_bytes(bytes[at * 8..at * 8 + 8].try_into().unwrap()));
                    at += 1;
                }
            }
        }
        if let Ok(dir) = std::env::var("K3_GOLDEN_DUMP") {
            if rt.rank("ep").unwrap_or(0) == 0 && step <= steps {
                let pos: Vec<usize> = out[me].iter().map(Vec::len).collect();
                dump_rows(rt, batch.own(), &pos, step, Path::new(&dir))?;
            }
        }
        step += 1;
    }
    for o in out.iter_mut().flat_map(|o| o.iter_mut()) {
        o.truncate(steps + free);
    }
    let time = |rt: &mut Runtime, program: &str, e: &BTreeMap<String, u64>| -> anyhow::Result<f64> {
        if !rt.is_captured(program, e) {
            rt.capture(program, e)?;
        }
        Ok(rt.time_captured(program, e, iters)? as f64)
    };
    // no decode step ran when the span covered the whole feed
    let ms = if iters > 0 && !vars.is_empty() { Some(time(rt, "decode", &vars)?) } else { None };
    let span_ms = match (&span_vars, iters) {
        (Some(e), 1..) => Some(time(rt, "decode_span", e)?),
        _ => None,
    };
    Ok((out, ms, span_ms))
}

/// Debug aid (`K3_GOLDEN_DUMP=<dir>`): every row's per-sequence state
/// (the whole slot), its per-token state (the slots fed so far) and the
/// `hidden` buffer after a step, as `r<row>-p<pos>-<state>.bin` /
/// `s<step>-hidden.bin`, for diffing two runs that must agree.
fn dump_rows(rt: &Runtime, leases: &[Lease], pos: &[usize], step: usize, dir: &Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir)?;
    let states: Vec<(String, bool, u64)> = rt
        .state_sizes()
        .iter()
        .map(|(n, st, _)| {
            (n.to_string(), st.is_per_seq(), if st.is_per_seq() { st.bytes_per_seq } else { st.bytes_per_token })
        })
        .collect();
    let tables: Vec<String> = rt.seq_tables().map(str::to_string).collect();
    for (r, lease) in leases.iter().enumerate() {
        for (name, per_seq, bytes) in &states {
            let data = if *per_seq {
                let Some(table) = tables.iter().find(|t| t.starts_with(&format!("{name}."))) else { continue };
                let lines = lease.seq_lines(table)? as u64;
                let line0 = lease.seq_line(table, 0)? as u64;
                rt.read_state_at(name, (line0 * (bytes / lines)) as usize, *bytes as usize)?
            } else {
                // whole pages, in position order (a page's inner layout is the kernels')
                let page = rt.page() as usize;
                let mut v = Vec::new();
                for &p in &lease.page_ids()[..pos[r].div_ceil(page)] {
                    v.extend(rt.read_state_at(name, p as usize * page * *bytes as usize, page * *bytes as usize)?);
                }
                v
            };
            std::fs::write(dir.join(format!("r{r}-p{}-{name}.bin", pos[r])), data)?;
        }
    }
    let bufs = std::env::var("K3_GOLDEN_DUMP_BUFS").unwrap_or_else(|_| "hidden".to_string());
    for name in bufs.split(',').filter(|n| rt.manifest.buffers.contains_key(*n)) {
        std::fs::write(dir.join(format!("s{step}-{name}.bin")), rt.read_buffer(name)?)?;
    }
    Ok(())
}

/// A seeded random prompt in the fixture's vocabulary, `len` tokens.
fn random_feed(len: usize, vocab: i64, seed: u64) -> Vec<i64> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x >> 11) % vocab as u64) as i64
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn run_rank(
    json: &str,
    kernels: &Path,
    gpu: usize,
    topo: &Topology,
    files: &[PathBuf],
    golden: &Golden,
    graph: bool,
    iters: usize,
    seqs: usize,
    mixed: bool,
    distinct: bool,
    seed: u64,
    free: usize,
    fork: Option<usize>,
    span: usize,
    span_from: usize,
    rendezvous: &dyn Fn(&mut Runtime) -> kern_runtime::Result<()>,
) -> anyhow::Result<Outcome> {
    let manifest = kern_manifest::Verified::from_json(json)?;
    let table = &manifest.buffers["block_table"];
    let per_row = match table.shape.as_slice() {
        [_, Dim::Const(pages)] => pages * table.domain.as_ref().map(|d| d.stride).unwrap_or(1),
        s => anyhow::bail!("unexpected block_table shape {s:?}"),
    };
    let per_row = per_row as usize;
    // The tray group this rank runs one batch with (`tp` in the manifest's
    // topology); alone, a group of one. Every rank leases the whole tray
    // batch's rows.
    let (me, tp) = topo.groups.get("tp").map(|g| (g.index as usize, g.size as usize)).unwrap_or((0, 1));
    // Every row of the tray batch, forks included, leased on this rank.
    let rows = ((seqs.max(1) + 2 * fork.is_some() as usize) * tp) as u64;
    let capacity = kern_runtime::Capacity { tokens: Some(per_row as u64 * rows), seqs: rows };
    let mut rt = Runtime::load(&manifest, kernels, gpu, Some(capacity), Some(topo))?;
    let maps: Vec<memmap2::Mmap> = files
        .iter()
        .map(|f| {
            let file = std::fs::File::open(f).map_err(|e| anyhow::anyhow!("{}: {e}", f.display()))?;
            Ok(unsafe { memmap2::Mmap::map(&file)? })
        })
        .collect::<anyhow::Result<_>>()?;
    let blobs: Vec<&[u8]> = maps.iter().map(|m| &m[..]).collect();
    rt.load_weights(&kern_runtime::Safetensors::parse(&blobs)?)?;
    rendezvous(&mut rt)?;
    // A tray manifest's one-time setup after the peers are mapped (the
    // allreduce's Lamport stages are poisoned, not zeroed).
    let vars =
        BTreeMap::from([("tokens".to_string(), 1u64), ("seqs".to_string(), 1u64), ("rows".to_string(), tp as u64)]);
    for (name, p) in &manifest.programs {
        if p.once {
            rt.run(name, &vars)?;
        }
    }

    let steps = golden.feed.len();
    let vocab = manifest.buffers["embed"].shape[0].clone();
    let vocab = match vocab {
        Dim::Const(v) => v as i64,
        _ => 163840,
    };
    let mut out = Outcome {
        tokens: Vec::new(),
        free: Vec::new(),
        table: Vec::new(),
        checked: 0,
        exact: 0,
        excused: 0,
        failures: Vec::new(),
        step_ms: None,
        span_ms: None,
    };
    // Every distinct feed first runs as a batch of `seqs` copies of itself,
    // so the mixed batch can be held to it row by row at the same batch
    // size (cuBLAS picks its kernels by m, so B = 1 and B = 8 legitimately
    // differ at near-ties; what must not differ is a row's result with other
    // rows' content around it). `--mixed` gives rows 1.. a random prompt
    // (`--distinct`: a different one per row, and per rank of the group);
    // plain mode feeds the fixture everywhere and only checks row agreement.
    // The feeds of every rank in the group are built here, since each rank
    // stages the whole tray batch; the schedule below is the same on all of
    // them, so the group's runs stay in step.
    let feeds_of = |q: usize| -> Vec<Vec<i64>> {
        (0..seqs)
            .map(|r| {
                if mixed && r > 0 {
                    random_feed(steps, vocab, if distinct { seed + (q * seqs + r) as u64 } else { seed })
                } else {
                    golden.feed.clone()
                }
            })
            .collect()
    };
    let feeds: Vec<Vec<Vec<i64>>> = (0..tp).map(feeds_of).collect();
    let mut solo: Vec<Option<Vec<i64>>> = vec![None; seqs];
    if mixed && seqs > 1 {
        for r in 0..seqs {
            if let Some(same) = (0..r).find(|&q| feeds[me][q] == feeds[me][r]) {
                solo[r] = solo[same].clone();
                continue;
            }
            let copies: Vec<Vec<Vec<i64>>> = (0..tp).map(|q| vec![feeds[q][r].clone(); seqs]).collect();
            let strays: Vec<Vec<i64>> = (0..tp).map(|q| feeds[q][r].clone()).collect();
            let shape = fork.map(|at| (at, strays.as_slice()));
            let (t, _, _) = run_batch(&mut rt, &copies, me, per_row, graph, 0, 0, shape, 0, 0)?;
            solo[r] = Some(t[me][0].clone());
        }
    }
    // The stray child: the fixture up to the fork, random after it. Every
    // reference batch forks the same way at the same step, so a row meets
    // the same batch sizes (the same kernels) in both runs.
    let stray: Option<Vec<i64>> = fork.map(|at| {
        let mut f = golden.feed[..at].to_vec();
        f.extend(random_feed(steps - at, vocab, seed + 97));
        f
    });
    let strays: Vec<Vec<i64>> = vec![stray.clone().unwrap_or_default(); tp];
    let stray_solo = match (&stray, fork) {
        (Some(f), Some(at)) => {
            let copies = vec![vec![f.clone(); seqs]; tp];
            let (t, _, _) = run_batch(&mut rt, &copies, me, per_row, graph, 0, 0, Some((at, strays.as_slice())), 0, 0)?;
            Some(t[me][0].clone())
        }
        _ => None,
    };
    let (table, ms, span_ms) = run_batch(
        &mut rt,
        &feeds,
        me,
        per_row,
        graph,
        iters,
        free,
        fork.map(|at| (at, strays.as_slice())),
        span,
        span_from,
    )?;
    let tokens = &table[me];
    out.step_ms = ms;
    out.span_ms = span_ms;
    out.tokens = tokens[0][..steps].to_vec();
    out.free = tokens[0][steps..].to_vec();
    if tp > 1 {
        let err = i32::from_le_bytes(rt.read_output("tp_err")?[..4].try_into().unwrap());
        if err != 0 {
            out.failures.push(format!("a tray collective never heard from group rank {}", err - 1));
        }
    }
    for (step, &got) in tokens[0][..steps].iter().enumerate() {
        if golden.argmax[step] < 0 {
            continue; // fed but unchecked (long-prompt fixture, tools/k3_oracle_dump.py --check-last)
        }
        out.checked += 1;
        let (exact, excused) = golden.accept(step, got);
        if exact {
            out.exact += 1;
        } else if excused {
            out.excused += 1;
        } else {
            out.failures.push(format!(
                "step {step}: got {got}, reference {} (margin {:.1} ulp, top5 {:?})",
                golden.argmax[step],
                golden.margin_ulp(step),
                golden.top5[step]
            ));
        }
    }
    // Plain mode: every row is the fixture, so every row is held to it
    // (rows past 0 step one token at a time whatever `--span` says).
    if !mixed {
        for r in 1..seqs {
            let bad: Vec<String> = (0..steps)
                .filter(|&i| golden.argmax[i] >= 0 && !matches!(golden.accept(i, tokens[r][i]), (true, _) | (_, true)))
                .map(|i| format!("step {i}: got {}, reference {}", tokens[r][i], golden.argmax[i]))
                .collect();
            if !bad.is_empty() {
                out.failures.push(format!("row {r} off the fixture at {} steps: {}", bad.len(), bad.join("; ")));
            }
        }
    }
    for r in 0..seqs {
        let Some(want) = &solo[r] else { continue };
        if &tokens[r][..steps] != want.as_slice() {
            let first = (0..steps).find(|&i| tokens[r][i] != want[i]).unwrap();
            out.failures.push(format!(
                "row {r} diverges from a batch of its own copies at step {first}: got {}, expected {}",
                tokens[r][first], want[first]
            ));
        }
    }
    if let (Some(at), Some(want)) = (fork, &stray_solo) {
        let twin = &tokens[seqs][..steps - at];
        match (0..steps - at).find(|&i| twin[i] != tokens[0][at + i]) {
            Some(i) => out.failures.push(format!(
                "the twin forked at step {at} diverges from row 0 at step {}: got {}, row 0 has {}",
                at + i,
                twin[i],
                tokens[0][at + i]
            )),
            None => println!("  fork at step {at}: the twin's {} tokens match row 0's", steps - at),
        }
        let stray = &tokens[seqs + 1][..steps - at];
        match (0..steps - at).find(|&i| stray[i] != want[at + i]) {
            Some(i) => println!(
                "  the stray forked at step {at} leaves its from-scratch run at step {} ({} of {} match)",
                at + i,
                i,
                steps - at
            ),
            None => println!("  the stray's {} tokens match its from-scratch run", steps - at),
        }
    }
    out.table = table;
    Ok(out)
}

fn main() {
    let mut manifest = PathBuf::from("examples/k3-4l-ep1.json");
    let mut weights = PathBuf::from("/data/kern-k3/4l");
    let mut fixture = PathBuf::from("tests/fixtures/k3_4l_greedy.json");
    let mut cubins = PathBuf::from("target/cubins");
    let mut gpus: Vec<usize> = vec![0];
    let mut graph = false;
    let mut iters = 0usize;
    let mut margin_abs: Option<f32> = None;
    let mut world: Option<usize> = None;
    let mut rank_base = 0usize;
    let mut rendezvous_addr: Option<String> = None;
    let mut seqs = 1usize;
    let mut mixed = false;
    let mut distinct = false;
    let mut seed = 1u64;
    let mut free = 0usize;
    let mut fork: Option<usize> = None;
    let mut span = 0usize;
    let mut span_from = 0usize;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut v = || args.next().expect("value");
        match a.as_str() {
            "--manifest" => manifest = PathBuf::from(v()),
            "--weights" => weights = PathBuf::from(v()),
            "--fixture" => fixture = PathBuf::from(v()),
            "--cubins" => cubins = PathBuf::from(v()),
            "--gpus" => gpus = v().split(',').map(|s| s.parse().unwrap()).collect(),
            "--graph" => graph = true,
            "--iters" => iters = v().parse().unwrap(),
            "--margin-abs" => margin_abs = Some(v().parse().unwrap()),
            "--world" => world = Some(v().parse().unwrap()),
            "--rank-base" => rank_base = v().parse().unwrap(),
            "--rendezvous" => rendezvous_addr = Some(v()),
            "--seqs" => seqs = v().parse().unwrap(),
            "--mixed" => mixed = true,
            "--distinct" => distinct = true,
            "--seed" => seed = v().parse().unwrap(),
            "--free" => free = v().parse().unwrap(),
            "--fork" => fork = Some(v().parse().unwrap()),
            "--span" => span = v().parse().unwrap(),
            "--span-from" => span_from = v().parse().unwrap(),
            _ => panic!("unknown arg {a}"),
        }
    }
    let json = std::fs::read_to_string(&manifest).expect("manifest");
    let mut golden = Golden::load(&fixture);
    golden.noise_abs = margin_abs;
    let golden = Arc::new(golden);
    let n = gpus.len();
    let world = world.unwrap_or(n);
    assert!(rank_base + n <= world, "ranks {rank_base}..{} exceed world {world}", rank_base + n);
    // A `tp` group is a tray batch: `tp` consecutive local ranks, all in
    // this process.
    let tp =
        kern_manifest::types::Manifest::from_json(&json).ok().and_then(|m| m.group_size("tp")).unwrap_or(1) as usize;
    assert!(n.is_multiple_of(tp), "{n} local ranks do not split into tray groups of {tp}");
    let kernels = std::env::temp_dir().join(format!("kern-k3-golden-{}", std::process::id()));
    stage_cubins(&cubins, &kernels);
    println!(
        "{}: {} layers, EP{world} ranks {rank_base}..{} on gpus {gpus:?}{}, {} fixture steps, {}{}",
        manifest.display(),
        golden.num_layers,
        rank_base + n,
        if tp > 1 { format!(", tray batches of TP{tp}") } else { String::new() },
        golden.feed.len(),
        if graph { "graph replay" } else { "eager" },
        if seqs > 1 {
            format!(", {seqs} rows/rank{}", if mixed { " (row 0 fixture, rest random)" } else { "" })
        } else {
            String::new()
        }
    );
    if world > n {
        let addr = rendezvous_addr.clone().expect("--rendezvous is required when the world spans processes");
        if rank_base == 0 {
            let port = addr.rsplit(':').next().unwrap().to_string();
            let listener = TcpListener::bind(format!("0.0.0.0:{port}")).expect("bind rendezvous port");
            std::thread::spawn(move || serve_rendezvous(listener, world).expect("rendezvous server"));
        }
    }

    let posted: Arc<Mutex<Vec<Option<Handles>>>> = Arc::new(Mutex::new(vec![None; n]));
    let table: Arc<Mutex<Option<Vec<Handles>>>> = Arc::new(Mutex::new(None));
    let gate = Arc::new(Barrier::new(n));
    let results: Arc<Mutex<Vec<Option<Result<Outcome, String>>>>> =
        Arc::new(Mutex::new((0..n).map(|_| None).collect()));
    let mut threads = Vec::new();
    for (local, &gpu) in gpus.iter().enumerate() {
        let rank = rank_base + local;
        let (json, kernels, posted, table, gate, results, golden, weights, addr) = (
            json.clone(),
            kernels.clone(),
            posted.clone(),
            table.clone(),
            gate.clone(),
            results.clone(),
            golden.clone(),
            weights.clone(),
            rendezvous_addr.clone(),
        );
        threads.push(std::thread::spawn(move || {
            let files = weight_files(&weights, golden.num_layers, world, rank, tp, local % tp);
            let rendezvous = |rt: &mut Runtime| -> kern_runtime::Result<()> {
                let mine = rt.export_handles()?;
                posted.lock().unwrap()[local] = Some(mine);
                gate.wait();
                if local == 0 {
                    let ours: Vec<(u64, Handles)> = posted
                        .lock()
                        .unwrap()
                        .iter()
                        .enumerate()
                        .map(|(i, m)| ((rank_base + i) as u64, m.clone().unwrap()))
                        .collect();
                    let members = if world > n {
                        exchange(addr.as_deref().unwrap(), &ours, world)
                            .map_err(|e| kern_runtime::Error::Api(format!("{e:#}")))?
                    } else {
                        ours.into_iter().map(|(_, m)| m).collect()
                    };
                    *table.lock().unwrap() = Some(members);
                }
                gate.wait();
                let members = table.lock().unwrap().clone().unwrap();
                rt.import_peers("ep", &members)?;
                if tp > 1 {
                    let group: Vec<Handles> =
                        posted.lock().unwrap()[local / tp * tp..][..tp].iter().map(|m| m.clone().unwrap()).collect();
                    rt.import_peers("tp", &group)?;
                }
                Ok(())
            };
            let mut topo = Topology::one("ep", rank as u64, world as u64);
            if tp > 1 {
                topo.groups
                    .insert("tp".to_string(), kern_runtime::GroupRank { index: (local % tp) as u64, size: tp as u64 });
            }
            let r = run_rank(
                &json,
                &kernels,
                gpu,
                &topo,
                &files,
                &golden,
                graph,
                iters,
                seqs,
                mixed,
                distinct,
                seed,
                free,
                fork,
                span,
                span_from,
                &rendezvous,
            );
            results.lock().unwrap()[local] = Some(r.map_err(|e| format!("{e:#}")));
        }));
    }
    for th in threads {
        th.join().unwrap();
    }
    let results = results.lock().unwrap();
    let mut ok = true;
    let mut first: Option<Vec<i64>> = None;
    for (local, r) in results.iter().enumerate() {
        let rank = rank_base + local;
        match r {
            Some(Ok(o)) => {
                let steps = golden.feed.len();
                let fed = if o.checked == steps { String::new() } else { format!(" (of {steps} fed)") };
                println!(
                    "rank {rank} gpu {}: {}/{}{fed} exact, {} excused inside the noise floor, {} wrong{}",
                    gpus[local],
                    o.exact,
                    o.checked,
                    o.excused,
                    o.failures.len(),
                    o.step_ms.map(|ms| format!("; {ms:.3} ms/step (captured, {iters} iters)")).unwrap_or_default()
                );
                if let Some(ms) = o.span_ms {
                    println!("  span step ({span} rows of one sequence): {ms:.3} ms (captured, {iters} iters)");
                }
                for f in &o.failures {
                    println!("  {f}");
                }
                if !o.free.is_empty() {
                    println!("  free continuation: {:?}", o.free);
                }
                ok &= o.failures.is_empty();
                match &first {
                    None => first = Some(o.tokens.clone()),
                    Some(t) if *t != o.tokens => {
                        println!("  rank {rank} disagrees with rank {rank_base} on the sampled tokens");
                        ok = false;
                    }
                    _ => {}
                }
                // A tray batch is replicated: every rank of the group holds
                // every rank's rows, and the copies must be the same tokens.
                let head = local / tp * tp;
                if let Some(Ok(h)) = &results[head] {
                    if local != head && h.table != o.table {
                        println!("  rank {rank}'s copy of the tray batch differs from rank {}'s", rank_base + head);
                        ok = false;
                    }
                }
            }
            Some(Err(e)) => {
                println!("rank {rank} gpu {}: FAILED: {e}", gpus[local]);
                ok = false;
            }
            None => unreachable!(),
        }
    }
    println!("{}", if ok { "PASS" } else { "FAIL" });
    if !ok {
        std::process::exit(1);
    }
}
