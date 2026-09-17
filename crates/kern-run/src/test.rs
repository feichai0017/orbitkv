//! `kern test`: the evidence for a kernel swap, over the loaded runtime.
//!
//! The harness itself is `kern-test` (`docs/test.md`): the static diff,
//! the seeded workload recorded on A, replayed on B, noise floor, perf
//! and the verdict, written over [`kern_test::Side`]. This module
//! resolves the flags against `kern.toml`, loads A, records, drops A,
//! loads B, replays; and prints or archives the report. The side is
//! [`Ranks`]: the manifest on one GPU, or on one GPU per rank of its
//! topology, each rank a [`Caller`] over its own runtime.

use std::ops::Range;
use std::path::PathBuf;
use std::time::Instant;

use crate::config::{Config, Target};
use crate::{Caller, Given, Inputs};
use anyhow::{bail, Context, Result};
use clap::Args;
use kern_manifest::types::{DType, Provision};
use kern_manifest::{Protocol, Verified};
use kern_runtime::{Capacity, HostWeights, Resident, Runtime, Scratch};
use kern_test::compare::{Cmp, LogitStats, TOP};
use kern_test::report::{plural, row, Report, Verdict};
use kern_test::{At, Options, Side, Vars};
use serde_json::json;

/// Flags of `kern test`; anything not given comes from the target /
/// `[test]` in kern.toml, then from the defaults.
#[derive(Args, Clone)]
pub struct TestOpts {
    /// Reference manifest A (assumed correct)
    #[arg(long)]
    pub reference: Option<PathBuf>,
    /// Candidate manifest B
    #[arg(long)]
    manifest: Option<PathBuf>,
    /// Directory of cubins for both manifests; steps resolve by their pinned
    /// sha256, so one dir holds every version (file names are labels)
    #[arg(long)]
    kernels: Option<PathBuf>,
    /// Checkpoint directories or .safetensors files, tensors bound by name
    /// across all of them
    #[arg(long)]
    weights: Vec<String>,
    /// HF tokenizer.json (only needed with --prompt)
    #[arg(long)]
    tokenizer: Option<PathBuf>,
    /// Real-text prompt for the tap's prefill instead of seeded random
    /// tokens (decode tokens stay seeded)
    #[arg(long)]
    prompt: Option<String>,
    /// Prefill length in tokens; 0 = drawn from the seed: half the time
    /// uniform in [1, capacity − steps], half the time a structural
    /// boundary (a page, a chunk, the `tokens` max, ±1)
    #[arg(long, default_value_t = 0)]
    prefill: u64,
    /// Decode steps; the seed picks a length in [steps/2, steps] (default 32)
    #[arg(long)]
    decode_steps: Option<u64>,
    /// How far the end-to-end distribution may move, KL(A‖B) in nats on
    /// any logits row, and still PASS (with logit evidence); an argmax
    /// flip within it is a tie, one beyond it a FAIL (default 0.01)
    #[arg(long)]
    logit_kl: Option<f64>,
    /// CUDA device ordinals, one per rank of the manifest's topology (a
    /// single ordinal starts the ranks there, consecutively); default 0
    #[arg(long, value_delimiter = ',')]
    gpu: Vec<usize>,
    /// State capacity in tokens; rounded down to the manifest's page unit
    /// (default 4096)
    #[arg(long)]
    capacity: Option<u64>,
    /// Prefill chunk; 0 = drawn from the seed among the `tokens` max, 512,
    /// a page and a random size
    #[arg(long, default_value_t = 0)]
    chunk: u64,
    /// Replays for span timing (minimum is reported)
    #[arg(long, default_value_t = 20)]
    iters: usize,
    /// Skip capturing both decode programs as CUDA graphs for the step time
    #[arg(long)]
    no_graph_step: bool,
    /// Skip sweeping prefill over the `tokens` var range
    #[arg(long)]
    no_sweep: bool,
    /// Device peak memory bandwidth in GB/s, for the roofline column
    #[arg(long, default_value_t = 8000.0)]
    peak_bw: f64,
    /// Write the test report as JSON here (a directory when several
    /// targets run: one file per target)
    #[arg(long)]
    out: Option<PathBuf>,
    /// Only print the structural diff
    #[arg(long)]
    diff_only: bool,
    /// Skip the timing section
    #[arg(long)]
    no_perf: bool,
    /// Skip the noise-floor re-runs
    #[arg(long)]
    no_noise: bool,
    /// Seed for the workload (default 0x5eed)
    #[arg(long)]
    seed: Option<u64>,
    /// Print the report as one JSON object instead of text lines
    #[arg(long)]
    json: bool,
}

/// Resolved options: the shared inputs, plus what only `kern test` has
/// an opinion about.
struct Opts {
    inputs: Inputs,
    /// The reference manifest A; B is `inputs.manifest`.
    a: PathBuf,
    prompt: Option<String>,
    gpus: Vec<usize>,
    capacity: u64,
    out: Option<PathBuf>,
    diff_only: bool,
    json: bool,
    harness: Options,
}

impl TestOpts {
    fn resolve(self, cfg: Option<&Config>, t: Option<&Target>) -> Result<Opts> {
        let given = Given {
            manifest: self.manifest,
            reference: self.reference,
            kernels: self.kernels,
            weights: self.weights,
            tokenizer: self.tokenizer,
            ..Default::default()
        };
        let inputs = Inputs::resolve(given, cfg, t)?;
        let a = inputs
            .reference
            .clone()
            .ok_or_else(|| crate::inputs::need(cfg, "reference").context("kern test is A/B"))?;
        let test = cfg.map(|c| &c.test);
        let harness = Options {
            a: a.display().to_string(),
            b: inputs.manifest.display().to_string(),
            prompt: None,
            prefill: self.prefill,
            decode_steps: self.decode_steps.or_else(|| test.and_then(|x| x.decode_steps)).unwrap_or(32),
            logit_kl: self.logit_kl.or_else(|| test.and_then(|x| x.logit_kl)).unwrap_or(0.01),
            chunk: self.chunk,
            iters: self.iters,
            graph_step: !self.no_graph_step,
            sweep: !self.no_sweep,
            peak_bw: self.peak_bw,
            perf: !self.no_perf,
            noise: !self.no_noise,
            seed: self.seed.or_else(|| test.and_then(|x| x.seed)).unwrap_or(0x5eed),
        };
        Ok(Opts {
            a,
            prompt: self.prompt.or_else(|| test.and_then(|x| x.prompt.clone())),
            gpus: match self.gpu.is_empty() {
                true => vec![inputs.gpu],
                false => self.gpu,
            },
            capacity: self.capacity.or_else(|| cfg.and_then(|c| c.capacity)).unwrap_or(4096),
            out: self.out,
            diff_only: self.diff_only,
            json: self.json,
            harness,
            inputs,
        })
    }
}

/// `kern test`: returns the exit code (0 PASS, 1 FAIL, 2 INCONCLUSIVE).
pub fn run(o: TestOpts, cfg: Option<&Config>, target: Option<&Target>) -> Result<i32> {
    execute(o.resolve(cfg, target)?)
}

/// The side: the manifest loaded once per rank, each on its GPU, every
/// rank fed the same sequence. Bytes move from the calling thread (every
/// runtime entry point rebinds its context); whatever runs fans out to
/// one thread per rank, so a collective inside the range finds its peers
/// issuing, and returns when the slowest rank has.
struct Ranks {
    ranks: Vec<Caller>,
}

impl Ranks {
    /// Every rank's weights, the rest of each rank dropped.
    fn into_resident(self) -> Vec<Resident> {
        self.ranks.into_iter().map(|c| c.into_runtime().into_resident()).collect()
    }

    fn kept_bytes(&self) -> u64 {
        self.ranks.iter().map(|c| c.rt.kept_bytes()).sum()
    }

    /// `f` on every rank at once; the results in rank order, the first
    /// error if any.
    fn each<T: Send>(&mut self, what: &str, f: impl Fn(&mut Caller) -> Result<T> + Sync) -> Result<Vec<T>> {
        crate::each(&mut self.ranks, what, f)
    }

    fn rt(&self, rank: usize) -> &Runtime {
        &self.ranks[rank].rt
    }
    fn rt_mut(&mut self, rank: usize) -> &mut Runtime {
        &mut self.ranks[rank].rt
    }
}

impl Side for Ranks {
    type Buf = Scratch;

    fn manifest(&self) -> &Verified {
        &self.rt(0).manifest
    }
    fn provision(&self) -> Provision {
        self.rt(0).provision()
    }
    fn page(&self) -> u64 {
        self.rt(0).page()
    }
    fn vocab(&self) -> u64 {
        self.ranks[0].vocab()
    }
    fn ranks(&self) -> usize {
        self.ranks.len()
    }
    fn stage(&mut self, ids: &[i64]) -> Result<Vars> {
        let vars = self.ranks.iter_mut().map(|c| c.stage(ids)).collect::<Result<Vec<_>>>()?;
        Ok(vars.into_iter().next().expect("a side has a rank"))
    }
    fn reset(&mut self) {
        self.ranks.iter_mut().for_each(Caller::reset)
    }
    fn advance(&mut self, n: u64) {
        self.ranks.iter_mut().for_each(|c| c.advance(n))
    }
    fn calls(&self, program: &str) -> Result<usize> {
        Ok(self.rt(0).call_count(program)?)
    }
    fn run(&mut self, program: &str, vars: &Vars, calls: Range<usize>) -> Result<()> {
        let what = format!("`{program}` calls {}..{}", calls.start, calls.end);
        self.each(&what, |c| Ok(c.rt.run_range(program, vars, calls.start, calls.end)?)).map(drop)
    }
    fn read(&self, rank: usize, buffer: &str, bytes: usize) -> Result<Vec<u8>> {
        self.rt(rank).read_buffer_prefix(buffer, bytes).with_context(|| format!("rank {rank}: reading `{buffer}`"))
    }
    fn save(&self, rank: usize, buffer: &str, at: Range<usize>) -> Result<Scratch> {
        self.rt(rank).save_buffer(buffer, at.clone()).with_context(|| format!("rank {rank}: saving `{buffer}` {at:?}"))
    }
    fn load(&mut self, rank: usize, buffer: &str, at: Range<usize>, from: &Scratch) -> Result<()> {
        self.rt_mut(rank)
            .load_buffer(buffer, at.clone(), from)
            .with_context(|| format!("rank {rank}: loading `{buffer}` {at:?}"))
    }
    fn bytes(&self, rank: usize, from: &Scratch, at: Range<usize>) -> Result<Vec<u8>> {
        self.rt(rank).read_scratch(from, at.clone()).with_context(|| format!("rank {rank}: reading scratch at {at:?}"))
    }
    fn compare(&self, rank: usize, dtype: DType, a: At<Scratch>, b: At<Scratch>) -> Result<Cmp> {
        let what = name_of(&a);
        let c = self
            .rt(rank)
            .compare(dtype, on_device(a), on_device(b))
            .with_context(|| format!("rank {rank}: comparing {what}"))?;
        Ok(cmp_of(c))
    }
    fn changed(&self, rank: usize, a: At<Scratch>, b: At<Scratch>) -> Result<Vec<Range<usize>>> {
        let what = name_of(&b);
        self.rt(rank)
            .changed(on_device(a), on_device(b))
            .with_context(|| format!("rank {rank}: changed blocks of {what}"))
    }
    fn logits(
        &self,
        rank: usize,
        dtype: DType,
        cols: usize,
        a: At<Scratch>,
        b: At<Scratch>,
    ) -> Result<Vec<LogitStats>> {
        let rows = self
            .rt(rank)
            .logits(dtype, cols, TOP, on_device(a), on_device(b))
            .with_context(|| format!("rank {rank}: logits rows of {cols}"))?;
        Ok(rows.into_iter().map(logit_of).collect())
    }
    fn state_bytes(&self, state: &str) -> Result<usize> {
        Ok(self.rt(0).state_bytes(state)?)
    }
    fn read_state(&self, rank: usize, state: &str, at: Range<usize>) -> Result<Vec<u8>> {
        self.rt(rank)
            .read_state_at(state, at.start, at.len())
            .with_context(|| format!("rank {rank}: reading state `{state}` at {at:?}"))
    }
    fn write_state(&mut self, rank: usize, state: &str, at: usize, bytes: &[u8]) -> Result<()> {
        self.rt_mut(rank)
            .write_state_at(state, at, bytes)
            .with_context(|| format!("rank {rank}: writing state `{state}` at {at}"))
    }
    fn save_state(&self, rank: usize, state: &str) -> Result<Scratch> {
        self.rt(rank).save_state(state).with_context(|| format!("rank {rank}: saving state `{state}`"))
    }
    fn load_state(&mut self, rank: usize, state: &str, from: &Scratch) -> Result<()> {
        self.rt_mut(rank).load_state(state, from).with_context(|| format!("rank {rank}: loading state `{state}`"))
    }
    fn zero_states(&mut self) -> Result<()> {
        self.ranks
            .iter_mut()
            .enumerate()
            .try_for_each(|(q, c)| c.rt.zero_states().with_context(|| format!("rank {q}: zeroing states")))
    }
    fn time(&mut self, program: &str, vars: &Vars, calls: Range<usize>, iters: usize) -> Result<Vec<f32>> {
        let what = format!("timing `{program}` calls {}..{}", calls.start, calls.end);
        let per_rank = self.each(&what, |c| Ok(c.rt.time_range(program, vars, calls.start, calls.end, iters)?))?;
        // the slowest rank per call: what a step waits for
        Ok((0..calls.len()).map(|i| per_rank.iter().map(|t| t[i]).fold(0.0, f32::max)).collect())
    }
    fn time_graph(&mut self, program: &str, vars: &Vars, iters: usize) -> Result<f32> {
        let t = self.each(&format!("timing `{program}` as a graph"), |c| {
            c.rt.capture(program, vars)?;
            Ok(c.rt.time_captured(program, vars, iters)?)
        })?;
        Ok(t.into_iter().fold(0.0, f32::max))
    }
}

fn on_device(at: At<'_, Scratch>) -> kern_runtime::At<'_> {
    match at {
        At::Buffer(n, r) => kern_runtime::At::Buffer(n, r),
        At::State(n, r) => kern_runtime::At::State(n, r),
        At::Scratch(s, r) => kern_runtime::At::Scratch(s, r),
    }
}

fn name_of(at: &At<Scratch>) -> String {
    match at {
        At::Buffer(n, _) => format!("`{n}`"),
        At::State(n, _) => format!("state `{n}`"),
        At::Scratch(..) => "scratch".into(),
    }
}

/// The device counts as kern-test's [`Cmp`]: the one place the two
/// crates' definitions meet, so the GPU oracle test goes through it too.
pub fn cmp_of(c: kern_runtime::Cmp) -> Cmp {
    Cmp::from_counts(
        c.n as usize,
        c.n_diff as usize,
        c.signed_zero as usize,
        c.nan_one_side as usize,
        c.measured as usize,
        c.max_ulp,
        c.max_abs,
    )
}

/// A device logits row as kern-test's [`LogitStats`].
pub fn logit_of(l: kern_runtime::Logit) -> LogitStats {
    LogitStats::from_parts(
        cmp_of(l.cmp),
        l.argmax_a as usize,
        l.argmax_b as usize,
        l.top1,
        l.top2,
        l.kl,
        l.top as usize,
        l.rank_in_b as usize,
    )
}

/// The GPU of each rank: the list as given, or `n` from the one given.
fn gpus_of(given: &[usize], n: usize) -> Result<Vec<usize>> {
    match given {
        [g] => Ok((*g..*g + n).collect()),
        gs if gs.len() == n => Ok(gs.to_vec()),
        gs => bail!("the manifest runs as {n} ranks; --gpu names {} devices", gs.len()),
    }
}

/// Load the manifest on every rank's GPU, bind each rank's weights (over
/// what the last side left resident on it, if anything), connect the
/// peers, run what the manifest runs once.
fn load_side(m: &Verified, o: &Opts, host_weights: &HostWeights, resident: Option<Vec<Resident>>) -> Result<Ranks> {
    let n = crate::ranks_of(m)?;
    let gpus = gpus_of(&o.gpus, n)?;
    let resident: Vec<Option<Resident>> = match resident {
        Some(r) if r.len() == n => r.into_iter().map(Some).collect(),
        Some(r) => bail!("{} ranks left their weights resident; this side runs {n}", r.len()),
        None => (0..n).map(|_| None).collect(),
    };
    // The test drives one sequence (`Caller` leases one and writes its
    // lines into every table column), so one slot is all the per-sequence
    // states need. Sizing them for the manifest's whole batch made every
    // whole-state read of a span copy 128 slots: on qwen3.8-27b (154 MB of
    // GDN state per slot) a run grew past 700 GB of host memory.
    let capacity = Capacity { tokens: Some(o.capacity), seqs: 1 };
    let loaded: Vec<Result<crate::Sent<Runtime>>> = std::thread::scope(|s| {
        let handles: Vec<_> = gpus
            .iter()
            .zip(resident)
            .enumerate()
            .map(|(q, (&gpu, resident))| {
                let topo = crate::topology_of(m, q);
                s.spawn(move || -> Result<crate::Sent<Runtime>> {
                    let topology = m.topology.is_some().then_some(&topo);
                    let mut rt = match resident {
                        Some(r) => {
                            Runtime::load_over(m, &o.inputs.kernels, gpu, Some(capacity), topology, host_weights, r)
                        }
                        None => Runtime::load_with_host_weights(
                            m,
                            &o.inputs.kernels,
                            gpu,
                            Some(capacity),
                            topology,
                            host_weights,
                        ),
                    }
                    .with_context(|| format!("rank {q} on gpu {gpu}"))?;
                    o.inputs.weights.bind(&mut rt, &topo).with_context(|| format!("rank {q}: binding weights"))?;
                    Ok(crate::Sent(rt))
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("a rank's load thread panicked")).collect()
    });
    let mut rts: Vec<Runtime> = loaded.into_iter().map(|r| r.map(|s| s.0)).collect::<Result<_>>()?;
    crate::connect_peers(m, &mut rts)?;
    Ok(Ranks { ranks: rts.into_iter().map(Caller::new).collect::<Result<_>>()? })
}

fn tokens_of(tok: &tokenizers::Tokenizer, s: &str) -> Result<Vec<i64>> {
    let ids: Vec<i64> =
        tok.encode(s, false).map_err(|e| anyhow::anyhow!("encode: {e}"))?.get_ids().iter().map(|&u| u as i64).collect();
    if ids.is_empty() {
        bail!("prompt is empty: {s:?}");
    }
    Ok(ids)
}

/// Text streams a section's lines as it finishes; JSON waits for the end.
struct Out {
    json: bool,
}

impl Out {
    fn show(&self, lines: &[String]) {
        if !self.json {
            for l in lines {
                println!("{l}");
            }
        }
    }
}

/// Archive, verdict line, JSON: the end of every path through `execute`.
fn finish(o: &Opts, report: Report) -> Result<i32> {
    let out = Out { json: o.json };
    if let Some(p) = &o.out {
        std::fs::write(p, serde_json::to_string_pretty(&json!({"summary": report.summary, "detail": report.detail}))?)?;
        out.show(&[row("out", p.display().to_string(), None)]);
    }
    // A structural diff alone has no verdict; identical programs do (a
    // no-op swap passes).
    let sum = &report.summary;
    if sum.tap.is_some() || sum.diff.programs.is_empty() {
        out.show(&sum.verdict.lines());
    }
    if o.json {
        println!("{}", serde_json::to_string_pretty(sum)?);
    }
    Ok(report.code())
}

fn execute(mut o: Opts) -> Result<i32> {
    let t_start = Instant::now();
    let ja = std::fs::read_to_string(&o.a).with_context(|| format!("reading {}", o.a.display()))?;
    let jb = std::fs::read_to_string(&o.inputs.manifest)
        .with_context(|| format!("reading {}", o.inputs.manifest.display()))?;
    let ma = Verified::from_json(&ja).with_context(|| format!("A ({}) failed verification", o.a.display()))?;
    let mb = Verified::from_json(&jb).with_context(|| format!("B ({}) failed verification", o.harness.b))?;
    Protocol::check(&ma).with_context(|| format!("A ({}) does not fit the serving protocol", o.a.display()))?;
    Protocol::check(&mb).with_context(|| format!("B ({}) does not fit the serving protocol", o.harness.b))?;
    let out = Out { json: o.json };
    let (a, b) = (o.harness.a.clone(), o.harness.b.clone());
    out.show(&[row("kern test", format!("A {a} → B {b}"), None)]);

    // ---- 1. static diff
    let diff = kern_test::diff::diff(&ma, &mb);
    out.show(&diff.lines());
    let verdict = |code: i32, summary: &str| Verdict::new(code, summary.into(), t_start.elapsed().as_secs_f32());
    if o.diff_only {
        return finish(&o, Report::of_diff(&a, &b, diff, verdict(0, "structural diff only, nothing run")));
    }
    if diff.spans.is_empty() {
        return finish(&o, Report::of_diff(&a, &b, diff, verdict(0, "nothing to test: the programs are identical")));
    }
    o.harness.prompt = match &o.prompt {
        Some(text) => {
            let Some(tk) = &o.inputs.tokenizer else {
                bail!("--prompt needs a tokenizer (--tokenizer or the target's `tokenizer`)")
            };
            let tokenizer = tokenizers::Tokenizer::from_file(tk).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
            Some(tokens_of(&tokenizer, text)?)
        }
        None => None,
    };

    // ---- 2. A: load, record, keep its weights; B: load over them, replay.
    // Never both loaded: A's states, workspace and programs are gone
    // before B allocates its own.
    let host_weights = HostWeights::new();
    let load = |m: &Verified, side: &str, resident: Option<Vec<Resident>>| -> Result<(Ranks, f32)> {
        let t = Instant::now();
        let r = load_side(m, &o, &host_weights, resident).with_context(|| format!("loading {side}"))?;
        let s = t.elapsed().as_secs_f32();
        let gpus = gpus_of(&o.gpus, r.ranks())?;
        let kept = match r.kept_bytes() {
            0 => String::new(),
            b => format!(" · {:.1} GiB of weights kept from A", b as f64 / (1u64 << 30) as f64),
        };
        out.show(&[row(
            "load",
            format!(
                "{side}: {} on gpu {}{kept}",
                plural(r.ranks(), "rank"),
                gpus.iter().map(|g| g.to_string()).collect::<Vec<_>>().join(",")
            ),
            Some(s),
        )]);
        Ok((r, s))
    };
    let (mut side_a, load_a) = load(&ma, "A", None)?;
    let mut rec = kern_test::record(&o.harness, diff, &mb, &mut side_a, &mut |lines: &[String]| out.show(lines))?;
    let (mut side_b, load_b) = load(&mb, "B", Some(side_a.into_resident()))?;
    rec.load_s = load_a + load_b;
    let report = kern_test::replay(&o.harness, rec, &mut side_b, &mut |lines: &[String]| out.show(lines))?;
    finish(&o, report)
}
