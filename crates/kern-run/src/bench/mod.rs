//! Single-device performance evidence, driven by the manifest's protocol.
//!
//! Two questions, two costs. Where does a program's time go: every call is
//! bracketed in a second copy of the graph and attributed, which is cheap
//! and always done. What would a call cost on its own, cold and warm:
//! every distinct call is lifted out of the program and measured against
//! its own pre-image, which is most of the wall clock and only done for
//! `--isolate`. The first answers what to work on, the second what the
//! work would be worth.
//!
//! Nothing here names a program, a buffer or a model. A workload gives
//! shapes ([`workload`]), `Protocol` turns a shape into a program, and the
//! fill table says what to stage. Samples are kept verbatim, slow tails
//! included; no minimum-only or outlier filtering.
//!
//! A manifest with a topology runs as its ranks, every rank fed the same
//! shape with its own sequences, in lockstep: a call that reads its peers
//! finds them issuing. Attribution is per rank; a call cannot be lifted
//! out of one rank's program alone, so `--isolate` is single-device.

pub mod workload;

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{ensure, Context, Result};
use kern_manifest::types::{Arg, BufferKind, Dim, Manifest};
use kern_manifest::{Protocol, Verified};
use kern_pool::Lease;
use kern_runtime::profile::{Anchor, Probe, ProgramSamples};
use kern_runtime::{Capacity, HostWeights, Runtime};
use kern_test::report::row;
use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::config::{Config, Target};
use crate::{le_bytes_i32, Vars};
use workload::{Plan, Scenario, Workload};

/// The report schema. Readers check it before they read anything else.
const VERSION: u32 = 2;

/// Ops named on the mix line before it says how many are left.
const SHOW: usize = 6;

#[derive(clap::Args, Debug)]
pub struct BenchOpts {
    /// Sweep of call shapes, sample count and seed (TOML); no ABI names
    #[arg(long)]
    workload: PathBuf,
    /// Portable JSON with raw samples and all call locations; no machine paths
    #[arg(long)]
    out: PathBuf,
    /// Also measure every distinct call on its own, L2 cold and warm
    #[arg(long)]
    isolate: bool,
    /// Manifest JSON (must pass verification)
    #[arg(long)]
    manifest: Option<PathBuf>,
    /// Directory of cubins; steps resolve by their pinned sha256, so one dir
    /// holds every version (file names are labels)
    #[arg(long)]
    kernels: Option<PathBuf>,
    /// Checkpoint directories or .safetensors files, tensors bound by name
    /// across all of them
    #[arg(long)]
    weights: Vec<String>,
    /// HF tokenizer.json; the prose every context is built from goes through it
    #[arg(long)]
    tokenizer: Option<PathBuf>,
    /// CUDA device ordinal; a manifest with a topology takes this one and
    /// the next ranks-1 after it
    #[arg(long)]
    gpu: Option<usize>,
}

#[derive(Serialize, Debug)]
struct Stats {
    n: usize,
    min: f64,
    p10: f64,
    p50: f64,
    p90: f64,
    max: f64,
    mean: f64,
    cv: f64,
    tail_ratio: f64,
    block_medians: Vec<f64>,
}

fn stats(samples: &[f64]) -> Stats {
    assert!(!samples.is_empty() && samples.iter().all(|x| x.is_finite() && *x >= 0.));
    let mut v = samples.to_vec();
    v.sort_by(f64::total_cmp);
    let q = |p: f64| {
        let at = p * (v.len() - 1) as f64;
        let lo = at.floor() as usize;
        let hi = at.ceil() as usize;
        v[lo] + (v[hi] - v[lo]) * (at - lo as f64)
    };
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    let variance = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / v.len() as f64;
    let block_medians = samples
        .chunks(samples.len().div_ceil(4))
        .map(|b| {
            let mut b = b.to_vec();
            b.sort_by(f64::total_cmp);
            b[b.len() / 2]
        })
        .collect();
    Stats {
        n: v.len(),
        min: v[0],
        p10: q(0.1),
        p50: q(0.5),
        p90: q(0.9),
        max: *v.last().unwrap(),
        mean,
        cv: if mean > 0. { variance.sqrt() / mean } else { 0. },
        tail_ratio: if q(0.5) > 0. { q(0.9) / q(0.5) } else { 1. },
        block_medians,
    }
}

/// Share of a program's attributed time by op, largest first. The shares
/// are of the traced total, not of the graph: bracketing every call costs
/// time that belongs to no call, and dividing by the graph would quietly
/// hand that time to whichever op happened to be measured.
fn mix(calls: &[(String, f64)]) -> Vec<(String, f64)> {
    let mut by: BTreeMap<&str, f64> = BTreeMap::new();
    for (op, us) in calls {
        *by.entry(op).or_default() += us;
    }
    let total: f64 = by.values().sum();
    let mut v: Vec<(String, f64)> =
        by.into_iter().map(|(op, us)| (op.to_string(), if total > 0. { us / total } else { 0. })).collect();
    v.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    v
}

fn series(v: Vec<f64>) -> Value {
    json!({"stats":stats(&v),"samples_us":v})
}

fn digest(b: &[u8]) -> String {
    hex::encode(Sha256::digest(b))
}

fn anchors(v: Vec<Anchor>) -> Value {
    Value::Array(
        v.into_iter()
            .map(|a| {
                json!({"name":a.name,"payload_bytes":a.bytes,
        "traffic_bytes":a.traffic_bytes,"flops":a.flops,"timing":series(a.samples_us)})
            })
            .collect(),
    )
}

fn telemetry(gpu: usize) -> Value {
    let fields = "clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used";
    let output = std::process::Command::new("nvidia-smi")
        .args(["-i", &gpu.to_string(), &format!("--query-gpu={fields}"), "--format=csv,noheader,nounits"])
        .output();
    match output {
        Ok(o) if o.status.success() => {
            let text = String::from_utf8_lossy(&o.stdout);
            let values = text.trim().split(',').map(|v| v.trim().parse::<f64>().ok()).collect::<Vec<_>>();
            json!({"sm_mhz":values.first(),"memory_mhz":values.get(1),"temperature_c":values.get(2),
                "power_w":values.get(3),"power_limit_w":values.get(4),"gpu_util_pct":values.get(5),"memory_mib":values.get(6)})
        }
        _ => Value::Null,
    }
}

/// Conservative grouping: retain state offsets, scalar args, ABI and all
/// buffer shapes/offsets and alias relationships. Weight names can vary.
fn signature(m: &Manifest, program: &str, index: usize, vars: &Vars) -> Value {
    let c = &m.programs[program].calls[index];
    let mut aliases: BTreeMap<&str, usize> = BTreeMap::new();
    let args: Vec<Value> = c
        .args
        .iter()
        .map(|a| match a {
            Arg::Buf { buf, offset } => {
                let next = aliases.len();
                let alias = *aliases.entry(buf).or_insert(next);
                let b = &m.buffers[buf];
                let shape: Vec<u64> = b
                    .shape
                    .iter()
                    .map(|d| match d {
                        Dim::Const(n) => *n,
                        Dim::Var(v) => vars[v],
                    })
                    .collect();
                json!({"alias":alias,"dtype":b.dtype,"shape":shape,"kind":b.kind,"offset":offset})
            }
            Arg::State { state, offset } => json!({"state":state,"offset":offset}),
            Arg::Var { var } => json!({"scalar":vars[var]}),
            Arg::Expr { expr } => json!({"scalar":expr.eval(vars).expect("verified expression")}),
            _ => serde_json::to_value(a).unwrap(),
        })
        .collect();
    // Vars remains in the key because impl-private geometry may read vars
    // not forwarded by the call's public interface.
    json!({"op":c.op,"args":args,"vars":vars})
}

/// One call staged for `leases`, tables included: a scenario's batch
/// changes the leases from call to call, so every table is rewritten
/// with the rows it is about to be read with.
fn stage(
    rt: &mut Runtime,
    p: &Protocol,
    leases: &[Lease],
    positions: &[usize],
    rows: usize,
    ids: &[i64],
) -> Result<Vars> {
    let vars = crate::stage(rt, p, leases, positions, rows, ids)?;
    for t in &p.page_tables {
        rt.write_input_at(&t.name, &le_bytes_i32(&crate::page_rows(t, leases, leases.len())?), &vars)?;
    }
    // Line tables use their declared maximum column stride, even when
    // only a prefix of columns is active.
    for t in &p.line_tables {
        rt.write_input_at(&t.name, &le_bytes_i32(&crate::line_rows(t, leases, p.groups.max as usize)?), &vars)?;
    }
    Ok(vars)
}

fn corpus(tokenizer: &std::path::Path, seed: u64) -> Result<Vec<i64>> {
    let tok = tokenizers::Tokenizer::from_file(tokenizer).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let topics = [
        "A botanist mapped the small plants growing between the stones. Some leaves stored water, while others curled away from the afternoon sunlight.",
        "The engineer compared three implementations using measured latency and memory traffic. A smaller matrix did not always finish sooner, because the scheduling algorithm changed.",
        "At the harbor, a wooden boat carried baskets of oranges and a box of weather instruments. The crew checked the wind before crossing the bay.",
        "The library catalog described an expedition through the mountains. Its notebooks contained sketches, temperature readings, and careful accounts of conversations in each village.",
        "A musician rehearsed the passage slowly, listening for uneven intervals. Later the ensemble adjusted its timing until the melody became clear across the room.",
        "An astronomer explained why a distant planet was hard to observe. Its atmosphere, orbital period, and reflected light each offered a different piece of evidence.",
    ];
    let text = (0..320)
        .map(|i| format!("Observation {}: {}\n", i as u64 + seed, topics[(i * 5 + seed as usize) % topics.len()]))
        .collect::<String>();
    Ok(tok
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?
        .get_ids()
        .iter()
        .map(|x| *x as i64)
        .collect())
}

/// `rows` tokens of prose for sequence `seq` from position `pos`. Every
/// sequence reads the corpus from its own offset, so the rows of a batch
/// are different text and not one prompt copied across the batch.
fn tokens(corpus: &[i64], seq: usize, pos: usize, rows: usize) -> Vec<i64> {
    (0..rows).map(|i| corpus[(seq * 173 + pos + i) % corpus.len()]).collect()
}

fn output_fingerprints(rt: &Runtime, p: &Protocol, groups: usize) -> Result<Vec<Value>> {
    p.fills
        .iter()
        .filter(|f| rt.manifest.buffers[&f.name].kind == BufferKind::Output)
        .map(|f| {
            let bytes = rt.read_output(&f.name)?;
            let values: Vec<i64> = f.decode(&bytes).into_iter().take(groups * f.width as usize).collect();
            Ok(json!({"name":f.name,"values":values}))
        })
        .collect()
}

/// Build every sequence's real prefix by running the chunk program over
/// real prose. Each sequence gets its own lease and its own tokens, from
/// sequence `first` on: never duplicate one lease across the batch to
/// save preparation time.
fn prefixes(
    rt: &mut Runtime,
    p: &Protocol,
    leases: &[Lease],
    context: &[usize],
    corpus: &[i64],
    first: usize,
) -> Result<()> {
    let chunk = p.chunk().context("a prefix needs a variable-row program")?.name.clone();
    for (i, &length) in context.iter().enumerate() {
        let mut pos = 0;
        while pos < length {
            let q = (length - pos).min(p.rows.max as usize);
            let vars = stage(rt, p, &leases[i..i + 1], &[pos], q, &tokens(corpus, first + i, pos, q))?;
            if !rt.is_captured(&chunk, &vars) {
                rt.capture(&chunk, &vars)?;
            }
            rt.run_captured(&chunk, &vars)?;
            pos += q;
        }
    }
    Ok(())
}

/// Lift each distinct call out of the program, measure it cold and warm,
/// and step the trajectory so the next call sees the inputs it would see.
/// The caller has already re-staged: `Probe::program` leaves persistent
/// state at its pre-image, not where the walk needs it. Returns the cases
/// and, for every call, the case it belongs to.
fn isolate_calls(
    rt: &Runtime,
    probe: &Probe,
    program: &str,
    vars: &Vars,
    samples: usize,
) -> Result<(Vec<Value>, Vec<usize>)> {
    let m = &rt.manifest;
    let mut seen: BTreeMap<String, usize> = BTreeMap::new();
    let mut cases = Vec::new();
    let mut of_call = Vec::new();
    for (i, c) in m.programs[program].calls.iter().enumerate() {
        let sig = signature(m, program, i, vars);
        let key = digest(&serde_json::to_vec(&sig)?);
        let case = match seen.get(&key) {
            Some(&case) => case,
            None => {
                let s = probe.call(rt, program, vars, i, samples).with_context(|| format!("call {i} ({})", c.op))?;
                seen.insert(key.clone(), cases.len());
                cases.push(json!({"id":key,"signature":sig,"representative_call":i,
                    "cold":series(s.cold_us),"warm":series(s.warm_us)}));
                cases.len() - 1
            }
        };
        of_call.push(case);
        rt.run_range(program, vars, i, i + 1)?;
    }
    Ok((cases, of_call))
}

/// One line, one fact, in the shape `kern test` reports in.
fn say(key: &str, body: String) {
    println!("{}", row(key, body, None));
}

fn timed(key: &str, body: String, since: Instant) {
    println!("{}", row(key, body, Some(since.elapsed().as_secs_f32())));
}

/// The id carries the whole shape. What it cannot carry is how much prefix
/// has to run before any measurement starts, which is what a long scenario
/// spends its time on.
fn scenario_line(s: &Scenario, chunk_rows: usize) -> String {
    match s.shape.context.iter().map(|n| n.div_ceil(chunk_rows)).sum::<usize>() {
        0 => s.id.clone(),
        1 => format!("{} · prefix 1 chunk", s.id),
        n => format!("{} · prefix {n} chunks", s.id),
    }
}

fn program_line(g: &Stats, calls: usize) -> String {
    format!(
        "{:.3} ms · p10–p90 {:.3}–{:.3} · cv {:.1}% · {calls} calls",
        g.p50 / 1e3,
        g.p10 / 1e3,
        g.p90 / 1e3,
        g.cv * 100.
    )
}

/// The largest shares by name, then how many ops are left unnamed.
fn mix_line(shares: &[(String, f64)]) -> String {
    let named = shares.iter().take(SHOW).map(|(op, f)| format!("{op} {:.1}%", f * 100.)).collect::<Vec<_>>();
    match shares.len().saturating_sub(SHOW) {
        0 => named.join(" · "),
        rest => format!("{} · +{rest} more", named.join(" · ")),
    }
}

/// Everything about the report that is settled before the first sample:
/// what was measured, on what, and under which protocol.
fn preamble(
    m: &Verified,
    json_bytes: &[u8],
    w: &Workload,
    plan: &Plan,
    probe: &Probe,
    isolated: bool,
    plan_ranks: usize,
) -> Result<Value> {
    Ok(json!({"version":VERSION,"model":m.model,"manifest_sha256":digest(json_bytes),
        "runner_sha256":digest(&std::fs::read(std::env::current_exe()?)?),
        "created_unix":SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        "isolated":isolated,"ranks":plan_ranks,
        "hardware":{"device":probe.device,"sm_count":probe.sm_count,"l2_bytes":probe.l2_bytes,
            "eviction_bytes":probe.eviction_bytes,"driver_version":probe.driver_version},
        "protocol":{"timer":"CUDA graph events","op_cache_modes":["cold_l2","warm_replay"],
            "program_entry_cache":"cold_l2; natural reuse between calls","restore":"declared writes before each sample",
            "samples_per_mode":w.samples,"tail_policy":"all measured samples retained; no outlier filtering",
            "warmup":"3 call executions; 4 whole-program executions",
            "grouping":"same op, resolved arguments, buffer shape, aliasing, offsets and vars; weight names omitted",
            "context_data":"deterministic diverse prose, actual model prefix execution",
            "units":"microseconds; bandwidth GB/s uses read+write traffic"},
        "modules":m.modules.iter().map(|(n,x)|json!({"name":n,"sha256":x.sha256})).collect::<Vec<_>>(),
        "workload":{"samples":w.samples,"seed":w.seed,"scenarios":&plan.scenarios,"dropped":&plan.dropped},
        "scenarios":[]}))
}

/// One rank's evidence of one scenario before it is written down.
/// `cases` and `of_call` are empty unless the calls were isolated, and
/// then `outputs` are the walk's, checked equal to the whole program's.
struct Measured {
    vars: Vars,
    program: ProgramSamples,
    cases: Vec<Value>,
    of_call: Vec<usize>,
    outputs: Vec<Value>,
    telemetry: (Value, Value),
    isolate_s: Option<f64>,
}

/// The report's record of one scenario on one rank: the manifest's calls
/// annotated with what was measured of each.
fn record(m: &Manifest, s: &Scenario, rank: usize, x: Measured, elapsed_s: f64) -> Value {
    let calls: Vec<Value> = m.programs[&s.program]
        .calls
        .iter()
        .zip(x.program.attributed_us)
        .enumerate()
        .map(|(i, (c, us))| {
            let mut v = json!({"index":i,"label":c.label,"op":c.op,"args":c.args,
                "launches":m.ops[&c.op].imp.launches.iter().map(|l|json!({"entry":l.entry(),"module":l.module()})).collect::<Vec<_>>(),
                "in_program":series(us)});
            if let Some(&case) = x.of_call.get(i) {
                v["case"] = json!(case);
            }
            v
        })
        .collect();
    let check = if x.of_call.is_empty() { Value::Null } else { json!("matches whole-program token outputs") };
    json!({"scenario":s,"program":s.program,"rank":rank,"vars":x.vars,
        "graph":series(x.program.graph_us),"instrumented":series(x.program.instrumented_us),
        "cases":x.cases,"calls":calls,"outputs":x.outputs,"output_check":check,
        "telemetry_before":x.telemetry.0,"telemetry_after":x.telemetry.1,"elapsed_s":elapsed_s})
}

/// One rank of the manifest: its runtime, the probe over it, the GPU it
/// is on and its index, which picks the prose its sequences read.
struct Rank {
    rt: Runtime,
    probe: Probe,
    gpu: usize,
    rank: usize,
}

/// One rank's part of a scenario: lease, prefix, stage, measure. Every
/// rank runs this at once for the same shape, so the graphs replay in
/// lockstep and a peer read finds its peer issuing.
fn measure(
    r: &mut Rank,
    p: &Protocol,
    s: &Scenario,
    corpus: &[i64],
    samples: usize,
    isolate: bool,
) -> Result<Measured> {
    let shape = &s.shape;
    let before = telemetry(r.gpu);
    let leases = shape.context.iter().map(|n| r.rt.lease(n + shape.rows)).collect::<kern_runtime::Result<Vec<_>>>()?;
    let seq = |i: usize| r.rank * shape.groups + i;
    prefixes(&mut r.rt, p, &leases, &shape.context, corpus, r.rank * shape.groups)?;
    let input: Vec<i64> =
        shape.context.iter().enumerate().flat_map(|(i, &pos)| tokens(corpus, seq(i), pos, shape.rows)).collect();
    let vars = stage(&mut r.rt, p, &leases, &shape.context, shape.rows, &input)?;
    let program = r.probe.program(&r.rt, &s.program, &vars, samples)?;
    let whole = output_fingerprints(&r.rt, p, shape.groups)?;
    let (cases, of_call, outputs, isolate_s) = match isolate {
        false => (Vec::new(), Vec::new(), whole, None),
        true => {
            let at = Instant::now();
            // `Probe::program` left the persistent state at its pre-image.
            stage(&mut r.rt, p, &leases, &shape.context, shape.rows, &input)?;
            let (cases, of_call) = isolate_calls(&r.rt, &r.probe, &s.program, &vars, samples)
                .with_context(|| format!("{}: isolating calls", s.id))?;
            let walked = output_fingerprints(&r.rt, p, shape.groups)?;
            ensure!(walked == whole, "{}: profiled trajectory differs from whole program output", s.id);
            (cases, of_call, walked, Some(at.elapsed().as_secs_f64()))
        }
    };
    Ok(Measured { vars, program, cases, of_call, outputs, telemetry: (before, telemetry(r.gpu)), isolate_s })
}

/// What every scenario is measured against: loaded once, and unchanged by
/// any of them. The manifest is the runtimes' own, so a scenario cannot be
/// measured against a different one than it runs on.
struct Bench {
    ranks: Vec<Rank>,
    protocol: Protocol,
    corpus: Vec<i64>,
    samples: usize,
    isolate: bool,
}

impl Bench {
    /// Every rank on its GPU, peers connected, what the manifest runs once
    /// run, a probe over each.
    fn load(m: &Verified, inputs: &crate::Inputs, protocol: Protocol, plan: &Plan, isolate: bool) -> Result<Bench> {
        let n = crate::ranks_of(m)?;
        ensure!(
            n == 1 || !isolate,
            "--isolate needs a single-device manifest: a call that reads its peers cannot be replayed on one rank"
        );
        let gpus: Vec<usize> = (inputs.gpu..inputs.gpu + n).collect();
        let host_weights = HostWeights::new();
        let capacity = Capacity { tokens: Some(plan.tokens), seqs: plan.seqs };
        let loaded: Vec<Result<crate::Sent<Runtime>>> = std::thread::scope(|s| {
            let handles: Vec<_> = gpus
                .iter()
                .enumerate()
                .map(|(q, &gpu)| {
                    let (topo, host_weights) = (crate::topology_of(m, q), &host_weights);
                    s.spawn(move || -> Result<crate::Sent<Runtime>> {
                        let topology = m.topology.is_some().then_some(&topo);
                        let mut rt = Runtime::load_with_host_weights(
                            m,
                            &inputs.kernels,
                            gpu,
                            Some(capacity),
                            topology,
                            host_weights,
                        )
                        .with_context(|| format!("rank {q} on gpu {gpu}"))?;
                        inputs.weights.bind(&mut rt, &topo).with_context(|| format!("rank {q}: binding weights"))?;
                        Ok(crate::Sent(rt))
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().expect("a rank's load thread panicked")).collect()
        });
        let mut rts: Vec<Runtime> = loaded.into_iter().map(|r| r.map(|s| s.0)).collect::<Result<_>>()?;
        crate::connect_peers(m, &mut rts)?;
        let probes = crate::each(&mut rts, "running what the manifest runs once", |rt| {
            crate::run_once(rt, &protocol)?;
            Ok(Probe::new(rt)?)
        })?;
        let ranks = rts
            .into_iter()
            .zip(probes)
            .zip(gpus)
            .enumerate()
            .map(|(rank, ((rt, probe), gpu))| Rank { rt, probe, gpu, rank })
            .collect();
        Ok(Bench { ranks, protocol, corpus: Vec::new(), samples: 0, isolate })
    }

    /// One scenario end to end on every rank: measure, say, record. The
    /// terminal shows the slowest rank, which is what a step waits for;
    /// the report keeps every rank.
    fn scenario(&mut self, s: &Scenario) -> Result<Vec<Value>> {
        let started = Instant::now();
        say("scenario", scenario_line(s, self.protocol.rows.max as usize));
        let (p, corpus, samples, isolate) = (&self.protocol, &self.corpus, self.samples, self.isolate);
        let measured = crate::each(&mut self.ranks, &s.id, |r| measure(r, p, s, corpus, samples, isolate))?;
        let p50 = |x: &Measured| stats(&x.program.graph_us).p50;
        let (q, slow) = measured.iter().enumerate().max_by(|a, b| p50(a.1).total_cmp(&p50(b.1))).unwrap();
        let m = &self.ranks[0].rt.manifest;
        let calls = &m.programs[&s.program].calls;
        let ranks = match measured.len() {
            1 => String::new(),
            n => format!(" · rank {q} slowest of {n}"),
        };
        timed("program", program_line(&stats(&slow.program.graph_us), calls.len()) + &ranks, started);
        let attributed: Vec<(String, f64)> =
            calls.iter().zip(&slow.program.attributed_us).map(|(c, us)| (c.op.clone(), stats(us).p50)).collect();
        say("mix", mix_line(&mix(&attributed)));
        if let Some(t) = slow.isolate_s {
            println!("{}", row("isolate", format!("{} cases · outputs match", slow.cases.len()), Some(t as f32)));
        }
        let elapsed_s = started.elapsed().as_secs_f64();
        Ok(measured.into_iter().enumerate().map(|(q, x)| record(m, s, q, x, elapsed_s)).collect())
    }
}

pub fn run(o: BenchOpts, cfg: Option<&Config>, target: Option<&Target>) -> Result<()> {
    let given = crate::Given {
        manifest: o.manifest.clone(),
        kernels: o.kernels.clone(),
        weights: o.weights.clone(),
        tokenizer: o.tokenizer.clone(),
        gpu: o.gpu,
        ..Default::default()
    };
    let inputs = crate::Inputs::resolve(given, cfg, target)?;
    let w = Workload::read(&o.workload)?;
    let json_bytes = std::fs::read(&inputs.manifest)?;
    let m = Verified::from_json(std::str::from_utf8(&json_bytes)?)?;
    let protocol = Protocol::check(&m)?;
    let plan = Plan::check(&w, &protocol, kern_pool::page_unit(&m) as usize)?;
    let mut bench = Bench::load(&m, &inputs, protocol, &plan, o.isolate)?;
    bench.corpus = corpus(inputs.tokenizer()?, w.seed)?;
    bench.samples = w.samples;

    let started = Instant::now();
    let (rt, probe) = (&bench.ranks[0].rt, &bench.ranks[0].probe);
    let before = anchors(probe.calibrate(rt, w.samples)?);
    let ranks = match bench.ranks.len() {
        1 => String::new(),
        n => format!(" · rank 0 of {n}"),
    };
    timed(
        "calibrate",
        format!("{} · {} SMs · L2 {} MiB{ranks}", probe.device, probe.sm_count, probe.l2_bytes >> 20),
        started,
    );
    say(
        "plan",
        format!(
            "{} scenarios · {} dropped · {} tokens of state · {} seqs",
            plan.scenarios.len(),
            plan.dropped.len(),
            plan.tokens,
            plan.seqs
        ),
    );
    for d in &plan.dropped {
        say("drop", format!("{} · {}", d.shape, d.why));
    }

    let mut report = preamble(&m, &json_bytes, &w, &plan, probe, o.isolate, bench.ranks.len())?;
    report["calibration_before"] = before;
    write(&o.out, &report)?;

    for s in &plan.scenarios {
        let records = bench.scenario(s)?;
        report["scenarios"].as_array_mut().unwrap().extend(records);
        write(&o.out, &report)?;
    }
    report["calibration_after"] = anchors(bench.ranks[0].probe.calibrate(&bench.ranks[0].rt, w.samples)?);
    write(&o.out, &report)?;
    timed("out", o.out.display().to_string(), started);
    Ok(())
}

fn write(out: &std::path::Path, report: &Value) -> Result<()> {
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    Ok(std::fs::write(out, serde_json::to_vec(report)?)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shares_are_of_the_traced_total_and_sorted() {
        let by = mix(&[("gemm".into(), 30.), ("attn".into(), 50.), ("gemm".into(), 20.)]);
        assert_eq!(by, [("attn".to_string(), 0.5), ("gemm".to_string(), 0.5)]);
        assert_eq!(mix(&[]), []);
    }

    #[test]
    fn the_mix_line_names_a_few_ops_and_counts_the_rest() {
        let shares: Vec<(String, f64)> = (0..SHOW + 2).map(|i| (format!("op{i}"), 0.1)).collect();
        assert!(mix_line(&shares).ends_with("op5 10.0% · +2 more"));
        assert_eq!(mix_line(&shares[..2]), "op0 10.0% · op1 10.0%");
        assert_eq!(mix_line(&[]), "");
    }

    #[test]
    fn tails_and_order_are_preserved() {
        let s = stats(&[1., 1., 1., 1., 1., 1., 1., 1., 10., 30., 50., 100.]);
        assert_eq!((s.n, s.max, s.p50), (12, 100., 1.));
        assert!(s.tail_ratio > 40. && s.cv > 1.);
        assert!(s.block_medians[3] > s.block_medians[0]);
    }
}
