//! `kern run`: end-to-end bs=1 greedy decode over a kern manifest.
//!
//! The prompt goes through the manifest's chunk program; every step after
//! that is one call of the program taking one sequence of `--rows` rows,
//! read back as the tokens it hands the sequence. A plain decode step and
//! a speculative round differ only in the rows the manifest declares for
//! them, so this loop does not know which one it is running.
//!
//! Logging goes to stderr via `tracing` (filter with `RUST_LOG`, default
//! `info`); stdout carries only the generated text.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{ensure, Context, Result};
use clap::Args;

use crate::config::{Config, Target};
use crate::{Caller, Given, Inputs};
use kern_manifest::protocol::Rows;
use kern_manifest::types::BufferKind;
use kern_manifest::Verified;
use kern_runtime::{Capacity, Runtime, Topology};
use tracing::info;

/// Flags of `kern run`; anything not given comes from the target in
/// kern.toml, then from the defaults.
#[derive(Args, Debug, Clone)]
pub struct RunOpts {
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

    /// HF tokenizer.json
    #[arg(long)]
    tokenizer: Option<PathBuf>,

    /// Raw (template-free) prompt
    #[arg(long)]
    prompt: Option<String>,

    /// The prompt as token ids, bypassing the tokenizer (a served request's
    /// ids, replayed as they were)
    #[arg(long, value_delimiter = ',', conflicts_with = "prompt")]
    prompt_ids: Vec<i64>,

    /// Max new tokens to generate
    #[arg(long)]
    steps: Option<usize>,

    /// CUDA device ordinal
    #[arg(long)]
    gpu: Option<usize>,

    /// State capacity in tokens (KV pages etc.); rounded down to the
    /// manifest's page unit. Default: what one sequence can reach (the
    /// manifest's page-table row)
    #[arg(long)]
    capacity: Option<u64>,

    /// Prefill chunk in tokens: the manifest's `tokens` bound unless a
    /// smaller one is asked for
    #[arg(long)]
    chunk: Option<u64>,

    /// Debug: launch every program eagerly, ignoring the manifest's `graph`
    #[arg(long)]
    eager: bool,

    /// Rows per sequence of a decode step: a shape some program of the
    /// manifest declares (1 for a plain step, its block for a speculative
    /// round). Default: the widest declared
    #[arg(long)]
    rows: Option<u64>,

    /// Extra token ids that end generation (comma-separated); the eos ids
    /// the checkpoint declares in generation_config.json always apply
    #[arg(long, value_delimiter = ',')]
    stop_tokens: Vec<i64>,

    /// Debug: dump activations of the first prefill chunk and `--probe-steps`
    /// decode steps into this directory, then exit: after every call whose
    /// label matches `--probe-labels`, the buffer it writes (live rows),
    /// plus the logits the step's tokens are taken from and the tokens.
    /// Programs run call-range by call-range so nothing executes twice.
    #[arg(long)]
    probe_dir: Option<PathBuf>,
    /// Call labels `--probe-dir` dumps after: comma-separated, a label
    /// matches one it equals or ends with (the file is named by the label
    /// minus its last `.part`)
    #[arg(long, default_value = "embed,.down_proj")]
    probe_labels: String,
    /// Decode steps `--probe-dir` dumps
    #[arg(long, default_value_t = 2)]
    probe_steps: usize,
}

/// Resolved options: the shared inputs, plus what only `kern run` has an
/// opinion about.
struct Opts {
    inputs: Inputs,
    tokenizer: PathBuf,
    prompt: String,
    prompt_ids: Vec<i64>,
    steps: usize,
    capacity: Option<u64>,
    chunk: Option<u64>,
    eager: bool,
    rows: Option<u64>,
    probe_dir: Option<PathBuf>,
    probe_labels: String,
    probe_steps: usize,
}

impl RunOpts {
    fn resolve(self, cfg: Option<&Config>, t: Option<&Target>) -> Result<Opts> {
        let given = Given {
            manifest: self.manifest,
            kernels: self.kernels,
            weights: self.weights,
            tokenizer: self.tokenizer,
            stop_tokens: self.stop_tokens,
            gpu: self.gpu,
            ..Default::default()
        };
        let inputs = Inputs::resolve(given, cfg, t)?;
        ensure!(
            !inputs.stop_tokens.is_empty(),
            "no stop tokens: no eos_token_id in the weights dir(s)' generation_config.json / config.json and no --stop-tokens"
        );
        Ok(Opts {
            tokenizer: inputs.tokenizer()?.to_path_buf(),
            prompt: self
                .prompt
                .or_else(|| cfg.and_then(|c| c.run.prompt.clone()))
                .unwrap_or_else(|| "The capital of France is".into()),
            prompt_ids: self.prompt_ids,
            steps: self.steps.or_else(|| cfg.and_then(|c| c.run.steps)).unwrap_or(32),
            capacity: self.capacity.or_else(|| cfg.and_then(|c| c.capacity)),
            chunk: self.chunk.or_else(|| cfg.and_then(|c| c.run.chunk)),
            eager: self.eager,
            rows: self.rows,
            probe_dir: self.probe_dir,
            probe_labels: self.probe_labels,
            probe_steps: self.probe_steps,
            inputs,
        })
    }
}

/// `kern run`.
pub fn run(o: RunOpts, cfg: Option<&Config>, target: Option<&Target>) -> Result<()> {
    execute(o.resolve(cfg, target)?)
}

fn human(bytes: u64) -> String {
    match bytes {
        b if b >= 1 << 30 => format!("{:.1} GiB", b as f64 / (1u64 << 30) as f64),
        b if b >= 1 << 20 => format!("{:.1} MiB", b as f64 / (1u64 << 20) as f64),
        b if b >= 1 << 10 => format!("{:.1} KiB", b as f64 / 1024.0),
        b => format!("{b} B"),
    }
}

fn ellipsize(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(n - 1).collect::<String>())
    }
}

/// What was loaded, before anything runs: the sizes the manifest asked
/// for and the module every op resolved to. A run that is wrong in a
/// boring way (the wrong kernels dir, a state far bigger than expected)
/// is visible here rather than in the output.
fn banner(rt: &Runtime, manifest: &Path, kernels: &Path, load: Duration) {
    let m = &rt.manifest;
    info!("manifest `{}` (schema v{}, {}): verified", m.model, m.schema_version, manifest.display());
    for (name, v) in &m.vars {
        info!("  var      {name} ∈ [{}, {}] (caller-provided per call)", kern_manifest::types::Var::MIN, v.max);
    }
    for (name, st, alloc) in rt.state_sizes() {
        if st.bytes_per_token > 0 {
            info!(
                "  state    {name}: opaque, {} B/token × capacity {} = {}",
                st.bytes_per_token,
                rt.capacity(),
                human(alloc)
            );
        } else if st.is_per_seq() {
            info!(
                "  state    {name}: opaque, {} per sequence × {} slots = {}",
                human(st.bytes_per_seq),
                rt.seq_slots(),
                human(alloc)
            );
        } else {
            info!("  state    {name}: opaque, fixed {}", human(alloc));
        }
    }
    let mut by_kind: BTreeMap<String, (usize, u64)> = BTreeMap::new();
    for (_, kind, bytes) in rt.buffer_sizes() {
        let e = by_kind.entry(kind.to_string()).or_default();
        e.0 += 1;
        e.1 += bytes;
    }
    let kinds = ["weight", "workspace", "carry", "input", "output"]
        .iter()
        .filter_map(|c| by_kind.get(*c).map(|(n, b)| format!("{c} {n} ({})", human(*b))))
        .collect::<Vec<_>>()
        .join(" | ");
    info!("  buffers  {kinds}");
    for (name, p) in &m.programs {
        let shape = match &p.batch {
            Some(b) => format!(", {} × {:?} per call", b.groups, b.rows),
            None if p.once => ", once after load".into(),
            None => String::new(),
        };
        info!("  program  `{name}`: {} calls{shape}", p.calls.len());
    }
    info!(
        "op resolution: {} of the {} modules the manifest pins loaded from {}, entries matched by \
         cuFuncGetParamInfo layout vs declared params ({load:?}):",
        rt.module_count(),
        m.modules.len(),
        kernels.display(),
    );
    let ones = BTreeMap::from_iter(m.vars.keys().map(|v| (v.clone(), 1)));
    for (name, modules) in rt.op_resolution() {
        let op = &m.ops[&name];
        for (li, (l, module)) in op.imp.launches.iter().zip(&modules).enumerate() {
            let label = if li == 0 { name.clone() } else { format!("  ·launch{li}") };
            let sm = match l.kernel().and_then(|k| k.shared_mem.as_ref()) {
                Some(e) => format!(", shmem {:?}", e.eval(&ones).unwrap_or(0)),
                None => String::new(),
            };
            let block = l.kernel().map_or(String::new(), |k| format!(", block {:?}", k.block));
            info!(
                "  {label:<18} {:<44} {:>2} params{block}{sm} <- {module}",
                ellipsize(l.entry(), 44),
                l.params_of(op).len(),
            );
        }
    }
}

fn execute(o: Opts) -> Result<()> {
    let manifest_json = std::fs::read_to_string(&o.inputs.manifest)
        .with_context(|| format!("reading manifest {}", o.inputs.manifest.display()))?;
    let t0 = Instant::now();
    let verified = Verified::from_json(&manifest_json)?;
    // One sequence: its reach, unless told otherwise (a manifest without
    // paged state takes the runtime's fit).
    let capacity = o
        .capacity
        .or_else(|| kern_runtime::seq_capacity(&verified))
        .map(|tokens| Capacity { tokens: Some(tokens), seqs: 1 });
    let mut rt = Runtime::load(&verified, &o.inputs.kernels, o.inputs.gpu, capacity, None)?;
    rt.set_eager(o.eager);
    banner(&rt, &o.inputs.manifest, &o.inputs.kernels, t0.elapsed());

    let t0 = Instant::now();
    o.inputs.weights.bind(&mut rt, &Topology::default())?;
    let n_weights = rt.buffer_sizes().iter().filter(|(_, k, _)| *k == BufferKind::Weight).count();
    info!("weights: {n_weights} buffers assembled from {} in {:?}", o.inputs.weights, t0.elapsed());

    let tokenizer = tokenizers::Tokenizer::from_file(&o.tokenizer).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    info!("tokenizer {} · stop tokens {:?}", o.tokenizer.display(), o.inputs.stop_tokens);
    let prompt_ids: Vec<i64> = if o.prompt_ids.is_empty() {
        tokenizer
            .encode(o.prompt.as_str(), false)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?
            .get_ids()
            .iter()
            .map(|&u| u as i64)
            .collect()
    } else {
        o.prompt_ids.clone()
    };
    ensure!(!prompt_ids.is_empty(), "empty prompt");
    info!("prompt: {} tokens {prompt_ids:?}", prompt_ids.len());

    let mut caller = Caller::new(rt)?;
    let rows = o.rows.or_else(|| caller.protocol.row_shapes().last().copied()).unwrap_or(1);
    let step = caller.forward(Rows::Const(rows))?;
    let chunk_f = caller.chunk_forward()?;
    info!(
        "protocol: prompt through `{}` (rows as fed{}), steps through `{}` ({rows} rows{}), fills {:?}",
        chunk_f.name,
        if chunk_f.emits.is_some() { ", emits a token" } else { "" },
        step.name,
        if step.count.is_some() { ", counted" } else { "" },
        caller.protocol.fills.iter().map(|f| format!("{}={}", f.fill, f.name)).collect::<Vec<_>>()
    );
    ensure!(
        prompt_ids.len() + rows as usize <= caller.limit(),
        "prompt of {} tokens plus a {rows}-row step does not fit the sequence's {} token slots (raise --capacity)",
        prompt_ids.len(),
        caller.limit()
    );

    let chunk = chunk_size(o.chunk, caller.protocol.rows.max)?;
    if let Some(dir) = &o.probe_dir {
        return crate::probe::probe(&mut caller, &prompt_ids, dir, &o.probe_labels, chunk, &step, o.probe_steps);
    }

    let n_pre = caller.prefill_len(prompt_ids.len())?;
    let mut generated: Vec<i64> = Vec::new();
    if n_pre > 0 {
        let t = Instant::now();
        let first = caller.prefill(&prompt_ids[..n_pre], chunk)?;
        let dt = t.elapsed();
        let pos = caller.pos;
        let n_chunks = (pos as u64).div_ceil(chunk);
        info!(
            "prefill: {pos} tokens in {n_chunks} chunk(s) of <= {chunk} ({dt:?}, {:.0} tok/s{})",
            pos as f64 / dt.as_secs_f64(),
            if first.is_some() { ", emits the first token" } else { "" }
        );
        if let Some(first) = first {
            if o.inputs.stop_tokens.contains(&first) {
                info!("stop token {first} at pos {pos}");
                println!("{}", o.prompt);
                return Ok(());
            }
            generated.push(first);
        }
    }

    let vars = caller.protocol.vars(1, rows, rows);
    let mut decode_ns: u128 = 0;
    let mut steps = 0u32;
    let mut taken = 0usize;

    'steps: while generated.len() < o.steps {
        let pos = caller.pos as usize;
        let tok = if pos < prompt_ids.len() { prompt_ids[pos] } else { *generated.last().unwrap() };
        caller.stage_rows(tok, rows)?;
        let t = Instant::now();
        caller.rt.issue(&step.name, &vars)?;
        caller.rt.synchronize()?;
        let out = caller.emitted(&step)?;
        decode_ns += t.elapsed().as_nanos();
        steps += 1;
        caller.advance(out.len() as u64);
        taken += out.len() - 1;
        for next in out {
            if o.inputs.stop_tokens.contains(&next) {
                info!("stop token {next} at pos {}", caller.pos);
                break 'steps;
            }
            generated.push(next);
            if generated.len() >= o.steps {
                break 'steps;
            }
        }
        if caller.pos as usize + rows as usize > caller.limit() {
            break;
        }
    }

    let gen_u32: Vec<u32> = generated.iter().map(|&t| t as u32).collect();
    let text = tokenizer.decode(&gen_u32, false).map_err(|e| anyhow::anyhow!("decode: {e}"))?;
    info!("generated ids: {generated:?}");
    info!(
        "{} tokens generated in {steps} steps of {rows} rows, {:.2} ms/step ({:.1} tok/s{})",
        generated.len(),
        decode_ns as f64 / 1e6 / steps.max(1) as f64,
        generated.len() as f64 * 1e9 / decode_ns.max(1) as f64,
        if rows > 1 {
            format!(
                ", {:.2} tokens/step, {:.1}% of the {} drafted rows taken",
                generated.len() as f64 / steps.max(1) as f64,
                taken as f64 * 100.0 / (steps.max(1) as u64 * (rows - 1)) as f64,
                rows - 1
            )
        } else {
            String::new()
        },
    );
    println!("{}{}", o.prompt, text);
    Ok(())
}

/// The prefill chunk: the manifest's `tokens` bound, or the smaller one
/// asked for; a bigger one is the caller's mistake, not clamped.
fn chunk_size(asked: Option<u64>, max: u64) -> Result<u64> {
    match asked {
        None => Ok(max),
        Some(c) if (1..=max).contains(&c) => Ok(c),
        Some(c) => anyhow::bail!("--chunk {c}: the manifest's `tokens` bound is {max}; a chunk can only be smaller"),
    }
}
