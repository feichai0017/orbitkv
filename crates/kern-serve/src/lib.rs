//! `kern-serve`: an OpenAI-compatible HTTP endpoint over a kern manifest.
//!
//! The HTTP/protocol stack is pegainfer's frontend (vLLM's Rust server
//! crates underneath: completions, chat completions, streaming, chat
//! templates, stop strings). This crate contributes the engine behind it:
//! `scheduler::KernScheduler`, the pegainfer `Scheduler` contract over a
//! `tray::Tray` — one `kern_runtime::Runtime` per GPU, driven in lockstep
//! (KV pages are the runtimes' leases). The crate's public surface is
//! [`serve`] and its option structs.

#![deny(unsafe_code)]

pub mod logline;
mod scheduler;
mod tray;

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Args;
use kern_manifest::Verified;
use kern_runtime::{Capacity, Runtime, Topology};
use pegainfer_frontend::engine::{
    drive, scheduler_pair, Engine, EngineInfo, KvCapacity, LaunchedEngine, LiveScheduler,
};
use pegainfer_frontend::vllm;
use tracing::info;

use scheduler::{KernScheduler, Policy};
use tray::Tray;

/// The manifest and its artifacts, as named on the command line.
pub struct Artifacts {
    pub manifest: PathBuf,
    pub kernels: PathBuf,
    /// Weight entries, files or a weight cache; see [`kern_run::Weights`].
    pub weights: Vec<String>,
}

#[derive(Args, Debug, Clone)]
pub struct ServeOpts {
    /// Model id served by the API (default: the manifest's `model`)
    #[arg(long)]
    pub served_model_name: Option<String>,

    #[arg(long, default_value_t = 8000)]
    pub port: u16,

    /// CUDA device ordinals, one per rank of the tray, in rank order (a
    /// manifest with a topology needs every group's members; `tp` groups
    /// are consecutive ranks). Default: 0
    #[arg(long, value_delimiter = ',')]
    pub gpus: Vec<usize>,

    /// KV pool in tokens per rank (rounded down to the page); every request
    /// reserves its worst case `prompt + max_tokens` at admission. Default:
    /// whatever device memory is left once weights, activations and scratch
    /// are allocated, less 1 GiB
    #[arg(long)]
    pub capacity: Option<u64>,

    /// Prefill chunk in tokens: the manifest's `tokens` bound unless a
    /// smaller one is asked for
    #[arg(long)]
    pub chunk: Option<u64>,

    /// Cap on concurrently running sequences per rank (≤ the manifest's
    /// `seqs` bound)
    #[arg(long, default_value_t = 256)]
    pub max_seqs: usize,

    /// Debug: launch every program eagerly, ignoring the manifest's `graph`
    #[arg(long)]
    pub eager: bool,

    /// Rows per sequence of a step: a shape some program of the manifest
    /// declares (1 for a plain step, its block for a speculative round).
    /// Default: the widest declared
    #[arg(long)]
    pub rows: Option<u64>,

    /// Extra stop token ids (generation_config.json's eos ids always apply)
    #[arg(long, value_delimiter = ',')]
    pub stop_tokens: Vec<u32>,

    /// Pinned host memory (GiB) per rank for snapshots parked off the
    /// device: a lease short of pages or slots parks the coldest snapshot
    /// there instead of dropping it, and a prompt hitting one wakes it
    /// (0: off)
    #[arg(long, default_value_t = 0.0)]
    pub host_gib: f64,
}

pub fn serve(o: ServeOpts, art: Artifacts) -> Result<()> {
    let weights = kern_run::Weights::parse(&art.weights)?;
    info!(
        manifest = %art.manifest.display(),
        kernels = %art.kernels.display(),
        %weights,
        kern = %*kern_run::VERSION,
        "loading"
    );
    let gpus = if o.gpus.is_empty() { vec![0] } else { o.gpus.clone() };
    let model_path = weights.dirs().remove(0);
    let mut stop_tokens: Vec<u32> = kern_run::eos_ids(&model_path).into_iter().map(|x| x as u32).collect();
    stop_tokens.extend(&o.stop_tokens);
    stop_tokens.sort_unstable();
    stop_tokens.dedup();
    anyhow::ensure!(
        !stop_tokens.is_empty(),
        "no stop tokens: none in {}/generation_config.json or config.json and no --stop-tokens",
        model_path.display()
    );
    info!(ids = ?stop_tokens, "stop tokens");

    let manifest_json = std::fs::read_to_string(&art.manifest)
        .with_context(|| format!("reading manifest {}", art.manifest.display()))?;
    let manifest =
        Verified::from_json(&manifest_json).with_context(|| format!("manifest {}", art.manifest.display()))?;
    let served_name = o.served_model_name.clone().unwrap_or_else(|| manifest.model.clone());
    // Every sequence of a tray batch group holds a token slot on each of
    // its `t` ranks, and each rank its pad.
    let t = manifest.group_size("tp").unwrap_or(1) as usize;
    let capacity = Capacity { tokens: o.capacity, seqs: ((o.max_seqs + 1) * t) as u64 };

    // The scheduler thread owns the tray for its whole life: load there,
    // report readiness, then drive.
    let (handle, backend) = scheduler_pair();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<scheduler::Facts>>();
    let host_bytes = (o.host_gib * (1u64 << 30) as f64) as u64;
    let policy =
        Policy { chunk: o.chunk.map(|c| c as usize), max_seqs: o.max_seqs, stop_tokens, rows: o.rows, host_bytes };
    let join = std::thread::Builder::new()
        .name("kern-scheduler".into())
        .spawn(move || {
            let load = || -> Result<KernScheduler> {
                let t0 = Instant::now();
                let bind = |rt: &mut Runtime, topo: &Topology| weights.bind(rt, topo);
                let tray = Tray::load(&manifest, &art.kernels, &gpus, capacity, &bind, host_bytes, o.eager)?;
                info!(model = %tray.manifest().model, gpus = ?gpus, load_s = logline::secs(t0.elapsed()), "tray loaded");
                KernScheduler::new(tray, policy)
            };
            match load() {
                Ok(sched) => {
                    let _ = ready_tx.send(Ok(sched.facts()));
                    // A scheduler that panicked would leave the port open
                    // and every request hanging: the process goes with it.
                    if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drive(sched, backend))).is_err() {
                        std::process::exit(101);
                    }
                }
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                }
            }
        })
        .context("spawning the scheduler thread")?;

    // The port opens once the engine is up: `serving` follows readiness.
    let (model, model_dir, port) = (served_name.clone(), model_path.clone(), o.port);
    let engine = async move {
        let facts = tokio::task::spawn_blocking(move || ready_rx.recv())
            .await
            .context("scheduler thread died before reporting readiness")?
            .context("scheduler thread died before reporting readiness")??;
        info!(model = %model, port, model_dir = %model_dir.display(), "serving");
        Ok(LaunchedEngine::Stepped(Engine {
            schedulers: vec![LiveScheduler { handle, join }],
            info: EngineInfo {
                kv_capacity: Some(KvCapacity { total_blocks: facts.total_blocks, block_size: facts.block_size }),
                servable_len: Some(facts.max_request_tokens as u32),
            },
            lora: None,
        }))
    };

    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    rt.block_on(async move {
        // Needs the runtime: it spawns the signal listener.
        let shutdown = vllm::shutdown_token_from_ctrl_c();
        vllm::serve_with_engine_count(engine, &model_path, vec![served_name], o.port, None, 1, shutdown).await
    })
}
