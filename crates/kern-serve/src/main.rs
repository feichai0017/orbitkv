//! `kern-serve`: OpenAI-compatible HTTP over a manifest.
//!
//! `--manifest`, `--kernels` and `--weights` name the artifact, the same
//! three flags `kern run` takes; the first weights entry's checkpoint
//! directory is also what the frontend reads (config, tokenizer, chat
//! template, stop tokens). A server names every input on its command line; `kern.toml`
//! is the kernel-development loop's file and this binary never reads it.

#![deny(unsafe_code)]

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

#[derive(Parser)]
#[command(name = "kern-serve", version, about = "serve a kern manifest over an OpenAI-compatible HTTP endpoint")]
struct Cli {
    /// Manifest JSON (must pass verification)
    #[arg(long)]
    manifest: PathBuf,
    /// Directory of cubins, resolved by their pinned sha256
    #[arg(long)]
    kernels: PathBuf,
    /// Checkpoint directories or .safetensors files, one flag each (a
    /// manifest with a topology may write `{ep}` / `{tp}` and `*` for the
    /// rank's shard)
    #[arg(long, required = true)]
    weights: Vec<String>,
    #[command(flatten)]
    opts: kern_serve::ServeOpts,
}

fn main() -> Result<()> {
    kern_serve::logline::init();
    let cli = Cli::parse();
    let art = kern_serve::Artifacts { manifest: cli.manifest, kernels: cli.kernels, weights: cli.weights };
    kern_serve::serve(cli.opts, art)
}
