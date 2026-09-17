//! Where a command's inputs come from: the flag, else the target in
//! `kern.toml`, else what the checkpoint itself declares, else a default.
//!
//! `kern run`, `kern test` and `kern bench` need the same six things and
//! resolve them the same way, so a missing one reads the same whichever
//! command asked for it. What is left to each command is what only it has
//! an opinion about: how much state to allocate, how long to generate,
//! which workload to profile.

use std::path::PathBuf;

use anyhow::Result;

use crate::config::{Config, Target};
use crate::Weights;

/// What the flags gave. Everything is optional: the target fills the rest.
#[derive(Default)]
pub struct Given {
    pub manifest: Option<PathBuf>,
    pub reference: Option<PathBuf>,
    pub kernels: Option<PathBuf>,
    pub weights: Vec<String>,
    pub tokenizer: Option<PathBuf>,
    pub stop_tokens: Vec<i64>,
    pub gpu: Option<usize>,
}

/// The artifacts a command runs on, resolved.
#[derive(Debug)]
pub struct Inputs {
    pub manifest: PathBuf,
    /// The manifest to compare against; only `kern test` needs one.
    pub reference: Option<PathBuf>,
    pub kernels: PathBuf,
    pub weights: Weights,
    /// The flag, the target's, else the first checkpoint's own.
    pub tokenizer: Option<PathBuf>,
    /// The checkpoints' eos ids and the flag's, in that order, no repeats.
    pub stop_tokens: Vec<i64>,
    pub gpu: usize,
}

impl Inputs {
    pub fn resolve(g: Given, cfg: Option<&Config>, t: Option<&Target>) -> Result<Inputs> {
        let entries = match g.weights.is_empty() {
            true => t.map(|t| t.weights.clone()).filter(|w| !w.is_empty()).ok_or_else(|| need(cfg, "weights"))?,
            false => g.weights,
        };
        let weights = Weights::parse(&entries)?;
        let ck = crate::checkpoint(&weights.dirs());
        Ok(Inputs {
            manifest: g.manifest.or_else(|| t.map(|t| t.manifest.clone())).ok_or_else(|| need(cfg, "manifest"))?,
            reference: g.reference.or_else(|| t.and_then(|t| t.reference.clone())),
            kernels: g.kernels.or_else(|| t.map(|t| t.kernels.clone())).ok_or_else(|| need(cfg, "kernels"))?,
            tokenizer: g.tokenizer.or_else(|| t.and_then(|t| t.tokenizer.clone())).or(ck.tokenizer),
            stop_tokens: once_each(ck.stop_tokens.into_iter().chain(g.stop_tokens)),
            gpu: g.gpu.or_else(|| cfg.and_then(|c| c.gpu)).unwrap_or(0),
            weights,
        })
    }

    /// The tokenizer, for a command that cannot run without one.
    pub fn tokenizer(&self) -> Result<&std::path::Path> {
        self.tokenizer.as_deref().ok_or_else(|| {
            anyhow::anyhow!("no --tokenizer, no `tokenizer` in the target, and no tokenizer.json in the weights dir(s)")
        })
    }
}

/// What an input nobody gave reads as, whether or not there was a
/// `kern.toml` to look in.
pub(crate) fn need(cfg: Option<&Config>, what: &str) -> anyhow::Error {
    match cfg {
        Some(c) => anyhow::anyhow!("no --{what}, and the target in {} does not give one", c.path.display()),
        None => anyhow::anyhow!("no --{what}, and no {} found at or above the cwd", crate::config::FILE),
    }
}

fn once_each(ids: impl Iterator<Item = i64>) -> Vec<i64> {
    ids.fold(Vec::new(), |mut v, id| {
        if !v.contains(&id) {
            v.push(id);
        }
        v
    })
}
