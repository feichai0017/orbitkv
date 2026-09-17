//! `kern.toml`: what the manifest cannot know — where things are on this
//! machine, which manifest is the reference, how the tap is seeded.
//!
//! Found by walking up from the cwd (or given with `--config`). Relative
//! paths are relative to the file; a leading `~` expands. Targets are
//! names the user picks; kern does not interpret them. Everything the
//! manifest already knows (page size, programs, buffer kinds, module
//! hashes) stays out of here. No `kern.toml` → every command takes its
//! inputs from flags, as before.
//!
//! ```toml
//! gpu = 0
//! capacity = 4096                       # optional; without it kern run takes one sequence's reach
//!
//! [targets.a]
//! manifest  = "examples/x.json"        # B: the manifest under work
//! reference = "ref/x.json"             # A: a copy you trust (kern test)
//! kernels   = "kernels-x"              # one dir, both versions, by sha
//! weights   = ["/weights/x.safetensors"]  # checkpoint dirs / .safetensors files
//! tokenizer = "/weights/tokenizer.json"  # kern run; kern test only with a prompt
//!
//! [kernels]                            # kern kernels
//! dumps   = ["~/dumps/x"]              # capture dumps to extract from
//! sources = "tools/kernels-src"        # handwritten .cu, built by nvcc
//!
//! [test]
//! seed = 0x5eed
//! decode_steps = 32
//! logit_kl = 0.01
//!
//! [run]
//! steps = 32
//! ```

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::Deserialize;

#[derive(Deserialize, Debug, Default, Clone)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub(crate) gpu: Option<usize>,
    pub(crate) capacity: Option<u64>,
    #[serde(default)]
    pub targets: BTreeMap<String, Target>,
    #[serde(default)]
    pub kernels: Kernels,
    #[serde(default)]
    pub(crate) test: Test,
    #[serde(default)]
    pub(crate) run: Run,
    /// Where the file was found.
    #[serde(skip)]
    pub path: PathBuf,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct Target {
    pub manifest: PathBuf,
    pub reference: Option<PathBuf>,
    pub kernels: PathBuf,
    #[serde(default)]
    pub(crate) weights: Vec<String>,
    pub(crate) tokenizer: Option<PathBuf>,
}

#[derive(Deserialize, Debug, Default, Clone)]
#[serde(deny_unknown_fields)]
pub struct Kernels {
    #[serde(default)]
    pub dumps: Vec<PathBuf>,
    pub sources: Option<PathBuf>,
}

#[derive(Deserialize, Debug, Default, Clone)]
#[serde(deny_unknown_fields)]
pub struct Test {
    pub(crate) seed: Option<u64>,
    pub(crate) decode_steps: Option<u64>,
    pub(crate) logit_kl: Option<f64>,
    pub(crate) prompt: Option<String>,
}

#[derive(Deserialize, Debug, Default, Clone)]
#[serde(deny_unknown_fields)]
pub struct Run {
    pub(crate) prompt: Option<String>,
    pub(crate) steps: Option<usize>,
    pub(crate) chunk: Option<u64>,
}

pub(crate) const FILE: &str = "kern.toml";

impl Config {
    /// The nearest `kern.toml` at or above the cwd, or the one given.
    pub fn find(explicit: Option<&Path>) -> Result<Option<Config>> {
        if let Some(p) = explicit {
            return Ok(Some(Self::load(p)?));
        }
        let mut dir = std::env::current_dir()?;
        loop {
            let p = dir.join(FILE);
            if p.is_file() {
                return Ok(Some(Self::load(&p)?));
            }
            if !dir.pop() {
                return Ok(None);
            }
        }
    }

    fn load(path: &Path) -> Result<Config> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let mut c: Config = toml::from_str(&text).with_context(|| format!("{}", path.display()))?;
        c.path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        let dir = c.path.parent().map(Path::to_path_buf).unwrap_or_default();
        for t in c.targets.values_mut() {
            t.manifest = abs(&dir, &t.manifest);
            t.reference = t.reference.as_ref().map(|p| abs(&dir, p));
            t.kernels = abs(&dir, &t.kernels);
            t.weights = t.weights.iter().map(|w| abs(&dir, Path::new(w)).to_string_lossy().into_owned()).collect();
            t.tokenizer = t.tokenizer.as_ref().map(|p| abs(&dir, p));
        }
        c.kernels.dumps = c.kernels.dumps.iter().map(|p| abs(&dir, p)).collect();
        c.kernels.sources = c.kernels.sources.as_ref().map(|p| abs(&dir, p));
        Ok(c)
    }

    pub fn dir(&self) -> &Path {
        self.path.parent().unwrap_or(Path::new("."))
    }

    /// Exactly one target: the named one, or the only one.
    pub fn one(&self, name: Option<&str>) -> Result<(&String, &Target)> {
        match name {
            Some(n) => self.targets.get_key_value(n).ok_or_else(|| {
                anyhow::anyhow!("no target `{n}` in {} (targets: {})", self.path.display(), self.names())
            }),
            None if self.targets.len() == 1 => Ok(self.targets.iter().next().unwrap()),
            None if self.targets.is_empty() => bail!("{} declares no targets", self.path.display()),
            None => bail!("{} has several targets ({}); name one", self.path.display(), self.names()),
        }
    }

    /// The named targets, or all of them.
    pub fn select(&self, names: &[String]) -> Result<Vec<(&String, &Target)>> {
        if names.is_empty() {
            if self.targets.is_empty() {
                bail!("{} declares no targets", self.path.display());
            }
            return Ok(self.targets.iter().collect());
        }
        names.iter().map(|n| self.one(Some(n))).collect()
    }

    fn names(&self) -> String {
        self.targets.keys().cloned().collect::<Vec<_>>().join(", ")
    }
}

fn abs(dir: &Path, p: &Path) -> PathBuf {
    let p = match p.strip_prefix("~") {
        Ok(rest) => match std::env::var_os("HOME") {
            Some(h) => PathBuf::from(h).join(rest),
            None => p.to_path_buf(),
        },
        Err(_) => p.to_path_buf(),
    };
    if p.is_absolute() {
        p
    } else {
        dir.join(p)
    }
}
