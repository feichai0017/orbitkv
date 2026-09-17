//! `kern`: run a manifest, test a kernel swap, gather cubins, or say
//! what a manifest declares without a GPU.
//!
//! Inputs come from flags, else from the nearest `kern.toml` (see
//! `kern_run::config`). Targets are names the user picks there.

#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, ensure, Context, Result};
use clap::{Parser, Subcommand};
use kern_run::config::{Config, Target};
use kern_run::run::RunOpts;
use kern_run::test::TestOpts;

#[derive(Parser)]
#[command(
    name = "kern",
    version = kern_run::VERSION.as_str(),
    about = "model-agnostic GPU runtime: run a manifest, test a kernel swap, gather cubins, verify a manifest"
)]
struct Cli {
    /// kern.toml to use (default: the nearest one at or above the cwd; ignored by verify)
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Measure a sweep of call shapes: where a program's time goes, and what its calls cost alone
    Bench {
        /// Target in kern.toml (needed when it declares several)
        target: Option<String>,
        #[command(flatten)]
        opts: kern_run::bench::BenchOpts,
    },
    /// Greedy bs=1 generation over a target's manifest
    Run {
        /// Target in kern.toml (needed when it declares several)
        target: Option<String>,
        #[command(flatten)]
        opts: RunOpts,
    },
    /// Serve a target through the separately installed kern-serve binary
    Server {
        /// Target in kern.toml
        target: String,
        /// Arguments forwarded to kern-serve (for example --port 8000)
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<std::ffi::OsString>,
    },
    /// A/B a kernel swap: a target's `reference` (A) against its
    /// `manifest` (B). Exit 0 PASS, 1 FAIL, 2 INCONCLUSIVE
    Test {
        /// Target in kern.toml (needed when it declares several)
        target: Option<String>,
        #[command(flatten)]
        opts: TestOpts,
    },
    /// Build the handwritten cubins (`[kernels].sources`) and land every
    /// cubin pinned by each target's manifest and reference into its
    /// kernels dir, from `[kernels].dumps` and the builds
    Kernels { targets: Vec<String> },
    /// Verify a manifest and print its serving protocol: the axes, every
    /// fill, the tables, and the call shape each program accepts. No GPU.
    /// Exit 1 when verification or the protocol fails
    Verify {
        /// Manifest JSON to verify (does not read kern.toml)
        manifest: PathBuf,
    },
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .with_target(false)
        .without_time()
        .init();
    let cli = Cli::parse();
    let cfg = match &cli.cmd {
        Cmd::Verify { .. } => None,
        _ => Config::find(cli.config.as_deref())?,
    };
    match cli.cmd {
        Cmd::Bench { target, opts } => kern_run::bench::run(opts, cfg.as_ref(), target_of(cfg.as_ref(), target)?),
        Cmd::Run { target, opts } => kern_run::run::run(opts, cfg.as_ref(), target_of(cfg.as_ref(), target)?),
        Cmd::Server { target, args } => {
            let cfg = cfg.as_ref().context("kern server needs a kern.toml with targets")?;
            let (_, target) = cfg.one(Some(&target))?;
            kern_run::server::run(target, &args)
        }
        Cmd::Test { target, opts } => {
            let code = kern_run::test::run(opts, cfg.as_ref(), target_of(cfg.as_ref(), target)?)?;
            if code != 0 {
                std::process::exit(code);
            }
            Ok(())
        }
        Cmd::Kernels { targets } => kernels(cfg.as_ref(), &targets),
        Cmd::Verify { manifest } => {
            if !verify(&manifest) {
                std::process::exit(1);
            }
            Ok(())
        }
    }
}

/// The target a command runs on: the named one, else the only one, else
/// none at all — every command takes its inputs from flags when there is
/// no `kern.toml` to look in.
fn target_of(cfg: Option<&Config>, name: Option<String>) -> Result<Option<&Target>> {
    match cfg {
        Some(c) if !c.targets.is_empty() => Ok(Some(c.one(name.as_deref())?.1)),
        _ => {
            ensure!(
                name.is_none(),
                "no kern.toml with targets found at or above the cwd; `{}` cannot be looked up",
                name.unwrap_or_default()
            );
            Ok(None)
        }
    }
}

/// `kern verify`: verification, then the protocol, both reported in full.
fn verify(path: &Path) -> bool {
    use kern_manifest::protocol::{Axis, Rows};
    let json = match std::fs::read_to_string(path) {
        Ok(json) => json,
        Err(e) => {
            tracing::error!("{}: failed to read manifest: {e}", path.display());
            return false;
        }
    };
    let m = match kern_manifest::Verified::from_json(&json) {
        Ok(m) => m,
        Err(e) => {
            tracing::error!("{}: {e}", path.display());
            return false;
        }
    };
    let p = match kern_manifest::Protocol::check(&m) {
        Ok(p) => p,
        Err(e) => {
            tracing::error!("{}: {e}", path.display());
            return false;
        }
    };
    tracing::info!("{}: `{}` schema v{}, verified", path.display(), m.model, m.schema_version);
    let axis = |a: Axis| match a {
        Axis::Rows => format!("[{}]", p.rows.var),
        Axis::Groups => format!("[{}]", p.groups.var),
        Axis::Tray => format!("[{}]", p.tray.as_ref().map_or("tray", |t| t.var.as_str())),
        Axis::Fixed(n) => format!("[{n}]"),
    };
    println!("  rows      `{}` <= {}", p.rows.var, p.rows.max);
    println!("  groups    `{}` <= {}", p.groups.var, p.groups.max);
    if let Some(t) = &p.tray {
        println!("  tray      `{}` <= {}", t.var, t.max);
    }
    for f in &p.fills {
        let w = if f.width > 1 { format!(" x {}", f.width) } else { String::new() };
        println!("  fill      {:<11} `{}` {} {}{w}", f.fill.to_string(), f.name, f.dtype, axis(f.axis));
    }
    for t in &p.page_tables {
        println!("  pages     `{}` [{}, {}]", t.name, p.groups.var, t.width);
    }
    for t in &p.line_tables {
        let w = if t.width > 1 { format!(", {}", t.width) } else { String::new() };
        println!("  lines     `{}` [{}, {}{w}]", t.name, t.lines, axis(t.axis).trim_matches(['[', ']']));
    }
    for f in &p.forwards {
        let rows = match (f.rows, p.span.as_ref().filter(|_| f.span)) {
            (Rows::Const(_), Some(s)) => format!("1 row, one of them a run of up to {} (`{}`)", s.max, s.var),
            (Rows::Const(r), None) => format!("{r} rows"),
            (Rows::Var, _) => "rows as fed".into(),
        };
        let emits = match f.emits {
            Some(i) => format!(", hands back `{}`", p.fills[i].name),
            None => ", state only".into(),
        };
        let count = match f.count {
            Some(i) => format!(" counted by `{}`", p.fills[i].name),
            None => String::new(),
        };
        println!("  forward   `{}`: <= {} sequences of {rows}{emits}{count}", f.name, f.groups);
    }
    for o in &p.once {
        println!("  once      `{o}`");
    }
    true
}

/// `kern kernels`: the two tools scripts, driven from kern.toml.
fn kernels(cfg: Option<&Config>, targets: &[String]) -> Result<()> {
    let Some(cfg) = cfg else { bail!("kern kernels needs a kern.toml ([targets], [kernels])") };
    let tools = tools_dir(cfg)?;
    if let Some(src) = &cfg.kernels.sources {
        sh(Command::new(tools.join("build_kernels.sh")).env("KERN_SRC", src))?;
    }
    let dumps: Vec<String> = cfg.kernels.dumps.iter().map(|p| p.display().to_string()).collect();
    for (name, t) in cfg.select(targets)? {
        for m in std::iter::once(&t.manifest).chain(t.reference.iter()) {
            eprintln!("{name}: {} → {}", m.display(), t.kernels.display());
            // the script wants at least one search dir; the kernels dir itself is harmless
            let d = if dumps.is_empty() { t.kernels.display().to_string() } else { dumps.join(":") };
            sh(Command::new(tools.join("extract_kernels.sh")).arg(m).arg(&d).arg(&t.kernels))?;
        }
    }
    Ok(())
}

/// The repo's `tools/`: next to the kern.toml or any directory above it,
/// else next to this binary's `target/` (never the build-time path — the
/// binary is built in a container).
fn tools_dir(cfg: &Config) -> Result<PathBuf> {
    let mut cands: Vec<PathBuf> = cfg.dir().ancestors().map(|d| d.join("tools")).collect();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(root) = exe.parent().and_then(Path::parent).and_then(Path::parent) {
            cands.push(root.join("tools"));
        }
    }
    cands.into_iter().find(|d| d.join("extract_kernels.sh").is_file()).ok_or_else(|| {
        anyhow::anyhow!("no tools/extract_kernels.sh above {} or next to the binary", cfg.path.display())
    })
}

fn sh(cmd: &mut Command) -> Result<()> {
    let st = cmd.status().with_context(|| format!("running {cmd:?}"))?;
    ensure!(st.success(), "{cmd:?} exited with {st}");
    Ok(())
}
