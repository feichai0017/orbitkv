//! Run one program of a manifest on one GPU with inputs from files and
//! outputs to files: the gate for a vendored kernel's op
//! (tools/kernel-capture/README.md, step 6) — the reference launcher's dumps
//! go in, what comes out is diffed against its dumps.
//!
//!   program_io --manifest m.json [--cubins target/cubins] [--gpu 0] [--program p]
//!              --vars var=value ... --in name=path ... --out name=path ...
//!              [--dump name=path ...] [--graph] [--iters 9]
//!
//! Inputs are written whole (`write_input`; a shorter file fills a prefix),
//! outputs are read whole; `--dump` reads any buffer whole (a weight as
//! bound, a carry a `once` program computed). A manifest with weight buffers gets `--weights`
//! checkpoints (a safetensors file or a directory of shards), in order. `--graph` captures after one eager run and writes outputs
//! after repeated replay; its median timing excludes capture and file I/O.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use kern_runtime::{Runtime, Topology};

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

fn pair(s: &str) -> (String, String) {
    let (a, b) = s.split_once('=').unwrap_or_else(|| panic!("`{s}`: expected name=value"));
    (a.to_string(), b.to_string())
}

fn main() -> anyhow::Result<()> {
    let mut manifest = PathBuf::new();
    let mut cubins = PathBuf::from("target/cubins");
    let mut gpu = 0usize;
    let mut program = String::new();
    let mut graph = false;
    let mut iters = 1usize;
    let mut vars = BTreeMap::new();
    let mut ins: Vec<(String, String)> = Vec::new();
    let mut outs: Vec<(String, String)> = Vec::new();
    let mut dumps: Vec<(String, String)> = Vec::new();
    let mut weights: Vec<PathBuf> = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut v = || args.next().expect("value");
        match a.as_str() {
            "--graph" => graph = true,
            "--iters" => iters = v().parse()?,
            "--manifest" => manifest = PathBuf::from(v()),
            "--cubins" => cubins = PathBuf::from(v()),
            "--gpu" => gpu = v().parse()?,
            "--program" => program = v(),
            "--vars" => {
                let (k, val) = pair(&v());
                vars.insert(k, val.parse::<u64>()?);
            }
            "--in" => ins.push(pair(&v())),
            "--out" => outs.push(pair(&v())),
            "--dump" => dumps.push(pair(&v())),
            "--weights" => weights.push(PathBuf::from(v())),
            _ => anyhow::bail!("unknown arg {a}"),
        }
    }
    anyhow::ensure!(iters > 0, "--iters must be positive");
    let json = std::fs::read_to_string(&manifest)?;
    let m = kern_manifest::Verified::from_json(&json)?;
    if program.is_empty() {
        anyhow::ensure!(m.programs.len() == 1, "--program: the manifest has {} programs", m.programs.len());
        program = m.programs.keys().next().unwrap().clone();
    }
    let kernels = std::env::temp_dir().join(format!("kern-program-io-{}", std::process::id()));
    stage_cubins(&cubins, &kernels);
    let mut rt = Runtime::load(&m, &kernels, gpu, None, Some(&Topology::default()))?;
    let maps = kern_run::map_weights(&weights)?;
    rt.load_weights(&kern_runtime::Safetensors::parse(&maps.iter().map(|m| &m[..]).collect::<Vec<_>>())?)?;
    for (name, path) in &ins {
        rt.write_input(name, &std::fs::read(path)?)?;
    }
    rt.run(&program, &vars)?;
    if graph {
        rt.capture(&program, &vars)?;
        rt.run_captured(&program, &vars)?;
        println!("graph_median_ms={}", rt.time_captured(&program, &vars, iters)?);
    } else {
        for _ in 1..iters {
            rt.run(&program, &vars)?;
        }
    }
    for (name, path) in &outs {
        std::fs::write(path, rt.read_output(name)?)?;
    }
    for (name, path) in &dumps {
        let bytes = rt.buffer_sizes().into_iter().find(|(n, _, _)| n == name).map(|(_, _, b)| b);
        let bytes = bytes.ok_or_else(|| anyhow::anyhow!("no buffer `{name}`"))?;
        std::fs::write(path, rt.read_buffer_prefix(name, bytes as usize)?)?;
    }
    println!("`{program}` ran with {vars:?}; {} inputs in, {} outputs out", ins.len(), outs.len());
    Ok(())
}
