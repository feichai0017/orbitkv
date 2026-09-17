//! E1 gate: one pruned-K3 MoE layer as a kern program at EP4 on one tray,
//! checked against the same layer at EP1 and against a host reference.
//!
//!   cargo run --release -p kern-runtime --example k3_moe_ep -- \
//!       --weights weights/k3-moe-l1 [--gpus 0,1,2,3] [--iters 20] [--cubins target/cubins]
//!
//! `--weights` holds what tools/export_k3_moe.py wrote: ep1.safetensors,
//! ep4-r<i>.safetensors and inputs.safetensors (x, topk_idx, topk_weight,
//! y_ref). The EP1 world runs all ranks' tokens on the first GPU; the EP4
//! world runs each rank's slice on its own GPU. Rank r's output must equal
//! rows [rT, (r+1)T) of the EP1 output bit for bit, and the EP1 output must
//! sit within tolerance of y_ref. Prints the captured per-layer time.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Barrier, Mutex};

use kern_runtime::{PeerHandle, Runtime, Topology};

type Posted = Arc<Mutex<Vec<Option<BTreeMap<String, PeerHandle>>>>>;
type Results = Arc<Mutex<Vec<Option<Result<(Vec<u8>, f64), String>>>>>;

const HIDDEN: usize = 3584;
const TOPK: usize = 16;

fn stage_cubins(cubins: &Path, kernels: &Path) {
    std::fs::create_dir_all(kernels).unwrap();
    for name in ["k3_mega_moe", "k3_mega_stage"] {
        let bytes = std::fs::read(cubins.join(format!("{name}.cubin")))
            .expect("cubin (tools/build_k3_mega.sh, build_kernels.sh)");
        let sha = hex::encode(<sha2::Sha256 as sha2::Digest>::digest(&bytes));
        std::fs::write(kernels.join(format!("{name}-{}.cubin", &sha[..12])), &bytes).unwrap();
    }
}

struct Inputs {
    x: Vec<u8>,
    topk_idx: Vec<u8>,
    topk_weight: Vec<u8>,
    y_ref: Vec<f32>,
    ranks: usize,
    tokens_per_rank: usize,
}

fn read_inputs(path: &Path) -> Inputs {
    let bytes = std::fs::read(path).expect("inputs.safetensors");
    let st = safetensors::SafeTensors::deserialize(&bytes).expect("safetensors");
    let (_, meta) = safetensors::SafeTensors::read_metadata(&bytes).unwrap();
    let md = meta.metadata().clone().unwrap_or_default();
    let t = |n: &str| st.tensor(n).unwrap_or_else(|_| panic!("tensor {n}")).data().to_vec();
    let y_ref: Vec<f32> = t("y_ref").as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
    Inputs {
        x: t("x"),
        topk_idx: t("topk_idx"),
        topk_weight: t("topk_weight"),
        y_ref,
        ranks: md["ranks"].parse().unwrap(),
        tokens_per_rank: md["tokens_per_rank"].parse().unwrap(),
    }
}

fn bf16_to_f32(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<2>().0.iter().map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16)).collect()
}

/// Load, stage inputs for rows [row0, row0 + rows), run once, return y rows
/// and the captured per-run time.
#[allow(clippy::too_many_arguments)]
fn run_world(
    manifest: &kern_manifest::Verified,
    kernels: &Path,
    gpu: usize,
    topo: &Topology,
    weights: &[u8],
    inp: &Inputs,
    row0: usize,
    rows: usize,
    iters: usize,
    rendezvous: &dyn Fn(&mut Runtime) -> kern_runtime::Result<()>,
    sync: &dyn Fn(),
) -> kern_runtime::Result<(Vec<u8>, f64)> {
    let mut rt =
        Runtime::load(manifest, kernels, gpu, Some(kern_runtime::Capacity { tokens: Some(1), seqs: 1 }), Some(topo))?;
    rt.load_weights(&kern_runtime::Safetensors::parse(&[weights])?)?;
    rendezvous(&mut rt)?;
    let vars: BTreeMap<String, u64> = [("tokens".to_string(), rows as u64)].into();
    rt.write_input_at("x", &inp.x[row0 * HIDDEN * 2..(row0 + rows) * HIDDEN * 2], &vars)?;
    rt.write_input_at("topk_idx", &inp.topk_idx[row0 * TOPK * 4..(row0 + rows) * TOPK * 4], &vars)?;
    rt.write_input_at("topk_weight", &inp.topk_weight[row0 * TOPK * 4..(row0 + rows) * TOPK * 4], &vars)?;
    sync();
    rt.run("moe", &vars)?;
    let y = rt.read_output("y")?[..rows * HIDDEN * 2].to_vec();
    // Timing: every rank replays the captured program in lockstep.
    rt.capture("moe", &vars)?;
    sync();
    let ms = rt.time_captured("moe", &vars, iters)?;
    let y2 = rt.read_output("y")?[..rows * HIDDEN * 2].to_vec();
    if y2 != y {
        eprintln!("gpu {gpu}: output changed across captured replays");
    }
    Ok((y, ms as f64 * 1e3))
}

fn main() {
    let mut weights = PathBuf::from("weights/k3-moe-l1");
    let mut cubins = PathBuf::from("target/cubins");
    let mut gpus: Vec<usize> = vec![0, 1, 2, 3];
    let mut iters = 20usize;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut v = || args.next().expect("value");
        match a.as_str() {
            "--weights" => weights = PathBuf::from(v()),
            "--cubins" => cubins = PathBuf::from(v()),
            "--gpus" => gpus = v().split(',').map(|s| s.parse().unwrap()).collect(),
            "--iters" => iters = v().parse().unwrap(),
            _ => panic!("unknown arg {a}"),
        }
    }
    let inp = read_inputs(&weights.join("inputs.safetensors"));
    let (n, t) = (inp.ranks, inp.tokens_per_rank);
    assert_eq!(gpus.len(), n, "--gpus must name one GPU per rank ({n})");
    let kernels = std::env::temp_dir().join(format!("kern-k3-moe-{}", std::process::id()));
    stage_cubins(&cubins, &kernels);
    let load = |path: String| {
        let json = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        kern_manifest::Verified::from_json(&json).unwrap_or_else(|e| panic!("{path}: {e}"))
    };
    let json1 = load("examples/k3-moe-l1-ep1.json".into());
    let json_n = load(format!("examples/k3-moe-l1-ep{n}.json"));
    let inp = Arc::new(inp);

    // ---- EP1 oracle: all n*t tokens on the first GPU.
    let w1 = std::fs::read(weights.join("ep1.safetensors")).expect("ep1.safetensors");
    let self_import = |rt: &mut Runtime| -> kern_runtime::Result<()> {
        let mine = rt.export_handles()?;
        rt.import_peers("ep", &[mine])
    };
    let (y1, ms1) = run_world(
        &json1,
        &kernels,
        gpus[0],
        &Topology::one("ep", 0, 1),
        &w1,
        &inp,
        0,
        n * t,
        iters,
        &self_import,
        &|| {},
    )
    .unwrap_or_else(|e| panic!("EP1: {e}"));
    println!("EP1 on gpu {}: {} tokens, {ms1:.1} us/layer (captured)", gpus[0], n * t);
    drop(w1);

    // Reference check on the EP1 output.
    let y1f = bf16_to_f32(&y1);
    let mut max_abs = 0f32;
    let mut sum_sq_err = 0f64;
    let mut sum_sq_ref = 0f64;
    let mut bad = 0usize;
    for (a, r) in y1f.iter().zip(&inp.y_ref) {
        let e = (a - r).abs();
        max_abs = max_abs.max(e);
        sum_sq_err += (e as f64).powi(2);
        sum_sq_ref += (*r as f64).powi(2);
        if e > 0.05 * r.abs() + 0.05 {
            bad += 1;
        }
    }
    let rel_rms = (sum_sq_err / sum_sq_ref.max(1e-30)).sqrt();
    println!(
        "EP1 vs reference: max |err| {max_abs:.4}, rel RMS {rel_rms:.2e}, {bad}/{} elements outside 5% + 0.05",
        y1f.len()
    );

    // ---- EP n: rank r on gpus[r], rows [r*t, (r+1)*t).
    let posted: Posted = Arc::new(Mutex::new(vec![None; n]));
    let gate = Arc::new(Barrier::new(n));
    let results: Results = Arc::new(Mutex::new(vec![None; n]));
    let mut threads = Vec::new();
    for (rank, &gpu) in gpus.iter().enumerate() {
        let (json, kernels, posted, gate, results, inp, weights) = (
            json_n.clone(),
            kernels.clone(),
            posted.clone(),
            gate.clone(),
            results.clone(),
            inp.clone(),
            weights.clone(),
        );
        threads.push(std::thread::spawn(move || {
            let w = std::fs::read(weights.join(format!("ep{n}-r{rank}.safetensors"))).expect("rank shard");
            let rendezvous = |rt: &mut Runtime| -> kern_runtime::Result<()> {
                let mine = rt.export_handles()?;
                posted.lock().unwrap()[rank] = Some(mine);
                gate.wait();
                let members: Vec<_> = posted.lock().unwrap().iter().map(|m| m.clone().unwrap()).collect();
                rt.import_peers("ep", &members)
            };
            let sync = || {
                gate.wait();
            };
            let r = run_world(
                &json,
                &kernels,
                gpu,
                &Topology::one("ep", rank as u64, n as u64),
                &w,
                &inp,
                rank * t,
                t,
                iters,
                &rendezvous,
                &sync,
            );
            results.lock().unwrap()[rank] = Some(r.map_err(|e| e.to_string()));
        }));
    }
    for th in threads {
        th.join().unwrap();
    }
    let mut ok = bad == 0;
    for (rank, r) in results.lock().unwrap().iter().enumerate() {
        match r {
            Some(Ok((y, ms))) => {
                let want = &y1[rank * t * HIDDEN * 2..(rank + 1) * t * HIDDEN * 2];
                let diff = y.iter().zip(want).filter(|(a, b)| a != b).count();
                let same = diff == 0;
                ok &= same;
                println!(
                    "EP{n} rank {rank} gpu {}: {t} tokens, {ms:.1} us/layer (captured), vs EP1 rows: {}",
                    gpus[rank],
                    if same { "bit-identical".to_string() } else { format!("{diff} bytes differ") }
                );
            }
            Some(Err(e)) => {
                ok = false;
                println!("EP{n} rank {rank}: {e}");
            }
            None => {
                ok = false;
                println!("EP{n} rank {rank}: no result");
            }
        }
    }
    println!("{}", if ok { "PASS" } else { "FAIL" });
    std::process::exit(if ok { 0 } else { 1 });
}
