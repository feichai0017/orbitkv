//! MegaMoE kernel bench at the pruned-K3 layer shape (hidden 3584,
//! intermediate 3072, 224 experts, top-16): time the `moe` program of
//! examples/k3-moe-l1-ep<R>.json over a sweep of per-rank token counts and
//! routing distributions, and print what the kernel achieved against the
//! work it was given — (token, expert) pairs, experts touched, weight bytes,
//! FLOPs, TFLOP/s, TB/s — per rank, so the busiest rank (the one a lockstep
//! step waits for) is visible.
//!
//!   k3_moe_bench --weights /data/<user>/kern-k3/moe-l1 --ranks 4 [--gpus 0,1,2,3]
//!       [--tokens 1,8,32,64,128,256,512,1024] [--routing uniform,narrow:16,file:<path>]
//!       [--iters 20] [--cubins target/cubins] [--manifest examples/k3-moe-l1-ep<R>.json]
//!       [--y-out <dir>]   (writes each case's y per rank, `y-<case>-r<rank>.bin`, to diff builds)
//!
//! Routing: `uniform` draws 16 distinct experts per token uniformly;
//! `narrow:<k>` confines every token to a fixed set of k experts spread
//! evenly over the ranks (k >= 16); `file:<path>` replays real router output,
//! a little-endian i32 [n, 16] array of global expert ids (dumped from a
//! k3_golden run with `K3_GOLDEN_DUMP_BUFS=topk_idx`), tokens taken in order.
//! Activations are seeded random bf16 in [-1, 1); weights 1/16 each.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Barrier, Mutex};

use kern_runtime::{Runtime, Topology};

const HIDDEN: usize = 3584;
const INTER: usize = 3072;
const EXPERTS: usize = 224;
const TOPK: usize = 16;
/// l1 [6144, 1792] u8 + l2 [3584, 1536] u8 + scale factors (i32 [28, 6144] + [24, 3584]).
const BYTES_PER_EXPERT: f64 = (6144 * 1792 + 3584 * 1536 + 4 * (28 * 6144 + 24 * 3584)) as f64;
/// Two GEMMs per (token, expert): [1, 3584] x [3584, 6144] and [1, 3072] x [3072, 3584].
const FLOP_PER_PAIR: f64 = 2.0 * (HIDDEN * 2 * INTER + INTER * HIDDEN) as f64;

fn stage_cubins(cubins: &Path, kernels: &Path) {
    std::fs::create_dir_all(kernels).unwrap();
    for name in ["k3_mega_moe", "k3_mega_stage"] {
        let bytes = std::fs::read(cubins.join(format!("{name}.cubin")))
            .expect("cubin (tools/build_k3_mega.sh, build_kernels.sh)");
        let sha = hex::encode(<sha2::Sha256 as sha2::Digest>::digest(&bytes));
        std::fs::write(kernels.join(format!("{name}-{}.cubin", &sha[..12])), &bytes).unwrap();
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

#[derive(Clone)]
enum Routing {
    Uniform,
    Narrow(usize),
    File(Arc<Vec<i32>>),
}

impl Routing {
    fn parse(s: &str) -> Routing {
        if s == "uniform" {
            Routing::Uniform
        } else if let Some(k) = s.strip_prefix("narrow:") {
            let k: usize = k.parse().expect("narrow:<k>");
            assert!((TOPK..=EXPERTS).contains(&k), "narrow:<k> needs 16 <= k <= 224");
            Routing::Narrow(k)
        } else if let Some(p) = s.strip_prefix("file:") {
            let bytes = std::fs::read(p).expect("routing file");
            let ids: Vec<i32> = bytes.as_chunks::<4>().0.iter().map(|c| i32::from_le_bytes(*c)).collect();
            assert!(ids.len().is_multiple_of(TOPK) && ids.iter().all(|&e| (0..EXPERTS as i32).contains(&e)));
            Routing::File(Arc::new(ids))
        } else {
            panic!("routing: uniform | narrow:<k> | file:<path>")
        }
    }

    fn name(&self) -> String {
        match self {
            Routing::Uniform => "uniform".into(),
            Routing::Narrow(k) => format!("narrow:{k}"),
            Routing::File(ids) => format!("file({} tokens)", ids.len() / TOPK),
        }
    }

    /// Global expert ids for `tokens` tokens, [tokens, 16], 16 distinct per token.
    fn draw(&self, tokens: usize, ranks: usize, seed: u64) -> Vec<i32> {
        let mut rng = Rng(seed | 1);
        let mut pick = |pool: &[usize]| -> Vec<i32> {
            let mut chosen: Vec<i32> = Vec::with_capacity(TOPK);
            while chosen.len() < TOPK {
                let e = pool[rng.below(pool.len())] as i32;
                if !chosen.contains(&e) {
                    chosen.push(e);
                }
            }
            chosen
        };
        match self {
            Routing::Uniform => {
                let pool: Vec<usize> = (0..EXPERTS).collect();
                (0..tokens).flat_map(|_| pick(&pool)).collect()
            }
            Routing::Narrow(k) => {
                // k experts spread evenly over the ranks: the first k/ranks of each rank's slice.
                let per = EXPERTS / ranks;
                let pool: Vec<usize> = (0..ranks).flat_map(|r| (0..k / ranks).map(move |i| r * per + i)).collect();
                (0..tokens).flat_map(|_| pick(&pool)).collect()
            }
            Routing::File(ids) => {
                let n = ids.len() / TOPK;
                (0..tokens).flat_map(|t| ids[(t % n) * TOPK..(t % n + 1) * TOPK].iter().copied()).collect()
            }
        }
    }
}

/// Per-rank work implied by a routing: (pairs, experts touched) for each rank.
fn load(idx: &[i32], ranks: usize) -> Vec<(usize, usize)> {
    let per = EXPERTS / ranks;
    let mut pairs = vec![0usize; ranks];
    let mut touched = vec![false; EXPERTS];
    for &e in idx {
        pairs[e as usize / per] += 1;
        touched[e as usize] = true;
    }
    (0..ranks).map(|r| (pairs[r], touched[r * per..(r + 1) * per].iter().filter(|&&t| t).count())).collect()
}

fn bf16_random(n: usize, seed: u64) -> Vec<u8> {
    let mut rng = Rng(seed | 1);
    (0..n)
        .flat_map(|_| {
            let f = (rng.next() >> 40) as f32 / (1u64 << 23) as f32 * 2.0 - 1.0;
            (f.to_bits() >> 16).to_le_bytes()
        })
        .collect()
}

struct Case {
    tokens: usize,
    routing: Routing,
}

/// One rank's runtime, driven through every case in lockstep with its peers.
#[allow(clippy::too_many_arguments)]
fn run_rank(
    manifest: &kern_manifest::Verified,
    kernels: &Path,
    gpu: usize,
    rank: usize,
    ranks: usize,
    weights: &[u8],
    cases: &[Case],
    iters: usize,
    y_out: Option<&Path>,
    rendezvous: &dyn Fn(&mut Runtime) -> kern_runtime::Result<()>,
    sync: &dyn Fn(),
) -> kern_runtime::Result<Vec<f64>> {
    let topo = Topology::one("ep", rank as u64, ranks as u64);
    let mut rt =
        Runtime::load(manifest, kernels, gpu, Some(kern_runtime::Capacity { tokens: Some(1), seqs: 1 }), Some(&topo))?;
    rt.load_weights(&kern_runtime::Safetensors::parse(&[weights])?)?;
    rendezvous(&mut rt)?;
    let mut out = Vec::new();
    for (i, c) in cases.iter().enumerate() {
        let t = c.tokens;
        // Every rank draws the whole tray's routing from the same seed and takes its own rows.
        let idx = c.routing.draw(t * ranks, ranks, 0x9e37_79b9 + i as u64);
        let mine = &idx[rank * t * TOPK..(rank + 1) * t * TOPK];
        let vars: BTreeMap<String, u64> = [("tokens".to_string(), t as u64)].into();
        let x = bf16_random(t * HIDDEN, 7 + rank as u64);
        let idx_bytes: Vec<u8> = mine.iter().flat_map(|e| e.to_le_bytes()).collect();
        let w_bytes: Vec<u8> = (0..t * TOPK).flat_map(|_| (1.0f32 / TOPK as f32).to_le_bytes()).collect();
        rt.write_input_at("x", &x, &vars)?;
        rt.write_input_at("topk_idx", &idx_bytes, &vars)?;
        rt.write_input_at("topk_weight", &w_bytes, &vars)?;
        sync();
        rt.run("moe", &vars)?;
        if let Some(dir) = y_out {
            let y = rt.read_output("y")?;
            std::fs::write(dir.join(format!("y-{i}-r{rank}.bin")), &y[..t * HIDDEN * 2]).expect("y-out");
        }
        rt.capture("moe", &vars)?;
        sync();
        let ms = rt.time_captured("moe", &vars, iters)?;
        out.push(ms as f64 * 1e3);
    }
    Ok(out)
}

fn main() {
    let mut weights = PathBuf::from("weights/k3-moe-l1");
    let mut cubins = PathBuf::from("target/cubins");
    let mut gpus: Vec<usize> = vec![0, 1, 2, 3];
    let mut ranks = 4usize;
    let mut iters = 20usize;
    let mut tokens: Vec<usize> = vec![1, 8, 32, 64, 128, 256, 512, 1024];
    let mut routings: Vec<Routing> = vec![Routing::Uniform];
    let mut manifest: Option<PathBuf> = None;
    let mut y_out: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut v = || args.next().expect("value");
        match a.as_str() {
            "--weights" => weights = PathBuf::from(v()),
            "--cubins" => cubins = PathBuf::from(v()),
            "--gpus" => gpus = v().split(',').map(|s| s.parse().unwrap()).collect(),
            "--ranks" => ranks = v().parse().unwrap(),
            "--iters" => iters = v().parse().unwrap(),
            "--tokens" => tokens = v().split(',').map(|s| s.parse().unwrap()).collect(),
            "--routing" => routings = v().split(',').map(Routing::parse).collect(),
            "--manifest" => manifest = Some(PathBuf::from(v())),
            "--y-out" => y_out = Some(PathBuf::from(v())),
            _ => panic!("unknown arg {a}"),
        }
    }
    gpus.truncate(ranks);
    if let Some(d) = &y_out {
        std::fs::create_dir_all(d).expect("--y-out dir");
    }
    assert_eq!(gpus.len(), ranks, "--gpus must name one GPU per rank");
    let manifest = manifest.unwrap_or_else(|| PathBuf::from(format!("examples/k3-moe-l1-ep{ranks}.json")));
    let json = std::fs::read_to_string(&manifest).unwrap_or_else(|e| panic!("{}: {e}", manifest.display()));
    let manifest = kern_manifest::Verified::from_json(&json).unwrap_or_else(|e| panic!("{}: {e}", manifest.display()));
    let kernels = std::env::temp_dir().join(format!("kern-k3-moe-bench-{}", std::process::id()));
    stage_cubins(&cubins, &kernels);
    let cases: Arc<Vec<Case>> = Arc::new(
        routings.iter().flat_map(|r| tokens.iter().map(move |&t| Case { tokens: t, routing: r.clone() })).collect(),
    );

    type Posted = Arc<Mutex<Vec<Option<BTreeMap<String, kern_runtime::PeerHandle>>>>>;
    let posted: Posted = Arc::new(Mutex::new(vec![None; ranks]));
    let gate = Arc::new(Barrier::new(ranks));
    let results: Arc<Mutex<Vec<Option<Result<Vec<f64>, String>>>>> = Arc::new(Mutex::new(vec![None; ranks]));
    let mut threads = Vec::new();
    for (rank, &gpu) in gpus.iter().enumerate() {
        let (manifest, kernels, posted, gate, results, cases, weights, y_out) = (
            manifest.clone(),
            kernels.clone(),
            posted.clone(),
            gate.clone(),
            results.clone(),
            cases.clone(),
            weights.clone(),
            y_out.clone(),
        );
        threads.push(std::thread::spawn(move || {
            let file =
                if ranks == 1 { "ep1.safetensors".to_string() } else { format!("ep{ranks}-r{rank}.safetensors") };
            let w = std::fs::read(weights.join(&file)).unwrap_or_else(|e| panic!("{file}: {e}"));
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
            let r = run_rank(
                &manifest,
                &kernels,
                gpu,
                rank,
                ranks,
                &w,
                &cases,
                iters,
                y_out.as_deref(),
                &rendezvous,
                &sync,
            );
            results.lock().unwrap()[rank] = Some(r.map_err(|e| e.to_string()));
        }));
    }
    for th in threads {
        th.join().unwrap();
    }
    let results = results.lock().unwrap();
    let times: Vec<&Vec<f64>> = results
        .iter()
        .enumerate()
        .map(|(r, x)| match x {
            Some(Ok(v)) => v,
            Some(Err(e)) => panic!("rank {r}: {e}"),
            None => panic!("rank {r}: no result"),
        })
        .collect();

    println!(
        "ranks {ranks}, {} iters; per rank: pairs = (token, expert) routed to it, experts touched of {}, \
         weight MB read if each touched expert is read once; TFLOP/s and TB/s use the busiest rank and the max time",
        iters,
        EXPERTS / ranks
    );
    println!(
        "{:<18} {:>6} {:>10} {:>10} {:>9} {:>8} {:>9} {:>8} {:>7}",
        "routing", "tok/r", "pairs max", "pairs mean", "experts", "MB", "us", "TFLOP/s", "TB/s"
    );
    for (i, c) in cases.iter().enumerate() {
        let idx = c.routing.draw(c.tokens * ranks, ranks, 0x9e37_79b9 + i as u64);
        let l = load(&idx, ranks);
        let pairs_max = l.iter().map(|x| x.0).max().unwrap();
        let pairs_mean = l.iter().map(|x| x.0).sum::<usize>() as f64 / ranks as f64;
        let experts_max = l.iter().map(|x| x.1).max().unwrap();
        let us = times.iter().map(|t| t[i]).fold(0.0, f64::max);
        let mb = experts_max as f64 * BYTES_PER_EXPERT / 1e6;
        println!(
            "{:<18} {:>6} {:>10} {:>10.0} {:>9} {:>8.0} {:>9.1} {:>8.0} {:>7.2}",
            c.routing.name(),
            c.tokens,
            pairs_max,
            pairs_mean,
            experts_max,
            mb,
            us,
            pairs_max as f64 * FLOP_PER_PAIR / (us * 1e-6) / 1e12,
            mb * 1e6 / (us * 1e-6) / 1e12
        );
    }
}
