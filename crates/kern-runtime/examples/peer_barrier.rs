//! E0/K0 gate: N runtimes on N GPUs of one tray, one SPMD manifest, a
//! cross-rank barrier over exported flag words. Each rank exports its
//! handles, imports the others', then replays a burst of barriers as a
//! captured graph; the per-barrier time is the fabric round trip plus the
//! spin. A timeout is reported, never hung on.
//!
//!   cargo run --release -p kern-runtime --example peer_barrier -- \
//!       --cubin target/cubins/peer_barrier.cubin [--gpus 0,1,2,3] [--iters 200] [--burst 64] [--drop <rank>]
//!
//! `--drop r` makes rank r skip the barrier: the others must report a
//! timeout naming it, within the 2 s budget, and nothing may hang.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Barrier, Mutex};
use std::time::Instant;

use kern_runtime::{PeerHandle, Runtime, Topology};

fn main() {
    let mut cubin = PathBuf::from("target/cubins/peer_barrier.cubin");
    let mut gpus: Vec<usize> = vec![0, 1, 2, 3];
    let mut iters = 200usize;
    let mut burst = 64usize;
    let mut drop: Option<usize> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut v = || args.next().expect("value");
        match a.as_str() {
            "--cubin" => cubin = PathBuf::from(v()),
            "--gpus" => gpus = v().split(',').map(|s| s.parse().unwrap()).collect(),
            "--iters" => iters = v().parse().unwrap(),
            "--burst" => burst = v().parse().unwrap(),
            "--drop" => drop = Some(v().parse().unwrap()),
            _ => panic!("unknown arg {a}"),
        }
    }
    let n = gpus.len();
    let bytes = std::fs::read(&cubin).expect("cubin");
    let sha = hex::encode(<sha2::Sha256 as sha2::Digest>::digest(&bytes));
    let kernels = std::env::temp_dir().join(format!("kern-peer-barrier-{}", std::process::id()));
    std::fs::create_dir_all(&kernels).unwrap();
    std::fs::write(kernels.join(format!("peer_barrier-{}.cubin", &sha[..12])), &bytes).unwrap();

    let call = serde_json::json!({
        "op": "barrier",
        "args": [
            { "buf": "flags" }, { "buf": "flags_peers" }, { "buf": "epoch" }, { "buf": "err" },
            { "rank": "ep" }, { "i32": n }, { "i64": 2_000_000_000i64 }
        ]
    });
    let manifest = serde_json::json!({
        "schema_version": 5,
        "model": "peer-barrier",
        "topology": { "groups": { "ep": n } },
        "buffers": {
            "flags": { "dtype": "u32", "shape": [n], "kind": "carry", "export": true },
            "flags_peers": { "dtype": "u64", "shape": [n], "kind": "peer", "of": "flags", "group": "ep" },
            "epoch": { "dtype": "u32", "shape": [1], "kind": "carry" },
            "err": { "dtype": "i32", "shape": [1], "kind": "output" }
        },
        "modules": { "peer_barrier": { "source": "peer_barrier.cubin", "sha256": sha } },
        "ops": {
            "barrier": {
                "params": ["inout buffer<u32>", "in buffer<u64>", "inout buffer<u32>", "out buffer<i32>", "i32", "i32", "i64"],
                "impl": { "launches": [
                    { "module": "peer_barrier", "entry": "kern_peer_barrier", "block": [32, 1, 1], "grid": [1, 1, 1] }
                ] }
            }
        },
        "programs": {
            "barrier": {"calls": [call.clone()]},
            "burst": {"calls": std::iter::repeat_n(call, burst).collect::<Vec<_>>()}
        }
    });
    let json = serde_json::to_string_pretty(&manifest).unwrap();
    let verified = kern_manifest::Verified::from_json(&json).unwrap_or_else(|e| panic!("{e}"));
    std::fs::write(kernels.join("peer-barrier.json"), &json).unwrap();
    eprintln!("manifest + cubin in {}", kernels.display());

    // Rendezvous: every rank posts its handles, then everyone imports.
    let posted: Arc<Mutex<Vec<Option<BTreeMap<String, PeerHandle>>>>> = Arc::new(Mutex::new(vec![None; n]));
    let gate = Arc::new(Barrier::new(n));
    let results = Arc::new(Mutex::new(vec![None; n]));
    let mut threads = Vec::new();
    for (rank, &gpu) in gpus.iter().enumerate() {
        let (verified, kernels, posted, gate, results) =
            (verified.clone(), kernels.clone(), posted.clone(), gate.clone(), results.clone());
        threads.push(std::thread::spawn(move || {
            let run = || -> kern_runtime::Result<(f64, f64, i32)> {
                let topo = Topology::one("ep", rank as u64, n as u64);
                let mut rt = Runtime::load(
                    &verified,
                    &kernels,
                    gpu,
                    Some(kern_runtime::Capacity { tokens: Some(1), seqs: 1 }),
                    Some(&topo),
                )?;
                let mine = rt.export_handles()?;
                posted.lock().unwrap()[rank] = Some(mine);
                gate.wait();
                let members: Vec<_> = posted.lock().unwrap().iter().map(|m| m.clone().unwrap()).collect();
                rt.import_peers("ep", &members)?;
                let vars = BTreeMap::new();
                // Everyone must be past import before the first barrier.
                gate.wait();
                let mut err = 0;
                if drop != Some(rank) {
                    rt.run("barrier", &vars)?;
                    err = i32::from_le_bytes(rt.read_output("err")?[..4].try_into().unwrap());
                }
                gate.wait();
                if err != 0 || drop.is_some() {
                    return Ok((f64::NAN, f64::NAN, err));
                }
                // Eager: wall time per run() including launch and sync.
                gate.wait();
                let t0 = Instant::now();
                for _ in 0..iters {
                    rt.run("barrier", &vars)?;
                }
                let eager_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
                // Captured burst: GPU-side time per barrier.
                rt.capture("burst", &vars)?;
                gate.wait();
                let ms = rt.time_captured("burst", &vars, iters)?;
                let err = i32::from_le_bytes(rt.read_output("err")?[..4].try_into().unwrap());
                Ok((eager_us, ms as f64 * 1e3 / burst as f64, err))
            };
            let r = run();
            results.lock().unwrap()[rank] = Some(r.map_err(|e| e.to_string()));
        }));
    }
    for t in threads {
        t.join().unwrap();
    }
    let mut ok = true;
    for (rank, r) in results.lock().unwrap().iter().enumerate() {
        match r {
            Some(Ok(_)) if drop == Some(rank) => println!("rank {rank}: dropped out, no barrier run"),
            Some(Ok((eager, graph, 0))) => println!(
                "rank {rank} gpu {}: eager {eager:.2} us/barrier (run+sync), captured burst {graph:.2} us/barrier",
                gpus[rank]
            ),
            Some(Ok((_, _, err))) => {
                let expected = drop.is_some_and(|d| (d == rank && *err == 0) || (d != rank && *err == d as i32 + 1));
                ok &= expected;
                println!(
                    "rank {rank}: barrier timed out waiting for rank {}{}",
                    err - 1,
                    if expected { " (as expected)" } else { "" }
                );
            }
            Some(Err(e)) => {
                ok = false;
                println!("rank {rank}: {e}")
            }
            None => {
                ok = false;
                println!("rank {rank}: no result")
            }
        }
    }
    std::process::exit(if ok { 0 } else { 1 });
}
