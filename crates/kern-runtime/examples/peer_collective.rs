//! E5 gate, first stone: the tray-local collectives on N runtimes of one
//! tray — the allreduce of tools/kernels-src/peer_allreduce.cu (TensorRT-LLM's
//! protocol, one-shot and two-shot each forced so both are measured at every
//! size) and the all-gather of peer_collective.cu. Every rank fills its input
//! with a rank-and-row pattern, runs each once eagerly and checks every
//! element against the host in the "own rows first" layout, then times
//! captured bursts at the sizes a K3 decode step exchanges (f32 [R*B, 7168]
//! partials, bf16 [B, 3584] and f32 [B, 7168] rows). `--ar-cubin` must be
//! built for the number of GPUs given.
//!
//!   cargo run --release -p kern-runtime --example peer_collective -- \
//!       --cubin target/cubins/peer_collective.cubin --ar-cubin target/cubins/peer_allreduce.cubin \
//!       [--gpus 0,1,2,3] [--iters 50] [--burst 32]
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Barrier, Mutex};

use kern_runtime::{PeerHandle, Runtime, Topology};

const H: usize = 7168;
const B_MAX: usize = 64;
const RB_MAX: usize = 4 * H;
const TIMEOUT_NS: i64 = 2_000_000_000;

/// Rank `r`'s allreduce input at tray row `t`, column `i`: small integers, so
/// the f32 sum over ranks is exact in any order.
fn ar_value(r: usize, t: usize, i: usize) -> f32 {
    (((r + 1) * (t * 7 + i)) % 13) as f32
}

fn ag_byte(r: usize, j: usize, k: usize) -> u8 {
    (r * 31 + j * 7 + k) as u8
}

fn main() {
    let mut cubin = PathBuf::from("target/cubins/peer_collective.cubin");
    let mut ar_cubin = PathBuf::from("target/cubins/peer_allreduce.cubin");
    let mut gpus: Vec<usize> = vec![0, 1, 2, 3];
    let mut iters = 50usize;
    let mut burst = 32usize;
    let mut verify = true;
    let mut grid = 128usize;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut v = || args.next().expect("value");
        match a.as_str() {
            "--cubin" => cubin = PathBuf::from(v()),
            "--ar-cubin" => ar_cubin = PathBuf::from(v()),
            "--gpus" => gpus = v().split(',').map(|s| s.parse().unwrap()).collect(),
            "--iters" => iters = v().parse().unwrap(),
            "--burst" => burst = v().parse().unwrap(),
            "--no-verify" => verify = false,
            "--grid" => grid = v().parse().unwrap(),
            _ => panic!("unknown arg {a}"),
        }
    }
    let n = gpus.len();
    let bytes = std::fs::read(&cubin).expect("cubin");
    let sha = hex::encode(<sha2::Sha256 as sha2::Digest>::digest(&bytes));
    let kernels = std::env::temp_dir().join(format!("kern-peer-collective-{}", std::process::id()));
    std::fs::create_dir_all(&kernels).unwrap();
    std::fs::write(kernels.join(format!("peer_collective-{}.cubin", &sha[..12])), &bytes).unwrap();
    let ar_bytes = std::fs::read(&ar_cubin).expect("allreduce cubin");
    let ar_sha = hex::encode(<sha2::Sha256 as sha2::Digest>::digest(&ar_bytes));
    std::fs::write(kernels.join(format!("peer_allreduce-{}.cubin", &ar_sha[..12])), &ar_bytes).unwrap();

    let ag_region = B_MAX * RB_MAX / 8;
    let ag = serde_json::json!({ "op": "allgather", "args": [
        { "buf": "ag_x" }, { "buf": "ag_y" }, { "buf": "ag_sym" }, { "buf": "ag_peers" }, { "buf": "ag_epochs" },
        { "buf": "err" }, { "rank": "tp" }, { "i32": n }, { "var": "tokens" }, { "var": "rb" }, { "i32": ag_region },
        { "i64": TIMEOUT_NS }
    ]});
    // The TensorRT-LLM protocol (peer_allreduce.cu): one-shot Lamport (mode 1)
    // and two-shot (mode 2) over the same peer buffers, forced per program so
    // both are measured at every size.
    let tr_stage = (n * n * B_MAX * H * 4) as u64;
    let tr = |mode: i32| {
        serde_json::json!({ "op": "allreduce", "args": [
            { "buf": "ar_x" }, { "buf": "ar_y" }, { "buf": "tr_comm" }, { "buf": "tr_comm_peers" },
            { "buf": "tr_flags" }, { "buf": "tr_flag_peers" }, { "buf": "tr_lamport" }, { "buf": "tr_lamport_peers" },
            { "buf": "tr_state" }, { "buf": "err" }, { "rank": "tp" }, { "var": "tokens" }, { "i32": H },
            { "i64": tr_stage }, { "i32": mode }, { "i64": TIMEOUT_NS }
        ]})
    };
    let tr_init =
        serde_json::json!({ "op": "lamport_init", "args": [{ "buf": "tr_lamport" }, { "i64": 3 * tr_stage }] });
    let ag_params = [
        "in buffer<u8>",
        "out buffer<u8>",
        "inout buffer<u8>",
        "in buffer<u64>",
        "inout buffer<u32>",
        "out buffer<i32>",
        "i32",
        "i32",
        "i32",
        "i32",
        "i32",
        "i64",
    ];
    let manifest = serde_json::json!({
        "schema_version": 5,
        "model": "peer-collective",
        "vars": { "tokens": { "max": B_MAX }, "rows": { "max": n * B_MAX }, "rb": { "max": RB_MAX } },
        "topology": { "groups": { "tp": n } },
        "buffers": {
            "ar_x": { "dtype": "f32", "shape": ["rows", H], "kind": "input" },
            "ar_y": { "dtype": "f32", "shape": ["rows", H], "kind": "output" },
            "ag_x": { "dtype": "u8", "shape": ["tokens", "rb"], "kind": "input" },
            "ag_y": { "dtype": "u8", "shape": ["rows", "rb"], "kind": "output" },
            "ag_sym": { "dtype": "u8", "shape": [2 * n * ag_region * 16], "kind": "carry", "export": true },
            "ag_peers": { "dtype": "u64", "shape": [n], "kind": "peer", "of": "ag_sym", "group": "tp" },
            "ag_epochs": { "dtype": "u32", "shape": [grid], "kind": "carry" },
            "err": { "dtype": "i32", "shape": [1], "kind": "output" },
            "tr_comm": { "dtype": "u8", "shape": [2 * n * B_MAX * H * 4], "kind": "carry", "export": true },
            "tr_comm_peers": { "dtype": "u64", "shape": [n], "kind": "peer", "of": "tr_comm", "group": "tp" },
            "tr_flags": { "dtype": "i32", "shape": [n * 256], "kind": "carry", "export": true },
            "tr_flag_peers": { "dtype": "u64", "shape": [n], "kind": "peer", "of": "tr_flags", "group": "tp" },
            "tr_lamport": { "dtype": "u8", "shape": [3 * tr_stage], "kind": "carry", "export": true },
            "tr_lamport_peers": { "dtype": "u64", "shape": [n], "kind": "peer", "of": "tr_lamport", "group": "tp" },
            "tr_state": { "dtype": "i32", "shape": [8], "kind": "carry" }
        },
        "modules": {
            "peer_collective": { "source": "peer_collective.cubin", "sha256": sha },
            "peer_allreduce": { "source": "peer_allreduce.cubin", "sha256": ar_sha }
        },
        "ops": {
            "allgather": {
                "params": ag_params,
                "impl": { "launches": [{ "module": "peer_collective", "entry": "kern_peer_allgather",
                                         "block": [256, 1, 1], "grid": [grid, 1, 1] }] }
            },
            "allreduce": {
                "params": ["in buffer<f32>", "out buffer<f32>", "inout buffer<u8>", "in buffer<u64>", "inout buffer<i32>",
                           "in buffer<u64>", "inout buffer<u8>", "in buffer<u64>", "inout buffer<i32>", "out buffer<i32>",
                           "i32", "i32", "i32", "i64", "i32", "i64"],
                "impl": { "launches": [{ "module": "peer_allreduce", "entry": "kern_peer_allreduce_f32",
                                         "block": [H / 4 / 8, 1, 1], "grid": [152, 1, 1], "cluster": [8, 1, 1] }] }
            },
            "lamport_init": {
                "params": ["inout buffer<u8>", "i64"],
                "impl": { "launches": [{ "module": "peer_allreduce", "entry": "kern_peer_lamport_init",
                                         "block": [256, 1, 1], "grid": [256, 1, 1] }] }
            }
        },
        "programs": {
            "allgather": [ag.clone()],
            "ag_burst": std::iter::repeat_n(ag, burst).collect::<Vec<_>>(),
            "tr_init": [tr_init],
            "tr1": {"calls": [tr(1)]},
            "tr2": {"calls": [tr(2)]},
            "tr1_burst": {"calls": std::iter::repeat_n(tr(1), burst).collect::<Vec<_>>()},
            "tr2_burst": {"calls": std::iter::repeat_n(tr(2), burst).collect::<Vec<_>>()}
        }
    });
    let json = serde_json::to_string_pretty(&manifest).unwrap();
    let verified = kern_manifest::Verified::from_json(&json).unwrap_or_else(|e| panic!("{e}"));
    std::fs::write(kernels.join("peer-collective.json"), &json).unwrap();
    eprintln!("manifest + cubin in {}", kernels.display());

    // (rows per rank, allgather row bytes): the shapes a K3 step exchanges.
    let configs: Vec<(usize, usize)> = vec![
        (1, 2 * 3584),
        (4, 2 * 3584),
        (6, 2 * 3584),
        (8, 2 * 3584),
        (16, 2 * 3584),
        (16, 4 * H),
        (24, 4 * H),
        (32, 4 * H),
        (64, 4 * H),
    ];

    let posted: Arc<Mutex<Vec<Option<BTreeMap<String, PeerHandle>>>>> = Arc::new(Mutex::new(vec![None; n]));
    let gate = Arc::new(Barrier::new(n));
    let results: Arc<Mutex<Vec<Option<Result<Vec<String>, String>>>>> = Arc::new(Mutex::new(vec![None; n]));
    let mut threads = Vec::new();
    for (rank, &gpu) in gpus.iter().enumerate() {
        let (verified, kernels, posted, gate, results, configs) =
            (verified.clone(), kernels.clone(), posted.clone(), gate.clone(), results.clone(), configs.clone());
        threads.push(std::thread::spawn(move || {
            let run = || -> Result<Vec<String>, String> {
                let topo = Topology::one("tp", rank as u64, n as u64);
                let e = |e: kern_runtime::Error| e.to_string();
                let mut rt =
                    Runtime::load(&verified, &kernels, gpu, Some(kern_runtime::Capacity { tokens: Some(1), seqs: 1 }), Some(&topo)).map_err(e)?;
                let mine = rt.export_handles().map_err(e)?;
                posted.lock().unwrap()[rank] = Some(mine);
                gate.wait();
                let members: Vec<_> = posted.lock().unwrap().iter().map(|m| m.clone().unwrap()).collect();
                rt.import_peers("tp", &members).map_err(e)?;
                let env0 = BTreeMap::from([("tokens".to_string(), 1u64), ("rows".to_string(), n as u64), ("rb".to_string(), 16u64)]);
                rt.run("tr_init", &env0).map_err(e)?;
                gate.wait();
                let mut lines = Vec::new();
                for &(b, rb) in &configs {
                    let rows = n * b;
                    let vars = BTreeMap::from([
                        ("tokens".to_string(), b as u64),
                        ("rows".to_string(), rows as u64),
                        ("rb".to_string(), rb as u64),
                    ]);
                    // Local row j is tray row (rank*b + j) mod rows.
                    let tray = |j: usize| (rank * b + j) % rows;
                    let gx: Vec<u8> = (0..b).flat_map(|j| (0..rb).map(move |k| ag_byte(rank, j, k))).collect();
                    rt.write_input_at("ag_x", &gx, &vars).map_err(e)?;
                    gate.wait();
                    rt.run("allgather", &vars).map_err(e)?;
                    let err = i32::from_le_bytes(rt.read_output("err").map_err(e)?[..4].try_into().unwrap());
                    if err != 0 { return Err(format!("B={b}: a slot from rank {} never arrived", err - 1)); }
                    // `off` shifts every input so each program's sum differs from
                    // the last one's and a stale output cannot pass.
                    let check_ar = |rt: &Runtime, what: &str, off: f32| -> Result<(), String> {
                        let y = rt.read_output("ar_y").map_err(e)?;
                        let mut bad = 0usize;
                        for j in 0..if verify { rows } else { 0 } {
                            for i in 0..H {
                                let got = f32::from_le_bytes(y[(j * H + i) * 4..][..4].try_into().unwrap());
                                let want: f32 = (0..n).map(|r| ar_value(r, tray(j), i) + off).sum();
                                if got != want {
                                    if bad == 0 {
                                        eprintln!("rank {rank} B={b}: {what} row {j} col {i}: got {got}, want {want}");
                                    }
                                    bad += 1;
                                }
                            }
                        }
                        if bad != 0 { return Err(format!("B={b}: {bad} wrong {what} elements")); }
                        Ok(())
                    };
                    for (prog, what, off) in [("tr1", "one-shot", 1.0f32), ("tr2", "two-shot", 2.0f32)] {
                        let x: Vec<u8> = (0..rows)
                            .flat_map(|j| (0..H).flat_map(move |i| (ar_value(rank, tray(j), i) + off).to_le_bytes()))
                            .collect();
                        rt.write_input_at("ar_x", &x, &vars).map_err(e)?;
                        gate.wait();
                        rt.run(prog, &vars).map_err(e)?;
                        let err = i32::from_le_bytes(rt.read_output("err").map_err(e)?[..4].try_into().unwrap());
                        if err != 0 { return Err(format!("B={b}: {what}: rank {} never arrived", err - 1)); }
                        check_ar(&rt, what, off)?;
                    }
                    let gy = rt.read_output("ag_y").map_err(e)?;
                    let mut bad = 0usize;
                    for d in 0..if verify { n } else { 0 } {
                        let src = (rank + d) % n;
                        for j in 0..b {
                            for k in 0..rb {
                                if gy[(d * b + j) * rb + k] != ag_byte(src, j, k) {
                                    if bad == 0 {
                                        eprintln!("rank {rank} B={b} rb={rb}: allgather block {d} row {j} byte {k} wrong");
                                    }
                                    bad += 1;
                                }
                            }
                        }
                    }
                    if bad != 0 { return Err(format!("B={b} rb={rb}: {bad} wrong allgather bytes")); }
                    rt.capture("ag_burst", &vars).map_err(e)?;
                    gate.wait();
                    let ag_us = rt.time_captured("ag_burst", &vars, iters).map_err(e)? as f64 * 1e3 / burst as f64;
                    rt.capture("tr1_burst", &vars).map_err(e)?;
                    rt.capture("tr2_burst", &vars).map_err(e)?;
                    gate.wait();
                    let tr1_us = rt.time_captured("tr1_burst", &vars, iters).map_err(e)? as f64 * 1e3 / burst as f64;
                    gate.wait();
                    let tr2_us = rt.time_captured("tr2_burst", &vars, iters).map_err(e)? as f64 * 1e3 / burst as f64;
                    let err = i32::from_le_bytes(rt.read_output("err").map_err(e)?[..4].try_into().unwrap());
                    if err != 0 { return Err(format!("B={b}: burst: a slot from rank {} never arrived", err - 1)); }
                    lines.push(format!(
                        "B={b:>2} rows={rows:>3}: allreduce f32 [{rows},{H}] ({:.2} MB) one-shot {tr1_us:.2} us, two-shot {tr2_us:.2} us; allgather [{b},{rb}] -> [{rows},{rb}] ({:.2} MB) {ag_us:.2} us",
                        (rows * H * 4) as f64 / 1e6,
                        (rows * rb) as f64 / 1e6
                    ));
                }
                Ok(lines)
            };
            let r = run();
            results.lock().unwrap()[rank] = Some(r);
        }));
    }
    for t in threads {
        t.join().unwrap();
    }
    let mut ok = true;
    for (rank, r) in results.lock().unwrap().iter().enumerate() {
        match r {
            Some(Ok(lines)) => {
                println!("rank {rank} gpu {}: all elements verified", gpus[rank]);
                if rank == 0 {
                    for l in lines {
                        println!("  {l}");
                    }
                }
            }
            Some(Err(e)) => {
                ok = false;
                println!("rank {rank}: {e}");
            }
            None => {
                ok = false;
                println!("rank {rank}: no result");
            }
        }
    }
    println!("{}", if ok { "PASS" } else { "FAIL" });
    std::process::exit(if ok { 0 } else { 1 });
}
