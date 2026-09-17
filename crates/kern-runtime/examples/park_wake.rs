//! Park a checkpoint into the host tier and wake it, on a real manifest's
//! shapes: every byte comes back, and how long each way takes.
//!
//!   cargo run --release -p kern-runtime --example park_wake -- \
//!       --manifest examples/qwen3.8-27b.json --kernels kernels-qwen38 [--gpu 0] [--tokens 98000] [--wake 98000] [--every-page] [--host-gib 16]
//!
//! Leases `tokens`, fills its pages and its slot with a pattern keyed by
//! position, checkpoints it, parks the checkpoint (timed to the transfer
//! stream's completion), zeroes the states once its pages are back in the
//! pool, wakes it (timed to the checkpoint being handed out), leases a
//! sequence from it and checks the pattern at the new pages. No weights
//! are loaded: the states are all this touches.

use std::path::PathBuf;
use std::time::Instant;

use kern_runtime::Runtime;

/// The byte at `i` of a region whose pattern base is `base`.
fn pattern(base: u64, len: usize) -> Vec<u8> {
    (0..len as u64).map(|i| ((base + i) ^ ((base + i) >> 12) ^ ((base + i) >> 24)) as u8).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut manifest = PathBuf::from("examples/qwen3.8-27b.json");
    let mut kernels = PathBuf::from("kernels-qwen38");
    let mut gpu = 0usize;
    let mut tokens = 98_000usize;
    let mut host_gib = 16u64;
    let mut wake_at: Option<usize> = None;
    let mut every_page = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut v = || args.next().expect("value");
        match a.as_str() {
            "--manifest" => manifest = PathBuf::from(v()),
            "--kernels" => kernels = PathBuf::from(v()),
            "--gpu" => gpu = v().parse()?,
            "--tokens" => tokens = v().parse()?,
            "--host-gib" => host_gib = v().parse()?,
            "--wake" => wake_at = Some(v().parse()?),
            "--every-page" => every_page = true,
            _ => return Err(format!("unknown arg {a}").into()),
        }
    }
    let json = std::fs::read_to_string(&manifest)?;
    let m = kern_manifest::Verified::from_json(&json)?;
    let unit = kern_pool::page_unit(&m) as usize;
    let pages = tokens.div_ceil(unit);
    let capacity = ((pages + 1) * unit) as u64;
    let mut rt =
        Runtime::load(&m, &kernels, gpu, Some(kern_runtime::Capacity { tokens: Some(capacity), seqs: 4 }), None)?;
    let t = Instant::now();
    rt.reserve_host(host_gib << 30)?;
    println!("host tier {host_gib} GiB reserved in {:.2} s", t.elapsed().as_secs_f64());

    // Every pooled state, its bytes per page (0: per-seq) and per slot.
    let states: Vec<(String, u64, u64)> =
        m.states.iter().map(|(n, s)| (n.clone(), s.bytes_per_token * unit as u64, s.bytes_per_seq)).collect();
    let mut lease = rt.lease(tokens + 1)?;
    let mut bytes = 0u64;
    for (si, (name, page_bytes, slot_bytes)) in states.iter().enumerate() {
        let base = (si as u64) << 48;
        if *page_bytes > 0 {
            for k in 0..pages {
                let off = lease.slot(k * unit) as usize * (page_bytes / unit as u64) as usize;
                rt.write_state_at(name, off, &pattern(base + k as u64 * page_bytes, *page_bytes as usize))?;
                bytes += page_bytes;
            }
        }
        if let (true, Some(slot)) = (*slot_bytes > 0, lease.seq_slot()) {
            let off = slot as usize * *slot_bytes as usize;
            rt.write_state_at(name, off, &pattern(base + (1 << 40), *slot_bytes as usize))?;
            bytes += slot_bytes;
        }
    }
    let gib = bytes as f64 / (1u64 << 30) as f64;
    println!("{tokens} tokens: {pages} pages of {unit}, {gib:.2} GiB of state written");
    // A serving scheduler checkpoints a stateless sequence page by page,
    // so the chain it parks is one node per page.
    if every_page {
        for k in 1..pages {
            drop(rt.checkpoint(&mut lease, k * unit)?);
        }
    }
    let cp = rt.checkpoint(&mut lease, tokens)?;
    drop(lease);
    rt.synchronize()?;

    let t = Instant::now();
    let Ok(room) = rt.room(cp)? else { return Err("the host tier is too small for this checkpoint".into()) };
    let parked = rt.park(room)?;
    let issued = t.elapsed();
    rt.synchronize()?;
    let park = t.elapsed();
    println!(
        "park: {:.1} ms ({:.0} GiB/s), {:.1} ms to issue; host tier {:.2} GiB used",
        park.as_secs_f64() * 1e3,
        gib / park.as_secs_f64(),
        issued.as_secs_f64() * 1e3,
        rt.host_tier().map_or(0.0, |(u, _)| u as f64 / (1u64 << 30) as f64)
    );
    rt.zero_states()?;

    let t = Instant::now();
    // A prompt hitting part of a parked checkpoint wakes that part alone.
    let wake_at = wake_at.unwrap_or(tokens);
    let wake_pages = wake_at.div_ceil(unit);
    let mut waking = rt.wake(&parked, wake_at, tokens + 1)?;
    let issued = t.elapsed();
    let row = loop {
        match rt.awake(waking)? {
            Ok(l) => break l,
            Err(w) => waking = w,
        }
    };
    let wake = t.elapsed();
    println!(
        "wake: {:.1} ms ({:.0} GiB/s), {:.1} ms to issue; prefix {}",
        wake.as_secs_f64() * 1e3,
        gib / wake.as_secs_f64(),
        issued.as_secs_f64() * 1e3,
        row.prefix()
    );

    let mut bad = 0usize;
    for (si, (name, page_bytes, slot_bytes)) in states.iter().enumerate() {
        let base = (si as u64) << 48;
        if *page_bytes > 0 {
            for (k, &page) in row.page_ids()[..wake_pages].iter().enumerate() {
                let off = page as usize * *page_bytes as usize;
                let got = rt.read_state_at(name, off, *page_bytes as usize)?;
                if got != pattern(base + k as u64 * page_bytes, *page_bytes as usize) {
                    bad += 1;
                    if bad <= 3 {
                        println!("  state `{name}` page {k} differs");
                    }
                }
            }
        }
        if let (true, Some(slot), true) = (*slot_bytes > 0, row.seq_slot(), wake_at == tokens) {
            let got = rt.read_state_at(name, slot as usize * *slot_bytes as usize, *slot_bytes as usize)?;
            if got != pattern(base + (1 << 40), *slot_bytes as usize) {
                bad += 1;
                println!("  state `{name}` slot differs");
            }
        }
    }
    println!("{}", if bad == 0 { "PASS: every byte back" } else { "FAIL" });
    if bad > 0 {
        std::process::exit(1);
    }
    Ok(())
}
