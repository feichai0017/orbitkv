//! Replaying B against a [`Recording`] of A: the workload with every run
//! started from A's state image and every span from A's inputs (so what
//! B writes is the span's own doing), B's free run for the end-to-end
//! logits, B's timings, and the verdict over all of it.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use anyhow::Result;
use serde_json::{json, Value};

use crate::compare::{BufCmp, LogitRow, TOP};
use crate::diff::{access, live_bytes};
use crate::harness::{at_rank, domain_violations, free_run, image, logit_rows, replay_span, Recording, StateCmp};
use crate::report::*;
use crate::{At, Options, Side, Vars};
use kern_manifest::types::DType;

/// Replay B against `rec` and judge it. `out` receives each section's
/// lines as it finishes.
pub fn replay<S: Side>(
    o: &Options,
    rec: Recording<S::Buf>,
    b: &mut S,
    out: &mut dyn FnMut(&[String]),
) -> Result<Report> {
    let t_start = Instant::now();
    let elapsed = |t: &Instant| t.elapsed().as_secs_f32();
    let ranks = rec.ranks;
    anyhow::ensure!(b.ranks() == ranks, "A recorded {ranks} ranks; B runs as {}", b.ranks());
    let (ma, mb) = (&rec.ma, &rec.mb);
    let mut sum = Summary { a: o.a.clone(), b: o.b.clone(), diff: rec.diff.clone(), ..Default::default() };
    let undriven = rec.undriven();
    let shared = rec.shared_states();

    // ---- 2. the workload on B, run by run from A's image, span by span from A's inputs
    let t_tap = Instant::now();
    // program -> buffer -> [(span label, cmp)]
    let mut local_res: BTreeMap<String, BTreeMap<String, Vec<(String, BufCmp)>>> = BTreeMap::new();
    let mut local_states: BTreeMap<String, BTreeMap<String, Vec<(String, StateCmp)>>> = BTreeMap::new();
    b.reset();
    for run in &rec.runs {
        image(b, run)?;
        b.stage(&run.tokens)?;
        let (p, e) = (run.program.as_str(), &run.vars);
        let mut ib = 0;
        for sr in &run.spans {
            b.run(p, e, ib..sr.span.b.start)?;
            ib = sr.span.b.end;
            let rp = replay_span(b, ma, sr, run, true)?;
            for q in 0..ranks {
                let label = at_rank(ranks, q, &format!("{} {}", run.label, sr.span.label()));
                for (n, c) in &rp.bufs[q] {
                    local_res
                        .entry(p.into())
                        .or_default()
                        .entry(n.clone())
                        .or_default()
                        .push((label.clone(), c.clone()));
                }
                for (st, c) in &rp.states[q] {
                    local_states
                        .entry(p.into())
                        .or_default()
                        .entry(st.clone())
                        .or_default()
                        .push((label.clone(), c.clone()));
                }
            }
        }
        b.run(p, e, ib..b.calls(p)?)?;
        b.advance(run.advance);
    }
    // B free-runs the same workload from zero state, nothing injected: what
    // a caller would get.
    let t_free = Instant::now();
    let b_logits = free_run(b, ma, mb, &rec.runs)?;
    let free_t = elapsed(&t_free);
    let tap = Tap {
        seed: format!("{:#x}", o.seed),
        prefill: rec.wl.prefill.len(),
        how: rec.wl.how.to_string(),
        chunk: rec.wl.chunk as u64,
        decode: rec.wl.decode.len(),
        vocab: rec.wl.vocab as usize,
        ranks,
        runs: rec.runs.len(),
        spans: rec.kept.len(),
        snapshot_bytes: rec.snapshot_bytes().0,
        snapshot_pieces: rec.snapshot_bytes().1,
        state_pre_image_bytes: rec.pre_image_bytes(),
        load_s: rec.load_s,
        record_s: rec.record_s,
        free_run_ms: free_t * 1e3,
        elapsed_s: elapsed(&t_tap),
    };
    out(&tap.lines());
    sum.tap = Some(tap);

    // ---- 2a. the comparisons, counted; the differing ones named
    let mut all_local: Vec<Finding> = Vec::new();
    let (mut compared, mut n_bit, mut n_val) = (0usize, 0usize, 0usize);
    for (p, bufs) in &local_res {
        for (bn, res) in bufs {
            for (l, c) in res {
                compared += 1;
                if c.identical() {
                    n_bit += 1;
                    continue;
                }
                n_val += c.value_identical() as usize;
                all_local.push(Finding::of_buf(p, l, bn, c));
            }
        }
    }
    for (p, sts) in &local_states {
        for (st, res) in sts {
            for (l, c) in res {
                compared += 1;
                if c.n_diff == 0 && c.outside == 0 {
                    n_bit += 1;
                    continue;
                }
                let mut t = format!("{}/{} bytes of the write-set differ", c.n_diff, c.set);
                if c.outside > 0 {
                    t += &format!(" · wrote {} outside A's write-set", kb(c.outside));
                }
                all_local.push(Finding::text(p, l, &format!("state {st}"), t));
            }
        }
    }
    // Nothing compared is not everything identical.
    let local_bit = compared > 0 && n_bit == compared;
    let local_identical = compared > 0 && n_bit + n_val == compared;
    let mut outputs = Vec::new();
    let mut e2e_differs: Vec<String> = Vec::new();
    for (name, per_rank) in &rec.outputs {
        for (q, a_bytes) in per_rank.iter().enumerate() {
            let c = crate::compare::compare(ma.buffers[name].dtype, a_bytes, &b.read(q, name, a_bytes.len())?);
            if !c.value_identical() {
                e2e_differs.push(at_rank(ranks, q, name));
            }
            outputs.push(Finding::of("", "end-to-end", &at_rank(ranks, q, name), &c));
        }
    }
    let e2e_outputs_identical = e2e_differs.is_empty();
    // Post-condition on what a caller would get: every output lies in
    // its declared domain.
    let names: Vec<String> = rec.outputs.keys().cloned().collect();
    let vars = rec.runs.last().map(|r| r.vars.clone()).unwrap_or_default();
    let violations = domain_violations(b, mb, &vars, "B", &names)?;
    let mut e2e_states = Vec::new();
    for (q, img) in rec.states.iter().enumerate() {
        for name in &shared {
            let len = b.state_bytes(name)?;
            let d = b.compare(q, DType::U8, At::Scratch(&img[name], 0..len), At::State(name, 0..len))?.n_diff;
            // KERN_TEST_DUMP=<dir>: both sides' final image of every state, for
            // locating a whole-state difference the spans do not explain.
            if let Ok(dir) = std::env::var("KERN_TEST_DUMP") {
                std::fs::write(format!("{dir}/{name}-r{q}-a.bin"), b.bytes(q, &img[name], 0..len)?)?;
                std::fs::write(format!("{dir}/{name}-r{q}-b.bin"), b.read_state(q, name, 0..len)?)?;
            }
            e2e_states.push(StateE2e { name: at_rank(ranks, q, name), bytes: len, differ: d });
        }
    }
    let (findings, omitted) = cap(all_local.clone(), Finding::severity);
    let local = Local {
        compared,
        bit_identical: n_bit,
        value_identical: n_val,
        findings,
        omitted,
        outputs,
        violations,
        states: e2e_states,
        one_sided: rec.one_sided.iter().cloned().collect(),
        undriven: undriven.clone(),
    };
    let domain_ok = local.violations.is_empty();
    out(&local.lines());
    sum.local = Some(local);

    // ---- 2b. logits: the end-to-end oracle
    let t_log = Instant::now();
    let logit_rows = logit_rows(b, ma, ranks, &rec.logits, &b_logits)?;
    let have_logits = !logit_rows.is_empty();
    let logits_bit = have_logits && logit_rows.iter().all(|r| r.stats.cmp.identical());
    let kl_max = logit_rows.iter().map(|r| r.stats.kl).fold(0.0, f64::max);
    let n_flips = logit_rows.iter().filter(|r| r.stats.flip()).count();
    let n_within = logit_rows.iter().filter(|r| r.stats.flip() && r.stats.kl <= o.logit_kl).count();
    let wide_flip = logit_rows
        .iter()
        .filter(|r| r.stats.flip() && r.stats.kl > o.logit_kl)
        .max_by(|x, y| x.stats.kl.total_cmp(&y.stats.kl));
    let logits_within = have_logits && kl_max <= o.logit_kl;
    let worst_kl = logit_rows.iter().max_by(|x, y| x.stats.kl.total_cmp(&y.stats.kl));
    let logits_at = worst_kl.map_or(String::new(), |w| w.label.clone());
    let worst_top = logit_rows.iter().min_by_key(|r| r.stats.top);
    let top_full = |r: &&LogitRow| r.stats.top == TOP.min(r.stats.cmp.n);
    let logits = Logits {
        rows: logit_rows.len(),
        differ: logit_rows.iter().filter(|r| !r.stats.cmp.identical()).count(),
        flips: n_flips,
        within: n_within,
        kl_max,
        kl_at: logits_at.clone(),
        limit_kl: o.logit_kl,
        top_min: worst_top.map_or(0, |w| w.stats.top),
        top_at: worst_top.map_or(String::new(), |w| w.label.clone()),
        top_differ: logit_rows.iter().filter(|r| !top_full(r)).count(),
        flipped: logit_rows
            .iter()
            .filter(|r| r.stats.flip())
            .map(|r| Flip {
                row: r.label.clone(),
                argmax_a: r.stats.argmax_a,
                argmax_b: r.stats.argmax_b,
                margin_a: r.stats.margin_a,
                kl: r.stats.kl,
                rank_in_b: r.stats.rank_in_b,
                within: r.stats.kl <= o.logit_kl,
            })
            .collect(),
        elapsed_s: elapsed(&t_log),
    };
    out(&logits.lines());
    sum.logits = Some(logits);
    let logit_detail: Vec<Value> = logit_rows
        .iter()
        .filter(|r| !r.stats.cmp.identical())
        .map(|r| json!({"row": r.label, "cmp": r.stats.cmp, "argmax_a": r.stats.argmax_a, "argmax_b": r.stats.argmax_b, "margin_a": r.stats.margin_a, "kl": r.stats.kl, "top": r.stats.top, "rank_in_b": r.stats.rank_in_b}))
        .collect();

    // ---- 3. noise floor: measured on A while it was loaded
    let (noise_clean, noisy_states, noise_res, all_noise) = match &rec.noise {
        Some(n) => {
            out(&n.report.lines());
            sum.noise = Some(n.report.clone());
            (n.report.compared == n.report.clean, n.noisy_states.clone(), n.res.clone(), n.findings.clone())
        }
        None => (true, BTreeSet::new(), BTreeMap::new(), Vec::new()),
    };
    let floor = rec.noise.as_ref().and_then(|n| n.report.floor.clone());
    // A's spans may be noisy while its end-to-end distribution reproduces;
    // only the latter says whether a flip beyond the limit is B's doing.
    let a_reproduces = floor.as_ref().is_none_or(|f| f.kl_max <= o.logit_kl && f.flips == 0);

    // ---- 4. perf: B's side of every timing, then attribution
    if let Some(pf) = &rec.perf {
        let t_p = Instant::now();
        // (program, kernel) -> per side (bytes, ms, count)
        let mut per_kernel: BTreeMap<(String, String), [(usize, f32, usize); 2]> = BTreeMap::new();
        let mut state_traffic = false;
        // Σ over a side's spans of its per-call times; feeds the roofline
        // accumulator when `roof`.
        let mut attribute = |side: usize, pname: &str, times: &[f32], e: &Vars, roof: bool| -> (f32, f32) {
            let m = if side == 0 { ma } else { mb };
            let mut span_ms = 0f32;
            for sg in rec.spans_of(pname) {
                let r = if side == 0 { sg.a } else { sg.b };
                for i in r {
                    span_ms += times[i];
                    if !roof {
                        continue;
                    }
                    let acc = access(m, pname, i..i + 1);
                    let bytes: usize = acc.reads.iter().map(|n| live_bytes(m, n, e)).sum::<usize>()
                        + acc.writes.iter().map(|n| live_bytes(m, n, e)).sum::<usize>();
                    state_traffic |= !(acc.state_reads.is_empty() && acc.state_writes.is_empty());
                    let ent = per_kernel.entry((pname.into(), m.programs[pname].calls[i].op.clone())).or_default();
                    ent[side].0 += bytes;
                    ent[side].1 += times[i];
                    ent[side].2 += 1;
                }
            }
            (times.iter().sum(), span_ms)
        };
        // derived = A's step with A's spans swapped for B's spans (both timed
        // eager); the gap to the measurement is launch-gap / L2 interaction.
        let derived = |a: (f32, f32), bb: (f32, f32)| a.0 - a.1 + bb.1;
        let mut steps = Vec::new();
        for st in &pf.steps {
            let p = st.program.as_str();
            let e = b.stage(&vec![st.token; st.rows as usize])?;
            let tb = b.time(p, &e, 0..b.calls(p)?, o.iters)?;
            let graph_ms = match st.graph_ms {
                Some(ga) => Some([ga, b.time_graph(p, &e, 100)?]),
                None => None,
            };
            let (a, bb) = (attribute(0, p, &st.times, &e, true), attribute(1, p, &tb, &e, true));
            steps.push(StepPerf {
                program: p.into(),
                rows: st.rows,
                spans: rec.spans_of(p).len(),
                eager_ms: [a.0, bb.0],
                derived_ms: derived(a, bb),
                graph_ms,
                span_ms: [a.1, bb.1],
            });
        }
        let chunk_name = rec.chunk.as_ref().map_or("", |f| f.name.as_str());
        let mut sweep = Vec::new();
        for sw in &pf.sweep {
            b.reset();
            let e = b.stage(&sw.tokens)?;
            let tb = b.time(chunk_name, &e, 0..b.calls(chunk_name)?, sw.iters)?;
            let (a, bb) =
                (attribute(0, chunk_name, &sw.times, &e, sw.tapped), attribute(1, chunk_name, &tb, &e, sw.tapped));
            if sw.tapped {
                steps.push(StepPerf {
                    program: chunk_name.into(),
                    rows: sw.rows,
                    spans: rec.spans_of(chunk_name).len(),
                    eager_ms: [a.0, bb.0],
                    derived_ms: derived(a, bb),
                    graph_ms: None,
                    span_ms: [a.1, bb.1],
                });
            }
            sweep.push(SweepPt {
                rows: sw.rows,
                eager_ms: [a.0, bb.0],
                derived_ms: derived(a, bb),
                span_ms: [a.1, bb.1],
            });
        }
        let roofline = per_kernel
            .iter()
            .map(|((program, op), sides)| {
                let per = sides.iter().find(|s| s.2 > 0).map_or(0, |s| s.0 / s.2);
                let side = |(bytes, ms, n): (usize, f32, usize)| -> (f32, f64, f64) {
                    if n == 0 {
                        return (f32::NAN, f64::NAN, f64::NAN);
                    }
                    let gbs = bytes as f64 / 1e9 / (ms as f64 / 1e3);
                    (ms / n as f32 * 1e3, gbs, gbs / o.peak_bw * 100.0)
                };
                let (a, bb) = (side(sides[0]), side(sides[1]));
                Roof {
                    program: program.clone(),
                    op: op.clone(),
                    calls: sides[0].2.max(sides[1].2),
                    bytes_per_call: per,
                    us_per_call: [a.0, bb.0],
                    gbs: [a.1, bb.1],
                    peak_pct: [a.2, bb.2],
                }
            })
            .collect();
        let perf = Perf {
            iters: o.iters,
            steps,
            sweep_program: chunk_name.into(),
            sweep,
            roofline,
            peak_bw_gbs: o.peak_bw,
            state_traffic,
            elapsed_s: elapsed(&t_p),
        };
        out(&perf.lines());
        sum.perf = Some(perf);
    }

    // Is every local difference inside A's own noise band?
    let states_within = local_states.iter().all(|(p, sts)| {
        sts.iter().all(|(st, res)| {
            res.iter().all(|(_, c)| c.n_diff == 0 && c.outside == 0) || noisy_states.contains(&(p.clone(), st.clone()))
        })
    });
    let within_noise = !local_identical
        && !noise_clean
        && states_within
        && local_res.iter().all(|(p, bufs)| {
            bufs.iter().all(|(bn, res)| {
                let worst_local = res
                    .iter()
                    .filter_map(|(_, c)| if c.value_identical() { None } else { c.cmp.max_ulp.or(Some(u64::MAX)) })
                    .max();
                let worst_noise = noise_res.get(p).and_then(|nb| nb.get(bn)).and_then(|nr| {
                    nr.iter()
                        .filter_map(|(_, c)| if c.value_identical() { None } else { c.max_ulp.or(Some(u64::MAX)) })
                        .max()
                });
                match (worst_local, worst_noise) {
                    (None, _) => true,
                    (Some(l), Some(n)) => l <= n,
                    (Some(_), None) => false,
                }
            })
        });

    // ---- verdict
    let n_rows = logit_rows.len();
    let (code, text) = if !domain_ok {
        (1, "B writes a value outside a declared domain end to end".to_string())
    } else if let (Some(f), true) = (wide_flip, a_reproduces) {
        (
            1,
            format!(
                "B changes the argmax end-to-end at {}: A {} → B {}, KL {:.2e} above the limit {:.0e} (A's margin {:.4}, A's token is B's #{})",
                f.label, f.stats.argmax_a, f.stats.argmax_b, f.stats.kl, o.logit_kl, f.stats.margin_a, f.stats.rank_in_b
            ),
        )
    } else if !undriven.is_empty() {
        (2, "a changed program was not tapped — the workload driver can't stage it".to_string())
    } else if local_bit {
        (0, "bit-identical at every span".to_string())
    } else if local_identical {
        (0, "value-identical at every span (only signed zeros differ)".to_string())
    } else if logits_bit {
        (0, format!("spans differ, but the end-to-end logits are bit-identical on all {n_rows} rows"))
    } else if logits_within {
        (
            0,
            format!(
                "logit evidence: end-to-end KL ≤ {kl_max:.2e} (limit {:.0e}) on {n_rows} rows, argmax agrees{}",
                o.logit_kl,
                if n_within > 0 {
                    format!(" except {n_within} flip{} within the limit", if n_within == 1 { "" } else { "s" })
                } else {
                    String::new()
                }
            ),
        )
    } else if within_noise {
        (0, "differences at every span lie within A's own noise floor".to_string())
    } else if have_logits {
        let band = match &floor {
            Some(f) if !a_reproduces => {
                format!(
                    " — A against itself reaches KL {:.2e} with {} flip{}",
                    f.kl_max,
                    f.flips,
                    if f.flips == 1 { "" } else { "s" }
                )
            }
            None if !noise_clean => " — A itself is not deterministic at some spans".to_string(),
            _ => String::new(),
        };
        (2, format!("spans differ; end-to-end KL up to {kl_max:.2e} at {} on {n_rows} rows (limit {:.0e}), {n_flips} argmax flip{} ({n_within} within the limit){band}", logits_at, o.logit_kl, if n_flips == 1 { "" } else { "s" }))
    } else {
        (2, "spans differ beyond bit/value identity and no driven program writes logits — no oracle".to_string())
    };
    sum.verdict = Verdict::new(code, text, elapsed(&t_start) + rec.record_s + rec.load_s);
    let detail = json!({
        "within_noise": within_noise, "end_to_end_outputs_differ": e2e_differs, "end_to_end_outputs_identical": e2e_outputs_identical,
        "local": all_local, "logit_rows": logit_detail, "noise": all_noise,
    });
    Ok(Report { summary: sum, detail })
}
