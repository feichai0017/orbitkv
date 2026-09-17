//! The report: one value per section, built when the section is done.
//! Text prints a section as one line per fact: the key first, ` · `
//! between facts, `a → b` for the two sides, the section's time last.
//! `--json` prints the sections as one object at the end instead.
//! Identical things are counted, differing things are named, worst first
//! and capped: a PASS is a dozen lines, and the last line starts with the
//! verdict. The archive is the same sections plus every differing
//! comparison.

use serde::Serialize;
use serde_json::Value;

use crate::compare::{BufCmp, Cmp, TOP};
use crate::diff::Diff;

/// Differing comparisons a section names before saying "more".
pub const SHOW: usize = 8;

pub fn row(key: &str, body: impl AsRef<str>, took: Option<f32>) -> String {
    let t = took.map_or(String::new(), |s| format!("   {}", secs(s)));
    format!("{key:<9} {}{t}", body.as_ref())
}

pub fn secs(s: f32) -> String {
    if s >= 1.0 {
        format!("{s:.1}s")
    } else {
        format!("{:.1}ms", s * 1e3)
    }
}

pub fn us(ms: f32) -> String {
    if ms >= 1.0 {
        format!("{ms:.3} ms")
    } else {
        format!("{:.1} µs", ms * 1e3)
    }
}

pub fn pct(a: f32, b: f32) -> String {
    let p = (b - a) / a.max(1e-9) * 100.0;
    format!("{}{:.1}%", if p < 0.0 { "−" } else { "+" }, p.abs())
}

/// `a → b` with the relative change; the swap's number.
pub fn pair(a: f32, b: f32) -> String {
    format!("{} → {} {}", us(a), us(b), pct(a, b))
}

pub fn kb(bytes: usize) -> String {
    match bytes {
        b if b >= 1 << 20 => format!("{:.1} MB", b as f64 / 1e6),
        b if b >= 1 << 10 => format!("{:.0} KB", b as f64 / 1e3),
        b => format!("{b} B"),
    }
}

pub fn plural(n: usize, one: &str) -> String {
    format!("{n} {one}{}", if n == 1 { "" } else { "s" })
}

pub fn spans(n: usize) -> String {
    format!("{n} span{}", if n == 1 { "" } else { "s" })
}

/// One comparison in words.
pub fn cell(c: &Cmp) -> String {
    if c.identical() {
        return "bit-identical".into();
    }
    if c.value_identical() {
        return format!("±0 only ({})", c.signed_zero);
    }
    let mut s = format!("{}/{} differ", c.n_diff - c.signed_zero, c.n);
    if let Some(u) = c.max_ulp {
        s += &format!(" · max {u} ulp");
    } else {
        s += &format!(" · max |Δ| {:.2e}", c.max_abs);
    }
    if c.nan_only_one_side > 0 {
        s += &format!(" · {} nan", c.nan_only_one_side);
    }
    s
}

/// A differing comparison: where, and what was seen.
#[derive(Serialize, Clone, Debug)]
pub struct Finding {
    pub program: String,
    pub span: String,
    pub buffer: String,
    pub what: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cmp: Option<Cmp>,
}

impl Finding {
    pub fn of(program: &str, span: &str, buffer: &str, c: &Cmp) -> Finding {
        Finding {
            program: program.into(),
            span: span.into(),
            buffer: buffer.into(),
            what: cell(c),
            cmp: Some(c.clone()),
        }
    }
    pub fn of_buf(program: &str, span: &str, buffer: &str, c: &BufCmp) -> Finding {
        let mut what = cell(&c.cmp);
        if c.outside > 0 {
            what += &format!(" · wrote {} outside A's write-set", kb(c.outside));
        }
        Finding { program: program.into(), span: span.into(), buffer: buffer.into(), what, cmp: Some(c.cmp.clone()) }
    }
    pub fn text(program: &str, span: &str, buffer: &str, what: String) -> Finding {
        Finding { program: program.into(), span: span.into(), buffer: buffer.into(), what, cmp: None }
    }
    pub fn line(&self, key: &str, prefix: &str) -> String {
        row(key, format!("{prefix}{} {} {}: {}", self.program, self.span, self.buffer, self.what), None)
    }
    /// A finding without a comparison (a state, a crash) outranks any with one.
    pub fn severity(&self) -> (usize, u64) {
        self.cmp.as_ref().map_or((usize::MAX, u64::MAX), Cmp::severity)
    }
}

/// Sort worst first and keep [`SHOW`]; the count of the rest.
pub fn cap<T>(mut all: Vec<T>, key: impl Fn(&T) -> (usize, u64)) -> (Vec<T>, usize) {
    all.sort_by_key(|f| std::cmp::Reverse(key(f)));
    let omitted = all.len().saturating_sub(SHOW);
    all.truncate(SHOW);
    (all, omitted)
}

fn more(key: &str, omitted: usize, what: &str) -> Option<String> {
    (omitted > 0).then(|| row(key, format!("… {omitted} more differing {what} (all in --out)"), None))
}

#[derive(Serialize, Debug)]
pub struct Tap {
    pub seed: String,
    pub prefill: usize,
    pub how: String,
    pub chunk: u64,
    pub decode: usize,
    pub vocab: usize,
    pub ranks: usize,
    /// Program runs of the workload, each replayed on B from A's image.
    pub runs: usize,
    /// Spans kept with their state write-set (the first run of each program).
    pub spans: usize,
    pub snapshot_bytes: usize,
    pub snapshot_pieces: usize,
    pub state_pre_image_bytes: usize,
    pub load_s: f32,
    pub record_s: f32,
    pub free_run_ms: f32,
    pub elapsed_s: f32,
}

impl Tap {
    pub fn lines(&self) -> Vec<String> {
        let mut s = format!(
            "seed {} · prefill {} ({}) in chunks of {} · decode {} · vocab {} · {} runs replayed · {} kept ({} in {} pieces) · load {} · record {}",
            self.seed,
            self.prefill,
            self.how,
            self.chunk,
            self.decode,
            self.vocab,
            self.runs,
            spans(self.spans),
            kb(self.snapshot_bytes),
            self.snapshot_pieces,
            secs(self.load_s),
            secs(self.record_s)
        );
        if self.ranks > 1 {
            s += &format!(" · {} ranks", self.ranks);
        }
        if self.state_pre_image_bytes > 0 {
            s += &format!(" · state pre-image {}", kb(self.state_pre_image_bytes));
        }
        vec![row("tap", s, Some(self.elapsed_s))]
    }
}

#[derive(Serialize, Debug)]
pub struct StateE2e {
    pub name: String,
    pub bytes: usize,
    pub differ: usize,
}

/// The tap's comparisons: B's span on A's inputs, per run, plus the end
/// state of B's free run against A.
#[derive(Serialize, Debug)]
pub struct Local {
    pub compared: usize,
    pub bit_identical: usize,
    pub value_identical: usize,
    pub findings: Vec<Finding>,
    pub omitted: usize,
    pub outputs: Vec<Finding>,
    /// An output B produced outside its declared domain.
    pub violations: Vec<String>,
    pub states: Vec<StateE2e>,
    pub one_sided: Vec<String>,
    pub undriven: Vec<String>,
}

impl Local {
    pub fn lines(&self) -> Vec<String> {
        let mut head = format!("{}/{} bit-identical", self.bit_identical, self.compared);
        if self.value_identical > 0 {
            head += &format!(" · {} value-identical (±0 only)", self.value_identical);
        }
        for o in &self.outputs {
            head += &format!(" · output {} {}", o.buffer, o.what);
        }
        for s in &self.states {
            head += &if s.differ == 0 {
                format!(" · state {} bit-identical ({})", s.name, kb(s.bytes))
            } else {
                format!(
                    " · state {} {} of {} bytes differ ({:.2}%)",
                    s.name,
                    s.differ,
                    s.bytes,
                    s.differ as f64 * 100.0 / s.bytes.max(1) as f64
                )
            };
        }
        let mut v = vec![row("local", head, None)];
        v.extend(self.violations.iter().map(|t| row("local", format!("✗ domain: {t}"), None)));
        v.extend(self.findings.iter().map(|f| f.line("local", "✗ ")));
        v.extend(more("local", self.omitted, "comparisons"));
        if !self.one_sided.is_empty() {
            v.push(row(
                "local",
                format!(
                    "declared differently or written on one side only, neither injected nor compared: {}",
                    self.one_sided.join(", ")
                ),
                None,
            ));
        }
        v.extend(self.undriven.iter().map(|p| {
            row(
                "local",
                format!("✗ {p}: changed but not tapped — the workload drives only the programs with a `batch`"),
                None,
            )
        }));
        v
    }
}

#[derive(Serialize, Debug)]
pub struct Flip {
    pub row: String,
    pub argmax_a: usize,
    pub argmax_b: usize,
    pub margin_a: f64,
    pub kl: f64,
    pub rank_in_b: usize,
    /// The row's KL is within the limit: a tie that broke the other way.
    pub within: bool,
}

/// The end-to-end oracle over every logits row of the workload.
#[derive(Serialize, Debug)]
pub struct Logits {
    pub rows: usize,
    pub differ: usize,
    pub flips: usize,
    /// Flips whose row stays within the KL limit.
    pub within: usize,
    pub kl_max: f64,
    pub kl_at: String,
    pub limit_kl: f64,
    /// The smallest top-[`TOP`](crate::compare::TOP) overlap over the rows, and where.
    pub top_min: usize,
    pub top_at: String,
    /// Rows whose top-[`TOP`](crate::compare::TOP) sets differ.
    pub top_differ: usize,
    pub flipped: Vec<Flip>,
    pub elapsed_s: f32,
}

impl Logits {
    pub fn lines(&self) -> Vec<String> {
        if self.rows == 0 {
            return vec![row(
                "logits",
                "none: no driven program writes a `logits*` buffer — the verdict falls back to span identity",
                Some(self.elapsed_s),
            )];
        }
        let mut s = format!("{}/{} argmax agree", self.rows - self.flips, self.rows);
        if self.flips > 0 {
            s += &format!(" ({} within the KL limit, {} beyond)", self.within, self.flips - self.within);
        }
        if self.differ == 0 {
            s += &format!(" · bit-identical on all rows (limit KL {:.0e})", self.limit_kl);
        } else {
            s += &format!(" · max KL {:.2e} at {} (limit {:.0e})", self.kl_max, self.kl_at, self.limit_kl);
            s += &if self.top_differ == 0 {
                format!(" · top-{TOP} same on all rows")
            } else {
                format!(
                    " · top-{TOP} differs on {} rows, overlap down to {}/{TOP} at {}",
                    self.top_differ, self.top_min, self.top_at
                )
            };
        }
        let mut v = vec![row("logits", s, Some(self.elapsed_s))];
        v.extend(self.flipped.iter().map(|f| {
            row(
                "logits",
                format!(
                    "{} {}: A {} → B {} · A's margin {:.4} · KL {:.2e} · A's token is B's #{}",
                    if f.within { "flip" } else { "✗ flip" },
                    f.row,
                    f.argmax_a,
                    f.argmax_b,
                    f.margin_a,
                    f.kl,
                    f.rank_in_b
                ),
                None,
            )
        }));
        v
    }
}

/// A against itself end to end: the workload run twice on A, its logits
/// rows compared. The band any end-to-end judgement of B sits in.
#[derive(Serialize, Debug, Clone)]
pub struct Floor {
    pub rows: usize,
    pub kl_max: f64,
    pub kl_at: String,
    pub flips: usize,
}

/// A's span re-run from its own snapshot against its own output.
#[derive(Serialize, Debug, Clone)]
pub struct Noise {
    pub compared: usize,
    pub clean: usize,
    pub findings: Vec<Finding>,
    pub omitted: usize,
    pub states: Vec<String>,
    pub floor: Option<Floor>,
    pub elapsed_s: f32,
}

impl Noise {
    pub fn lines(&self) -> Vec<String> {
        let mut s = format!("{}/{} clean", self.clean, self.compared);
        if self.clean < self.compared || !self.states.is_empty() {
            s += " · A is not deterministic at these spans; B is judged against this band";
        }
        let mut v = vec![row("noise", s, Some(self.elapsed_s))];
        v.extend(self.findings.iter().map(|f| f.line("noise", "")));
        v.extend(more("noise", self.omitted, "comparisons"));
        v.extend(self.states.iter().map(|t| row("noise", t, None)));
        if let Some(f) = &self.floor {
            v.push(row(
                "noise",
                format!(
                    "A against itself end to end: {} rows · max KL {:.2e} at {} · {} argmax flip{}",
                    f.rows,
                    f.kl_max,
                    f.kl_at,
                    f.flips,
                    if f.flips == 1 { "" } else { "s" }
                ),
                None,
            ));
        }
        v
    }
}

/// One program timed whole on both sides; `derived` is A's step with A's
/// spans swapped for B's (both eager), so measured − derived is the
/// launch-gap / L2 interaction of the swap.
#[derive(Serialize, Debug)]
pub struct StepPerf {
    pub program: String,
    pub rows: u64,
    pub spans: usize,
    pub eager_ms: [f32; 2],
    pub derived_ms: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph_ms: Option<[f32; 2]>,
    pub span_ms: [f32; 2],
}

#[derive(Serialize, Debug)]
pub struct SweepPt {
    pub rows: u64,
    pub eager_ms: [f32; 2],
    pub derived_ms: f32,
    pub span_ms: [f32; 2],
}

/// Static bytes moved by a changed kernel against the time it took.
#[derive(Serialize, Debug)]
pub struct Roof {
    pub program: String,
    pub op: String,
    pub calls: usize,
    pub bytes_per_call: usize,
    pub us_per_call: [f32; 2],
    pub gbs: [f64; 2],
    pub peak_pct: [f64; 2],
}

#[derive(Serialize, Debug)]
pub struct Perf {
    pub iters: usize,
    pub steps: Vec<StepPerf>,
    pub sweep_program: String,
    pub sweep: Vec<SweepPt>,
    pub roofline: Vec<Roof>,
    pub peak_bw_gbs: f64,
    /// Some changed kernel touches opaque state; its traffic is not counted.
    pub state_traffic: bool,
    pub elapsed_s: f32,
}

impl Perf {
    pub fn lines(&self) -> Vec<String> {
        let mut v = Vec::new();
        for (i, s) in self.steps.iter().enumerate() {
            let mut t = format!("{} rows {} · eager {}", s.program, s.rows, pair(s.eager_ms[0], s.eager_ms[1]));
            if let Some([ga, gb]) = s.graph_ms {
                t += &format!(" · tpot {} ({:.0} → {:.0} tok/s)", pair(ga, gb), 1e3 / ga, 1e3 / gb);
            }
            t += &format!(" · {} {}", spans(s.spans), pair(s.span_ms[0], s.span_ms[1]));
            v.push(row("perf", t, (i == 0).then_some(self.elapsed_s)));
        }
        if self.sweep.len() > 1 {
            let pts: Vec<String> =
                self.sweep.iter().map(|p| format!("{} rows {}", p.rows, pair(p.eager_ms[0], p.eager_ms[1]))).collect();
            v.push(row("sweep", format!("{} · {}", self.sweep_program, pts.join(" · ")), None));
        }
        for r in &self.roofline {
            let side = |i: usize| -> String {
                if r.us_per_call[i].is_nan() {
                    "—".into()
                } else {
                    format!("{} · {:.1} GB/s · {:.2}% of peak", us(r.us_per_call[i] / 1e3), r.gbs[i], r.peak_pct[i])
                }
            };
            v.push(row(
                "roofline",
                format!(
                    "{} {} ×{} · {}/call{} · A {} · B {}",
                    r.program,
                    r.op,
                    r.calls,
                    kb(r.bytes_per_call),
                    if self.state_traffic { " + opaque state" } else { "" },
                    side(0),
                    side(1)
                ),
                None,
            ));
        }
        v
    }
}

#[derive(Serialize, Debug, Default, Clone)]
pub struct Verdict {
    pub code: i32,
    pub pass: bool,
    pub summary: String,
    pub elapsed_s: f32,
}

impl Verdict {
    pub fn new(code: i32, summary: String, elapsed_s: f32) -> Verdict {
        Verdict { code, pass: code == 0, summary, elapsed_s }
    }
    pub fn lines(&self) -> Vec<String> {
        let tag = match self.code {
            0 => "PASS",
            1 => "FAIL",
            _ => "INCONCLUSIVE",
        };
        vec![row(tag, &self.summary, Some(self.elapsed_s))]
    }
}

/// The report: every section, as `--json` prints it.
#[derive(Serialize, Debug, Default)]
pub struct Summary {
    pub a: String,
    pub b: String,
    pub diff: Diff,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tap: Option<Tap>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub local: Option<Local>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logits: Option<Logits>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub noise: Option<Noise>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub perf: Option<Perf>,
    pub verdict: Verdict,
}

/// The summary plus every differing comparison: what `--out` archives.
#[derive(Serialize, Debug)]
pub struct Report {
    pub summary: Summary,
    pub detail: Value,
}

impl Report {
    /// A run that stopped at the static diff: nothing measured, so the
    /// verdict is the one the caller states (a no-op swap passes, a
    /// `--diff-only` run has no verdict of its own).
    pub fn of_diff(a: &str, b: &str, diff: Diff, verdict: Verdict) -> Report {
        let summary = Summary { a: a.into(), b: b.into(), diff, verdict, ..Default::default() };
        Report { summary, detail: Value::Object(Default::default()) }
    }

    /// The exit code: 0 PASS, 1 FAIL, 2 INCONCLUSIVE.
    pub fn code(&self) -> i32 {
        self.summary.verdict.code
    }
}
