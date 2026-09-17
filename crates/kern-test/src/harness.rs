//! Recording A: the seeded workload once, keeping at every span of every
//! program run what the span consumed and what A produced, and an image
//! of A's state at the start of each run; then A's noise floor and A's
//! timings. Everything B is later judged against lives in the
//! [`Recording`]; A is not needed after this. Generic over [`Side`];
//! every decision here is made on numbers the side handed back.
//!
//! A buffer is kept as a [`Snap`]: the first time whole, after that as
//! the 64-byte blocks that differ from the time before, found on the
//! device against a shadow copy. The same bytes seen twice cost nothing
//! (the moment is the same snapshot), and a span's write to a buffer is
//! exactly the delta between its pre- and post-snapshot: the range the
//! other side is compared on. A delta of half the buffer or more is a
//! whole again, so a chain is never longer than the writes that made it.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use kern_manifest::protocol::{Forward, Rows};
use kern_manifest::types::{BufferKind, DType, Manifest};
use kern_manifest::values;
use kern_manifest::{Protocol, Verified};

use crate::compare::{coalesce, outside, BufCmp, Cmp, LogitRow};
use crate::diff::{access, constants, frontier_inputs, live_bytes, row_elems, Diff, Span};
use crate::report::{cap, kb, row, Finding, Floor, Noise};
use crate::workload::{self, Rng, Workload};
use crate::{At, Options, Side, Vars};

/// `state -> [(offset, bytes)]`: the runs of a state a span changed.
pub(crate) type Runs = BTreeMap<String, Vec<(usize, Vec<u8>)>>;

/// Gaps under this many bytes between changed blocks are kept too: a
/// scattered write-set as few pieces, each piece one copy and one
/// comparison.
const GAP: usize = 256 << 10;

/// A buffer's bytes at one moment of the recording, on the device: a
/// whole copy of the live prefix (the base) and, over it, the pieces
/// written since, in load order. A later piece covers an earlier one;
/// an earlier piece a later one covers whole is dropped, so the list
/// is as long as the distinct regions written since the base.
pub(crate) struct Snap<B> {
    len: usize,
    base: Arc<B>,
    pieces: Vec<(Range<usize>, Arc<B>)>,
}

impl<B> Snap<B> {
    pub fn len(&self) -> usize {
        self.len
    }
    /// The snapshot into a side's buffer: the base, then every piece.
    pub fn load<S: Side<Buf = B>>(&self, c: &mut S, q: usize, name: &str) -> Result<()> {
        c.load(q, name, 0..self.len, &self.base)?;
        self.pieces.iter().try_for_each(|(r, b)| c.load(q, name, r.clone(), b))
    }
    /// Where bytes `r` of this snapshot are: the last piece holding all
    /// of them, else the base.
    pub fn piece(&self, r: Range<usize>) -> (&B, Range<usize>) {
        match self.pieces.iter().rev().find(|(at, _)| at.start <= r.start && r.end <= at.end) {
            Some((at, b)) => (b, r.start - at.start..r.end - at.start),
            None => (&self.base, r),
        }
    }
    /// Every device allocation this snapshot holds: its address and size.
    fn allocs(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        std::iter::once((Arc::as_ptr(&self.base) as usize, self.len))
            .chain(self.pieces.iter().map(|(r, b)| (Arc::as_ptr(b) as usize, r.len())))
    }
    /// Over `prev`, the bytes at `ranges` as they are now.
    fn over<S: Side<Buf = B>>(prev: &Snap<B>, c: &S, q: usize, name: &str, ranges: &[Range<usize>]) -> Result<Self> {
        let covered = |r: &Range<usize>| ranges.iter().any(|n| n.start <= r.start && r.end <= n.end);
        let mut pieces: Vec<_> = prev.pieces.iter().filter(|(r, _)| !covered(r)).cloned().collect();
        for r in ranges {
            pieces.push((r.clone(), Arc::new(c.save(q, name, r.clone())?)));
        }
        Ok(Snap { len: prev.len, base: prev.base.clone(), pieces })
    }
}

/// The latest snapshot of every buffer on every rank, with a whole copy
/// of it on the device (the shadow) the next delta is found against.
struct Chains<B> {
    last: BTreeMap<(usize, String), (Arc<Snap<B>>, Arc<B>)>,
}

impl<B> Chains<B> {
    fn new() -> Self {
        Chains { last: BTreeMap::new() }
    }
    /// The buffer's live prefix now, and what changed since it was last
    /// snapped (everything, the first time). Unchanged is the same
    /// snapshot; changed by half or more is a new base.
    fn snap<S: Side<Buf = B>>(
        &mut self,
        c: &S,
        q: usize,
        name: &str,
        len: usize,
    ) -> Result<(Arc<Snap<B>>, Vec<Range<usize>>)> {
        let key = (q, name.to_string());
        if let Some((last, shadow)) = self.last.get(&key).filter(|(s, _)| s.len() == len) {
            let ranges = coalesce(c.changed(q, At::Scratch(shadow, 0..len), At::Buffer(name, 0..len))?, GAP);
            if ranges.is_empty() {
                return Ok((last.clone(), ranges));
            }
            let fresh = Arc::new(c.save(q, name, 0..len)?);
            let delta: usize = ranges.iter().map(|r| r.len()).sum();
            let snap = if delta * 2 >= len {
                Arc::new(Snap { len, base: fresh.clone(), pieces: Vec::new() })
            } else {
                Arc::new(Snap::over(last, c, q, name, &ranges)?)
            };
            self.last.insert(key, (snap.clone(), fresh));
            return Ok((snap, ranges));
        }
        let fresh = Arc::new(c.save(q, name, 0..len)?);
        let snap = Arc::new(Snap { len, base: fresh.clone(), pieces: Vec::new() });
        self.last.insert(key, (snap.clone(), fresh));
        Ok((snap, std::iter::once(0..len).collect()))
    }
}

/// A buffer a span writes, as A saw it: before, after, and the ranges the
/// span changed (A's write-set: the comparison's range).
pub(crate) struct Out<B> {
    pub pre: Arc<Snap<B>>,
    pub post: Arc<Snap<B>>,
    pub wrote: Vec<Range<usize>>,
}

/// How a side's write to a state compares with A's, on a span: bytes of
/// A's write-set (`set`), how many of them the side wrote differently
/// (`n_diff`), and how many bytes it changed outside A's write-set.
#[derive(Clone, Default)]
pub(crate) struct StateCmp {
    pub set: usize,
    pub n_diff: usize,
    pub outside: usize,
}

/// One span replayed on a side: every written buffer and every touched
/// state against A, per rank.
pub(crate) struct Replayed {
    pub bufs: Vec<BTreeMap<String, BufCmp>>,
    pub states: Vec<BTreeMap<String, StateCmp>>,
}

/// One span of one program run, as A saw it, per rank.
pub(crate) struct SpanRec<B> {
    pub span: Span,
    /// A's frontier inputs.
    pub inputs: Vec<Vec<(String, Arc<Snap<B>>)>>,
    /// What A wrote that both sides write.
    pub ref_out: Vec<BTreeMap<String, Out<B>>>,
    /// Kept runs only: pre-image and post-image of every state byte the
    /// span changed. A span with inout state is not idempotent (replaying
    /// it on its own output shifts the conv window again, advances the SSM
    /// again), so every replay first writes the pre-image back, and the
    /// post-image when it is done. States are opaque; this is byte-level.
    pub pre: Vec<Runs>,
    pub post: Vec<Runs>,
}

/// One program run of the workload.
pub(crate) struct Run<B> {
    pub program: String,
    /// `chunk 3` / `step 7`.
    pub label: String,
    pub vars: Vars,
    pub tokens: Vec<i64>,
    /// Positions the cursor moves past this run.
    pub advance: u64,
    /// Per rank: A's shared states at the start of the run. B starts the
    /// run from it; with per-layer write-sets that is the pre-image of
    /// every span of the run at one copy instead of a read per span.
    pub image: Vec<BTreeMap<String, B>>,
    pub spans: Vec<SpanRec<B>>,
}

/// A side's `logits*` buffer after one run, per rank: live bytes and
/// where they are kept (on the device); `cols` elements per row.
pub(crate) struct LogitsAt<B> {
    pub label: String,
    pub buffer: String,
    pub cols: usize,
    pub bufs: Vec<(usize, B)>,
}

pub(crate) struct NoiseRec {
    pub report: Noise,
    /// program -> buffer -> [(span label, cmp)]
    pub res: BTreeMap<String, BTreeMap<String, Vec<(String, Cmp)>>>,
    pub noisy_states: BTreeSet<(String, String)>,
    pub findings: Vec<Finding>,
}

pub(crate) struct StepRec {
    pub program: String,
    pub rows: u64,
    pub token: i64,
    pub times: Vec<f32>,
    pub graph_ms: Option<f32>,
}

pub(crate) struct SweepRec {
    pub rows: u64,
    pub tokens: Vec<i64>,
    pub iters: usize,
    pub times: Vec<f32>,
    pub tapped: bool,
}

pub(crate) struct PerfRec {
    pub steps: Vec<StepRec>,
    pub sweep: Vec<SweepRec>,
}

/// Everything B is judged against. Device bytes ([`Side::Buf`]) outlive
/// the side that recorded them.
pub struct Recording<B> {
    /// Seconds spent loading the sides; the caller adds each load.
    pub load_s: f32,
    pub(crate) record_s: f32,
    pub(crate) ma: Manifest,
    pub(crate) mb: Manifest,
    pub(crate) diff: Diff,
    pub(crate) wl: Workload,
    pub(crate) ranks: usize,
    pub(crate) chunk: Option<Forward>,
    steps: Vec<Forward>,
    /// Programs either side runs once after load: their spans are each
    /// side's own setup, not something the workload drives.
    once: Vec<String>,
    pub(crate) runs: Vec<Run<B>>,
    /// `(run, span)` of the spans with state images: the first run of each
    /// driven program. The noise floor replays these.
    pub(crate) kept: Vec<(usize, usize)>,
    pub(crate) logits: Vec<LogitsAt<B>>,
    /// Output buffers after the workload, per rank.
    pub(crate) outputs: BTreeMap<String, Vec<Vec<u8>>>,
    /// Per rank: A's shared states after the workload.
    pub(crate) states: Vec<BTreeMap<String, B>>,
    pub(crate) one_sided: BTreeSet<String>,
    pub(crate) noise: Option<NoiseRec>,
    pub(crate) perf: Option<PerfRec>,
}

impl<B> Recording<B> {
    pub(crate) fn spans_of(&self, program: &str) -> Vec<Span> {
        self.diff.spans.get(program).cloned().unwrap_or_default()
    }
    /// Programs with spans the workload does not drive.
    pub(crate) fn undriven(&self) -> Vec<String> {
        let driven: Vec<&str> = self.chunk.iter().chain(&self.steps).map(|f| f.name.as_str()).collect();
        self.diff.spans.keys().filter(|p| !driven.contains(&p.as_str()) && !self.once.contains(p)).cloned().collect()
    }
    /// States both sides declare identically: the ones A's images fit.
    pub(crate) fn shared_states(&self) -> Vec<String> {
        self.ma
            .states
            .iter()
            .filter(|(n, a)| self.mb.states.get(*n).is_some_and(|b| declares_same(*a, b)))
            .map(|(n, _)| n.clone())
            .collect()
    }
    /// What the spans' inputs and reference outputs hold on the device:
    /// bytes and allocations, each counted once.
    pub(crate) fn snapshot_bytes(&self) -> (usize, usize) {
        let mut seen = BTreeMap::new();
        let snaps = self.runs.iter().flat_map(|r| &r.spans).flat_map(|sr| {
            sr.inputs
                .iter()
                .flatten()
                .map(|(_, s)| s)
                .chain(sr.ref_out.iter().flatten().flat_map(|(_, o)| [&o.pre, &o.post]))
        });
        for (ptr, bytes) in snaps.flat_map(|s| s.allocs()) {
            seen.insert(ptr, bytes);
        }
        (seen.values().sum(), seen.len())
    }
    pub(crate) fn pre_image_bytes(&self) -> usize {
        self.kept
            .iter()
            .map(|&(r, s)| {
                self.runs[r].spans[s].pre.iter().flatten().flat_map(|(_, rs)| rs).map(|(_, b)| b.len()).sum::<usize>()
            })
            .sum()
    }
}

/// The workload's programs: the chunk program over the prompt, then every
/// fixed-rows forward in rotation (a caller may switch between them at any
/// step: same state contract).
fn driven(p: &Protocol) -> (Option<Forward>, Vec<Forward>) {
    let chunk = p.chunk().cloned();
    let steps = p.forwards.iter().filter(|f| matches!(f.rows, Rows::Const(_))).cloned().collect();
    (chunk, steps)
}

fn rows_of(f: &Forward) -> u64 {
    match f.rows {
        Rows::Const(r) => r,
        Rows::Var => 1,
    }
}

/// A span label on a rank, when there are several.
pub(crate) fn at_rank(ranks: usize, q: usize, label: &str) -> String {
    if ranks > 1 {
        format!("rank {q} {label}")
    } else {
        label.to_string()
    }
}

fn write_runs<S: Side>(c: &mut S, q: usize, runs: &Runs) -> Result<()> {
    for (name, rs) in runs {
        for (off, bytes) in rs {
            c.write_state(q, name, *off, bytes)?;
        }
    }
    Ok(())
}

/// Put a side's shared states to the image a run started from.
pub(crate) fn image<S: Side>(c: &mut S, run: &Run<S::Buf>) -> Result<()> {
    for (q, img) in run.image.iter().enumerate() {
        for (n, b) in img {
            c.load_state(q, n, b)?;
        }
    }
    Ok(())
}

/// Buffers a program writes on both sides whose last name segment is
/// `logits*` (`logits`, `logits_blk`, `decode.target_head.logits`): the
/// end-to-end oracle.
fn logits_of(ma: &Manifest, mb: &Manifest, prog: &str) -> Vec<String> {
    access(ma, prog, 0..ma.programs[prog].calls.len())
        .writes
        .into_iter()
        .filter(|n| n.rsplit('.').next().is_some_and(|last| last.starts_with("logits")) && alike(ma, mb, n))
        .collect()
}

/// Two declarations, field for field.
fn declares_same<T: serde::Serialize>(a: &T, b: &T) -> bool {
    serde_json::to_value(a).ok() == serde_json::to_value(b).ok()
}

/// A buffer both sides declare with the same dtype and shape: the only
/// kind whose bytes mean the same thing on either side, so the only kind
/// A can hand over or B's copy can be compared against.
fn alike(ma: &Manifest, mb: &Manifest, n: &str) -> bool {
    ma.buffers.get(n).zip(mb.buffers.get(n)).is_some_and(|(x, y)| x.dtype == y.dtype && x.shape == y.shape)
}

fn read_logits<S: Side>(
    c: &S,
    ma: &Manifest,
    mb: &Manifest,
    prog: &str,
    e: &Vars,
    label: &str,
) -> Result<Vec<LogitsAt<S::Buf>>> {
    logits_of(ma, mb, prog)
        .into_iter()
        .map(|n| {
            let len = live_bytes(ma, &n, e);
            let bufs = (0..c.ranks()).map(|q| Ok((len, c.save(q, &n, 0..len)?))).collect::<Result<_>>()?;
            Ok(LogitsAt { label: label.to_string(), cols: row_elems(ma, &n, e), buffer: n, bufs })
        })
        .collect()
}

/// The workload once more from zero state with nothing injected, on
/// either side: what a caller would get, as its `logits*` reads per run.
pub(crate) fn free_run<S: Side>(
    c: &mut S,
    ma: &Manifest,
    mb: &Manifest,
    runs: &[Run<S::Buf>],
) -> Result<Vec<LogitsAt<S::Buf>>> {
    c.zero_states()?;
    c.reset();
    let mut out = Vec::new();
    for run in runs {
        let e = c.stage(&run.tokens)?;
        c.run(&run.program, &e, 0..c.calls(&run.program)?)?;
        out.extend(read_logits(c, ma, mb, &run.program, &e, "")?);
        c.advance(run.advance);
    }
    Ok(out)
}

/// The end-to-end rows of two sides' `logits*` reads, in workload order:
/// one per rank and, for a buffer of several rows, per row; compared
/// where they are.
pub(crate) fn logit_rows<S: Side>(
    c: &S,
    ma: &Manifest,
    ranks: usize,
    a: &[LogitsAt<S::Buf>],
    b: &[LogitsAt<S::Buf>],
) -> Result<Vec<LogitRow>> {
    let n_runs_with = |label: &str| a.iter().filter(|x| x.label == label).count();
    let mut rows_out = Vec::new();
    for (la, lb) in a.iter().zip(b) {
        let dt = ma.buffers[&la.buffer].dtype;
        let w = dt.bytes() as usize;
        for q in 0..ranks {
            let (len, x) = &la.bufs[q];
            let (_, y) = &lb.bufs[q];
            if *len == 0 {
                continue;
            }
            // a live prefix that is not whole rows is one row of what there is
            let cols = if *len % (la.cols * w) == 0 { la.cols } else { *len / w };
            let stats = c.logits(q, dt, cols, At::Scratch(x, 0..*len), At::Scratch(y, 0..*len))?;
            let rows = stats.len();
            for (r, st) in stats.into_iter().enumerate() {
                let lbl = if n_runs_with(&la.label) > 1 || rows > 1 {
                    format!("{} {}{}", la.label, la.buffer, if rows > 1 { format!("[{r}]") } else { String::new() })
                } else {
                    la.label.clone()
                };
                rows_out.push(LogitRow { label: at_rank(ranks, q, &lbl), stats: st });
            }
        }
    }
    Ok(rows_out)
}

/// Replay one recorded span on a side from the recording and compare what
/// it wrote with A's, where it is. Every buffer the span reads or writes
/// is first put to A's snapshot before the span and the state to A's
/// pre-image, so the span sees what A saw on either side and what it
/// writes is its own doing; the state goes to A's post-image after, so
/// the next span's reads see the reference, not this replay's output.
/// A buffer is compared on A's write-set; a state on A's write-set with
/// the bytes changed outside it counted.
pub(crate) fn replay_span<S: Side>(
    c: &mut S,
    m: &Manifest,
    sr: &SpanRec<S::Buf>,
    run: &Run<S::Buf>,
    side_b: bool,
) -> Result<Replayed> {
    let ranks = c.ranks();
    let mut pre_bufs: Vec<BTreeMap<String, S::Buf>> = Vec::new();
    let mut pre_states: Vec<BTreeMap<String, S::Buf>> = Vec::new();
    for q in 0..ranks {
        write_runs(c, q, &sr.pre[q])?;
        for (n, s) in &sr.inputs[q] {
            s.load(c, q, n)?;
        }
        for (n, o) in &sr.ref_out[q] {
            if !sr.inputs[q].iter().any(|(i, s)| i == n && Arc::ptr_eq(s, &o.pre)) {
                o.pre.load(c, q, n)?;
            }
        }
        pre_bufs.push(
            sr.ref_out[q].iter().map(|(n, o)| Ok((n.clone(), c.save(q, n, 0..o.pre.len())?))).collect::<Result<_>>()?,
        );
        pre_states.push(sr.pre[q].keys().map(|st| Ok((st.clone(), c.save_state(q, st)?))).collect::<Result<_>>()?);
    }
    let r = if side_b { sr.span.b.clone() } else { sr.span.a.clone() };
    c.run(&run.program, &run.vars, r)?;
    let (mut bufs, mut states) = (Vec::new(), Vec::new());
    for q in 0..ranks {
        let mut bq = BTreeMap::new();
        for (n, o) in &sr.ref_out[q] {
            let dt = m.buffers[n].dtype;
            let mut cmp = Cmp::default();
            for w in &o.wrote {
                let (b, r) = o.post.piece(w.clone());
                cmp = cmp.merge(c.compare(q, dt, At::Buffer(n, w.clone()), At::Scratch(b, r))?);
            }
            let len = o.pre.len();
            let changed = c.changed(q, At::Scratch(&pre_bufs[q][n], 0..len), At::Buffer(n, 0..len))?;
            bq.insert(n.clone(), BufCmp { cmp, outside: outside(&changed, &o.wrote) });
        }
        bufs.push(bq);
        let mut sq = BTreeMap::new();
        for (st, runs) in &sr.pre[q] {
            let len = c.state_bytes(st)?;
            let img = &pre_states[q][st];
            let set: usize = runs.iter().map(|(_, b)| b.len()).sum();
            // A's write-set is small and comes to the host; the rest of the
            // state is compared where it is
            let (mut n_diff, mut inside) = (0usize, 0usize);
            for (off, ap) in &sr.post[q][st] {
                let now = c.read_state(q, st, *off..off + ap.len())?;
                let was = c.bytes(q, img, *off..off + ap.len())?;
                n_diff += ap.iter().zip(&now).filter(|(x, y)| x != y).count();
                inside += was.iter().zip(&now).filter(|(x, y)| x != y).count();
            }
            let total = c.compare(q, DType::U8, At::Scratch(img, 0..len), At::State(st, 0..len))?.n_diff;
            sq.insert(st.clone(), StateCmp { set, n_diff, outside: total - inside });
        }
        states.push(sq);
        write_runs(c, q, &sr.post[q])?;
    }
    Ok(Replayed { bufs, states })
}

/// Values a side produced that lie outside their buffer's declared
/// domain, among `names`, as `side name[i] = v`; only buffers with a
/// domain are read.
pub(crate) fn domain_violations<S: Side>(
    c: &S,
    m: &Manifest,
    vars: &Vars,
    side: &str,
    names: &[String],
) -> Result<Vec<String>> {
    let mut v = Vec::new();
    for q in 0..c.ranks() {
        for name in names {
            let Some(d) = &m.buffers[name].domain else { continue };
            let r = d.resolve(m, vars, &c.provision())?;
            let vals = values::to_f64(m.buffers[name].dtype, &c.read(q, name, live_bytes(m, name, vars))?);
            if let Some(i) = vals.iter().position(|x| !r.contains(*x)) {
                v.push(format!("{} {name}[{i}] = {} outside domain", at_rank(c.ranks(), q, side), vals[i]));
            }
        }
    }
    Ok(v)
}

/// Record A on the workload `o` and the diff's spans: what every span
/// consumed and produced, then A's noise floor and timings.
/// `mb` is B's manifest: which buffers and states the two sides share,
/// and what B's `once` programs make their own.
pub fn record<S: Side>(
    o: &Options,
    diff: Diff,
    mb: &Verified,
    a: &mut S,
    out: &mut dyn FnMut(&[String]),
) -> Result<Recording<S::Buf>> {
    let t0 = Instant::now();
    let ma: Manifest = (**a.manifest()).clone();
    let pa = Protocol::check(a.manifest()).context("A does not fit the serving protocol")?;
    let (chunk_f, step_fs) = driven(&pa);
    let ranks = a.ranks();
    let wl = workload::sample(o, &ma, &pa, a.provision(), a.page())?;
    let mut rec = Recording {
        load_s: 0.0,
        record_s: 0.0,
        ma: ma.clone(),
        mb: (**mb).clone(),
        diff,
        wl: wl.clone(),
        ranks,
        chunk: chunk_f.clone(),
        steps: step_fs.clone(),
        once: Vec::new(),
        runs: Vec::new(),
        kept: Vec::new(),
        logits: Vec::new(),
        outputs: BTreeMap::new(),
        states: Vec::new(),
        one_sided: ma
            .states
            .keys()
            .filter(|n| {
                mb.states.contains_key(*n) && !mb.states.get(*n).is_some_and(|b| declares_same(&ma.states[*n], b))
            })
            .cloned()
            .collect(),
        noise: None,
        perf: None,
    };
    let shared = rec.shared_states();
    let pb = Protocol::check(mb).context("B does not fit the serving protocol")?;
    let mut fixed = constants(&ma, &pa.once);
    fixed.extend(constants(mb, &pb.once));
    rec.once = pa.once.iter().chain(&pb.once).cloned().collect();
    let mb: &Manifest = mb;
    let chunk_name = chunk_f.as_ref().map_or("", |f| f.name.as_str());

    // ---- the workload: one run at a time, spans recorded as A passes them
    let mut chains = Chains::new();
    a.reset();
    let mut i = 0;
    let mut n_chunks = 0usize;
    while i < wl.prefill.len() {
        let c = (wl.prefill.len() - i).min(wl.chunk);
        let tokens = wl.prefill[i..i + c].to_vec();
        let keep = n_chunks == 0;
        record_run(
            a,
            &mut rec,
            &mut chains,
            chunk_name,
            format!("chunk {n_chunks}"),
            tokens,
            c as u64,
            keep,
            &shared,
            &fixed,
        )?;
        a.advance(c as u64);
        i += c;
        n_chunks += 1;
    }
    // Every step stages the drawn token in a forward's rows at the cursor
    // and advances one position: a wide forward's extra rows are
    // overwritten by the next step, on both sides alike.
    for (k, &tok) in wl.decode.iter().enumerate() {
        let f = &step_fs[k % step_fs.len()];
        let tokens = vec![tok; rows_of(f) as usize];
        record_run(
            a,
            &mut rec,
            &mut chains,
            &f.name,
            format!("step {k}"),
            tokens,
            1,
            k < step_fs.len(),
            &shared,
            &fixed,
        )?;
        a.advance(1);
    }
    // What a caller would get: the outputs and the states after the workload.
    let e_last = rec.runs.last().map(|r| r.vars.clone()).unwrap_or_default();
    for (name, b) in &ma.buffers {
        if b.kind == BufferKind::Output && alike(&ma, mb, name) {
            let len = live_bytes(&ma, name, &e_last);
            rec.outputs.insert(name.clone(), (0..ranks).map(|q| a.read(q, name, len)).collect::<Result<_>>()?);
        }
    }
    for q in 0..ranks {
        let img = shared.iter().map(|n| Ok((n.clone(), a.save_state(q, n)?))).collect::<Result<_>>()?;
        rec.states.push(img);
    }
    let workload_s = t0.elapsed().as_secs_f32();

    // ---- noise floor: A's kept spans replayed from their own recording,
    // and the whole workload once more for the end-to-end band
    if o.noise {
        let t_n = Instant::now();
        let again = free_run(a, &ma, mb, &rec.runs)?;
        let rows = logit_rows(a, &ma, ranks, &rec.logits, &again)?;
        let worst = rows.iter().max_by(|x, y| x.stats.kl.total_cmp(&y.stats.kl));
        let floor = worst.map(|w| Floor {
            rows: rows.len(),
            kl_max: w.stats.kl,
            kl_at: w.label.clone(),
            flips: rows.iter().filter(|r| r.stats.flip()).count(),
        });
        let mut res: BTreeMap<String, BTreeMap<String, Vec<(String, Cmp)>>> = BTreeMap::new();
        let mut noisy_states = BTreeSet::new();
        let mut findings = Vec::new();
        let mut state_noise = Vec::new();
        let (mut compared, mut clean) = (0usize, 0usize);
        let mut cur = None;
        for &(r, s) in &rec.kept {
            let run = &rec.runs[r];
            if cur != Some(r) {
                image(a, run)?;
                cur = Some(r);
            }
            let sr = &run.spans[s];
            let rp = replay_span(a, &ma, sr, run, false)?;
            for q in 0..ranks {
                let label = at_rank(ranks, q, &format!("{} {}", run.label, sr.span.label()));
                for (n, c) in &rp.bufs[q] {
                    compared += 1;
                    if c.identical() {
                        clean += 1;
                    } else {
                        findings.push(Finding::of_buf(&run.program, &label, n, c));
                    }
                    res.entry(run.program.clone())
                        .or_default()
                        .entry(n.clone())
                        .or_default()
                        .push((label.clone(), c.cmp.clone()));
                }
                for (name, sc) in &rp.states[q] {
                    if sc.n_diff > 0 {
                        noisy_states.insert((run.program.clone(), name.clone()));
                        state_noise.push(format!("state {name}: {} bytes at {} {label}", sc.n_diff, run.program));
                    }
                }
            }
        }
        let (shown, omitted) = cap(findings.clone(), Finding::severity);
        let report = Noise {
            compared,
            clean,
            findings: shown,
            omitted,
            states: state_noise,
            floor,
            elapsed_s: t_n.elapsed().as_secs_f32(),
        };
        rec.noise = Some(NoiseRec { report, res, noisy_states, findings });
    }

    // ---- perf: A's side of every timing
    if o.perf {
        let undriven = rec.undriven();
        let mut steps = Vec::new();
        let last_tok = *wl.decode.last().unwrap_or(&0);
        for f in &step_fs {
            let p = f.name.as_str();
            if rec.spans_of(p).is_empty() || undriven.iter().any(|u| u == p) {
                continue;
            }
            let e = a.stage(&vec![last_tok; rows_of(f) as usize])?;
            let times = a.time(p, &e, 0..a.calls(p)?, o.iters)?;
            let graph_ms = if o.graph_step { Some(a.time_graph(p, &e, 100)?) } else { None };
            steps.push(StepRec { program: p.into(), rows: rows_of(f), token: last_tok, times, graph_ms });
        }
        let mut sweep = Vec::new();
        if chunk_f.is_some() && !rec.spans_of(chunk_name).is_empty() && !undriven.iter().any(|u| u == chunk_name) {
            // The manifest's widest chunk, within what the sequence was provisioned.
            let max = pa.rows.max.min(a.provision().tokens);
            let tap_len = wl.prefill.len().min(wl.chunk) as u64;
            let mut points: BTreeSet<u64> = [tap_len].into();
            if o.sweep {
                points.extend([1u64, 16, 128, 512, 2048, 4096, max].into_iter().filter(|&t| t <= max));
            }
            let vocab = a.vocab();
            let mut rng = Rng(o.seed);
            for &t in &points {
                let tokens: Vec<i64> = (0..t).map(|_| rng.below(vocab) as i64).collect();
                a.reset();
                let e = a.stage(&tokens)?;
                let iters = if t == tap_len { o.iters } else { o.iters.min(10) };
                let times = a.time(chunk_name, &e, 0..a.calls(chunk_name)?, iters)?;
                sweep.push(SweepRec { rows: t, tokens, iters, times, tapped: t == tap_len });
            }
        }
        rec.perf = Some(PerfRec { steps, sweep });
    }
    rec.record_s = t0.elapsed().as_secs_f32();
    let image_bytes = image_bytes(&rec, a)?;
    let (snap_bytes, pieces) = rec.snapshot_bytes();
    out(&[row(
        "record",
        format!(
            "A: {} runs · {} spans kept ({} in {} pieces) · state images {} · workload {}",
            rec.runs.len(),
            rec.kept.len(),
            kb(snap_bytes),
            pieces,
            kb(image_bytes),
            crate::report::secs(workload_s)
        ),
        Some(rec.record_s),
    )]);
    Ok(rec)
}

/// Bytes the per-run state images hold.
fn image_bytes<S: Side>(rec: &Recording<S::Buf>, c: &S) -> Result<usize> {
    let mut n = 0;
    for run in &rec.runs {
        for img in &run.image {
            for name in img.keys() {
                n += c.state_bytes(name)?;
            }
        }
    }
    Ok(n)
}

/// Stage and run one program on A, recording every span on the way.
#[allow(clippy::too_many_arguments)]
fn record_run<S: Side>(
    a: &mut S,
    rec: &mut Recording<S::Buf>,
    chains: &mut Chains<S::Buf>,
    pname: &str,
    label: String,
    tokens: Vec<i64>,
    advance: u64,
    keep: bool,
    shared: &[String],
    fixed: &BTreeSet<String>,
) -> Result<()> {
    let ranks = rec.ranks;
    let e = a.stage(&tokens)?;
    let mut image = Vec::new();
    for q in 0..ranks {
        let mut img = BTreeMap::new();
        for n in shared {
            let b = a.save_state(q, n).with_context(|| format!("imaging state `{n}` before {pname} {label}"))?;
            img.insert(n.clone(), b);
        }
        image.push(img);
    }
    let (ma, mb) = (&rec.ma, &rec.mb);
    let segs = rec.spans_of(pname);
    let mut spans = Vec::new();
    let mut ia = 0;
    for span in &segs {
        a.run(pname, &e, ia..span.a.start)?;
        ia = span.a.end;
        // Inputs both sides have, declared alike, and A can hand over: a
        // buffer only B knows (an intermediate of its own), a weight, or
        // what a `once` program made of one (B packs it its own way) is
        // B's to produce; one B declares with another shape B reads as its
        // own, and the report says so.
        let (names, apart): (Vec<String>, Vec<String>) = frontier_inputs(ma, pname, span.a.clone())
            .union(&frontier_inputs(mb, pname, span.b.clone()))
            .filter(|n| ma.buffers.contains_key(*n) && mb.buffers.contains_key(*n))
            .filter(|n| ma.buffers[*n].kind != BufferKind::Weight && !fixed.contains(*n))
            .cloned()
            .partition(|n| alike(ma, mb, n));
        rec.one_sided.extend(apart);
        let (aa, ab) = (access(ma, pname, span.a.clone()), access(mb, pname, span.b.clone()));
        // Compared: what both sides write into the same declaration.
        let (written, apart): (Vec<String>, Vec<String>) = aa
            .writes
            .union(&ab.writes)
            .cloned()
            .partition(|n| aa.writes.contains(n) && ab.writes.contains(n) && alike(ma, mb, n));
        rec.one_sided.extend(apart);
        let live = |n: &String| live_bytes(ma, n, &e);
        let mut inputs = Vec::new();
        let mut pre_out: Vec<BTreeMap<String, Arc<Snap<S::Buf>>>> = Vec::new();
        for q in 0..ranks {
            let mut v = Vec::new();
            for n in names.iter().filter(|n| live(n) > 0) {
                let (s, _) = chains
                    .snap(a, q, n, live(n))
                    .with_context(|| format!("recording `{n}` at {pname} span {}", span.label()))?;
                v.push((n.clone(), s));
            }
            // a buffer the span writes without reading: its bytes before,
            // so the write-set is exactly the span's
            let mut pre = BTreeMap::new();
            for n in written.iter().filter(|n| live(n) > 0) {
                let s = match v.iter().find(|(i, _)| i == n) {
                    Some((_, s)) => s.clone(),
                    None => chains.snap(a, q, n, live(n))?.0,
                };
                pre.insert(n.clone(), s);
            }
            inputs.push(v);
            pre_out.push(pre);
        }
        // State is compared on the span's write-set — the bytes A's run
        // changed — not on the whole allocation (the rest is other layers'
        // history). Whole-state reads only on the kept run; every other run
        // B starts from A's image of the run.
        let touched: Vec<String> =
            aa.state_writes.union(&ab.state_writes).filter(|s| shared.contains(s)).cloned().collect();
        let mut a_pre: Vec<BTreeMap<String, S::Buf>> = Vec::new();
        if keep {
            for q in 0..ranks {
                a_pre.push(touched.iter().map(|st| Ok((st.clone(), a.save_state(q, st)?))).collect::<Result<_>>()?);
            }
        }
        a.run(pname, &e, span.a.clone())?;
        let (mut pre, mut post) = (Vec::new(), Vec::new());
        for q in 0..ranks {
            let (mut pq, mut oq) = (Runs::new(), Runs::new());
            if keep {
                // the write-set: the blocks the span changed, found where the
                // state is; only those bytes come to the host
                for st in &touched {
                    let len = a.state_bytes(st)?;
                    let img = &a_pre[q][st];
                    let ranges = a.changed(q, At::Scratch(img, 0..len), At::State(st, 0..len))?;
                    let (mut before, mut after) = (Vec::new(), Vec::new());
                    for r in ranges {
                        before.push((r.start, a.bytes(q, img, r.clone())?));
                        after.push((r.start, a.read_state(q, st, r)?));
                    }
                    pq.insert(st.clone(), before);
                    oq.insert(st.clone(), after);
                }
            }
            pre.push(pq);
            post.push(oq);
        }
        let mut ref_out = Vec::new();
        for (q, pre) in pre_out.into_iter().enumerate() {
            let mut outs = BTreeMap::new();
            for (n, pre) in pre {
                let (post, wrote) = chains
                    .snap(a, q, &n, live(&n))
                    .with_context(|| format!("keeping `{n}` after {pname} span {}", span.label()))?;
                outs.insert(n, Out { pre, post, wrote });
            }
            ref_out.push(outs);
        }
        spans.push(SpanRec { span: span.clone(), inputs, ref_out, pre, post });
    }
    a.run(pname, &e, ia..a.calls(pname)?)?;
    let logits_label = if label.starts_with("chunk") { format!("prefill {label}") } else { label.clone() };
    rec.logits.extend(read_logits(a, ma, mb, pname, &e, &logits_label)?);
    let r = rec.runs.len();
    if keep {
        rec.kept.extend((0..spans.len()).map(|s| (r, s)));
    }
    rec.runs.push(Run { program: pname.into(), label, vars: e, tokens, advance, image, spans });
    Ok(())
}
