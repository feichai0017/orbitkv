//! The structural diff of two manifests and the dataflow view of a span:
//! which ops changed, where the programs' call lists differ, what a run of
//! calls reads from outside itself and what it writes. Pure functions of
//! the manifests.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use kern_manifest::types::{Arg, BufferKind, Call, Dim, Dir, Manifest, Op, ParamType};
use serde::Serialize;
use serde_json::json;

use crate::report::{row, spans};
use crate::Vars;

/// A run of calls that differs between the two sides: A's calls `a` and
/// B's calls `b` of the same program, aligned so that everything before
/// and after is shared. Either side may be empty (calls added or
/// removed).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Span {
    pub a: Range<usize>,
    pub b: Range<usize>,
}

impl Span {
    pub fn label(&self) -> String {
        format!("A[{}..{}) B[{}..{})", self.a.start, self.a.end, self.b.start, self.b.end)
    }
}

/// Which ops differ and how.
pub fn op_changes(ma: &Manifest, mb: &Manifest) -> BTreeMap<String, &'static str> {
    let names: BTreeSet<&String> = ma.ops.keys().chain(mb.ops.keys()).collect();
    names
        .into_iter()
        .filter_map(|n| {
            let kind = match (ma.ops.get(n), mb.ops.get(n)) {
                (Some(_), None) => "removed",
                (None, Some(_)) => "added",
                (Some(x), Some(y)) if json!(x.params) != json!(y.params) => "interface",
                (Some(x), Some(y)) if json!(x.imp) != json!(y.imp) => "impl",
                _ => return None,
            };
            Some((n.clone(), kind))
        })
        .collect()
}

/// Align two call lists (LCS over canonical call keys; a call of a changed
/// op never matches across sides) and return the spans, in order. Gaps
/// between spans are matched one to one, so they are the same length on
/// both sides.
pub fn align(pa: &[Call], pb: &[Call], changed: &BTreeMap<String, &str>) -> Vec<Span> {
    let key = |c: &Call, side: &str| {
        if changed.contains_key(&c.op) {
            format!("{}@{side}#{}", c.op, json!(c.args))
        } else {
            format!("{}#{}", c.op, json!(c.args))
        }
    };
    let ka: Vec<String> = pa.iter().map(|c| key(c, "A")).collect();
    let kb: Vec<String> = pb.iter().map(|c| key(c, "B")).collect();
    let (n, m) = (ka.len(), kb.len());
    let mut lcs = vec![vec![0u32; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            lcs[i][j] = if ka[i] == kb[j] { lcs[i + 1][j + 1] + 1 } else { lcs[i + 1][j].max(lcs[i][j + 1]) };
        }
    }
    let mut spans = Vec::new();
    let (mut i, mut j) = (0, 0);
    let (mut ia, mut ib) = (0, 0);
    while i < n && j < m {
        if ka[i] == kb[j] {
            if i > ia || j > ib {
                spans.push(Span { a: ia..i, b: ib..j });
            }
            i += 1;
            j += 1;
            ia = i;
            ib = j;
        } else if lcs[i + 1][j] >= lcs[i][j + 1] {
            i += 1;
        } else {
            j += 1;
        }
    }
    if ia < n || ib < m {
        spans.push(Span { a: ia..n, b: ib..m });
    }
    spans
}

/// What a run of calls touches, by name.
#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub struct Access {
    pub reads: BTreeSet<String>,
    pub writes: BTreeSet<String>,
    pub state_reads: BTreeSet<String>,
    pub state_writes: BTreeSet<String>,
}

/// What a buffer argument stands for: itself, or for a `peer` buffer the
/// exported buffer or state it holds the group's addresses of. A kernel
/// given the peers reads and writes every rank's copy of the target, this
/// rank's included, so the target is what the span touches; the address
/// array itself is a load-time constant.
fn target(m: &Manifest, buf: &str) -> (String, bool) {
    match (&m.buffers[buf].kind, &m.buffers[buf].of) {
        (BufferKind::Peer, Some(of)) => (of.clone(), m.states.contains_key(of)),
        _ => (buf.to_string(), false),
    }
}

pub fn access(m: &Manifest, prog: &str, calls: Range<usize>) -> Access {
    let mut acc = Access::default();
    for c in &m.programs[prog].calls[calls] {
        let op = &m.ops[&c.op];
        for (arg, p) in c.args.iter().zip(&op.params) {
            match (arg, p) {
                (Arg::Buf { buf, .. }, ParamType::Buf { dir, .. }) => {
                    let (name, is_state) = target(m, buf);
                    let peer = name != *buf;
                    let (reads, writes) = if is_state {
                        (&mut acc.state_reads, &mut acc.state_writes)
                    } else {
                        (&mut acc.reads, &mut acc.writes)
                    };
                    if peer || matches!(dir, Dir::In | Dir::InOut) {
                        reads.insert(name.clone());
                    }
                    if peer || matches!(dir, Dir::Out | Dir::InOut) {
                        writes.insert(name);
                    }
                }
                (Arg::State { state, .. }, ParamType::State { dir }) => {
                    if matches!(dir, Dir::In | Dir::InOut) {
                        acc.state_reads.insert(state.clone());
                    }
                    if matches!(dir, Dir::Out | Dir::InOut) {
                        acc.state_writes.insert(state.clone());
                    }
                }
                _ => {}
            }
        }
    }
    acc
}

/// Buffers a run of calls reads before it writes them: what the span
/// consumes from outside.
pub fn frontier_inputs(m: &Manifest, prog: &str, calls: Range<usize>) -> BTreeSet<String> {
    let mut written = BTreeSet::new();
    let mut inputs = BTreeSet::new();
    for c in &m.programs[prog].calls[calls] {
        let op = &m.ops[&c.op];
        for (arg, p) in c.args.iter().zip(&op.params) {
            if let (Arg::Buf { buf, .. }, ParamType::Buf { dir, .. }) = (arg, p) {
                let (name, is_state) = target(m, buf);
                if is_state {
                    continue;
                }
                let peer = name != *buf;
                if (peer || matches!(dir, Dir::In | Dir::InOut)) && !written.contains(&name) {
                    inputs.insert(name.clone());
                }
                if peer || matches!(dir, Dir::Out | Dir::InOut) {
                    written.insert(name);
                }
            }
        }
    }
    inputs
}

/// Buffers the `once` programs write: load-time constants (a packed
/// weight, a rope table), each side's own like its weights, never handed
/// from A to B.
pub fn constants(m: &Manifest, once: &[String]) -> BTreeSet<String> {
    once.iter().flat_map(|p| access(m, p, 0..m.programs[p].calls.len()).writes).collect()
}

/// Elements of a buffer at `vars` (its live prefix).
pub fn live_elems(m: &Manifest, name: &str, vars: &Vars) -> usize {
    m.buffers[name]
        .shape
        .iter()
        .map(|d| match d {
            Dim::Const(c) => *c as usize,
            Dim::Var(s) => vars[s] as usize,
        })
        .product()
}

pub fn live_bytes(m: &Manifest, name: &str, vars: &Vars) -> usize {
    live_elems(m, name, vars) * m.buffers[name].dtype.bytes() as usize
}

/// Elements per row of the leading dimension (1 for a vector).
pub fn row_elems(m: &Manifest, name: &str, vars: &Vars) -> usize {
    m.buffers[name].shape[1..]
        .iter()
        .map(|d| match d {
            Dim::Const(c) => *c as usize,
            Dim::Var(s) => vars[s] as usize,
        })
        .product::<usize>()
        .max(1)
}

#[derive(Serialize, Debug, Clone, PartialEq, Eq)]
pub struct OpDiff {
    pub op: String,
    pub kind: String,
    pub detail: String,
}

#[derive(Serialize, Debug, Clone, PartialEq, Eq)]
pub struct ProgDiff {
    pub program: String,
    pub spans: usize,
    pub calls: usize,
    pub shared: usize,
    /// Each distinct span shape: `ops   reads → writes`, with its count.
    pub shape: Vec<String>,
}

/// The static diff: what the report's first lines say, and the spans the
/// rest of the harness works on.
#[derive(Serialize, Debug, Clone, Default, PartialEq, Eq)]
pub struct Diff {
    pub ops: Vec<OpDiff>,
    pub buffers: Vec<OpDiff>,
    pub programs: Vec<ProgDiff>,
    /// Programs on one side only.
    pub skipped: Vec<String>,
    /// Some span reads or writes different buffers on the two sides.
    pub frontier_mismatch: bool,
    /// Per program on both sides with at least one span: its spans.
    #[serde(skip)]
    pub spans: BTreeMap<String, Vec<Span>>,
}

fn launch_desc(m: &Manifest, op: &Op) -> String {
    let ls = &op.imp.launches;
    let mut names: Vec<String> = ls
        .iter()
        .map(|l| match l.module().and_then(|n| m.modules.get(n)) {
            Some(md) => md.source.rsplit('/').next().unwrap_or(&md.source).to_string(),
            None => l.entry().to_string(),
        })
        .collect();
    names.dedup();
    let n = names.join(" + ");
    if ls.len() > 1 {
        format!("{} launches: {n}", ls.len())
    } else {
        n
    }
}

/// Diff A against B.
pub fn diff(ma: &Manifest, mb: &Manifest) -> Diff {
    let changed = op_changes(ma, mb);
    let mut d = Diff::default();
    for (k, kind) in &changed {
        let detail = match *kind {
            "interface" => format!("{} → {} params", ma.ops[k].params.len(), mb.ops[k].params.len()),
            "impl" => format!("{} → {}", launch_desc(ma, &ma.ops[k]), launch_desc(mb, &mb.ops[k])),
            "added" => launch_desc(mb, &mb.ops[k]),
            _ => launch_desc(ma, &ma.ops[k]),
        };
        d.ops.push(OpDiff { op: k.clone(), kind: kind.to_string(), detail });
    }
    for name in ma.buffers.keys().chain(mb.buffers.keys()).collect::<BTreeSet<_>>() {
        let what = match (ma.buffers.get(name), mb.buffers.get(name)) {
            (Some(x), Some(y)) if json!(x) != json!(y) => "changed",
            (Some(_), None) => "removed",
            (None, Some(_)) => "added",
            _ => continue,
        };
        d.buffers.push(OpDiff { op: name.clone(), kind: what.into(), detail: String::new() });
    }
    for (pname, pa) in &ma.programs {
        let Some(pb) = mb.programs.get(pname) else {
            d.skipped.push(format!("`{pname}` only in A"));
            continue;
        };
        let spans = align(&pa.calls, &pb.calls, &changed);
        if spans.is_empty() {
            continue;
        }
        // Group spans by shape: (A ops, B ops, reads, writes).
        let mut groups: BTreeMap<(String, String, String, String), usize> = BTreeMap::new();
        for s in &spans {
            let ka = pa.calls[s.a.clone()].iter().map(|c| c.op.as_str()).collect::<Vec<_>>().join("+");
            let kb = pb.calls[s.b.clone()].iter().map(|c| c.op.as_str()).collect::<Vec<_>>().join("+");
            let ia = frontier_inputs(ma, pname, s.a.clone());
            let ib = frontier_inputs(mb, pname, s.b.clone());
            let wa = access(ma, pname, s.a.clone()).writes;
            let wb = access(mb, pname, s.b.clone()).writes;
            if ia != ib || wa != wb {
                d.frontier_mismatch = true;
            }
            // Weights are what makes one layer's span differ from the
            // next's; the shape is the same, so they stay out of the key.
            let reads =
                ia.iter().filter(|n| ma.buffers[*n].kind != BufferKind::Weight).cloned().collect::<Vec<_>>().join(", ");
            let writes = wa.iter().cloned().collect::<Vec<_>>().join(", ");
            *groups.entry((ka, kb, reads, writes)).or_default() += 1;
        }
        let shared = pa.calls.len() - spans.iter().map(|s| s.a.len()).sum::<usize>();
        let shape = groups
            .iter()
            .map(|((ka, kb, reads, writes), n)| {
                let what = if ka == kb {
                    ka.clone()
                } else if ka.is_empty() {
                    format!("∅ → {kb}")
                } else if kb.is_empty() {
                    format!("{ka} → ∅")
                } else {
                    format!("{ka} → {kb}")
                };
                let count = if groups.len() > 1 { format!("{n}× ") } else { String::new() };
                format!("{count}{what} {reads} → {writes}")
            })
            .collect();
        d.programs.push(ProgDiff { program: pname.clone(), spans: spans.len(), calls: pa.calls.len(), shared, shape });
        d.spans.insert(pname.clone(), spans);
    }
    for pname in mb.programs.keys() {
        if !ma.programs.contains_key(pname) {
            d.skipped.push(format!("`{pname}` only in B"));
        }
    }
    d
}

impl Diff {
    pub fn lines(&self) -> Vec<String> {
        let mut v = Vec::new();
        if self.ops.is_empty() && self.buffers.is_empty() {
            v.push(row("diff", "no interface or implementation differs", None));
        }
        v.extend(self.ops.iter().map(|o| row("diff", format!("{} {} {}", o.op, o.kind, o.detail), None)));
        v.extend(self.buffers.iter().map(|b| row("diff", format!("buffer {} {}", b.op, b.kind), None)));
        v.extend(self.programs.iter().map(|p| {
            row(
                "diff",
                format!(
                    "{} {} of {} calls ({} shared) · {}",
                    p.program,
                    spans(p.spans),
                    p.calls,
                    p.shared,
                    p.shape.join(" · ")
                ),
                None,
            )
        }));
        v.extend(self.skipped.iter().map(|s| row("diff", format!("{s}, skipped"), None)));
        if self.frontier_mismatch {
            v.push(row(
                "diff",
                "⚠ some spans read or write different buffers on the two sides — not a span-internal replacement",
                None,
            ));
        }
        v
    }
}
