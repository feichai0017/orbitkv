//! The serving protocol: what a manifest tells a driver about how to call
//! it, read off the manifest's own declarations and settled before any
//! GPU is touched.
//!
//! A manifest speaks to two readers. The runtime reads structure —
//! shapes, `index_into`, dataflow — and executes blindly. A serving loop
//! needs three more facts, and v4 puts each in the manifest rather than in
//! the loop's source: which buffer carries which role in a call (`fill`
//! on inputs and outputs), what shape of call each program accepts
//! (`batch` on programs: `groups` sequences of `rows` rows), and which
//! programs run once after load (`once`). Everything else the loop wants
//! — the var a call's rows go in, which buffers are page tables, which
//! outputs a program produces — follows from those and from structure the
//! runtime already reads, so the loop never names a buffer, a program or
//! a var: [`Protocol::check`] is the one place a name is read, and what
//! it hands out is typed.
//!
//! The axes. A call has `groups` sequences of `rows` rows each on this
//! rank. The var the rows go in is the one the `slot` fill's buffer is
//! over ([`Axis::Rows`]); the var the sequences go in is the `seq_len`
//! buffer's ([`Axis::Groups`]). A manifest whose tray batch spans several
//! ranks (a `tp` group) lays some buffers out over a third var, the whole
//! tray's rows with this rank's first ([`Axis::Tray`]); any fill or line
//! table over a var that is neither of the first two names it. Fixed
//! lengths ([`Axis::Fixed`]) are for what is declared at its bound: the
//! `cu_seqlens` buffer, the collectives' one-word `error` flag.
//!
//! A program with a `batch` is a [`Forward`]. Which tokens it hands back is
//! not a role but dataflow: the `tokens` output it writes (one per
//! sequence, or one per row when the buffer is `[groups, rows]`), and the
//! `count` output it writes, if any, says how many of a sequence's the
//! caller takes. A decode step, a prefill chunk and a speculative round
//! are the same call to the driver — stage `rows` rows per sequence, run,
//! read `tokens` and `count` — which is why there is no role: the shape
//! is the role.
//!
//! One shape rides another: a program whose `batch` names a `span` var
//! takes a decode step's call (`rows` 1) in which one sequence feeds a
//! run of consecutive tokens, a row each, the var set to the run's length
//! and the `span_at` fill to its first row — a prompt chunk in the middle
//! of a batch, for a model whose recurrent kernels take the run as one
//! sequence and whose others see rows like any other. The driver stages a
//! run as that many rows of the sequence (each with its position, slot,
//! length and page-table row) and reads the last row's token.

use crate::types::*;
use crate::Verified;
use std::collections::{BTreeMap, BTreeSet};

/// The axis a caller-facing buffer's first dimension spans.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    /// One entry per row of this rank's call.
    Rows,
    /// One entry per sequence of this rank's call.
    Groups,
    /// One entry per row of the whole tray batch: this rank's rows first,
    /// then the other members' in group order from it.
    Tray,
    /// A fixed length declared at the bound.
    Fixed(u64),
}

/// A var a call is sized by, with its bound.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bound {
    pub var: String,
    pub max: u64,
}

/// A caller-facing buffer: its role, element type and layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Filled {
    pub name: String,
    pub fill: Fill,
    pub dtype: DType,
    pub axis: Axis,
    /// Entries per axis element: `w` for a `[groups, w]` buffer, else 1.
    pub width: u64,
}

impl Filled {
    /// `v` as the buffer's element type, little-endian.
    pub fn encode(&self, v: &[i64]) -> Vec<u8> {
        match self.dtype {
            DType::I32 => v.iter().flat_map(|x| (*x as i32).to_le_bytes()).collect(),
            _ => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        }
    }

    /// The buffer's bytes as values.
    pub fn decode(&self, b: &[u8]) -> Vec<i64> {
        match self.dtype {
            DType::I32 => b.as_chunks::<4>().0.iter().map(|c| i32::from_le_bytes(*c) as i64).collect(),
            _ => b.as_chunks::<8>().0.iter().map(|c| i64::from_le_bytes(*c)).collect(),
        }
    }
}

/// A page table: an `i32 [groups, width]` input indexing a paged state,
/// each sequence's row from its lease.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageTable {
    pub name: String,
    pub width: usize,
}

/// A line table: an `i32 [lines, cols]` or `[lines, cols, width]` input
/// indexing a per-sequence state, its columns the sequences of this rank
/// ([`Axis::Groups`]) or of the whole tray ([`Axis::Tray`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineTable {
    pub name: String,
    pub lines: usize,
    pub width: usize,
    pub axis: Axis,
}

/// Rows per sequence of a call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rows {
    /// Exactly this many, the layout the program's kernels expect.
    Const(u64),
    /// As many as the call feeds: one sequence, the rows var set per call.
    Var,
}

/// A program a serving loop drives: the shape it accepts and what it hands back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Forward {
    pub name: String,
    /// Upper bound on sequences per call.
    pub groups: u64,
    pub rows: Rows,
    /// One sequence of the call feeds a run of rows sized by
    /// [`Protocol::span`]; only a call with a run goes through it.
    pub span: bool,
    /// The `tokens` output it writes, as an index into [`Protocol::fills`];
    /// `None` for a call that only advances state.
    pub emits: Option<usize>,
    /// The `count` output it writes, likewise; without one the caller takes
    /// one token per sequence.
    pub count: Option<usize>,
}

impl Forward {
    /// Whether the program accepts a call of `groups` sequences of `rows`
    /// with no run among them.
    fn accepts(&self, groups: u64, rows: Rows) -> bool {
        !self.span && self.rows == rows && groups <= self.groups
    }
}

/// Every reason a manifest does not fit the protocol, reported together.
#[derive(Debug)]
pub struct ProtocolErrors(pub Vec<String>);

impl std::fmt::Display for ProtocolErrors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("manifest does not fit the serving protocol:")?;
        for e in &self.0 {
            write!(f, "\n  - {e}")?;
        }
        Ok(())
    }
}

impl std::error::Error for ProtocolErrors {}

impl std::ops::Deref for ProtocolErrors {
    type Target = [String];
    fn deref(&self) -> &[String] {
        &self.0
    }
}

/// The manifest's serving contract, checked once and read everywhere.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Protocol {
    /// The var a call's rows on this rank go in.
    pub rows: Bound,
    /// The var a call's sequences go in.
    pub groups: Bound,
    /// The var the tray batch's rows go in, for a manifest whose batch
    /// spans a rank group.
    pub tray: Option<Bound>,
    /// The var a run's length goes in, when some program takes one.
    pub span: Option<Bound>,
    /// Every buffer with a fill, in name order.
    pub fills: Vec<Filled>,
    pub page_tables: Vec<PageTable>,
    pub line_tables: Vec<LineTable>,
    /// Every program with a batch, in name order.
    pub forwards: Vec<Forward>,
    /// Programs run once after load, in name order.
    pub once: Vec<String>,
}

impl Protocol {
    /// Read the contract off a verified manifest, or say everything that
    /// is missing. Verification comes first: the protocol is built on
    /// buffers and programs that are already known to be consistent.
    pub fn check(m: &Verified) -> Result<Protocol, ProtocolErrors> {
        let mut errs = Vec::new();
        let var_max = |v: &str| m.vars.get(v).map(|v| v.max);
        let axis_var = |b: &Buffer| match b.shape.first() {
            Some(Dim::Var(v)) => Some(v.clone()),
            _ => None,
        };
        let one = |fill: Fill| -> Option<(&String, &Buffer)> {
            let mut it = m.buffers.iter().filter(|(_, b)| b.fill == Some(fill));
            let first = it.next();
            if it.next().is_some() {
                None
            } else {
                first
            }
        };

        // The axes come from the two fills every call needs.
        let axis_of = |fill: Fill, errs: &mut Vec<String>| -> Option<Bound> {
            let all: Vec<&String> = m.buffers.iter().filter(|(_, b)| b.fill == Some(fill)).map(|(n, _)| n).collect();
            match all.as_slice() {
                [] => {
                    errs.push(format!("no input has fill `{fill}`"));
                    None
                }
                [name] => {
                    let b = &m.buffers[*name];
                    match (b.shape.as_slice(), axis_var(b).and_then(|v| var_max(&v).map(|max| Bound { var: v, max }))) {
                        ([Dim::Var(_)], Some(bound)) => Some(bound),
                        _ => {
                            errs.push(format!("`{name}` (fill `{fill}`) is shaped {:?}, expected [<var>]", b.shape));
                            None
                        }
                    }
                }
                _ => {
                    errs.push(format!("fill `{fill}` is on {} buffers ({all:?}), expected one", all.len()));
                    None
                }
            }
        };
        let rows = axis_of(Fill::Slot, &mut errs);
        let groups = axis_of(Fill::SeqLen, &mut errs);
        if let (Some(r), Some(g)) = (&rows, &groups) {
            if r.var == g.var {
                errs.push(format!(
                    "`slot` and `seq_len` are both over var `{}`; rows and sequences need their own",
                    r.var
                ));
            }
        }
        let (Some(rows), Some(groups)) = (rows, groups) else {
            return Err(ProtocolErrors(errs));
        };

        // A third var, over which something spans the tray batch.
        let mut tray_vars: BTreeSet<String> = BTreeSet::new();
        for b in m.buffers.values().filter(|b| b.fill.is_some() || is_line_table(m, b)) {
            let col = if is_line_table(m, b) { b.shape.get(1) } else { b.shape.first() };
            if let Some(Dim::Var(v)) = col {
                if *v != rows.var && *v != groups.var {
                    tray_vars.insert(v.clone());
                }
            }
        }
        if tray_vars.len() > 1 {
            errs.push(format!("fills and line tables span {} vars besides rows and sequences ({tray_vars:?}), at most one names the tray batch", tray_vars.len()));
        }
        let tray = tray_vars.first().and_then(|v| var_max(v).map(|max| Bound { var: v.clone(), max }));
        let classify = |v: &str| -> Axis {
            if v == rows.var {
                Axis::Rows
            } else if v == groups.var {
                Axis::Groups
            } else {
                Axis::Tray
            }
        };

        // Fills.
        let mut fills: Vec<Filled> = Vec::new();
        for (name, b) in &m.buffers {
            let Some(fill) = b.fill else { continue };
            let (axis, width) = match b.shape.as_slice() {
                [Dim::Var(v)] => (classify(v), 1),
                [Dim::Const(c)] => (Axis::Fixed(*c), 1),
                [Dim::Var(v), Dim::Const(w)] => (classify(v), *w),
                s => {
                    errs.push(format!(
                        "`{name}` (fill `{fill}`) is shaped {s:?}, expected [<axis>], [<n>] or [<axis>, w]"
                    ));
                    continue;
                }
            };
            let ok = match fill {
                Fill::Token => width == 1 && matches!(axis, Axis::Rows | Axis::Tray | Axis::Groups),
                Fill::Position | Fill::Valid | Fill::Slot => width == 1 && axis == Axis::Rows,
                Fill::SeqLen | Fill::Count => width == 1 && axis == Axis::Groups,
                Fill::CuSeqlens => matches!(axis, Axis::Fixed(n) if n > groups.max),
                Fill::Tokens => axis == Axis::Groups || (axis == Axis::Tray && width == 1),
                Fill::SpanAt | Fill::Error => axis == Axis::Fixed(1) && b.dtype == DType::I32,
                Fill::Blocks => matches!(axis, Axis::Fixed(n) if n >= 2) && b.dtype == DType::I32,
            };
            if !ok {
                errs.push(format!(
                    "`{name}` (fill `{fill}`) is shaped {:?}: {}",
                    b.shape,
                    match fill {
                        Fill::Token => "expected [rows], [tray] (one per row) or [groups] (each sequence's first)",
                        Fill::Position | Fill::Valid | Fill::Slot => "expected [rows]",
                        Fill::SeqLen | Fill::Count => "expected [groups]",
                        Fill::CuSeqlens => "expected [n] with n >= groups + 1",
                        Fill::Tokens => "expected [groups], [groups, w] or [tray]",
                        Fill::SpanAt | Fill::Error => "expected i32 [1]",
                        Fill::Blocks => "expected i32 [members + 1]",
                    }
                ));
                continue;
            }
            fills.push(Filled { name: name.clone(), fill, dtype: b.dtype, axis, width });
        }
        for (fill, axis) in [(Fill::Token, Axis::Rows), (Fill::Token, Axis::Tray)] {
            let n = fills.iter().filter(|f| f.fill == fill && f.axis == axis).count();
            if n > 1 {
                errs.push(format!("fill `{fill}` over {axis:?} is on {n} buffers, expected one"));
            }
        }
        if !fills.iter().any(|f| f.fill == Fill::Token && f.axis != Axis::Groups) {
            errs.push("no input has fill `token` over the rows: nothing carries the tokens a call feeds".into());
        }
        if fills.iter().filter(|f| f.fill == Fill::Token && f.axis == Axis::Groups).count() > 1 {
            errs.push("fill `token` over the sequences is on more than one buffer".into());
        }
        for fill in [Fill::Position, Fill::Valid, Fill::CuSeqlens, Fill::SpanAt, Fill::Blocks, Fill::Count, Fill::Error]
        {
            if one(fill).is_none() && m.buffers.values().any(|b| b.fill == Some(fill)) {
                errs.push(format!("fill `{fill}` is on more than one buffer"));
            }
        }
        if fills.iter().any(|f| f.fill == Fill::Token && f.axis == Axis::Tray) && tray.is_none() {
            errs.push("a `token` fill spans the tray but no var names the tray batch".into());
        }
        if one(Fill::Blocks).is_some() && tray.is_none() {
            errs.push("fill `blocks` without a var naming the tray batch".into());
        }

        // Tables: the inputs indexing a state that have no role of their own
        // (the `slot` fill indexes the paged state too, one slot per row).
        let mut page_tables = Vec::new();
        let mut line_tables = Vec::new();
        for (name, b) in &m.buffers {
            if b.kind != BufferKind::Input || b.fill.is_some() {
                continue;
            }
            let Some(st) = b.domain.as_ref().and_then(|d| d.index_into.as_deref()).and_then(|s| m.states.get(s)) else {
                continue;
            };
            if b.dtype != DType::I32 {
                errs.push(format!("table `{name}` is {}, the runtime hands out i32 page and line indices", b.dtype));
                continue;
            }
            if st.is_per_seq() {
                let (lines, cols, width) = match b.shape.as_slice() {
                    [Dim::Const(l), Dim::Var(c)] => (*l, c, 1),
                    [Dim::Const(l), Dim::Var(c), Dim::Const(w)] => (*l, c, *w),
                    s => {
                        errs.push(format!("line table `{name}` is shaped {s:?}, expected [lines, <groups|tray>] or [lines, <groups|tray>, w]"));
                        continue;
                    }
                };
                let axis = classify(cols);
                if axis == Axis::Rows {
                    errs.push(format!("line table `{name}` has a column per row (`{cols}`); lines are per sequence"));
                    continue;
                }
                line_tables.push(LineTable { name: name.clone(), lines: lines as usize, width: width as usize, axis });
            } else {
                match b.shape.as_slice() {
                    [Dim::Var(g), Dim::Const(w)] if *g == groups.var => {
                        page_tables.push(PageTable { name: name.clone(), width: *w as usize });
                    }
                    s => errs.push(format!("page table `{name}` is shaped {s:?}, expected [<groups>, n]")),
                }
            }
        }

        // Forwards: the shape, and by dataflow what each hands back.
        let mut forwards: Vec<Forward> = Vec::new();
        let mut once = Vec::new();
        let mut span_var: Option<Bound> = None;
        for (pname, p) in &m.programs {
            if p.once {
                once.push(pname.clone());
            }
            let Some(batch) = &p.batch else { continue };
            let ctx = format!("program `{pname}`");
            let rows_of = match &batch.rows {
                Dim::Const(r) => {
                    if batch.groups * r > rows.max {
                        errs.push(format!(
                            "{ctx}: {} sequences of {r} rows exceed the {} rows `{}` allows",
                            batch.groups, rows.max, rows.var
                        ));
                    }
                    Rows::Const(*r)
                }
                Dim::Var(v) => {
                    if *v != rows.var {
                        errs.push(format!(
                            "{ctx}: batch.rows is var `{v}`, but the rows of a call go in `{}`",
                            rows.var
                        ));
                    }
                    if batch.groups != 1 {
                        errs.push(format!("{ctx}: rows set per call means one sequence, not {} groups", batch.groups));
                    }
                    Rows::Var
                }
            };
            if batch.groups > groups.max {
                errs.push(format!("{ctx}: {} groups exceed the {} `{}` allows", batch.groups, groups.max, groups.var));
            }
            let span = batch.span.as_ref().and_then(|v| {
                if rows_of != Rows::Const(1) {
                    let rows = match &batch.rows {
                        Dim::Const(r) => r.to_string(),
                        Dim::Var(v) => v.clone(),
                    };
                    errs.push(format!("{ctx}: a span rides a call of one row per sequence, not `{rows}`"));
                }
                if *v == rows.var || *v == groups.var {
                    errs.push(format!(
                        "{ctx}: batch.span is `{v}`, which sizes the call itself; a run needs its own var"
                    ));
                }
                // A verified manifest's batch.span names a declared var.
                let max = var_max(v).expect("verified: batch.span names a var");
                if max > rows.max {
                    errs.push(format!(
                        "{ctx}: a run of {max} rows (`{v}`) exceeds the {} rows `{}` allows",
                        rows.max, rows.var
                    ));
                    None
                } else {
                    Some(Bound { var: v.clone(), max })
                }
            });
            if let Some(s) = &span {
                match &span_var {
                    Some(other) if *other != *s => errs.push(format!(
                        "{ctx}: batch.span is `{}`, but `{}` sizes a run in another program; one var sizes every run",
                        s.var, other.var
                    )),
                    Some(_) => {}
                    None => span_var = Some(s.clone()),
                }
            }
            let written: BTreeSet<&str> = p
                .calls
                .iter()
                .flat_map(|c| {
                    let op = m.ops.get(&c.op);
                    c.args.iter().enumerate().filter_map(move |(j, a)| match a {
                        Arg::Buf { buf, .. }
                            if op.is_none_or(|op| {
                                matches!(op.params.get(j).and_then(|p| p.dir()), Some(Dir::Out | Dir::InOut) | None)
                            }) =>
                        {
                            Some(buf.as_str())
                        }
                        _ => None,
                    })
                })
                .collect();
            let writes = |fill: Fill| -> Vec<usize> {
                fills
                    .iter()
                    .enumerate()
                    .filter(|(_, f)| f.fill == fill && written.contains(f.name.as_str()))
                    .map(|(i, _)| i)
                    .collect()
            };
            let emits = match writes(Fill::Tokens).as_slice() {
                [] => None,
                [i] => Some(*i),
                many => {
                    errs.push(format!(
                        "{ctx}: writes {} `tokens` outputs ({:?}); a call hands back one",
                        many.len(),
                        many.iter().map(|&i| &fills[i].name).collect::<Vec<_>>()
                    ));
                    None
                }
            };
            if let Some(i) = emits {
                let f = &fills[i];
                let ok = match rows_of {
                    Rows::Const(r) => f.width == 1 || f.width == r,
                    Rows::Var => f.width == 1,
                };
                if !ok {
                    errs.push(format!(
                        "{ctx}: hands back `{}` of {} per sequence, but a call has {} rows per sequence",
                        f.name,
                        f.width,
                        match rows_of {
                            Rows::Const(r) => r.to_string(),
                            Rows::Var => "a variable number of".into(),
                        }
                    ));
                }
            }
            let count = match writes(Fill::Count).as_slice() {
                [] => None,
                [i] => Some(*i),
                _ => unreachable!("one `count` fill"),
            };
            if let Some(c) = count {
                match emits {
                    Some(i) if fills[i].width > 1 => {}
                    _ => errs.push(format!(
                        "{ctx}: writes `{}` (fill `count`) but no `tokens` output of several per sequence to count",
                        fills[c].name
                    )),
                }
            }
            let spanned = batch.span.is_some();
            if let Some(same) =
                forwards.iter().find(|f| f.groups == batch.groups && f.rows == rows_of && f.span == spanned)
            {
                errs.push(format!("{ctx} and `{}` accept the same call shape; a shape names one program", same.name));
            }
            forwards.push(Forward {
                name: pname.clone(),
                groups: batch.groups,
                rows: rows_of,
                span: spanned,
                emits,
                count,
            });
        }
        if forwards.is_empty() {
            errs.push("no program declares a `batch`: nothing for a serving loop to drive".into());
        } else if forwards.iter().all(|f| f.emits.is_none()) {
            errs.push("no program with a `batch` writes a `tokens` output: no call hands a token back".into());
        }
        if span_var.is_some() && one(Fill::SpanAt).is_none() {
            errs.push("a program takes a run of rows (batch.span) but no input has fill `span_at`".into());
        }

        if errs.is_empty() {
            Ok(Protocol { rows, groups, tray, span: span_var, fills, page_tables, line_tables, forwards, once })
        } else {
            Err(ProtocolErrors(errs))
        }
    }

    /// The program for a call of `groups` sequences of `rows`: the one with
    /// the tightest bound that accepts it.
    pub fn forward(&self, groups: u64, rows: Rows) -> Option<&Forward> {
        self.forwards.iter().filter(|f| f.accepts(groups, rows)).min_by_key(|f| f.groups)
    }

    /// The program that takes one sequence's rows as fed (a prefill chunk).
    pub fn chunk(&self) -> Option<&Forward> {
        self.forward(1, Rows::Var)
    }

    /// The program for a call of `groups` sequences of one row, one of
    /// them a run: the one with the tightest bound that takes it.
    pub fn spanned(&self, groups: u64) -> Option<&Forward> {
        self.forwards.iter().filter(|f| f.span && groups <= f.groups).min_by_key(|f| f.groups)
    }

    /// Every constant rows-per-sequence some program accepts, ascending.
    pub fn row_shapes(&self) -> Vec<u64> {
        let mut v: Vec<u64> = self
            .forwards
            .iter()
            .filter(|f| !f.span)
            .filter_map(|f| if let Rows::Const(r) = f.rows { Some(r) } else { None })
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Most sequences any program accepts at `rows` per sequence, no run
    /// among them.
    pub fn max_groups(&self, rows: Rows) -> u64 {
        self.forwards.iter().filter(|f| !f.span && f.rows == rows).map(|f| f.groups).max().unwrap_or(0)
    }

    /// The var vars of a call: `b` sequences of `per` rows on this rank,
    /// `tray` rows in the whole tray batch (the sum of its members' blocks;
    /// this rank's `b * per` when it is alone).
    pub fn vars(&self, b: u64, per: u64, tray: u64) -> BTreeMap<String, u64> {
        let mut vars = BTreeMap::from([(self.rows.var.clone(), b * per), (self.groups.var.clone(), b)]);
        if let Some(t) = &self.tray {
            vars.insert(t.var.clone(), tray);
        }
        vars
    }

    /// The buffer carrying `fill` over `axis`, if any.
    pub fn filled(&self, fill: Fill, axis: Axis) -> Option<&Filled> {
        self.fills.iter().find(|f| f.fill == fill && f.axis == axis)
    }

    /// The one buffer carrying `fill`, whatever its axis (roles that sit on
    /// at most one buffer).
    pub fn any(&self, fill: Fill) -> Option<&Filled> {
        self.fills.iter().find(|f| f.fill == fill)
    }

    /// The tokens fed, one per row: over this rank's rows or the tray's.
    pub fn token_rows(&self) -> &Filled {
        self.filled(Fill::Token, Axis::Rows).or_else(|| self.filled(Fill::Token, Axis::Tray)).expect("checked")
    }

    pub fn slots(&self) -> &Filled {
        self.filled(Fill::Slot, Axis::Rows).expect("checked")
    }

    pub fn seq_lens(&self) -> &Filled {
        self.filled(Fill::SeqLen, Axis::Groups).expect("checked")
    }
}

/// An input indexing a per-sequence state.
fn is_line_table(m: &Manifest, b: &Buffer) -> bool {
    b.kind == BufferKind::Input
        && b.domain
            .as_ref()
            .and_then(|d| d.index_into.as_deref())
            .and_then(|s| m.states.get(s))
            .is_some_and(State::is_per_seq)
}
