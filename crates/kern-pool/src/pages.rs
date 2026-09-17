//! Token-slot and sequence-slot ownership of the states: the device tier.
//!
//! The runtime provisions every paged state as `capacity` tokens and every
//! per-sequence state as `seq_slots` slots; the kernels address them
//! through the manifest's tables — inputs whose domain `index_into`s a
//! state: page tables (`stride` tokens per entry) and slot lists (stride 1)
//! over a paged state, line tables (`stride` bytes per line) over a
//! per-sequence one. Which slot holds what is the caller's business, but
//! the only way to name a slot is a [`Lease`]: pages and a sequence slot
//! come out of the pool as a lease, slots, table rows and line indices are
//! computed from it, and everything goes back when it drops. Nothing can
//! free a page twice, free a page it never leased, or address a slot past
//! its lease.
//!
//! Pages are in the runtime's page unit — the lcm of every page table's
//! stride — so one lease serves every paged state at once (a 16-token
//! draft table sees 49 entries per 784-token page of the target's table).
//! A lease is all-or-nothing: a caller takes the pages its worst case
//! needs and holds them, so the pool never fragments.
//!
//! A per-sequence state is `seq_slots` slots of `bytes_per_seq`; slot 0 is
//! never leased (a kernel may read line index 0 as the null line), the
//! rest go one per lease. A line table is shaped `[lines, seqs]` (or
//! `[lines]`): row `r` names, for every sequence of the batch, line `r` of
//! its slot — `slot × lines_per_slot + r`. A wide table `[lines, seqs, w]`
//! has `w` entries per (line, sequence) cell for kernels that take a
//! per-sequence list of lines: the caller puts the line in one of them (the
//! contract of the program says which) and 0, the null line, in the rest.
//!
//! # Checkpoints
//!
//! A [`Checkpoint`] is the first `len` tokens of a sequence kept after the
//! sequence is gone, so a later sequence with the same prefix starts at
//! `len` instead of 0 (see [`crate::store`]). Three rules hold for every
//! operation here: whole pages are shared through the chain, a partial
//! page is copied for whoever newly holds it, and the only writer of a
//! page is a live lease into its own pages. So [`Pool::checkpoint`] shares
//! the lease's whole pages and copies the page `len` ends inside (the
//! lease keeps writing its own), [`Pool::retire`] seals a finished lease's
//! pages in place (nothing is copied: the writer is gone),
//! [`Pool::restore`] and [`Pool::fork`] share whole pages and copy the
//! partial one for the new lease. A paged state alone makes a whole-page
//! checkpoint free — a node, no bytes move — so a caller leaves one at
//! every page boundary; a recurrent state makes any checkpoint cost a slot
//! copy, so a caller leaves one where a request ends (`retire` hands the
//! finished sequence's slot over without a copy) and at explicit
//! breakpoints only.
//!
//! A restored lease's first `len` positions are read-only — the lease
//! refuses to name a slot inside its prefix. The pool decides all of this
//! on the host and returns the byte moves as [`Copies`]; the runtime is the
//! shell that runs them on the stream.
//!
//! # Memory
//!
//! Pages and slots come out of one budget of physical chunks
//! ([`crate::chunks`]): every pooled state is an address range reserved
//! once, a page or a slot exists while its chunks are mapped there, and
//! chunks stay where they were last used. When a lease finds pages (or a
//! slot) short but free objects of the other kind are holding chunks, the
//! pool plans a [`Remap`] that unmakes those and makes what is short, says
//! [`Denied::Remapping`], and the caller asks again once the shell has run
//! the plan and reported it landed ([`Pool::take_pending`],
//! [`Pool::complete`]). One remap at a time. [`Denied::Busy`] is only for
//! when everything is held: something has to go first. So the manifest's
//! `seqs` bound only sizes a step's batch; how many sequences can sleep
//! as checkpoints is the budget's business.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::ops::Range;
use std::sync::{Arc, Mutex};

use kern_manifest::types::{BufferKind, Dim, Manifest};

use crate::chunks::{Chunks, Kind, Remap};
use crate::error::{bail, Result};
use crate::store::{ancestor, chain_pages, lock, sealed, Copies, HostTwin, Node, SlotOwn, Storage, Store};

/// A page table input: `stride` tokens per entry, `width` entries per row.
struct Table {
    stride: u64,
    width: usize,
}

/// A line table input over a per-sequence state: `rows` lines per
/// sequence it names, out of `per_slot` lines a slot holds, `width`
/// entries per (line, sequence) cell.
struct SeqTable {
    rows: usize,
    per_slot: i32,
    width: usize,
}

/// A pooled state: its arena in the chunk pool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pooled {
    pub state: String,
    pub kind: Kind,
    /// Bytes per page or per slot.
    pub object: u64,
    /// Pages or slots the arena is reserved for.
    objects: usize,
    /// Chunk positions reserved.
    pub positions: usize,
}

/// Where an object stands: not backed, free, handed out, or in a remap
/// that makes or unmakes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status {
    Absent,
    Free,
    Held,
    Arriving,
    Leaving,
}

impl Status {
    /// Usable now: a caller can be handed it (free) or holds it.
    fn exists(self) -> bool {
        matches!(self, Status::Free | Status::Held)
    }
}

/// How many objects of one kind exist and how many are handed out, kept
/// as their statuses change. The serving loop reads all four numbers
/// every step, and a pool is millions of pages at a 1M-token context, so
/// counting them by scanning is most of the gap between two steps.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct Tally {
    live: usize,
    held: usize,
}

impl Tally {
    fn moved(&mut self, from: Status, to: Status) {
        fn step(n: &mut usize, was: bool, now: bool) {
            match (was, now) {
                (false, true) => *n += 1,
                (true, false) => *n -= 1,
                _ => {}
            }
        }
        step(&mut self.live, from.exists(), to.exists());
        step(&mut self.held, from == Status::Held, to == Status::Held);
    }
}

/// The accounting behind one mutex: chunks, every page's and slot's
/// status, and the remap not yet landed.
struct Inner {
    chunks: Chunks,
    pages: Vec<Status>,
    page_tally: Tally,
    /// Slot 0 is `Held` for good: the null line.
    slots: Vec<Status>,
    slot_tally: Tally,
    free_pages: BTreeSet<i32>,
    free_slots: BTreeSet<i32>,
    /// Built, not yet taken by the shell.
    pending: Option<Remap>,
    /// Taken, not yet completed.
    in_flight: bool,
    remapped: bool,
}

/// The device tier: shared between the runtime and every live lease and
/// checkpoint so a drop returns pages directly. One caller thread; the
/// mutex is for `Send`, never contended.
pub struct Pool {
    /// Tokens per page.
    unit: u64,
    /// Pages one sequence may hold: what the narrowest table row fits.
    max_pages: usize,
    tables: BTreeMap<String, Table>,
    seq_tables: BTreeMap<String, SeqTable>,
    pooled: Vec<Pooled>,
    inner: Mutex<Inner>,
}

/// The first `len` tokens of a sequence that is gone, on the device. Made
/// by [`Pool::checkpoint`] / [`Pool::retire`] / [`crate::Host::wake`],
/// spent by [`Pool::restore`] and [`crate::Host::park`].
pub type Checkpoint = Store<Pool>;

impl sealed::Sealed for Pool {}

impl Storage for Pool {
    type Page = i32;
    type Slot = i32;
    type Twin = HostTwin;

    fn give_page(&self, page: i32) {
        self.release(&[page], None);
    }

    fn give_slot(&self, slot: i32) {
        self.release(&[], Some(slot));
    }
}

/// The page tables: every input whose domain `index_into`s a paged state
/// (an index a kernel writes — a carry — is the manifest's business, not
/// the host's).
fn tables(m: &Manifest) -> BTreeMap<String, Table> {
    m.buffers
        .iter()
        .filter_map(|(name, b)| {
            if b.kind != BufferKind::Input {
                return None;
            }
            let d = b.domain.as_ref()?;
            if !m.states.get(d.index_into.as_deref()?).is_some_and(|s| !s.is_per_seq()) {
                return None;
            }
            let Some(Dim::Const(width)) = b.shape.last() else { return None };
            Some((name.clone(), Table { stride: d.stride.max(1), width: *width as usize }))
        })
        .collect()
}

/// The line tables: every input whose domain `index_into`s a per-sequence
/// state, shaped `[lines]`, `[lines, seqs]` or `[lines, seqs, w]`.
fn seq_tables(m: &Manifest) -> Result<BTreeMap<String, SeqTable>> {
    let mut out = BTreeMap::new();
    for (name, b) in &m.buffers {
        if b.kind != BufferKind::Input {
            continue;
        }
        let Some(d) = b.domain.as_ref() else { continue };
        let Some(st) = d.index_into.as_deref().and_then(|s| m.states.get(s)) else { continue };
        if !st.is_per_seq() {
            continue;
        }
        let (rows, width) = match b.shape.as_slice() {
            [Dim::Const(rows)] | [Dim::Const(rows), Dim::Var(_)] => (*rows, 1),
            [Dim::Const(rows), Dim::Var(_), Dim::Const(w)] => (*rows, *w as usize),
            s => bail!(
                Manifest,
                "`{name}` indexes a per-sequence state: expected shape [lines], [lines, seqs] or [lines, seqs, w], got {s:?}"
            ),
        };
        let per_slot = st.bytes_per_seq / d.stride.max(1);
        if rows > per_slot {
            bail!(Manifest, "`{name}` names {rows} lines per sequence, the state holds {per_slot}");
        }
        out.insert(name.clone(), SeqTable { rows: rows as usize, per_slot: per_slot as i32, width });
    }
    Ok(out)
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

fn lcm(a: u64, b: u64) -> u64 {
    a / gcd(a, b) * b
}

/// The page unit: the lcm of every page table's stride (a per-sequence
/// state's stride is bytes per line, not tokens: not a page).
pub fn page_unit(m: &Manifest) -> u64 {
    m.buffers
        .values()
        .filter_map(|b| b.domain.as_ref())
        .filter(|d| d.index_into.as_deref().and_then(|s| m.states.get(s)).is_some_and(|s| !s.is_per_seq()))
        .map(|d| d.stride.max(1))
        .fold(1u64, lcm)
}

/// Chunks that hold `tokens` of every paged state and `first_slots` of
/// every per-sequence state. Every pooled state is its own arena of whole
/// chunks, so each rounds up on its own: rounding the sum once leaves the
/// last page or slot of a state short of a chunk.
pub fn chunks_for(m: &Manifest, tokens: u64, first_slots: u64, chunk: u64) -> u64 {
    m.states
        .values()
        .map(|s| (tokens * s.bytes_per_token).div_ceil(chunk) + (first_slots * s.bytes_per_seq).div_ceil(chunk))
        .sum()
}

/// Tokens one sequence can hold, in whole pages of `unit`: what the
/// narrowest page-table row references. `None` when nothing is paged.
pub fn row_tokens(m: &Manifest, unit: u64) -> Option<u64> {
    tables(m).values().map(|t| t.width as u64 * t.stride / unit * unit).min()
}

/// The pooled states of `m` in manifest order, paged ones first, sized for
/// a budget of `chunks` chunks: a state is paged, per-sequence or fixed,
/// never a mix.
fn pooled(m: &Manifest, unit: u64, chunk: u64, chunks: u32, tokens: Option<u64>) -> Result<Vec<Pooled>> {
    let mut paged = Vec::new();
    let mut per_seq = Vec::new();
    for (name, s) in &m.states {
        match (s.bytes_per_token > 0, s.bytes > 0, s.bytes_per_seq > 0) {
            (true, false, false) => paged.push((name.clone(), unit * s.bytes_per_token)),
            (false, false, true) => per_seq.push((name.clone(), s.bytes_per_seq)),
            (false, _, false) => {}
            _ => bail!(
                Manifest,
                "state `{name}` mixes per-token, per-sequence and fixed bytes; a pooled state has one layout"
            ),
        }
    }
    let budget = chunk * chunks as u64;
    let page_bytes: u64 = paged.iter().map(|(_, b)| b).sum();
    let slot_bytes: u64 = per_seq.iter().map(|(_, b)| b).sum();
    // What the chunks hold, or the tokens asked for when that is fewer: a
    // chunk is whole, a capacity is not.
    let pages = budget.checked_div(page_bytes).map_or(chunks as usize, |n| n as usize);
    let pages = tokens.map_or(pages, |t| pages.min((t / unit) as usize));
    let slots = budget.checked_div(slot_bytes).map_or(0, |n| n as usize);
    let entry = |(state, object): (String, u64), kind, objects: usize| Pooled {
        state,
        kind,
        object,
        objects,
        positions: (object * objects as u64).div_ceil(chunk) as usize,
    };
    Ok(paged
        .into_iter()
        .map(|p| entry(p, Kind::Page, pages))
        .chain(per_seq.into_iter().map(|p| entry(p, Kind::Slot, slots)))
        .collect())
}

impl Inner {
    /// The only way a status changes, so no tally can drift from what the
    /// vectors say.
    fn set_page(&mut self, p: usize, to: Status) {
        self.page_tally.moved(self.pages[p], to);
        self.pages[p] = to;
    }

    fn set_slot(&mut self, s: usize, to: Status) {
        self.slot_tally.moved(self.slots[s], to);
        self.slots[s] = to;
    }

    /// Slot 0 is held for good, so it never counts as something a caller
    /// would have to give back.
    fn anything_held(&self) -> bool {
        self.page_tally.held > 0 || self.slot_tally.held > 1
    }

    /// The `n` lowest absent objects of `v` (skipping slot 0 through
    /// `from`).
    fn absent(v: &[Status], from: usize, n: usize) -> Vec<usize> {
        v.iter().enumerate().skip(from).filter(|(_, &s)| s == Status::Absent).map(|(i, _)| i).take(n).collect()
    }

    /// Plan what a request for `need_p` pages and `need_s` slots is
    /// short of, out of free chunks and, when only one kind is short,
    /// chunks taken from the free objects of the other the request does
    /// not need itself — the highest first, so what stays is packed low.
    /// The plan waits in `pending`. `Busy` when what is held would have
    /// to go first, `ExceedsPool` when even then nothing could give.
    fn rebalance(&mut self, need_p: usize, need_s: usize) -> std::result::Result<(), Denied> {
        let never = |me: &Inner| if me.anything_held() { Denied::Busy } else { Denied::ExceedsPool };
        let (free_p, free_s) = (self.free_pages.len(), self.free_slots.len());
        let (d_p, d_s) = (need_p.saturating_sub(free_p), need_s.saturating_sub(free_s));
        let pages = Inner::absent(&self.pages, 0, d_p);
        let slots = Inner::absent(&self.slots, 1, d_s);
        if pages.len() < d_p || slots.len() < d_s {
            return Err(never(self));
        }
        let mut sim = self.chunks.clone();
        let mut plan = Remap::default();
        // Costs counted per object over-count a chunk two targets share:
        // a chunk too many taken, never too few.
        let cost: usize = pages.iter().map(|&p| sim.cost(Kind::Page, p)).sum::<usize>()
            + slots.iter().map(|&s| sim.cost(Kind::Slot, s)).sum::<usize>();
        let mut sources: Vec<(Kind, i32)> = Vec::new();
        let candidates: Vec<(Kind, i32)> = if d_s == 0 {
            self.free_slots.iter().rev().take(free_s - need_s).map(|&s| (Kind::Slot, s)).collect()
        } else if d_p == 0 {
            self.free_pages.iter().rev().take(free_p - need_p).map(|&p| (Kind::Page, p)).collect()
        } else {
            Vec::new()
        };
        for (k, o) in candidates {
            if sim.free() >= cost {
                break;
            }
            sim.unmake(k, o as usize, &mut plan);
            sources.push((k, o));
        }
        if sim.free() < cost {
            return Err(never(self));
        }
        for &p in &pages {
            sim.make(Kind::Page, p, &mut plan);
        }
        for &s in &slots {
            sim.make(Kind::Slot, s, &mut plan);
        }
        self.chunks = sim;
        for (k, o) in sources {
            match k {
                Kind::Page => {
                    self.free_pages.remove(&o);
                    self.set_page(o as usize, Status::Leaving);
                }
                Kind::Slot => {
                    self.free_slots.remove(&o);
                    self.set_slot(o as usize, Status::Leaving);
                }
            }
        }
        for p in pages {
            self.set_page(p, Status::Arriving);
        }
        for s in slots {
            self.set_slot(s, Status::Arriving);
        }
        self.pending = Some(plan);
        self.remapped = true;
        Ok(())
    }
}

impl Pool {
    /// The pool of `m` over `chunks` chunks of `chunk` bytes, in whole
    /// pages of [`page_unit`]: the first `first_slots` sequence slots
    /// (slot 0 among them) exist from the start, then as many pages as the
    /// chunks left hold, at most `tokens` of them when a capacity was
    /// asked for. The [`Remap`] returned makes that initial layout; the
    /// pool already counts it as landed.
    pub fn new(
        m: &Manifest,
        chunk: u64,
        chunks: u32,
        first_slots: usize,
        tokens: Option<u64>,
    ) -> Result<(Pool, Remap)> {
        let unit = page_unit(m);
        let tables = tables(m);
        let pooled = pooled(m, unit, chunk, chunks, tokens)?;
        let arenas: Vec<(Kind, u64, usize)> = pooled.iter().map(|p| (p.kind, p.object, p.objects)).collect();
        let pages_max = pooled.iter().find(|p| p.kind == Kind::Page).map_or(chunks as usize, |p| p.objects);
        let slots_max = pooled.iter().find(|p| p.kind == Kind::Slot).map_or(0, |p| p.objects);
        let max_pages = row_tokens(m, unit).map_or(pages_max, |t| (t / unit) as usize).min(pages_max);
        let mut inner = Inner {
            chunks: Chunks::new(chunk, &arenas, chunks),
            pages: vec![Status::Absent; pages_max],
            page_tally: Tally::default(),
            slots: vec![Status::Absent; slots_max],
            slot_tally: Tally::default(),
            free_pages: BTreeSet::new(),
            free_slots: BTreeSet::new(),
            pending: None,
            in_flight: false,
            remapped: false,
        };
        let mut plan = Remap::default();
        for s in 0..first_slots {
            if s >= slots_max || inner.chunks.cost(Kind::Slot, s) > inner.chunks.free() {
                bail!(Api, "{chunks} chunks of {chunk} bytes hold {s} sequence slots, not the {first_slots} asked for");
            }
            inner.chunks.make(Kind::Slot, s, &mut plan);
            inner.set_slot(s, if s == 0 { Status::Held } else { Status::Free });
            if s > 0 {
                inner.free_slots.insert(s as i32);
            }
        }
        for p in 0..pages_max {
            if inner.chunks.cost(Kind::Page, p) > inner.chunks.free() {
                break;
            }
            inner.chunks.make(Kind::Page, p, &mut plan);
            inner.set_page(p, Status::Free);
            inner.free_pages.insert(p as i32);
        }
        let pool = Pool { unit, max_pages, tables, seq_tables: seq_tables(m)?, pooled, inner: Mutex::new(inner) };
        Ok((pool, plan))
    }

    pub fn unit(&self) -> u64 {
        self.unit
    }

    /// The pooled states, in arena order.
    pub fn pooled(&self) -> &[Pooled] {
        &self.pooled
    }

    pub fn chunk(&self) -> u64 {
        lock(&self.inner).chunks.chunk()
    }

    /// Pages that exist now (free or held).
    pub fn total(&self) -> usize {
        lock(&self.inner).page_tally.live
    }

    /// Pages the arena could hold if every chunk were a page.
    pub fn pages_max(&self) -> usize {
        lock(&self.inner).pages.len()
    }

    /// Pages held by a lease or a checkpoint.
    pub fn used(&self) -> usize {
        lock(&self.inner).page_tally.held
    }

    pub fn max_seq_tokens(&self) -> usize {
        self.max_pages * self.unit as usize
    }

    pub fn tables(&self) -> impl Iterator<Item = &str> {
        self.tables.keys().map(String::as_str)
    }

    /// Whether the manifest has per-sequence states (slots at all).
    pub fn has_slots(&self) -> bool {
        self.pooled.iter().any(|p| p.kind == Kind::Slot)
    }

    /// Sequence slots that exist now, slot 0 among them (0 without
    /// per-sequence states).
    pub fn slots(&self) -> usize {
        lock(&self.inner).slot_tally.live
    }

    /// Slots the arena could hold if every chunk were a slot.
    pub fn slots_max(&self) -> usize {
        lock(&self.inner).slots.len()
    }

    /// Sequence slots held by a lease or a checkpoint, slot 0 aside.
    pub fn slots_used(&self) -> usize {
        lock(&self.inner).slot_tally.held.saturating_sub(1)
    }

    pub fn seq_tables(&self) -> impl Iterator<Item = &str> {
        self.seq_tables.keys().map(String::as_str)
    }

    /// Whether any remap was ever planned: the initial layout is gone.
    pub fn remapped(&self) -> bool {
        lock(&self.inner).remapped
    }

    /// The remap waiting to be executed, if any; from here until
    /// [`Pool::complete`] one is in flight and no other is planned.
    pub fn take_pending(&self) -> Option<Remap> {
        let mut g = lock(&self.inner);
        let plan = g.pending.take()?;
        g.in_flight = true;
        Some(plan)
    }

    /// The remap landed: what it made is free, what it unmade is gone.
    pub fn complete(&self, plan: Remap) {
        let mut g = lock(&self.inner);
        for (k, o) in plan.made {
            match k {
                Kind::Page => {
                    g.set_page(o as usize, Status::Free);
                    g.free_pages.insert(o);
                }
                Kind::Slot => {
                    g.set_slot(o as usize, Status::Free);
                    g.free_slots.insert(o);
                }
            }
        }
        for (k, o) in plan.unmade {
            match k {
                Kind::Page => g.set_page(o as usize, Status::Absent),
                Kind::Slot => g.set_slot(o as usize, Status::Absent),
            }
        }
        g.in_flight = false;
    }

    pub(crate) fn pages_for(&self, tokens: usize) -> std::result::Result<usize, Denied> {
        let need = tokens.div_ceil(self.unit as usize);
        if need > self.max_pages {
            return Err(Denied::ExceedsRow { limit: self.max_seq_tokens() });
        }
        Ok(need)
    }

    /// `fresh` free pages, the lowest, and, when the manifest has
    /// per-sequence states, a slot; all or nothing. Short of either, a
    /// remap is planned when free objects of the other kind can give the
    /// chunks.
    pub(crate) fn take(
        self: &Arc<Pool>,
        fresh: usize,
    ) -> std::result::Result<(Vec<i32>, Option<Arc<SlotOwn<Pool>>>), Denied> {
        let want_slot = self.has_slots();
        let mut g = lock(&self.inner);
        if fresh <= g.free_pages.len() && (!want_slot || !g.free_slots.is_empty()) {
            let taken: Vec<i32> = (0..fresh).map(|_| g.free_pages.pop_first().expect("counted")).collect();
            for &p in &taken {
                g.set_page(p as usize, Status::Held);
            }
            let slot = want_slot.then(|| g.free_slots.pop_first().expect("counted"));
            if let Some(s) = slot {
                g.set_slot(s as usize, Status::Held);
            }
            return Ok((taken, slot.map(|s| SlotOwn::new(s, self))));
        }
        if g.pending.is_some() || g.in_flight {
            return Err(Denied::Remapping);
        }
        g.rebalance(fresh, want_slot as usize)?;
        Err(Denied::Remapping)
    }

    fn release(&self, pages: &[i32], slot: Option<i32>) {
        let mut g = lock(&self.inner);
        for &p in pages {
            debug_assert_eq!(g.pages[p as usize], Status::Held);
            g.set_page(p as usize, Status::Free);
            g.free_pages.insert(p);
        }
        if let Some(s) = slot {
            debug_assert_eq!(g.slots[s as usize], Status::Held);
            g.set_slot(s as usize, Status::Free);
            g.free_slots.insert(s);
        }
    }

    /// A fresh sequence: the pages `tokens` need and a sequence slot.
    pub fn lease(self: &Arc<Pool>, tokens: usize) -> std::result::Result<Lease, Denied> {
        let need = self.pages_for(tokens)?;
        let (pages, slot) = self.take(need)?;
        Ok(Lease { chain: None, shared: 0, pages, slot, prefix: 0, pool: Arc::clone(self) })
    }

    /// A sequence slot alone, no pages: this rank's share of a sequence
    /// whose positions live on another rank (a tensor-parallel peer holds
    /// a slice of every row's recurrent state and none of its KV). Such a
    /// lease names no position; its checkpoints, forks and wakes move the
    /// slot only, at whatever length the caller says.
    pub fn lease_slot(self: &Arc<Pool>) -> std::result::Result<Lease, Denied> {
        let (pages, slot) = self.take(0)?;
        Ok(Lease { chain: None, shared: 0, pages, slot, prefix: 0, pool: Arc::clone(self) })
    }

    /// The first `len` tokens of `lease` as a checkpoint the lease's
    /// sequence keeps running past: its whole pages up to there become
    /// shared, the page `len` ends inside is copied into a page of the
    /// checkpoint's own (the lease keeps writing its own), its state slot
    /// (when the manifest has one) is copied into a fresh slot — the
    /// [`Copies`] say which. `len` is 1 to the lease's tokens (anything
    /// from 1 for a slot-only lease).
    pub fn checkpoint(
        self: &Arc<Pool>,
        lease: &mut Lease,
        len: usize,
    ) -> std::result::Result<(Checkpoint, Copies), Denied> {
        assert!(
            len >= 1 && (!lease.paged() || len <= lease.tokens()),
            "checkpoint of {len} tokens out of a lease of {}",
            lease.tokens()
        );
        let unit = self.unit as usize;
        let (full, tail) = if lease.paged() { (len / unit, len % unit) } else { (0, 0) };
        let (fresh, slot) = self.take((tail > 0) as usize)?;
        let chain = (full > 0).then(|| lease.share(full));
        let (chain, pages) = match fresh.first() {
            Some(&p) => (Some(Node::new(p, chain, self, HostTwin::default())), vec![(lease.pages[full], p)]),
            None => (chain, Vec::new()),
        };
        let copies = Copies { pages, slot: lease.seq_slot().zip(slot.as_ref().map(|s| s.id)) };
        Ok((Store { len, pages: full + (tail > 0) as usize, chain, slot }, copies))
    }

    /// The first `len` tokens of a finished sequence as a checkpoint:
    /// the lease's pages past `len` return, the rest and its state slot
    /// move over as they are. Nothing is copied.
    pub fn retire(self: &Arc<Pool>, mut lease: Lease, len: usize) -> Checkpoint {
        assert!(
            len >= 1 && (!lease.paged() || len <= lease.tokens()),
            "retiring {len} tokens out of a lease of {}",
            lease.tokens()
        );
        let pages = if lease.paged() { len.div_ceil(self.unit as usize) } else { 0 };
        let chain = (pages > 0).then(|| lease.share(pages));
        Store { len, pages, chain, slot: lease.slot.take() }
    }

    /// A sequence continuing from the first `len` tokens of `cp` with room
    /// for `tokens` (more than `len`): those whole pages shared, a copy of
    /// the page `len` ends inside when it does, fresh pages for the rest,
    /// a fresh slot with the checkpoint's state copied in. `len` is the
    /// checkpoint's own length or, for one without a state slot, any
    /// whole number of its pages. The lease names positions from `len` on.
    /// A slot-only checkpoint restores to a slot-only lease at its own
    /// length; `tokens` is not its business.
    pub fn restore(
        self: &Arc<Pool>,
        cp: &Checkpoint,
        len: usize,
        tokens: usize,
    ) -> std::result::Result<(Lease, Copies), Denied> {
        let unit = self.unit as usize;
        let Some(cp_chain) = &cp.chain else {
            assert!(len == cp.len, "restoring {len} tokens of a slot-only checkpoint of {}", cp.len);
            let (_, slot) = self.take(0)?;
            let copies = Copies { pages: Vec::new(), slot: cp.slot_id().zip(slot.as_ref().map(|s| s.id)) };
            let lease = Lease { chain: None, shared: 0, pages: Vec::new(), slot, prefix: len, pool: Arc::clone(self) };
            return Ok((lease, copies));
        };
        assert!(
            len == cp.len || (cp.slot.is_none() && len >= 1 && len < cp.len && len.is_multiple_of(unit)),
            "restoring {len} tokens of a checkpoint of {} ({} slot)",
            cp.len,
            if cp.slot.is_some() { "with a" } else { "no" }
        );
        assert!(tokens > len, "restoring {len} tokens into room for {tokens}");
        let need = self.pages_for(tokens)?;
        let full = len / unit;
        let partial = if len.is_multiple_of(unit) { None } else { Some(cp_chain.page) };
        let (fresh, slot) = self.take(need - full)?;
        // The chain through the whole pages: the node `full` pages deep.
        let chain = ancestor(&cp.chain, cp.pages - full);
        let mut pages = chain_pages(&chain);
        pages.extend_from_slice(&fresh);
        let copies = Copies {
            pages: partial.map(|p| (p, fresh[0])).into_iter().collect(),
            slot: cp.slot_id().zip(slot.as_ref().map(|s| s.id)),
        };
        Ok((Lease { chain, shared: full, pages, slot, prefix: len, pool: Arc::clone(self) }, copies))
    }

    /// A sequence branched off the first `len` tokens of `parent`, which
    /// keeps running, with room for `tokens`: the whole pages shared, the
    /// page `len` ends inside copied, a fresh slot with the parent's state
    /// copied in. A recurrent state is the parent's as of now, so with one
    /// `len` must be the parent's position. The lease names positions from
    /// `len` on. A slot-only parent forks a slot-only child: the state
    /// copied, nothing else.
    pub fn fork(
        self: &Arc<Pool>,
        parent: &mut Lease,
        len: usize,
        tokens: usize,
    ) -> std::result::Result<(Lease, Copies), Denied> {
        if !parent.paged() {
            assert!(len >= 1, "forking a slot-only lease at 0 tokens");
            let (_, slot) = self.take(0)?;
            let copies = Copies { pages: Vec::new(), slot: parent.seq_slot().zip(slot.as_ref().map(|s| s.id)) };
            let lease = Lease { chain: None, shared: 0, pages: Vec::new(), slot, prefix: len, pool: Arc::clone(self) };
            return Ok((lease, copies));
        }
        assert!(len >= 1 && len <= parent.tokens(), "forking {len} tokens out of a lease of {}", parent.tokens());
        assert!(tokens > len, "forking {len} tokens into room for {tokens}");
        let unit = self.unit as usize;
        let need = self.pages_for(tokens)?;
        let full = len / unit;
        let partial = if len.is_multiple_of(unit) { None } else { Some(parent.pages[full]) };
        let (fresh, slot) = self.take(need - full)?;
        let chain = (full > 0).then(|| parent.share(full));
        let mut pages = chain_pages(&chain);
        pages.extend_from_slice(&fresh);
        let copies = Copies {
            pages: partial.map(|p| (p, fresh[0])).into_iter().collect(),
            slot: parent.seq_slot().zip(slot.as_ref().map(|s| s.id)),
        };
        Ok((Lease { chain, shared: full, pages, slot, prefix: len, pool: Arc::clone(self) }, copies))
    }
}

/// Why [`Runtime::lease`](crate::Runtime::lease) said no.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Denied {
    /// More tokens than one page-table row can reference; never fits.
    ExceedsRow { limit: usize },
    /// More pages than the pool has, even empty; never fits.
    ExceedsPool,
    /// Fits, but not right now: pages or sequence slots all held.
    Busy,
    /// Fits once the remap in flight lands; ask again, evict nothing.
    Remapping,
    /// The host tier has no room for the checkpoint; drop a parked one.
    HostFull,
}

impl fmt::Display for Denied {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Denied::ExceedsRow { limit } => write!(f, "longer than a page-table row ({limit} tokens)"),
            Denied::ExceedsPool => write!(f, "no layout of the state budget holds it"),
            Denied::Busy => write!(f, "pages or sequence slots busy"),
            Denied::Remapping => write!(f, "pages or sequence slots being remapped"),
            Denied::HostFull => write!(f, "no room in the host tier"),
        }
    }
}

impl std::error::Error for Denied {}

/// Pages and a sequence slot leased to one sequence: the only handle to
/// its token slots and its per-sequence state, and the only writer of
/// its own pages. Dropping it returns them to the runtime. The first
/// `shared` pages are held through a chain (checkpoints hold them too),
/// the rest are the lease's own. A lease restored from a checkpoint
/// starts with `prefix` positions already filled; it never names a slot
/// inside them.
pub struct Lease {
    chain: Option<Arc<Node<Pool>>>,
    shared: usize,
    /// Every page in order: the chain's, then the lease's own.
    pages: Vec<i32>,
    slot: Option<Arc<SlotOwn<Pool>>>,
    prefix: usize,
    pool: Arc<Pool>,
}

impl Lease {
    pub(crate) fn new(
        chain: Option<Arc<Node<Pool>>>,
        shared: usize,
        pages: Vec<i32>,
        slot: Option<Arc<SlotOwn<Pool>>>,
        prefix: usize,
        pool: &Arc<Pool>,
    ) -> Lease {
        Lease { chain, shared, pages, slot, prefix, pool: Arc::clone(pool) }
    }

    /// The chain through the first `keep` pages (at least 1), moving own
    /// pages into nodes as needed.
    fn share(&mut self, keep: usize) -> Arc<Node<Pool>> {
        while self.shared < keep {
            let page = self.pages[self.shared];
            self.chain = Some(Node::new(page, self.chain.take(), &self.pool, HostTwin::default()));
            self.shared += 1;
        }
        ancestor(&self.chain, self.shared - keep).expect("keep is at least 1")
    }

    /// Pages held.
    pub fn pages(&self) -> usize {
        self.pages.len()
    }

    /// Whether the lease holds pages at all (a slot-only lease names no
    /// position).
    pub fn paged(&self) -> bool {
        !self.pages.is_empty()
    }

    /// The page ids, in position order (a harness reading a restored
    /// prefix back; programs get them through `extend_row`).
    pub fn page_ids(&self) -> &[i32] {
        &self.pages
    }

    /// Token slots held (whole pages, so at least what was asked for).
    pub fn tokens(&self) -> usize {
        self.pages.len() * self.pool.unit as usize
    }

    /// Positions already filled when the lease was handed out: 0 for a
    /// fresh sequence, the checkpoint's length for a restored one.
    pub fn prefix(&self) -> usize {
        self.prefix
    }

    /// The token slot of position `pos` of the sequence, to write into;
    /// `pos` is past the shared prefix and past every page moved into a
    /// chain: a node's page is never written after the node exists, and
    /// the lease that sealed it is no exception.
    pub fn slot(&self, pos: usize) -> i64 {
        assert!(pos >= self.prefix, "position {pos} is inside the shared prefix of {} tokens", self.prefix);
        let unit = self.pool.unit as usize;
        assert!(pos >= self.shared * unit, "position {pos} is inside a sealed page ({} pages sealed)", self.shared);
        let page = *self.pages.get(pos / unit).expect("position past the lease") as i64;
        page * unit as i64 + (pos % unit) as i64
    }

    /// The token slots of consecutive positions (a `slot_mapping` list).
    pub fn slots(&self, positions: Range<usize>) -> Vec<i64> {
        positions.map(|p| self.slot(p)).collect()
    }

    /// Append the sequence's row of page table `table`: one entry per
    /// `stride` tokens of every page held, then the first entry repeated to
    /// the row's width (entries past the sequence length are never
    /// dereferenced, but the domain wants valid page ids in them).
    pub fn extend_row(&self, table: &str, out: &mut Vec<i32>) -> Result<()> {
        let Some(t) = self.pool.tables.get(table) else {
            bail!(Api, "`{table}` is not a page table of this manifest");
        };
        let per_page = (self.pool.unit / t.stride) as i32;
        let start = out.len();
        for &p in &self.pages {
            out.extend((0..per_page).map(|k| p * per_page + k));
        }
        let fill = out.get(start).copied().unwrap_or(0);
        out.resize(start + t.width, fill);
        Ok(())
    }

    /// The sequence slot held in every per-sequence state (`None` when
    /// the manifest has none). Slot 0 is never handed out.
    pub fn seq_slot(&self) -> Option<i32> {
        self.slot.as_ref().map(|s| s.id)
    }

    /// Byte range of this sequence's slot in a per-sequence state of
    /// `bytes_per_seq`.
    pub fn seq_bytes(&self, bytes_per_seq: u64) -> Option<Range<usize>> {
        let s = self.seq_slot()? as u64;
        Some((s * bytes_per_seq) as usize..((s + 1) * bytes_per_seq) as usize)
    }

    /// Line `row` of this sequence in line table `table`: the index its
    /// entry `[row, i]` holds when the sequence is column `i` of a batch.
    pub fn seq_line(&self, table: &str, row: usize) -> Result<i32> {
        let Some(t) = self.pool.seq_tables.get(table) else {
            bail!(Api, "`{table}` is not a line table of this manifest");
        };
        if row >= t.rows {
            bail!(Api, "`{table}` has {} lines per sequence, asked for line {row}", t.rows);
        }
        let Some(slot) = self.seq_slot() else {
            bail!(Api, "lease holds no sequence slot");
        };
        Ok(slot * t.per_slot + row as i32)
    }

    /// Lines per sequence line table `table` names.
    pub fn seq_lines(&self, table: &str) -> Result<usize> {
        match self.pool.seq_tables.get(table) {
            Some(t) => Ok(t.rows),
            None => bail!(Api, "`{table}` is not a line table of this manifest"),
        }
    }

    /// Entries per (line, sequence) cell of line table `table`: 1, or the
    /// `w` of a wide `[lines, seqs, w]` table.
    pub fn seq_width(&self, table: &str) -> Result<usize> {
        match self.pool.seq_tables.get(table) {
            Some(t) => Ok(t.width),
            None => bail!(Api, "`{table}` is not a line table of this manifest"),
        }
    }
}

impl fmt::Debug for Lease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Lease({} pages, {} shared", self.pages.len(), self.shared)?;
        if let Some(s) = self.seq_slot() {
            write!(f, ", slot {s}")?;
        }
        write!(f, ", prefix {})", self.prefix)
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        self.pool.release(&self.pages[self.shared..], None);
    }
}

impl Checkpoint {
    /// The sequence slot holding the state, when the manifest has one.
    pub fn seq_slot(&self) -> Option<i32> {
        self.slot_id()
    }

    /// The page ids, root first (a harness reading a checkpoint back).
    pub fn page_ids(&self) -> Vec<i32> {
        chain_pages(&self.chain)
    }
}
