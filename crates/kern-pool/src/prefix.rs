//! The prefix index: which prefixes of past sequences are kept, on the
//! device or parked on the host, and the longest one a new prompt can
//! start from.
//!
//! The index is a radix tree over tokens. An edge is a run of tokens, a
//! node is a prefix length, and an entry sits on the node whose path
//! spells the entry's tokens: a checkpoint (`R`) on the device, a parked
//! copy (`P`) on the host, or both. The tree only finds; the bytes and
//! what they share are the entries' own business ([`crate::Store`]), so
//! the tree may branch at any token while no page is ever split. A lookup
//! walks the prompt (never its last token: that one must still go through
//! a program) and takes the longest usable entry: one with a recurrent
//! state is usable at its own length only (the state is the state after
//! exactly those tokens); one without is usable at its own length when the
//! prompt covers it, and at any whole page of the tokens the prompt shares
//! with it otherwise. Same tokens, same path: a prefix two sessions both
//! typed is one entry.
//!
//! Room is the caller's answer to a `Busy` lease: [`Prefix::evict`]
//! moves the resident entry hit least recently to the host through the
//! caller's copy, dropping the coldest parked ones until it fits, or
//! drops it when there is no host tier; dropping is all that frees
//! anything (pages shared with a live sequence or another entry stay).
//! Recency is a counter, not a clock. A hit touches every entry on the
//! prompt's path, the deepest coldest, so a chain ages together and its
//! leaf goes first. A stateless resident entry made one page past
//! another on the same path replaces it: a sequence checkpointing every
//! page keeps one entry that grows.
//!
//! No clock, no hash, no hash map: a replay makes the same decisions in
//! the same order.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::host::Parked;
use crate::pages::Checkpoint;
use crate::store::{Storage, Store};

/// Where an entry's bytes are: on the device, or on the host alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier {
    Resident,
    Parked,
}

/// What the index needs to know of what it keeps: a checkpoint, a parked
/// one, or a caller's bundle of several (a tensor-parallel tray's
/// per-rank checkpoints of one sequence). A clone is another holder of
/// the same bytes.
pub trait Kept: Clone {
    /// Tokens held; never 0.
    fn tokens(&self) -> usize;
    /// Whether a recurrent state is held, which pins the entry to its
    /// exact length.
    fn has_slot(&self) -> bool;
}

impl<T: Storage> Kept for Store<T> {
    fn tokens(&self) -> usize {
        Store::tokens(self)
    }

    fn has_slot(&self) -> bool {
        Store::has_slot(self)
    }
}

/// What a lookup found: a holder of the entry, so it stays usable
/// whatever the index does to the entry afterwards.
#[derive(Debug, Clone)]
pub enum Found<R, P> {
    Resident(R),
    Parked(P),
}

/// An entry a prompt can continue from: its first `len` tokens are
/// already in `found`.
#[derive(Debug, Clone)]
pub struct Hit<R, P> {
    pub len: usize,
    pub found: Found<R, P>,
}

/// What [`Prefix::evict`] did to the coldest resident entry, whose
/// tokens it names: parked it, or dropped it; either after dropping
/// `dropped` cold parked entries for the room.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Evicted {
    Parked { key: Arc<[i64]>, dropped: usize },
    Dropped { key: Arc<[i64]>, dropped: usize },
}

struct Entry<R, P> {
    key: Arc<[i64]>,
    device: Option<R>,
    host: Option<P>,
    slot: bool,
    used: u64,
}

impl<R: Clone, P: Clone> Entry<R, P> {
    fn tier(&self) -> Tier {
        if self.device.is_some() {
            Tier::Resident
        } else {
            Tier::Parked
        }
    }

    fn found(&self) -> Found<R, P> {
        match (&self.device, &self.host) {
            (Some(r), _) => Found::Resident(r.clone()),
            (None, Some(p)) => Found::Parked(p.clone()),
            (None, None) => unreachable!("an entry holds something"),
        }
    }
}

struct Node<R, P> {
    /// The edge from the parent: never empty except at the root.
    tokens: Vec<i64>,
    /// By the first token of the child's edge.
    children: BTreeMap<i64, Node<R, P>>,
    entry: Option<Entry<R, P>>,
    /// Stateless entries in this subtree, this node's included, and
    /// those of them resident: a prompt diverging inside an entry is
    /// usable at the whole pages it shares with any of them, a resident
    /// one first.
    paged_below: usize,
    resident_below: usize,
}

impl<R, P> Node<R, P> {
    fn new(tokens: Vec<i64>) -> Node<R, P> {
        Node { tokens, children: BTreeMap::new(), entry: None, paged_below: 0, resident_below: 0 }
    }

    fn below(&self, resident: bool) -> usize {
        if resident {
            self.resident_below
        } else {
            self.paged_below
        }
    }
}

fn lcp(a: &[i64], b: &[i64]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// The node at `key`, made if absent (splitting the edge it falls inside).
fn at_mut<'a, R, P>(root: &'a mut Node<R, P>, key: &[i64]) -> &'a mut Node<R, P> {
    let mut node = root;
    let mut pos = 0;
    loop {
        if pos == key.len() {
            return node;
        }
        let first = key[pos];
        let fresh = !node.children.contains_key(&first);
        let child = node.children.entry(first).or_insert_with(|| Node::new(key[pos..].to_vec()));
        if fresh {
            return child;
        }
        let common = lcp(&child.tokens, &key[pos..]);
        if common < child.tokens.len() {
            let rest = child.tokens.split_off(common);
            let tail = Node {
                tokens: rest,
                children: std::mem::take(&mut child.children),
                entry: child.entry.take(),
                paged_below: child.paged_below,
                resident_below: child.resident_below,
            };
            child.children.insert(tail.tokens[0], tail);
        }
        pos += common;
        node = child;
    }
}

/// The node `key` tokens down a path the tree spells exactly.
fn path_mut<'a, R, P>(root: &'a mut Node<R, P>, key: &[i64]) -> &'a mut Node<R, P> {
    let mut node = root;
    let mut pos = 0;
    while pos < key.len() {
        let child = node.children.get_mut(&key[pos]).expect("on the path");
        pos += child.tokens.len();
        node = child;
    }
    node
}

/// The node at `key`, if the tree has one.
fn at<'a, R, P>(root: &'a Node<R, P>, key: &[i64]) -> Option<&'a Node<R, P>> {
    let mut node = root;
    let mut pos = 0;
    while pos < key.len() {
        let child = node.children.get(&key[pos])?;
        let n = child.tokens.len();
        if key.len() - pos < n || child.tokens[..] != key[pos..pos + n] {
            return None;
        }
        pos += n;
        node = child;
    }
    Some(node)
}

/// Add `paged` and `resident` to the subtree counts of every node from
/// the root to `key`.
fn bump<R, P>(root: &mut Node<R, P>, key: &[i64], paged: isize, resident: isize) {
    let mut node = root;
    let mut pos = 0;
    loop {
        node.paged_below = (node.paged_below as isize + paged) as usize;
        node.resident_below = (node.resident_below as isize + resident) as usize;
        if pos == key.len() {
            return;
        }
        let child = node.children.get_mut(&key[pos]).expect("path exists");
        pos += child.tokens.len();
        node = child;
    }
}

/// Take the entry at `key` out, pruning the node it leaves empty and
/// merging a node left with one child into it.
fn take_entry<R, P>(node: &mut Node<R, P>, key: &[i64]) -> Option<Entry<R, P>> {
    let removed = if key.is_empty() {
        node.entry.take()?
    } else {
        let first = key[0];
        let child = node.children.get_mut(&first)?;
        let n = child.tokens.len();
        if key.len() < n || child.tokens[..] != key[..n] {
            return None;
        }
        let removed = take_entry(child, &key[n..])?;
        if child.entry.is_none() {
            if child.children.is_empty() {
                node.children.remove(&first);
            } else if child.children.len() == 1 {
                let (_, grand) = child.children.pop_first().expect("one child");
                child.tokens.extend(grand.tokens);
                child.children = grand.children;
                child.entry = grand.entry;
            }
        }
        removed
    };
    if !removed.slot {
        node.paged_below -= 1;
        node.resident_below -= removed.device.is_some() as usize;
    }
    Some(removed)
}

/// A stateless entry somewhere in `node`'s subtree (a resident one when
/// `resident`), the node's own first, then the first child's in token
/// order; `None` when there is none.
fn paged_in_mut<R, P>(node: &mut Node<R, P>, resident: bool) -> Option<&mut Node<R, P>> {
    let mut cur = node;
    loop {
        if cur.entry.as_ref().is_some_and(|e| !e.slot && (!resident || e.device.is_some())) {
            return Some(cur);
        }
        cur = cur.children.values_mut().find(|c| c.below(resident) > 0)?;
    }
}

pub struct Prefix<R = Checkpoint, P = Parked> {
    unit: usize,
    root: Node<R, P>,
    /// Recency stamp → entry key, per tier: the eviction order.
    lru: [BTreeMap<u64, Arc<[i64]>>; 2],
    count: [usize; 2],
    clock: u64,
}

fn slot_of(t: Tier) -> usize {
    t as usize
}

impl<R: Kept, P: Kept> Prefix<R, P> {
    /// An index over sequences paged in `unit` tokens.
    pub fn new(unit: usize) -> Prefix<R, P> {
        assert!(unit >= 1);
        Prefix { unit, root: Node::new(Vec::new()), lru: [BTreeMap::new(), BTreeMap::new()], count: [0, 0], clock: 0 }
    }

    /// Entries in both tiers.
    pub fn entries(&self) -> usize {
        self.count[0] + self.count[1]
    }

    /// Entries in a tier.
    pub fn count(&self, tier: Tier) -> usize {
        self.count[slot_of(tier)]
    }

    /// Restamp `e` with `tick`, in its tier's order.
    fn restamp(lru: &mut [BTreeMap<u64, Arc<[i64]>>; 2], e: &mut Entry<R, P>, tick: u64) {
        let order = &mut lru[slot_of(e.tier())];
        order.remove(&e.used);
        e.used = tick;
        order.insert(tick, Arc::clone(&e.key));
    }

    /// Keep `r`, whose tokens are `key`. The same tokens are here already:
    /// a parked entry becomes resident too, a resident one counts as used
    /// and the new one is dropped. A stateless entry one page past a
    /// resident stateless one on the same path replaces it.
    pub fn insert(&mut self, key: &[i64], r: R) {
        assert_eq!(key.len(), r.tokens(), "an entry's key is its tokens");
        assert!(!key.is_empty(), "an entry holds at least one token");
        let slot = r.has_slot();
        self.clock += 1;
        let tick = self.clock;
        let node = at_mut(&mut self.root, key);
        match node.entry.as_mut() {
            Some(e) => {
                assert_eq!(e.slot, slot, "an entry's state is its tokens'");
                let was = e.tier();
                if e.device.is_none() {
                    e.device = Some(r);
                    self.count[slot_of(Tier::Parked)] -= 1;
                    self.count[slot_of(Tier::Resident)] += 1;
                    self.lru[slot_of(was)].remove(&e.used);
                    e.used = tick;
                    self.lru[slot_of(Tier::Resident)].insert(tick, Arc::clone(&e.key));
                    if !slot {
                        bump(&mut self.root, key, 0, 1);
                    }
                } else {
                    Self::restamp(&mut self.lru, e, tick);
                }
            }
            None => {
                let key: Arc<[i64]> = Arc::from(key);
                self.lru[slot_of(Tier::Resident)].insert(tick, Arc::clone(&key));
                self.count[slot_of(Tier::Resident)] += 1;
                node.entry = Some(Entry { key: Arc::clone(&key), device: Some(r), host: None, slot, used: tick });
                if !slot {
                    bump(&mut self.root, &key, 1, 1);
                }
            }
        }
        if !slot && key.len() > self.unit && key.len().is_multiple_of(self.unit) {
            let shallow = &key[..key.len() - self.unit];
            let redundant = at(&self.root, shallow)
                .and_then(|n| n.entry.as_ref())
                .is_some_and(|e| !e.slot && e.device.is_some() && e.host.is_none());
            if redundant {
                self.remove(shallow);
            }
        }
    }

    /// The longest entry usable for `tokens` (the last token is never
    /// covered): a resident one before a parked one of the same length.
    /// A hit touches it and every entry on the prompt's path above it.
    pub fn lookup(&mut self, tokens: &[i64]) -> Option<Hit<R, P>> {
        let usable = tokens.len().checked_sub(1)?;
        let q = &tokens[..usable];
        let unit = self.unit;
        // How far the tree spells the prompt: `pos` is the deepest node on
        // its path, `matched` how many tokens it shares (into the edge
        // below `pos` when more than `pos`).
        let (pos, matched) = {
            let mut node = &self.root;
            let mut pos = 0;
            loop {
                if pos == usable {
                    break (pos, pos);
                }
                let Some(child) = node.children.get(&q[pos]) else { break (pos, pos) };
                let common = lcp(&child.tokens, &q[pos..]);
                if common < child.tokens.len() {
                    break (pos, pos + common);
                }
                pos += common;
                node = child;
            }
        };
        // Ticks for the path, root highest, so a chain ages together and
        // its leaf goes first; the clock skips the ticks not handed out.
        let base = self.clock + usable as u64 + 2;
        self.clock = base;
        let mut i = 0u64;
        let Prefix { root, lru, .. } = self;
        let mut best: Option<(usize, std::cmp::Reverse<Tier>, Found<R, P>)> = None;
        let mut offer = |len: usize, e: &Entry<R, P>| {
            let cand = (len, std::cmp::Reverse(e.tier()));
            if len > 0 && best.as_ref().is_none_or(|(l, t, _)| cand > (*l, *t)) {
                best = Some((len, std::cmp::Reverse(e.tier()), e.found()));
            }
        };
        // A stateless entry off the path is usable at the whole pages the
        // prompt shares with it: `at` tokens through a sibling of the
        // path at depth `at`, `matched` through the edge the prompt
        // continues into below `pos`. The longest wins, a resident one
        // before a parked one at the same length; one sharing no whole
        // page serves nothing and is not touched.
        let mut cands: Vec<(usize, usize, i64, bool)> = Vec::new();
        let mut node = &mut *root;
        let mut at = 0;
        loop {
            if let Some(e) = node.entry.as_mut() {
                Self::restamp(lru, e, base - i);
                i += 1;
                offer(at, e);
            }
            let next = if at < pos { Some(q[at]) } else { (matched > pos).then(|| q[pos]) };
            for resident in [true, false] {
                let sibling = node.children.iter().find(|(&k, c)| c.below(resident) > 0 && next != Some(k));
                if let Some((&k, _)) = sibling {
                    cands.push((at, at, k, resident));
                    break;
                }
            }
            if at == pos {
                if let Some((k, c)) = next.and_then(|k| node.children.get(&k).map(|c| (k, c))) {
                    if c.paged_below > 0 {
                        cands.push((matched, at, k, c.resident_below > 0));
                    }
                }
                break;
            }
            let child = node.children.get_mut(&q[at]).expect("on the path");
            at += child.tokens.len();
            node = child;
        }
        let usable = cands.iter().filter(|&&(s, ..)| s >= unit);
        if let Some(&(shared, depth, k, resident)) = usable.max_by_key(|&&(s, _, _, r)| (s / unit * unit, r)) {
            let parent = path_mut(root, &q[..depth]);
            let n = paged_in_mut(parent.children.get_mut(&k).expect("a child"), resident).expect("counted");
            let e = n.entry.as_mut().expect("a stateless entry");
            Self::restamp(lru, e, base - i);
            offer(shared / unit * unit, e);
        }
        let (len, _, found) = best?;
        Some(Hit { len, found })
    }

    /// The key of the entry in `tier` used least recently.
    fn coldest(&self, tier: Tier) -> Option<Arc<[i64]>> {
        self.lru[slot_of(tier)].values().next().cloned()
    }

    /// Room for a `Busy` lease: the resident entry hit least recently
    /// goes to the host through `park` (the runtime's copy, `Ok(Err(r))`
    /// handing the checkpoint back when the host is full), the parked
    /// entries hit least recently dropped until it fits; without a host
    /// tier (`park` is `None`), or once nothing parked is left to drop,
    /// it is dropped. `None` when nothing is resident. An entry on the
    /// host already just lets its device copy go. On an error the entry
    /// is gone and the error is the caller's.
    pub fn evict<E>(
        &mut self,
        park: Option<impl FnMut(R) -> std::result::Result<std::result::Result<P, R>, E>>,
    ) -> std::result::Result<Option<Evicted>, E> {
        let Some(key) = self.coldest(Tier::Resident) else { return Ok(None) };
        let mut dropped = 0;
        if let Some(mut park) = park {
            loop {
                if self.park(&key, &mut park)? {
                    return Ok(Some(Evicted::Parked { key, dropped }));
                }
                match self.coldest(Tier::Parked) {
                    Some(c) => {
                        self.remove(&c);
                        dropped += 1;
                    }
                    None => break,
                }
            }
        }
        self.remove(&key);
        Ok(Some(Evicted::Dropped { key, dropped }))
    }

    /// `true` when the entry at `key` is parked, `false` when `park`
    /// handed the checkpoint back and the entry stays resident.
    fn park<E>(
        &mut self,
        key: &[i64],
        park: &mut impl FnMut(R) -> std::result::Result<std::result::Result<P, R>, E>,
    ) -> std::result::Result<bool, E> {
        let node = at_mut(&mut self.root, key);
        let e = node.entry.as_mut().expect("entry");
        let cp = e.device.clone().expect("resident");
        let parked = if e.host.is_some() { Ok(Ok(None)) } else { park(cp).map(|r| r.map(Some)) };
        match parked {
            Ok(Ok(p)) => {
                if let Some(p) = p {
                    assert_eq!(p.tokens(), key.len(), "parking an entry of {} tokens", key.len());
                    e.host = Some(p);
                }
                e.device = None;
                self.lru[slot_of(Tier::Resident)].remove(&e.used);
                self.lru[slot_of(Tier::Parked)].insert(e.used, Arc::clone(&e.key));
                self.count[slot_of(Tier::Resident)] -= 1;
                self.count[slot_of(Tier::Parked)] += 1;
                if !e.slot {
                    bump(&mut self.root, key, 0, -1);
                }
                Ok(true)
            }
            Ok(Err(_)) => Ok(false),
            Err(err) => {
                self.remove(key);
                Err(err)
            }
        }
    }

    /// Drop the entry at `key`; `false` when there is none.
    fn remove(&mut self, key: &[i64]) {
        let Some(e) = take_entry(&mut self.root, key) else { return };
        self.lru[slot_of(e.tier())].remove(&e.used);
        self.count[slot_of(e.tier())] -= 1;
    }
}
