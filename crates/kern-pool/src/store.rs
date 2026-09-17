//! What a prefix's bytes are in a tier: a chain of frozen pages and a
//! state slot, held by value.
//!
//! A [`Storage`] is a budget of pages and slots — the device ([`crate::Pool`])
//! or the host block ([`crate::Host`]) — and a [`Store`] is the first `len`
//! tokens of a sequence in one of them: the pages holding those tokens,
//! shared through a chain, and the slot holding the recurrent state after
//! them. A store is a value: cloning it is another holder, dropping the
//! last holder returns what nobody else holds. `Store<Pool>` is a
//! [`crate::Checkpoint`], `Store<Host>` a [`crate::Parked`].
//!
//! A [`Node`] is one page and the chain before it, reference-counted, so
//! holding a node holds every page up to it and a page returns when its
//! last holder lets go. **A node's page is never written after the node
//! exists**: nodes are made from a lease's whole pages, from a partial page
//! copied for a new holder, or from a retiring lease's last page, and the
//! only writer of any page is a live lease into its own pages. That is what
//! lets the host tier keep one copy per device node: the device node knows
//! its host twin, and a twin still alive is the same bytes.

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};

pub(crate) mod sealed {
    pub trait Sealed {}
}

/// A budget of pages and slots that hands them back one at a time; the
/// device pool and the host block implement it, nothing else.
pub trait Storage: sealed::Sealed + Send + Sync + 'static {
    type Page: Copy + Ord + fmt::Debug + Send + Sync;
    type Slot: Copy + Ord + fmt::Debug + Send + Sync;
    /// What a node knows of its copy in the other tier.
    type Twin: Default + Send + Sync;
    fn give_page(&self, page: Self::Page);
    fn give_slot(&self, slot: Self::Slot);
}

pub(crate) fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A frozen page and the chain before it. Reachable only as the type
/// behind a [`Storage::Twin`]; nothing outside the crate builds or reads one.
pub struct Node<T: Storage> {
    pub(crate) page: T::Page,
    parent: Option<Arc<Node<T>>>,
    tier: Arc<T>,
    pub(crate) twin: T::Twin,
}

impl<T: Storage> Node<T> {
    pub(crate) fn new(page: T::Page, parent: Option<Arc<Node<T>>>, tier: &Arc<T>, twin: T::Twin) -> Arc<Node<T>> {
        Arc::new(Node { page, parent, tier: Arc::clone(tier), twin })
    }
}

impl<T: Storage> Drop for Node<T> {
    /// Unwind the chain in a loop: a recursive drop of a 65k-page chain
    /// would overflow the stack.
    fn drop(&mut self) {
        self.tier.give_page(self.page);
        let mut next = self.parent.take();
        while let Some(n) = next {
            match Arc::try_unwrap(n) {
                Ok(mut node) => next = node.parent.take(),
                Err(_) => break,
            }
        }
    }
}

/// The device node's link to its host copy, set when it is parked or
/// when it is woken from one.
pub type HostTwin = Mutex<Weak<Node<crate::Host>>>;

/// A slot handed back when its last holder drops.
pub(crate) struct SlotOwn<T: Storage> {
    pub(crate) id: T::Slot,
    tier: Arc<T>,
}

impl<T: Storage> SlotOwn<T> {
    pub(crate) fn new(id: T::Slot, tier: &Arc<T>) -> Arc<SlotOwn<T>> {
        Arc::new(SlotOwn { id, tier: Arc::clone(tier) })
    }
}

impl<T: Storage> Drop for SlotOwn<T> {
    fn drop(&mut self) {
        self.tier.give_slot(self.id);
    }
}

/// The nodes of a chain, root first.
pub(crate) fn chain_nodes<T: Storage>(chain: &Option<Arc<Node<T>>>) -> Vec<Arc<Node<T>>> {
    let mut out = Vec::new();
    let mut cur = chain.clone();
    while let Some(n) = cur {
        cur = n.parent.clone();
        out.push(n);
    }
    out.reverse();
    out
}

/// The pages of a chain, root first.
pub(crate) fn chain_pages<T: Storage>(chain: &Option<Arc<Node<T>>>) -> Vec<T::Page> {
    chain_nodes(chain).iter().map(|n| n.page).collect()
}

/// The node `depth` pages up the chain from `chain` (0: `chain` itself).
pub(crate) fn ancestor<T: Storage>(chain: &Option<Arc<Node<T>>>, depth: usize) -> Option<Arc<Node<T>>> {
    let mut cur = chain.clone();
    for _ in 0..depth {
        cur = cur.and_then(|n| n.parent.clone());
    }
    cur
}

/// The first `len` tokens of a sequence in tier `T`: the pages holding
/// them (shared with whoever else holds them) and, when the manifest has
/// per-sequence states, a slot with the state after those tokens. A
/// store with a slot is usable at its own length only; one without at any
/// whole page of it. Holding one holds its bytes; dropping the last holder
/// releases what it alone held.
pub struct Store<T: Storage> {
    pub(crate) len: usize,
    pub(crate) pages: usize,
    /// The chain through the pages; `None` for a slot-only store.
    pub(crate) chain: Option<Arc<Node<T>>>,
    pub(crate) slot: Option<Arc<SlotOwn<T>>>,
}

impl<T: Storage> Clone for Store<T> {
    fn clone(&self) -> Store<T> {
        Store { len: self.len, pages: self.pages, chain: self.chain.clone(), slot: self.slot.clone() }
    }
}

impl<T: Storage> Store<T> {
    /// Tokens held; never 0.
    pub fn tokens(&self) -> usize {
        self.len
    }

    pub(crate) fn paged(&self) -> bool {
        self.chain.is_some()
    }

    /// Whether a state slot is held.
    pub fn has_slot(&self) -> bool {
        self.slot.is_some()
    }

    pub(crate) fn slot_id(&self) -> Option<T::Slot> {
        self.slot.as_ref().map(|s| s.id)
    }
}

impl<T: Storage> fmt::Debug for Store<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Store({} tokens, {} pages", self.len, self.pages)?;
        if let Some(s) = &self.slot {
            write!(f, ", slot {:?}", s.id)?;
        }
        write!(f, ")")
    }
}

/// The byte moves that realize a pool decision, as (from, to) page and
/// slot numbers in the source and destination tiers: `pages` for every
/// paged state, `slot` for every per-sequence state. Empty when nothing
/// moves. A plan is executed by the shell that asked for it, right away,
/// while the handles it was made with are alive.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Copies<S = i32, D = i32> {
    pub pages: Vec<(S, D)>,
    pub slot: Option<(S, D)>,
}
