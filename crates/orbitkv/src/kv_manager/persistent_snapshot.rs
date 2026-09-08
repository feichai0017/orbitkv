use std::sync::Arc;

#[cfg(test)]
use std::cell::Cell;

use super::identity::{PageLease, ViewVersion};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct RootEntry {
    pub(super) class_id: u16,
    pub(super) backend_domain: u16,
    pub(super) logical_ordinal: u64,
    pub(super) temporal_cell_index: u64,
    pub(super) temporal_cycle: u64,
    pub(super) page: PageLease,
    pub(super) backend_index: u64,
}

/// Immutable AVL sequence keyed by the already-contiguous logical ordinal.
/// Cloning a root is O(1); append, tail replacement, and front retirement copy
/// only one logarithmic path, while cached first/last metadata makes steady
/// partial-tail checks O(1). Hot publication is O(C + delta * log R), where C
/// is class count, delta is changed bindings, and R is the resident root size;
/// zero-retirement paths never construct an ordered iterator.
#[derive(Debug)]
pub(super) struct RootTreeNode {
    entry: RootEntry,
    first: RootEntry,
    last: RootEntry,
    pub(super) left: Option<Arc<Self>>,
    pub(super) right: Option<Arc<Self>>,
    height: u32,
    len: usize,
}

#[derive(Clone, Debug, Default)]
pub(super) struct PersistentRootEntries {
    pub(super) root: Option<Arc<RootTreeNode>>,
}

impl PartialEq for PersistentRootEntries {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().eq(other.iter())
    }
}

impl Eq for PersistentRootEntries {}

impl PersistentRootEntries {
    pub(super) fn len(&self) -> usize {
        root_tree_len(self.root.as_deref())
    }

    pub(super) fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    pub(super) fn front(&self) -> Option<&RootEntry> {
        self.root.as_deref().map(|node| &node.first)
    }

    pub(super) fn back(&self) -> Option<&RootEntry> {
        self.root.as_deref().map(|node| &node.last)
    }

    pub(super) fn push_back(&mut self, entry: RootEntry) {
        debug_assert!(
            self.back()
                .is_none_or(|back| back.logical_ordinal < entry.logical_ordinal)
        );
        self.root = Some(root_tree_insert_max(self.root.as_ref(), entry));
    }

    pub(super) fn pop_front(&mut self) -> Option<RootEntry> {
        let root = self.root.as_ref()?;
        let (next, entry) = root_tree_remove_min(root);
        self.root = next;
        Some(entry)
    }

    pub(super) fn pop_back(&mut self) -> Option<RootEntry> {
        let root = self.root.as_ref()?;
        let (next, entry) = root_tree_remove_max(root);
        self.root = next;
        Some(entry)
    }

    pub(super) fn extend(&mut self, entries: impl IntoIterator<Item = RootEntry>) {
        for entry in entries {
            self.push_back(entry);
        }
    }

    pub(super) fn iter(&self) -> RootTreeIter<'_> {
        RootTreeIter::new(self.root.as_deref(), self.len())
    }
}

pub(super) struct RootTreeIter<'a> {
    stack: Vec<&'a RootTreeNode>,
    remaining: usize,
}

impl<'a> RootTreeIter<'a> {
    fn new(root: Option<&'a RootTreeNode>, remaining: usize) -> Self {
        #[cfg(test)]
        ROOT_ITERATOR_ALLOCS.set(ROOT_ITERATOR_ALLOCS.get() + 1);
        let mut iter = Self {
            stack: Vec::new(),
            remaining,
        };
        iter.push_left(root);
        iter
    }

    fn push_left(&mut self, mut node: Option<&'a RootTreeNode>) {
        while let Some(current) = node {
            #[cfg(test)]
            ROOT_NODE_VISITS.set(ROOT_NODE_VISITS.get() + 1);
            self.stack.push(current);
            node = current.left.as_deref();
        }
    }
}

impl<'a> Iterator for RootTreeIter<'a> {
    type Item = &'a RootEntry;

    fn next(&mut self) -> Option<Self::Item> {
        let node = self.stack.pop()?;
        self.push_left(node.right.as_deref());
        self.remaining -= 1;
        Some(&node.entry)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for RootTreeIter<'_> {}

#[cfg(test)]
std::thread_local! {
    static ROOT_NODE_VISITS: Cell<u64> = const { Cell::new(0) };
    static ROOT_ITERATOR_ALLOCS: Cell<u64> = const { Cell::new(0) };
    static ROOT_PATH_NODES_CLONED: Cell<u64> = const { Cell::new(0) };
}

#[cfg(test)]
pub(super) fn root_instrumentation() -> (u64, u64, u64) {
    (
        ROOT_NODE_VISITS.get(),
        ROOT_ITERATOR_ALLOCS.get(),
        ROOT_PATH_NODES_CLONED.get(),
    )
}

fn root_tree_height(root: Option<&RootTreeNode>) -> u32 {
    root.map_or(0, |node| node.height)
}

fn root_tree_len(root: Option<&RootTreeNode>) -> usize {
    root.map_or(0, |node| node.len)
}

fn root_tree_node(
    entry: RootEntry,
    left: Option<Arc<RootTreeNode>>,
    right: Option<Arc<RootTreeNode>>,
) -> Arc<RootTreeNode> {
    #[cfg(test)]
    ROOT_PATH_NODES_CLONED.set(ROOT_PATH_NODES_CLONED.get() + 1);
    let first = left.as_ref().map_or(entry, |node| node.first);
    let last = right.as_ref().map_or(entry, |node| node.last);
    Arc::new(RootTreeNode {
        entry,
        first,
        last,
        height: 1 + root_tree_height(left.as_deref()).max(root_tree_height(right.as_deref())),
        len: 1 + root_tree_len(left.as_deref()) + root_tree_len(right.as_deref()),
        left,
        right,
    })
}

fn root_tree_balance(
    entry: RootEntry,
    left: Option<Arc<RootTreeNode>>,
    right: Option<Arc<RootTreeNode>>,
) -> Arc<RootTreeNode> {
    let balance = i64::from(root_tree_height(left.as_deref()))
        - i64::from(root_tree_height(right.as_deref()));
    if balance > 1 {
        let left_root = left.as_ref().expect("left-heavy AVL node has a left child");
        if root_tree_height(left_root.left.as_deref())
            >= root_tree_height(left_root.right.as_deref())
        {
            let rotated_right = root_tree_node(entry, left_root.right.clone(), right);
            return root_tree_node(left_root.entry, left_root.left.clone(), Some(rotated_right));
        }
        let pivot = left_root
            .right
            .as_ref()
            .expect("left-right AVL rotation has a pivot");
        let rotated_left =
            root_tree_node(left_root.entry, left_root.left.clone(), pivot.left.clone());
        let rotated_right = root_tree_node(entry, pivot.right.clone(), right);
        return root_tree_node(pivot.entry, Some(rotated_left), Some(rotated_right));
    }
    if balance < -1 {
        let right_root = right
            .as_ref()
            .expect("right-heavy AVL node has a right child");
        if root_tree_height(right_root.right.as_deref())
            >= root_tree_height(right_root.left.as_deref())
        {
            let rotated_left = root_tree_node(entry, left, right_root.left.clone());
            return root_tree_node(
                right_root.entry,
                Some(rotated_left),
                right_root.right.clone(),
            );
        }
        let pivot = right_root
            .left
            .as_ref()
            .expect("right-left AVL rotation has a pivot");
        let rotated_left = root_tree_node(entry, left, pivot.left.clone());
        let rotated_right = root_tree_node(
            right_root.entry,
            pivot.right.clone(),
            right_root.right.clone(),
        );
        return root_tree_node(pivot.entry, Some(rotated_left), Some(rotated_right));
    }
    root_tree_node(entry, left, right)
}

fn root_tree_insert_max(root: Option<&Arc<RootTreeNode>>, entry: RootEntry) -> Arc<RootTreeNode> {
    root.map_or_else(
        || root_tree_node(entry, None, None),
        |node| {
            let right = Some(root_tree_insert_max(node.right.as_ref(), entry));
            root_tree_balance(node.entry, node.left.clone(), right)
        },
    )
}

fn root_tree_remove_min(root: &Arc<RootTreeNode>) -> (Option<Arc<RootTreeNode>>, RootEntry) {
    match root.left.as_ref() {
        None => (root.right.clone(), root.entry),
        Some(left) => {
            let (next_left, removed) = root_tree_remove_min(left);
            (
                Some(root_tree_balance(root.entry, next_left, root.right.clone())),
                removed,
            )
        }
    }
}

fn root_tree_remove_max(root: &Arc<RootTreeNode>) -> (Option<Arc<RootTreeNode>>, RootEntry) {
    match root.right.as_ref() {
        None => (root.left.clone(), root.entry),
        Some(right) => {
            let (next_right, removed) = root_tree_remove_max(right);
            (
                Some(root_tree_balance(root.entry, root.left.clone(), next_right)),
                removed,
            )
        }
    }
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ClassRoot {
    pub(super) entries: PersistentRootEntries,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct RequestSnapshot {
    pub(super) boundary: u64,
    pub(super) view_version: ViewVersion,
    pub(super) roots: Arc<[ClassRoot]>,
}

impl RequestSnapshot {
    pub(super) fn resident_count(&self) -> usize {
        self.roots.iter().map(|root| root.entries.len()).sum()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.roots.iter().all(|root| root.entries.is_empty())
    }
}
#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct HotPathInstrumentation {
    pub(super) hot_root_entries_visited: u64,
    pub(super) root_node_visits: u64,
    pub(super) root_iterator_allocs: u64,
    pub(super) path_nodes_cloned: u64,
    pub(super) device_view_entries_materialized: u64,
    pub(super) snapshot_entries_cloned: u64,
    pub(super) delta_entries_touched: u64,
    pub(super) retirement_entries_touched: u64,
}
