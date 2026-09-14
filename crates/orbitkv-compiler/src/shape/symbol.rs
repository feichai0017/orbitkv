//! Symbolic dimension variables.
//!
//! A [`Symbol`] names one dynamic dimension — the `s` in a `[s, 128]` shape.
//! Everything that means "a dim variable" says `Symbol`, and every
//! symbol-to-concrete-size mapping says [`DynMap`].
//!
//! A `Symbol` is a `Copy` handle into a generational arena, the same shape as
//! [`Expression`](crate::shape::Expression) and its `Vec<Term>`: the name lives
//! in the arena, and the handle is a pointer plus a generation. That is what
//! keeps [`Term`](crate::shape::Term) `Copy` while a dim name is an owned,
//! arbitrary-length string.
//!
//! `GenerationalBox` supplies only `Copy`, `Clone` and `Debug`, so equality,
//! ordering and hashing are written by hand here and read *through* the handle,
//! comparing names rather than arena slots. `Expression` does the same for its
//! `Hash` and `PartialEq`.
//!
//! `Ord` compares names, not handles, because CUDA assigns each dim a
//! `dyn_dims[]` slot by sorting the dim set and the host uploads values in that
//! same order. Sorting by name keeps that a function of the graph alone; handle
//! order is allocation order.
//!
//! One name is reserved. `"z"` is the runtime loop index, not a dimension:
//! [`Expression::dyn_vars`](crate::shape::Expression::dyn_vars) filters it out
//! and generated kernels declare their own `long long const_z`, so a dimension
//! named `z` would vanish from the buffer planner and collide with that local.
//! See [`RESERVED_INDEX_NAME`] and [`Symbol::try_new_dim`].

use std::fmt;
use std::sync::{OnceLock, RwLock};

use generational_box::{AnyStorage, GenerationalBox, Owner, SyncStorage};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A `Copy` handle to a name in the arena, as `ExprBox` is to a `Vec<Term>`.
type NameBox = GenerationalBox<String, SyncStorage>;

/// Names one symbolic dimension.
///
/// `Copy`, and compares/hashes/orders by name.
///
/// Most code never names a constructor: the dim-taking methods on `Graph` and
/// `CompileOptions` are generic over `impl Into<Symbol>`, so `set_dim("seq", 8)`
/// and `set_dim('s', 8)` both work. Reach for [`sym`] when you need a value
/// directly, or [`Symbol::try_new_dim`] for a name from a frontend.
#[derive(Clone, Copy)]
pub struct Symbol(NameBox);

/// Concrete sizes for the symbolic dimensions of a graph, resolved at runtime.
pub type DynMap = FxHashMap<Symbol, usize>;

/// The runtime loop index. Reserved — never hand this out as a dimension.
///
/// A `&str` rather than a `Symbol` because a `Symbol` is an arena handle and so
/// cannot be a `const`. Use [`Symbol::reserved_index`] for the value.
const RESERVED_INDEX_NAME: &str = "z";

/// A name that cannot be a dimension.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvalidSymbolName {
    /// Not `[A-Za-z][A-Za-z0-9_]*`.
    ///
    /// The alphabet makes `const_<name>` a usable identifier and `"<name>"` a
    /// valid egglog literal by construction, so no codegen site re-checks. The
    /// `__` rule is C++'s, not C's: C++ reserves every identifier containing a
    /// doubled underscore, not just leading ones.
    ///
    /// Rejected rather than sanitized — sanitizing is not injective, so `a.b`
    /// and `a-b` would land on one `#define` and one `MVar`.
    Malformed(String),
    /// The reserved loop index — see [`RESERVED_INDEX_NAME`].
    Reserved,
}

impl fmt::Display for InvalidSymbolName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Malformed(name) => write!(
                f,
                "dim name {name:?} must match [A-Za-z][A-Za-z0-9_]* with no \
                 doubled underscore, so that const_<name> is never an identifier \
                 C++ reserves to the implementation, and <name> is a valid \
                 egglog literal"
            ),
            Self::Reserved => write!(
                f,
                "{RESERVED_INDEX_NAME:?} is the reserved runtime loop index, not a \
                 dimension: kernels declare their own const_{RESERVED_INDEX_NAME} \
                 local and dyn_vars() filters it out, so a dim with this name would \
                 be dropped from the buffer planner"
            ),
        }
    }
}

static NAME_OWNER: OnceLock<Owner<SyncStorage>> = OnceLock::new();
static NAME_INTERNER: OnceLock<RwLock<FxHashMap<String, NameBox>>> = OnceLock::new();

/// One arena slot per distinct name.
///
/// Not needed for correctness — equality reads through the handle and compares
/// names — but without it every `Symbol::new` would claim a fresh slot that is
/// never reclaimed. Mirrors `EXPRESSION_INTERNER`.
fn intern(name: &str) -> NameBox {
    let interner = NAME_INTERNER.get_or_init(|| RwLock::new(FxHashMap::default()));
    if let Some(existing) = interner.read().unwrap().get(name) {
        return *existing;
    }
    let mut guard = interner.write().unwrap();
    if let Some(existing) = guard.get(name) {
        return *existing;
    }
    let boxed = NAME_OWNER
        .get_or_init(SyncStorage::owner)
        .insert(name.to_string());
    guard.insert(name.to_string(), boxed);
    boxed
}

fn is_well_formed(name: &str) -> bool {
    // No is_empty check: starts_with already rejects "".
    name.starts_with(|c: char| c.is_ascii_alphabetic())
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !name.contains("__")
}

impl Symbol {
    /// The reserved runtime loop index.
    pub fn reserved_index() -> Symbol {
        Symbol(intern(RESERVED_INDEX_NAME))
    }

    /// Build a symbol, allowing the reserved index.
    ///
    /// Panics if the name is malformed. Prefer [`Symbol::try_new_dim`] for
    /// names arriving from a frontend: it also rejects the reserved index, and
    /// reports rather than unwinding.
    pub fn new(name: &str) -> Symbol {
        Self::try_new(name).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Build a symbol, allowing the reserved index.
    fn try_new(name: &str) -> Result<Symbol, InvalidSymbolName> {
        if !is_well_formed(name) {
            return Err(InvalidSymbolName::Malformed(name.to_string()));
        }
        Ok(Symbol(intern(name)))
    }

    /// Build a symbol for a *dimension*, rejecting the reserved loop index.
    ///
    /// This is the guard a frontend should use on a name it did not choose.
    pub fn try_new_dim(name: &str) -> Result<Symbol, InvalidSymbolName> {
        let sym = Self::try_new(name)?;
        if sym.is_reserved() {
            return Err(InvalidSymbolName::Reserved);
        }
        Ok(sym)
    }

    /// Whether this is the reserved loop index rather than a real dimension.
    pub fn is_reserved(&self) -> bool {
        *self.0.read() == RESERVED_INDEX_NAME
    }
}

/// Build a dim symbol. Shorthand for [`Symbol::new`].
pub fn sym(name: &str) -> Symbol {
    Symbol::new(name)
}

// GenerationalBox supplies only Copy/Clone/Debug, and a derived PartialEq would
// compare arena slot and generation — so two symbols named `s` would differ.
// These read through the handle and compare names.

impl PartialEq for Symbol {
    fn eq(&self, other: &Self) -> bool {
        *self.0.read() == *other.0.read()
    }
}

impl Eq for Symbol {}

impl PartialOrd for Symbol {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Symbol {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.read().cmp(&other.0.read())
    }
}

impl std::hash::Hash for Symbol {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Hash the name, not the handle: two symbols named `s` must agree.
        (**self.0.read()).hash(state);
    }
}

// No `Borrow<str>`: probing a DynMap with a bare `&str` would need one borrowed
// from the Symbol, but the name lives behind a read guard that outlives nothing.
// Callers build a Symbol to look up.

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0.read())
    }
}

/// Prints the bare name, matching how `Term` and `Expression` render dims.
impl fmt::Debug for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0.read())
    }
}

/// `expr('s')` reads better than `expr(sym("s"))` in hand-written models.
/// Validated like any other name, so `'%'` panics rather than minting `const_%`.
impl From<char> for Symbol {
    fn from(c: char) -> Self {
        let mut buf = [0u8; 4];
        Symbol::new(c.encode_utf8(&mut buf))
    }
}

/// Serializes as the name, so the encoding is unchanged from when a dim was a
/// `char`: `Term::Var('s')` was `{"Var":"s"}` and still is, and artifacts
/// written before this change still load.
impl Serialize for Symbol {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.0.read())
    }
}

impl<'de> Deserialize<'de> for Symbol {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let name = String::deserialize(deserializer)?;
        // try_new, not try_new_dim: the reserved index appears in serialized
        // strides and has to round-trip.
        Symbol::try_new(&name).map_err(serde::de::Error::custom)
    }
}

/// The C identifier a symbol takes in generated kernel source.
///
/// Kernels also declare a *local* `const_z` for the thread index, so this
/// namespace is shared with generated code — see [`RESERVED_INDEX_NAME`].
pub fn kernel_const_name(sym: &Symbol) -> String {
    format!("const_{sym}")
}

/// The name a symbol carries inside egglog, as the payload of `(MVar "…")`.
///
/// Kept separate from [`kernel_const_name`] even though both are the bare
/// spelling: the `.egg` rewrite rules match these literally, so this one is a
/// wire format that rules depend on, not a display detail.
pub fn egglog_var_name(sym: &Symbol) -> String {
    sym.to_string()
}

/// Inverse of [`egglog_var_name`], for names read back out of the e-graph.
///
/// Validates rather than trusting. A name that does not survive the round trip
/// means the boundary mangled it, and a mangled name misses in `dyn_map`, where
/// a miss silently becomes a dimension of 0.
pub fn symbol_from_egglog_name(name: &str) -> Symbol {
    Symbol::try_new(name).unwrap_or_else(|e| panic!("bad dim name out of egglog: {e}"))
}
#[cfg(test)]
#[path = "../../tests/unit/shape/symbol/mod.rs"]
mod tests;
