//! Manifest schema (format 5). Parsing is already strict: unknown fields,
//! duplicate names and malformed type strings are rejected at
//! deserialization time. Semantic checks (references, dtypes, dataflow,
//! bounds) live in [`crate::verify`]; what a serving loop needs to drive
//! the manifest (`fill`, `batch`) is projected by [`crate::protocol`].
//!
//! Vocabulary, one word per level so nothing collides:
//!
//! ```text
//! programs.<name>            a *program*: its calling shape and its calls  {"batch": {...}, "calls": [...]}
//! programs.<name>.calls[]    a *call* of an op            {"op": "attn", "args": [...]}
//! ops.<name>                 an op: interface + impl      {"params": [...], "impl": {...}}
//! ops.<name>.impl.launches[] a *launch* of a module entry {"module": "argmax", "entry": "kern_argmax_partial"}
//! modules.<name>             an artifact the launches pin {"source": "argmax.cubin", "sha256": "..."}
//! vars.<name>                a per-call scalar the caller supplies, bounded
//! states.<name>              opaque persistent memory, sized by the runtime
//! buffers.<name>             typed tensors: input / output / weight / workspace / carry / peer;
//!                            `fill` names the role a caller-facing one plays in a call
//! topology.groups.<name>     a rank group and its size; the manifest is SPMD over it
//! ```

use serde::de::{Error as DeError, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fmt;
use std::marker::PhantomData;
use std::str::FromStr;

/// The one format this crate reads and writes.
pub(crate) const SCHEMA_VERSION: u32 = 5;

/// Deserialize a JSON object into a map, rejecting duplicate keys (plain
/// serde silently keeps the last one).
fn unique_map<'de, D, V>(deserializer: D) -> Result<BTreeMap<String, V>, D::Error>
where
    D: Deserializer<'de>,
    V: Deserialize<'de>,
{
    struct UniqueMap<V>(PhantomData<V>);

    impl<'de, V: Deserialize<'de>> Visitor<'de> for UniqueMap<V> {
        type Value = BTreeMap<String, V>;

        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("a map with unique keys")
        }

        fn visit_map<A: MapAccess<'de>>(self, mut access: A) -> Result<Self::Value, A::Error> {
            let mut map = BTreeMap::new();
            while let Some((key, value)) = access.next_entry::<String, V>()? {
                if map.insert(key.clone(), value).is_some() {
                    return Err(A::Error::custom(format!("duplicate name `{key}`")));
                }
            }
            Ok(map)
        }
    }

    deserializer.deserialize_map(UniqueMap(PhantomData))
}

/// The resolved contract: vars, states, buffers, modules, ops and programs.
/// The wire format also accepts optional `constants`, expanded during
/// deserialization. Use [`crate::json_schema`] for the complete wire schema;
/// this type's derived schema describes the resolved numeric form.
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    /// Wire-format version; must be `5`.
    pub schema_version: u32,
    /// Free-form model label, e.g. `"qwen3-4b"`.
    pub model: String,
    /// Rank groups a multi-GPU manifest is SPMD over, e.g. `{"groups": {"ep": 4}}`; every rank loads the same manifest.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub topology: Option<Topology>,
    /// Per-call scalars the caller supplies, e.g. `{"tokens": {"max": 2048}}`.
    #[serde(default, deserialize_with = "unique_map", skip_serializing_if = "BTreeMap::is_empty")]
    pub vars: BTreeMap<String, Var>,
    /// Opaque persistent memory the runtime provisions by size, e.g. a paged KV cache.
    #[serde(default, deserialize_with = "unique_map", skip_serializing_if = "BTreeMap::is_empty")]
    pub states: BTreeMap<String, State>,
    /// Typed tensors: inputs, outputs, weights, workspace and carries.
    #[serde(deserialize_with = "unique_map")]
    pub buffers: BTreeMap<String, Buffer>,
    /// Code artifacts the launches pin, by name, e.g. `{"argmax": {"source": "argmax.cubin", "sha256": "2537…"}}`.
    #[serde(deserialize_with = "unique_map")]
    pub modules: BTreeMap<String, Module>,
    /// Operators: a typed interface plus the launches that implement it.
    #[serde(deserialize_with = "unique_map")]
    pub ops: BTreeMap<String, Op>,
    /// Named straight-line programs, e.g. `"decode": {"batch": {"groups": 256, "rows": 1}, "calls": [...]}`.
    #[serde(deserialize_with = "unique_map")]
    pub programs: BTreeMap<String, Program>,
}

impl Manifest {
    /// Sequence slots the runtime provisions for every `bytes_per_seq`
    /// state: the widest column axis of any line table (an input indexing
    /// a per-sequence state, shaped `[lines, cols]` or `[lines, cols, w]`
    /// — every rank holds a slice of every row's state, so the axis may
    /// span a tray batch), 1 without one — plus one so a batched caller
    /// can hold a padding lease, plus slot 0, which is never leased —
    /// kernels may treat line index 0 as the null line.
    pub fn seq_slots(&self) -> u64 {
        let cols = self
            .buffers
            .values()
            .filter(|b| b.kind == BufferKind::Input)
            .filter(|b| {
                b.domain
                    .as_ref()
                    .and_then(|d| d.index_into.as_deref())
                    .and_then(|s| self.states.get(s))
                    .is_some_and(State::is_per_seq)
            })
            .filter_map(|b| match b.shape.get(1) {
                Some(Dim::Var(v)) => self.vars.get(v).map(|v| v.max),
                Some(Dim::Const(c)) => Some(*c),
                None => None,
            })
            .max();
        cols.map_or(1, |c| c.max(1)) + 2
    }

    /// Size of a declared topology group, if any.
    pub fn group_size(&self, group: &str) -> Option<u64> {
        self.topology.as_ref()?.groups.get(group).copied()
    }

    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }

    /// Serialize the resolved contract with numeric literals.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("manifest serialization cannot fail")
    }
}

/// A program: a straight-line call list plus, when a serving loop may drive it, the shape of one call.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Program {
    /// The shape of one call — `groups` sequences of `rows` rows each — for a program a serving loop drives; absent for one only a harness runs (a single layer under test, a barrier).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch: Option<Batch>,
    /// Run once after load (after every peer is imported), never per step: a tray manifest's collective setup. Takes no per-call input.
    #[serde(default, skip_serializing_if = "is_false")]
    pub once: bool,
    /// Driven through a CUDA graph: the runtime captures the call list at each var assignment the program is called with and replays it with one launch. For a program of fixed shape per call whose launches are many and short (a decode step, a speculative round); a program whose rows vary per call (a prefill chunk) runs launch by launch.
    #[serde(default, skip_serializing_if = "is_false")]
    pub graph: bool,
    /// The calls, in order, e.g. `[{"op": "embedding", "args": [...]}, ...]`.
    pub calls: Vec<Call>,
}

/// The shape of one call of a program: up to `groups` sequences, each contributing `rows` rows, e.g. `{"groups": 256, "rows": 1}` (a decode step over a batch) or `{"groups": 1, "rows": "tokens"}` (a prefill chunk).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Batch {
    /// Upper bound on the sequences one call covers; a caller with fewer pads up to a bucket, e.g. `256`.
    pub groups: u64,
    /// Rows per sequence: a constant the program's kernels are laid out for (`1` for a decode step, `8` for a speculative round — exact, not a bound), or the var whose value is the row count of the call (`"tokens"`: one sequence, as many rows as the chunk).
    pub rows: Dim,
    /// The var whose value is the length of the call's span: one of the sequences (rows `1` each) feeds a run of that many consecutive tokens, a row each, starting at the row the `span_at` fill names, e.g. `"span"` (a prompt chunk riding a decode step). Absent for a call whose sequences all feed `rows`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub span: Option<String>,
}

/// The role a caller-facing buffer plays in a call: what the serving loop writes into an input or reads from an output. A closed set; the runtime never reads it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Fill {
    /// Input: the token ids fed, one per row (a buffer over the row axis) or each sequence's first (over the sequence axis: the anchor of a speculative round).
    Token,
    /// Input: each row's position in its sequence.
    Position,
    /// Input: one for a real row, zero for serving bucket padding.
    Valid,
    /// Input: each row's token slot in the paged states, from the sequence's lease.
    Slot,
    /// Input: each sequence's length after this call (rows already in the state plus this call's).
    SeqLen,
    /// Input: exclusive prefix sums of rows per sequence, `groups + 1` entries.
    CuSeqlens,
    /// Input: the first row of the call's span (`batch.span`), one word; 0 when the call has none.
    SpanAt,
    /// Input: the tray batch's blocks — exclusive prefix sums of each member's rows in member order, `members + 1` entries, the last the tray's rows.
    Blocks,
    /// Output: the tokens a call produced — one per sequence (`[groups]`) or one per row (`[groups, rows]`, the program's `rows`).
    Tokens,
    /// Output: how many of a sequence's `tokens` the caller takes, per sequence; without one it takes one.
    Count,
    /// Output: a one-word error flag of the collectives, read after every call; nonzero is a failed step.
    Error,
}

impl fmt::Display for Fill {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Fill::Token => "token",
            Fill::Position => "position",
            Fill::Valid => "valid",
            Fill::Slot => "slot",
            Fill::SeqLen => "seq_len",
            Fill::CuSeqlens => "cu_seqlens",
            Fill::SpanAt => "span_at",
            Fill::Blocks => "blocks",
            Fill::Tokens => "tokens",
            Fill::Count => "count",
            Fill::Error => "error",
        })
    }
}

/// Rank groups, e.g. `{"groups": {"ep": 4}}`. Sizes are fixed in the manifest; the loading rank's index in each group is a load-time constant that `{"rank": "<group>"}` args receive and `peer` buffers are ordered by.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Topology {
    /// Group name to member count, e.g. `{"ep": 4}`.
    #[serde(deserialize_with = "unique_map")]
    pub groups: BTreeMap<String, u64>,
}

/// A per-call scalar the caller supplies, bounded `1..=max`; the only kind of number that may size a shape or a grid.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Var {
    /// Upper bound, e.g. `2048` for the token count of a prefill chunk.
    pub max: u64,
}

impl Var {
    /// Lower bound of every var.
    pub const MIN: u64 = 1;
}

/// Opaque persistent memory; the runtime provisions the bytes and hands the base pointer to `inout state` params. Exactly one of the three sizes is non-zero. States are always allocated through the driver's virtual-memory API with a fabric-shareable handle when the device has one, so a `peer` buffer may be `of` a state.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct State {
    /// Bytes per token slot, scaled by the capacity — a paged KV cache, e.g. `147456`.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub bytes_per_token: u64,
    /// Fixed byte count independent of capacity and sequences, e.g. `4096`.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub bytes: u64,
    /// Bytes per sequence slot, one slot per live sequence — a recurrent conv/SSM state, e.g. `154140672`. The runtime starts with `seqs.max + 2` slots — slot 0 (never leased; kernels may read line index 0 as null), one per sequence, one for a batched caller's padding — and grows them out of the state budget as checkpoints keep the states of sleeping sequences.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub bytes_per_seq: u64,
}

impl State {
    /// Whether this state is one slot per sequence.
    pub fn is_per_seq(&self) -> bool {
        self.bytes_per_seq > 0
    }
}

fn is_zero(v: &u64) -> bool {
    *v == 0
}

/// A typed tensor, e.g. `{"dtype": "bf16", "shape": ["tokens", 2560], "kind": "workspace"}`.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Buffer {
    /// Element type, e.g. `"bf16"`.
    pub dtype: DType,
    /// Extents, constants or var names, e.g. `["tokens", 2560]`.
    pub shape: Vec<Dim>,
    /// Who provides the buffer and how long its contents live.
    pub kind: BufferKind,
    /// Immutable weights may reside in shared, GPU-addressable host memory.
    #[serde(default, skip_serializing_if = "Placement::is_device")]
    pub placement: Placement,
    /// `input` / `output` buffers only: the role a serving loop fills or reads it for, e.g. `"token"`; absent for one a harness stages by name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fill: Option<Fill>,
    /// Optional prior on the contents; the runtime rejects out-of-domain input writes and `kern test` synthesizes values from it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub domain: Option<Domain>,
    /// Allocate through the driver's virtual-memory API with a fabric-shareable handle so other ranks can map it; the address of every rank's copy lands in the `peer` buffers `of` it. e.g. `true`.
    #[serde(default, skip_serializing_if = "is_false")]
    pub export: bool,
    /// `peer` buffers only: the exported buffer or state whose per-rank base addresses this buffer holds, e.g. `"flags"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub of: Option<String>,
    /// `peer` buffers only: the topology group the addresses are indexed by, e.g. `"ep"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group: Option<String>,
    /// `weight` buffers only: the checkpoint tensors laid end to end into this buffer, in order, e.g. `[{"tensor": "q_proj.weight"}, {"tensor": "k_proj.weight"}]`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub bind: Vec<Segment>,
}

/// Physical placement of a buffer; host placement is restricted to immutable weights.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Placement {
    #[default]
    Device,
    Host,
}

impl Placement {
    fn is_device(&self) -> bool {
        *self == Self::Device
    }
}

/// One piece of a weight buffer: a checkpoint tensor, or a rectangle of it, copied whole into the next bytes of the buffer. The tensor is read as a matrix `[rows, cols]` (its first axis by the product of the rest); `rows` / `cols` take a half-open range of it, e.g. `{"tensor": "fc.weight", "cols": [0, 5120]}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Segment {
    /// Tensor name in the safetensors header, e.g. `"model.layers.0.self_attn.q_proj.weight"`.
    pub tensor: TensorSource,
    /// Half-open row range `[from, to)` of the tensor's first axis, fixed or selected by a declared rank group; the whole axis when absent, e.g. `[0, 1024]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rows: Option<Rows>,
    /// Half-open column range `[from, to)` over the product of the remaining axes; every column when absent (a strided copy otherwise), e.g. `[5120, 10240]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cols: Option<[u64; 2]>,
}

/// A checkpoint tensor, fixed or selected by a declared rank group.
/// Selection is a load-time table lookup, never a string template or expression.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged, deny_unknown_fields)]
pub enum TensorSource {
    Named(String),
    Ranked { group: String, tensors: Vec<String> },
}

/// A row range of a checkpoint tensor: one range, or one per rank of a declared group (a table sharded across the group's devices, each rank loading its slice), e.g. `{"group": "ep", "ranges": [[0, 8], [8, 16]]}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged, deny_unknown_fields)]
pub enum Rows {
    Range([u64; 2]),
    Ranked { group: String, ranges: Vec<[u64; 2]> },
}

impl From<[u64; 2]> for Rows {
    fn from(range: [u64; 2]) -> Self {
        Self::Range(range)
    }
}

impl From<String> for TensorSource {
    fn from(name: String) -> Self {
        Self::Named(name)
    }
}

impl fmt::Display for TensorSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Named(name) => f.write_str(name),
            Self::Ranked { group, .. } => write!(f, "rank-selected tensor in {group}"),
        }
    }
}

/// A prior on a buffer's contents: bounds (`{"min": 0, "max": "tokens"}`) or an index into a buffer/state (`{"index_into": "kv", "stride": 16}`).
#[derive(Debug, Clone, Default, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Domain {
    /// Inclusive lower bound, e.g. `0`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) min: Option<Bound>,
    /// Inclusive upper bound, a literal or a var expression, e.g. `"tokens"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) max: Option<Bound>,
    /// Buffer or state whose rows / token slots the elements index, e.g. `"model.embed_tokens.weight"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index_into: Option<String>,
    /// Rows or token slots per index (default `1`), e.g. `16` for a paged KV block table.
    #[serde(default = "one", skip_serializing_if = "is_one")]
    pub stride: u64,
    /// Require a non-decreasing sequence, e.g. `cu_seqlens`.
    #[serde(default, skip_serializing_if = "is_false")]
    pub(crate) monotone: bool,
}

fn one() -> u64 {
    1
}
fn is_one(v: &u64) -> bool {
    *v == 1
}
fn is_false(v: &bool) -> bool {
    !*v
}

/// A domain bound: an integer, a float, or a var expression, e.g. `0`, `1e-6`, `"tokens"`.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum Bound {
    Int(i64),
    Float(f64),
    Expr(Expr),
}

impl Bound {
    pub(crate) fn eval(&self, vars: &BTreeMap<String, u64>) -> Result<f64, EvalError> {
        Ok(match self {
            Bound::Int(v) => *v as f64,
            Bound::Float(v) => *v,
            Bound::Expr(e) => e.eval(vars)? as f64,
        })
    }
}

/// A domain with its bounds evaluated for one var environment and one
/// state capacity. `lo`/`hi` are inclusive; `None` is unbounded.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResolvedDomain {
    pub lo: Option<f64>,
    pub hi: Option<f64>,
    pub monotone: bool,
}

impl ResolvedDomain {
    pub fn contains(&self, v: f64) -> bool {
        if v.is_nan() {
            return false;
        }
        self.lo.is_none_or(|lo| v >= lo) && self.hi.is_none_or(|hi| v <= hi)
    }
}

impl Domain {
    /// Row count of `index_into`'s target at `vars`, or what the runtime
    /// provisioned for a state; `None` when the name resolves to nothing.
    fn target_rows(&self, m: &Manifest, vars: &BTreeMap<String, u64>, p: &Provision) -> Result<Option<u64>, EvalError> {
        let Some(t) = &self.index_into else { return Ok(None) };
        if let Some(b) = m.buffers.get(t) {
            return Ok(Some(match b.shape.first() {
                Some(Dim::Const(c)) => *c,
                Some(Dim::Var(s)) => *vars.get(s).ok_or_else(|| EvalError::UnknownVar(s.clone()))?,
                None => 0,
            }));
        }
        if let Some(st) = m.states.get(t) {
            // A per-sequence state is addressed in lines of `stride` bytes,
            // `seq_slots` slots of them; `resolve` divides by the stride
            // again, so hand back the byte count.
            return Ok(Some(if st.is_per_seq() { p.seq_slots * st.bytes_per_seq } else { p.tokens }));
        }
        Ok(None)
    }

    /// Evaluate the bounds. Verification guarantees the references resolve;
    /// an unknown `index_into` target here yields an unbounded domain.
    pub fn resolve(
        &self,
        m: &Manifest,
        vars: &BTreeMap<String, u64>,
        p: &Provision,
    ) -> Result<ResolvedDomain, EvalError> {
        if self.index_into.is_some() {
            let rows = self.target_rows(m, vars, p)?;
            let hi = rows.map(|r| (r / self.stride.max(1)).saturating_sub(1) as f64);
            return Ok(ResolvedDomain { lo: Some(0.0), hi, monotone: self.monotone });
        }
        Ok(ResolvedDomain {
            lo: self.min.as_ref().map(|b| b.eval(vars)).transpose()?,
            hi: self.max.as_ref().map(|b| b.eval(vars)).transpose()?,
            monotone: self.monotone,
        })
    }
}

/// What the runtime provisioned the states for: the token slots every
/// paged state can address and the sequence slots every per-sequence one
/// can, the bounds an `index_into` domain resolves against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Provision {
    pub tokens: u64,
    pub seq_slots: u64,
}

/// One shape extent: a constant or a var name, e.g. `2560` or `"tokens"`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum Dim {
    Const(u64),
    Var(String),
}

/// Who provides a buffer and how long its contents live.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum BufferKind {
    /// Written by the runtime before each run, e.g. `token_ids`.
    Input,
    /// Read back by the runtime after each run, e.g. `next_token`.
    Output,
    /// Assembled at load time from the checkpoint tensors its `bind` names, e.g. `model.embed_tokens.weight`.
    Weight,
    /// Runtime-owned scratch, dead between runs, e.g. `hidden`.
    Workspace,
    /// Written by one program and read by another, kept between runs, e.g. the `fc_out` hidden states a draft reads.
    Carry,
    /// `u64[group size]` of device addresses: every group member's copy of the buffer or state named by `of` (`export`ed), this rank's own included. Filled by the runtime once the peers' handles are imported, read-only to ops.
    Peer,
}

impl fmt::Display for BufferKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            BufferKind::Input => "input",
            BufferKind::Output => "output",
            BufferKind::Weight => "weight",
            BufferKind::Workspace => "workspace",
            BufferKind::Carry => "carry",
            BufferKind::Peer => "peer",
        })
    }
}

/// Element type: `bf16`, `f16`, `f32`, `fp8e4m3`, `i32`, `u32`, `i64`, `u64`, `u8`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    Bf16,
    F16,
    F32,
    Fp8E4m3,
    Fp8E8m0,
    I8,
    I32,
    U32,
    I64,
    U64,
    U8,
}

impl DType {
    pub fn bytes(self) -> u64 {
        match self {
            DType::Bf16 | DType::F16 => 2,
            DType::F32 | DType::I32 | DType::U32 => 4,
            DType::I64 | DType::U64 => 8,
            DType::Fp8E4m3 | DType::Fp8E8m0 | DType::I8 | DType::U8 => 1,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            DType::Bf16 => "bf16",
            DType::F16 => "f16",
            DType::F32 => "f32",
            DType::Fp8E4m3 => "fp8e4m3",
            DType::Fp8E8m0 => "fp8e8m0",
            DType::I8 => "i8",
            DType::I32 => "i32",
            DType::U32 => "u32",
            DType::I64 => "i64",
            DType::U64 => "u64",
            DType::U8 => "u8",
        }
    }
}

impl FromStr for DType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        Ok(match s {
            "bf16" => DType::Bf16,
            "f16" => DType::F16,
            "f32" => DType::F32,
            "fp8e4m3" => DType::Fp8E4m3,
            "fp8e8m0" => DType::Fp8E8m0,
            "i8" => DType::I8,
            "i32" => DType::I32,
            "u32" => DType::U32,
            "i64" => DType::I64,
            "u64" => DType::U64,
            "u8" => DType::U8,
            _ => return Err(format!("unknown dtype `{s}`")),
        })
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl schemars::JsonSchema for DType {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "DType".into()
    }

    fn json_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({
            "description": "Element type of a buffer or scratch, e.g. `\"bf16\"`.",
            "type": "string",
            "enum": ["bf16", "f16", "f32", "fp8e4m3", "fp8e8m0", "i8", "i32", "u32", "i64", "u64", "u8"],
        })
    }
}

impl Serialize for DType {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for DType {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        String::deserialize(d)?.parse().map_err(D::Error::custom)
    }
}

/// Direction of a pointer param: `in`, `out`, `inout`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dir {
    In,
    Out,
    InOut,
}

impl fmt::Display for Dir {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Dir::In => "in",
            Dir::Out => "out",
            Dir::InOut => "inout",
        })
    }
}

/// A by-value scalar param: `i32`, `i64`, `f32`, `u8`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarType {
    I32,
    I64,
    F32,
    /// One-byte scalar, e.g. a bool flag.
    U8,
}

impl fmt::Display for ScalarType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            ScalarType::I32 => "i32",
            ScalarType::I64 => "i64",
            ScalarType::F32 => "f32",
            ScalarType::U8 => "u8",
        })
    }
}

/// One op or launch parameter, written as a string: `"in buffer<bf16>"`, `"inout state"`, `"i32"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamType {
    /// A buffer pointer with its element type and direction, e.g. `"out buffer<bf16>"`.
    Buf { dtype: DType, dir: Dir },
    /// An opaque state base pointer, e.g. `"inout state"`.
    State { dir: Dir },
    /// A by-value scalar, e.g. `"i32"`.
    Scalar(ScalarType),
    /// An aggregate of `n` bytes passed by value (a kernel whose ABI takes
    /// structs, or a bare 128-byte `CUtensorMap`), launch-private: wired with
    /// a `{"pack": {...}}` launch arg.
    Bytes(u32),
}

impl ParamType {
    /// Size of the param slot in the kernel ABI, cross-checked against
    /// `cuFuncGetParamInfo` when the module is loaded.
    pub fn size_bytes(self) -> u64 {
        match self {
            ParamType::Buf { .. } | ParamType::State { .. } => 8,
            ParamType::Bytes(n) => n as u64,
            ParamType::Scalar(ScalarType::I64) => 8,
            ParamType::Scalar(ScalarType::U8) => 1,
            ParamType::Scalar(_) => 4,
        }
    }

    /// Direction of a pointer param; `None` for scalars.
    pub fn dir(self) -> Option<Dir> {
        match self {
            ParamType::Buf { dir, .. } | ParamType::State { dir } => Some(dir),
            ParamType::Scalar(_) | ParamType::Bytes(_) => None,
        }
    }
}

impl FromStr for ParamType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        let s = s.trim();
        match s {
            "i32" => return Ok(ParamType::Scalar(ScalarType::I32)),
            "i64" => return Ok(ParamType::Scalar(ScalarType::I64)),
            "f32" => return Ok(ParamType::Scalar(ScalarType::F32)),
            "u8" => return Ok(ParamType::Scalar(ScalarType::U8)),
            _ => {}
        }
        if let Some(n) = s.strip_prefix("bytes<").and_then(|r| r.strip_suffix('>')) {
            return match n.parse::<u32>() {
                Ok(n) if n > 0 => Ok(ParamType::Bytes(n)),
                _ => Err(format!("invalid byte count in param type `{s}`")),
            };
        }
        let (dir_s, rest) = s.split_once(' ').ok_or_else(|| format!("invalid param type `{s}`"))?;
        let dir = match dir_s {
            "in" => Dir::In,
            "out" => Dir::Out,
            "inout" => Dir::InOut,
            _ => return Err(format!("invalid direction `{dir_s}` in param type `{s}`")),
        };
        let rest = rest.trim();
        if rest == "state" {
            return Ok(ParamType::State { dir });
        }
        if let Some(dt) = rest.strip_prefix("buffer<").and_then(|r| r.strip_suffix('>')) {
            let dtype = dt.parse::<DType>().map_err(|e| format!("{e} in param type `{s}`"))?;
            return Ok(ParamType::Buf { dtype, dir });
        }
        Err(format!("invalid param type `{s}`"))
    }
}

impl fmt::Display for ParamType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParamType::Buf { dtype, dir } => write!(f, "{dir} buffer<{dtype}>"),
            ParamType::State { dir } => write!(f, "{dir} state"),
            ParamType::Scalar(st) => write!(f, "{st}"),
            ParamType::Bytes(n) => write!(f, "bytes<{n}>"),
        }
    }
}

impl schemars::JsonSchema for ParamType {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "ParamType".into()
    }

    fn json_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({
            "description": "One parameter: a scalar (`\"i32\"`, `\"i64\"`, `\"f32\"`, `\"u8\"`), \
                a directional pointer (`\"in buffer<bf16>\"`, `\"out buffer<f32>\"`, \
                `\"inout state\"`), or a launch-private `\"tensormap\"` (128-byte TMA descriptor) \
                or `\"bytes<n>\"` (an n-byte aggregate filled by a `pack` launch arg).",
            "type": "string",
            "pattern": "^(i32|i64|f32|u8|bytes<[1-9][0-9]*>|(in|out|inout) (state|buffer<(bf16|f16|f32|fp8e4m3|fp8e8m0|i8|i32|u32|i64|u64|u8)>))$",
        })
    }
}

impl Serialize for ParamType {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for ParamType {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        String::deserialize(d)?.parse().map_err(D::Error::custom)
    }
}

/// A code artifact, e.g. `{"source": "hf:kernels-community/activation/build/.../_activation.abi3.so", "sha256": "73748b54..."}`.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Module {
    /// Local file name (`argmax.cubin`) or registry ref (`hf:org/repo/path[@rev]`); a label, not the identity.
    pub source: String,
    /// Hex sha256 of the artifact bytes; the runtime matches modules by this.
    pub sha256: String,
}

/// An operator: the typed interface a call binds, plus the launches that implement it.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Op {
    /// Interface params in call order, e.g. `["out buffer<bf16>", "in buffer<bf16>"]`.
    pub params: Vec<ParamType>,
    /// The implementation; swapping it leaves every call untouched.
    #[serde(rename = "impl")]
    pub imp: Impl,
}

/// An op implementation: private scratch buffers and one or more launches in order.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Impl {
    #[serde(default, deserialize_with = "unique_map", skip_serializing_if = "BTreeMap::is_empty")]
    /// Implementation-private buffers, e.g. `{"pmax": {"dtype": "f32", "shape": [1, 64]}}`.
    pub scratch: BTreeMap<String, Scratch>,
    /// Launches in execution order.
    pub launches: Vec<Launch>,
}

/// A private buffer of one implementation, sized like a workspace buffer.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Scratch {
    /// Element type, e.g. `"f32"`.
    pub dtype: DType,
    /// Extents, constants or var names, e.g. `["tokens", 8]`.
    pub shape: Vec<Dim>,
}

/// One launch of an implementation: a kernel entry in a pinned module, or a runtime built-in.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum Launch {
    Kernel(KernelLaunch),
    Extern(ExternLaunch),
}

/// A device kernel launch, e.g. `{"module": "argmax", "entry": "kern_argmax_partial_bf16", "block": [1024, 1, 1], "grid": [1, 64, 1]}`.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct KernelLaunch {
    /// Name of a `modules` entry, e.g. `"argmax"`.
    pub module: String,
    /// Kernel symbol in the module, e.g. `"kern_argmax_partial_bf16"`.
    pub entry: String,
    /// This launch's own ABI when it differs from the op's params, e.g. `["in buffer<bf16>", "out buffer<f32>", "i32"]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    params: Option<Vec<ParamType>>,
    /// Threads per block, e.g. `[1024, 1, 1]`.
    pub block: [u32; 3],
    /// Blocks per launch, as expressions, e.g. `[{"ceil_div": ["tokens", 128]}, 1, 1]`.
    pub grid: [Expr; 3],
    /// Dynamic shared memory in bytes, e.g. `{"mul": ["tokens", 512]}`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shared_mem: Option<Expr>,
    /// Thread-block cluster shape, e.g. `[2, 1, 1]`; the grid must be a multiple of it on every axis. Launched with `cuLaunchKernelEx`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cluster: Option<[u32; 3]>,
    /// Programmatic dependent launch, e.g. `true`: the kernel may start before the previous launch on the stream has finished, and it promises to execute `griddepcontrol.wait` before reading or writing anything that launch produces or consumes. Only reads of weights may precede the wait.
    #[serde(default, skip_serializing_if = "is_false")]
    pub pdl: bool,
    /// Where each launch param comes from (default: the op's params in order), e.g. `[{"param": 0}, {"scratch": "pmax"}, {"i32": 64}]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    args: Option<Vec<LaunchArg>>,
}

/// A runtime built-in launch, e.g. `{"entry": "extern:cublaslt_bf16_tn"}`.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ExternLaunch {
    /// `extern:<name>`: `cublaslt_bf16_tn` (C = A·Wᵀ) or `cublaslt_bf16_tn_acc` (C += A·Wᵀ).
    pub entry: String,
    /// This launch's own ABI when it differs from the op's params.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    params: Option<Vec<ParamType>>,
    /// Where each launch param comes from (default: the op's params in order).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    args: Option<Vec<LaunchArg>>,
}

impl Launch {
    pub(crate) fn is_extern(&self) -> bool {
        matches!(self, Launch::Extern(_))
    }

    pub fn entry(&self) -> &str {
        match self {
            Launch::Kernel(k) => &k.entry,
            Launch::Extern(e) => &e.entry,
        }
    }

    /// The pinned module, for a kernel launch.
    pub fn module(&self) -> Option<&str> {
        match self {
            Launch::Kernel(k) => Some(&k.module),
            Launch::Extern(_) => None,
        }
    }

    pub fn kernel(&self) -> Option<&KernelLaunch> {
        match self {
            Launch::Kernel(k) => Some(k),
            Launch::Extern(_) => None,
        }
    }

    /// The launch ABI, defaulting to the op's interface.
    pub fn params_of<'a>(&'a self, op: &'a Op) -> &'a [ParamType] {
        let own = match self {
            Launch::Kernel(k) => &k.params,
            Launch::Extern(e) => &e.params,
        };
        own.as_deref().unwrap_or(&op.params)
    }

    /// The wiring, defaulting to forwarding the op's params in order.
    pub fn args_of(&self, op: &Op) -> Cow<'_, [LaunchArg]> {
        let own = match self {
            Launch::Kernel(k) => &k.args,
            Launch::Extern(e) => &e.args,
        };
        match own {
            Some(a) => Cow::Borrowed(a),
            None => Cow::Owned((0..op.params.len()).map(|param| LaunchArg::Param { param }).collect()),
        }
    }
}

/// A launch argument: a forwarded op param, a scratch buffer, a literal, or this rank's index in a group, e.g. `{"param": 0}`, `{"scratch": "pmax"}`, `{"i32": 64}`, `{"rank": "ep"}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged, deny_unknown_fields)]
pub enum LaunchArg {
    /// Forward the op's param at this index.
    Param { param: usize },
    /// Pass the named scratch buffer.
    Scratch { scratch: String },
    /// A literal `i32`.
    I32 { i32: i32 },
    /// A literal `i64`.
    I64 { i64: i64 },
    /// A literal `f32`.
    F32 { f32: f32 },
    /// A literal `u8`.
    U8 { u8: u8 },
    /// This rank's index in the named topology group, a load-time constant.
    Rank { rank: String },
    /// An aggregate assembled from fields at byte offsets, for a `bytes<n>` param.
    Pack { pack: Pack },
}

/// An aggregate param image, e.g. `{"size": 24, "fields": [{"at": 0, "param": 3}, {"at": 8, "i32": 64}, {"at": 12, "var": "tokens"}]}`:
/// the bytes a kernel compiled from a struct-taking language (CUTLASS, the CuTe DSL) reads as one parameter.
///
/// Bytes no field covers are zero. A pointer field carries the bound buffer's
/// address (offset honoured), a scalar field its value in the field's width.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Pack {
    /// Image size in bytes; equals the `bytes<n>` of the param it binds to.
    pub size: u32,
    /// The fields, each at its byte offset.
    pub fields: Vec<Field>,
}

/// One field of a `pack`: a byte offset plus where the value comes from, e.g. `{"at": 8, "var": "tokens"}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Field {
    /// Byte offset of the field in the image.
    pub at: u32,
    /// Field width in bytes; defaults to the source's own width (8 for a
    /// pointer or i64, 4 for i32 / f32 / var / expr, 1 for u8).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    /// Where the value comes from.
    #[serde(flatten)]
    pub src: FieldSrc,
}

/// The value of a pack field, e.g. `{"param": 3}` (the bound buffer's pointer or the scalar's value), `{"var": "tokens"}`, `{"i64": 4608}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged, deny_unknown_fields)]
pub enum FieldSrc {
    /// The op's interface param: a pointer (with the call's offset) or a scalar.
    Param { param: usize },
    /// A scratch buffer's pointer.
    Scratch { scratch: String },
    /// A literal `i32`.
    I32 { i32: i32 },
    /// A literal `i64`.
    I64 { i64: i64 },
    /// A literal `f32`.
    F32 { f32: f32 },
    /// A literal `u8`.
    U8 { u8: u8 },
    /// The current value of a var.
    Var { var: String },
    /// The value of a var expression.
    Expr { expr: Expr },
    /// This rank's index in the named topology group.
    Rank { rank: String },
    /// A 128-byte TMA descriptor over an interface buffer or state, encoded
    /// at load time; sits at a 64-byte aligned offset (a bare `CUtensorMap`
    /// param is a `bytes<128>` with this field at 0, a cute `TiledCopy` has
    /// its dynamic strides after it).
    TensorMap { tensormap: TensorMap },
}

impl Field {
    /// The field's width: explicit, else the source's natural width
    /// (`None` for a param source, whose width is the param's).
    fn natural_width(&self) -> Option<u32> {
        self.width.or(match &self.src {
            FieldSrc::Param { .. } => None,
            FieldSrc::Scratch { .. } | FieldSrc::I64 { .. } => Some(8),
            FieldSrc::I32 { .. }
            | FieldSrc::F32 { .. }
            | FieldSrc::Var { .. }
            | FieldSrc::Expr { .. }
            | FieldSrc::Rank { .. } => Some(4),
            FieldSrc::U8 { .. } => Some(1),
            FieldSrc::TensorMap { .. } => Some(128),
        })
    }
}

impl Pack {
    /// Structural diagnostics: every field inside the image and no two
    /// overlapping. `width_of` supplies a param field's width.
    pub(crate) fn check(&self, width_of: impl Fn(usize) -> Option<u32>) -> Vec<String> {
        let mut errs = Vec::new();
        let mut spans: Vec<(u32, u32, usize)> = Vec::new();
        for (i, f) in self.fields.iter().enumerate() {
            let w = match f.natural_width().or_else(|| match &f.src {
                FieldSrc::Param { param } => width_of(*param),
                _ => None,
            }) {
                Some(w) if w > 0 => w,
                _ => {
                    errs.push(format!("field #{i}: width unknown"));
                    continue;
                }
            };
            if matches!(f.src, FieldSrc::TensorMap { .. }) {
                if w != 128 {
                    errs.push(format!("field #{i}: a tensormap field is 128 bytes, not {w}"));
                }
                if !f.at.is_multiple_of(64) {
                    errs.push(format!("field #{i}: tensormap at {} is not 64-byte aligned", f.at));
                }
            }
            match f.at.checked_add(w) {
                Some(end) if end <= self.size => spans.push((f.at, end, i)),
                _ => errs.push(format!("field #{i} at {} spans {w} bytes, past the {} byte image", f.at, self.size)),
            }
        }
        spans.sort_unstable();
        for w in spans.windows(2) {
            if w[1].0 < w[0].1 {
                errs.push(format!("field #{} overlaps field #{}", w[1].2, w[0].2));
            }
        }
        errs
    }
}

/// A tiled TMA descriptor (`cuTensorMapEncodeTiled`) over the buffer bound to interface param `param`, e.g. `{"param": 5, "dtype": "u8", "dims": [3584, 1024], "strides": [3584], "box": [128, 96], "swizzle": 128}`.
///
/// `dims` are in elements, innermost first; `strides` are the byte strides of
/// `dims[1..]`; `box` is the smem tile in elements. Element strides are 1.
/// The outermost dim may be 0: the descriptor then spans as many of that dim
/// as the bound buffer holds past the call's offset (a paged cache whose
/// page count is the runtime's, not the manifest's).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TensorMap {
    /// The interface buffer param the descriptor addresses (the call's offset is honoured).
    pub param: usize,
    /// Element type as TMA sees it; `u4` is 16 packed nibbles per 8 bytes (`CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B`).
    pub dtype: TmaDType,
    /// Global extent per dimension in elements, innermost first (1 to 5 dims); the outermost may be 0 (span the buffer).
    pub dims: Vec<u64>,
    /// Byte stride of each dimension after the first (`dims.len() - 1` entries, multiples of 16).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub strides: Vec<u64>,
    /// Box (smem tile) extent per dimension in elements, each 1..=256.
    #[serde(rename = "box")]
    pub box_: Vec<u32>,
    /// Shared-memory swizzle span in bytes: 0 (none), 32, 64 or 128.
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub swizzle: u32,
    /// L2 promotion in bytes: 0 (none), 64, 128 or 256.
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub l2_promotion: u32,
    /// Fill out-of-bounds elements with NaN instead of zero.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub oob_nan: bool,
}

fn is_zero_u32(v: &u32) -> bool {
    *v == 0
}

/// TMA element types the runtime encodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum TmaDType {
    U8,
    U16,
    U32,
    I32,
    U64,
    I64,
    F16,
    Bf16,
    F32,
    /// Float32 storage converted to TF32 by TMA.
    Tf32,
    /// 4-bit elements, 16 per 8-byte unit (`16U4_ALIGN16B`); dims and box count nibbles.
    U4,
    /// Packed 4-bit shared-memory output (`16U4_ALIGN8B`).
    U4packed,
}

impl TmaDType {
    /// Element size in bits.
    fn bits(self) -> u64 {
        match self {
            TmaDType::U4 | TmaDType::U4packed => 4,
            TmaDType::U8 => 8,
            TmaDType::U16 | TmaDType::F16 | TmaDType::Bf16 => 16,
            TmaDType::U32 | TmaDType::I32 | TmaDType::F32 | TmaDType::Tf32 => 32,
            TmaDType::U64 | TmaDType::I64 => 64,
        }
    }
}

impl TensorMap {
    /// Structural checks that need no buffer: rank, box limits, stride and
    /// swizzle alignment. Returns every violation.
    pub(crate) fn check(&self) -> Vec<String> {
        let mut errs = Vec::new();
        let n = self.dims.len();
        if !(1..=5).contains(&n) {
            errs.push(format!("dims has {n} entries, expected 1 to 5"));
            return errs;
        }
        if self.strides.len() != n - 1 {
            errs.push(format!("strides has {} entries, expected dims - 1 = {}", self.strides.len(), n - 1));
        }
        if self.box_.len() != n {
            errs.push(format!("box has {} entries, expected {n}", self.box_.len()));
        }
        if let Some((i, _)) = self.dims[..n - 1].iter().enumerate().find(|(_, &d)| d == 0) {
            errs.push(format!("dims[{i}] is 0; only the outermost dim may span the buffer"));
        }
        if n == 1 && self.dims[0] == 0 {
            errs.push("dims[0] is 0".into());
        }
        if let Some((i, &b)) = self.box_.iter().enumerate().find(|(_, &b)| !(1..=256).contains(&b)) {
            errs.push(format!("box[{i}] = {b}, must be 1..=256"));
        }
        if let Some((i, &s)) = self.strides.iter().enumerate().find(|(_, &s)| s % 16 != 0 || s == 0) {
            errs.push(format!("strides[{i}] = {s} is not a positive multiple of 16"));
        }
        let bits = self.dtype.bits();
        if !(self.dims[0] * bits).is_multiple_of(128) {
            errs.push(format!("dims[0] = {} elements is not a multiple of 16 bytes", self.dims[0]));
        }
        if let Some(&b0) = self.box_.first() {
            let inner = b0 as u64 * bits / 8;
            if !inner.is_multiple_of(16) {
                errs.push(format!("box[0] = {b0} elements spans {inner} bytes, not a multiple of 16"));
            }
            if self.swizzle != 0 && inner > self.swizzle as u64 {
                errs.push(format!("box[0] spans {inner} bytes, more than the {} byte swizzle span", self.swizzle));
            }
        }
        if ![0, 32, 64, 128].contains(&self.swizzle) {
            errs.push(format!("swizzle {} is not one of 0, 32, 64, 128", self.swizzle));
        }
        if ![0, 64, 128, 256].contains(&self.l2_promotion) {
            errs.push(format!("l2_promotion {} is not one of 0, 64, 128, 256", self.l2_promotion));
        }
        errs
    }

    /// Bytes the descriptor can address from its base: the last element's
    /// end. `None` if the shape is malformed (see [`Self::check`]).
    pub fn footprint(&self) -> Option<u64> {
        if self.dims.is_empty() || self.strides.len() + 1 != self.dims.len() {
            return None;
        }
        let mut bytes = self.dims[0].checked_mul(self.dtype.bits())?.div_ceil(8);
        for (d, s) in self.dims[1..].iter().zip(&self.strides) {
            bytes = bytes.checked_add(d.checked_sub(1)?.checked_mul(*s)?)?;
        }
        Some(bytes)
    }
}

impl fmt::Display for LaunchArg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LaunchArg::Param { param } => write!(f, "interface param #{param}"),
            LaunchArg::Scratch { scratch } => write!(f, "scratch `{scratch}`"),
            LaunchArg::I32 { i32: v } => write!(f, "i32 literal {v}"),
            LaunchArg::I64 { i64: v } => write!(f, "i64 literal {v}"),
            LaunchArg::F32 { f32: v } => write!(f, "f32 literal {v}"),
            LaunchArg::U8 { u8: v } => write!(f, "u8 literal {v}"),
            LaunchArg::Rank { rank } => write!(f, "rank in group `{rank}`"),
            LaunchArg::Pack { pack } => write!(f, "pack of {} bytes", pack.size),
        }
    }
}

/// A `source` of the form `hf:<org>/<repo>/<path>[@<revision>]` (revision defaults to `main`), fetched into a content-addressed cache at load time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegistryRef {
    pub org: String,
    pub repo: String,
    pub path: String,
    pub revision: String,
}

impl RegistryRef {
    /// `None` if `s` is a plain local file name (no registry prefix);
    /// otherwise the parsed ref or why it is malformed.
    pub fn parse(s: &str) -> Option<Result<RegistryRef, String>> {
        let rest = s.strip_prefix("hf:")?;
        let malformed = || format!("invalid registry ref `{s}`: expected hf:<org>/<repo>/<path>[@revision]");
        let (rest, revision) = match rest.rsplit_once('@') {
            Some((r, rev)) if !rev.is_empty() && !rev.contains('/') => (r, rev),
            Some(_) => return Some(Err(malformed())),
            None => (rest, "main"),
        };
        let Some((org, rest)) = rest.split_once('/') else {
            return Some(Err(malformed()));
        };
        let Some((repo, path)) = rest.split_once('/') else {
            return Some(Err(malformed()));
        };
        if org.is_empty()
            || repo.is_empty()
            || path.is_empty()
            || path.split('/').any(|seg| seg.is_empty() || seg == "." || seg == "..")
        {
            return Some(Err(malformed()));
        }
        Some(Ok(RegistryRef {
            org: org.to_string(),
            repo: repo.to_string(),
            path: path.to_string(),
            revision: revision.to_string(),
        }))
    }
}

/// A scalar expression: a constant, a var name, `{"ceil_div": [e, c]}` or `{"mul": [e, c]}`, e.g. `{"ceil_div": ["tokens", 128]}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum Expr {
    /// A constant, e.g. `64`.
    Const(u64),
    /// A var by name, e.g. `"tokens"`.
    Var(String),
    /// `ceil(e / c)`, e.g. `{"ceil_div": ["tokens", 128]}`.
    CeilDiv { ceil_div: (Box<Expr>, u64) },
    /// `e * c`, e.g. `{"mul": ["tokens", 32]}`.
    Mul { mul: (Box<Expr>, u64) },
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EvalError {
    #[error("unknown var `{0}`")]
    UnknownVar(String),
    #[error("arithmetic overflow")]
    Overflow,
    #[error("division by zero")]
    DivByZero,
}

impl Expr {
    pub fn eval(&self, vars: &BTreeMap<String, u64>) -> Result<u64, EvalError> {
        match self {
            Expr::Const(c) => Ok(*c),
            Expr::Var(var) => vars.get(var).copied().ok_or_else(|| EvalError::UnknownVar(var.clone())),
            Expr::CeilDiv { ceil_div: (inner, c) } => {
                if *c == 0 {
                    return Err(EvalError::DivByZero);
                }
                let x = inner.eval(vars)?;
                Ok(x.checked_add(c - 1).ok_or(EvalError::Overflow)? / c)
            }
            Expr::Mul { mul: (inner, c) } => inner.eval(vars)?.checked_mul(*c).ok_or(EvalError::Overflow),
        }
    }
}

/// One op call, e.g. `{"label": "l0.attn", "op": "attn", "args": [{"buf": "q"}, {"state": "kv"}, ...]}`.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Call {
    /// Human-readable name for diagnostics, e.g. `"l0.attn"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Name of the op called, e.g. `"attn"`.
    pub op: String,
    /// One argument per interface param, in order.
    pub args: Vec<Arg>,
}

/// A call argument, e.g. `{"buf": "hidden"}`, `{"state": "kv", "offset": 65536}`, `{"var": "tokens"}`, `{"i32": 2560}`, `{"rank": "ep"}`.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged, deny_unknown_fields)]
pub enum Arg {
    /// A buffer plus an optional byte offset into it, e.g. the v slice of a fused qkv buffer.
    Buf {
        /// Buffer name, e.g. `"qkv"`.
        buf: String,
        /// Byte offset added to the base pointer (default `0`), e.g. `4096`.
        #[serde(default, skip_serializing_if = "is_zero")]
        offset: u64,
    },
    /// A state plus an optional byte offset into it, e.g. one layer's region of a KV cache.
    State {
        /// State name, e.g. `"kv"`.
        state: String,
        /// Byte offset added to the base pointer (default `0`), e.g. `65536`.
        #[serde(default, skip_serializing_if = "is_zero")]
        offset: u64,
    },
    /// The current value of a var, e.g. `{"var": "tokens"}`.
    Var { var: String },
    /// The value of a var expression, e.g. `{"expr": {"mul": ["tokens", 32]}}`.
    Expr { expr: Expr },
    /// A literal `i32`.
    I32 { i32: i32 },
    /// A literal `i64`.
    I64 { i64: i64 },
    /// A literal `f32`.
    F32 { f32: f32 },
    /// A literal `u8`.
    U8 { u8: u8 },
    /// This rank's index in the named topology group, a load-time constant, e.g. `{"rank": "ep"}`.
    Rank { rank: String },
}

impl fmt::Display for Arg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Arg::Buf { buf, offset: 0 } => write!(f, "buffer `{buf}`"),
            Arg::Buf { buf, offset } => write!(f, "buffer `{buf}`+{offset}"),
            Arg::State { state, offset: 0 } => write!(f, "state `{state}`"),
            Arg::State { state, offset } => write!(f, "state `{state}`+{offset}"),
            Arg::Var { var } => write!(f, "var `{var}`"),
            Arg::Expr { .. } => write!(f, "expression"),
            Arg::I32 { i32: v } => write!(f, "i32 literal {v}"),
            Arg::I64 { i64: v } => write!(f, "i64 literal {v}"),
            Arg::F32 { f32: v } => write!(f, "f32 literal {v}"),
            Arg::U8 { u8: v } => write!(f, "u8 literal {v}"),
            Arg::Rank { rank } => write!(f, "rank in group `{rank}`"),
        }
    }
}
