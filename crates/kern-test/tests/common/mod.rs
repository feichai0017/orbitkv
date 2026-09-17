//! The fake side: a host interpreter over a tiny manifest, and the fixture
//! family it runs. The model is two layers of `scale` (elementwise, the op
//! under swap) and `mix` (a recurrence over a per-token state: not
//! idempotent, reads the previous token's slot), an `embed` in front and a
//! `head` behind that writes bf16 logits and the argmax token. Every
//! behaviour is keyed by the launch entry the manifest names, so a B
//! variant is a manifest edit and nothing else — the way a real swap is.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::ops::Range;

use anyhow::{bail, Result};
use kern_manifest::protocol::Axis;
use kern_manifest::types::{Arg, DType, Dim, Fill, Provision};
use kern_manifest::values;
use kern_manifest::{verify, Manifest, Protocol, Verified};
use kern_test::compare::{changed_blocks, compare, logit_stats, Cmp, LogitStats};
use kern_test::{At, Options, Side, Vars};

pub const D: usize = 4;
pub const VOCAB: usize = 16;
pub const LAYERS: usize = 2;
pub const CAPACITY: u64 = 16;
pub const SLAB: usize = 4096;

/// One manifest of the family: which entry each op launches, and the
/// structural variations.
#[derive(Clone, Debug)]
pub struct Fixture {
    pub scale: &'static str,
    pub mix: &'static str,
    pub head: &'static str,
    /// At this layer, the `scale` call goes to a second op `scale_alt`
    /// launching this entry: a change at one layer only.
    pub alt: Option<(usize, &'static str)>,
    /// Name of the logits buffer; anything not starting with `logits` is
    /// no oracle.
    pub logits: &'static str,
    /// An extra program without a `batch` (a harness-only layer): changed
    /// but undriven.
    pub probe: bool,
    /// Ranks of the `ep` group the manifest is SPMD over; above 1 the
    /// `scale` op takes the rank as an argument.
    pub ranks: usize,
    /// A `gather` program over a `peer` buffer of `hidden`, for the diff
    /// (its op is a module kernel the fake never runs).
    pub peer: bool,
    /// A `prep` program run once after load that fills a carry `table`,
    /// through the entry named.
    pub once: Option<&'static str>,
    /// `act` declared one column wider: the same name, another shape.
    pub wide_act: bool,
    /// A 4 KB `slab` carry the `scale` op writes one 64-byte block of per
    /// layer (a persistent kernel's workspace: big, touched in places).
    pub slab: bool,
}

impl Default for Fixture {
    fn default() -> Self {
        Fixture {
            scale: "scale",
            mix: "mix",
            head: "head",
            alt: None,
            logits: "logits",
            probe: false,
            ranks: 1,
            peer: false,
            once: None,
            wide_act: false,
            slab: false,
        }
    }
}

impl Fixture {
    pub fn scale(self, e: &'static str) -> Self {
        Fixture { scale: e, ..self }
    }
    pub fn mix(self, e: &'static str) -> Self {
        Fixture { mix: e, ..self }
    }
    pub fn head(self, e: &'static str) -> Self {
        Fixture { head: e, ..self }
    }
    pub fn alt(self, layer: usize, e: &'static str) -> Self {
        Fixture { alt: Some((layer, e)), ..self }
    }
    pub fn logits(self, name: &'static str) -> Self {
        Fixture { logits: name, ..self }
    }
    pub fn probe(self) -> Self {
        Fixture { probe: true, ..self }
    }
    pub fn ranks(self, n: usize) -> Self {
        Fixture { ranks: n, ..self }
    }
    pub fn peer(self) -> Self {
        Fixture { peer: true, ..self }
    }
    pub fn once(self, e: &'static str) -> Self {
        Fixture { once: Some(e), ..self }
    }
    pub fn wide_act(self) -> Self {
        Fixture { wide_act: true, ..self }
    }
    pub fn slab(self) -> Self {
        Fixture { slab: true, ..self }
    }

    pub fn manifest(&self) -> Verified {
        let op = |params: &[&str], entry: &str| serde_json::json!({"params": params, "impl": {"launches": [{"entry": format!("extern:{entry}")}]}});
        let mut ops = serde_json::Map::new();
        ops.insert("embed".into(), op(&["in buffer<i64>", "out buffer<f32>"], "embed"));
        let ranked = self.ranks > 1;
        let mut scale_params: Vec<&str> = vec!["in buffer<f32>", "out buffer<f32>"];
        if ranked {
            scale_params.push("i32");
        }
        if self.slab {
            scale_params.extend(["inout buffer<u8>", "i32"]);
        }
        let scale_params = &scale_params[..];
        ops.insert("scale".into(), op(scale_params, self.scale));
        if let Some((_, e)) = self.alt {
            ops.insert("scale_alt".into(), op(scale_params, e));
        }
        ops.insert(
            "mix".into(),
            op(
                &["in buffer<f32>", "in buffer<i64>", "in buffer<i32>", "inout state", "i32", "out buffer<f32>"],
                self.mix,
            ),
        );
        ops.insert("head".into(), op(&["in buffer<f32>", "out buffer<bf16>", "out buffer<i64>"], self.head));
        let mut calls = vec![serde_json::json!({"op": "embed", "args": [{"buf": "token_ids"}, {"buf": "hidden"}]})];
        for l in 0..LAYERS {
            let sc = match self.alt {
                Some((k, _)) if k == l => "scale_alt",
                _ => "scale",
            };
            let mut args = vec![serde_json::json!({"buf": "hidden"}), serde_json::json!({"buf": "act"})];
            if ranked {
                args.push(serde_json::json!({"rank": "ep"}));
            }
            if self.slab {
                args.extend([serde_json::json!({"buf": "slab"}), serde_json::json!({"i32": l})]);
            }
            calls.push(serde_json::json!({"op": sc, "args": args}));
            calls.push(serde_json::json!({"op": "mix", "args": [
                {"buf": "act"}, {"buf": "slot_mapping"}, {"buf": "seq_lens"}, {"state": "kv"}, {"i32": l},
                {"buf": "hidden"}]}));
        }
        calls.push(serde_json::json!({"op": "head", "args": [
            {"buf": "hidden"}, {"buf": self.logits}, {"buf": "next_token"}]}));
        let mut programs = serde_json::json!({
            "prefill": {"batch": {"groups": 1, "rows": "tokens"}, "calls": calls},
            "decode": {"batch": {"groups": 1, "rows": 1}, "calls": calls},
        });
        if self.probe {
            programs["probe"] = serde_json::json!({"calls": calls});
        }
        if let Some(e) = self.once {
            ops.insert("fill_table".into(), op(&["out buffer<f32>"], e));
            programs["prep"] =
                serde_json::json!({"once": true, "calls": [{"op": "fill_table", "args": [{"buf": "table"}]}]});
        }
        let mut modules = serde_json::json!({});
        if self.peer {
            modules["toy"] = serde_json::json!({"source": "toy.cubin", "sha256": "ab".repeat(32)});
            ops.insert("gather".into(), serde_json::json!({
                "params": ["in buffer<f32>", "in buffer<u64>", "out buffer<f32>"],
                "impl": {"launches": [{"module": "toy", "entry": "gather_k", "block": [128, 1, 1], "grid": [1, 1, 1]}]},
            }));
            programs["gather"] = serde_json::json!({"calls": [
                calls[0],
                {"op": "gather", "args": [{"buf": "hidden"}, {"buf": "hidden_peers"}, {"buf": "act"}]}]});
        }
        let mut m = serde_json::json!({
            "schema_version": 5, "model": "fake",
            "vars": {"tokens": {"max": 8}, "seqs": {"max": 1}},
            "states": {"kv": {"bytes_per_token": LAYERS * D * 4}},
            "buffers": {
                "token_ids": {"kind": "input", "dtype": "i64", "shape": ["tokens"], "fill": "token",
                              "domain": {"min": 0, "max": VOCAB - 1}},
                "slot_mapping": {"kind": "input", "dtype": "i64", "shape": ["tokens"], "fill": "slot",
                                 "domain": {"index_into": "kv"}},
                "seq_lens": {"kind": "input", "dtype": "i32", "shape": ["seqs"], "fill": "seq_len"},
                "hidden": {"kind": "workspace", "dtype": "f32", "shape": ["tokens", D]},
                "act": {"kind": "workspace", "dtype": "f32", "shape": ["tokens", if self.wide_act { D + 1 } else { D }]},
                self.logits: {"kind": "output", "dtype": "bf16", "shape": ["tokens", VOCAB]},
                "next_token": {"kind": "output", "dtype": "i64", "shape": ["seqs"], "fill": "tokens",
                               "domain": {"min": 0, "max": VOCAB - 1}},
            },
            "modules": modules,
            "ops": ops,
            "programs": programs,
        });
        if ranked {
            m["topology"] = serde_json::json!({"groups": {"ep": self.ranks}});
        }
        if self.once.is_some() {
            m["buffers"]["table"] = serde_json::json!({"kind": "carry", "dtype": "f32", "shape": [D]});
        }
        if self.slab {
            m["buffers"]["slab"] = serde_json::json!({"kind": "carry", "dtype": "u8", "shape": [SLAB]});
        }
        if self.peer {
            m["buffers"]["hidden"]["export"] = true.into();
            m["buffers"]["hidden_peers"] = serde_json::json!(
                {"kind": "peer", "dtype": "u64", "shape": [self.ranks], "of": "hidden", "group": "ep"});
        }
        let m: Manifest = serde_json::from_value(m).expect("fixture parses");
        verify(m).unwrap_or_else(|e| panic!("fixture does not verify: {e}"))
    }
}

/// The interpreter over one manifest of the family: one [`Rank`] per
/// member of its `ep` group, all fed the same sequence.
pub struct Fake {
    m: Verified,
    protocol: Protocol,
    ranks: Vec<Rank>,
    pos: i64,
}

/// One rank's memory.
struct Rank {
    q: usize,
    bufs: BTreeMap<String, Vec<u8>>,
    states: BTreeMap<String, Vec<u8>>,
    /// How often `scale_noisy` has seen each input: it flips sign on every
    /// repeat, so no replay reproduces the original.
    seen: BTreeMap<Vec<u8>, u64>,
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
}

fn bytes_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn i64s(b: &[u8]) -> Vec<i64> {
    b.as_chunks::<8>().0.iter().map(|c| i64::from_le_bytes(*c)).collect()
}

impl Fake {
    pub fn new(m: Verified) -> Fake {
        let protocol = Protocol::check(&m).expect("fixture has a protocol");
        let max: Vars = m.vars.iter().map(|(k, v)| (k.clone(), v.max)).collect();
        let n = m.group_size("ep").unwrap_or(1) as usize;
        let ranks = (0..n)
            .map(|q| Rank {
                q,
                bufs: m
                    .buffers
                    .keys()
                    .map(|n| (n.clone(), vec![0u8; kern_test::diff::live_bytes(&m, n, &max)]))
                    .collect(),
                states: m
                    .states
                    .iter()
                    .map(|(n, s)| (n.clone(), vec![0u8; (s.bytes_per_token * CAPACITY) as usize]))
                    .collect(),
                seen: BTreeMap::new(),
            })
            .collect();
        Fake { m, protocol, ranks, pos: 0 }
    }

    fn rows(&self, vars: &Vars) -> usize {
        vars[&self.protocol.rows.var] as usize
    }
}

impl Rank {
    fn call(&mut self, m: &Verified, t: usize, op: &str, args: &[Arg]) -> Result<()> {
        let entry = m.ops[op].imp.launches[0].entry().to_string();
        let buf = |k: usize| match &args[k] {
            Arg::Buf { buf, .. } => buf.clone(),
            _ => panic!("expected a buffer arg"),
        };
        match entry.as_str() {
            "extern:embed" => {
                let ids = i64s(&self.bufs[&buf(0)][..t * 8]);
                let h: Vec<f32> = ids
                    .iter()
                    .flat_map(|&tok| (0..D).map(move |d| (((tok + 1) * (d as i64 + 1)) % 7 - 3) as f32))
                    .collect();
                self.bufs.get_mut(&buf(1)).unwrap()[..t * D * 4].copy_from_slice(&bytes_f32(&h));
            }
            e if e.starts_with("extern:scale") => {
                let raw = self.bufs[&buf(0)][..t * D * 4].to_vec();
                let x = f32s(&raw);
                // The noisy variant is ±3%, the sign flipping each time the
                // same input comes back: A's own band (6%) always contains
                // its distance to the exact op (3%).
                let seen = self.seen.entry(raw).or_insert(0);
                *seen += 1;
                let noisy = if *seen % 2 == 1 { 1.0 + 2f32.powi(-5) } else { 1.0 - 2f32.powi(-5) };
                let y: Vec<f32> = x
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| match e {
                        "extern:scale" | "extern:scale_same" | "extern:scale_slab_leak" => v * 0.5,
                        "extern:scale_negzero" if v == 0.0 => -0.0,
                        "extern:scale_negzero" => v * 0.5,
                        "extern:scale_round" => v * 0.5 * (1.0 + 2f32.powi(-9)),
                        "extern:scale_drift" => v * 0.5 * 1.25,
                        "extern:scale_wrong" => v * 0.5 + 1.0,
                        "extern:scale_nan" if i == 0 => f32::NAN,
                        "extern:scale_nan" => v * 0.5,
                        "extern:scale_dim3" if i % D == 3 => v * 0.5 + 1.0,
                        "extern:scale_dim3" => v * 0.5,
                        "extern:scale_noisy" => v * 0.5 * noisy,
                        "extern:scale_rank1_wrong" if self.q == 1 => v * 0.5 + 1.0,
                        "extern:scale_rank1_wrong" => v * 0.5,
                        _ => panic!("no such scale: {e}"),
                    })
                    .collect();
                self.bufs.get_mut(&buf(1)).unwrap()[..t * D * 4].copy_from_slice(&bytes_f32(&y));
                // the slab: this layer's block gets the row sums; the leaky
                // variant also stamps a block nobody asked for
                if let Some(Arg::I32 { i32: layer }) = args.iter().find(|a| matches!(a, Arg::I32 { .. })) {
                    let slab = args.iter().find_map(|a| match a {
                        Arg::Buf { buf, .. } if buf == "slab" => Some(buf.clone()),
                        _ => None,
                    });
                    if let Some(slab) = slab {
                        let sums: Vec<u8> = (0..64).map(|i| (y[i % y.len()] as i64 + i as i64) as u8).collect();
                        let s = self.bufs.get_mut(&slab).unwrap();
                        let at = *layer as usize * 64;
                        s[at..at + 64].copy_from_slice(&sums);
                        if e == "extern:scale_slab_leak" {
                            s[2048..2048 + 64].copy_from_slice(&sums);
                        }
                    }
                }
            }
            e if e.starts_with("extern:mix") => {
                let act = f32s(&self.bufs[&buf(0)][..t * D * 4]);
                let slots = i64s(&self.bufs[&buf(1)][..t * 8]);
                let Arg::State { state, .. } = &args[3] else { panic!("expected a state arg") };
                let Arg::I32 { i32: layer } = args[4] else { panic!("expected the layer") };
                let cell = |slot: i64| (slot as usize * LAYERS + layer as usize) * D * 4;
                let st = self.states.get_mut(state).unwrap();
                let mut h = vec![0f32; t * D];
                for (r, &slot) in slots.iter().enumerate() {
                    let c = cell(slot);
                    let mut cur = f32s(&st[c..c + D * 4]);
                    for d in 0..D {
                        cur[d] += act[r * D + d];
                    }
                    st[c..c + D * 4].copy_from_slice(&bytes_f32(&cur));
                    if e == "extern:mix_leak" && ((slot + 1) as u64) < CAPACITY {
                        let n = cell(slot + 1);
                        st[n..n + D * 4].copy_from_slice(&bytes_f32(&act[r * D..(r + 1) * D]));
                    }
                    let prev = if slot > 0 { f32s(&st[cell(slot - 1)..cell(slot - 1) + D * 4]) } else { vec![0.0; D] };
                    for d in 0..D {
                        h[r * D + d] = cur[d] + 0.5 * prev[d];
                    }
                }
                self.bufs.get_mut(&buf(5)).unwrap()[..t * D * 4].copy_from_slice(&bytes_f32(&h));
            }
            e if e.starts_with("extern:head") => {
                let h = f32s(&self.bufs[&buf(0)][..t * D * 4]);
                let w = |d: usize, v: usize| (((3 * d + 5 * v + d * v) % 17) % 5) as f32 - 2.0;
                let mut logits = vec![0f32; t * VOCAB];
                for r in 0..t {
                    for v in 0..VOCAB {
                        logits[r * VOCAB + v] = (0..3).map(|d| h[r * D + d] * w(d, v)).sum();
                    }
                    if e == "extern:head_swap" {
                        let row = &mut logits[r * VOCAB..(r + 1) * VOCAB];
                        let (top1, m1) =
                            row.iter().enumerate().fold((0, f32::MIN), |m, (i, &x)| if x > m.1 { (i, x) } else { m });
                        let (top2, m2) = row
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| *i != top1)
                            .fold((0, f32::MIN), |m, (i, &x)| if x > m.1 { (i, x) } else { m });
                        let margin = m1 - m2;
                        row[top1] -= 0.6 * margin;
                        row[top2] += 0.6 * margin;
                    }
                }
                let last = &logits[(t - 1) * VOCAB..t * VOCAB];
                let argmax =
                    last.iter().enumerate().fold((0, f32::MIN), |m, (i, &x)| if x > m.1 { (i, x) } else { m }).0;
                let tok = if e == "extern:head_bad" { 99 } else { argmax as i64 };
                let lg = values::from_f64(DType::Bf16, &logits.iter().map(|&x| x as f64).collect::<Vec<_>>());
                self.bufs.get_mut(&buf(1)).unwrap()[..lg.len()].copy_from_slice(&lg);
                self.bufs.get_mut(&buf(2)).unwrap()[..8].copy_from_slice(&tok.to_le_bytes());
            }
            e => bail!("no such op: {e}"),
        }
        Ok(())
    }
}

impl Fake {
    fn at<'a>(&'a self, rank: usize, at: At<'a, Vec<u8>>) -> &'a [u8] {
        match at {
            At::Buffer(name, r) => &self.ranks[rank].bufs[name][r],
            At::State(name, r) => &self.ranks[rank].states[name][r],
            At::Scratch(b, r) => &b[r],
        }
    }
}

impl Side for Fake {
    type Buf = Vec<u8>;

    fn manifest(&self) -> &Verified {
        &self.m
    }
    fn provision(&self) -> Provision {
        Provision { tokens: CAPACITY, seq_slots: 0 }
    }
    fn page(&self) -> u64 {
        4
    }
    fn vocab(&self) -> u64 {
        VOCAB as u64
    }
    fn ranks(&self) -> usize {
        self.ranks.len()
    }
    fn stage(&mut self, ids: &[i64]) -> Result<Vars> {
        let c = ids.len();
        let pos = self.pos;
        let vars = self.protocol.vars(1, c as u64, c as u64);
        let p = self.protocol.clone();
        let mut put = |f: &kern_manifest::protocol::Filled, v: &[i64]| {
            let b = f.encode(v);
            for r in &mut self.ranks {
                r.bufs.get_mut(&f.name).unwrap()[..b.len()].copy_from_slice(&b);
            }
        };
        put(p.token_rows(), ids);
        put(p.slots(), &(pos..pos + c as i64).collect::<Vec<_>>());
        put(p.seq_lens(), &[pos + c as i64]);
        if let Some(f) = p.filled(Fill::Position, Axis::Rows) {
            put(f, &(pos..pos + c as i64).collect::<Vec<_>>());
        }
        Ok(vars)
    }
    fn reset(&mut self) {
        self.pos = 0;
    }
    fn advance(&mut self, n: u64) {
        self.pos += n as i64;
    }
    fn calls(&self, program: &str) -> Result<usize> {
        Ok(self.m.programs[program].calls.len())
    }
    fn run(&mut self, program: &str, vars: &Vars, calls: Range<usize>) -> Result<()> {
        let cs = self.m.programs[program].calls[calls].to_vec();
        let t = self.rows(vars);
        for r in &mut self.ranks {
            for c in &cs {
                r.call(&self.m, t, &c.op, &c.args)?;
            }
        }
        Ok(())
    }
    fn read(&self, rank: usize, buffer: &str, bytes: usize) -> Result<Vec<u8>> {
        Ok(self.ranks[rank].bufs[buffer][..bytes].to_vec())
    }
    fn save(&self, rank: usize, buffer: &str, at: Range<usize>) -> Result<Vec<u8>> {
        Ok(self.ranks[rank].bufs[buffer][at].to_vec())
    }
    fn load(&mut self, rank: usize, buffer: &str, at: Range<usize>, from: &Vec<u8>) -> Result<()> {
        let n = at.len();
        self.ranks[rank].bufs.get_mut(buffer).unwrap()[at].copy_from_slice(&from[..n]);
        Ok(())
    }
    fn bytes(&self, _rank: usize, from: &Vec<u8>, at: Range<usize>) -> Result<Vec<u8>> {
        Ok(from[at].to_vec())
    }
    fn compare(&self, rank: usize, dtype: DType, a: At<Vec<u8>>, b: At<Vec<u8>>) -> Result<Cmp> {
        Ok(compare(dtype, self.at(rank, a), self.at(rank, b)))
    }
    fn changed(&self, rank: usize, a: At<Vec<u8>>, b: At<Vec<u8>>) -> Result<Vec<Range<usize>>> {
        Ok(changed_blocks(self.at(rank, a), self.at(rank, b)))
    }
    fn logits(
        &self,
        rank: usize,
        dtype: DType,
        cols: usize,
        a: At<Vec<u8>>,
        b: At<Vec<u8>>,
    ) -> Result<Vec<LogitStats>> {
        let row = cols * dtype.bytes() as usize;
        let (x, y) = (self.at(rank, a), self.at(rank, b));
        Ok(x.chunks_exact(row).zip(y.chunks_exact(row)).map(|(p, q)| logit_stats(dtype, p, q)).collect())
    }
    fn state_bytes(&self, state: &str) -> Result<usize> {
        Ok(self.ranks[0].states[state].len())
    }
    fn read_state(&self, rank: usize, state: &str, at: Range<usize>) -> Result<Vec<u8>> {
        Ok(self.ranks[rank].states[state][at].to_vec())
    }
    fn write_state(&mut self, rank: usize, state: &str, at: usize, bytes: &[u8]) -> Result<()> {
        self.ranks[rank].states.get_mut(state).unwrap()[at..at + bytes.len()].copy_from_slice(bytes);
        Ok(())
    }
    fn save_state(&self, rank: usize, state: &str) -> Result<Vec<u8>> {
        Ok(self.ranks[rank].states[state].clone())
    }
    fn load_state(&mut self, rank: usize, state: &str, from: &Vec<u8>) -> Result<()> {
        self.ranks[rank].states.get_mut(state).unwrap().copy_from_slice(from);
        Ok(())
    }
    fn zero_states(&mut self) -> Result<()> {
        for r in &mut self.ranks {
            for s in r.states.values_mut() {
                s.fill(0);
            }
        }
        Ok(())
    }
    fn time(&mut self, _program: &str, _vars: &Vars, calls: Range<usize>, _iters: usize) -> Result<Vec<f32>> {
        Ok(vec![0.01; calls.len()])
    }
    fn time_graph(&mut self, _program: &str, _vars: &Vars, _iters: usize) -> Result<f32> {
        Ok(0.5)
    }
}

/// The options every fixture runs with unless a test says otherwise: a
/// short, fixed workload; no timing.
pub fn options() -> Options {
    Options {
        a: "A".into(),
        b: "B".into(),
        prompt: None,
        prefill: 5,
        decode_steps: 4,
        logit_kl: 0.01,
        chunk: 3,
        iters: 1,
        graph_step: false,
        sweep: false,
        peak_bw: 1.0,
        perf: false,
        noise: true,
        seed: 0x5eed,
    }
}

/// Run A against B with `o`: record A, drop it, replay B. The report and
/// the lines it printed.
pub fn test(a: &Fixture, b: &Fixture, o: &Options) -> Result<(kern_test::Report, Vec<String>)> {
    let (ma, mb) = (a.manifest(), b.manifest());
    let diff = kern_test::diff::diff(&ma, &mb);
    let mut lines = diff.lines();
    let mut out = |ls: &[String]| lines.extend_from_slice(ls);
    let rec = kern_test::record(o, diff, &mb, &mut Fake::new(ma), &mut out)?;
    let report = kern_test::replay(o, rec, &mut Fake::new(mb), &mut out)?;
    lines.extend(report.summary.verdict.lines());
    Ok((report, lines))
}

/// The verdict's code and its first words.
pub fn verdict(r: &kern_test::Report) -> (i32, String) {
    let v = &r.summary.verdict;
    (v.code, v.summary.split(':').next().unwrap_or("").to_string())
}

pub fn dim(d: &Dim) -> u64 {
    match d {
        Dim::Const(c) => *c,
        Dim::Var(_) => 0,
    }
}
