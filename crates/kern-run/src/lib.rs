//! One sequence driven over a manifest's serving protocol. The runtime is
//! model-agnostic and executes blindly; [`Protocol`] is what the manifest
//! declares about calling it (which buffer carries which role, what shape
//! of call each program takes); [`Caller`] is the one sequence, holding
//! one [`Lease`] of the runtime's token slots for its whole life and
//! staging every call through the protocol's fills. Nothing here names a
//! buffer, a program or a var. `kern run` (generation) and `kern test`
//! (A/B evidence) both drive the runtime through it.

#![deny(unsafe_code)]

pub mod bench;
pub mod config;
pub mod inputs;
pub mod probe;
pub mod run;
pub mod server;
pub mod test;
pub mod weights;

pub use inputs::{Given, Inputs};
pub use weights::Weights;

use std::collections::BTreeMap;
use std::sync::LazyLock;
use std::time::Duration;

use anyhow::{bail, ensure, Context, Result};
use kern_manifest::protocol::{Axis, Forward, LineTable, PageTable, Rows};
use kern_manifest::types::Fill;
use kern_manifest::{Protocol, Verified};
use kern_pool::Lease;
use kern_runtime::{GroupRank, PeerHandle, Runtime, Topology};

/// What `kern --version` prints: the crate version, the commit it was built
/// from, and the CUDA API the runtime binds; the three facts a bug report
/// needs. The commit comes from `build.rs`.
pub static VERSION: LazyLock<String> = LazyLock::new(|| {
    let (major, minor) = (kern_runtime::CUDA_API / 1000, kern_runtime::CUDA_API % 1000 / 10);
    format!("{} ({}, cuda {major}.{minor})", env!("CARGO_PKG_VERSION"), env!("KERN_COMMIT"))
});

/// What a checkpoint directory says besides its tensors: `tokenizer.json`
/// and the ids that end generation (`generation_config.json`'s
/// `eos_token_id`, else `config.json`'s; one id or a list). Over several
/// directories ([`Weights::dirs`]) the first tokenizer wins and the eos
/// ids are the union in order.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Checkpoint {
    tokenizer: Option<std::path::PathBuf>,
    stop_tokens: Vec<i64>,
}

fn checkpoint(dirs: &[std::path::PathBuf]) -> Checkpoint {
    let mut c = Checkpoint::default();
    for d in dirs.iter().filter(|p| p.is_dir()) {
        let t = d.join("tokenizer.json");
        if c.tokenizer.is_none() && t.is_file() {
            c.tokenizer = Some(t);
        }
        for id in eos_ids(d) {
            if !c.stop_tokens.contains(&id) {
                c.stop_tokens.push(id);
            }
        }
    }
    c
}

/// The eos ids one HF directory declares: `generation_config.json`'s,
/// else `config.json`'s, else none.
pub fn eos_ids(dir: &std::path::Path) -> Vec<i64> {
    let read = |f: &str| -> Option<serde_json::Value> {
        serde_json::from_str(&std::fs::read_to_string(dir.join(f)).ok()?).ok()
    };
    ["generation_config.json", "config.json"]
        .iter()
        .filter_map(|f| read(f))
        .map(|v| eos_of(&v))
        .find(|ids| !ids.is_empty())
        .unwrap_or_default()
}

/// `eos_token_id` as a list, whether it was written as one id or several.
fn eos_of(v: &serde_json::Value) -> Vec<i64> {
    match v.get("eos_token_id") {
        Some(serde_json::Value::Number(n)) => n.as_i64().into_iter().collect(),
        Some(serde_json::Value::Array(a)) => a.iter().filter_map(|x| x.as_i64()).collect(),
        _ => Vec::new(),
    }
}

pub fn le_bytes_i32(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// The var vars of one call.
pub(crate) type Vars = BTreeMap<String, u64>;

/// The safetensors a `--weights` entry stands for: the file itself, or
/// every `*.safetensors` under a directory (a checkpoint's shards) in
/// name order, mapped read-only. Nothing is read up front: the runtime
/// parses headers and copies each bound segment straight out of the map.
pub fn map_weights(paths: &[std::path::PathBuf]) -> Result<Vec<memmap2::Mmap>> {
    shard_files(paths)?.iter().map(|f| map_file(f)).collect()
}

/// The safetensors files the entries name: a file itself, or every
/// `*.safetensors` under a directory, in name order.
pub fn shard_files(paths: &[std::path::PathBuf]) -> Result<Vec<std::path::PathBuf>> {
    let mut files = Vec::new();
    for p in paths {
        if p.is_dir() {
            let mut shards: Vec<_> = std::fs::read_dir(p)
                .with_context(|| format!("weights dir {}", p.display()))?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|f| f.extension().is_some_and(|x| x == "safetensors"))
                .collect();
            ensure!(!shards.is_empty(), "weights dir {}: no .safetensors in it", p.display());
            shards.sort();
            files.extend(shards);
        } else {
            files.push(p.clone());
        }
    }
    Ok(files)
}

#[allow(unsafe_code)]
fn map_file(f: &std::path::Path) -> Result<memmap2::Mmap> {
    let file = std::fs::File::open(f).with_context(|| format!("weights {}", f.display()))?;
    // Mapped, not read: a 50 GB checkpoint costs no DRAM of its own and
    // two runtimes on one host share the page cache.
    unsafe { memmap2::Mmap::map(&file) }.with_context(|| format!("mapping weights {}", f.display()))
}

/// Stage one call: `rows` rows for each lease, at that sequence's
/// position, `ids` in row order. Returns the call's vars.
///
/// Every fill a caller of one rank produces is written; `blocks` is the
/// tray batch's and only a caller that spans a rank group can fill it.
fn stage(
    rt: &mut Runtime,
    p: &Protocol,
    leases: &[Lease],
    positions: &[usize],
    rows: usize,
    ids: &[i64],
) -> Result<Vars> {
    let b = leases.len();
    let vars = p.vars(b as u64, rows as u64, (b * rows) as u64);
    for f in &p.fills {
        let v: Vec<i64> = match (f.fill, f.axis) {
            // Each sequence's first token: the anchor a drafting program
            // splices its own rows from.
            (Fill::Token, Axis::Groups) => ids.chunks(rows).map(|x| x[0]).collect(),
            (Fill::Token, _) => ids.to_vec(),
            (Fill::Valid, _) => vec![1; b * rows],
            (Fill::Position, _) => positions.iter().flat_map(|&pos| (pos..pos + rows).map(|v| v as i64)).collect(),
            (Fill::Slot, _) => leases.iter().zip(positions).flat_map(|(l, &pos)| l.slots(pos..pos + rows)).collect(),
            (Fill::SeqLen, _) => positions.iter().map(|&pos| (pos + rows) as i64).collect(),
            (Fill::CuSeqlens, _) => (0..=b).map(|i| (i * rows) as i64).collect(),
            (Fill::SpanAt, _) => vec![0],
            (Fill::Blocks | Fill::Tokens | Fill::Count | Fill::Error, _) => continue,
        };
        rt.write_input_at(&f.name, &f.encode(&v), &vars)?;
    }
    Ok(vars)
}

/// `rows` rows of a page table: row `i` from lease `i`, the last lease
/// repeating past the ones given.
fn page_rows(t: &PageTable, leases: &[Lease], rows: usize) -> Result<Vec<i32>> {
    let mut v = Vec::with_capacity(rows * t.width);
    for r in 0..rows {
        leases[r.min(leases.len() - 1)].extend_row(&t.name, &mut v)?;
    }
    Ok(v)
}

/// A line table whole: cell `[line, col]` carries column `col`'s lease's
/// line in entry 0 and zeros through the rest of a wide cell (a program
/// that moves along one does so on the device). Columns past the leases
/// given repeat the last.
fn line_rows(t: &LineTable, leases: &[Lease], cols: usize) -> Result<Vec<i32>> {
    let mut v = Vec::with_capacity(t.lines * cols * t.width);
    for line in 0..t.lines {
        for c in 0..cols {
            v.push(leases[c.min(leases.len() - 1)].seq_line(&t.name, line)?);
            v.extend(std::iter::repeat_n(0, t.width - 1));
        }
    }
    Ok(v)
}

/// What the manifest runs once after load (the derived tables a weight
/// prep program computes), with every var at 1: a once program takes no
/// call shape.
fn run_once(rt: &Runtime, p: &Protocol) -> Result<()> {
    let vars: Vars = rt.manifest.vars.keys().map(|v| (v.clone(), 1)).collect();
    p.once.iter().try_for_each(|name| rt.run(name, &vars).with_context(|| format!("`{name}`")))
}

/// Ranks a manifest runs as: the size its topology groups share; 1
/// without a topology.
fn ranks_of(m: &Verified) -> Result<usize> {
    let sizes: Vec<u64> = m.topology.iter().flat_map(|t| t.groups.values().copied()).collect();
    match sizes.as_slice() {
        [] => Ok(1),
        [n, rest @ ..] if rest.iter().all(|r| r == n) => Ok(*n as usize),
        _ => bail!("the manifest's topology groups differ in size; a caller spans every group with all its ranks"),
    }
}

/// Rank `q`'s place in every group of the manifest's topology.
fn topology_of(m: &Verified, q: usize) -> Topology {
    Topology {
        groups: m
            .topology
            .iter()
            .flat_map(|t| &t.groups)
            .map(|(g, &size)| (g.clone(), GroupRank { index: q as u64, size }))
            .collect(),
    }
}

/// Every rank's peer buffers imported from every other, group by group,
/// until no rank has one unfilled. Nothing to do without a topology.
fn connect_peers(m: &Verified, rts: &mut [Runtime]) -> Result<()> {
    let Some(topo) = &m.topology else { return Ok(()) };
    let handles: Vec<BTreeMap<String, PeerHandle>> =
        rts.iter().map(Runtime::export_handles).collect::<Result<_, _>>()?;
    for g in topo.groups.keys() {
        for (q, rt) in rts.iter_mut().enumerate() {
            rt.import_peers(g, &handles).with_context(|| format!("rank {q}: peers of `{g}`"))?;
        }
    }
    for (q, rt) in rts.iter().enumerate() {
        let pending = rt.pending_peers();
        ensure!(pending.is_empty(), "rank {q}: peer buffers {pending:?} still unfilled after every group was imported");
    }
    Ok(())
}

/// A rank that has not returned in this long is hung: a collective
/// waiting for a peer that failed. Nothing in the process can go on.
const HUNG: Duration = Duration::from_secs(600);

/// A rank moved to a thread once, or lent to one for a call that is
/// joined before the borrow ends. The `Runtime` inside holds raw CUDA
/// handles; it is used from one thread at a time and binds its context on
/// every entry, so either is sound.
struct Sent<T>(T);
#[allow(unsafe_code)]
unsafe impl<T> Send for Sent<T> {}
struct Lent<'a, R>(&'a mut R);
#[allow(unsafe_code)]
unsafe impl<R> Send for Lent<'_, R> {}

/// `f` on every rank at once, so a collective inside it finds its peers
/// issuing; the results in rank order, the first error if any.
fn each<R, T: Send>(ranks: &mut [R], what: &str, f: impl Fn(&mut R) -> Result<T> + Sync) -> Result<Vec<T>> {
    let n = ranks.len();
    let (tx, rx) = std::sync::mpsc::channel();
    let mut out: Vec<Option<Result<T>>> = (0..n).map(|_| None).collect();
    std::thread::scope(|s| {
        for (q, r) in ranks.iter_mut().enumerate() {
            let (lent, tx, f) = (Lent(r), tx.clone(), &f);
            s.spawn(move || {
                let lent = lent;
                let _ = tx.send((q, f(lent.0)));
            });
        }
        drop(tx);
        for _ in 0..n {
            match rx.recv_timeout(HUNG) {
                Ok((q, r)) => out[q] = Some(r),
                Err(_) => {
                    let hung: Vec<String> =
                        out.iter().enumerate().filter(|(_, r)| r.is_none()).map(|(q, _)| q.to_string()).collect();
                    eprintln!(
                        "rank {} did not return within {}s at {what}: a collective waiting for a peer that failed",
                        hung.join(", "),
                        HUNG.as_secs()
                    );
                    std::process::exit(2);
                }
            }
        }
    });
    out.into_iter()
        .enumerate()
        .map(|(q, r)| r.expect("every rank replied").with_context(|| format!("rank {q}: {what}")))
        .collect()
}

/// A runtime plus the single sequence: its token slots and position cursor.
struct Caller {
    rt: Runtime,
    protocol: Protocol,
    /// The sequence's slots: as many as one page-table row (or the whole
    /// state) holds, leased once for the caller's life.
    lease: Lease,
    /// Tokens already in the state (next slot to fill).
    pos: i64,
}

impl Caller {
    fn into_runtime(self) -> Runtime {
        self.rt
    }

    /// Leases the sequence's slots, writes its row into every table, and
    /// runs what the manifest runs once. A table has a row per sequence
    /// the manifest allows; this caller is sequence 0, but every row must
    /// hold valid page ids, so its lease fills them all.
    fn new(mut rt: Runtime) -> Result<Caller> {
        let protocol = Protocol::check(&rt.manifest)?;
        let lease = rt.lease(rt.max_seq_tokens().min(rt.capacity() as usize))?;
        let one = std::slice::from_ref(&lease);
        for t in &protocol.page_tables {
            let rows = page_rows(t, one, protocol.groups.max as usize)?;
            rt.write_input(&t.name, &le_bytes_i32(&rows))?;
        }
        for t in &protocol.line_tables {
            let cols = match t.axis {
                Axis::Tray => protocol.tray.as_ref().map_or(1, |b| b.max),
                _ => protocol.groups.max,
            };
            rt.write_input(&t.name, &le_bytes_i32(&line_rows(t, one, cols as usize)?))?;
        }
        run_once(&rt, &protocol)?;
        Ok(Caller { rt, protocol, lease, pos: 0 })
    }

    /// Token slots the sequence can hold.
    fn limit(&self) -> usize {
        self.lease.tokens()
    }

    /// Stage one call's rows at the cursor: `ids` as consecutive positions
    /// of this one sequence. Does not advance. Returns the call's vars.
    fn stage(&mut self, ids: &[i64]) -> Result<Vars> {
        let pos = self.pos as usize;
        let one = std::slice::from_ref(&self.lease);
        crate::stage(&mut self.rt, &self.protocol, one, &[pos], ids.len(), ids)
    }

    /// Stage a fixed-rows call at the cursor: `tok` is the sequence's next
    /// token, in row 0 and again in every other row. A program that
    /// drafts its own rows (a speculative round) overwrites rows 1.. on
    /// the device; a one-row step reads only row 0.
    fn stage_rows(&mut self, tok: i64, rows: u64) -> Result<Vars> {
        self.stage(&vec![tok; rows as usize])
    }

    /// Reset the cursor (a new prompt reuses the slots from position 0).
    fn reset(&mut self) {
        self.pos = 0;
    }

    fn advance(&mut self, n: u64) {
        self.pos += n as i64;
    }

    /// The forward for one sequence of `rows` rows per call.
    fn forward(&self, rows: Rows) -> Result<Forward> {
        self.protocol.forward(1, rows).cloned().ok_or_else(|| {
            anyhow::anyhow!(
                "no program takes one sequence of {} rows; the manifest declares rows {:?}{}",
                match rows {
                    Rows::Const(r) => r.to_string(),
                    Rows::Var => "as many as fed".into(),
                },
                self.protocol.row_shapes(),
                if self.protocol.chunk().is_some() { " and a chunk" } else { "" }
            )
        })
    }

    /// The program a chunk of the prompt goes through.
    fn chunk_forward(&self) -> Result<Forward> {
        self.forward(Rows::Var)
    }

    /// How many of a prompt's tokens go through the chunk program: all of
    /// them when it hands a token back (a hybrid model's chunked prefill
    /// is a different arithmetic from its decode kernel, and the
    /// reference runs the last prompt token through the former), else all
    /// but the last, which goes through the step program.
    fn prefill_len(&self, prompt: usize) -> Result<usize> {
        Ok(prompt - usize::from(self.chunk_forward()?.emits.is_none()))
    }

    /// Chunked prefill of `ids` (eager or graph-captured full chunks),
    /// advancing the cursor past them. Returns the token the last chunk
    /// handed back, if the chunk program emits one, and whether a graph
    /// was captured.
    fn prefill(&mut self, ids: &[i64], chunk: u64) -> Result<Option<i64>> {
        let f = self.chunk_forward()?;
        let mut last = None;
        let mut i = 0usize;
        while i < ids.len() {
            let c = ((ids.len() - i) as u64).min(chunk) as usize;
            let e = self.stage(&ids[i..i + c])?;
            self.rt.issue(&f.name, &e)?;
            self.rt.synchronize()?;
            self.advance(c as u64);
            last = self.emitted(&f)?.first().copied();
            i += c;
        }
        Ok(last)
    }

    /// Vocabulary size as declared by the token fill's domain (1000 if none).
    fn vocab(&self) -> u64 {
        let m = &self.rt.manifest;
        m.buffers[&self.protocol.token_rows().name]
            .domain
            .as_ref()
            .and_then(|d| d.resolve(m, &self.protocol.vars(1, 1, 1), &self.rt.provision()).ok())
            .and_then(|r| r.hi)
            .map_or(1000, |hi| hi as u64 + 1)
    }

    /// What the last run of `f` handed back for this sequence: its `tokens`
    /// output's first cell, in order, cut to its `count` (one without a
    /// count).
    fn emitted(&self, f: &Forward) -> Result<Vec<i64>> {
        let Some(i) = f.emits else { return Ok(Vec::new()) };
        let t = &self.protocol.fills[i];
        let mut v = t.decode(&self.rt.read_output(&t.name)?);
        let n = match f.count {
            Some(c) => {
                let c = &self.protocol.fills[c];
                let n = c.decode(&self.rt.read_output(&c.name)?)[0];
                ensure!(n >= 1 && n <= t.width as i64, "`{}` says {n} of the {} rows are taken", c.name, t.width);
                n as usize
            }
            None => 1,
        };
        v.truncate(n);
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn eos_is_one_id_or_a_list() {
        assert_eq!(eos_of(&json!({"eos_token_id": 7})), vec![7]);
        assert_eq!(eos_of(&json!({"eos_token_id": [3, 5]})), vec![3, 5]);
        assert_eq!(eos_of(&json!({"eos_token_id": [3, "x"]})), vec![3]);
        assert_eq!(eos_of(&json!({"bos_token_id": 1})), Vec::<i64>::new());
    }

    #[test]
    fn a_checkpoint_dir_names_its_tokenizer_and_eos() {
        let root = std::env::temp_dir().join(format!("kern-checkpoint-{}", std::process::id()));
        let (a, b) = (root.join("a"), root.join("b"));
        std::fs::create_dir_all(&a).unwrap();
        std::fs::create_dir_all(&b).unwrap();
        std::fs::write(a.join("tokenizer.json"), "{}").unwrap();
        std::fs::write(a.join("config.json"), r#"{"eos_token_id": 1}"#).unwrap();
        std::fs::write(a.join("generation_config.json"), r#"{"eos_token_id": [2, 3]}"#).unwrap();
        std::fs::write(b.join("tokenizer.json"), "{}").unwrap();
        std::fs::write(b.join("config.json"), r#"{"eos_token_id": [3, 4]}"#).unwrap();
        let file = root.join("w.safetensors");
        std::fs::write(&file, "").unwrap();
        // generation_config beats config; first tokenizer wins; ids union in order; a file adds nothing
        let ck = checkpoint(&[file, a.clone(), b]);
        assert_eq!((ck.tokenizer, ck.stop_tokens), (Some(a.join("tokenizer.json")), vec![2, 3, 4]));
        assert_eq!(checkpoint(&[root.join("missing")]), Checkpoint::default());
        std::fs::remove_dir_all(&root).unwrap();
    }
}
