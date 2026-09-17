//! Fixtures shared by the runtime's host-side tests: pools over tiny
//! manifests whose states are one byte per token or a few bytes per
//! sequence, so every page and slot is a handful of chunks.

#![allow(dead_code)]

use std::sync::Arc;

use kern_manifest::types::Manifest;
use kern_pool::Pool;

fn manifest(json: &str) -> Manifest {
    Manifest::from_json(json).expect("fixture parses")
}

/// Two paged states: kv in 16-token pages (a row of 3), a draft state in
/// 4-token pages (a row of 16). Page unit 16, a page is 16 bytes in each
/// of two arenas.
pub fn two_paged() -> Manifest {
    manifest(
        r#"{
        "schema_version": 5, "model": "t", "vars": {"tokens": {"max": 8}, "seqs": {"max": 2}},
        "states": {"kv": {"bytes_per_token": 1}, "draft_kv": {"bytes_per_token": 1}},
        "buffers": {
            "slot_mapping": {"kind": "input", "dtype": "i64", "shape": ["tokens"], "domain": {"index_into": "kv"}},
            "block_table": {"kind": "input", "dtype": "i32", "shape": ["seqs", 3], "domain": {"index_into": "kv", "stride": 16}},
            "draft_block_table": {"kind": "input", "dtype": "i32", "shape": [16], "domain": {"index_into": "draft_kv", "stride": 4}}
        },
        "modules": {}, "ops": {}, "programs": {}
    }"#,
    )
}

/// One paged state of 16 bytes a page, plus a recurrent state of 3 lines
/// of 8 bytes per sequence and its line table.
pub fn hybrid() -> Manifest {
    manifest(
        r#"{
        "schema_version": 5, "model": "t", "vars": {"tokens": {"max": 8}, "seqs": {"max": 2}},
        "states": {"kv": {"bytes_per_token": 1}, "gdn": {"bytes_per_seq": 24}},
        "buffers": {
            "block_table": {"kind": "input", "dtype": "i32", "shape": ["seqs", 3], "domain": {"index_into": "kv", "stride": 16}},
            "line_index": {"kind": "input", "dtype": "i32", "shape": [3, "seqs"], "domain": {"index_into": "gdn", "stride": 8}}
        },
        "modules": {}, "ops": {}, "programs": {}
    }"#,
    )
}

/// kv paged in 4 tokens, a row of 8 pages.
pub fn paged4() -> Manifest {
    manifest(
        r#"{
        "schema_version": 5, "model": "t", "vars": {"tokens": {"max": 8}, "seqs": {"max": 2}},
        "states": {"kv": {"bytes_per_token": 1}},
        "buffers": {
            "block_table": {"kind": "input", "dtype": "i32", "shape": ["seqs", 8], "domain": {"index_into": "kv", "stride": 4}}
        },
        "modules": {}, "ops": {}, "programs": {}
    }"#,
    )
}

/// kv paged in 4 tokens plus a recurrent state of one 8-byte line.
pub fn hybrid4() -> Manifest {
    manifest(
        r#"{
        "schema_version": 5, "model": "t", "vars": {"tokens": {"max": 8}, "seqs": {"max": 1}},
        "states": {"kv": {"bytes_per_token": 1}, "rec": {"bytes_per_seq": 8}},
        "buffers": {
            "block_table": {"kind": "input", "dtype": "i32", "shape": ["seqs", 8], "domain": {"index_into": "kv", "stride": 4}},
            "line_index": {"kind": "input", "dtype": "i32", "shape": [1, "seqs"], "domain": {"index_into": "rec", "stride": 8}}
        },
        "modules": {}, "ops": {}, "programs": {}
    }"#,
    )
}

/// A pool over `m` with `chunks` chunks of `chunk` bytes, the initial
/// remap taken as landed.
pub fn pool_of(m: &Manifest, chunk: u64, chunks: u32, first_slots: usize) -> Arc<Pool> {
    Arc::new(Pool::new(m, chunk, chunks, first_slots, None).expect("fixture lays out").0)
}

/// `two_paged` over 8-byte chunks: a page is 2 chunks per arena, 16 chunks
/// hold 4 pages.
pub fn pool() -> Arc<Pool> {
    pool_of(&two_paged(), 8, 16, 0)
}

/// `hybrid` over 8-byte chunks, 20 of them: 4 slots (slot 0 among them)
/// and 4 pages, no chunk spare.
pub fn hybrid_pool() -> Arc<Pool> {
    pool_of(&hybrid(), 8, 20, 4)
}

/// `paged4` over 4-byte chunks: 8 pages of one chunk.
pub fn pool4() -> Arc<Pool> {
    pool_of(&paged4(), 4, 8, 0)
}

/// `hybrid4` over 4-byte chunks, 18 of them: 3 slots and 6 pages.
pub fn hybrid_pool4() -> Arc<Pool> {
    pool_of(&hybrid4(), 4, 18, 3)
}

/// A deterministic xorshift source, `next(n)` in `0..n`.
pub struct Rand(u64);

impl Rand {
    pub fn new(seed: u64) -> Rand {
        Rand(seed)
    }

    pub fn next(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 33) as usize % n
    }
}

/// Land the remap the pool has planned.
pub fn land(p: &Pool) {
    p.complete(p.take_pending().expect("a remap planned"));
}
