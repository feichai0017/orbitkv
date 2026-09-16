//! OrbitKV Next: model semantics and execution-island compilation above `kern`.
//!
//! This crate deliberately contains no CUDA runtime, allocator, weight loader,
//! or serving loop. Those mechanisms belong to the pinned `kern` substrate.

pub mod compiler;
pub mod ir;
pub mod lower;
pub mod model;
