//! The model and compiler layer of the source-integrated OrbitKV Next engine.
//!
//! CUDA execution, state allocation, and serving remain behind the local
//! `kern-manifest` boundary in the sibling `kern-*` crates.

pub mod compiler;
pub mod ir;
pub mod lower;
pub mod model;
pub mod oracle;
