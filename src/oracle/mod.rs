//! Executable-oracle inspection and comparison for compiler migration.

mod inventory;
mod qwen38;
mod qwen38_fp8;

pub use inventory::{
    BatchInventory, BufferInventory, ManifestDiff, ManifestInventory, NamedDiff, ProgramDiff, ProgramInventory,
    RowsInventory, StateInventory,
};
pub use qwen38::{OracleError, Qwen38Oracle, Qwen38OracleReport, Qwen38StatePacking};
pub use qwen38_fp8::{CheckpointError, Qwen38Fp8Checkpoint, Qwen38Fp8Report, TensorContract};
