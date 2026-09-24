use serde::{Deserialize, Serialize};

use crate::Digest;

/// Allowed storage transform and original scalar representation. Checkpoints
/// and opaque layouts remain exact.
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum StorageFormat {
    #[default]
    Exact,
    Fp8FromBf16,
    Fp8FromFp16,
}

/// Physical scalar representation of a state component.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateDType {
    Fp8E4M3,
    Fp8E5M2,
    Bf16,
    Fp16,
    Fp32,
    Int8,
    Opaque(String),
}

/// Byte layout needed to decide whether two engine pages are interchangeable.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateLayout {
    MhaLayerFirst,
    MhaPageFirst,
    MlaCompressed,
    Recurrent,
    Opaque(String),
}

/// All physical facts that affect byte compatibility across engines.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct StateFormat {
    pub model: Digest,
    pub implementation: Digest,
    pub dtype: StateDType,
    pub layout: StateLayout,
    pub block_tokens: u32,
    pub tensor_parallel_size: u32,
    pub pipeline_parallel_size: u32,
    pub context_parallel_size: u32,
}
