use serde::{Deserialize, Serialize};

pub type RegionId = [u8; 16];

/// Framework-owned local storage that OrbitKV may use as a transfer endpoint.
///
/// A generation is mandatory: reusing an index or byte range must never make
/// an old asynchronous completion refer to a new page occupant.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum LocalPageRef {
    CudaIpc {
        pool_id: u64,
        page_index: u32,
        generation: u64,
    },
    SharedHost {
        region_id: RegionId,
        offset: u64,
        len: u64,
        generation: u64,
    },
}

impl LocalPageRef {
    pub fn generation(&self) -> u64 {
        match self {
            Self::CudaIpc { generation, .. } | Self::SharedHost { generation, .. } => *generation,
        }
    }
}
