//! Bounded, host-observed costs. Observations never own execution resources.
//!
//! A path is one measurement boundary, not an additive edge: codec, prefetch
//! and SSD restore paths include child operations. No device timing is inferred.

use std::hash::{Hash, Hasher};
use std::sync::LazyLock;
use std::time::Duration;

const CAPACITY: usize = 512;
const MIN_SAMPLES: u64 = 4;
const MAX_AGE: Duration = Duration::from_secs(300);
const ALPHA: f64 = 0.2;

static ENABLED: LazyLock<bool> =
    LazyLock::new(|| std::env::var("ORBITKV_COST_OBSERVATIONS").as_deref() == Ok("1"));

pub(crate) fn enabled() -> bool {
    *ENABLED
}

mod estimates;
mod observation;
mod shadow;

pub(crate) use observation::{Observation, Outcome};
pub(crate) use shadow::shadow;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum CostPath {
    GpuLoadDirect,
    GpuLoadKernel,
    GpuSaveDirect,
    GpuSaveKernel,
    GpuDecode,
    GpuEncode,
    GpuSsdLoad,
    SsdUringRestore,
    SsdCufileRestore,
    GpuSsdSave,
    SsdRead,
    SsdWrite,
    SsdWriteBatch,
    SsdCufileRead,
    SsdCufileWrite,
    SsdPrefetch,
    #[cfg(feature = "mooncake")]
    RemoteRead,
    #[cfg(feature = "mooncake")]
    RemoteAuthorization,
}

impl CostPath {
    fn is_raw_copy(self) -> bool {
        matches!(
            self,
            Self::GpuLoadDirect | Self::GpuLoadKernel | Self::GpuSaveDirect | Self::GpuSaveKernel
        )
    }

    fn is_restore_route(self) -> bool {
        matches!(self, Self::SsdUringRestore | Self::SsdCufileRestore)
    }

    fn label(self) -> &'static str {
        match self {
            Self::GpuLoadDirect => "gpu_load_direct",
            Self::GpuLoadKernel => "gpu_load_kernel",
            Self::GpuSaveDirect => "gpu_save_direct",
            Self::GpuSaveKernel => "gpu_save_kernel",
            Self::GpuDecode => "gpu_decode",
            Self::GpuEncode => "gpu_encode",
            Self::GpuSsdLoad => "gpu_ssd_load",
            Self::SsdUringRestore => "ssd_uring_restore",
            Self::SsdCufileRestore => "ssd_cufile_restore",
            Self::GpuSsdSave => "gpu_ssd_save",
            Self::SsdRead => "ssd_read",
            Self::SsdWrite => "ssd_write",
            Self::SsdWriteBatch => "ssd_write_batch",
            Self::SsdCufileRead => "ssd_cufile_read",
            Self::SsdCufileWrite => "ssd_cufile_write",
            Self::SsdPrefetch => "ssd_prefetch",
            #[cfg(feature = "mooncake")]
            Self::RemoteRead => "remote_read",
            #[cfg(feature = "mooncake")]
            Self::RemoteAuthorization => "remote_authorization",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum Representation {
    Raw,
    Ans,
    Fp8,
    TurboQuant,
    Mixed,
    Unknown,
}

impl From<orbitkv_state::StorageFormat> for Representation {
    fn from(format: orbitkv_state::StorageFormat) -> Self {
        use orbitkv_state::StorageFormat;
        match format {
            StorageFormat::Ans | StorageFormat::Ans16 | StorageFormat::AnsFp8 => Self::Ans,
            StorageFormat::Fp8FromBf16 | StorageFormat::Fp8FromFp16 => Self::Fp8,
            StorageFormat::TurboQuant { .. } => Self::TurboQuant,
            _ => Self::Raw,
        }
    }
}

/// Neither request IDs nor state keys belong here. Resources identify a GPU,
/// disk owner or peer incarnation; they are never exported as metric labels.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct CostKey {
    path: CostPath,
    resource: u64,
    representation: Representation,
    size: u8,
    fragments: u8,
    dma_ranges: u8,
    source_size: u8,
    source_fragments: u8,
    ssd_size: u8,
    ssd_fragments: u8,
}

impl CostKey {
    pub(crate) fn with_dma_ranges(self, ranges: usize) -> Self {
        Self {
            dma_ranges: bucket(ranges as u64),
            ..self
        }
    }

    pub(crate) fn with_path(self, path: CostPath) -> Self {
        Self { path, ..self }
    }
    pub(crate) fn with_ssd_shape(
        self,
        source_bytes: u64,
        source_fragments: usize,
        target_bytes: u64,
        target_fragments: usize,
    ) -> Self {
        Self {
            source_size: bucket(source_bytes),
            source_fragments: bucket(source_fragments as u64),
            ssd_size: bucket(target_bytes),
            ssd_fragments: bucket(target_fragments as u64),
            ..self
        }
    }
    pub(crate) fn with_path_resource(self, path: CostPath, resource: u64) -> Self {
        Self {
            path,
            resource,
            ..self
        }
    }
    pub(crate) fn new(
        path: CostPath,
        resource: u64,
        representation: Representation,
        bytes: u64,
        fragments: usize,
    ) -> Self {
        Self {
            path,
            resource,
            representation,
            size: bucket(bytes),
            fragments: bucket(fragments as u64),
            dma_ranges: 0,
            source_size: 0,
            source_fragments: 0,
            ssd_size: 0,
            ssd_fragments: 0,
        }
    }
}

fn bucket(value: u64) -> u8 {
    (u64::BITS - value.leading_zeros()) as u8
}

pub(crate) fn resource_id(value: &impl Hash) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}
