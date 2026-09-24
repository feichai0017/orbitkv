use std::num::NonZeroUsize;
use std::path::PathBuf;

pub(crate) const SSD_ALIGNMENT: usize = 512;

/// Default write queue depth for SSD writer thread (blocks dropped if full)
pub const DEFAULT_SSD_WRITE_QUEUE_DEPTH: usize = 8;

/// Default prefetch queue depth (limits read tail latency)
pub const DEFAULT_SSD_PREFETCH_QUEUE_DEPTH: usize = 2;

/// Default max concurrent writes (not critical path, keep low)
pub const DEFAULT_SSD_WRITE_INFLIGHT: usize = 2;

/// Default max concurrent prefetches
pub const DEFAULT_SSD_PREFETCH_INFLIGHT: usize = 16;

// ============================================================================
// Configuration
// ============================================================================

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum SsdWritePolicy {
    #[default]
    All,
    /// Admit demand hits or a repeated publication within bounded history.
    Reuse,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum SsdBackend {
    #[default]
    /// Try native cuFile on supported mounts; use io_uring when unavailable.
    Auto,
    Uring,
    /// Read/write through bounded GPU staging; fragmented saves seal in DRAM.
    /// Native GDS availability depends on the deployment's cuFile configuration.
    Cufile,
}

impl std::str::FromStr for SsdBackend {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "auto" => Ok(Self::Auto),
            "uring" => Ok(Self::Uring),
            "cufile" => Ok(Self::Cufile),
            _ => Err("SSD backend must be auto, uring or cufile".into()),
        }
    }
}

impl std::str::FromStr for SsdWritePolicy {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "all" => Ok(Self::All),
            "reuse" => Ok(Self::Reuse),
            _ => Err("SSD write policy must be all or reuse".into()),
        }
    }
}

/// Lossless SSD representation; engine KV quantization is independent.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum SsdCompression {
    #[default]
    None,
    Lz4,
}

impl std::str::FromStr for SsdCompression {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "none" => Ok(Self::None),
            "lz4" => Ok(Self::Lz4),
            _ => Err("SSD compression must be none or lz4".into()),
        }
    }
}

/// Configuration for the SSD cache (logical ring).
///
/// Supports one or more cache directories. When multiple paths are provided,
/// cache shards are distributed across them in round-robin order so that I/O
/// is balanced across independent devices.
#[derive(Debug, Clone)]
pub struct SsdCacheConfig {
    /// Cache data directories. Each path receives a subset of the total shards.
    pub cache_paths: Vec<PathBuf>,
    /// Total logical capacity of the cache (bytes).
    pub capacity_bytes: u64,
    /// Number of cache files per path. 1 keeps the existing single-file SSD layout
    /// when only one path is configured. With multiple paths each path receives this
    /// many shards so that every device is utilised.
    pub shards: NonZeroUsize,
    /// Max pending write batches. New sealed blocks are dropped if the queue is full.
    pub write_queue_depth: usize,
    pub write_policy: SsdWritePolicy,
    /// Max pending prefetch batches (limits read tail latency).
    pub prefetch_queue_depth: usize,
    /// Max concurrent block writes (not critical path, keep low).
    pub write_inflight: usize,
    /// Max concurrent block prefetches.
    pub prefetch_inflight: usize,
    pub backend: SsdBackend,
    pub compression: SsdCompression,
    /// Temporary encoded buffers, independent of the pinned cache pool.
    pub codec_budget: usize,
}

impl Default for SsdCacheConfig {
    fn default() -> Self {
        Self {
            cache_paths: vec![PathBuf::from("/tmp/orbitkv-ssd-cache/cache.bin")],
            capacity_bytes: 512 * 1024 * 1024 * 1024, // 512GB
            shards: NonZeroUsize::new(1).unwrap(),
            write_queue_depth: DEFAULT_SSD_WRITE_QUEUE_DEPTH,
            write_policy: SsdWritePolicy::All,
            prefetch_queue_depth: DEFAULT_SSD_PREFETCH_QUEUE_DEPTH,
            write_inflight: DEFAULT_SSD_WRITE_INFLIGHT,
            prefetch_inflight: DEFAULT_SSD_PREFETCH_INFLIGHT,
            backend: SsdBackend::Auto,
            compression: SsdCompression::None,
            codec_budget: 64 * 1024 * 1024,
        }
    }
}
