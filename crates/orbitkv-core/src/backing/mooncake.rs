use std::path::Path;
use std::ptr::NonNull;
use std::sync::Arc;
use std::time::Instant;

use log::{error, info};
use orbitkv_transfer::{AUTO_MEMORY_LOCATION, P2P_METADATA, TransferEngine};

use crate::memory::pool::PinnedAllocator;

/// Mooncake Transfer Engine plus the lifetime of the registered pinned pool.
pub(crate) struct MooncakeTransport {
    engine: TransferEngine,
    transfer_endpoint: String,
    /// Base pointers of registered regions, kept for unregister on drop.
    registered_ptrs: Vec<NonNull<u8>>,
}

// SAFETY: The registered pointers point to CUDA-pinned memory that is
// fixed in physical memory and safe to access from any thread. The Vec
// is only read during Drop, which is exclusive.
unsafe impl Send for MooncakeTransport {}
unsafe impl Sync for MooncakeTransport {}

impl MooncakeTransport {
    pub(crate) fn engine(&self) -> &TransferEngine {
        &self.engine
    }

    pub(crate) fn transfer_endpoint(&self) -> &str {
        &self.transfer_endpoint
    }

    fn new(
        nic_names: &[String],
        allocator: &PinnedAllocator,
        advertise_addr: &str,
    ) -> Result<Self, String> {
        let t0 = Instant::now();
        for nic in nic_names {
            let path = Path::new("/sys/class/infiniband").join(nic);
            if !path.exists() {
                return Err(format!("configured RDMA NIC {nic:?} does not exist"));
            }
        }
        let bind_host = host_from_endpoint(advertise_addr)?;
        let local_server_name = format!("{bind_host}:0");
        let engine =
            TransferEngine::new(P2P_METADATA, &local_server_name, &bind_host, 0, nic_names)
                .map_err(|e| e.to_string())?;
        let transfer_endpoint = engine.local_segment_name().map_err(|e| e.to_string())?;

        let regions: Vec<(NonNull<u8>, usize)> = allocator.memory_regions();
        for &(ptr, len) in &regions {
            unsafe {
                engine
                    .register_memory(ptr, len, AUTO_MEMORY_LOCATION)
                    .map_err(|e| e.to_string())?;
            }
        }

        let registered_ptrs: Vec<NonNull<u8>> = regions.iter().map(|&(ptr, _)| ptr).collect();

        info!(
            "Mooncake Transfer Engine initialised: endpoint={}, nics={}, registered {} memory region(s), elapsed={:?}",
            transfer_endpoint,
            nic_names.len(),
            registered_ptrs.len(),
            t0.elapsed(),
        );

        Ok(Self {
            engine,
            transfer_endpoint,
            registered_ptrs,
        })
    }
}

impl Drop for MooncakeTransport {
    fn drop(&mut self) {
        for &ptr in &self.registered_ptrs {
            if let Err(e) = unsafe { self.engine.unregister_memory(ptr) } {
                error!("Failed to unregister Mooncake memory region: {e}");
            }
        }
    }
}

fn host_from_endpoint(endpoint: &str) -> Result<String, String> {
    let endpoint = endpoint
        .strip_prefix("http://")
        .or_else(|| endpoint.strip_prefix("https://"))
        .unwrap_or(endpoint);
    endpoint
        .rsplit_once(':')
        .map(|(host, _)| host.trim_matches(['[', ']']).to_string())
        .filter(|host| !host.is_empty())
        .ok_or_else(|| format!("invalid advertise address: {endpoint}"))
}

/// Create a [`MooncakeTransport`].
pub(crate) fn new_mooncake(
    nic_names: &[String],
    allocator: &PinnedAllocator,
    advertise_addr: &str,
) -> Result<Arc<MooncakeTransport>, String> {
    MooncakeTransport::new(nic_names, allocator, advertise_addr)
        .map(Arc::new)
        .map_err(|e| {
            format!("Failed to initialise Mooncake Transfer Engine (nics={nic_names:?}): {e}")
        })
}
