use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use log::{error, info};
use orbitkv_transfer::{AUTO_MEMORY_LOCATION, MemoryRegistration, P2P_METADATA, TransferEngine};

use crate::memory::pool::PinnedAllocator;

/// Mooncake TENT plus the lifetime of the registered pinned pool.
pub(crate) struct MooncakeTransport {
    engine: Arc<TransferEngine>,
    registrations: Vec<MemoryRegistration>,
    _pool: Arc<PinnedAllocator>,
    transfer_endpoint: String,
}

impl MooncakeTransport {
    pub(crate) fn engine(&self) -> &TransferEngine {
        &self.engine
    }

    pub(crate) fn transfer_endpoint(&self) -> &str {
        &self.transfer_endpoint
    }

    pub(crate) fn new(
        nic_names: &[String],
        allocator: Arc<PinnedAllocator>,
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
        let engine = Arc::new(
            TransferEngine::new(P2P_METADATA, &local_server_name, &bind_host, 0, nic_names)
                .map_err(|e| e.to_string())?,
        );
        let transfer_endpoint = engine.local_segment_name().map_err(|e| e.to_string())?;

        let regions = allocator.memory_regions();
        let registrations = regions
            .iter()
            .map(|&(ptr, len)| unsafe {
                engine
                    .register_memory_owned(ptr, len, AUTO_MEMORY_LOCATION)
                    .map_err(|e| e.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;

        info!(
            "Mooncake TENT initialised: endpoint={}, nics={}, registered {} memory region(s), elapsed={:?}",
            transfer_endpoint,
            nic_names.len(),
            registrations.len(),
            t0.elapsed(),
        );

        Ok(Self {
            engine,
            registrations,
            _pool: allocator,
            transfer_endpoint,
        })
    }
}

impl Drop for MooncakeTransport {
    fn drop(&mut self) {
        // Unregister before the pinned-pool owner can release its backing.
        for registration in self.registrations.drain(..) {
            if let Err(error) = registration.unregister() {
                // The token retries once more from Drop while its engine Arc
                // and the pool backing are still alive.
                error!("Failed to unregister Mooncake memory region: {error}");
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

#[cfg(test)]
#[path = "../../tests/unit/peer/transport.rs"]
mod tests;
