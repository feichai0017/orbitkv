use std::sync::Arc;

use super::replica::ReplicaSet;
use crate::block::StateKey;
use crate::peer::catalog::CatalogClient;
use crate::storage::{dram::DramStore, ssd::SsdStore};

// One catalog budget across every bounded batch in a candidate discovery.
pub(crate) const DISCOVERY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(3);

pub(crate) async fn discover(
    dram: &DramStore,
    ssd: Option<&Arc<SsdStore>>,
    catalog: Option<&Arc<CatalogClient>>,
    namespace: &str,
    hashes: &[Vec<u8>],
    deadline: tokio::time::Instant,
) -> Vec<ReplicaSet> {
    #[cfg(not(feature = "mooncake"))]
    let _ = (deadline, catalog);
    let keys: Vec<_> = hashes
        .iter()
        .map(|hash| StateKey::new(namespace.to_owned(), hash.clone()))
        .collect();
    let mut candidates: Vec<_> = keys
        .iter()
        .cloned()
        .zip(dram.discover(&keys))
        .map(|(key, dram)| {
            let mut candidates = ReplicaSet::new(key);
            if let Some(dram) = dram {
                candidates.set_memory(dram);
            }
            candidates
        })
        .collect();
    if let Some(ssd) = ssd {
        for (candidate, backing) in candidates.iter_mut().zip(ssd.discover(&keys)) {
            if let Some(backing) = backing {
                candidate.set_ssd(backing);
            }
        }
    }
    #[cfg(feature = "mooncake")]
    if let Some(catalog) = catalog {
        for (candidate, cached) in candidates.iter_mut().zip(catalog.cached_blocks(&keys)) {
            if let Some(cached) = cached {
                candidate.set_peer_dram(cached.replicas);
            }
        }
        let missing: Vec<_> = candidates
            .iter()
            .enumerate()
            .filter_map(|(i, candidate)| (!candidate.is_available()).then_some(i))
            .collect();
        let hashes: Vec<_> = missing.iter().map(|&i| hashes[i].clone()).collect();
        if !hashes.is_empty() && tokio::time::Instant::now() < deadline {
            let remote =
                match tokio::time::timeout_at(deadline, catalog.locate_blocks(namespace, &hashes))
                    .await
                {
                    Ok(Ok(remote)) => remote.into_iter().map(Some).collect(),
                    Ok(Err(error)) => {
                        log::warn!("candidate discovery failed: {error}");
                        Vec::new()
                    }
                    Err(_) => {
                        // Healthy peers may have published evidence before a
                        // different peer exhausted this discovery's deadline.
                        let keys: Vec<_> = missing.iter().map(|&i| keys[i].clone()).collect();
                        catalog.cached_blocks(&keys)
                    }
                };
            for (i, candidate) in missing.into_iter().zip(remote) {
                if let Some(candidate) = candidate
                    && candidates[i].key == candidate.key
                {
                    candidates[i].set_peer_dram(candidate.replicas);
                }
            }
        }
    }
    candidates
}
