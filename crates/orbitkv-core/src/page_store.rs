//! Opaque host pages for framework storage adapters.
//!
//! Pages enter the same pinned read cache and SSD write path as GPU offloads.
//! Their namespace is supplied by the adapter and must include model, rank,
//! pool, and byte-layout identity.

use std::num::NonZeroU64;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::backing::SSD_ALIGNMENT;
use crate::block::{PrefetchStatus, RawBlock, SealedBlock, Segment};
use crate::offload::{RawSaveBatch, RawSaveLayer};
use crate::{EngineError, OrbitKVEngine};
use orbitkv_common::NumaNode;

const PAGE_HEADER_BYTES: usize = 8;
const PAGE_FETCH_TIMEOUT: Duration = Duration::from_secs(35);

impl OrbitKVEngine {
    /// Copy one framework-owned host page into the Cache Manager's tiered store.
    pub async fn put_host_page(
        &self,
        namespace: &str,
        key: &[u8],
        data: &[u8],
    ) -> Result<(), EngineError> {
        if namespace.is_empty() || key.is_empty() || data.is_empty() {
            return Err(EngineError::InvalidArgument(
                "host page requires a namespace, key, and non-empty data".into(),
            ));
        }
        let size = data
            .len()
            .checked_add(PAGE_HEADER_BYTES)
            .and_then(|size| size.checked_add(SSD_ALIGNMENT - 1))
            .map(|size| size / SSD_ALIGNMENT * SSD_ALIGNMENT)
            .ok_or_else(|| EngineError::InvalidArgument("host page size overflow".into()))?;
        let size = NonZeroU64::new(size as u64)
            .ok_or_else(|| EngineError::InvalidArgument("empty host page".into()))?;
        let numa_node = self
            .topology
            .gpu_numa_nodes()
            .first()
            .copied()
            .or_else(|| self.topology.numa_nodes().first().copied())
            .unwrap_or(NumaNode(0));
        let alloc = self
            .storage
            .allocate(size, Some(numa_node))
            .ok_or_else(|| EngineError::Storage("host page pool exhausted".into()))?;
        let ptr = alloc.as_non_null();
        // SAFETY: `alloc` owns `size` writable bytes and remains alive in Segment.
        unsafe {
            std::ptr::write_bytes(ptr.as_ptr(), 0, size.get() as usize);
            std::ptr::copy_nonoverlapping(
                (data.len() as u64).to_le_bytes().as_ptr(),
                ptr.as_ptr(),
                PAGE_HEADER_BYTES,
            );
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                ptr.as_ptr().add(PAGE_HEADER_BYTES),
                data.len(),
            );
        }
        self.storage.send_raw_insert(RawSaveBatch {
            namespace: namespace.to_string(),
            total_slots: 1,
            numa_node,
            layers: vec![RawSaveLayer {
                slot_id: 0,
                padded_block_size: size.get() as usize,
                blocks: vec![RawBlock::single_segment(Segment::new(
                    ptr,
                    size.get() as usize,
                    alloc,
                ))],
                block_hashes: vec![key.to_vec()],
            }],
        });
        self.storage.flush_write_pipeline().await;
        if !self.has_host_page(namespace, key).await? {
            return Err(EngineError::Storage(
                "host page was rejected by cache admission".into(),
            ));
        }
        Ok(())
    }

    /// Fetch a page from DRAM, SSD, or a peer, if available.
    pub async fn get_host_page(
        &self,
        namespace: &str,
        key: &[u8],
    ) -> Result<Option<Vec<u8>>, EngineError> {
        let Some(block) = self.load_host_block(namespace, key).await? else {
            return Ok(None);
        };
        Ok(Some(host_page_bytes(&block)?.to_vec()))
    }

    /// Check page availability without copying the page into the control reply.
    pub async fn has_host_page(&self, namespace: &str, key: &[u8]) -> Result<bool, EngineError> {
        let Some(block) = self.load_host_block(namespace, key).await? else {
            return Ok(false);
        };
        host_page_bytes(&block)?;
        Ok(true)
    }

    async fn load_host_block(
        &self,
        namespace: &str,
        key: &[u8],
    ) -> Result<Option<Arc<SealedBlock>>, EngineError> {
        if namespace.is_empty() || key.is_empty() {
            return Err(EngineError::InvalidArgument(
                "host page requires a namespace and key".into(),
            ));
        }
        let req_id = format!("host-page-{}", uuid::Uuid::new_v4());
        let hashes = vec![key.to_vec()];
        let deadline = Instant::now() + PAGE_FETCH_TIMEOUT;
        loop {
            match self
                .storage
                .check_prefix_and_prefetch(&req_id, namespace, &hashes, true)
                .await
            {
                PrefetchStatus::Ready { mut blocks, .. } => return Ok(blocks.pop()),
                PrefetchStatus::Loading if Instant::now() < deadline => {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
                PrefetchStatus::Loading => {
                    return Err(EngineError::Storage("host page fetch timed out".into()));
                }
            }
        }
    }
}

fn host_page_bytes(block: &SealedBlock) -> Result<&[u8], EngineError> {
    let raw = block
        .get_slot(0)
        .ok_or_else(|| EngineError::Storage("host page has no slot".into()))?;
    if raw.num_segments() != 1 {
        return Err(EngineError::Storage(
            "host page has incompatible segment layout".into(),
        ));
    }
    let ptr = raw
        .segment_ptr(0)
        .ok_or_else(|| EngineError::Storage("host page has no data".into()))?;
    let len = raw.segment_size(0).unwrap_or(0);
    if len < PAGE_HEADER_BYTES {
        return Err(EngineError::Storage("host page header is truncated".into()));
    }
    // SAFETY: `block` retains the pinned allocation for the returned slice.
    let bytes = unsafe { std::slice::from_raw_parts(ptr.as_ptr(), len) };
    let actual = u64::from_le_bytes(bytes[..PAGE_HEADER_BYTES].try_into().unwrap()) as usize;
    if actual == 0 || actual > len - PAGE_HEADER_BYTES {
        return Err(EngineError::Storage("host page length is invalid".into()));
    }
    Ok(&bytes[PAGE_HEADER_BYTES..PAGE_HEADER_BYTES + actual])
}
