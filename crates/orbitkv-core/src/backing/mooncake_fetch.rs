// Candidate discovery -> requester planning -> source authorization -> Mooncake READ.

use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::Arc;
use std::time::{Duration, Instant};

use log::{info, warn};
use orbitkv_proto::proto::engine::TransferBlockInfo;
use orbitkv_transfer::{TransferOp, TransferSlice};

use crate::memory::numa::NumaNode;

use opentelemetry::KeyValue;

use super::fetch_plan::{
    FetchPlan, FetchSegment, SegmentFetcher, SegmentOutcome, execute_fetch_plan,
};
use super::transfer_lock_guard::{TransferCompletions, TransferLockGuard};
use super::{AllocateFn, MooncakeTransport, PrefetchResult};
use crate::block::{RawBlock, SealedBlock, Segment, StateKey};
use crate::internode::CatalogClient;
use crate::metrics::core_metrics;

/// Minimum usable transfer timeout. If the server's lock timeout minus the
/// safety margin falls below this, we use this floor to avoid instant timeouts.
const MIN_TRANSFER_TIMEOUT: Duration = Duration::from_secs(10);

/// Stop submitting work before the source marks the session overdue.
const LOCK_TIMEOUT_MARGIN: Duration = Duration::from_secs(60);

/// Upper bound for a single pinned-pool allocation while staging a Mooncake fetch.
/// LRU reclaim must carve a contiguous hole of the requested size, so a
/// whole-prefix slab can force eviction of far more bytes than the fetch needs.
const FETCH_CHUNK_BYTES: u64 = 256 * 1024 * 1024;

/// Mooncake remote block fetch backing store.
///
/// When all requested blocks are missing locally, queries Catalog for their
/// location, picks the best remote node, and uses gRPC authorization plus a
/// Mooncake READ to fetch them.
pub(crate) struct MooncakeFetchStore {
    catalog_client: Arc<CatalogClient>,
    completions: Arc<TransferCompletions>,
    membership: Arc<orbitkv_catalog::MembershipView>,
    transfer: Arc<MooncakeTransport>,
    allocate_fn: AllocateFn,
}

#[tonic::async_trait]
impl SegmentFetcher for MooncakeFetchStore {
    async fn fetch_segment(&self, segment: &FetchSegment, req_id: &str) -> SegmentOutcome {
        if !self.membership.permits(&segment.owner) {
            return SegmentOutcome::Rejected;
        }
        let remote_addr = &segment.owner.endpoint;
        let namespace = &segment.records[0].key.namespace;
        let block_hashes: Vec<_> = segment.records.iter().map(|r| r.key.hash.clone()).collect();
        let t0 = Instant::now();

        // Query the OrbitKV authority before exposing any physical addresses.
        let query_start = Instant::now();
        let authorization = self
            .completions
            .authorize(segment, self.membership.owner().incarnation)
            .await;
        let query_elapsed = query_start.elapsed();
        core_metrics().remote_stage_duration_seconds.record(
            query_elapsed.as_secs_f64(),
            &[
                KeyValue::new("stage", "authorization"),
                KeyValue::new("status", if authorization.is_ok() { "ok" } else { "error" }),
            ],
        );
        let (lock_guard, response) = match authorization {
            Ok(cr) => cr,
            Err(error) if error.code() == tonic::Code::FailedPrecondition => {
                core_metrics()
                    .remote_fetch_total
                    .add(1, &[KeyValue::new("status", "rejected")]);
                for record in &segment.records {
                    self.catalog_client.reject_candidate(
                        &record.key,
                        &orbitkv_state::ReplicaLocation {
                            owner: segment.owner.clone(),
                            sequence: record.sequence,
                        },
                    );
                }
                return SegmentOutcome::Rejected;
            }
            Err(e) => {
                warn!("Remote query to {remote_addr} failed: {e}");
                core_metrics()
                    .remote_fetch_total
                    .add(1, &[KeyValue::new("status", "error")]);
                return SegmentOutcome::Failed;
            }
        };

        // The guard moves into the blocking transfer with the destination buffers.
        // Cancelling this future cannot release either while the READ is running.
        if response.transfer_endpoint.is_empty()
            || response.blocks.len() != block_hashes.len()
            || response
                .blocks
                .iter()
                .zip(&block_hashes)
                .any(|(block, hash)| block.block_hash != *hash)
        {
            warn!("Remote query to {remote_addr} returned invalid transfer authorization");
            drop(lock_guard);
            core_metrics()
                .remote_fetch_total
                .add(1, &[KeyValue::new("status", "error")]);
            return SegmentOutcome::Failed;
        }

        // Mooncake READ all blocks + build SealedBlocks.
        let transfer_timeout = transfer_timeout_from_server(response.lock_timeout_secs);
        let blocks = response.blocks;
        let total_bytes: u64 = blocks
            .iter()
            .flat_map(|b| &b.slots)
            .map(|s| s.k_size + s.v_size)
            .sum();
        let (result, transfer_timing) = match fetch_blocks_via_mooncake(
            &self.transfer,
            &self.allocate_fn,
            namespace,
            &response.transfer_endpoint,
            &blocks,
            transfer_timeout,
            lock_guard,
        )
        .await
        {
            Ok(r) => r,
            Err(e) => {
                warn!("Mooncake transfer from {remote_addr} failed: {e}");
                self.transfer
                    .engine()
                    .invalidate_segment(&response.transfer_endpoint);
                core_metrics()
                    .remote_fetch_total
                    .add(1, &[KeyValue::new("status", "error")]);
                return SegmentOutcome::Failed;
            }
        };

        let elapsed = t0.elapsed();
        let mb = total_bytes as f64 / (1024.0 * 1024.0);
        let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
        let throughput_mib_s = if elapsed.as_secs_f64() > 0.0 {
            mb / elapsed.as_secs_f64()
        } else {
            0.0
        };
        info!(
            "Mooncake fetch summary: req_id={req_id} remote={remote_addr} blocks={}/{} slots={} descs={} slabs={} bytes_mib={mb:.1} total_ms={elapsed_ms:.2} tp_mib_s={throughput_mib_s:.0}",
            result.len(),
            block_hashes.len(),
            transfer_timing.slot_count,
            transfer_timing.transfer_desc_count,
            transfer_timing.numa_slab_count,
        );
        info!(
            "Mooncake fetch stages: req_id={req_id} remote={remote_addr} query_ms={:.2} build_transfer_tasks_ms={:.2} transfer_wait_ms={:.2} rebuild_ms={:.2}",
            query_elapsed.as_secs_f64() * 1000.0,
            transfer_timing.build_transfer_tasks.as_secs_f64() * 1000.0,
            transfer_timing.mooncake_wait.as_secs_f64() * 1000.0,
            transfer_timing.rebuild.as_secs_f64() * 1000.0,
        );
        let m = core_metrics();
        let ok = &[KeyValue::new("status", "ok")];
        m.remote_fetch_total.add(1, ok);
        m.remote_fetch_duration_seconds
            .record(elapsed.as_secs_f64(), ok);
        m.remote_fetch_bytes.add(total_bytes, ok);
        for (stage, duration) in [
            ("allocation", transfer_timing.build_transfer_tasks),
            ("read", transfer_timing.mooncake_wait),
            ("rebuild", transfer_timing.rebuild),
        ] {
            m.remote_stage_duration_seconds.record(
                duration.as_secs_f64(),
                &[KeyValue::new("stage", stage), KeyValue::new("status", "ok")],
            );
        }
        SegmentOutcome::Fetched(result)
    }
}

impl MooncakeFetchStore {
    pub(crate) fn new(
        catalog_client: Arc<CatalogClient>,
        transfer: Arc<MooncakeTransport>,
        allocate_fn: AllocateFn,
        membership: Arc<orbitkv_catalog::MembershipView>,
    ) -> Self {
        info!(
            "Mooncake remote fetch enabled (advertise={})",
            membership.owner().endpoint
        );
        Self {
            catalog_client,
            completions: Arc::new(TransferCompletions::default()),
            membership,
            transfer,
            allocate_fn,
        }
    }

    /// Use cached positive evidence before consulting the directory.
    pub(crate) async fn query_plan(
        &self,
        namespace: &str,
        hashes: &[Vec<u8>],
    ) -> Option<FetchPlan> {
        if !self.membership.permits(self.membership.owner()) {
            return None;
        }
        let mut candidates = match self.catalog_client.locate_blocks(namespace, hashes).await {
            Ok(candidates) => candidates,
            Err(e) => {
                warn!("Candidate discovery failed: {e}");
                return None;
            }
        };
        for row in &mut candidates {
            row.replicas
                .retain(|replica| self.membership.permits(&replica.owner));
        }
        FetchPlan::new(candidates)
    }

    pub(crate) async fn fetch_plan(
        &self,
        plan: &FetchPlan,
        req_id: &str,
        namespace: &str,
        hashes: &[Vec<u8>],
    ) -> PrefetchResult {
        if !plan.matches(namespace, hashes) {
            warn!("Remote fetch plan does not match the requested state");
            return Vec::new();
        }
        let started_at = Instant::now();
        let (fetched, attempts, completed) = execute_fetch_plan(self, plan, req_id).await;
        let metrics = core_metrics();
        metrics
            .remote_fetch_plan_segments
            .record(attempts as u64, &[]);
        metrics
            .remote_fetch_plan_completed_segments
            .record(completed as u64, &[]);
        info!(
            "Mooncake fetch plan: req_id={req_id} attempted_segments={attempts} completed_segments={completed} planned_blocks={} fetched_blocks={} total_ms={:.2}",
            plan.block_count(),
            fetched.len(),
            started_at.elapsed().as_secs_f64() * 1000.0
        );
        fetched
    }
}

/// One fetched slot: its Mooncake-staged segments plus the NUMA node they sit on.
/// The NUMA travels with the slot so a re-served block advertises real topology.
type StagedSlot = (Vec<SegmentAlloc>, NumaNode);
/// A staged block awaiting SealedBlock rebuild: its hash and per-slot allocations.
type StagedBlock = (Vec<u8>, Vec<StagedSlot>);

/// Allocate local memory, execute one Mooncake READ batch, and rebuild blocks.
async fn fetch_blocks_via_mooncake(
    transfer: &Arc<MooncakeTransport>,
    allocate_fn: &AllocateFn,
    namespace: &str,
    transfer_endpoint: &str,
    blocks: &[TransferBlockInfo],
    transfer_timeout: Duration,
    lock_guard: TransferLockGuard,
) -> Result<(PrefetchResult, TransferTiming), String> {
    if blocks.is_empty() {
        return Ok((Vec::new(), TransferTiming::default()));
    }

    let mut slabs = ChunkedSlabs::new(
        allocate_fn,
        FETCH_CHUNK_BYTES,
        sum_segment_bytes_by_numa(blocks)?,
    );

    // (block_hash, Vec<(slot_segments, slot_numa)>) — for building SealedBlock afterwards.
    // The per-slot NUMA is preserved so a re-served fetched block advertises real topology.
    let mut block_allocs: Vec<StagedBlock> = Vec::new();
    let mut slot_count = 0usize;
    let build_start = Instant::now();

    let (all_descs, mut timing) = {
        let mut all_descs: Vec<TransferSlice> = Vec::new();

        for block_info in blocks {
            slot_count += block_info.slots.len();
            let mut slot_allocs = Vec::with_capacity(block_info.slots.len());

            for slot in &block_info.slots {
                let mut segments = Vec::new();
                let numa = NumaNode(slot.numa_node);

                // K segment
                if slot.k_size > 0 {
                    let len = usize::try_from(slot.k_size)
                        .map_err(|_| format!("K size exceeds usize: {}", slot.k_size))?;
                    let (local_ptr, alloc) = slabs.alloc_segment(numa, len, "K")?;
                    if slot.k_ptr == 0 {
                        return Err("remote K ptr is null".to_string());
                    }
                    all_descs.push(TransferSlice {
                        local: local_ptr,
                        remote_address: slot.k_ptr,
                        length: len,
                    });
                    segments.push(SegmentAlloc {
                        ptr_addr: local_ptr.as_ptr() as u64,
                        alloc,
                        size: len,
                    });
                }

                // V segment (split KV)
                if slot.v_size > 0 && slot.v_ptr != 0 {
                    let len = usize::try_from(slot.v_size)
                        .map_err(|_| format!("V size exceeds usize: {}", slot.v_size))?;
                    let (local_ptr, alloc) = slabs.alloc_segment(numa, len, "V")?;
                    all_descs.push(TransferSlice {
                        local: local_ptr,
                        remote_address: slot.v_ptr,
                        length: len,
                    });
                    segments.push(SegmentAlloc {
                        ptr_addr: local_ptr.as_ptr() as u64,
                        alloc,
                        size: len,
                    });
                }

                slot_allocs.push((segments, numa));
            }

            block_allocs.push((block_info.block_hash.clone(), slot_allocs));
        }

        if all_descs.is_empty() {
            let timing = TransferTiming {
                build_transfer_tasks: build_start.elapsed(),
                slot_count,
                numa_slab_count: slabs.chunk_count,
                ..TransferTiming::default()
            };
            return Ok((Vec::new(), timing));
        }

        let transfer_desc_count = all_descs.len();

        let timing = TransferTiming {
            build_transfer_tasks: build_start.elapsed(),
            transfer_desc_count,
            slot_count,
            numa_slab_count: slabs.chunk_count,
            ..TransferTiming::default()
        };
        (all_descs, timing)
    };

    let wait_start = Instant::now();
    let transfer = Arc::clone(transfer);
    let transfer_endpoint = transfer_endpoint.to_string();
    let (block_allocs, transferred) = lock_guard
        .run_with_buffers(block_allocs, move || {
            transfer.engine().submit_and_wait(
                TransferOp::Read,
                &transfer_endpoint,
                &all_descs,
                transfer_timeout,
            )
        })
        .await
        .map_err(|error| format!("Mooncake READ task failed: {error}"))?;
    transferred.map_err(|error| format!("Mooncake READ failed: {error}"))?;
    timing.mooncake_wait = wait_start.elapsed();

    // Build SealedBlocks from allocated memory
    let rebuild_start = Instant::now();
    let mut result: PrefetchResult = Vec::with_capacity(block_allocs.len());
    for (hash, slot_allocs) in block_allocs {
        let key = StateKey::new(namespace.to_string(), hash);
        let slots: Vec<(RawBlock, NumaNode)> = slot_allocs
            .into_iter()
            .map(|(segs, numa)| {
                let segments: Vec<Segment> = segs
                    .into_iter()
                    .map(|sa| {
                        let ptr = NonNull::new(sa.ptr_addr as *mut u8)
                            .expect("slab segment pointer must be non-null");
                        Segment::new(ptr, sa.size, sa.alloc)
                    })
                    .collect();
                (RawBlock::new(segments), numa)
            })
            .collect();
        let sealed = Arc::new(SealedBlock::from_slots(slots));
        result.push((key, sealed));
    }
    timing.rebuild = rebuild_start.elapsed();

    Ok((result, timing))
}

/// Total staged bytes per NUMA node for one fetch batch. Used to right-size
/// the last chunk of each NUMA so small fetches don't over-allocate.
fn sum_segment_bytes_by_numa(
    blocks: &[TransferBlockInfo],
) -> Result<HashMap<NumaNode, u64>, String> {
    let mut bytes_per_numa: HashMap<NumaNode, u64> = HashMap::new();
    for block_info in blocks {
        for slot in &block_info.slots {
            let numa = NumaNode(slot.numa_node);
            let mut add = 0u64;
            if slot.k_size > 0 {
                add += slot.k_size;
            }
            if slot.v_size > 0 && slot.v_ptr != 0 {
                add = add
                    .checked_add(slot.v_size)
                    .ok_or_else(|| format!("segment bytes overflow on {numa}"))?;
            }
            let total = bytes_per_numa.entry(numa).or_insert(0);
            *total = total
                .checked_add(add)
                .ok_or_else(|| format!("numa bytes overflow while summing segments on {numa}"))?;
        }
    }
    Ok(bytes_per_numa)
}

/// Bump allocator over bounded pinned chunks, one active chunk per NUMA node.
/// Each staged segment holds an Arc to its own chunk, so starting a fresh
/// chunk never invalidates previously staged segments, and fetched blocks are
/// freed chunk-by-chunk on eviction instead of all-or-nothing per fetch.
///
/// A chunk is sized `min(remaining bytes on that NUMA, chunk_bytes)`: the cap
/// bounds LRU-reclaim amplification on large fetches, the remaining-bytes
/// clamp keeps small fetches from grabbing a whole `chunk_bytes` slab (which
/// would fail outright on pools smaller than the cap).
struct ChunkedSlabs<'a> {
    allocate_fn: &'a AllocateFn,
    chunk_bytes: u64,
    current: HashMap<NumaNode, NumaSlab>,
    /// Bytes of this batch not yet staged, per NUMA.
    remaining: HashMap<NumaNode, u64>,
    chunk_count: usize,
}

impl<'a> ChunkedSlabs<'a> {
    fn new(
        allocate_fn: &'a AllocateFn,
        chunk_bytes: u64,
        remaining: HashMap<NumaNode, u64>,
    ) -> Self {
        Self {
            allocate_fn,
            chunk_bytes,
            current: HashMap::new(),
            remaining,
            chunk_count: 0,
        }
    }

    fn alloc_segment(
        &mut self,
        numa: NumaNode,
        len: usize,
        segment_kind: &str,
    ) -> Result<(NonNull<u8>, Arc<crate::memory::pool::PinnedAllocation>), String> {
        if let Some(slab) = self.current.get_mut(&numa)
            && let Ok(seg) = slab.allocate(len, segment_kind)
        {
            self.consume_remaining(numa, len);
            return Ok(seg);
        }

        // No chunk on this NUMA yet, or the current one can't fit the segment.
        let remaining = *self
            .remaining
            .get(&numa)
            .expect("remaining bytes tracked for every NUMA in the batch");
        let chunk = remaining.min(self.chunk_bytes).max(len as u64);
        let allocation = (self.allocate_fn)(chunk, Some(numa)).ok_or_else(|| {
            format!("failed to allocate fetch chunk ({chunk} bytes) on {numa} for {segment_kind}")
        })?;
        let capacity = usize::try_from(chunk)
            .map_err(|_| format!("fetch chunk size exceeds usize: {chunk}"))?;
        self.chunk_count += 1;
        self.current.insert(
            numa,
            NumaSlab {
                allocation,
                next_offset: 0,
                capacity,
            },
        );
        self.consume_remaining(numa, len);
        self.current
            .get_mut(&numa)
            .expect("chunk just inserted")
            .allocate(len, segment_kind)
    }

    fn consume_remaining(&mut self, numa: NumaNode, len: usize) {
        let rem = self
            .remaining
            .get_mut(&numa)
            .expect("remaining bytes tracked for every NUMA in the batch");
        *rem = rem.saturating_sub(len as u64);
    }
}

struct NumaSlab {
    allocation: Arc<crate::memory::pool::PinnedAllocation>,
    next_offset: usize,
    capacity: usize,
}

impl NumaSlab {
    fn allocate(
        &mut self,
        len: usize,
        segment_kind: &str,
    ) -> Result<(NonNull<u8>, Arc<crate::memory::pool::PinnedAllocation>), String> {
        let end = self.next_offset.checked_add(len).ok_or_else(|| {
            format!(
                "slab offset overflow while allocating {segment_kind}: offset={} len={len} capacity={}",
                self.next_offset, self.capacity
            )
        })?;
        if end > self.capacity {
            return Err(format!(
                "slab exhausted while allocating {segment_kind}: offset={} len={len} capacity={}",
                self.next_offset, self.capacity
            ));
        }

        let ptr = unsafe { self.allocation.as_non_null().as_ptr().add(self.next_offset) };
        self.next_offset = end;
        let ptr = NonNull::new(ptr).ok_or_else(|| "slab pointer is null".to_string())?;
        Ok((ptr, Arc::clone(&self.allocation)))
    }
}

struct SegmentAlloc {
    ptr_addr: u64,
    alloc: Arc<crate::memory::pool::PinnedAllocation>,
    size: usize,
}

#[derive(Default)]
struct TransferTiming {
    build_transfer_tasks: Duration,
    mooncake_wait: Duration,
    rebuild: Duration,
    transfer_desc_count: usize,
    slot_count: usize,
    numa_slab_count: usize,
}

/// Compute client-side transfer timeout from server's lock timeout.
/// This is a submission budget, not proof of source lifetime: Mooncake drains
/// already-submitted work even past the deadline, retaining source reservations.
/// Orphan revocation and cross-host failure qualification remain open.
fn transfer_timeout_from_server(lock_timeout_secs: u32) -> Duration {
    let server = Duration::from_secs(lock_timeout_secs as u64);
    server
        .saturating_sub(LOCK_TIMEOUT_MARGIN)
        .max(MIN_TRANSFER_TIMEOUT)
}

#[cfg(test)]
#[path = "../../tests/unit/backing/mooncake_fetch.rs"]
mod tests;
