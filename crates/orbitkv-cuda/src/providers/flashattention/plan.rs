//! Pointer-free scratch planning and captured allocation ownership.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use orbitkv_compiler::prelude::{DynMap, FxHashMap, NodeIndex};

use super::{FlashAttention, jit};
use crate::{providers::DeviceBuffer, resource::ResourceViolation};

// FA3's scheduler vectorizes int32 metadata in 16-byte units.
const METADATA_ALIGNMENT: usize = 16;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Plan {
    pub query_tokens: usize,
    pub requests: usize,
    pub context_pages: usize,
    pub bytes: usize,
    page_table: usize,
    kv_lengths: usize,
    split_counts: usize,
    query_tiles: usize,
    batch_order: usize,
    head_swizzle: usize,
    tile_counter: usize,
    lse: usize,
}

fn overflow() -> ResourceViolation {
    ResourceViolation::ArithmeticOverflow {
        resource: "FlashAttention scratch",
    }
}

impl Plan {
    pub fn new(op: &FlashAttention, dimensions: &DynMap) -> Result<Self, ResourceViolation> {
        let resolve = |expression: orbitkv_compiler::prelude::Expression| {
            expression
                .exec(dimensions)
                .ok_or(ResourceViolation::UnresolvedExpression {
                    resource: "FlashAttention dimensions",
                })
        };
        let query_tokens = resolve(op.query_tokens)?;
        let requests = resolve(op.requests)?;
        let context_pages = resolve(op.context_pages)?;
        let geometry_ok = query_tokens > 0
            && requests > 0
            && query_tokens >= requests
            && context_pages >= requests
            && op.supports_geometry();
        let kv_tokens = context_pages
            .checked_mul(op.page_size)
            .ok_or_else(overflow)?;
        let packed_queries = query_tokens
            .checked_mul(op.query_heads)
            .ok_or_else(overflow)?;
        if !geometry_ok
            || [query_tokens, requests, kv_tokens, packed_queries]
                .iter()
                .any(|&value| i32::try_from(value).is_err())
        {
            return Err(ResourceViolation::HostResourcePlanning {
                name: "FlashAttention geometry/index range",
            });
        }
        let mut bytes = 0usize;
        let mut reserve =
            |elements: usize, element_bytes: usize| -> Result<usize, ResourceViolation> {
                let offset = bytes
                    .checked_add(METADATA_ALIGNMENT - 1)
                    .ok_or_else(overflow)?
                    / METADATA_ALIGNMENT
                    * METADATA_ALIGNMENT;
                let allocation = elements.checked_mul(element_bytes).ok_or_else(overflow)?;
                bytes = offset.checked_add(allocation).ok_or_else(overflow)?;
                Ok(offset)
            };
        let table_elements = requests.checked_mul(context_pages).ok_or_else(overflow)?;
        let metadata_lanes = METADATA_ALIGNMENT / size_of::<i32>();
        let padded_requests = requests
            .div_ceil(metadata_lanes)
            .checked_mul(metadata_lanes)
            .ok_or_else(overflow)?;
        let page_table = reserve(table_elements, size_of::<i32>())?;
        let kv_lengths = reserve(requests, size_of::<i32>())?;
        let split_counts = reserve(padded_requests, size_of::<i32>())?;
        let query_tiles = reserve(padded_requests, size_of::<i32>())?;
        let batch_order = reserve(padded_requests, size_of::<i32>())?;
        let head_swizzle = reserve(padded_requests, size_of::<i32>())?;
        let tile_counter = reserve(1, size_of::<i32>())?;
        let lse = reserve(packed_queries, size_of::<f32>())?;
        Ok(Self {
            query_tokens,
            requests,
            context_pages,
            bytes,
            page_table,
            kv_lengths,
            split_counts,
            query_tiles,
            batch_order,
            head_swizzle,
            tile_counter,
            lse,
        })
    }
}

pub(super) struct Prepared {
    pub plan: Plan,
    pub stream_key: usize,
    _scratch: CudaSlice<u8>,
    scratch_ptr: u64,
    num_sm: i32,
}

impl Prepared {
    pub fn new(plan: Plan, stream: &Arc<CudaStream>) -> anyhow::Result<Self> {
        let capability = stream.context().compute_capability()?;
        anyhow::ensure!(
            capability == (9, 0),
            "FlashAttention-3 adapter requires compute capability 9.0"
        );
        let num_sm = stream.context().attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        )?;
        let scratch = unsafe { stream.alloc::<u8>(plan.bytes)? };
        // Obtain the pointer before capture. A device_ptr guard created during
        // capture records a slice event into that graph; destroying the graph
        // before the slice would leave its destructor waiting on a dead event.
        // This allocation belongs to one stream and is retained by each graph
        // that uses it, so launches need no additional per-call slice events.
        let scratch_ptr = scratch.device_ptr(stream).0;
        Ok(Self {
            _scratch: scratch,
            scratch_ptr,
            plan,
            stream_key: stream.cu_stream() as usize,
            num_sm,
        })
    }

    pub fn launch(
        &self,
        op: &FlashAttention,
        stream: &Arc<CudaStream>,
        output: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
    ) -> anyhow::Result<()> {
        let [query, key, value, indices, query_indptr, page_indptr, last]: [NodeIndex; 7] = inputs
            .try_into()
            .map_err(|_| anyhow::anyhow!("FlashAttention requires seven inputs"))?;
        let buffer = |node| {
            buffers
                .get(&node)
                .copied()
                .ok_or_else(|| anyhow::anyhow!("FlashAttention buffer {node:?} is missing"))
        };
        let [q, k, v, indices, qptr, pptr, last, output] = [
            buffer(query)?,
            buffer(key)?,
            buffer(value)?,
            buffer(indices)?,
            buffer(query_indptr)?,
            buffer(page_indptr)?,
            buffer(last)?,
            buffer(output)?,
        ];
        let p = &self.plan;
        let value_bytes = op.dtype.bits().div_ceil(8);
        let query_bytes = p
            .query_tokens
            .checked_mul(op.query_heads)
            .and_then(|n| n.checked_mul(op.head_dim))
            .and_then(|n| n.checked_mul(value_bytes))
            .ok_or_else(|| anyhow::anyhow!("FlashAttention query byte overflow"))?;
        let page_bytes = op
            .page_size
            .checked_mul(op.kv_heads)
            .and_then(|n| n.checked_mul(op.head_dim))
            .and_then(|n| n.checked_mul(value_bytes))
            .ok_or_else(|| anyhow::anyhow!("FlashAttention page byte overflow"))?;
        anyhow::ensure!(
            q.len() == query_bytes && output.len() >= query_bytes,
            "FlashAttention query/output byte count mismatch"
        );
        anyhow::ensure!(
            !k.is_empty() && k.len() == v.len() && k.len().is_multiple_of(page_bytes),
            "FlashAttention K/V storage must contain equal complete pages"
        );
        anyhow::ensure!(
            indices.len() == p.context_pages * size_of::<i32>()
                && qptr.len() == (p.requests + 1) * size_of::<i32>()
                && pptr.len() == qptr.len()
                && last.len() == p.requests * size_of::<i32>(),
            "FlashAttention CSR byte count mismatch"
        );
        let cache_pages = i32::try_from(k.len() / page_bytes)?;
        let ptr = |offset: usize| (self.scratch_ptr + offset as u64) as *mut i32;
        let arguments = jit::Launch {
            query: q.ptr() as _,
            key: k.ptr() as _,
            value: v.ptr() as _,
            page_indices: indices.ptr() as _,
            query_indptr: qptr.ptr() as _,
            page_indptr: pptr.ptr() as _,
            last_page_len: last.ptr() as _,
            output: output.ptr() as _,
            page_table: ptr(p.page_table),
            kv_lengths: ptr(p.kv_lengths),
            split_counts: ptr(p.split_counts),
            query_tiles: ptr(p.query_tiles),
            batch_order: ptr(p.batch_order),
            head_swizzle: ptr(p.head_swizzle),
            tile_counter: ptr(p.tile_counter),
            lse: ptr(p.lse).cast(),
            query_tokens: p.query_tokens as i32,
            requests: p.requests as i32,
            query_heads: op.query_heads as i32,
            kv_heads: op.kv_heads as i32,
            page_size: op.page_size as i32,
            context_pages: p.context_pages as i32,
            cache_pages,
            num_sm: self.num_sm,
            scale: op.scale as f32,
            window_left: op.window_left as i32,
        };
        let library = jit::ensure_compiled(
            crate::target::CudaTarget::from_context(stream.context())?,
            op.config(),
        )?;
        let result = unsafe { (library.run)(&arguments, stream.cu_stream().cast()) };
        anyhow::ensure!(
            result == 0,
            "FlashAttention execution failed: {}",
            library.error()
        );
        Ok(())
    }
}
