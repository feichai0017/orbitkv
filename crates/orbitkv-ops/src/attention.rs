use orbitkv_compiler::prelude::*;
use orbitkv_compiler::shape::Expression;

/// Gather entire rows from a 2D tensor using row indices.
///
/// - `data`: (R, D) tensor
/// - `indices`: (N,) Int tensor of row indices
/// - `d`: the number of columns (D), must match data's second dimension
///
/// Returns: (N, D) tensor where output[i] = data[indices[i]]
pub fn gather_rows(data: GraphTensor, indices: GraphTensor, d: usize) -> GraphTensor {
    assert_eq!(indices.dtype, DType::Int);
    let n = indices.dims1();

    // base[i] = indices[i] * D → flat starting position for each row
    let base = (indices * d).expand_dim(1, d); // (N, D) broadcast along cols

    // col[j] = j → column offsets 0..D
    let col = data.graph().arange(d as i32).expand_dim(0, n); // (N, D) broadcast along rows

    // flat_idx[i,j] = indices[i] * D + j
    let flat_idx = base + col;

    data.gather(flat_idx)
}

/// Scatter entire rows into a 2D tensor using row indices.
///
/// - `src`: (N, D) tensor of values to write
/// - `indices`: (N,) Int tensor of destination row indices
/// - `dest`: (R, D) tensor to write into (copied first, then overwritten at index positions)
/// - `d`: the number of columns (D)
///
/// Returns: (R, D) tensor where output = copy(dest); output[indices[i]] = src[i]
pub fn scatter_rows(
    src: GraphTensor,
    indices: GraphTensor,
    dest: GraphTensor,
    d: usize,
) -> GraphTensor {
    assert_eq!(indices.dtype, DType::Int);
    let n = indices.dims1();

    // Same index expansion as gather_rows
    let base = (indices * d).expand_dim(1, d);
    let col = src.graph().arange(d as i32).expand_dim(0, n);
    let flat_idx = base + col;

    src.scatter(flat_idx, dest)
}

/// Pure HLIR paged attention for one layer with causal masking.
///
/// Inputs:
/// - `q`:           (s, hidden)         f32 — query vectors
/// - `k_new`:       (s, kv_dim)         f32 — new key vectors
/// - `v_new`:       (s, kv_dim)         f32 — new value vectors
/// - `k_cache`:     (num_slots, kv_dim) f32 — key cache (preallocated)
/// - `v_cache`:     (num_slots, kv_dim) f32 — value cache (preallocated)
/// - `gather_idx`:  (ctx_len,)          Int — which cache slots to read
/// - `scatter_idx`: (s,)                Int — which cache slots to write new KV into
/// - `prev_seq`:    number of previously cached tokens (for causal mask offset)
/// - `n_heads`:     number of query heads
/// - `n_kv_heads`:  number of KV heads (for GQA)
/// - `head_dim`:    dimension per head
///
/// Returns: (attn_out, k_cache_new, v_cache_new)
///   - `attn_out`:     (s, hidden)         f32
///   - `k_cache_new`:  (num_slots, kv_dim) f32
///   - `v_cache_new`:  (num_slots, kv_dim) f32
#[allow(clippy::too_many_arguments)]
pub fn paged_attention(
    q: GraphTensor,
    k_new: GraphTensor,
    v_new: GraphTensor,
    k_cache: GraphTensor,
    v_cache: GraphTensor,
    gather_idx: GraphTensor,
    scatter_idx: GraphTensor,
    prev_seq: Expression,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> (GraphTensor, GraphTensor, GraphTensor) {
    let kv_dim = n_kv_heads * head_dim;
    let kv_groups = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let s = q.dims()[0];
    let ctx = gather_idx.dims()[0];
    let cx = q.graph();

    // ── Phase 1: Write new KV into cache ──
    let k_cache = scatter_rows(k_new, scatter_idx, k_cache, kv_dim);
    let v_cache = scatter_rows(v_new, scatter_idx, v_cache, kv_dim);

    // ── Phase 2: Gather context KV from cache ──
    let k = gather_rows(k_cache, gather_idx, kv_dim); // (ctx, kv_dim)
    let v = gather_rows(v_cache, gather_idx, kv_dim); // (ctx, kv_dim)

    // ── Phase 3: Reshape for multi-head attention ──
    // Q: (s, hidden) → (s, n_heads, head_dim) → (s, n_kv_heads, kv_groups, head_dim)
    //                 → (n_kv_heads, kv_groups, s, head_dim)
    let q = q
        .split_dims(1, head_dim) // (s, n_heads, head_dim)
        .split_dims(1, kv_groups) // (s, n_kv_heads, kv_groups, head_dim)
        .permute((1, 2, 0, 3)); // (n_kv_heads, kv_groups, s, head_dim)

    // K: (ctx, kv_dim) → (ctx, n_kv_heads, head_dim) → (n_kv_heads, head_dim, ctx)
    let k = k
        .split_dims(1, head_dim) // (ctx, n_kv_heads, head_dim)
        .permute((1, 2, 0)); // (n_kv_heads, head_dim, ctx)

    // V: (ctx, kv_dim) → (ctx, n_kv_heads, head_dim) → (n_kv_heads, ctx, head_dim)
    let v = v
        .split_dims(1, head_dim) // (ctx, n_kv_heads, head_dim)
        .permute((1, 0, 2)); // (n_kv_heads, ctx, head_dim)

    // ── Phase 4: Attention ──
    // Broadcast K, V over kv_groups dimension
    let k = k.expand_dim(1, kv_groups); // (n_kv_heads, kv_groups, head_dim, ctx)
    let v = v.expand_dim(1, kv_groups); // (n_kv_heads, kv_groups, ctx, head_dim)

    // QK^T: (n_kv_heads, kv_groups, s, head_dim) @ (n_kv_heads, kv_groups, head_dim, ctx)
    //     → (n_kv_heads, kv_groups, s, ctx)
    let scores = q.matmul(k) * scale;

    // Build causal mask: query at position prev_seq+i can attend to context j iff j <= prev_seq+i.
    // row_vals[i] = prev_seq + i, col_vals[j] = j
    // mask[i,j] = -1e9 where row_vals[i] < col_vals[j], else 0
    let z = Expression::from('z');
    let row_vals = cx.iota(z + prev_seq, s).expand_dim(1, ctx); // (s, ctx)
    let col_vals = cx.arange(ctx).expand_dim(0, s); // (s, ctx)
    let mask = row_vals
        .cast(DType::F32)
        .lt(col_vals.cast(DType::F32))
        .cast(DType::F32)
        * -1e9;

    // Broadcast (s, ctx) → (n_kv_heads, kv_groups, s, ctx)
    let mask = mask.expand_dim(0, n_kv_heads).expand_dim(1, kv_groups);
    let scores = scores + mask;

    // Softmax over context dimension (axis 3)
    let weights = scores.softmax(3);

    // Weighted sum: (n_kv_heads, kv_groups, s, ctx) @ (n_kv_heads, kv_groups, ctx, head_dim)
    //            → (n_kv_heads, kv_groups, s, head_dim)
    let out = weights.matmul(v);

    // ── Phase 5: Reshape output ──
    // (n_kv_heads, kv_groups, s, head_dim) → (s, n_kv_heads, kv_groups, head_dim)
    let mut out = out.permute((2, 0, 1, 3));
    out.shape = ShapeTracker::new((s, n_heads * head_dim));

    (out, k_cache, v_cache)
}

#[cfg(test)]
#[path = "../tests/unit/attention/mod.rs"]
mod tests;
