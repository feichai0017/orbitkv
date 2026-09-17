// The GDN "advance" pass of speculative decoding, recomputed instead of
// checkpointed ("The Mamba in the Llama", Wang et al. 2024): after the
// accept step the recurrent kernel re-runs the verify rows from the
// after-anchor state and stores after row `a` (the last accepted draft) —
// row 0 (the anchor) is already in that state, so it must be an identity
// update, and the conv line must be shifted so the next round reads its
// history at offset 0. Two small kernels do the parts vLLM's kernels
// don't:
//
//   kern_mask_row0:   dst = src with row 0 of every `rows_per_seq` rows set
//                     to -inf (bf16). On the gate inputs a/b that makes the
//                     fused sigmoid-gating kernel's update the identity:
//                     softplus(-inf) = 0 => g = 0 => exp(g) = 1, and
//                     sigmoid(-inf) = 0 => beta = 0.
//   kern_conv_shift:  conv line `idx[seq * idx_stride + a]` (token-major
//                     [state_len][dim] bf16 rows of `tok_bytes`): tokens
//                     a..a+2 -> 0..2 where a = nacc[seq] - 1, the three
//                     history taps of the next anchor. 16-byte vectors;
//                     every thread reads its three vectors before writing
//                     (the ranges overlap when a < 3). Line 0 (null) and
//                     a == 0 are no-ops.
//
//   nvcc -cubin -arch=sm_103a -o target/cubins/gdn_advance.cubin tools/kernels-src/gdn_advance.cu
#include <cstdint>

extern "C" __global__ void kern_mask_row0(
    uint16_t* __restrict__ dst, const uint16_t* __restrict__ src, int rows, int rows_per_seq, int cols) {
  const int r = blockIdx.x;
  if (r >= rows) return;
  const bool masked = (r % rows_per_seq) == 0;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    dst[(size_t)r * cols + c] = masked ? (uint16_t)0xFF80 : src[(size_t)r * cols + c];
  }
}

extern "C" __global__ void kern_conv_shift(
    const int* __restrict__ idx, int idx_stride, unsigned char* __restrict__ state,
    const int* __restrict__ nacc, long long line_bytes, long long tok_bytes) {
  const int seq = blockIdx.x;
  const long long a = (long long)nacc[seq] - 1;
  if (a <= 0) return;
  // The advance pass keeps the line in column `a` of its line-table cell
  // (column 0 during verify), so that is where it is found here.
  const long long line = idx[(size_t)seq * idx_stride + a];
  if (line <= 0) return;
  const long long v = (long long)blockIdx.y * blockDim.x + threadIdx.x;
  if (v * 16 >= tok_bytes) return;
  unsigned char* base = state + line * line_bytes + v * 16;
  const uint4 t0 = *reinterpret_cast<const uint4*>(base + a * tok_bytes);
  const uint4 t1 = *reinterpret_cast<const uint4*>(base + (a + 1) * tok_bytes);
  const uint4 t2 = *reinterpret_cast<const uint4*>(base + (a + 2) * tok_bytes);
  *reinterpret_cast<uint4*>(base) = t0;
  *reinterpret_cast<uint4*>(base + tok_bytes) = t1;
  *reinterpret_cast<uint4*>(base + 2 * tok_bytes) = t2;
}
