// Row-wise top-16 of bf16 logits (vLLM's LogitsProcessor.get_top_k_tokens
// for the DFlash2 candidate set: torch.topk(k=16), values as f32).
//
// grid.x = rows, block 1024.  Each thread keeps the top-16 of its strided
// slice in registers (insertion, value desc / index asc), then the block
// merges by 16 rounds of "every thread offers its best unconsumed
// candidate, block argmax takes one".  Ties: larger value first, then
// smaller index (torch leaves tie order unspecified; the draft only
// affects acceptance, never the output).
//
//   nvcc -cubin -arch=sm_103a -o kernels/topk_row.cubin tools/kernels-src/topk_row.cu
#include <cuda_bf16.h>

#define K 16
#define BLOCK 1024

__device__ static inline bool better(float v, int i, float bv, int bi) {
  return v > bv || (v == bv && i < bi);
}

extern "C" __global__ void kern_topk16_bf16(
    const __nv_bfloat16* __restrict__ logits, long long* __restrict__ ids,
    float* __restrict__ vals, int n, int row_stride) {
  const __nv_bfloat16* row = logits + (long long)blockIdx.x * row_stride;
  float lv[K];
  int li[K];
#pragma unroll
  for (int j = 0; j < K; ++j) { lv[j] = -INFINITY; li[j] = 0x7fffffff; }
  for (int i = threadIdx.x; i < n; i += BLOCK) {
    const float v = __bfloat162float(row[i]);
    if (!better(v, i, lv[K - 1], li[K - 1])) continue;
    int j = K - 1;
    while (j > 0 && better(v, i, lv[j - 1], li[j - 1])) { lv[j] = lv[j - 1]; li[j] = li[j - 1]; --j; }
    lv[j] = v; li[j] = i;
  }
  __shared__ float sv[BLOCK];
  __shared__ int si[BLOCK];
  int head = 0;
  for (int r = 0; r < K; ++r) {
    // offer
    sv[threadIdx.x] = head < K ? lv[head] : -INFINITY;
    si[threadIdx.x] = head < K ? li[head] : 0x7fffffff;
    __syncthreads();
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        if (better(sv[threadIdx.x + stride], si[threadIdx.x + stride], sv[threadIdx.x], si[threadIdx.x])) {
          sv[threadIdx.x] = sv[threadIdx.x + stride];
          si[threadIdx.x] = si[threadIdx.x + stride];
        }
      }
      __syncthreads();
    }
    const float wv = sv[0];
    const int wi = si[0];
    if (threadIdx.x == 0) {
      ids[(long long)blockIdx.x * K + r] = wi;
      vals[(long long)blockIdx.x * K + r] = wv;
    }
    if (head < K && li[head] == wi && lv[head] == wv) ++head;  // the winner advances
    __syncthreads();
  }
}
