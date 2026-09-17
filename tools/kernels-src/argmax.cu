// Greedy sampling：bf16 logits 行 argmax，两段式（单 block 版实测 55µs/步，
// 单 SM 读 300KB 是瓶颈）。平局取最小下标（与 CPU 逐个扫描语义一致）：
// stage1 每 block 处理连续 chunk（块间下标天然有序），stage2 归并分部结果。
//   nvcc -cubin -arch=sm_103a -o kernels/argmax.cubin tools/kernels-src/argmax.cu
#include <cuda_bf16.h>
#include <limits.h>

struct Pair {
  float v;
  int i;
};

__device__ static inline Pair pick(Pair a, Pair b) {
  return (b.v > a.v || (b.v == a.v && b.i < a.i)) ? b : a;
}

template <int BLOCK>
__device__ static inline Pair block_reduce(Pair best) {
  __shared__ float sv[BLOCK];
  __shared__ int si[BLOCK];
  sv[threadIdx.x] = best.v;
  si[threadIdx.x] = best.i;
  __syncthreads();
  for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      Pair o = {sv[threadIdx.x + stride], si[threadIdx.x + stride]};
      Pair m = pick({sv[threadIdx.x], si[threadIdx.x]}, o);
      sv[threadIdx.x] = m.v;
      si[threadIdx.x] = m.i;
    }
    __syncthreads();
  }
  return {sv[0], si[0]};
}

// grid [tokens, NB]，block 1024：block (t,b) 扫行 t 的第 b 个连续 chunk，
// 写出 pmax/pidx[t*NB+b]。
extern "C" __global__ void kern_argmax_partial_bf16(
    const __nv_bfloat16* __restrict__ logits, float* __restrict__ pmax,
    int* __restrict__ pidx, int n) {
  int nb = gridDim.y;
  int chunk = (n + nb - 1) / nb;
  int lo = blockIdx.y * chunk;
  int hi = min(lo + chunk, n);
  const __nv_bfloat16* row = logits + (long long)blockIdx.x * n;
  Pair best = {-__builtin_inff(), INT_MAX};
  for (int j = lo + threadIdx.x; j < hi; j += blockDim.x) {
    best = pick(best, {__bfloat162float(row[j]), j});
  }
  best = block_reduce<1024>(best);
  if (threadIdx.x == 0) {
    pmax[blockIdx.x * nb + blockIdx.y] = best.v;
    pidx[blockIdx.x * nb + blockIdx.y] = best.i;
  }
}

// grid [tokens]，block 64（= NB）：归并 stage1 的分部结果，写 token id。
extern "C" __global__ void kern_argmax_final_i64(
    const float* __restrict__ pmax, const int* __restrict__ pidx,
    long long* __restrict__ out, int nb) {
  Pair best = {-__builtin_inff(), INT_MAX};
  if (threadIdx.x < nb) {
    best = {pmax[blockIdx.x * nb + threadIdx.x],
            pidx[blockIdx.x * nb + threadIdx.x]};
  }
  best = block_reduce<64>(best);
  if (threadIdx.x == 0) out[blockIdx.x] = best.i;
}
