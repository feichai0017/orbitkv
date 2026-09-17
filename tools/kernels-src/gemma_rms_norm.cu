// Gemma-style RMSNorm, bit-exact with vLLM's eager GemmaRMSNorm path
// (`ir.ops.rms_norm` / `fused_add_rms_norm`: a chain of ATen ops in f32):
//
//   z   = f32(x[r]) (+ f32(res[r]))          fused: res[r] := bf16(z)
//   var = mean(z^2)                           ATen reduce_kernel<512,1,MeanOps>
//   y   = bf16((z * rsqrtf(var + eps)) * w1)  w1 = f32(weight) + 1 (exported)
//
// The only non-trivial part is reproducing ATen's summation order for the
// mean, since a different order moves ulps and greedy decoding notices:
//   * the reduction is vectorized by 4 over the contiguous row; virtual lane
//     l (0 <= l < W) accumulates acc[i] += v[(l + k*W)*4 + i] for k = 0.. and
//     then s_l = ((acc0 + acc1) + acc2) + acc3;
//   * W is ATen's block width, a function of (N/4, rows) via
//     set_block_dimension: 512 for 1 row, 256 for 2-3, 128 for 4-7, 64 for
//     8-15, 32 for >= 16 (N/4 >= 512; for N=256 W caps at 64);
//   * if W > 32: shared-memory halving  s[l] += s[l+off], off = W/2 .. 32;
//     then the warp: s[l] += shfl_down(s, off), off = 16, 8, 4, 2, 1
//     (torch 2.13 reversed the classic 1..16 order on CUDA; check
//     ATen/native/cuda/Reduce.cuh block_x_reduce when bumping torch);
//   * mean = s_0 * factor, factor = f32(rows) / f32(rows*N).
// One 512-thread block per row emulates exactly that (threads >= W idle
// during the reduction).
//
// Rows are addressed as token/head pairs so the same kernel does the hidden
// norm (heads=1) and the per-head q/k norms reading strided views of the
// fused qkv projection and writing contiguous outputs:
//   row r -> token r/heads, head r%heads
//   in  = in  + token*in_tstride  + head*in_hstride
//   out = out + token*out_tstride + head*out_hstride
//
//   nvcc -cubin -arch=sm_103a -o kernels/gemma_rms_norm.cubin tools/kernels-src/gemma_rms_norm.cu
#include <cuda_bf16.h>

#define NT 512

__device__ static inline int last_pow2(int n) {
  n |= (n >> 1); n |= (n >> 2); n |= (n >> 4); n |= (n >> 8); n |= (n >> 16);
  return (n - (n >> 1)) > 0 ? (n - (n >> 1)) : 1;
}

// ATen ReduceConfig::set_block_dimension(dim0 = N/vec, dim1 = rows), mnt=512.
__device__ static inline int aten_block_width(int dim0, int dim1) {
  const int max_threads = NT;
  int dim0_pow2 = dim0 < max_threads ? last_pow2(dim0) : max_threads;
  int dim1_pow2 = dim1 < max_threads ? last_pow2(dim1) : max_threads;
  int block_width = min(dim0_pow2, 32);
  int block_height = min(dim1_pow2, max_threads / block_width);
  block_width = min(dim0_pow2, max_threads / block_height);
  return block_width;
}

// Sum of sq[0..N) (already in shared memory as f32) in ATen's order for W
// lanes; returns the lane-0 result to every thread. N % 4 == 0.
__device__ static float aten_row_sum(const float* __restrict__ sq, int N, int W,
                                     float* __restrict__ s) {
  const int t = threadIdx.x;
  float value = 0.f;
  if (t < W) {
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int idx = t; idx * 4 + 3 < N; idx += W) {
      const float4 v = *reinterpret_cast<const float4*>(sq + idx * 4);
      acc[0] += v.x;
      acc[1] += v.y;
      acc[2] += v.z;
      acc[3] += v.w;
    }
    value = ((acc[0] + acc[1]) + acc[2]) + acc[3];
  }
  if (W > 32) {
    s[t] = value;
    for (int off = W / 2; off >= 32; off >>= 1) {
      __syncthreads();
      if (t < off && t + off < W) {
        value += s[t + off];
        s[t] = value;
      }
    }
  }
  __syncthreads();
  if (t < 32) {
    // torch >= 2.13 block_x_reduce: shuffle offsets decrease (16 .. 1).
    for (int off = 16; off > 0; off >>= 1) {
      float other = __shfl_down_sync(0xffffffffu, value, off);
      value += other;
    }
    if (t == 0) s[0] = value;
  }
  __syncthreads();
  return s[0];
}

// One 512-thread block per row. The row is N/8 chunks of 8 bf16 (16 bytes);
// thread t owns chunks t, t + NT, .. and issues every load it needs before
// touching shared memory, so a row costs one memory round trip, not one per
// loop iteration. N % 8 == 0 and N <= 8 * NT * MAXC; every row base is 16-byte
// aligned (strides are multiples of 8 elements).
#define MAXC 2

__device__ static inline void unpack8(const uint4 v, float f[8]) {
  const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    f[2 * i] = __bfloat162float(p[i].x);
    f[2 * i + 1] = __bfloat162float(p[i].y);
  }
}

__device__ static inline uint4 pack8(const float f[8]) {
  uint4 v;
  __nv_bfloat162* p = reinterpret_cast<__nv_bfloat162*>(&v);
#pragma unroll
  for (int i = 0; i < 4; i++) p[i] = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
  return v;
}

template <bool FUSED>
__device__ static inline void gemma_norm_row(
    __nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ in,
    __nv_bfloat16* __restrict__ res, const float* __restrict__ w1, int N,
    int rows, float eps, float* __restrict__ z, float* __restrict__ s) {
  const int nchunk = N / 8;
  uint4 xin[MAXC], rin[MAXC];
  float4 wv[MAXC][2];
#pragma unroll
  for (int k = 0; k < MAXC; k++) {
    const int c = threadIdx.x + k * NT;
    if (c < nchunk) {
      xin[k] = *reinterpret_cast<const uint4*>(in + c * 8);
      if (FUSED) rin[k] = *reinterpret_cast<const uint4*>(res + c * 8);
      wv[k][0] = *reinterpret_cast<const float4*>(w1 + c * 8);
      wv[k][1] = *reinterpret_cast<const float4*>(w1 + c * 8 + 4);
    }
  }
  // z = f32(x) (+ f32(res)); res := bf16(z); sq = z * z (ATen's pow(2))
  float zk[MAXC][8];
#pragma unroll
  for (int k = 0; k < MAXC; k++) {
    const int c = threadIdx.x + k * NT;
    if (c < nchunk) {
      float x[8], r[8], sq[8];
      unpack8(xin[k], x);
      if (FUSED) unpack8(rin[k], r);
#pragma unroll
      for (int e = 0; e < 8; e++) {
        float v = x[e];
        if (FUSED) v = __fadd_rn(v, r[e]);
        zk[k][e] = v;
        sq[e] = __fmul_rn(v, v);
      }
      if (FUSED) *reinterpret_cast<uint4*>(res + c * 8) = pack8(zk[k]);
      *reinterpret_cast<float4*>(s + NT + c * 8) = make_float4(sq[0], sq[1], sq[2], sq[3]);
      *reinterpret_cast<float4*>(s + NT + c * 8 + 4) = make_float4(sq[4], sq[5], sq[6], sq[7]);
    }
  }
  __syncthreads();
  const int W = aten_block_width(N / 4, rows);
  const float sum = aten_row_sum(s + NT, N, W, s);
  // __fmul_rn / __fadd_rn: nvcc must not contract `sum * factor + eps` into
  // an FFMA (it does by default, and the single rounding moves rsqrt by an
  // ulp in ~10% of rows); ATen rounds after every op.
  const float factor = (float)rows / (float)((long long)rows * (long long)N);
  const float var = __fmul_rn(sum, factor);
  const float r = rsqrtf(__fadd_rn(var, eps));
#pragma unroll
  for (int k = 0; k < MAXC; k++) {
    const int c = threadIdx.x + k * NT;
    if (c < nchunk) {
      const float w[8] = {wv[k][0].x, wv[k][0].y, wv[k][0].z, wv[k][0].w,
                          wv[k][1].x, wv[k][1].y, wv[k][1].z, wv[k][1].w};
      float y[8];
#pragma unroll
      for (int e = 0; e < 8; e++) y[e] = __fmul_rn(__fmul_rn(zk[k][e], r), w[e]);
      *reinterpret_cast<uint4*>(out + c * 8) = pack8(y);
    }
  }
}

// Shared layout: z[N] (unused since the row rewrite, kept for the launch's
// smem size) | s[NT] | sq[N] floats -> dynamic shared 2*N*4 + NT*4.
extern "C" __global__ void kern_gemma_rms_norm_bf16(
    __nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ in,
    const float* __restrict__ w1, int N, int rows, int heads, int in_tstride,
    int in_hstride, int out_tstride, int out_hstride, float eps) {
    // Programmatic dependent launch: everything before this line may run
    // while the previous kernel is still finishing; nothing produced by it
    // is read or overwritten until the wait returns. The trigger right after
    // lets the next launch start its own prologue (a GEMM's weight stream).
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

  extern __shared__ float smem[];
  const int row = blockIdx.x;
  const int tok = row / heads, head = row % heads;
  gemma_norm_row<false>(
      out + (long long)tok * out_tstride + (long long)head * out_hstride,
      in + (long long)tok * in_tstride + (long long)head * in_hstride, nullptr,
      w1, N, rows, eps, smem, smem + N);
}

// Fused residual add: res[r] += x[r] (in f32, stored bf16), out = norm(sum).
// heads = 1 semantics (hidden norm); res is contiguous [rows, N].
extern "C" __global__ void kern_gemma_fused_add_rms_norm_bf16(
    __nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ in,
    __nv_bfloat16* __restrict__ res, const float* __restrict__ w1, int N,
    int rows, int in_tstride, int out_tstride, float eps) {
    // Programmatic dependent launch: everything before this line may run
    // while the previous kernel is still finishing; nothing produced by it
    // is read or overwritten until the wait returns. The trigger right after
    // lets the next launch start its own prologue (a GEMM's weight stream).
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

  extern __shared__ float smem[];
  const int row = blockIdx.x;
  gemma_norm_row<true>(out + (long long)row * out_tstride,
                       in + (long long)row * in_tstride,
                       res + (long long)row * N, w1, N, rows, eps, smem,
                       smem + N);
}
