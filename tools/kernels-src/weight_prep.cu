// What a checkpoint does not contain, computed once after load by the
// manifest's `once` program: the derived tables the captured kernels read
// as pointers. Every kernel here is elementwise over a constant-shaped
// buffer; the `once` program's grids are literals.
//
//   kern_weight_p1_bf16_f32: out[i] = float(w[i]) + 1.0f — the tensor
//     vLLM's GemmaRMSNorm feeds ATen (`weight.float() + 1`), bit-exact.
//   kern_cast_bf16_f32:      out[i] = float(w[i]) (GDN's A_log, kept in
//     f32 by vLLM and read as such by the Triton kernels).
//   kern_rope_table_bf16:    cos[r*stride + j] / sin[...] for r < rows,
//     j < half, as vLLM's RotaryEmbedding builds its cache in f32 on the
//     device: inv = 1 / base^(2j / (2 half)), angle = r * inv, then bf16.
//     One buffer with two halves (cos | sin, stride = 2 half) or two
//     buffers (stride = half) are both a call away. grid.x = rows, block
//     >= half.
//   kern_fill_{f32,i32,i64,u8}: p[i] = v — the ones / zeros the kernels
//     want as pointers (kv scales, offsets, flags).
//   kern_iota_i32:           p[i*stride] = start + i*step — chunk and
//     program index tables.
//
//   nvcc -cubin -arch=sm_103a -o kernels/weight_prep.cubin tools/kernels-src/weight_prep.cu
#include <cuda_bf16.h>

extern "C" __global__ void kern_weight_p1_bf16_f32(const __nv_bfloat16* __restrict__ w, float* __restrict__ out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __bfloat162float(w[i]) + 1.0f;
}

extern "C" __global__ void kern_cast_bf16_f32(const __nv_bfloat16* __restrict__ w, float* __restrict__ out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __bfloat162float(w[i]);
}

extern "C" __global__ void kern_rope_table_bf16(__nv_bfloat16* __restrict__ cos_t, __nv_bfloat16* __restrict__ sin_t,
                                                int rows, int half, int stride, float base) {
  // one block per row, a thread per column
  const int j = threadIdx.x;
  const int r = blockIdx.x;
  if (j >= half || r >= rows) return;
  // torch: arange(0, rot, 2) / rot in f32, base ** that, 1 / it, t * inv.
  const float inv = 1.0f / powf(base, (float)(2 * j) / (float)(2 * half));
  const float angle = (float)r * inv;
  cos_t[(long long)r * stride + j] = __float2bfloat16(cosf(angle));
  sin_t[(long long)r * stride + j] = __float2bfloat16(sinf(angle));
}

extern "C" __global__ void kern_fill_f32(float* __restrict__ p, int n, float v) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = v;
}

extern "C" __global__ void kern_fill_i32(int* __restrict__ p, int n, int v) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = v;
}

extern "C" __global__ void kern_fill_i64(long long* __restrict__ p, int n, long long v) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = v;
}

extern "C" __global__ void kern_fill_u8(unsigned char* __restrict__ p, int n, int v) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = (unsigned char)v;
}

extern "C" __global__ void kern_iota_i32(int* __restrict__ p, int n, int start, int step, int stride) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[(long long)i * stride] = start + i * step;
}
