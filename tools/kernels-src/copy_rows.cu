// Plain bf16 row copies (the ATen `contiguous()` / row-select copies that
// vLLM does between kernels and that the manifest cannot express with a
// byte offset alone).
//
//   kern_copy_rows_bf16: dst[r, :width] = src[r, :width], grid.x = rows
//     (strided view -> contiguous, e.g. the GDN z gate out of the fused
//     qkvz projection)
//   kern_last_row_bf16:  dst[0, :width] = src[rows-1, :width], grid 1
//     (the final-norm / lm_head row of a prefill chunk; `rows-1` is not in
//     the manifest expression set, so the kernel takes `rows`)
//
//   nvcc -cubin -arch=sm_103a -o kernels/copy_rows.cubin tools/kernels-src/copy_rows.cu
#include <cuda_bf16.h>

extern "C" __global__ void kern_copy_rows_bf16(
    __nv_bfloat16* __restrict__ dst, const __nv_bfloat16* __restrict__ src,
    int dst_stride, int src_stride, int width) {
  const long long r = blockIdx.x;
  const __nv_bfloat16* s = src + r * src_stride;
  __nv_bfloat16* d = dst + r * dst_stride;
  for (int j = threadIdx.x; j < width; j += blockDim.x) d[j] = s[j];
}

extern "C" __global__ void kern_last_row_bf16(
    __nv_bfloat16* __restrict__ dst, const __nv_bfloat16* __restrict__ src,
    int src_stride, int width, int rows) {
  const __nv_bfloat16* s = src + (long long)(rows - 1) * src_stride;
  for (int j = threadIdx.x; j < width; j += blockDim.x) dst[j] = s[j];
}
