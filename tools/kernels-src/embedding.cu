// MVP embedding gather：manifest `embedding` 核的实现（唯一自己写的核）。
// ABI 与 manifest 声明一致：grid.x = tokens，block 256。
//   nvcc -cubin -arch=sm_103a -o kernels/embedding.cubin tools/kernels-src/embedding.cu
#include <cuda_bf16.h>

extern "C" __global__ void kern_embedding_i64_bf16(
    const long long* __restrict__ ids, const __nv_bfloat16* __restrict__ table,
    __nv_bfloat16* __restrict__ out, int tokens, int hidden) {
  long long row = ids[blockIdx.x];
  const __nv_bfloat16* src = table + row * hidden;
  __nv_bfloat16* dst = out + (long long)blockIdx.x * hidden;
  for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
    dst[j] = src[j];
  }
}
