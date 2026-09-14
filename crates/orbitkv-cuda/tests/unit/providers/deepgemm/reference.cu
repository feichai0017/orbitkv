#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math_constants.h>

@QUANTIZER@

extern "C" __global__ void block_scaled_reference(
    __nv_bfloat16* output, const __nv_fp8_e4m3* input, const float* input_scales,
    const __nv_fp8_e4m3* weight, const float* weight_scales, int m, int n, int k) {
    using namespace orbitkv_fp8;
    constexpr int kWarpThreads = 32;
    constexpr int kReferenceWarps = @REFERENCE_THREADS@ / kWarpThreads;
    __shared__ float warp_sums[kReferenceWarps];
    int row = blockIdx.y;
    int output_column = blockIdx.x;
    int aligned_m = aligned_scale_rows(m);
    int k_blocks = (k + kScaleBlock - 1) / kScaleBlock;
    float sum = 0.0f;
    for (int column = threadIdx.x; column < k; column += blockDim.x) {
        int k_block = column / kScaleBlock;
        float input_scale = input_scales[k_block * aligned_m + row];
        float weight_scale = weight_scales[(output_column / kScaleBlock) * k_blocks + k_block];
        sum += (float)input[(long long)row * k + column]
            * (float)weight[(long long)output_column * k + column]
            * input_scale * weight_scale;
    }
    for (int offset = kWarpThreads / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    int warp = threadIdx.x / kWarpThreads;
    if ((threadIdx.x % kWarpThreads) == 0) warp_sums[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = threadIdx.x < kReferenceWarps ? warp_sums[threadIdx.x] : 0.0f;
        for (int offset = kReferenceWarps / 2; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (threadIdx.x == 0) output[(long long)row * n + output_column] = (__nv_bfloat16)sum;
    }
}
