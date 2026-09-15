// Shared numerical and physical contract for the combined and graph-visible
// quantizers. FP8 is row-major [M,K]. FP32 scales are [K/128, align4(M)].
// K is a positive multiple of 128. Padded scale rows are initialized to zero
// so the complete packed result is deterministic, including small M.
extern "C" __global__ void block_scaled_quantize(
    __nv_fp8_e4m3* output, float* scales, const __nv_bfloat16* input, int m, int k) {
    using namespace orbitkv_fp8;
    static_assert(kQuantizerThreads == 32, "quantizer reduction requires one CUDA warp");
    static_assert(kScaleBlock % kQuantizerThreads == 0, "whole values per lane");
    constexpr int kValuesPerLane = kScaleBlock / kQuantizerThreads;
    int row = blockIdx.y;
    int k_block = blockIdx.x;
    long long base = (long long)row * k + k_block * kScaleBlock + threadIdx.x;
    float values[kValuesPerLane];
    float maximum = 0.0f;
    #pragma unroll
    for (int i = 0; i < kValuesPerLane; ++i) {
        values[i] = __bfloat162float(input[base + i * kQuantizerThreads]);
        maximum = fmaxf(maximum, fabsf(values[i]));
    }
    const unsigned mask = __activemask();
    #pragma unroll
    for (int offset = kQuantizerThreads / 2; offset > 0; offset >>= 1)
        maximum = fmaxf(maximum, __shfl_xor_sync(mask, maximum, offset));
    // These F32 roundings are part of the quantization contract. Replacing
    // either product with division changes FP8 codes near a midpoint.
    float scale = __fmul_rn(fmaxf(maximum, kQuantizationAmaxFloor), kFp8InverseMaxFinite);
    float inverse_scale = __frcp_rn(scale);
    int scale_rows = aligned_scale_rows(m);
    if (threadIdx.x == 0) scales[(long long)k_block * scale_rows + row] = scale;
    if (row == 0 && threadIdx.x < scale_rows - m)
        scales[(long long)k_block * scale_rows + m + threadIdx.x] = 0.0f;
    #pragma unroll
    for (int i = 0; i < kValuesPerLane; ++i)
        output[base + i * kQuantizerThreads] = (__nv_fp8_e4m3)(__fmul_rn(values[i], inverse_scale));
}
