// Included into a versioned DeepGEMM wrapper. The constants and layout are
// named by @NUMERICAL_ABI@ in the rendered wrapper.
namespace orbitkv_fp8 {
constexpr int kScaleBlock = 128;
constexpr int kQuantizerThreads = 32;
constexpr int kScaleRowAlignment = 4;
constexpr float kFp8MaxFinite = 448.0f;
constexpr float kFp8InverseMaxFinite = 0.00223214296f;
constexpr float kQuantizationAmaxFloor = 1.0e-4f;
__host__ __device__ constexpr int aligned_scale_rows(int rows) {
    return (rows + kScaleRowAlignment - 1) / kScaleRowAlignment * kScaleRowAlignment;
}
}

extern "C" __global__ void block_scaled_quantize(
    __nv_fp8_e4m3* output, float* scales, const __nv_bfloat16* input, int m, int k) {
    using namespace orbitkv_fp8;
    static_assert(kQuantizerThreads == 32);
    constexpr int kValuesPerLane = kScaleBlock / kQuantizerThreads;
    const int row = blockIdx.y;
    const int k_block = blockIdx.x;
    const long long base = static_cast<long long>(row) * k + k_block * kScaleBlock + threadIdx.x;
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
    const float scale = __fmul_rn(fmaxf(maximum, kQuantizationAmaxFloor), kFp8InverseMaxFinite);
    const float inverse_scale = __frcp_rn(scale);
    const int scale_rows = aligned_scale_rows(m);
    if (threadIdx.x == 0)
        scales[static_cast<long long>(k_block) * scale_rows + row] = scale;
    if (row == 0 && threadIdx.x < scale_rows - m)
        scales[static_cast<long long>(k_block) * scale_rows + m + threadIdx.x] = 0.0f;
#pragma unroll
    for (int i = 0; i < kValuesPerLane; ++i)
        output[base + i * kQuantizerThreads] =
            static_cast<__nv_fp8_e4m3>(__fmul_rn(values[i], inverse_scale));
}
