// Frozen OrbitKV quantizer from cb15ff241b5c, solely for bit parity and timing.
// Shared numerical and physical contract for the combined and graph-visible
// quantizers. FP8 is row-major [M,K]. FP32 scales are [K/128, align4(M)].
// K is a positive multiple of 128. Padded scale rows are initialized to zero
// so the complete packed result is deterministic, including small M.
extern "C" __global__ void previous_block_scaled_quantize(
    __nv_fp8_e4m3* output, float* scales, const __nv_bfloat16* input, int m, int k) {
    using namespace orbitkv_fp8;
    __shared__ float maxima[kScaleBlock];
    int row = blockIdx.y;
    int k_block = blockIdx.x;
    int column = k_block * kScaleBlock + threadIdx.x;
    float value = __bfloat162float(input[(long long)row * k + column]);
    maxima[threadIdx.x] = fabsf(value);
    __syncthreads();
    for (int offset = kScaleBlock / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) maxima[threadIdx.x] = fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + offset]);
        __syncthreads();
    }
    float scale = fmaxf(maxima[0], kQuantizationAmaxFloor) / kFp8MaxFinite;
    int scale_rows = aligned_scale_rows(m);
    if (threadIdx.x == 0) scales[(long long)k_block * scale_rows + row] = scale;
    if (row == 0 && threadIdx.x < scale_rows - m)
        scales[(long long)k_block * scale_rows + m + threadIdx.x] = 0.0f;
    output[(long long)row * k + column] = (__nv_fp8_e4m3)(value / scale);
}
