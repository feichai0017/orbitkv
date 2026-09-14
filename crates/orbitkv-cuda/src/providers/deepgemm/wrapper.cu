#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdio>

#include <deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh>

using namespace deep_gemm;

// Provider identity is deliberately embedded in the compiled image and its
// cache key. A selected OrbitKV schedule therefore names reproducible upstream
// kernel semantics rather than an unversioned runtime library.
static constexpr char kDeepGemmRevision[] = "@PROVIDER_REVISION@";
static thread_local char last_error[512] = {};

@QUANTIZER@

// Fixed SM90 1D2D kernel template requirements, independent of matrix shape.
static constexpr int kOperandSwizzleBytes = 128;
static constexpr int kTmaLoadThreads = 128;

using GemmKernel = decltype(&sm90_fp8_gemm_1d2d_impl<
    cute::UMMA::Major::K,
    0, @N@, @K@,
    1,
    @BLOCK_M@, @BLOCK_N@, @BLOCK_K@,
    kOperandSwizzleBytes, kOperandSwizzleBytes, @SWIZZLE_D@,
    @STAGES@,
    kTmaLoadThreads, @MATH_THREADS@,
    @CLUSTER_SIZE@, @MULTICAST_ON_A@,
    @NUM_SMS@, GemmType::Normal,
    cutlass::bfloat16_t,
    epilogue::transform::EpilogueIdentity>);

static GemmKernel gemm_kernel() {
    return &sm90_fp8_gemm_1d2d_impl<
        cute::UMMA::Major::K,
        0, @N@, @K@,
        1,
        @BLOCK_M@, @BLOCK_N@, @BLOCK_K@,
        kOperandSwizzleBytes, kOperandSwizzleBytes, @SWIZZLE_D@,
        @STAGES@,
        kTmaLoadThreads, @MATH_THREADS@,
        @CLUSTER_SIZE@, @MULTICAST_ON_A@,
        @NUM_SMS@, GemmType::Normal,
        cutlass::bfloat16_t,
        epilogue::transform::EpilogueIdentity>;
}

static CUresult make_tma(
    CUtensorMap* map, CUtensorMapDataType type, void* pointer,
    uint64_t inner, uint64_t outer, uint64_t outer_stride_bytes,
    uint32_t box_inner, uint32_t box_outer, CUtensorMapSwizzle swizzle) {
    uint64_t dimensions[2] = {inner, outer};
    uint64_t strides[1] = {outer_stride_bytes};
    uint32_t box[2] = {box_inner, box_outer};
    uint32_t element_strides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        map, type, 2, pointer, dimensions, strides, box, element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

static int fail(const char* call, int status) {
    std::snprintf(last_error, sizeof(last_error), "%s failed with CUDA status %d", call, status);
    return status == 0 ? -1 : status;
}

extern "C" const char* orbitkv_deepgemm_last_error() {
    return last_error;
}

extern "C" int orbitkv_deepgemm_run_prequantized(
    const void* weight, const void* weight_scale,
    void* quantized, void* activation_scale, void* output, int m, void* raw_stream) {
    last_error[0] = '\0';
    auto stream = reinterpret_cast<cudaStream_t>(raw_stream);
    CUtensorMap tensor_map_a{}, tensor_map_b{}, tensor_map_d{}, tensor_map_sfa{};
    if (auto status = make_tma(&tensor_map_a, CU_TENSOR_MAP_DATA_TYPE_UINT8, quantized,
                               @K@, m, @K@, kOperandSwizzleBytes, @BLOCK_M@, CU_TENSOR_MAP_SWIZZLE_128B); status != CUDA_SUCCESS)
        return fail("A TMA descriptor", static_cast<int>(status));
    if (auto status = make_tma(&tensor_map_b, CU_TENSOR_MAP_DATA_TYPE_UINT8, const_cast<void*>(weight),
                               @K@, @N@, @K@, kOperandSwizzleBytes, @BLOCK_N@, CU_TENSOR_MAP_SWIZZLE_128B); status != CUDA_SUCCESS)
        return fail("B TMA descriptor", static_cast<int>(status));
    if (auto status = make_tma(&tensor_map_d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, output,
                               @N@, m, @N@ * sizeof(__nv_bfloat16), @SWIZZLE_D@ / sizeof(__nv_bfloat16), @BLOCK_M@,
                               CU_TENSOR_MAP_SWIZZLE_@SWIZZLE_D@B); status != CUDA_SUCCESS)
        return fail("D TMA descriptor", static_cast<int>(status));
    if (auto status = make_tma(&tensor_map_sfa, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, activation_scale,
                               orbitkv_fp8::aligned_scale_rows(m), @K@ / orbitkv_fp8::kScaleBlock,
                               orbitkv_fp8::aligned_scale_rows(m) * sizeof(float),
                               @BLOCK_M@, 1, CU_TENSOR_MAP_SWIZZLE_NONE); status != CUDA_SUCCESS)
        return fail("A-scale TMA descriptor", static_cast<int>(status));

    auto kernel = gemm_kernel();
    if (auto status = cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            @SMEM_BYTES@); status != cudaSuccess)
        return fail("cudaFuncSetAttribute", static_cast<int>(status));

    void* sfb = const_cast<void*>(weight_scale);
    int* grouped_layout = nullptr;
    unsigned rows = static_cast<unsigned>(m), n = @N@, k = @K@;
    void* arguments[] = {
        &sfb, &grouped_layout, &rows, &n, &k,
        &tensor_map_a, &tensor_map_b, &tensor_map_d, &tensor_map_sfa,
    };
    cudaLaunchAttribute attributes[2]{};
    unsigned attribute_count = 0;
    if constexpr (@CLUSTER_M@ * @CLUSTER_N@ > 1) {
        attributes[attribute_count].id = cudaLaunchAttributeClusterDimension;
        attributes[attribute_count].val.clusterDim = {@CLUSTER_M@ * @CLUSTER_N@, 1, 1};
        ++attribute_count;
    }
    attributes[attribute_count].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attributes[attribute_count].val.programmaticStreamSerializationAllowed = 1;
    ++attribute_count;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(@NUM_SMS@, 1, 1);
    config.blockDim = dim3(kTmaLoadThreads + @MATH_THREADS@, 1, 1);
    config.dynamicSmemBytes = @SMEM_BYTES@;
    config.stream = stream;
    config.attrs = attributes;
    config.numAttrs = attribute_count;
    if (auto status = cudaLaunchKernelExC(&config, reinterpret_cast<void*>(kernel), arguments);
        status != cudaSuccess)
        return fail("DeepGEMM launch", static_cast<int>(status));
    (void)kDeepGemmRevision;
    return 0;
}

extern "C" int orbitkv_deepgemm_run(
    const void* input, const void* weight, const void* weight_scale,
    void* quantized, void* activation_scale, void* output, int m, void* raw_stream) {
    last_error[0] = '\0';
    auto stream = reinterpret_cast<cudaStream_t>(raw_stream);
    block_scaled_quantize<<<dim3(@K@ / orbitkv_fp8::kScaleBlock, m, 1), orbitkv_fp8::kScaleBlock, 0, stream>>>(
        reinterpret_cast<__nv_fp8_e4m3*>(quantized),
        reinterpret_cast<float*>(activation_scale),
        reinterpret_cast<const __nv_bfloat16*>(input), m, @K@);
    if (auto status = cudaGetLastError(); status != cudaSuccess)
        return fail("activation quantization launch", static_cast<int>(status));
    return orbitkv_deepgemm_run_prequantized(
        weight, weight_scale, quantized, activation_scale, output, m, raw_stream);
}
