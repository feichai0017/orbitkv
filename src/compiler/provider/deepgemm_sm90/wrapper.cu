#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdio>

#include <deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh>

using namespace deep_gemm;

extern "C" __attribute__((used, visibility("default")))
const char orbitkv_deepgemm_revision[] = "@PROVIDER_REVISION@";
extern "C" __attribute__((used, visibility("default")))
const char orbitkv_deepgemm_numerical_abi[] = "@NUMERICAL_ABI@";

@QUANTIZER@

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

static thread_local char orbitkv_deepgemm_error[512] = {};

static GemmKernel orbitkv_deepgemm_kernel() {
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

static CUresult orbitkv_make_tma(
    CUtensorMap* map, CUtensorMapDataType dtype, void* pointer,
    uint64_t inner, uint64_t outer, uint64_t outer_stride_bytes,
    uint32_t box_inner, uint32_t box_outer, CUtensorMapSwizzle swizzle) {
    uint64_t dimensions[2] = {inner, outer};
    uint64_t strides[1] = {outer_stride_bytes};
    uint32_t box[2] = {box_inner, box_outer};
    uint32_t element_strides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        map, dtype, 2, pointer, dimensions, strides, box, element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

static int orbitkv_fail(const char* call, int status) {
    std::snprintf(orbitkv_deepgemm_error, sizeof(orbitkv_deepgemm_error),
                  "%s failed with CUDA status %d", call, status);
    return status == 0 ? -1 : status;
}

extern "C" const char* orbitkv_deepgemm_last_error() {
    return orbitkv_deepgemm_error;
}

extern "C" int orbitkv_deepgemm_run_prequantized(
    const void* weight, const void* weight_scale, void* quantized,
    void* activation_scale, void* output, int rows, void* raw_stream) {
    orbitkv_deepgemm_error[0] = '\0';
    if (rows <= 0 || rows > @ROW_LIMIT@)
        return orbitkv_fail("row limit", rows);
    auto stream = reinterpret_cast<cudaStream_t>(raw_stream);
    CUtensorMap tensor_map_a{}, tensor_map_b{}, tensor_map_d{}, tensor_map_sfa{};
    if (auto status = orbitkv_make_tma(
            &tensor_map_a, CU_TENSOR_MAP_DATA_TYPE_UINT8, quantized, @K@, rows, @K@,
            kOperandSwizzleBytes, @BLOCK_M@, CU_TENSOR_MAP_SWIZZLE_128B); status != CUDA_SUCCESS)
        return orbitkv_fail("A TMA descriptor", static_cast<int>(status));
    if (auto status = orbitkv_make_tma(
            &tensor_map_b, CU_TENSOR_MAP_DATA_TYPE_UINT8, const_cast<void*>(weight), @K@, @N@, @K@,
            kOperandSwizzleBytes, @BLOCK_N@, CU_TENSOR_MAP_SWIZZLE_128B); status != CUDA_SUCCESS)
        return orbitkv_fail("B TMA descriptor", static_cast<int>(status));
    if (auto status = orbitkv_make_tma(
            &tensor_map_d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, output, @N@, rows,
            @N@ * sizeof(__nv_bfloat16), @SWIZZLE_D@ / sizeof(__nv_bfloat16), @BLOCK_M@,
            CU_TENSOR_MAP_SWIZZLE_@SWIZZLE_D@B); status != CUDA_SUCCESS)
        return orbitkv_fail("D TMA descriptor", static_cast<int>(status));
    if (auto status = orbitkv_make_tma(
            &tensor_map_sfa, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, activation_scale,
            orbitkv_fp8::aligned_scale_rows(rows), @K@ / orbitkv_fp8::kScaleBlock,
            orbitkv_fp8::aligned_scale_rows(rows) * sizeof(float), @BLOCK_M@, 1,
            CU_TENSOR_MAP_SWIZZLE_NONE); status != CUDA_SUCCESS)
        return orbitkv_fail("A-scale TMA descriptor", static_cast<int>(status));

    auto kernel = orbitkv_deepgemm_kernel();
    if (auto status = cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, @SMEM_BYTES@); status != cudaSuccess)
        return orbitkv_fail("cudaFuncSetAttribute", static_cast<int>(status));

    void* sfb = const_cast<void*>(weight_scale);
    int* grouped_layout = nullptr;
    unsigned m = static_cast<unsigned>(rows), n = @N@, k = @K@;
    void* arguments[] = {
        &sfb, &grouped_layout, &m, &n, &k,
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
        return orbitkv_fail("DeepGEMM launch", static_cast<int>(status));
    return 0;
}

extern "C" int orbitkv_deepgemm_run(
    const void* input, const void* weight, const void* weight_scale,
    void* quantized, void* activation_scale, void* output, int rows, void* raw_stream) {
    orbitkv_deepgemm_error[0] = '\0';
    auto stream = reinterpret_cast<cudaStream_t>(raw_stream);
    block_scaled_quantize<<<dim3(@K@ / orbitkv_fp8::kScaleBlock, rows, 1),
                              orbitkv_fp8::kQuantizerThreads, 0, stream>>>(
        reinterpret_cast<__nv_fp8_e4m3*>(quantized),
        reinterpret_cast<float*>(activation_scale),
        reinterpret_cast<const __nv_bfloat16*>(input), rows, @K@);
    if (auto status = cudaGetLastError(); status != cudaSuccess)
        return orbitkv_fail("activation quantization launch", static_cast<int>(status));
    return orbitkv_deepgemm_run_prequantized(
        weight, weight_scale, quantized, activation_scale, output, rows, raw_stream);
}
