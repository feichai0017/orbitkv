#include <cuda_bf16.h>
@DYNAMIC_DEFINES@
extern "C" __global__ void @ENTRY@(
    float* output, const float* query, const float* key, const float* value,
    const float* log_decay, const float* update_gate, @STATE_PARAMETERS@,
    const int* query_indptr@DYNAMIC_PARAMETER@
) {
    const long long request = blockIdx.x;
    const long long head = blockIdx.y;
    const long long column = static_cast<long long>(blockIdx.z) * blockDim.x + threadIdx.x;
    if (request >= @REQUESTS@ || head >= @VALUE_HEADS@) return;
    const int begin = query_indptr[request];
    const int end = query_indptr[request + 1];
    if (begin < 0 || end < begin || end > @TOKENS@) {
        asm volatile("trap;");
        return;
    }
    __shared__ float normalized_query[@KEY_WIDTH@];
    __shared__ float normalized_key[@KEY_WIDTH@];
    const long long base = (request * @VALUE_HEADS@ + head) * @KEY_WIDTH@ * @VALUE_WIDTH@;
    float state[@KEY_WIDTH@];
    #pragma unroll
    for (int width = 0; width < @KEY_WIDTH@; ++width) {
        if (column < @VALUE_WIDTH@) {
            const long long state_index = base + static_cast<long long>(width) * @VALUE_WIDTH@ + column;
            @STATE_LOAD@
        } else {
            state[width] = 0.0f;
        }
    }
    const long long key_head = head / (@VALUE_HEADS@ / @KEY_HEADS@);
    constexpr bool round_normalized_qk_to_bf16 = @ROUND_NORMALIZED_QK_TO_BF16@;
    for (int token = begin; token < end; ++token) {
        const long long qk_base = (static_cast<long long>(token) * @KEY_HEADS@ + key_head) * @KEY_WIDTH@;
        float query_quadrants[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int width = threadIdx.x; width < @KEY_WIDTH@; width += blockDim.x) {
            const float q = width < @KEY_WIDTH@ ? query[qk_base + width] : 0.0f;
            if (width < @KEY_WIDTH@) normalized_query[width] = q;
            const float square = round_normalized_qk_to_bf16
                ? __bfloat162float(__float2bfloat16_rn(q * q)) : q * q;
            query_quadrants[width / blockDim.x] = square;
        }
        // Match PackedDeltaScan's 256-thread reduction tree. For K <= 128,
        // its stride-64 and stride-32 stages combine quadrants in this exact
        // order before the remaining warp-local tree. Reassociating these
        // additions makes otherwise equivalent bucket implementations drift.
        float query_sum = __fadd_rn(
            __fadd_rn(query_quadrants[0], query_quadrants[2]),
            __fadd_rn(query_quadrants[1], query_quadrants[3]));
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            query_sum = __fadd_rn(query_sum, __shfl_down_sync(0xffffffff, query_sum, offset));
        query_sum = __shfl_sync(0xffffffff, query_sum, 0);
        const float query_scale = rsqrtf(static_cast<float>(@KEY_WIDTH@));
        const float query_inverse = round_normalized_qk_to_bf16
            ? __bfloat162float(__float2bfloat16_rn(rsqrtf(__bfloat162float(
                __float2bfloat16_rn(__bfloat162float(__float2bfloat16_rn(query_sum))
                    + @NORMALIZATION_EPSILON@)))))
            : rsqrtf(query_sum + @NORMALIZATION_EPSILON@) * query_scale;
        float key_quadrants[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int width = threadIdx.x; width < @KEY_WIDTH@; width += blockDim.x) {
            const float k = width < @KEY_WIDTH@ ? key[qk_base + width] : 0.0f;
            if (width < @KEY_WIDTH@) normalized_key[width] = k;
            const float square = round_normalized_qk_to_bf16
                ? __bfloat162float(__float2bfloat16_rn(k * k)) : k * k;
            key_quadrants[width / blockDim.x] = square;
        }
        float key_sum = __fadd_rn(
            __fadd_rn(key_quadrants[0], key_quadrants[2]),
            __fadd_rn(key_quadrants[1], key_quadrants[3]));
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            key_sum = __fadd_rn(key_sum, __shfl_down_sync(0xffffffff, key_sum, offset));
        key_sum = __shfl_sync(0xffffffff, key_sum, 0);
        const float key_inverse = round_normalized_qk_to_bf16
            ? __bfloat162float(__float2bfloat16_rn(rsqrtf(__bfloat162float(
                __float2bfloat16_rn(__bfloat162float(__float2bfloat16_rn(key_sum))
                    + @NORMALIZATION_EPSILON@)))))
            : rsqrtf(key_sum + @NORMALIZATION_EPSILON@);
        for (int width = threadIdx.x; width < @KEY_WIDTH@; width += blockDim.x) {
            if constexpr (round_normalized_qk_to_bf16) {
                normalized_query[width] = __bfloat162float(
                    __float2bfloat16_rn(normalized_query[width] * query_inverse)) * query_scale;
                normalized_key[width] = __bfloat162float(
                    __float2bfloat16_rn(normalized_key[width] * key_inverse));
            } else {
                normalized_query[width] *= query_inverse;
                normalized_key[width] *= key_inverse;
            }
        }
        __syncthreads();
        const long long gate_index = static_cast<long long>(token) * @VALUE_HEADS@ + head;
        const float decay = expf(log_decay[gate_index]);
        const float beta = update_gate[gate_index];
        if (column < @VALUE_WIDTH@) {
            float memory = 0.0f;
            #pragma unroll
            for (int width = 0; width < @KEY_WIDTH@; ++width) {
                state[width] *= decay;
                memory = round_normalized_qk_to_bf16
                    ? __fadd_rn(memory, __fmul_rn(state[width], normalized_key[width]))
                    : fmaf(state[width], normalized_key[width], memory);
            }
            const float delta = round_normalized_qk_to_bf16
                ? __fmul_rn(__fsub_rn(value[gate_index * @VALUE_WIDTH@ + column], memory), beta)
                : (value[gate_index * @VALUE_WIDTH@ + column] - memory) * beta;
            float result = 0.0f;
            #pragma unroll
            for (int width = 0; width < @KEY_WIDTH@; ++width) {
                state[width] = round_normalized_qk_to_bf16
                    ? __fadd_rn(__fmul_rn(normalized_key[width], delta), state[width])
                    : fmaf(normalized_key[width], delta, state[width]);
                result = round_normalized_qk_to_bf16
                    ? __fadd_rn(result, __fmul_rn(state[width], normalized_query[width]))
                    : fmaf(state[width], normalized_query[width], result);
            }
            output[gate_index * @VALUE_WIDTH@ + column] = result;
        }
        __syncthreads();
    }
    if (column < @VALUE_WIDTH@) {
        #pragma unroll
        for (int width = 0; width < @KEY_WIDTH@; ++width)
            output[@TOKEN_VALUE_ELEMENTS@ + base + static_cast<long long>(width) * @VALUE_WIDTH@ + column] =
                @ROUND_FINAL_STATE_TO_BF16@
                    ? __bfloat162float(__float2bfloat16_rn(state[width]))
                    : state[width];
    }
}
