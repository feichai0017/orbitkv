#include <cuda_bf16.h>
@DYNAMIC_DEFINES@

extern "C" __global__ void packed_causal_convolution(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    const __nv_bfloat16* weights,
    const __nv_bfloat16* initial_history,
    const int* query_indptr@DYNAMIC_PARAMETER@
) {
    const long long request = blockIdx.x;
    const long long channel =
        static_cast<long long>(blockIdx.y) * blockDim.x + threadIdx.x;
    if (request >= @REQUESTS@ || channel >= @CHANNELS@) return;
    const int begin = query_indptr[request];
    const int end = query_indptr[request + 1];
    if (begin < 0 || end < begin || end > @TOKENS@) {
        asm volatile("trap;");
        return;
    }

    const long long history_width = @HISTORY_WIDTH@;
    const long long history_base = @TOKENS@ * @CHANNELS@
        + (request * @CHANNELS@ + channel) * history_width;
    const long long initial_base =
        (request * @CHANNELS@ + channel) * history_width;
    for (long long offset = 0; offset < history_width; ++offset) {
        output[history_base + offset] = initial_history[initial_base + offset];
    }

    for (int token = begin; token < end; ++token) {
        float value = 0.0f;
        const long long weight_base = channel * @KERNEL_WIDTH@;
        for (long long offset = 0; offset < history_width; ++offset) {
            value = fmaf(
                __bfloat162float(output[history_base + offset]),
                __bfloat162float(weights[weight_base + offset]),
                value
            );
        }
        value = fmaf(
            __bfloat162float(input[static_cast<long long>(token) * @CHANNELS@ + channel]),
            __bfloat162float(weights[weight_base + history_width]),
            value
        );
        // Convolution materializes BF16 before SiLU; fusion must retain that rounding.
        value = __bfloat162float(__float2bfloat16_rn(value));
        output[static_cast<long long>(token) * @CHANNELS@ + channel] =
            __float2bfloat16(value / (1.0f + expf(-value)));
        for (long long offset = 0; offset + 1 < history_width; ++offset) {
            output[history_base + offset] = output[history_base + offset + 1];
        }
        output[history_base + history_width - 1] =
            input[static_cast<long long>(token) * @CHANNELS@ + channel];
    }
}
