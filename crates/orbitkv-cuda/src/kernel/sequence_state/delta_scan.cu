@DYNAMIC_DEFINES@
extern "C" __global__ void packed_delta_scan(
    float* output,
    const float* query,
    const float* key,
    const float* value,
    const float* log_decay,
    const float* update_gate,
    const float* initial_state,
    const int* query_indptr@DYNAMIC_PARAMETER@
) {
    const long long request = blockIdx.x;
    const long long value_head = blockIdx.y;
    if (request >= @REQUESTS@ || value_head >= @VALUE_HEADS@) return;
    const int begin = query_indptr[request];
    const int end = query_indptr[request + 1];
    if (begin < 0 || end < begin || end > @TOKENS@) {
        asm volatile("trap;");
        return;
    }

    extern __shared__ float shared[];
    float* normalized_query = shared;
    float* normalized_key = normalized_query + @KEY_WIDTH@;
    float* reduction = normalized_key + @KEY_WIDTH@;
    const long long value_head_width = @KEY_WIDTH@ * @VALUE_WIDTH@;
    const long long state_base = @TOKEN_VALUE_ELEMENTS@
        + (request * @VALUE_HEADS@ + value_head) * value_head_width;
    const long long initial_base =
        (request * @VALUE_HEADS@ + value_head) * value_head_width;
    for (long long index = threadIdx.x; index < value_head_width; index += blockDim.x) {
        output[state_base + index] = initial_state[initial_base + index];
    }
    __syncthreads();

    const long long group_size = @VALUE_HEADS@ / @KEY_HEADS@;
    const long long key_head = value_head / group_size;
    for (int token = begin; token < end; ++token) {
        const long long qk_base =
            (static_cast<long long>(token) * @KEY_HEADS@ + key_head) * @KEY_WIDTH@;
        float query_sum = 0.0f;
        float key_sum = 0.0f;
        for (long long width = threadIdx.x; width < @KEY_WIDTH@; width += blockDim.x) {
            const float q = query[qk_base + width];
            const float k = key[qk_base + width];
            normalized_query[width] = q;
            normalized_key[width] = k;
            query_sum = fmaf(q, q, query_sum);
            key_sum = fmaf(k, k, key_sum);
        }
        reduction[threadIdx.x] = query_sum;
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
            __syncthreads();
        }
        const float query_inverse = rsqrtf(reduction[0] + @NORMALIZATION_EPSILON@)
            * rsqrtf(static_cast<float>(@KEY_WIDTH@));
        __syncthreads();
        reduction[threadIdx.x] = key_sum;
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
            __syncthreads();
        }
        const float key_inverse = rsqrtf(reduction[0] + @NORMALIZATION_EPSILON@);
        for (long long width = threadIdx.x; width < @KEY_WIDTH@; width += blockDim.x) {
            normalized_query[width] *= query_inverse;
            normalized_key[width] *= key_inverse;
        }
        __syncthreads();

        const long long gate_index =
            static_cast<long long>(token) * @VALUE_HEADS@ + value_head;
        const float decay = expf(log_decay[gate_index]);
        const float beta = update_gate[gate_index];
        const long long value_base = gate_index * @VALUE_WIDTH@;
        for (long long value_index = threadIdx.x; value_index < @VALUE_WIDTH@;
             value_index += blockDim.x) {
            float memory = 0.0f;
            for (long long width = 0; width < @KEY_WIDTH@; ++width) {
                const long long state_index = state_base + width * @VALUE_WIDTH@ + value_index;
                const float decayed = output[state_index] * decay;
                output[state_index] = decayed;
                memory = fmaf(decayed, normalized_key[width], memory);
            }
            const float delta = (value[value_base + value_index] - memory) * beta;
            float token_output = 0.0f;
            for (long long width = 0; width < @KEY_WIDTH@; ++width) {
                const long long state_index = state_base + width * @VALUE_WIDTH@ + value_index;
                const float next = fmaf(normalized_key[width], delta, output[state_index]);
                output[state_index] = next;
                token_output = fmaf(next, normalized_query[width], token_output);
            }
            output[value_base + value_index] = token_output;
        }
        __syncthreads();
    }
}
