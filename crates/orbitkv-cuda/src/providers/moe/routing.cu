
#include <cuda_bf16.h>

extern "C" __global__ void f32_to_bf16(unsigned long long in_ptr, unsigned long long out_ptr, int n) {
    const float* in_ = (const float*)in_ptr;
    __nv_bfloat16* out = (__nv_bfloat16*)out_ptr;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(in_[i]);
}

extern "C" __global__ void glu_activation_bf16(
    unsigned long long gate_up_ptr,
    unsigned long long out_ptr,
    int intermediate,
    int mode
) {
    const __nv_bfloat16* gate_up = (const __nv_bfloat16*)gate_up_ptr;
    __nv_bfloat16* out = (__nv_bfloat16*)out_ptr;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < intermediate) {
        float gate = __bfloat162float(gate_up[i]);
        float up   = __bfloat162float(gate_up[i + intermediate]);
        float activated;
        if (mode == 0) {
            activated = gate / (1.0f + expf(-gate));
        } else {
            float scaled = 1.5957691216f * gate * (1.0f + 0.044715f * gate * gate);
            activated = gate / (1.0f + expf(-scaled));
        }
        out[i] = __float2bfloat16(activated * up);
    }
}
