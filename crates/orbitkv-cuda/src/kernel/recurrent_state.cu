@DYNAMIC_DEFINES@
extern "C" {
__global__ void delta_state_update(
    float* state,
    const float* decay,
    const float* key,
    const float* delta@DYNAMIC_PARAMETER@
) {
    const long long const_z =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (const_z >= @TOTAL@) return;

    const long long state_index = @STATE_INDEX@;
    const float decayed = __fmul_rn(state[state_index], decay[@DECAY_INDEX@]);
    const float update = __fmul_rn(key[@KEY_INDEX@], delta[@DELTA_INDEX@]);
    state[state_index] = __fadd_rn(decayed, update);
}
}
