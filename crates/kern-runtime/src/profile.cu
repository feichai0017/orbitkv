// Calibration and cache preparation only; never part of a model program.
extern "C" __global__ void stream_copy(const ulonglong2* src, ulonglong2* dst, unsigned long long n) {
    for (unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (unsigned long long)gridDim.x * blockDim.x) dst[i] = src[i];
}
extern "C" __global__ void stream_read(const ulonglong2* src, unsigned long long* dst, unsigned long long n) {
    unsigned long long v = 0;
    for (unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (unsigned long long)gridDim.x * blockDim.x) {
        ulonglong2 x = src[i]; v ^= x.x ^ x.y;
    }
    dst[blockIdx.x * blockDim.x + threadIdx.x] = v;
}
extern "C" __global__ void seed_data(ulonglong2* dst, unsigned long long n) {
    for (unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (unsigned long long)gridDim.x * blockDim.x) {
        unsigned long long v = (i + 1) * 0x9e3779b97f4a7c15ull;
        dst[i] = make_ulonglong2(v ^ (v >> 29), v * 0xbf58476d1ce4e5b9ull);
    }
}
extern "C" __global__ void empty_probe() {}
// Activity-trace delimiters. Their durations are excluded from the span of
// the enclosed model kernels; they carry no model data and touch no cache.
extern "C" __global__ void profile_cold_start() {}
extern "C" __global__ void profile_warm_start() {}
extern "C" __global__ void profile_program_start() {}
extern "C" __global__ void profile_anchor_start() {}
extern "C" __global__ void profile_end() {}
