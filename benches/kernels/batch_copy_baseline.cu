
extern "C" __global__ void orbitkv_batch_copy(const unsigned long long* __restrict__ descs, int n) {
    for (int i = blockIdx.x; i < n; i += gridDim.x) {
        char* dst = (char*)descs[3 * i + 0];
        const char* src = (const char*)descs[3 * i + 1];
        unsigned long long size = descs[3 * i + 2];
        if (((((unsigned long long)dst) | ((unsigned long long)src)) & 15ULL) == 0ULL) {
            unsigned long long n16 = size >> 4;
            for (unsigned long long j = threadIdx.x; j < n16; j += blockDim.x) {
                ((int4*)dst)[j] = ((const int4*)src)[j];
            }
            for (unsigned long long j = (n16 << 4) + threadIdx.x; j < size; j += blockDim.x) {
                dst[j] = src[j];
            }
        } else {
            for (unsigned long long j = threadIdx.x; j < size; j += blockDim.x) {
                dst[j] = src[j];
            }
        }
    }
}
