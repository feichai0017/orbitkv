
extern "C" __global__ void transfer_blocks(const unsigned long long* __restrict__ desc,
                                           int n, int ctas) {
    const int tid = threadIdx.x;
    const long long work_count = (long long)n * ctas;
    for (long long work = blockIdx.x; work < work_count; work += gridDim.x) {
        const int i = work / ctas;
        const int shard = work % ctas;
        char* dst = (char*)desc[3 * i];
        const char* src = (const char*)desc[3 * i + 1];
        const unsigned long long size = desc[3 * i + 2];
        const unsigned long long start = (unsigned long long)shard * blockDim.x + tid;
        const unsigned long long stride = (unsigned long long)blockDim.x * ctas;
        if ((((unsigned long long)dst | (unsigned long long)src) & 15ULL) == 0) {
            int4* dst4 = (int4*)dst;
            const int4* src4 = (const int4*)src;
            const unsigned long long n4 = size / 16;
            for (unsigned long long j = start; j < n4; j += stride) {
                dst4[j] = src4[j];
            }
            for (unsigned long long j = n4 * 16 + start; j < size; j += stride) {
                dst[j] = src[j];
            }
        } else {
            for (unsigned long long j = start; j < size; j += stride) {
                dst[j] = src[j];
            }
        }
    }
}
