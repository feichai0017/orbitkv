// Device-side comparison for the harness (`kern test`). The host functions
// in kern-test (`compare`, `changed_blocks`, `logit_row`) are the
// definitions; every count and maximum here must agree with them exactly,
// only the floating-point sums (log-sum-exp, KL) may differ in rounding.
// Never part of a model program.

enum Kind { BF16 = 0, F16 = 1, F32 = 2, E4M3 = 3, E8M0 = 4, I8 = 5, U8 = 6, I32 = 7, U32 = 8, I64 = 9, U64 = 10 };

__device__ __forceinline__ int width(int kind) {
    switch (kind) {
        case BF16: case F16: return 2;
        case F32: case I32: case U32: return 4;
        case I64: case U64: return 8;
        default: return 1;
    }
}

// kern-test measures ulps on the four float formats with a mantissa; E8M0 is an exponent byte.
__device__ __forceinline__ bool is_float(int kind) { return kind <= E4M3; }

__device__ __forceinline__ unsigned short load16(const unsigned char* p) { return p[0] | (p[1] << 8); }
__device__ __forceinline__ unsigned int load32(const unsigned char* p) {
    return p[0] | (p[1] << 8) | (p[2] << 16) | ((unsigned int)p[3] << 24);
}
__device__ __forceinline__ unsigned long long load64(const unsigned char* p) {
    return load32(p) | ((unsigned long long)load32(p + 4) << 32);
}

// `kern_manifest::values::to_f64`, one element.
__device__ double value(int kind, const unsigned char* p) {
    switch (kind) {
        case BF16: return (double)__uint_as_float((unsigned int)load16(p) << 16);
        case F16: {
            unsigned short h = load16(p);
            double sign = (h & 0x8000) ? -1.0 : 1.0;
            int exp = (h >> 10) & 0x1f;
            double man = (double)(h & 0x3ff);
            if (exp == 0x1f) return man != 0.0 ? __longlong_as_double(0x7ff8000000000000ll) : sign * __longlong_as_double(0x7ff0000000000000ll);
            return sign * (exp == 0 ? man / 1024.0 * exp2(-14.0) : (1.0 + man / 1024.0) * exp2((double)(exp - 15)));
        }
        case F32: return (double)__uint_as_float(load32(p));
        case E4M3: {
            unsigned char b = p[0];
            double sign = (b & 0x80) ? -1.0 : 1.0;
            int exp = (b >> 3) & 0xf;
            double man = (double)(b & 7);
            if (exp == 0xf && man == 7.0) return __longlong_as_double(0x7ff8000000000000ll);
            return sign * (exp == 0 ? man / 8.0 * exp2(-6.0) : (1.0 + man / 8.0) * exp2((double)(exp - 7)));
        }
        case E8M0: {
            unsigned char b = p[0];
            if (b == 255) return __longlong_as_double(0x7ff8000000000000ll);
            return exp2((double)b - 127.0);
        }
        case I8: return (double)(signed char)p[0];
        case U8: return (double)p[0];
        case I32: return (double)(int)load32(p);
        case U32: return (double)load32(p);
        case I64: return (double)(long long)load64(p);
        default: return (double)load64(p);
    }
}

// `kern_manifest::values::ulp_distance`'s key: sign-magnitude bits to a
// monotone integer so adjacent floats differ by 1.
__device__ __forceinline__ long long ulp_key(int kind, const unsigned char* p) {
    long long bits, sign_bit;
    switch (kind) {
        case BF16: case F16: bits = load16(p); sign_bit = 1ll << 15; break;
        case F32: bits = load32(p); sign_bit = 1ll << 31; break;
        case E4M3: bits = p[0]; sign_bit = 1ll << 7; break;
        default: return 0; // never: pair() measures ulps for is_float kinds only
    }
    return (bits & sign_bit) ? sign_bit - (bits & (sign_bit - 1)) - sign_bit : bits;
}

// `f64::total_cmp`'s key, so an argmax orders NaNs the way the host does.
__device__ __forceinline__ long long total_key(double v) {
    long long bits = __double_as_longlong(v);
    return bits ^ (long long)(((unsigned long long)(bits >> 63)) >> 1);
}

struct Acc {
    unsigned long long n_diff, signed_zero, nan_one, measured, max_ulp, max_abs_bits;
    __device__ void zero() { n_diff = signed_zero = nan_one = measured = max_ulp = max_abs_bits = 0; }
    __device__ void merge(const Acc& o) {
        n_diff += o.n_diff; signed_zero += o.signed_zero; nan_one += o.nan_one; measured += o.measured;
        max_ulp = max(max_ulp, o.max_ulp); max_abs_bits = max(max_abs_bits, o.max_abs_bits);
    }
};

// One element pair, as kern-test's `compare` counts it.
__device__ __forceinline__ void pair(int kind, int w, const unsigned char* x, const unsigned char* y, Acc& a) {
    bool same = true;
    for (int i = 0; i < w; i++) same &= x[i] == y[i];
    if (same) return;
    a.n_diff++;
    double fx = value(kind, x), fy = value(kind, y);
    if (isnan(fx) != isnan(fy)) { a.nan_one++; return; }
    if (fx == fy) { a.signed_zero++; return; }
    double d = fabs(fx - fy);
    // a non-negative double's bits order like the value: max on the bits
    if (!isnan(d)) a.max_abs_bits = max(a.max_abs_bits, (unsigned long long)__double_as_longlong(d));
    if (is_float(kind) && !isnan(fx) && !isnan(fy)) {
        long long k = ulp_key(kind, x) - ulp_key(kind, y);
        a.max_ulp = max(a.max_ulp, (unsigned long long)(k < 0 ? -k : k));
        a.measured++;
    }
}

__device__ Acc block_merge(Acc a, Acc* sh) {
    sh[threadIdx.x] = a;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x].merge(sh[threadIdx.x + s]);
        __syncthreads();
    }
    Acc r = sh[0];
    __syncthreads();
    return r;
}

// out[6]: n_diff, signed_zero, nan_one, measured, max_ulp, max_abs bits;
// zero before the launch.
extern "C" __global__ void cmp_reduce(const unsigned char* a, const unsigned char* b, unsigned long long n,
                                      int kind, unsigned long long* out) {
    __shared__ Acc sh[256];
    int w = width(kind);
    Acc acc; acc.zero();
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x)
        pair(kind, w, a + i * w, b + i * w, acc);
    Acc r = block_merge(acc, sh);
    if (threadIdx.x == 0) {
        atomicAdd(&out[0], r.n_diff); atomicAdd(&out[1], r.signed_zero);
        atomicAdd(&out[2], r.nan_one); atomicAdd(&out[3], r.measured);
        atomicMax(&out[4], r.max_ulp); atomicMax(&out[5], r.max_abs_bits);
    }
}

// bits[i] set when 64-byte block i of `pre` and `post` differ anywhere;
// `bytes` need not be a multiple of 64. Zero the bitmap before the launch.
extern "C" __global__ void changed_blocks(const unsigned char* pre, const unsigned char* post,
                                          unsigned long long bytes, unsigned int* bits) {
    unsigned long long blocks = (bytes + 63) / 64;
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < blocks;
         i += (unsigned long long)gridDim.x * blockDim.x) {
        unsigned long long lo = i * 64, hi = min(lo + 64, bytes);
        bool same = true;
        if (hi - lo == 64 && ((((unsigned long long)pre) | ((unsigned long long)post)) & 15) == 0) {
            const ulonglong2* p = (const ulonglong2*)(pre + lo);
            const ulonglong2* q = (const ulonglong2*)(post + lo);
            for (int k = 0; k < 4; k++) { ulonglong2 x = p[k], y = q[k]; same &= x.x == y.x && x.y == y.y; }
        } else {
            for (unsigned long long j = lo; j < hi; j++) same &= pre[j] == post[j];
        }
        if (!same) atomicOr(&bits[i / 32], 1u << (i % 32));
    }
}

// ---- one logits row per block: kern-test's `logit_row` without the label

struct Best { long long key; unsigned int idx; };

// Larger key first, lower index on ties: the host's `sort_by(value desc, index asc)`.
__device__ __forceinline__ bool better(Best x, Best y) { return x.key > y.key || (x.key == y.key && x.idx < y.idx); }

__device__ Best block_best(Best b, Best* sh) {
    sh[threadIdx.x] = b;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && better(sh[threadIdx.x + s], sh[threadIdx.x])) sh[threadIdx.x] = sh[threadIdx.x + s];
        __syncthreads();
    }
    Best r = sh[0];
    __syncthreads();
    return r;
}

__device__ double block_sum(double v, double* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    double r = sh[0];
    __syncthreads();
    return r;
}

// The best element strictly after `prev` in the row's total order (`prev`
// with key LLONG_MAX and idx 0 means the first).
__device__ Best next_best(int kind, int w, const unsigned char* row, unsigned int cols, Best prev, Best* sh) {
    Best mine; mine.key = 0x8000000000000000ll; mine.idx = 0xffffffffu;
    for (unsigned int j = threadIdx.x; j < cols; j += blockDim.x) {
        Best c; c.key = total_key(value(kind, row + (unsigned long long)j * w)); c.idx = j;
        if (better(prev, c) && better(c, mine)) mine = c;
    }
    return block_best(mine, sh);
}

#define TOP_MAX 64

// out: 12 u64 per row — Acc (6), top1 bits, top2 bits, kl bits, argmax_a,
// argmax_b, (rank_in_b << 32 | top overlap).
extern "C" __global__ void logit_rows(const unsigned char* a, const unsigned char* b, unsigned int cols, int kind,
                                      int top, unsigned long long* out) {
    __shared__ union { Acc acc[256]; Best best[256]; double sum[256]; } sh;
    __shared__ unsigned int sel_a[TOP_MAX], sel_b[TOP_MAX];
    int w = width(kind);
    unsigned long long row = (unsigned long long)blockIdx.x * cols * w;
    const unsigned char* ra = a + row;
    const unsigned char* rb = b + row;

    Acc acc; acc.zero();
    for (unsigned int j = threadIdx.x; j < cols; j += blockDim.x) pair(kind, w, ra + (unsigned long long)j * w, rb + (unsigned long long)j * w, acc);
    acc = block_merge(acc, sh.acc);

    Best first; first.key = 0x7fffffffffffffffll; first.idx = 0;
    Best am_a = next_best(kind, w, ra, cols, first, sh.best);
    Best second = next_best(kind, w, ra, cols, am_a, sh.best);
    Best am_b = next_best(kind, w, rb, cols, first, sh.best);
    double top1 = value(kind, ra + (unsigned long long)am_a.idx * w);
    double top2 = cols > 1 ? value(kind, ra + (unsigned long long)second.idx * w) : -__longlong_as_double(0x7ff0000000000000ll);
    double mb = value(kind, rb + (unsigned long long)am_b.idx * w);

    double sa = 0.0, sb = 0.0;
    for (unsigned int j = threadIdx.x; j < cols; j += blockDim.x) {
        sa += exp(value(kind, ra + (unsigned long long)j * w) - top1);
        sb += exp(value(kind, rb + (unsigned long long)j * w) - mb);
    }
    double la = top1 + log(block_sum(sa, sh.sum));
    double lb = mb + log(block_sum(sb, sh.sum));
    // A's argmax as B ranks it: 1 + the elements B orders ahead of it.
    Best a_in_b; a_in_b.key = total_key(value(kind, rb + (unsigned long long)am_a.idx * w)); a_in_b.idx = am_a.idx;
    double kl = 0.0, ahead = 0.0;
    for (unsigned int j = threadIdx.x; j < cols; j += blockDim.x) {
        double x = value(kind, ra + (unsigned long long)j * w), y = value(kind, rb + (unsigned long long)j * w);
        kl += exp(x - la) * ((x - la) - (y - lb));
        Best c; c.key = total_key(y); c.idx = j;
        ahead += better(c, a_in_b) ? 1.0 : 0.0;
    }
    kl = block_sum(kl, sh.sum);
    unsigned long long rank_in_b = 1 + (unsigned long long)block_sum(ahead, sh.sum);

    int k = min(top, (int)cols);
    Best pa = first, pb = first;
    for (int r = 0; r < k; r++) {
        pa = next_best(kind, w, ra, cols, pa, sh.best);
        pb = next_best(kind, w, rb, cols, pb, sh.best);
        if (threadIdx.x == 0) { sel_a[r] = pa.idx; sel_b[r] = pb.idx; }
    }
    __syncthreads();
    unsigned int overlap = 0;
    if (threadIdx.x == 0) {
        for (int i = 0; i < k; i++)
            for (int j = 0; j < k; j++) overlap += sel_a[i] == sel_b[j];
        unsigned long long* o = out + (unsigned long long)blockIdx.x * 12;
        o[0] = acc.n_diff; o[1] = acc.signed_zero; o[2] = acc.nan_one; o[3] = acc.measured;
        o[4] = acc.max_ulp; o[5] = acc.max_abs_bits;
        o[6] = __double_as_longlong(top1); o[7] = __double_as_longlong(top2); o[8] = __double_as_longlong(kl);
        o[9] = am_a.idx; o[10] = am_b.idx; o[11] = (rank_in_b << 32) | overlap;
    }
}
