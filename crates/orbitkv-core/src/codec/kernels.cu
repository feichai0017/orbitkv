// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted for Rust-owned, standalone GPU storage encode/decode.
// TurboQuant MSE/NC equations follow LMCache's Apache-2.0 TurboQuant serde.
// OrbitKV uses a versioned integer sign generator and its own storage layout.
__device__ float from16(unsigned short x, int bf) {
  if (bf)
    return __uint_as_float((unsigned int)x << 16);
  float y;
  asm("cvt.f32.f16 %0, %1;" : "=f"(y) : "h"(x));
  return y;
}
__device__ unsigned short to16(float x, int bf) {
  if (bf) {
    unsigned int u = __float_as_uint(x);
    return (u + 0x7fff + ((u >> 16) & 1)) >> 16;
  }
  unsigned short y;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(y) : "f"(x));
  return y;
}
__device__ float fp8value(unsigned char b) {
  int e = (b >> 3) & 15, m = b & 7;
  float x = e == 0 ? m / 512.f : __uint_as_float(((e + 120) << 23) | (m << 20));
  return b & 128 ? -x : x;
}
__device__ unsigned char fp8bits(float x) {
#if __CUDA_ARCH__ >= 890
  unsigned short packed;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
      : "=h"(packed)
      : "f"(0.f), "f"(x));
  return (unsigned char)packed;
#else
  float a = fabsf(x);
  int lo = 0, hi = 126;
  while (lo < hi) {
    int mid = (lo + hi) / 2;
    if (fp8value(mid) < a)
      lo = mid + 1;
    else
      hi = mid;
  }
  int lower = max(0, lo - 1);
  float down = a - fp8value(lower), up = fp8value(lo) - a;
  int q = down < up || (down == up && !(lower & 1)) ? lower : lo;
  return q | ((__float_as_uint(x) >> 24) & 128);
#endif
}
extern "C" __global__ void fp8_encode(const unsigned short *src,
                                      unsigned char *dst, unsigned long long n,
                                      int bf, unsigned int *invalid) {
  for (unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (unsigned long long)gridDim.x * blockDim.x) {
    float x = from16(src[i], bf);
    if (!isfinite(x) || fabsf(x) > 448.f) {
      atomicOr(invalid, 1u);
      dst[i] = 0;
    } else
      dst[i] = fp8bits(x);
  }
}
extern "C" __global__ void fp8_decode(const unsigned char *src,
                                      unsigned short *dst, unsigned long long n,
                                      int bf) {
  for (unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (unsigned long long)gridDim.x * blockDim.x)
    dst[i] = to16(fp8value(src[i]), bf);
}
__device__ float sign_for(unsigned int seed, unsigned int i) {
  unsigned int x = seed ^ (i * 0x9e3779b9u);
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x & 1 ? -1.f : 1.f;
}
__device__ void hadamard(float *x, int dim) {
  int t = threadIdx.x;
  for (int stride = 1; stride < dim; stride <<= 1) {
    float a = x[t], b = x[t ^ stride];
    __syncthreads();
    x[t] = (t & stride) ? b - a : a + b;
    __syncthreads();
  }
  x[t] *= rsqrtf((float)dim);
  __syncthreads();
}
extern "C" __global__ void
turbo_encode(const unsigned short *src, unsigned char *dst, int vectors,
             int dim, int bits, int bf, int role, unsigned int seed,
             const float *centroids, unsigned int *invalid) {
  __shared__ float x[256];
  __shared__ unsigned char q[256];
  __shared__ float norm, scale, minimum;
  int t = threadIdx.x, packed = (dim * bits + 7) / 8;
  for (int v = blockIdx.x; v < vectors; v += gridDim.x) {
    int key = role == 2 ? (v % 2 == 0) : role;
    float value = from16(src[(unsigned long long)v * dim + t], bf);
    if (!isfinite(value))
      atomicOr(invalid, 1u);
    x[t] = isfinite(value) ? value : 0.f;
    __syncthreads();
    unsigned long long offset =
        role == 2 ? (unsigned long long)(v / 2) * (2 * packed + 6) +
                        (v % 2) * (packed + 2)
                  : (unsigned long long)v * (packed + (key ? 2 : 4));
    unsigned char *out = dst + offset;
    if (t == 0) {
      float sum = 0.f, low = x[0], high = x[0];
      for (int i = 0; i < dim; ++i) {
        sum += x[i] * x[i];
        low = fminf(low, x[i]);
        high = fmaxf(high, x[i]);
      }
      norm = sqrtf(sum);
      minimum = low;
      scale = fmaxf((high - low) / ((1 << bits) - 1), 1.e-8f);
      unsigned short first = to16(key ? norm : scale, 0),
                     second = to16(minimum, 0);
      if (!isfinite(from16(first, 0)) ||
          (!key && (!isfinite(from16(second, 0)) || from16(first, 0) == 0.f)))
        atomicOr(invalid, 1u);
      out[packed] = first;
      out[packed + 1] = first >> 8;
      if (!key) {
        out[packed + 2] = second;
        out[packed + 3] = second >> 8;
      }
    }
    __syncthreads();
    if (key) {
      x[t] = (norm > 0.f ? x[t] / norm : 0.f) * sign_for(seed, t);
      __syncthreads();
      hadamard(x, dim);
      int index = 0;
      while (index < (1 << bits) - 1 &&
             x[t] > (centroids[index] + centroids[index + 1]) * 0.5f)
        ++index;
      q[t] = index;
    } else
      q[t] =
          min((1 << bits) - 1, max(0, (int)((x[t] - minimum) / scale + 0.5f)));
    __syncthreads();
    if (t < packed) {
      unsigned int byte = 0;
      for (int bit = 0; bit < 8; ++bit) {
        int pos = t * 8 + bit;
        if (pos < dim * bits)
          byte |= ((q[pos / bits] >> (pos % bits)) & 1) << bit;
      }
      out[t] = byte;
    }
    __syncthreads();
  }
}
extern "C" __global__ void turbo_decode(const unsigned char *src,
                                        unsigned short *dst, int vectors,
                                        int dim, int bits, int bf, int role,
                                        unsigned int seed,
                                        const float *centroids) {
  __shared__ float x[256];
  __shared__ float correction;
  int t = threadIdx.x, packed = (dim * bits + 7) / 8;
  for (int v = blockIdx.x; v < vectors; v += gridDim.x) {
    int key = role == 2 ? (v % 2 == 0) : role;
    unsigned long long offset =
        role == 2 ? (unsigned long long)(v / 2) * (2 * packed + 6) +
                        (v % 2) * (packed + 2)
                  : (unsigned long long)v * (packed + (key ? 2 : 4));
    const unsigned char *in = src + offset;
    int pos = t * bits, byte = pos / 8, shift = pos % 8;
    unsigned int word = in[byte];
    if (byte + 1 < packed)
      word |= (unsigned int)in[byte + 1] << 8;
    unsigned int q = (word >> shift) & ((1 << bits) - 1);
    float first = from16(in[packed] | (unsigned short)in[packed + 1] << 8, 0);
    if (key) {
      x[t] = centroids[q];
      __syncthreads();
      if (t == 0) {
        float sum = 0;
        for (int i = 0; i < dim; ++i)
          sum += x[i] * x[i];
        correction = rsqrtf(sum + 1.e-16f);
      }
      __syncthreads();
      hadamard(x, dim);
      x[t] *= sign_for(seed, t) * correction * first;
    } else {
      float minimum =
          from16(in[packed + 2] | (unsigned short)in[packed + 3] << 8, 0);
      x[t] = q * first + minimum;
    }
    dst[(unsigned long long)v * dim + t] = to16(x[t], bf);
    __syncthreads();
  }
}
