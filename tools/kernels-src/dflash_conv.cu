// DFlash2 two-tap grouped conv (vLLM qwen3_dflash2.DFlashGroupedConv):
//
//   coef[t, side, tap, ch] = base[side, tap, ch] + delta[t, side, tap, group(ch)]
//   out[t, ch] = coef[t,side,0,ch] * x[t, ch]
//              + coef[t,side,1,ch] * x[t-1, ch] * (t % 8 != 0)
//
// group(ch) = ch / 16 (320 groups over 5120 channels), delta is the
// kernel_projection GEMM output viewed [T, 2 sides, 2 taps, 320 groups],
// side 0 = `prepare` (before attention / MLP), side 1 = `finish` (after).
// The `t % 8` mask isolates each 8-token draft block.  In vLLM this is a
// chain of ATen ops in bf16; every intermediate is rounded to bf16 in the
// same order here (add, mul, mul, add).  In place on x: one thread per
// channel walks the tokens in order, so x[t-1] is read before x[t] is
// written.
//
//   nvcc -cubin -arch=sm_103a -o kernels/dflash_conv.cubin tools/kernels-src/dflash_conv.cu
#include <cuda_bf16.h>

extern "C" __global__ void kern_dflash_conv_bf16(
    __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ delta,
    const __nv_bfloat16* __restrict__ base, int tokens, int dim, int groups,
    int delta_stride, int side) {
  const int ch = blockIdx.x * blockDim.x + threadIdx.x;
  if (ch >= dim) return;
  const int gsz = dim / groups;
  const int g = ch / gsz;
  const float b0 = __bfloat162float(base[(side * 2 + 0) * dim + ch]);
  const float b1 = __bfloat162float(base[(side * 2 + 1) * dim + ch]);
  __nv_bfloat16 prev = __float2bfloat16(0.0f);
  for (int t = 0; t < tokens; ++t) {
    const __nv_bfloat16 xt = x[(long long)t * dim + ch];
    const __nv_bfloat16* dl = delta + (long long)t * delta_stride + side * 2 * groups;
    const __nv_bfloat16 c0 = __float2bfloat16(b0 + __bfloat162float(dl[g]));
    const __nv_bfloat16 c1 = __float2bfloat16(b1 + __bfloat162float(dl[groups + g]));
    const __nv_bfloat16 t0 = __float2bfloat16(__bfloat162float(c0) * __bfloat162float(xt));
    __nv_bfloat16 t1 = __float2bfloat16(__bfloat162float(c1) * __bfloat162float(prev));
    if ((t & 7) == 0) t1 = __float2bfloat16(0.0f);
    x[(long long)t * dim + ch] = __float2bfloat16(__bfloat162float(t0) + __bfloat162float(t1));
    prev = xt;
  }
}
