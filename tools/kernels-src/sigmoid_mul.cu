// Gated attention output, bit-exact with vLLM's eager
// `attn_output * torch.sigmoid(gate)` (two ATen ops, two bf16 roundings):
//   g = bf16(1 / (1 + expf(-f32(gate))))     (ATen sigmoid, opmath f32)
//   out = bf16(f32(attn) * f32(g))           (ATen mul on bf16)
// attn / out are contiguous [tokens, heads*head_dim]; gate is a strided view
// (per-head [q | gate] halves of the fused qkv projection):
//   gate(t, h, d) = gate + t*gate_tstride + h*gate_hstride + d
//
//   nvcc -cubin -arch=sm_103a -o kernels/sigmoid_mul.cubin tools/kernels-src/sigmoid_mul.cu
#include <cuda_bf16.h>

// Grid [tokens, ceil(heads*head_dim / 2048)], block 256: 8 consecutive
// elements per thread, one 16-byte load per operand.
extern "C" __global__ void kern_sigmoid_mul_bf16(
    __nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ attn,
    const __nv_bfloat16* __restrict__ gate, int heads, int head_dim,
    int gate_tstride, int gate_hstride) {
    // Programmatic dependent launch: everything before this line may run
    // while the previous kernel is still finishing; nothing produced by it
    // is read or overwritten until the wait returns. The trigger right after
    // lets the next launch start its own prologue (a GEMM's weight stream).
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

  const long long t = blockIdx.x;
  const int n = heads * head_dim;
  const int j = (blockIdx.y * blockDim.x + threadIdx.x) * 8;
  if (j >= n) return;
  const int h = j / head_dim, d = j % head_dim;
  const uint4 gv = *reinterpret_cast<const uint4*>(gate + t * gate_tstride + (long long)h * gate_hstride + d);
  const uint4 av = *reinterpret_cast<const uint4*>(attn + t * n + j);
  const __nv_bfloat162* gp = reinterpret_cast<const __nv_bfloat162*>(&gv);
  const __nv_bfloat162* ap = reinterpret_cast<const __nv_bfloat162*>(&av);
  uint4 ov;
  __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&ov);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const float x0 = __bfloat162float(gp[i].x), x1 = __bfloat162float(gp[i].y);
    const __nv_bfloat162 g = __floats2bfloat162_rn(1.0f / (1.0f + expf(-x0)), 1.0f / (1.0f + expf(-x1)));
    op[i] = __floats2bfloat162_rn(__bfloat162float(ap[i].x) * __bfloat162float(g.x),
                                  __bfloat162float(ap[i].y) * __bfloat162float(g.y));
  }
  *reinterpret_cast<uint4*>(out + t * n + j) = ov;
}
