// Copies between one line of a per-sequence state and a workspace buffer,
// the line picked by an index the host wrote (a line table entry), for
// kernels that take a state row by plain pointer rather than by index: the
// FLA chunk kernel's h0/ht are `[N, HV, V, K]` tensors, so vLLM gathers
// `ssm_state[state_indices]` before it and scatters the result back after.
//
//   kern_line_gather:  dst[0:nbytes]                       = state[idx[0]*line_bytes + off ..]
//   kern_line_scatter: state[idx[0]*line_bytes + off ..]   = src[0:nbytes]
//
// 16-byte vectors, one thread each; grid.x covers nbytes/16 threads
// (nbytes, off and line_bytes are multiples of 16). Line index 0 is the
// null line: nothing is copied.
//
//   nvcc -cubin -arch=sm_103a -o kernels/line_copy.cubin tools/kernels-src/line_copy.cu
#include <cstdint>

extern "C" __global__ void kern_line_gather(
    const int* __restrict__ idx, const unsigned char* __restrict__ state,
    unsigned char* __restrict__ dst, long long line_bytes, long long off, long long nbytes) {
  const long long line = idx[0];
  if (line <= 0) return;
  const long long v = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (v * 16 >= nbytes) return;
  const uint4* s = reinterpret_cast<const uint4*>(state + line * line_bytes + off);
  reinterpret_cast<uint4*>(dst)[v] = s[v];
}

extern "C" __global__ void kern_line_scatter(
    const int* __restrict__ idx, unsigned char* __restrict__ state,
    const unsigned char* __restrict__ src, long long line_bytes, long long off, long long nbytes) {
  const long long line = idx[0];
  if (line <= 0) return;
  const long long v = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (v * 16 >= nbytes) return;
  uint4* d = reinterpret_cast<uint4*>(state + line * line_bytes + off);
  d[v] = reinterpret_cast<const uint4*>(src)[v];
}
