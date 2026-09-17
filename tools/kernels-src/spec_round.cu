// The device-side glue of a speculative round (`round` = draft → verify →
// draft_precompute → accept → advance, one program, one CUDA graph, one
// host sync per round): what a host used to do between the phases. The
// host stages only the anchor of every sequence; the round builds its own
// rows and says how many it took (the manifest's `count` output). Per
// sequence i of the batch, `block` rows (anchor + block-1 drafts); a draft
// output is `[seqs, dstride]` (dstride ≥ block-1: DSpark drafts one more
// than its 7-row round verifies).
//
//   kern_splice_draft:  draft's ids from the anchor and the mask —
//                       ids[i*block] = anchor[i], ids[i*block + 1 + j] = mask.
//   kern_splice_verify: verify's ids from draft's output —
//                       ids[i*block]         = anchor[i]
//                       ids[i*block + 1 + j] = drafts[i*dstride + j].
//   kern_spec_count:    a = longest prefix of drafts[i] matched by
//                       verify[i] (row j predicts what follows draft j);
//                       nacc[i] = a + 1: the rows of verify[i] the caller
//                       takes.
//   kern_spec_lines:    the sequence's line moves to entry nacc[i]-1 of its
//                       cell in every row of a wide line table:
//                       line_adv[(r*seqs_max + i)*block + e] = e == a ? line_in[(r*seqs_max + i)*block] : 0
//                       (entry 0 of `line_in` is the line the caller staged;
//                       0 is the null line). rows ≤ blockDim.x threads.
//   kern_ones_i32:      out[i] = 1 for i < n: a per-sequence constant a
//                       kernel reads through a pointer (the count a resumed
//                       state takes at a step's start).
//   Every kernel: grid.x = seqs, block-1 ≤ blockDim.x threads.
//
//   nvcc -cubin -arch=sm_103a -o target/cubins/spec_round.cubin tools/kernels-src/spec_round.cu

extern "C" __global__ void kern_splice_draft(
    const long long* __restrict__ anchor, long long* __restrict__ ids, int block, long long mask) {
  const int i = blockIdx.x;
  const int t = threadIdx.x;
  if (t == 0) ids[(long long)i * block] = anchor[i];
  if (t < block - 1) ids[(long long)i * block + 1 + t] = mask;
}

extern "C" __global__ void kern_splice_verify(
    const long long* __restrict__ anchor, const long long* __restrict__ drafts,
    long long* __restrict__ ids, int block, int dstride) {
  const int i = blockIdx.x;
  const int t = threadIdx.x;
  if (t == 0) ids[(long long)i * block] = anchor[i];
  if (t < block - 1) ids[(long long)i * block + 1 + t] = drafts[(long long)i * dstride + t];
}

extern "C" __global__ void kern_spec_count(
    const long long* __restrict__ drafts, const long long* __restrict__ verify,
    int* __restrict__ nacc, int block, int dstride) {
  const int i = blockIdx.x;
  if (threadIdx.x != 0) return;
  const long long* d = drafts + (long long)i * dstride;
  const long long* v = verify + (long long)i * block;
  int a = 0;
  while (a < block - 1 && d[a] == v[a]) ++a;
  nacc[i] = a + 1;
}

extern "C" __global__ void kern_spec_lines(
    const int* __restrict__ line_in, const int* __restrict__ nacc, int* __restrict__ line_adv,
    int block, int rows, int seqs_max) {
  const int i = blockIdx.x;
  const int a = nacc[i] - 1;
  for (int r = threadIdx.x; r < rows; r += blockDim.x) {
    const long long cell = ((long long)r * seqs_max + i) * block;
    const int line = line_in[cell];
    for (int e = 0; e < block; ++e) line_adv[cell + e] = e == a ? line : 0;
  }
}

extern "C" __global__ void kern_ones_i32(int* __restrict__ out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = 1;
}
