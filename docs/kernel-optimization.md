# CUDA kernel optimization

OrbitKV uses the [NVIDIA KDA workflow](https://github.com/NVlabs/kda) to define a
bounded kernel task, validate a candidate and measure it before changing the
production backend. The execution queue remains [S4](completion-plan.md#s4--finish-communication-execution-and-demonstrate-gains).

## Fragmented mapped-host copy

The current `KernelBackend` consumes three `u64` values per descriptor:
destination, source and byte count. One CTA handles each fragment, vectorizing
16-byte aligned addresses and copying remaining bytes exactly. The worker owns
its stream and descriptor scratch until physical completion. The direct DMA
backend already combines adjacent ranges; a new kernel must improve the actual
fragmented path rather than a deliberately uncoalesced DMA comparison.

CUDA device code lives in separate `.cu` files. The production backend embeds
`crates/orbitkv-core/src/transfer/kernel.cu` with Rust's `include_str!` and compiles
it through NVRTC; installed binaries do not need a source-file lookup.
`benches/kernel_copy.py` reads that same file as its baseline and
`benches/kernels/fragmented_copy.cu` as its **benchmark-only** candidate. The
candidate assigns multiple CTAs to large fragments in small batches. Moving
these sources preserves the previously measured CUDA bytes and production
launch behavior. The driver
checks both copy directions, empty and zero-size batches, independent address
misalignment, scalar tails, reordered ranges, source preservation, guard bytes
and batches exceeding the grid limit before collecting timing samples.

```bash
python -m benches.kernel_copy \
  --source-root /path/to/orbitkv \
  --output /var/tmp/orbitkv-kda/validation-001 --validate-only
python -m benches.kernel_copy \
  --source-root /path/to/orbitkv \
  --output /var/tmp/orbitkv-kda/measurement-001
```

Use a new external directory for every attempt. The driver requires PyTorch,
a working CUDA device and the CUDA toolkit's NVRTC library. Set `--nvrtc` for a
non-default toolkit path. It records GPU architecture, source/compiler inputs,
matched sample order, CUDA kernel time, descriptor-upload-plus-kernel time and
Python submit-to-drain time. The latter is a benchmark boundary, not native
restore latency. It does not require an inference engine or Cache Manager.

Promotion requires independent correctness reproduction, improvement on actual
fragment sizes in both directions and no more than 5% regression on the existing
4 KiB fragmented workload. Follow with native `transfer_h2d` and installed-engine
Publish/Restore gates under inference contention; preserve source/destination
holds, layer dependencies and final drain. Kernel timing alone cannot establish
a TTFT/ITL advantage or qualify a production replacement.

## Current candidate status

The A100 SM80 candidate passes all 30 byte/canary controls and three matched
measurement runs. The frozen raw sources, timings and failed attempts are at
`/root/orbitkv-artifacts/kda-fragment-copy-20261006/`, with an A100 mirror at
`/workspace/orbitkv-kda-fragment-copy-20261006/`. The component evidence is
**implemented and locally measured; independent acceptance and production
promotion remain open**. It does not qualify H20, inference contention or the
DMA route. No production kernel is changed.

## Other candidate requirements

- Encoding and checksum fusion: identify a measured extra HBM pass, preserve the
  stored byte format and IEEE CRC32 result, and validate corrupted input before
  writing GPU destinations. Measure encoding, checksum, D2H and SSD/peer I/O
  together.
- FP8 and TurboQuant: use the actual SM architecture, byte-level conversion
  controls and declared quality tolerances. A100 software conversion and H20
  hardware capabilities need separate profiles. Recurrent state remains exact.

Supply representative block sizes, tensor strides, fragmentation, concurrent
inference load and storage/transport budgets. Nsight Compute counters help
identify bandwidth, occupancy and instruction bottlenecks when the profiler and
device permissions are available. Without those counters, retain the limitation
and measured timings; do not invent a bottleneck classification.
