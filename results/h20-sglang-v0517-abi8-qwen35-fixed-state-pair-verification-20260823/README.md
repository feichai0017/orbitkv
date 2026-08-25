# OrbitKV ABI8 Qwen3.5 fixed-state pair verification

Status: real-model H20 pair verification passed; not qualification and no
performance GO.

This record binds clean source snapshot `7385ee586974ffd09dffecc415d52098f373e32a`
to official `Qwen/Qwen3.5-0.8B`, SGLang `v0.5.17` at
`29481685462732237d80d86076d6563e1f658102`, ABI8, and one NVIDIA H20
(`GPU-3a35e57b-fc54-5620-56ee-deaf5a9c40d3`). The execution profile is
page16 BF16 NHD, eager FA3 for Full attention, Triton for GDN
general/prefill/decode, FP32 recurrent state, `no_buffer`, Radix disabled, and
TP/PP/DP/DCP = 1.

All six stock/manager pairs passed across three epochs: B1 uses one request and
one measured iteration; B4 uses four requests and five measured iterations per
epoch. Every request used a fresh prompt, reported `cached_tokens=0`, produced
exactly 33 output tokens, and matched stock token-for-token. In total the
records cover 63 requests and 2,079 generated tokens per mode. Every manager
record reports exact token-manager transactions, real CUDA completion-event
observation, fixed-state prepare/clear/event/retire/ACK activity, zero
fixed-state copies, and a fully drained token and fixed-state pool.

The three-epoch aggregate is deliberately a negative performance result:

| Batch | Stock mean | OrbitKV mean | OrbitKV over stock |
| --- | ---: | ---: | ---: |
| B1 | 1.581859 s | 1.589885 s | +0.5074% slower |
| B4 | 0.915634 s | 0.949873 s | +3.7394% slower |

At 33 generated tokens per request, these means correspond to 20.8615 versus
20.7562 output tokens/s for B1 (-0.5048% manager throughput), and 144.1624
versus 138.9659 output tokens/s for B4 (-3.6046% manager throughput). These
throughput values are derived from the stored mean iteration times.

The B4 steady-state diagnostic after excluding each fresh process's first
iteration is 0.715970 s stock versus 0.766410 s OrbitKV, or +7.0451% slower.
That is 184.3654 versus 172.2316 output tokens/s, or -6.5814% manager
throughput. These derived numbers are explanatory only and are not part of the
pair contract.
The evidence therefore has `performance_go=false`, `qualified=false`, and
`hardware_attested=false`. The last field means that the portable verifier does
not independently attest the machine; each raw record nevertheless contains
the observed H20 UUID and GPU snapshots.

All 12/12 stored stock/manager stderr logs contain PyTorch's `CudaIPCTypes`
producer-exit warning. It is treated as a process-teardown warning because all
six pair records still pass and every manager record's JSON final token and
fixed-state census plus global cleanup reports a complete drain. It remains a
disclosed residual harness warning.

This run does not show an intrinsic same-capacity token-KV reduction: both
modes reserve the same Full KV tensor capacity, so that saving is **0%**. It
also does not qualify Prefix/Radix sharing, a production fixed-state copy or
replacement trigger, CUDA Graphs, overlap scheduling, speculation, distributed
or multi-GPU execution, other Qwen3.5 configurations, general hybrid-attention
support, or complete replacement of SGLang's KV block manager. SGLang still
owns tensor allocation, model kernels, scheduling, and execution.

The evidence is intentionally separate from the sealed Full/Full+SWA Prefix
qualification. Verify its hashes, source provenance, ABI, exact input identity,
fresh prompts, output equality, counters, drain state, and aggregate timings
from a trusted checkout with:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tools/verify_qwen35_h20_pair_evidence.py
```

`qualification/source.bundle` is a thin Git bundle containing the exact run
commit and requiring repository commit
`0eb74430b7e3228c45b137a539e6d266b51a5bed`. The separately copied
`qualification/source` tree is the audited qualification source closure used
for readable offline verification.

Online retrieval verification was performed on 2026-08-23. The immutable
Hugging Face revision, resolved reference, Git blob identities, model-file
hashes, and Xet/LFS shard identity are sealed in
`qualification/model-provenance.json` for offline verification; the model
config and safetensors index are included byte-for-byte under
`qualification/model/`.
