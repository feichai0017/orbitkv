# H20 SGLang token-relocation diagnostic

The strict verifier passes all 16 raw process records and all eight paired
comparisons in this directory. This is a verified diagnostic, not a
qualification record: `diagnostic_only=true`, `sealed=false`,
`source_dirty=true`, `hardware_attested=false`, `qualified=false`, and
`performance_go=false`. The raw runtime snapshots consistently record one
NVIDIA H20, but that is a recorded observation rather than independent
hardware attestation.

Separately, the H20 CUDA component-conformance run passed all seven cases: the
payload-oracle case, stale-member failure atomicity at B1/B4/B32, and real CUDA
copy/consumer execution at B1/B4/B32. Each CUDA case runs two cycles with
257-byte coordinate-bearing payloads on real non-default append, copy, and
consumer streams, including ACK-gated generation reuse and final drain. This
is narrow component conformance, not sealed L3/L4 or performance/capacity
qualification.

[H20 component-conformance JUnit record](component-conformance.xml)

## Pinned execution contract

- official SGLang `v0.5.17`, revision
  `29481685462732237d80d86076d6563e1f658102`;
- Qwen2.5-0.5B, Full attention, page16 BF16 NHD KV storage, eager single-GPU
  execution, and the token-indexed FlashInfer backend;
- Radix, overlap scheduling, and CUDA Graphs disabled;
- four order-balanced epochs: epochs 1 and 3 run Naive then Relocate, while
  epochs 2 and 4 run Relocate then Naive;
- B1 and B4, five iterations per process, 48 prompt tokens and 41 decode
  tokens per request, with two reclamation rounds per iteration; and
- identical model, prompts, sampling, policy, and configured capacity in each
  pair. The only allowed difference is Naive versus Relocate execution mode.

The paired reference deliberately uses FlashInfer in both modes. A separate
relocate-only FA3 end-to-end smoke passed, but FA3 page tables cannot represent
the sparse retained slots required by the Naive oracle. Naive+FA3 is therefore
invalid and now fails closed; the FA3 smoke contributes no same-policy timing
comparison.

## Correctness and lifecycle result

All 8/8 Naive/Relocate pairs have exact output-token equality. Every Relocate
process performs ten scheduler-batch relocation/copy events: B1 records 240
moves and copied tokens plus ten reclaimed pages, while B4 records 960 moves
and copied tokens plus 40 reclaimed pages. Every manager census drains fully,
and all failure, quarantine, and fail-stop counters are zero. These counters
show that the relocation and page-reclamation path was active for this
workload; byte-exact copy behavior is established separately by the CUDA
component-conformance record above. Neither establishes a configured-capacity
or end-to-end memory reduction.

## Diagnostic timing

Iteration 0 is excluded from every process, leaving 16 hot samples per mode
and batch size. Percentiles use the inclusive method.

| Case | Naive mean / median / p95 | Relocate mean / median / p95 | Relocate throughput delta | Relocate mean / p95 latency delta |
| --- | --- | --- | ---: | ---: |
| B1 | 0.993521 / 0.946700 / 1.258864 s | 0.923097 / 0.915447 / 0.971295 s | +7.629% | -7.088% / -22.844% |
| B4 | 0.981837 / 0.979866 / 0.990581 s | 0.994085 / 0.998254 / 1.019109 s | -1.232% | +1.247% / +2.880% |

The B1 pooled observation is favorable, but its per-epoch mean-latency deltas
are +2.204%, -6.419%, -20.024%, and -0.748%, so the small matrix does not
support a speedup claim. B4 is slightly slower. The authoritative conclusion
remains `performance_go=false`.

## Claim boundary

This archive establishes recorded-device execution of the pinned SGLang
token-relocation path, exact output-token equality, positive relocation and
reclamation activity, clean final drain, and zero failure/quarantine counters
for the workload above. It does not establish independent hardware
attestation, sealed L3 or L4 qualification, production readiness, asynchronous
copy/consumer overlap, Prefix or Hybrid/SWA relocation, MLA, CUDA Graphs,
multi-GPU behavior, a capacity advantage, or an end-to-end memory saving. It
also does not turn the B1 observation into a general latency or throughput
claim and does not qualify OrbitKV as a complete SGLang replacement.

Reproduce the strict archive check with:

```bash
python3 tools/verify_token_relocation_h20_evidence.py \
  results/h20-sglang-v0517-token-relocation-diagnostic-20260825
```
