# Compiler-constrained schedule benefit

This record qualifies a narrow same-engine serving improvement and records the
remaining product gap. The workload is the same released Full+Sliding
checkpoint and C2 16-request, 127-input/256-output trace used by the prior
stock-SGLang comparison.

OrbitKV now registers persistent K/V state as a required output-to-input alias.
Luminal rejects any candidate or loaded artifact that materializes one of those
updates. A 16-candidate search produced two buckets with all 36 K/V tensors
in-place and zero graph-visible K/V copy-back bytes.

Four alternating release-mode epochs against the previous two-candidate
OrbitKV artifact improved median output throughput from 593.57 to 679.52
token/s (+14.4%), reduced median TTFT from 98.14 to 66.68 ms (-32.0%), reduced
TPOT from 2.984 to 2.682 ms (-10.1%), and reduced end-to-end latency from
858.93 to 750.56 ms (-12.6%). Every request completed with 256 output tokens
and no reported error. Each arm was deterministic across its four epochs.

The new artifact also passes the existing independent B2 eight-token reference
probe and drains the OrbitKV manager completely. Its random-trace text digest
differs from the previous artifact, so this is not a strict cross-schedule
output-equivalence claim.

Against stock SGLang v0.5.17, four alternating epochs reached 678.64 versus
1135.11 token/s (0.598x). OrbitKV TPOT was 2.685 versus 1.692 ms (1.59x), TTFT
was 66.92 versus 14.28 ms (4.69x), and end-to-end latency was 751.75 versus
444.74 ms (1.69x). The configured OrbitKV K/V payload remains 40.4% smaller,
but the serving-competitiveness gate still fails.

The qualified contribution is therefore compiler-constrained persistent-state
selection with a measured same-engine benefit. It is not an OrbitKV-over-SGLang
speed claim or a complete replacement claim.
