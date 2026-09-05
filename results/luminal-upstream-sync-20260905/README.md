# Luminal upstream synchronization diagnostic

Status: source sync plus host and H20 regression evidence. This record does not
qualify broad model performance, throughput, capacity, or production use.

The OrbitKV Luminal fork merged upstream `c28a9fb`, including CUDA graph memory
reclamation and retained shape-specific graph variants on a shared arena. The
merge preserved OrbitKV's externally managed paged-attention metadata, stable
input allocations, persistent-input relocation, bucket introspection, and
explicit parent/child CUDA Graph composition.

Validation completed:

- Luminal CUDA test binary compilation and Clippy;
- stable-input child-graph replay on one H20;
- child-graph followed by a device-to-device copy node on one H20;
- released dense checkpoint prefill, decode capture, diagnostic replay,
  token-only replay, device/host greedy parity, persistent K/V updates, and
  four OrbitKV publications;
- 20 alternating matched fixed-step runs: eager median 4662.4 us, child-graph
  median 4435.4 us, replay/eager ratio 0.951.

The matched result confirms that the upstream sync did not reverse the narrow
fixed-signature benefit. It does not establish continuous-batching throughput
or benefits for other shapes and models.

## Source

- OrbitKV parent before this evidence commit: `b14a50f` plus upstream pin update
- OrbitKV Luminal fork merge: `9b8ac26243bae5bc5c37fb56ee3eb6554a05e7bc`
- merged Luminal upstream head: `c28a9fb07ac122c1c897588fd83147cfa2f9bcc6`
- checkpoint: `/workspace/models/qwen2.5-0.5b-instruct`
- accelerator: NVIDIA H20, compute capability 9.0
