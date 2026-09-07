# Serial model engine qualification

Status: passed on a real H20 with a released, unmodified Full+Sliding
checkpoint. This record qualifies the first single-process `orbitkv-engine`
composition root; it is not an HTTP, concurrency, latency, throughput, or
production result.

One initialized `ModelEngine` owned a `RuntimeSession`, per-class OrbitKV arenas,
and one compiled Luminal decoder on a dedicated execution thread. The test ran
three logical requests through the public server `Engine` contract:

- a 512-token fresh prompt with eight visible greedy output tokens;
- a 16-token fresh prompt whose first sampled token was a configured stop token
  and was not emitted as visible output;
- a 512-token request cancelled immediately after admission.

The first request produced
`[236743, 199, 236820, 34280, 236813, 208, 236820, 34280]`, matching the
previously qualified released-checkpoint reference prefix. Length, stop, and
cancellation all converged through request release and exact reclamation
acknowledgement. After each request, active requests, snapshots, pages, retiring
or quarantined pages, pending reclamations, page references, and reader pins
were zero.

The final same-source release-profile run finished in 29.99 seconds with warm
compiler and FlashInfer caches. This total includes graph construction/search
and is diagnostic, not a performance measurement.

## Reproduction

```bash
ORBITKV_MODEL_DIR=/workspace/models/gemma-3-270m-it \
ORBITKV_SEARCH_GRAPHS=2 \
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvshmem/13:/usr/local/cuda/lib64 \
cargo test --release --locked -p orbitkv-engine --features cuda \
  --test model_engine \
  released_hybrid_engine_streams_stops_cancels_and_drains \
  -- --ignored --nocapture
```
