# Released hybrid lifecycle qualification

Status: passed on a real H20 with a released, unmodified Full+Sliding
checkpoint. This record qualifies narrow model correctness and the OrbitKV page
lifecycle; it is not a latency, throughput, capacity, or production result.

The checkpoint has 18 dense decoder layers: 3 Full layers and 15 Sliding layers
with a native 512-token window. One configuration-driven Luminal graph was
searched into decode and prefill buckets and used one stable OrbitKV-owned arena
per attention class. The run performed:

- independent parity probes at 1, 2, 4, and 16 prompt tokens;
- a 512-token prefill followed by 33 consecutive decode steps;
- exact comparison of all 34 greedy output tokens with a separate Transformers
  execution;
- Sliding retirement, exact acknowledgement, and generation reuse;
- release and complete drain of the long request;
- a 16-token prefill for a second request using a generation greater than one;
- token-boundary cancellation/release and a second complete drain.

After the long decode, the Full class had 35 active pages and the Sliding class
had 33. After each release, active requests, snapshots, pages, retirements,
quarantines, pending reclamations, request references, and reader pins were zero,
and every arena page was free.

The direct explicit-CSR causal-prefill regression in the pinned Luminal fork
also passed against its independent BF16 reference with maximum absolute error
`1.93e-4` at output index 813.

The one-time search and the single-run timings in `summary.json` identify this
qualification only. They were not collected with repeated matched arms and must
not be used as a performance claim.

## Reproduction

```bash
ORBITKV_MODEL_DIR=/workspace/models/gemma-3-270m-it \
ORBITKV_SEARCH_GRAPHS=2 \
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvshmem/13:/usr/local/cuda/lib64 \
cargo test --locked -p orbitkv-executor --features cuda \
  --test model_execution \
  released_hybrid_checkpoint_crosses_window_and_drains \
  -- --ignored --nocapture
```

The independent reference was generated outside the product with PyTorch
`2.10.0a0+git89fc82e.aml`, Transformers `4.55.2`, and the same checkpoint, dtype,
input token IDs `0..511`, and greedy decoding.
