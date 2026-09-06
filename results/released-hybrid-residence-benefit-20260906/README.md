# Released hybrid residence benefit

Status: passed narrow same-executor compiler-benefit qualification on H20. This
is a batch-one model/runtime result, not an HTTP-serving or SGLang comparison.

One configuration-driven Luminal graph was searched once, then reused for ten
paired alternating epochs. Every arm ran the released checkpoint's native 3
Full / 15 Sliding layer schedule with BF16 weights, a 512-token prefill, and 255
decode steps. The only independent variable was OrbitKV physical residence:
compiler-derived retirement versus request-lifetime retention.

All 256 output tokens matched across every arm. The first 13 stable tokens also
matched the independent Transformers reference. Reference token 14 has only a
0.125 BF16 top-two logit margin and is therefore not used as a discrete-token
gate across independently searched valid graphs. Every request
finished with a complete drain. At the final boundary, compiled residence used
48 Full pages plus 32 Sliding pages while the baseline used 48 Full plus 48
Sliding pages. Total live payload fell by 3,932,160 bytes, or 27.8%.
Against 10,205,184 semantic-live bytes, Retention Amplification fell from 1.387
to 1.002.

With a deliberately tight 35-Full/33-Sliding-page budget, compiled residence
advanced to boundary 560; request-lifetime residence exhausted its Sliding pool
at boundary 529 and therefore had a maximum successful boundary of 528. This is
32 additional sequence positions, or 6.1% over the baseline boundary, for the
same registered arena budget.

Median complete test-path time was 1.075590 s for compiled residence and
1.083926 s for request-lifetime residence, a 0.77% reduction. The paired mean
improvement was 11.633 ms with a two-sided 95% Student-t interval of
5.697-17.570 ms. Median prefill and accumulated model-decode times were nearly
equal; median measured manager time was 6.727 ms versus 13.529 ms.

These byte counts describe live manager payload inside equally preallocated
device arenas. They do not mean CUDA returned memory to its allocator. The
timing is one request in an integration test and does not qualify TTFT, TPOT,
p95/p99, concurrent admission, serving throughput, or comparison with SGLang.

## Reproduction

```bash
ORBITKV_MODEL_DIR=/workspace/models/gemma-3-270m-it \
ORBITKV_SEARCH_GRAPHS=2 \
ORBITKV_RESIDENCE_BENCH_EPOCHS=10 \
ORBITKV_RESIDENCE_DECODE_TOKENS=256 \
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvshmem/13:/usr/local/cuda/lib64 \
cargo test --release --locked -p orbitkv-executor --features cuda \
  --test model_execution \
  released_checkpoint_compares_compiled_and_request_lifetime_residence \
  -- --ignored --nocapture
```
