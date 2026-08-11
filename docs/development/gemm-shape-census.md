# Dense GEMM shape census

Kernel selection starts from an observed model workload. A microbenchmark
shape alone is not sufficient.

The first census targets the paired Mistral.rs Qwen2.5-1.5B proof of concept.
The engine records each resolved unquantized dense-linear call before backend
selection. In the current Qwen path, these calls later use Mistral.rs GEMV or
Candle CUDA matmul.

The record is a call census, not a kernel trace. `host_calls` counts successful
linear dispatches. It does not count CUDA kernels, Graph replays, retries, or
warmup work. The capture must label prefill and decode from engine metadata.
It must not infer the phase from `M`.

Version 1 rejects runs that enable CUDA Graphs. A Graph replay does not return
through the Rust linear dispatch hook, so a host-call census would undercount
effective execution frequency.

Version 1 also admits one request and one unchunked prefill forward. The
`decode_forward_steps` field counts successful model decode forwards. It does
not count generated tokens across a batch.

## Request identity

The request digest covers the UTF-8 bytes of `canonical_request`. The producer
uses sorted keys, no insignificant whitespace, and unescaped UTF-8 characters.

The object contains ordered role/content messages and the generation settings
that affect the fixed smoke run.

## Record contract

The schema requires:

- a clean producer source and immutable source hashes
- exact model, hardware, toolchain, and workload identity
- local tensor shapes and strides after tensor-parallel sharding
- activation and weight dtypes, layouts, transpose flags, and post-ops
- the observed engine path and exact integer call count
- complete, unsampled, untimed capture

Run the local validator and summary tool:

```bash
python3 tools/gemm/shape_census.py validate run.jsonl
python3 tools/gemm/shape_census.py summarize run.jsonl > summary.json
```

The summary ranks shapes by call count, then FLOPs. It reports both call and
FLOP coverage because a frequent small-M launch and a large prefill GEMM have
different optimization value.

The tool does not choose an algorithm. A reviewer uses the validated profile
to freeze the first Loom admission contract. The first provider candidate must
then pass correctness, lifecycle, sanitizer, Graph, matched operator, and real
model gates.

For Mistral.rs decode, `M <= 8` and even `K` currently select its custom CUDA
GEMV before cuBLASLt. A Loom small-M algorithm must beat both that engine path
and the cuBLASLt baseline. Winning only the vendor microbenchmark is not enough
for engine admission.

Do not combine records when source, model, workload, hardware, capture
boundary, dtype, layout, transpose, or post-op metadata differs. Do not present
a single-request census as a production traffic distribution.

The shape census profiles the engine baseline. It does not compile or execute
a Loom kernel. Record cuda-oxide source, backend, and codegen provenance in the
later Loom provider benchmark, not in this host-dispatch record.
