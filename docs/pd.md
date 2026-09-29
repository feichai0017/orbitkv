# Prefill/decode transfer

P/D (prefill/decode disaggregation) places the prompt prefill and token decode
phases on different inference workers. The decode worker needs the prefill
worker's KV for the *same request* before it can continue. This is a request
handoff, not a cache lookup for a repeated prefix. A router or proxy also has
to coordinate the request; a KV transfer connector alone does not route it.

NIXL ([NVIDIA Inference Xfer Library](https://github.com/ai-dynamo/nixl)) is a
data-movement library used by inference systems. The pinned vLLM
release registers its own `NixlConnector`, `NixlPullConnector`, and
`NixlPushConnector` for P/D transfer. OrbitKV does not vendor or register a
NIXL connector. NIXL is not intrinsically vLLM-only: SGLang also documents
[P/D transfer with NIXL or Mooncake](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/pd_disaggregation.mdx).
The NIXL integration described here is
[vLLM's implementation](https://github.com/vllm-project/vllm/blob/main/docs/features/nixl_connector_usage.md).

| Path | Trigger | KV destination | Discovery/control | OrbitKV status |
| --- | --- | --- | --- | --- |
| OrbitKV external cache | Repeated-prefix lookup | Cache Manager DRAM/SSD, then engine HBM | Local index; global location index + exact source grant | GPU-validated locally; multi-node experimental |
| OrbitKV vLLM split P/D connectors | P-to-D request handoff | Decode worker's GPU KV pages | OrbitKV handshake and proxy; Mooncake TENT moves bytes | A100 same-host TCP output gate passes; H20→A100 byte gate passes, strict output gate fails |
| OrbitKV SGLang TENT adapter | P-to-D request handoff | Decode worker's GPU KV pages | SGLang 0.5.20 bootstrap/room protocol; OrbitKV Rust/TENT moves bytes | A100 same-host TCP output/restart gate passes; H20→A100 reuse passes, strict 64-token output gate fails |
| vLLM `NixlConnector` | P-to-D request handoff | Decode worker's GPU KV pages | vLLM's NIXL side channel and request router | Upstream vLLM connector, not OrbitKV code |

The OrbitKV `PdPrefillConnector` and `PdDecodeConnector` live in
`orbitkv.vllm.pd` and use Mooncake TENT to push KV directly from prefill to
decode. The old role-selecting `PdConnector` facade has been removed; each
process declares its role through its connector class instead of encoding it
again in `engine_id`. The handoff does not require an OrbitKV Cache
Manager, Catalog, or the remote-cache replica directory for that transfer.
See [the Mooncake P/D protocol](pd-mooncake-push.md) and the local
[`run_pd_local.sh`](../scripts/run_pd_local.sh) example. Its local proxy is
for P/D handoff and testing; it is not the planned KV-aware cache router.
The script requires an explicit model path, uses `.venv/vllm-release` by default,
and selects `PREFILL_GPU=0`, `DECODE_GPU=1` with `MC_FORCE_TCP=1`. Override
`VLLM_PYTHON` for another pinned environment. RDMA testing requires
`MC_FORCE_TCP=0` plus `PREFILL_NIC` and `DECODE_NIC`; the script does not infer
GPU/NIC affinity. Each child runs in its own process group, startup timeout is
fatal, and cleanup targets only those groups.

The decode connector can optionally report the physical handoff completion to
its node-local Cache Manager. Configure both
`orbitkv.pd.completion_observation_socket` and
`orbitkv.pd.completion_observation_instance_id` in the decode connector's
`kv_connector_extra_config`. The instance must already be registered on that
Manager and the reported device is taken from the actual decode KV tensor.
Standalone P/D remains the default and opens no Cache Manager connection.

The report covers decode wait enqueue through generation-fenced TENT terminal
completion. It carries the prefill control endpoint as a hashed source identity,
the nonzero transfer generation as freshness evidence (the TENT notification
scope generation in vLLM), destination
device, raw logical/wire bytes, target-layout fragments and terminal outcome.
Request IDs and cache keys never enter the Manager's cost index or metric
labels. Only admitted completed observations train estimates; failures,
cancellations and timeouts are diagnostic. This records one side of the future
choice but does not enable direct-restore-versus-handoff selection. SGLang now
reports only from its decode-owned commit boundary:
`DecodeTransferQueue._commit_transfer_to_req` runs after TP polling, metadata
validation and any required HiCache restore. Source-side `batch_transfer_sync`
return is never treated as DecodeReady. Terminal receiver failures report from
`failure_exception`; abort only marks the eventual terminal outcome.

The alternative is vLLM's built-in NIXL connector. The local
[`run_nixl_local.sh`](../scripts/run_nixl_local.sh) example uses that upstream
connector and a separate example proxy. You may also compose vLLM's NIXL
connector with `OrbitKVConnector` in `MultiConnector`: NIXL hands off the live
request, while OrbitKV can save completed blocks for reuse by later requests.
The two paths have different ownership and failure modes. See the
[deployment example](deployment.md).

## Why the adapter sizes differ

OrbitKV currently integrates at different boundaries in the two engines:

| Responsibility | vLLM split connectors | SGLang TENT adapter |
| --- | --- | --- |
| Request bootstrap, destination grants and handoff lifecycle | OrbitKV's scheduler/worker handshake and request state | SGLang's native bootstrap, rooms and request queues |
| Chunk/layer readiness and rank/layout mapping | OrbitKV's vLLM callbacks and layout plans | SGLang's native disaggregation implementation |
| Native registration, batch drain and memory lifetime | Shared Rust/TENT owner | The same shared Rust/TENT owner |
| Request routing | Example OrbitKV P/D proxy or external orchestrator | Separate SGLang router |

The checked OrbitKV tree contains 6,038 Python lines in `vllm/pd/`, including
794 proxy lines and 438 metric lines, versus 233 lines in `sglang/pd.py`.
These are source-line counts including comments, not equivalent feature or
complexity measurements. SGLang's adapter delegates its control lifecycle to
the pinned upstream engine; it has not eliminated that lifecycle. Completion
observation and plugin installation also live outside `sglang/pd.py`.

vLLM 0.29.0 does include its own Mooncake connector. OrbitKV chose a separate
split push protocol and native lifetime owner, so the extra code is not forced
by a lack of upstream P/D support. Before replacing it with a thinner adapter,
check the actual upstream transport contract, cancellation/drain behavior,
layout coverage and external-cache composition. Moving request bookkeeping to
Rust alone does not reduce the number of state machines. Keep one authoritative
handoff lifecycle; engine callbacks own GPU-page allocation and readiness,
while shared notification delivery, deduplication and native waiting belong to
the Rust transport owner. Remove obsolete forwarding and duplicate state only
with all consumers and fault/output gates updated.

## SGLang P/D over TENT

The pinned SGLang `0.5.20` release already owns the hard framework-specific
parts of disaggregation: bootstrap rooms, decode-page grants, TP/PP/CP mapping,
chunking, staging, request polling and terminal failure propagation. OrbitKV
does not copy that state machine. When `ORBITKV_SGLANG_TENT=1`, the SGLang
plugin installs `orbitkv.sglang.pd.SGLangTentTransferEngine` before SGLang
creates its process-wide transfer engine. SGLang's control plane remains in
place while its registered HBM/host regions and every payload batch are handed
to the same Rust TENT owner used by the rest of OrbitKV.

The upstream CLI value remains `--disaggregation-transfer-backend mooncake`
because that is SGLang's fixed backend routing key. It does **not** select the
legacy Transfer Engine when the OrbitKV opt-in is set: the wheel loads only
`libtent_shared.so`, and startup fails if TENT is unavailable. Registration is
RAII-owned in Rust. A synchronous SGLang batch returns success only after every
TENT task is terminal; timeout and partial-submit paths request cancellation,
drain the batch, retain the source/destination regions until the drain ends,
and invalidate the failed peer segment before SGLang marks the room failed.

For a same-host, two-GPU correctness run, enable TENT's TCP path and launch the
pinned SGLang processes with the OrbitKV plugin installed. The router is a separate package; the pinned SGLang source
uses `sglang-router 0.3.2`. Install it in the environment that runs the router:

```bash
uv pip install 'sglang-router==0.3.2'
export ORBITKV_SGLANG_TENT=1
export MC_FORCE_TCP=1

python -m sglang.launch_server --model-path /path/to/model \
  --host 127.0.0.1 --port 31000 --base-gpu-id 0 \
  --disaggregation-mode prefill --disaggregation-bootstrap-port 31500 \
  --disaggregation-transfer-backend mooncake

python -m sglang.launch_server --model-path /path/to/model \
  --host 127.0.0.1 --port 32000 --base-gpu-id 1 \
  --disaggregation-mode decode --disaggregation-bootstrap-port 31500 \
  --disaggregation-transfer-backend mooncake

python -m sglang_router.launch_router --pd-disaggregation --mini-lb \
  --prefill http://127.0.0.1:31000 31500 --decode http://127.0.0.1:32000 \
  --host 127.0.0.1 --port 30000
```

For RDMA, unset `MC_FORCE_TCP` and pass the appropriate
`--disaggregation-ib-device` value on both workers; the adapter resolves
SGLang's per-GPU mapping and supplies it as TENT's NIC filter. A transfer
timeout defaults to 30 seconds and can be changed with
`ORBITKV_SGLANG_TENT_TIMEOUT_S`. SGLang's optional failed-session background
probe must remain disabled for this revision (it is disabled by default): the
current TENT C ABI does not expose its peer-liveness probe, and OrbitKV rejects
startup if `SGLANG_ENABLE_FAILED_SESSION_PROBE=1`. Ordinary transfer failure,
cancellation and room teardown are supported and fail closed.

The external two-GPU correctness gate is:

```bash
cd python
../.venv/sglang-release/bin/python -m pytest -m e2e \
  tests/e2e/test_sglang_pd_e2e.py --model /path/to/model
```

It compares greedy P/D output with a monolithic SGLang control and asserts that
both P and D processes installed OrbitKV's Rust/TENT engine. The gate also
starts a Cache Manager and enables the OrbitKV external linker on both workers.
After a P/D restart, a continuation must recover past the last page boundary
that the original prefill alone could publish, proving that decode-produced
state was published and reused by the next prefill. Cache Manager load bytes
must increase. Run forced TCP first, then repeat the deployment on two hosts
with RDMA and external NIC counters before claiming GPUDirect.

### SGLang qualification on 2026-09-28

Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`, SGLang 0.5.20,
BF16, TP=1, 64-token pages, eager deterministic inference and forced TCP were
used with one Manager per worker. The 513-token natural-language prompt
produced 64 tokens with `ignore_eos=true`; both engines then restarted while
the Managers retained their caches. A follow-up appended those token IDs and
three additional tokens, and requested eight more output tokens.

| Deployment | Initial 64-token output vs A100 monolithic | Restarted 8-token continuation | Cache evidence |
| --- | --- | --- | --- |
| Two replicas on the same A100 | Exact token IDs and text | Exact token IDs and text | 576 cached tokens; 81 MiB Prefill H2D, including 9 MiB fetched from the Decode Manager; zero Decode H2D |
| H20 Prefill → A100 Decode | Differs at output index 20, after the first EOS, in `violet`/`Violet` casing | Exact token IDs and text | Same 576-token, 81 MiB H2D and 9 MiB remote-read evidence; zero Decode H2D |

The H20→A100 run passes the exercised restart/reuse path but **fails** the full
64-token equality gate. It is not a blanket heterogeneous-GPU correctness pass.
The same-A100 run drained request/source resources and passed both strict
output comparisons. Neither deployment establishes RDMA, throughput gains,
TP/PP behavior or mid-transfer fault recovery.

Raw outputs, launch commands, retained failure logs and driver snapshots are in
`benches/results/runs/two-host-natural-20260928/sglang-pd/` and
`sglang-pd-same-a100/`. The production fix keeps external hits out of Decode's
HiCache-only restore state machine; the maintained E2E also checks the actual
TENT engine-ready marker and passes the router's explicit bootstrap port.
The final normal release with bounded Restore partitioning repeats the
same-A100 gate successfully; its matching native hashes and outputs are under
`benches/results/runs/partitioned-restore-20260928/sglang-pd-same-a100/`,
with hashes in the parent directory.

To compose P/D with the external cache manually, point both workers at the same
node-local Manager and add these flags to both server commands:

```bash
export ORBITKV_SGLANG_ENDPOINT=unix:///run/orbitkv/orbitkv.sock

--radix-cache-backend orbitkv \
--enable-unified-cache-external-linker
```

The decode worker additionally needs
`--disaggregation-decode-enable-radix-cache`. OrbitKV rejects a composed P/D
configuration unless its live-transfer backend is the SGLang `mooncake` route
with `ORBITKV_SGLANG_TENT=1`; this prevents a deployment from silently sending
the live request through NIXL or the legacy Python Transfer Engine. P and D
share cache bytes only when their model, computation, TP/PP/CP and physical
layout identities match.

This first composition keeps the planner boundaries explicit. An external hit
is restored into prefill HBM, SGLang computes any missing suffix and hands the
request to decode through TENT, and completed decode state can be published for
a later prefill. Decode advertises only its resident HBM prefix and performs no
external lookup or restore; this keeps offloaded hits out of SGLang 0.5.20's
HiCache-only decode restore state machine. Its normal radix-cache retention and
write-through publication remain enabled. It does not yet let one cost decision choose between restoring
directly into decode HBM and routing through prefill; that requires comparable
completion targets and resource-admission evidence on both alternatives.

Neither OrbitKV's P/D paths nor the current global index provides production KV-aware
request routing. Production qualification still needs real multi-GPU and
cross-machine correctness, cancellation/restart tests, and throughput/latency
comparison against the vLLM NIXL baseline.
