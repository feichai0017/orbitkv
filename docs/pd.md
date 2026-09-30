# Prefill/decode transfer

P/D moves a live request's KV from its prefill worker to its decode worker.
External caching saves completed state for later requests. OrbitKV keeps these
as independent paths: the engine and its router own P/D, thin Python backends
adapt TENT, and cache connectors own historical reuse.

The native TENT profile requires the **experimental engine revisions below**.
Official dependency pins remain vLLM 0.30.0 and SGLang 0.5.20 for ordinary cache
serving; they do not supply these factories and lifecycle fixes. Local
qualification and independent acceptance are tracked in
[the completion plan](completion-plan.md#s54--native-pd-lifecycle-with-tent-payloads).

| Responsibility | vLLM | SGLang |
| --- | --- | --- |
| Routing | `vllm-router==0.1.15` | `sglang-router==0.3.2` |
| Bootstrap, allocation, scheduling, readiness and release | Native `MooncakeConnector` | Native disaggregation queues and rooms |
| Payload adapter | `orbitkv.vllm.transport.TentTransferEngine` | `orbitkv.sglang.pd.SGLangTentTransferEngine` |
| Memory registration, native completion and drain | Shared Rust/TENT owner | Shared Rust/TENT owner |
| Historical cache | `OrbitKVConnector` through `MultiConnector` | `UnifiedRadixCache` + `OrbitKVLinker` |
| P/D observations | Native connector statistics | Public `PDTransferEvent` callback |

The custom `PdPrefillConnector`, `PdDecodeConnector`, handshake state machines,
HTTP sender and proxy are retired. The partial-tail external-cache extension
is also removed: native P/D handles the live prompt tail, while the independent
cache stores complete engine blocks. Historical contracts and evidence remain
linked from [the retired protocol page](pd-mooncake-push.md).

## Native P/D with an explicit TENT backend

This separation follows the
[LMCache 0.5.5 MP recipe](https://github.com/LMCache/LMCache/blob/05a013b29da78cf2321b9b46ec5039dde2fb0bb0/docs/source/mp/disaggregated_prefill.rst).
It does not copy cache policy or router state into a second OrbitKV controller.

| Component | Required source |
| --- | --- |
| vLLM native P/D | [`fc4fc427db66b78c37da6bf071e542597447da0e`](https://github.com/feichai0017/vllm/commit/fc4fc427db66b78c37da6bf071e542597447da0e), based on main `91dab0eb`; includes factory, receive drain, shutdown and composition fixes |
| SGLang native P/D | [`d520742126f52f7bfef43079151059def1381098`](https://github.com/feichai0017/sglang/commit/d520742126f52f7bfef43079151059def1381098), based on official 0.5.20 `94602c9c`; includes factory, drain fencing and public observations |
| vLLM router | [`0.1.15`](https://github.com/vllm-project/router/tree/1fbcde7443d75b36befb61bc081f64c2a1f13a4b) |

Both workers must run the same selected revision. Build/install that engine
using its instructions, then install the OrbitKV wheel without an engine extra:
installing `[vllm]` or `[sglang]` would replace the experimental engine with the
official release pin. No engine submodule or optional release pin is advanced
by this P/D profile.

The patches adapt existing upstream work:
[vLLM #52954](https://github.com/vllm-project/vllm/pull/52954) filters unexpected
send completions; [#43836](https://github.com/vllm-project/vllm/pull/43836)
allows a child without a Prometheus exporter. Empty-pull completion is also
tracked by [#59347](https://github.com/vllm-project/vllm/pull/59347).
The SGLang patch adapts [#38961](https://github.com/sgl-project/sglang/pull/38961)
and adds per-attempt ACK tokens, timeout quarantine, partial-submit drain and
public observations. A fork commit is not an upstream merge or released API.

### vLLM setup

Select this configuration on P, and change `kv_role` to `kv_consumer` on D:

```json
{
  "kv_connector": "MooncakeConnector",
  "kv_role": "kv_producer",
  "kv_connector_extra_config": {
    "mooncake_protocol": "tcp",
    "transfer_engine_factory": "orbitkv.vllm.transport.TentTransferEngine"
  }
}
```

Let the engine create a fresh `engine_id` at each start. Set `MC_FORCE_TCP=1`
and `VLLM_MOONCAKE_BOOTSTRAP_PORT=8998` on both workers. After P and D listen on
8100 and 8200, run:

```bash
vllm-router --vllm-pd-disaggregation --kv-connector mooncake \
  --prefill http://127.0.0.1:8100 8998 --decode http://127.0.0.1:8200 \
  --host 127.0.0.1 --port 8000
```

[`scripts/run_pd_local.sh`](../scripts/run_pd_local.sh) launches this profile
on two local GPUs using `VLLM_PYTHON` (default `.venv/vllm-native-pd/bin/python`)
and `VLLM_ROUTER` (default `vllm-router` on `PATH`). It defaults to TCP.
Shutdown waits for the native workers; unresolved drain can keep it waiting.
`MC_FORCE_TCP=0` unsets the native variable and selects `mooncake_protocol=rdma`;
set `PREFILL_NIC`/`DECODE_NIC` for that separate qualification. TENT treats any
presence of `MC_FORCE_TCP`, including `0`, as forcing TCP.

For cache composition, wrap the native configuration and the ordinary
`OrbitKVConnector` configuration in native `MultiConnector`, with outer
`kv_role=kv_both`. The cache child uses `kv_role=kv_both` and
`orbitkv.mode=read_write`. Its socket, identity and layout settings remain
those of the [ordinary cache adapter](adapters.md).

Native selection chooses the first child advertising a load. Cache-first can
restore a historical prefix and compute the remainder locally; P/D-first
selects the live handoff. Only that selected child receives destination blocks.
An unselected cache query releases its leases. An unselected P/D child sends an
empty cleanup pull to retire the producer's source without writing or reporting
a decode completion. Saves can go to both owners because they read completed
engine state. Publication is best effort: a final asynchronous output callback
can expose a full-block hash after the last scheduled save. This profile does not combine two concurrent writers for disjoint
parts of one load or consume the proposed piecewise-prefix protocol.

### SGLang setup

Install the selected patched engine and `sglang-router==0.3.2`, then set:

```bash
export SGLANG_MOONCAKE_TRANSFER_ENGINE=orbitkv
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

The CLI routing key remains `mooncake`; the explicit factory selects OrbitKV's
Rust TENT implementation. Missing factory/public-callback APIs fail startup.
The adapter enables deferred decode release and rejects explicitly disabling it.
For RDMA, unset `MC_FORCE_TCP` and select `--disaggregation-ib-device`; GPU/NIC
routing still requires independent measurement. `ORBITKV_SGLANG_TENT_TIMEOUT_S`
defaults to 30 seconds. Failed-session probing remains disabled because the
consumed TENT ABI does not supply that probe.

For cache reuse, add `--radix-cache-backend orbitkv` and
`--enable-unified-cache-external-linker`, configure `ORBITKV_SGLANG_ENDPOINT`,
and enable `--disaggregation-decode-enable-radix-cache` on D. P may restore
historical state. D publishes completed state but does not start external
restores into live P/D destinations. A later P can reuse D-produced state after
both inference processes restart and the Manager retains its cache.

## Failure and lifetime contract

- Native allocation and request lifecycle remain the sole page-release authority.
  SGLang public callbacks are observations: callback exceptions disable the
  observer without changing readiness or release.
- vLLM retains receive destinations until every participating producer reports a
  terminal result. Timeout alone cannot finish a receive. Empty cleanup pulls
  never produce a scheduler completion. Shutdown closes admission, drains queued
  work and waits for synchronous native writes before releasing resources.
- SGLang retains aborted P source pages while any chunk is outstanding and holds
  D destinations until every notified writer returns its per-attempt ACK token.
  Duplicate or old ACKs cannot satisfy a new attempt. Partial layer submission
  drains all already-submitted futures, including siblings of a failed task.
- A SGLang hold timeout logs quarantine and retries ABORT using the same tokens;
  it does not free pages. Memory unload rejects outstanding holds. A partial
  transport failure isolates the native peer session; recreate the affected
  workers before admitting writes through a new session.
- Permanent peer loss can leave pages held. This profile does not claim automatic
  recovery through timeout, process disappearance, or lease expiry. Cross-host
  revocation and independent-failure-domain qualification remain in S3.

The six private SGLang P/D observation Hooks are removed. The public callback
reports decode admission, commit, failure, cancellation after drain, and continued
quarantine. Optional Manager observations use the transfer UUID and layout byte
count; unavailable NIC pressure is not measured by this callback. Ordinary
cache admission/graph Hooks and vLLM's cache restore boundary remain separate
S5.3 work.

## Reproduction and limits

Run from `python/` in the corresponding experimental environment:

```bash
python -m pytest -m integration tests/integration/test_native_pd_lifetime.py \
  --basetemp /var/tmp/orbitkv-native-pd/lifetime-001
python -m pytest -m e2e tests/e2e/test_vllm_native_pd_e2e.py \
  --model /path/to/dense-model --basetemp /var/tmp/orbitkv-native-pd/vllm-001
python -m pytest -m e2e tests/e2e/test_sglang_pd_e2e.py \
  --model /path/to/dense-model --basetemp /var/tmp/orbitkv-native-pd/sglang-001
```

The lifetime gate uses real GPU tensors, TENT and the native vLLM control channel
for delayed completion, partial write, cancellation, shutdown and page reuse.
Model gates compare exact outputs with monolithic controls, check physical
cache-load bytes and restart reuse, and require observed native preemption or
retraction in their pressure cases. The SGLang test-only plugin blocks an actual
GPU write and records public quarantine/release events; it is excluded from the
wheel. Keep all run directories, failures and controls outside the checkout.

Evidence and exact source/native hashes:
`/root/orbitkv-artifacts/native-pd-cutover-20260930/HANDOFF.md`.
The exercised profile is A100, dense Qwen3-8B, BF16, TP=1/PP=1, eager, same-host
TCP. It does not qualify cross-host HA, GPUDirect RDMA, heterogeneous GPUs/ranks,
hybrid P/D, native GDS, or a performance advantage. NIXL remains an upstream
alternative; OrbitKV ships no NIXL connector. See the
[comparison launcher](../scripts/run_nixl_local.sh) for that separate experiment.

### Historical SGLang qualification on 2026-09-28

Earlier class-substitution runs are preserved under
`/root/orbitkv-artifacts/s1-evidence-20260929/legacy-results/runs/two-host-natural-20260928/`.
Same-A100 initial and restarted outputs matched; the H20-to-A100 initial output
failed strict equality at token 20 even though restart reuse passed. These
results do not qualify the replacement factories or fault lifecycle. The full
historical record remains in the
[pre-cutover documentation](https://github.com/feichai0017/orbitkv/blob/0e3668ebca0f7d80a049b834390098de7eaf963b/docs/pd.md#historical-sglang-qualification-on-2026-09-28).
