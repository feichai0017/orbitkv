# Prefill/decode transfer and NIXL

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
| OrbitKV external cache | Repeated-prefix lookup | Cache Manager DRAM/SSD, then engine HBM | Local index; experimental remote Catalog + peer lease | GPU-validated locally; multi-node experimental |
| OrbitKV `PdConnector` | P-to-D request handoff | Decode worker's GPU KV pages | P/D request handshake and proxy; Mooncake Transfer Engine moves bytes | Experimental vLLM adapter |
| vLLM `NixlConnector` | P-to-D request handoff | Decode worker's GPU KV pages | vLLM's NIXL side channel and request router | Upstream vLLM connector, not OrbitKV code |

The OrbitKV P/D connector lives in `orbitkv.vllm.pd` and uses Mooncake to push
KV directly from prefill to decode. It does not require an OrbitKV Cache
Manager, Catalog, or the remote-cache replica directory for that transfer.
See [the Mooncake P/D protocol](pd-mooncake-push.md) and the local
[`run_pd_local.sh`](../scripts/run_pd_local.sh) example. Its local proxy is
for P/D handoff and testing; it is not the planned KV-aware cache router.

The alternative is vLLM's built-in NIXL connector. The local
[`run_nixl_local.sh`](../scripts/run_nixl_local.sh) example uses that upstream
connector and a separate example proxy. You may also compose vLLM's NIXL
connector with `OrbitKVConnector` in `MultiConnector`: NIXL hands off the live
request, while OrbitKV can save completed blocks for reuse by later requests.
The two paths have different ownership and failure modes. See the
[deployment example](deployment.md).

SGLang has its own disaggregated-serving facilities (including NIXL), but OrbitKV currently
provides **only** an SGLang external-cache linker. It does not provide a
SGLang P/D adapter or NIXL connector. P/D support for SGLang would require a
separate integration against SGLang's handoff protocol and a tested recovery
contract.

Neither OrbitKV's P/D path nor the current Catalog provides production KV-aware
request routing. Production qualification still needs real multi-GPU and
cross-machine correctness, cancellation/restart tests, and throughput/latency
comparison against the vLLM NIXL baseline. Historical benchmark figures from
earlier PegaFlow-based experiments are not OrbitKV release results.
