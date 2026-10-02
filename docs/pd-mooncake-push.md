# Retired OrbitKV P/D protocol

The custom vLLM split connectors, handshake state machine and HTTP proxy are
removed. Use [native P/D with TENT](pd.md) for current architecture, exact engine
revisions, launch configuration, composition and failure contracts.

The [pre-cutover protocol specification](https://github.com/feichai0017/orbitkv/blob/0e3668ebca0f7d80a049b834390098de7eaf963b/docs/pd-mooncake-push.md)
remains available for historical review. Its experiments remain under
`/root/orbitkv-artifacts/s1-evidence-20260929/legacy-results/runs/two-host-natural-20260928/vllm-pd-same-a100/`
and `vllm-pd-byte-probe/`. Retiring the implementation does not reclassify the
H20-to-A100 strict-output failure, establish RDMA support, or remove evidence.
