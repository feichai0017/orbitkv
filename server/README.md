# OrbitKV server

`server/` is the Rust control plane for the composed engine. Its async `Engine`
trait accepts validated request intent and returns an ordered local event
stream. No public server type can name, allocate, retire, or recycle a physical
KV page.

The API layer will provide OpenAI-compatible HTTP, SSE, WebSocket, tokenization,
validation, cancellation, and backpressure above this boundary. Conversation
and tool-loop design may reuse protocol ideas from the Apache-2.0
`vllm-project/agentic-api`; OrbitKV does not inherit its upstream HTTP inference
proxy or external process launcher. The local engine drives OrbitKV and Luminal
inside one Rust process.
