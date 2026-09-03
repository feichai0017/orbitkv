# OrbitKV server

`server/` is the Rust control plane for the composed engine. The public
`Engine` trait accepts only request and batch intent. No server type can name,
allocate, retire, or recycle a physical KV page.

The existing SGLang Rust server is a useful source for OpenAI HTTP handling,
tokenization, SSE, request validation, cancellation, and backpressure. Its
current Python-scheduler ring and PyO3 boundary are intentionally not part of
this crate. Those portions will be replaced by a local engine implementation
that drives OrbitKV and the Luminal executor in one Rust process.
