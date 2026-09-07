# OrbitKV server

`crates/orbitkv-server/` is the Rust control plane for the composed engine. Its async `Engine`
trait accepts validated request intent and returns an ordered local event
stream. No public server type can name, allocate, retire, or recycle a physical
KV page.

The optional `vllm-frontend` feature reuses the pinned vLLM Rust crates for the
OpenAI HTTP routes, tokenizer, Hugging Face chat templates, and SSE output. A
small local transport maps only tokenized Add/Abort requests and token events
to the async `Engine`; unsupported sampling, LoRA, multimodal, logprob,
distributed, utility, and KV-transfer semantics are explicitly rejected.

The feature does not import PegaInfer's scheduler, KV cache, model execution, or
CUDA runtime. PegaInfer's useful architectural idea is the frontend protocol
bridge; OrbitKV's local engine still drives the `orbitkv` manager and the Luminal
executor. The default build has no vLLM dependencies.

The source-level HTTP launcher is `serve_openai`. A runnable product still needs
a concrete scheduler implementation of `Engine`; until that is connected and
tested over HTTP, this is a protocol-complete skeleton rather than an
end-to-end serving claim.

Build and test this optional boundary with:

```bash
cargo test --locked -p orbitkv-server --features vllm-frontend
```

With tokenizer assets from a released checkpoint, the ignored HTTP smoke test
can be run by setting `ORBITKV_MODEL_DIR` and selecting
`openai_completion_reaches_the_local_engine`. It proves HTTP request parsing,
tokenization, local Add transport, event translation, detokenization, and an
OpenAI response; its test engine deliberately does not execute a model.
