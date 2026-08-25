# Qwen3.8-27B config provenance

- Repository: `Qwen/Qwen3.8-27B`
- Revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Source: `https://huggingface.co/Qwen/Qwen3.8-27B/raw/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json`
- Source SHA-256: `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab`
- Canonical JSON SHA-256: `1c8aa924e850dfad57c055a01e8663f277091c7760c1d444da88eb6079befed3`
- Canonicalization: parse as `serde_json::Value`, serialize with
  `serde_json::to_vec`, and terminate it with one newline. With the default
  `serde_json` map representation this recursively sorts object keys; its
  number formatting is part of this Rust-side contract.

`config.json` is a normalized, content-equivalent copy used only for compiler
tests. It proves plan compilation from the official model configuration; it
does not prove model loading, H20 execution, output correctness, capacity, or
performance.
