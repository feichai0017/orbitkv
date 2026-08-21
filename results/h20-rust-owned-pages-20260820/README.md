# H20 Rust-Owned SWA Pages

OrbitKV selected generation-checked physical SWA page IDs through Dense FFI
v2; SGLang consumed those IDs with its existing allocation kernels.

- 4K Prefix, 1K live tail, 128 continuation tokens, seven rounds.
- Output digest matched the pre-page-ownership path.
- Final pool: 198 free, 0 active, 0 retiring pages.
- Paired steady-state runtime ratio: 1.0067x; no speedup claim.
- Both paired paths include the same ~525 ms Holt lookup.

SGLang still owns ReqToToken, KV tensors, and attention kernel views.
