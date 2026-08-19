# H20 Dense SGLang Binding

OrbitKV restored a 4K logical pure-SWA Prefix as its compiler-proven 1K live
tail and decoded 128 tokens through typed in-process FFI, CUDA-event views, and
online retirement certificates. Seven Dense, Capsule-only, and full-capacity
reference rounds produced the same output digest.

The Dense path used 3,168 physical token slots versus 32,768 for the reference,
a 90.33% reduction. Its median time was 1.075x Capsule-only; this is not a
performance-win claim. OrbitKV owns compiled page binding and lifecycle on this
path, while SGLang still supplies physical allocation, ReqToToken, KV tensors,
and kernel views. It is therefore not yet a complete drop-in allocator
replacement.
