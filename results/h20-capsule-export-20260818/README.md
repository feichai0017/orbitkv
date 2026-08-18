# H20 Capsule Export Validation

A real 496,060,688-byte TinyMistral checkpoint executed through pinned SGLang
and OrbitKV owning mode. Before SGLang released the request, OrbitKV copied the
64-token GPU KV through `get_cpu_copy`, encoded 12 layers of BF16 K/V without
pickle, and published a 787,680-byte payload through Holt 0.9.2.

The same token prefix restored through Holt longest-prefix lookup. Payload
SHA-256 and all `[64, 2, 128]` K/V shapes matched.

Three paired export/no-export rounds produced the same output digest. Median
export overhead was 1.55%; one pair was a 16.2% outlier. This is export-only
evidence: hydration and TTFT improvement are not implemented or claimed.
