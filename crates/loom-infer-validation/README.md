# loom-infer-validation

`loom-infer-validation` contains non-published hardware correctness programs
for permanent Loom Infer providers. It depends on both product crates; neither
product crate depends on it.

The crate owns:

- H20 case orchestration and declared numerical limits.
- shared finite comparisons, bit mismatch counts, and stable digests.
- stable machine-readable `gate`, `case`, and `status` output.

Run validation through the repository targets:

```bash
make cuda-test
make h20
```

These executables are correctness and lifecycle gates. They are not benchmark,
engine, or serving APIs.
