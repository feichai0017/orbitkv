# oxide-infer-lab

`oxide-infer-lab` contains non-published hardware correctness programs
for permanent Oxide Infer providers. It depends on both product crates.
Neither product crate depends on it.

The crate groups code by validation responsibility:

```text
src/
|-- gates/         hardware correctness and lifecycle implementations
|-- benchmarks/    matched and tuning measurements
|-- support/       fixtures, comparisons, records, and reporting
|-- bin/           thin Cargo entry points
`-- lib.rs
```

Cargo binary names stay stable, but `src/bin` contains no operator
implementation. Shared deterministic fixtures prevent correctness and
performance runners from drifting to different input generation.

Run validation through the repository targets:

```bash
make cuda-test
make h20
make bench-ragged-graph-oxide
```

These executables are validation tooling, not product, engine, or serving
APIs. Correctness and performance remain separate entry points and claims.

The Graph benchmark fixes one long-GQA shape and measures one replay per
CUDA-event sample. Capture, planning, allocation, and correctness reads remain
outside the timed interval.
