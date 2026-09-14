# Rust 1.98 byte-conversion follow-up

GitHub CI for the [workspace migration](../workspace-integration-20260914/README.md)
used Rust 1.98, while local qualification used Rust 1.97.1. The new
`chunks_exact_to_as_chunks` Clippy lint rejected a fixed-width byte conversion.
The other jobs passed for the imported workspace.

Weight conversion now uses typed byte chunks, retains the incomplete-element
check, and no longer needs a fallible slice-to-array conversion. Reference
readers use the same typed API. One production file and five test files change;
model semantics, reference inputs, tolerance gates and graph rewrites do not.

## Validation

- Rust 1.98 CUDA and executor Clippy passes with all targets and warnings denied.
- Four weight-conversion tests pass, including all FP16/BF16 bit patterns,
  unaligned inputs, narrowing/rounding, empty inputs and incomplete elements.
- Formatting and source-layout checks pass.
- The rebuilt Rust 1.97.1 model binary passes **144 reference comparisons**
  across B1/B8 strict replay and profiling on one H20; every state drain passes.
  Maximum absolute logit error remains **0.6875** within the unchanged 1.0 gate.

These four processes reuse the migration's schema 11 decoder artifact and
preserve its hash. This is a correctness follow-up, with no new search or
serving-performance claim. The earlier record retains its original binaries
and source identities.

[Source identities](source.json) record the parent commit, the six changed
files, all 391 build inputs and this binary's hash. The source remained fixed
through qualification. [Model evidence](model.json) contains the per-process
comparisons; [checks](checks.json) records the lint and conversion gates.
