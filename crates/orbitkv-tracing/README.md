# OrbitKV tracing

Composable stage and Perfetto tracing for compilation and execution. This crate
is independent of CUDA and the compiler's graph types. `ORBITKV_STAGE_TRACE`
selects the output path for buffered stage events with target `orbitkv::stage`.

See [trace analysis](../../docs/stage-tracing.md) and
[source origin and licenses](../../docs/compiler-maintenance.md).
