# OrbitKV compiler

Symbolic tensor graphs, shapes, primitive IR, egglog equivalence compilation,
and reusable program-search utilities. The compiler builds a search space;
backend runtimes select and execute programs. The reference runtime supports
CPU correctness tests without CUDA.

```rust
use orbitkv_compiler::prelude::*;

let mut graph = Graph::new();
let input = graph.tensor(3);
let output = (input + 1.0).output();
let mut runtime = graph.compile(ReferenceRuntime::default(), CompileOptions::default());
runtime.set_data(input, vec![1.0, 2.0, 3.0]);
runtime.execute(&graph.dyn_map);
assert_eq!(runtime.get_f32(output), vec![2.0, 3.0, 4.0]);
```

See [architecture](../../docs/compiler.md) and
[source origin and licenses](../../docs/compiler-maintenance.md).
