# Source dependency policy

Third-party inference code is used as readable, pinned source. It is never
copied into this repository without provenance.

## Rules

1. Every Git source has an immutable commit in
   `third-party/sources.lock.toml`. Tags and branches are descriptive only.
2. Major inference projects are Git submodules so their source is present,
   searchable, patchable, and independently buildable in every developer
   checkout. `sources.lock.toml` adds role and structural checks to Git's commit
   pin.
3. Rust dependencies use crates.io versions or Git revisions and are resolved
   by the committed `Cargo.lock`. Python integrations are installed editable
   from the checked-out submodules.
4. A dependency is wrapped behind a provider or adapter owned here. Core schema
   crates cannot import CUDA, PyTorch, SGLang, or vendor APIs.
5. Local changes are maintained as a small patch series against the pinned
   revision. Forks must record upstream commit, patch purpose, and removal gate.
6. Updating a revision requires contract tests, qualification replay, license
   review, and a performance comparison.

## Isolated environments

The source trees share a repository but not one Python environment:

- `sglang`: SGLang, its compatible FlashInfer and DeepGEMM sources, and our
  in-process SGLang plugin; `sglang-kernel` is built from SGLang's own
  `python/sglang/kernels/aot` source and replaces the wheel dependency;
- `autodeploy`: TensorRT-LLM v1.2.0 plus our registered AutoDeploy transforms;
- `tensorrt-source`: complete TensorRT-LLM C++/CUDA wheel build, defaulting to
  SM90 on this H20 (`ALETHEIA_CUDA_ARCHITECTURES` overrides it);
- `kernels`: FlashInfer and DeepGEMM qualification work without the serving
  framework.

This avoids pretending that two large stacks with independent PyTorch/CUDA
constraints can safely share one lock. Deployment artifacts, not Python object
graphs, cross the environment boundary.
Native builds default to eight parallel jobs on this development machine;
`ALETHEIA_BUILD_JOBS` provides an explicit override.

## SGLang boundary

The integration uses SGLang's general `sglang.srt.plugins` hook framework in
the SGLang process. It observes the real
`TpModelWorker.forward_batch_generation` path while SGLang continues to own
requests, batches, KV pools, sampling, and workers. It does not start a second
server. The first trace measures host-side batch execution latency; GPU evidence
will come from CUDA events and profilers in qualification, not from this timer.
Kernel choices initially use SGLang's existing backend flags and source
wrappers. A future plan-controlled model runner requires either a narrow
upstream hook or an explicit patch, added only after trace evidence identifies
the required contract.

## AutoDeploy boundary

TensorRT-LLM AutoDeploy remains an offline candidate generator and NVIDIA
baseline. Our extension registers directly with its `TransformRegistry`; future
extensions may also use `CompileBackendRegistry`. AutoDeploy may export graph
inventories and artifacts into qualification, but TensorRT-LLM is not imported
inside the SGLang serving process.

The fast `autodeploy` environment edits Python source in place while using the
upstream-supported precompiled native substrate. It is not called a full source
build. `tensorrt-source` performs the full native build through upstream's
`scripts/build_wheel.py`.

AutoDeploy itself has no external setuptools-plugin discovery contract. Local
candidate tools must construct it through
`aletheia_autodeploy.registered_optimizer()`, which registers our transforms before
instantiating upstream `InferenceOptimizer`. This call order is deliberate and
covered by integration tests.
