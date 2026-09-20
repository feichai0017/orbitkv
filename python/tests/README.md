# Python Test Gates

`python/tests` is organized around developer workflow, not around how much code has accumulated. A test belongs here only when skipping it would materially reduce confidence to merge a PR for its trigger area. The default pytest invocation is intentionally small: it runs unit/helper contracts and excludes tests that start `orbitkv-cache-manager`, require CUDA, run vLLM, or create pressure workloads.

## What To Run

| Change area | Gate | Command | Failure boundary |
| --- | --- | --- | --- |
| Connector helper math, scheduler state, worker load failure handling, GPU registration | Default unit | `uv run --extra test pytest` | Python contract or local connector state-machine regression |
| Clean source-only Python changes, docs touching test layout, CI test dependency changes | Source-only default | `uv run --isolated --no-project --with pytest --with numpy --with 'requests>=2.26.0' pytest` | Default test accidentally depends on torch, vLLM, CUDA, or native extension |
| Server client, native extension, CUDA IPC registration, session lifecycle | Integration | `uv run --extra test pytest -m integration` | Server/native/GPU lifecycle regression |
| vLLM connector correctness, cache semantics, save/load/hit behavior, release candidate confidence | vLLM correctness E2E | `../.venv/vllm-release/bin/python -m pytest -m e2e tests/test_vllm_e2e_correctness.py --model /path/to/model` | Native prefix-cache control follows the same prompt plan; `long_warm` must load saved KV after vLLM restart. |
| SGLang direct GPU linker, CUDA IPC layout, or plugin registration | SGLang direct E2E | `../.venv/sglang-release/bin/python -m pytest -m e2e tests/test_sglang_direct_e2e.py --model /path/to/model` | Restores the same prompt after a RadixCache flush and after SGLang restarts against a live Cache Manager. |
| Warm-hit pressure, pending lease release, scheduler/cache concurrency | Stress | `uv run --extra test pytest -m stress tests/test_vllm_warm_hit_stress.py --model /data/models/Qwen3-4B --max-model-len 2048` | Real vLLM cache pressure regression |
| Wheel, loader path, installed console script, target CUDA runtime, published package | Release smoke | See Release Smoke | Packaging, loader, final artifact, or runtime contract regression |

A heavy test without a clear trigger should not be promoted into a routine gate. A generated fuzz workload used to live here, but it had no stable owner, cadence, data contract, or debugging path; it was removed from the main pytest surface instead of pretending to be a regular gate.

## Default Unit Gate

```bash
cd python
uv run --extra test pytest
```

This command must not start vLLM, `orbitkv-cache-manager`, or any GPU runtime. It still
imports every test module during pytest collection, so any top-level import used
by deselected heavy tests must be present in the `test` extra or moved behind a
fixture/helper boundary.

CI uses the source-only variant below so the unit gate does not compile the native extension or require CUDA:

```bash
cd python
uv run --isolated --no-project --with pytest --with numpy --with 'requests>=2.26.0' pytest
```

Runs:
- connector arithmetic and scheduler state-machine contracts (`test_combine_hashes.py`)
- connector load fault-tolerance unit tests with fake transport (`test_connector_fault_tolerance.py`)
- GPU registration layout with fake torch objects (`test_gpu_registration.py`)
- import-stub safety checks for default unit tests (`test_unit_stubs.py`)

This gate must collect and run without torch, vLLM, CUDA, external models, or a running OrbitKV server. Stub modules are allowed only inside tests that explicitly mock the connector boundary, and they must not shadow a real runtime during integration or E2E collection.

## Server Integration Gate

```bash
cd python
uv run --extra test pytest -m integration
```

Runs tests that start or require a local `orbitkv-cache-manager` but do not run vLLM:

- `test_local_control.py` proves Python-to-Cache Manager iceoryx2 ping, epoch fencing,
  UDS/memfd bootstrap, lifecycle registration/health with gRPC disabled,
  local publish, cold and warm `QueryBundle`, asynchronous
  restore with GPU byte verification, local lease release, and shutdown across
  a real process boundary;
- `test_cache_manager_client.py`
- `test_session_watcher.py`
- `test_sglang_direct_transfer.py` writes SGLang-shaped GPU pages through CUDA
  IPC, clears their source slots, and verifies a byte-exact restore into new
  slots. Run this when changing page registration or GPU transfer layout.

Requirements:
- built Python extension, for example `uv run maturin develop -r`
- a discoverable `orbitkv-cache-manager` binary from the installed package or Cargo target
- CUDA-capable GPU for tests that register real CUDA IPC tensors
- loader paths for the active Python and CUDA runtime, when the environment does not provide them globally


Set `ORBITKV_CACHE_MANAGER_BINARY` to an absolute path to test a specific build
(for example `target/debug/orbitkv-cache-manager`) instead of a previously
installed or release binary. Both integration and vLLM helpers honor this override.

## vLLM Correctness E2E Gate

Use the vLLM `0.29.0` release environment described in
[`python/README.md`](../README.md). The SGLang `0.5.20` environment is separate.
The default `uv run --extra test` environment intentionally has no GPU framework.

```bash
cd python
../.venv/vllm-release/bin/python -m pytest -m e2e tests/test_vllm_e2e_correctness.py \
  --model /path/to/model \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --max-model-len 4096
```

This is the main correctness E2E. It runs the same ordered prompt plan with
vLLM native prefix caching and with OrbitKV, then requires exact completion
equality at each plan step. The native control stays in one process to retain
its HBM cache; OrbitKV restarts vLLM between cold saves and warm loads. The
gate verifies that native `long_warm` had a prefix-cache hit, checks OrbitKV
save/hit/load activity, and requires OrbitKV `long_warm` to load KV bytes after
that restart.

This gate is required before merging PRs that change Python test gates, the
vLLM connector, cache semantics visible to the connector, save/load behavior,
query planning, or release confidence. The code author runs it before requesting
merge, and review reruns it independently on the GPU machine.

SGLang direct-linker changes require the SGLang direct E2E in the release
environment. It checks exact generated text, a nonzero external prefix hit
after `/flush_cache`, and an actual Cache Manager GPU load after the SGLang
process restarts while the Cache Manager remains alive. A separate namespace
provides a true cold inference control for the restarted process.

Requirements:
- vLLM installed in the active environment
- local model path, not an implicit network download
- GPU runtime compatible with the installed wheel variant
- enough free GPU memory for the model and configured context length

## Stress Gate

```bash
cd python
uv run --extra test pytest -m stress tests/test_vllm_warm_hit_stress.py \
  --model /data/models/Qwen3-4B \
  --max-model-len 2048
```

This is a targeted single-GPU vLLM scenario for warm-hit pressure. The checked profile uses `/data/models/Qwen3-4B`, `max_model_len=2048`, `gpu_memory_utilization=0.82`, `max_num_seqs=16`, and 12 concurrent repeated prompts; it has been validated on a 16GB GPU. Run it for cache warm-hit, pending lease release, scheduler/cache concurrency, or pressure-profile changes. It is not a default PR gate.

## Release Smoke

Release smoke validates the final installed package, not the source checkout. It should use a clean non-editable environment and record Python libdir, `PYTHONHOME`, `PYTHONPATH`, CUDA runtime path, package name/version, GPU, model path, and metrics excerpt.

Minimum checks:
- `orbitkv-cache-manager --help`
- minimal installed `orbitkv-cache-manager` startup and `/health` 200
- vLLM + `OrbitKVConnector` with `/v1/models`, one completion, one repeated long prompt, and non-zero save/load/hit/HLL metrics
