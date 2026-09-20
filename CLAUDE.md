# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OrbitKV is a framework-neutral state cache and physical planner for LLM
inference. Its current PegaFlow-derived data plane is validated with vLLM.
SGLang adapter contracts exist, but the executable HiCache backend remains an
M1 deliverable. Do not describe roadmap functionality as implemented.

## Build Commands

### Rust Core

```bash
cargo build              # Debug build
cargo build --release    # Release build
cargo test               # Run Rust tests
```

On CUDA 13 dev machines, pass `--no-default-features --features cuda-13,mooncake` to `cargo test`/`cargo clippy` (default `cuda-12` can fail with missing `libcudart` symbols).

### CI Checks (Local)

Run all CI checks locally before committing:

```bash
./scripts/check.sh       # Run fmt, typos, clippy, and cargo check
```

### Python Test Gates

| Gate | When to run | Command | Notes |
|------|-------------|---------|-------|
| Default unit | Every Python PR before review | `cd python && uv run --extra test pytest` | Must not start vLLM, `orbitkv-cache-manager`, or GPU runtime. Collection still imports deselected files, so top-level imports must be in `python[test]` or moved behind fixtures. |
| Source-only default | CI and dependency-boundary checks | `cd python && uv run --isolated --no-project --with pytest --with numpy --with 'requests>=2.26.0' pytest` | Proves default gate does not need torch, vLLM, CUDA, native extension build, or a running server. |
| Integration | Server/native/client/session lifecycle changes | `cd python && uv run --extra test pytest -m integration` | Requires built native extension, server binary, and GPU where the test uses CUDA IPC. |
| vLLM correctness E2E | Python test gates, vLLM connector, connector-visible cache semantics, save/load, query planning, or release-confidence changes | `cd python && uv run --extra test pytest -m e2e tests/test_vllm_e2e_correctness.py --model /data/models/Qwen3-4B --max-model-len 4096` | Merge-before gate: code author runs it, reviewer reruns it on the GPU machine. |
| Stress | Warm-hit pressure, pending unpin, scheduler/cache concurrency | `cd python && uv run --extra test pytest -m stress tests/test_vllm_warm_hit_stress.py --model /data/models/Qwen3-4B --max-model-len 2048` | Targeted single-GPU evidence, not default PR feedback. |
| Release smoke | Published wheel/image, loader path, installed console script, CUDA runtime | See `python/tests/README.md` | Validates final installed artifact, not the source checkout. |

Do not default to running all of `python/tests`. Current project taste is `uv` + pytest markers for Python and Cargo/CI for Rust; do not add an `xtask` wrapper until the gate contract is stable and repeated execution is the real bottleneck.

### Running Benchmarks

```bash
cargo bench --bench pinned_copy
cargo bench --bench uds_latency
```

## Architecture

### Workspace Design

All Rust packages live under `crates/`; the repository root is a virtual Cargo
workspace and intentionally has no `src/`. The Python distribution remains at
`python/` because it is both a maturin package and Python source tree.

1. **orbitkv-contract** (Rust): Framework-neutral state contract
   - State identity, byte-format compatibility, component kinds, page
     generations, and recovery bundles
   - Must not depend on vLLM, SGLang, CUDA, or transport implementations

2. **orbitkv-local** (Rust): Local control transport
   - Fixed 64-byte ABI over iceoryx2 request/response
   - UDS remains the planned bootstrap and file-descriptor passing path

3. **orbitkv-common** (Rust): Shared lightweight utilities
   - `logging.rs`: Unified log initialization (logforth-based)
   - `numa.rs`: NUMA topology detection and CPU affinity utilities
   - Depended on by all other crates to avoid heavy transitive dependencies

4. **orbitkv-core** (Rust): Core storage engine
   - `OrbitKVEngine`: Main engine managing GPU workers and KV cache storage
   - `storage/`: Modular block storage engine
     - `mod.rs`: `StorageEngine` — aggregates allocator, read cache, prefetch, write pipeline, SSD store, Mooncake remote fetch
     - `read_cache.rs`: Pin/unpin/consume operations on sealed blocks
     - `prefetch.rs`: Per-request SSD/remote prefetch state machine
     - `transfer_lock.rs`: Transfer lock manager — prevents LRU eviction during remote transfer
     - `write_path.rs`: Async insert worker thread for batched writes
   - `backing/`: SSD backing store
     - `ssd.rs`: `SsdBackingStore` coordinator
     - `ssd_cache.rs`: SSD-backed block storage
     - `uring.rs`: io_uring async I/O
   - `pinned_pool.rs` / `pinned_mem.rs`: NUMA-aware pinned memory allocator
   - `transfer.rs`: GPU-CPU transfer operations via CUDA
   - `cache.rs`: LRU cache for blocks
   - `gpu_worker.rs`: Per-GPU worker handling async operations
   - `internode/`: Cross-node communication
     - `metaserver_client.rs`: MetaServer registration, removal, query, and node heartbeat

5. **orbitkv-proto** (Rust): Protobuf definitions
   - gRPC service definitions built with prost/tonic

6. **orbitkv-server** (Rust): Cache Manager service
   - `endpoint/`: UDS/iceoryx2 inference endpoint
   - `orbitkv-core/src/internode/p2p_service.rs`: peer transfer gRPC service
   - `registry.rs`: Instance/worker registration
   - `http_server.rs`: HTTP health check and Prometheus metrics endpoint
   - `bin/orbitkv-router.rs`: P/D request router (coordinates P/D nodes; OrbitKV itself is a KV store)

7. **orbitkv-metaserver** (Rust): Cross-node block hash registry
   - `service.rs`: gRPC MetaServer service (insert/query block hashes)
   - `store.rs`: Multi-owner block hash store with TTL sweep (backed by DashMap)
   - Used for multi-node KV cache coordination — each Cache Manager registers its block hashes here

8. **orbitkv-transfer** (Rust): pinned upstream Mooncake Transfer Engine provider
   - `mooncake.rs`: Segment/BatchTransfer/notification wrapper for READ and WRITE

9. **python/** (Rust/PyO3 + Python): Python package (`orbitkv-llm` on PyPI)
   - `src/lib.rs`: PyO3 bindings exposing `OrbitKVEngine` and gRPC client
   - `orbitkv/vllm/`: canonical vLLM v1 connector
   - `orbitkv/connector/`: backward-compatible alias for `orbitkv.vllm`
   - `orbitkv/sglang/`: SGLang config and pool contracts; no runtime backend yet
   - `orbitkv/client/`: framework-neutral Cache Manager client exports
   - `orbitkv/ipc_wrapper.py`: CUDA IPC handle wrapper
   - CLI binaries: `orbitkv-cache-manager`, `orbitkv-metaserver` (installed via pip)

### Data Flow

```
vLLM/SGLang adapter <--control--> OrbitKV Cache Manager <--registered pages--> framework memory
                                    |
                             Pinned CPU Memory (KV cache storage)
                                   / \
                   SSD Cache (io_uring)  Remote Node (Mooncake READ)
                                              |
                                        MetaServer (block discovery)
```

### Key Concepts

- **Instance**: A model instance with specific num_layers and tp_size
- **Worker**: A tensor-parallel rank within an instance
- **Block**: Unit of KV cache storage, identified by content hash
- **Split Storage**: K and V segments stored separately for efficient batching
- **StateBundle**: Complete component set required to restore a logical boundary
- **Page Generation**: Identity guard that prevents stale references after page reuse

OrbitKV owns external replicas and transfer/storage policy. Frameworks retain
execution ownership during M0/M1. KV payload bytes must not be serialized into
gRPC; local data moves through registered CUDA IPC or shared-host pages, and
remote data moves through Mooncake over RDMA or TCP. See `docs/architecture.md` and
`docs/roadmap.md` for milestone-specific ownership boundaries.

## Code Conventions

### General

- Use English in comments
- Use `.venv` for Python virtual environment
- KV cache uses layer-first layout: all K blocks contiguous, followed by all V blocks

### Rust

- **Visibility**: Prefer `fn` (private) > `pub(crate)` > `pub`; use the minimal necessary visibility
- Prefer `NonNull` over `*mut` in unsafe code

### Python (3.10+)

- Use native generics (`list`, `dict`, `set`, `tuple`) instead of `typing.List`, `typing.Dict`, etc.
- Use PEP 604 union syntax (`X | Y`, `X | None`) instead of `typing.Union`, `typing.Optional`
- Logging: use `%s` formatting (`logger.info("x=%s", x)`) instead of f-strings to avoid evaluation overhead

## Testing Principles

- Do not add tests just for the sake of adding tests.
- Before adding or keeping a test, answer: would skipping this test materially reduce confidence to merge a PR in its trigger area? If not, delete it or keep it out of routine gates.
- Merge tests that protect the same contract; prefer table-driven cases with clear ids over copy-pasted methods.
- Delete tests with no clear consumer. A test that no dev, reviewer, CI job, release gate, or scheduled job uses is noise.
- Default Python tests must be dev friendly: fast, stable, no GPU, no vLLM, no native extension build, and failure messages that point to a Python contract.
- Heavy tests must declare their trigger: integration, e2e, stress, or release smoke. A heavy test without a trigger should not live in the main pytest surface.
- Stub and mock tests are allowed only for local contracts and must not shadow real runtime modules during integration/e2e collection.
- Prefer integration tests when they prove a real boundary that units cannot. Prefer units when they give faster, clearer feedback for local connector state machines.

## Compatibility & Durability

- We have a strict version handshake (`CARGO_PKG_VERSION` exact match at registration), so do NOT design for backward compatibility with older clients or older stored formats — bump the version and break freely.
- SSD cache is ephemeral: it is wiped on restart. Do NOT add migration, on-disk versioning, or cross-version SSD compatibility handling.

## Environment Variables

- `ORBITKV_ENGINE_ENDPOINT`: gRPC endpoint (default: `127.0.0.1:50055`)
- `ORBITKV_INSTANCE_ID`: Override instance ID
- `RUST_LOG`: Control Rust logging (e.g., `info,orbitkv_core=debug,orbitkv_server=debug`)

### Git commit message format

- We use commitizen commit message format.
- Do not commit directly to the master branch; create a feat/fix/chore/style/refactor/ci/... branch first.

## Key Files

- `crates/orbitkv-contract/src/lib.rs`: Framework-neutral state and recovery contracts
- `crates/orbitkv-local/src/lib.rs`: iceoryx2 local-control transport
- `crates/orbitkv-common/src/logging.rs`: Unified log initialization
- `crates/orbitkv-common/src/numa.rs`: NUMA topology detection and CPU affinity
- `crates/orbitkv-core/src/lib.rs`: Main OrbitKVEngine implementation
- `crates/orbitkv-core/src/storage/mod.rs`: StorageEngine (allocator, read cache, prefetch, write pipeline, Mooncake remote fetch)
- `crates/orbitkv-core/src/storage/read_cache.rs`: Pin/unpin/consume operations
- `crates/orbitkv-core/src/storage/prefetch.rs`: SSD/remote prefetch state machine
- `crates/orbitkv-core/src/storage/transfer_lock.rs`: Transfer lock manager for remote transfers
- `crates/orbitkv-core/src/storage/write_path.rs`: Async insert worker thread
- `crates/orbitkv-core/src/backing/ssd.rs`: SSD backing store coordinator
- `crates/orbitkv-core/src/internode/metaserver_client.rs`: MetaServer registration, query, and node heartbeat
- `crates/orbitkv-core/src/internode/p2p_service.rs`: peer transfer gRPC service
- `crates/orbitkv-metaserver/src/lib.rs`: MetaServer entry point and CLI
- `crates/orbitkv-metaserver/src/service.rs`: MetaServer gRPC service
- `crates/orbitkv-metaserver/src/store.rs`: Block hash store (LRU + TTL)
- `crates/orbitkv-transfer/src/mooncake.rs`: Mooncake Transfer Engine wrapper
- `python/src/lib.rs`: PyO3 bindings (Rust side)
- `python/orbitkv/orbitkv.pyi`: Type stubs for PyO3 bindings
- `python/orbitkv/vllm/scheduler.py`: vLLM scheduler-side connector
- `python/orbitkv/vllm/worker.py`: vLLM worker-side connector
- `python/orbitkv/sglang/`: SGLang adapter contracts; runtime backend pending M1
- `TODO.md`: Repository-wide milestone checklist
