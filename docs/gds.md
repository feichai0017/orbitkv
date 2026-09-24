# GPU storage

OrbitKV defaults to automatic SSD backend selection when SSD caching is configured.
Complete state groups can be written
from registered engine GPU pages to SSD, and selected SSD state can be restored
to those pages. Both directions use bounded, registered GPU staging.
vLLM and SGLang share this Rust implementation and their existing cache API.
`--ssd-backend auto` tries native cuFile and uses io_uring when unavailable.

Calling cuFile does **not** prove native GPUDirect Storage: its compatibility
mode performs host staging internally. Native-mode probes in this H20 container
have failed driver initialization (5001) or file registration (5027); automatic
selection falls back to io_uring. Native GDS throughput remains unqualified.

## Automatic selection

The Manager owns this decision; engine adapters use the same cache API.

| Mode | Behavior |
| --- | --- |
| `auto` (default) | Try cuFile on ext4/XFS with sufficient alignment capacity; disable both allow/force compatibility through NVIDIA's parameter API before driver initialization. Missing API/library, driver, mount or file-registration support selects io_uring. |
| `uring` | Use host staging; never load cuFile. Useful as a reproducible control. |
| `cufile` | Require cuFile initialization and follow NVIDIA configuration, including explicitly enabled CPU compatibility for functional development. Operation failures remain errors. |

Auto selection is currently cache-wide: every configured shard must register;
otherwise the whole SSD cache uses io_uring. It requires cuFile's parameter API
to enforce the native-only initialization policy. An already initialized
compatibility-capable driver is not accepted as native evidence. Other filesystem
types need separate qualification and an explicit `cufile` selection.

On a GPU storage buffer or I/O failure in `auto`, the Manager stops admitting
new cuFile work and switches subsequent operations to io_uring until restart.
Submitted work retains its file/extent/page ownership and completes or reports
its error; changing the backend does not revoke DMA or silently replay an
already submitted restore. The failing operation retains normal error semantics.
Logs identify the selection and fallback reason; `orbitkv_ssd_backend_fallbacks_total`
counts the transition. Per-mount/per-device isolation, retries after a cooldown,
and online latency-based selection remain future work.

This is capability and failure adaptation. Successful initialization does not
establish that cuFile beats io_uring for a workload, or replace native-path
statistics. SSD paths/capacity must still be configured explicitly.

## Data flow and ownership

| Operation | Path |
| --- | --- |
| DRAM hit | Pinned DRAM → engine GPU pages |
| SSD demand hit with `cufile` | SSD → registered GPU staging → engine GPU pages |
| Speculative prepare/warmup | SSD → pinned DRAM; GPU restore after consumption |
| Complete state group in one Publish | Engine GPU → registered GPU staging → SSD; also retain a hot DRAM copy |
| Fragmented/multi-writer Publish | Engine GPU → pinned DRAM; seal complete group → io_uring → SSD |
| Shared-cache remote recovery | Mooncake → pinned DRAM → engine GPU pages |

Candidate discovery remains metadata-only. A selected demand query acquires an
SSD extent lease instead of allocating and reading a host block. The compiled
recovery requirements still select the necessary windows and checkpoints. The
lease pins the source against ring overwrite until all query/GPU interests are
released. Cancellation, expiry and session teardown release unconsumed interests;
submitted restores keep their interests until completion.

Rust validates source segment sizes, slot offsets and GPU destinations before
issuing I/O. Adjacent ranges are merged into reads of at most **8 MiB** with
**4 KiB** aligned boundaries. Existing 512-byte storage segments can cause edge
overread; only actual component bytes are scattered into engine pages. Large
recurrent checkpoints are split across the bounded staging buffer.

Each instance/device lazily creates one SSD worker and one registered
8 MiB GPU buffer. It has its own CUDA stream, so synchronous disk I/O does not
block the ordinary DRAM restore or save worker. GPU-backed SSD reads and writes
share this lane and run sequentially; read preemption is not implemented. Hardware bandwidth and GPU
scheduling remain shared. Reserve this extra HBM in the engine's memory budget;
it is outside `--pool-size`. The buffer is reused until the worker drains. Query
reservations still charge logical source bytes under the existing query budget,
even when those bytes remain on SSD.

Before GPU writeback, Rust reserves an unpublished, 4 KiB aligned SSD extent.
Each chunk gathers only valid source bytes and zeroes padding, then completes
cuFileWrite before reusing staging. The complete object becomes queryable only
after all chunks finish. Failed or short writes abort the reservation; other
completed objects remain valid. Publish retains the source pages until D2H and
SSD work finish, including after a caller wait deadline. This extends page hold
time versus asynchronous DRAM writeback.

Direct writes require all slots of a storage group in the same Publish.
Fragmented layers and multi-writer TP/PP groups assemble in DRAM and keep
io_uring writeback. Both forms can subsequently restore through cuFile.
The existing `all`/`reuse` admission policy still applies.

Every read completes before scatter. The scatter stream drains before staging
is reused or completion is reported, including partial submission failures.
Short/failed reads fail the restore. A ring reservation that would overwrite an
active read or write tries the other shards, then drops the write if none has
space. It does not block the directory. SSD remains a best-effort cache recreated
at Manager startup.

## Copy avoidance and remote storage

Optimize request completion time and retained resources alongside copy count.
The native cuFile path avoids host staging for SSD demand reads. OrbitKV still
uses a registered GPU staging buffer and D2D scatter/gather to handle engine
layouts and alignment. Direct I/O into registered engine pages is a separate,
unimplemented optimization: it needs compatible offsets/sizes, reusable memory
registration, engine-page lifetime protection and a comparison against coalesced
staging. Many small direct operations can cost more than a larger staged read.

| Source and destination | Relevant mechanism | OrbitKV status |
| --- | --- | --- |
| Local SSD ↔ GPU | GDS/cuFile | Bounded GPU staging implemented; direct engine-page I/O pending |
| Remote shared-cache memory ↔ GPU | GPUDirect RDMA through a transport such as Mooncake TE | Manager recovery lands in requester DRAM before GPU restore; direct engine destinations are pending |
| Prefill GPU → decode GPU | Mooncake TE memory transport; GPUDirect RDMA on supported hardware | Experimental vLLM P/D connector writes engine GPU pages; GPU and cross-host qualification pending |
| Remote storage filesystem or mounted NVMe-oF ↔ GPU | GDS with a supported filesystem/network/storage stack | Separate deployment and qualification; not enabled by the current peer-memory path |

The pinned Mooncake TE includes an NVMe-oF transport that registers buffers with
cuFile and submits cuFile batch I/O. This is distinct from its RDMA memory
transport. The OrbitKV Manager registers its pinned DRAM pool and fetches remote
state into that pool; using Mooncake does not implicitly enable its storage
transport. Remote GDS requires an accessible storage namespace and a qualified
filesystem, network and GPU path. See the pinned
[NVMe-oF implementation](https://github.com/kvcache-ai/Mooncake/blob/719735896c86b56fabec6cf3e825fb2ea640597a/mooncake-transfer-engine/src/transport/nvmeof_transport/nvmeof_transport.cpp).

Direct placement in engine HBM can remove an extra GPU staging allocation and
D2D scatter when source and destination layouts match. The KV state still
occupies HBM, and destination pages must remain reserved until completion,
including after cancellation. CUDA IPC only shares access to an allocation;
sharing a handle does not itself move data. The
[experimental vLLM P/D connector](pd-mooncake-push.md) already registers engine
GPU tensors and submits remote writes, but remains outside the qualified shared
cache path. Direct placement in that shared-cache path is a follow-up to
qualify before adding remote SSD pools.

GDS does not schedule requests or select reusable model state. Recovery planning
selects the required ranges; the storage/transfer owners choose a viable data
path and keep leases until it completes. A future request-level selector should
measure queue delay, registration cost, I/O count, bytes/alignment amplification,
GPU scatter cost, source-page hold time and deadline. Use bounded concurrency,
read priority and hysteresis before changing paths based on latency. Those
policies require matched native hardware measurements; they are not inferred
from successful driver loading.

See [NVIDIA's buffering guidance](https://docs.nvidia.com/gpudirect-storage/best-practices-guide/index.html)
and [Mooncake's transport design](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/design/transfer-engine/index.md).

## Review against LMCache

Reviewed on 2026-09-24 against LMCache **v0.5.5**, commit
`05a013b29da78cf2321b9b46ec5039dde2fb0bb0`. Its
[legacy GDS backend documentation](https://github.com/LMCache/LMCache/blob/05a013b29da78cf2321b9b46ec5039dde2fb0bb0/docs/source/kv_cache/storage_backends/gds.rst)
marks in-process mode deprecated and recommends MP mode. The MP implementation
is the relevant reference for the Manager architecture.

LMCache MP also registers GPU staging; GDS does not imply direct I/O into every
engine page. Its
[GDS context](https://github.com/LMCache/LMCache/blob/05a013b29da78cf2321b9b46ec5039dde2fb0bb0/lmcache/v1/gpu_connector/gds_context.py)
keeps file/buffer registrations, splits transfers at registered-region boundaries,
and retains asynchronous submissions behind per-stream GPU events. Its
[cuFile binding](https://github.com/LMCache/LMCache/blob/05a013b29da78cf2321b9b46ec5039dde2fb0bb0/lmcache/v1/gpu_connector/_cufile_async.py)
uses `cuFileStreamRegister`, `cuFileReadAsync` and `cuFileWriteAsync`.

| Concern | LMCache MP reference | Current OrbitKV |
| --- | --- | --- |
| GPU registration | Reusable staging, registered in regions of at most 16 MiB | One reusable registered 8 MiB buffer per instance/device |
| File allocation | Preallocates its slab with `posix_fallocate` | Sets logical shard lengths; physical space is not reserved |
| Submission/completion | Stream-ordered asynchronous I/O; event-scoped submission lifetime | Synchronous cuFile calls and a scatter/gather stream drain for every batch |
| Batching | GPU context provides four chunk slots | One staging slot; read ranges merge within each source lease, not across adjacent leases |
| Tier policy | GDS L1 replaces pinned-DRAM L1 in that configuration | Complete-group GPU writeback also creates a DRAM copy; Publish waits for SSD completion |

The four-slot geometry comes from LMCache's
[CUDA cache context](https://github.com/LMCache/LMCache/blob/05a013b29da78cf2321b9b46ec5039dde2fb0bb0/lmcache/v1/platform/cuda/cache_context.py);
its [MP configuration](https://github.com/LMCache/LMCache/blob/05a013b29da78cf2321b9b46ec5039dde2fb0bb0/docs/source/mp/configuration.rst)
describes the L1 medium switch. Neither its slot count nor OrbitKV's 8 MiB size
establishes an optimum for another model or storage device.

The next implementation gates, in order, are:

1. **Physical allocation and native evidence.** Reserve GPU-storage file space
   and handle capacity exhaustion explicitly. Qualify first writes and overwrites
   separately: even preallocated but unwritten extents can require filesystem
   metadata work. `set_len`, `O_DIRECT`, or successful registration alone cannot
   prove a native path. See NVIDIA's
   [write-allocation guidance](https://docs.nvidia.com/gpudirect-storage/o-direct-guide/#block-allocation-for-writes).
2. **Bounded asynchronous I/O in Rust.** Start with a small registered slot pool
   and stream/event ownership. Keep each operation's argument storage, result
   storage, file, extent leases and GPU pages alive through completion. Check
   actual byte counts/errors before accepting a restore or publishing a write.
   Cancellation stops admission; submitted work still drains. Compare stream
   APIs with batch I/O for small scattered ranges rather than assuming one API
   wins at every size.
3. **Batching and read progress.** Merge compatible ranges by file across source
   leases, retaining every owner and respecting compiled required ranges.
   Bound bytes, operations and queued writes; let demand reads progress between
   write batches. Extra buffers must remain charged to a GPU budget.
4. **Placement and source-page hold time.** Measure the cost of producing both
   DRAM and SSD copies. Evaluate selective hot-DRAM admission and releasing engine
   pages after their final copy into owned staging; the staging and SSD reservation
   must survive until disk completion. Large objects still need bounded chunking.
   Publish visibility must never precede complete successful writes.

These are planned changes. The current implementation provides registration,
alignment and ownership foundations, but has no measured native-GDS performance
advantage. Preserve those guarantees while adding concurrency; benchmark both
engines with matched DRAM/SSD capacities, HBM budgets, working sets and native-I/O
statistics. Shared-cache direct engine-page I/O remains a separate later step.

## Enable and qualify

Install NVIDIA's GDS user-space library on the Manager and provide a supported
storage mount and compatible host driver/kernel configuration. Ordinary io_uring
deployments do not load cuFile. Explicit `cufile` selection fails startup on
library/file registration errors; GPU buffer registration errors fail the
operation. Default `auto` instead logs initialization fallback and disables new
cuFile admission after an operation failure as described above.

For native qualification, explicitly disallow compatibility mode:

```bash
CUFILE_ALLOW_COMPAT_MODE=false CUFILE_FORCE_COMPAT_MODE=false \
orbitkv-cache-manager --pool-size 8gb \
  --ssd-cache-path /mnt/nvme/orbitkv/cache.bin \
  --ssd-cache-capacity 100gb --ssd-backend cufile
```

Check the target mount with NVIDIA's `gdscheck.py -p` and `gdsio`, then inspect cuFile
statistics for actual reads. Library installation, file registration or an
OrbitKV cuFile byte counter alone is not native-path evidence. Local NVMe GDS
does not need a network RDMA adapter. Containers require suitable host mounts,
devices and kernel support; newer NVMe PCI P2PDMA configurations can operate
without `nvidia_fs`.

Run the explicit GPU gate with no serving process active during Cargo builds:

```bash
mkdir -p /mnt/nvme/orbitkv-tests
TMPDIR=/mnt/nvme/orbitkv-tests \
CUFILE_ALLOW_COMPAT_MODE=false CUFILE_FORCE_COMPAT_MODE=false \
cargo test --release -p orbitkv-core --no-default-features \
  --features cuda-13,mooncake --test cufile -- --ignored --test-threads=1
```

For functional development, set both cuFile environment values to `true` and
choose a writable mount on which cuFile can register files. Compatibility mode
does not make every mount usable: this container's `/tmp` ext4 mount fails file
registration, while `/workspace` overlay works in forced compatibility mode.
Set `TMPDIR` accordingly; for pytest, set `--basetemp` on that same mount. Record
this as **cuFile compatibility-mode correctness**, not native GDS performance.
The gate covers split/page-first layouts, unaligned components, checkpoints
larger than staging, ring pinning, cross-shard progress, cancellation and
short-read failure recovery. It also covers fragmented publication sealing
and restoring its io_uring-written payload through cuFile.

The development serving gates use **cuFile 1.16.1 from CUDA 13.1**. Their Torch
2.13/cu130 environments preload cuFile 1.15.1.6, which fails to register this
overlay mount even in compatibility mode. The commands below explicitly preload
the verified library. Qualify the library and mount together on another host;
do not infer native support from a successful compatibility run.

Both serving gates accept `--ssd-backend cufile` and require positive cuFile
reads after writes drain, DRAM eviction and an engine restart, plus positive
cuFile write bytes:

```bash
cd python
ORBITKV_CACHE_MANAGER_BINARY=../target/release/orbitkv-cache-manager-py \
LD_PRELOAD=/usr/local/cuda/lib64/libcufile.so.0 \
CUFILE_ALLOW_COMPAT_MODE=true CUFILE_FORCE_COMPAT_MODE=true \
../.venv/vllm-release/bin/python -m pytest -m e2e \
  tests/e2e/test_vllm_e2e_correctness.py --model /workspace/models/qwen3-8b \
  --vllm-cache-tier ssd --ssd-backend cufile \
  --max-model-len 4096 --orbitkv-pool-size 1gb \
  --basetemp=/workspace/orbitkv/benches/results/runs/cufile-vllm

ORBITKV_CACHE_MANAGER_BINARY=../target/release/orbitkv-cache-manager-py \
LD_PRELOAD=/usr/local/cuda/lib64/libcufile.so.0 \
CUFILE_ALLOW_COMPAT_MODE=true CUFILE_FORCE_COMPAT_MODE=true \
../.venv/sglang-release/bin/python -m pytest -m e2e \
  tests/e2e/test_sglang_direct_e2e.py -k ssd --model /workspace/models/qwen3-8b \
  --ssd-backend cufile \
  --basetemp=/workspace/orbitkv/benches/results/runs/cufile-sglang
```

Use a separate `--basetemp` for each run and run GPU gates sequentially. Change
both cuFile environment values to `false` for native-path qualification.

## Recorded functional qualification

Final checks on 2026-09-24 used one H20, cuFile 1.16.1, vLLM 0.29.0 and
SGLang 0.5.20. Explicit cuFile runs **forced CPU compatibility mode** on this
container's overlay mount. Default `auto` runs used its `/tmp` ext4 mount and
selected io_uring after cuFile file registration failed. These results establish
correctness, not native GDS throughput.

| Gate | Final result |
| --- | --- |
| Rust GPU layouts, checkpoints, pinning, failure recovery and auto fallback | 6 passed |
| Manager process faults, cancellation and resource ownership | 16 passed |
| vLLM / Qwen3-8B / SSD | 6 passed; 1 recurrent-only check skipped, with both explicit `cufile` and default `auto` |
| SGLang / Qwen3-8B / SSD | Passed with both explicit `cufile` and default `auto` |
| vLLM / Qwen3.8-27B-FP8 / SSD | 7 passed |
| SGLang / Qwen3.8-27B-FP8 / SSD | Passed |

Both models' explicit cuFile checks require cuFile writes and new reads after
DRAM eviction and engine restart. Default-auto checks require new SSD reads
through the selected backend after the same eviction and restart sequence.
Qwen3.8 also exercises full-attention and GDN
conv/recurrent state in the same request. Reproduce it with the
[Qwen3.8 engine settings](models.md#qwen38-on-h20) and `--ssd-backend cufile`.

## Bare-metal acceptance script

[`benches.gds`](../benches/gds.py) runs the native qualification sequence. It
rejects containers, verifies the selected ext4/XFS mount is NVMe-backed,
disables CPU compatibility, and enables per-process cuFile statistics through
a private config overlay. It does not change system configuration. A private
subdirectory under `--ssd-dir` holds test data and results; it never opens a raw
disk or overwrites an existing cache file.

Build the ordinary Manager, a separate `test-hooks` Manager, the matching Python
extension in **both** engine environments, and the Rust `cufile` test executable
before starting. Use the CUDA feature matching the machine. Obtain the test
executable path from `cargo test --release -p orbitkv-core --no-default-features
--features cuda-13,mooncake --test cufile --no-run`. Copy each Manager before
building the other variant. The script accepts these prebuilt artifacts and
never invokes Cargo while services are running.

```bash
.venv/vllm-release/bin/python -m benches.gds \
  --ssd-dir /mnt/nvme/qualification --model /models/Qwen3-8B \
  --manager /opt/orbitkv/manager \
  --fault-manager /opt/orbitkv/manager-faults \
  --core-test /opt/orbitkv/cufile-test \
  --gds-tools /usr/local/cuda/gds/tools \
  --cufile-library /usr/local/cuda/lib64/libcufile.so.0
```

Set `--gds-tools` to the directory containing NVIDIA's `gdscheck.py`, `gdsio` and
`gds_stats`. The sequence covers:

1. GPU/storage topology, `gdscheck`, and native `gdsio` writes and reads.
2. Exact GPU-byte recovery, pinning, fragmented saves and large checkpoints.
3. Real-process faults including stalled/failed cuFile writes and Manager death.
4. vLLM and SGLang Qwen3 correctness across engine restart and SSD recovery.
5. Matched io_uring/auto/cuFile serial and sustained mixed-prefix workloads, with
   the same seed, capacities and a working set larger than both DRAM and HBM KV.

Every auto/cuFile benchmark captures `gds_stats -p <manager-pid> -l 3` while the
Manager is alive. Success requires positive reads **and** writes and zero
POSIX, unaligned, sparse/inline or failed operations. Missing/unknown statistics
fail the gate, as does any automatic backend fallback. Ordinary io_uring work is not counted as native I/O. The final
`qualification.json` records all stage exits; per-run summaries retain TTFT,
throughput, storage counters and whole-workload Manager CPU usage. Linux process
I/O accounting is recorded separately from cuFile GPU I/O bytes. Raw logs remain
in this private run directory.

This script has source-level checks in `benches/tests/test_gds.py`. Its full
native sequence is pending an accessible bare-metal GPU/NVMe host; container
functional passes do not satisfy it.

## Metrics and next gates

- `orbitkv_ssd_cufile_read_bytes_total` / `orbitkv_ssd_cufile_write_bytes_total`:
  physical bytes, including alignment.
- `orbitkv_ssd_cufile_read_seconds` / `orbitkv_ssd_cufile_write_seconds`: synchronous I/O duration.
- `orbitkv_ssd_cufile_read_failures_total` / `orbitkv_ssd_cufile_write_failures_total`: failed or short I/O.
- `orbitkv_ssd_backend_fallbacks_total`: transitions from automatic cuFile admission to io_uring.
- `orbitkv_ssd_read_pinned_bytes`: SSD bytes held by restore leases.
- `orbitkv_ssd_gpu_staging_bytes`: allocated registered GPU staging memory.
- `orbitkv_ssd_pinned_write_skips_total`: reservations rejected to protect reads
  or in-flight writes.

The timeline's `source_ready` event now means the source lease is ready. In
cuFile demand mode, disk reading occurs during GPU restoration, not query
preparation; compare end-to-end TTFT and storage counters rather than treating
preparation time alone as an improvement.

Before claiming a performance improvement, compare io_uring and **verified native** GDS on the same
NVMe mount, model and working set larger than DRAM. Measure TTFT, throughput, CPU
use, physical/read amplification, staging budgets and lease drain. Keep
speculative preparation off in the first comparison. Automatic cost-based path
selection, prioritizing reads over writes and overlapping multiple cuFile
operations per GPU remain follow-up work. Complete-group direct writeback is
implemented; multi-writer GPU assembly remains separate work.

## Design references

- [FlexKV GDS implementation](https://github.com/taco-project/FlexKV/blob/738ddc141a198b4e20de6c5d1f0128e387f7fdb2/csrc/gds/gds_manager.cpp): file registration and GPU staging with layout transforms.
- [LMCache GDS L1](https://docs.lmcache.ai/mp/configuration.html#gds-l1-tier): explicit storage-medium selection. OrbitKV retains DRAM for hot data and speculative preparation.
- [NVIDIA installation and verification](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/) and [benchmarking guide](https://docs.nvidia.com/gpudirect-storage/configuration-guide/): native configuration and per-process path evidence.
- [NVIDIA cuFile API](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html) and [best practices](https://docs.nvidia.com/gpudirect-storage/best-practices-guide/): registration lifetime, completion, alignment and compatibility mode.
