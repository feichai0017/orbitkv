# GPU storage

OrbitKV has an opt-in cuFile SSD backend. Complete state groups can be written
from registered engine GPU pages to SSD, and selected SSD state can be restored
to those pages. Both directions use bounded, registered GPU staging.
vLLM and SGLang share this Rust implementation and their existing cache API.
The default remains `--ssd-backend uring`.

Calling cuFile does **not** prove native GPUDirect Storage: its compatibility
mode performs host staging internally. The H20 container's native-mode probe
fails at `cuFileDriverOpen` with `CU_FILE_DRIVER_NOT_INITIALIZED` (5001).
Native GDS throughput remains unqualified.

## Data flow and ownership

| Operation | Path |
| --- | --- |
| DRAM hit | Pinned DRAM → engine GPU pages |
| SSD demand hit with `cufile` | SSD → registered GPU staging → engine GPU pages |
| Speculative prepare/warmup | SSD → pinned DRAM; GPU restore after consumption |
| Complete state group in one Publish | Engine GPU → registered GPU staging → SSD; also retain a hot DRAM copy |
| Fragmented/multi-writer Publish | Engine GPU → pinned DRAM; seal complete group → io_uring → SSD |
| Remote recovery | Mooncake → pinned DRAM → engine GPU pages |

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

## Enable and qualify

Install NVIDIA's GDS user-space library on the Manager and provide a supported
storage mount and compatible host driver/kernel configuration. Ordinary io_uring
deployments do not load cuFile. Library/file registration errors fail startup;
GPU buffer registration errors fail the operation. There is no silent switch
to io_uring on a cuFile error.

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
SGLang 0.5.20. **CPU compatibility mode was forced** on this container's
overlay mount. These results establish correctness, not native GDS throughput.

| Gate | Final result |
| --- | --- |
| Rust cuFile GPU layouts, checkpoints, pinning and failure recovery | 5 passed |
| Manager process faults, cancellation and resource ownership | 16 passed |
| vLLM / Qwen3-8B / SSD | 6 passed; 1 recurrent-only check skipped |
| SGLang / Qwen3-8B / SSD | Passed |
| vLLM / Qwen3.8-27B-FP8 / SSD | 7 passed |
| SGLang / Qwen3.8-27B-FP8 / SSD | Passed |

Both models' serving checks require cuFile writes and new reads after DRAM
eviction and engine restart. Qwen3.8 also exercises full-attention and GDN
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
5. Matched io_uring/cuFile serial and sustained mixed-prefix workloads, with
   the same seed, capacities and a working set larger than both DRAM and HBM KV.

Every cuFile benchmark captures `gds_stats -p <manager-pid> -l 3` while the
Manager is alive. Success requires positive reads **and** writes and zero
POSIX, unaligned, sparse/inline or failed operations. Missing/unknown statistics
fail the gate. Ordinary io_uring work is not counted as native I/O. The final
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
- `orbitkv_ssd_read_pinned_bytes`: SSD bytes held by restore leases.
- `orbitkv_ssd_gpu_staging_bytes`: allocated registered GPU staging memory.
- `orbitkv_ssd_pinned_write_skips_total`: reservations rejected to protect reads
  or in-flight writes.

The timeline's `source_ready` event now means the source lease is ready. In
cuFile demand mode, disk reading occurs during GPU restoration, not query
preparation; compare end-to-end TTFT and storage counters rather than treating
preparation time alone as an improvement.

Before changing defaults, compare io_uring and **verified native** GDS on the same
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
