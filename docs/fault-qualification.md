# Single-node fault qualification

The deterministic process gate runs a real Cache Manager, registered CUDA IPC
buffers and SSD reads. It uses a separate `test-hooks` build; default release
binaries contain no fault barriers. Each test owns its Manager and a private
barrier directory. Ordinary completion and model-output gates run separately.

| Fault | Required behavior |
| --- | --- |
| SSD completion paused after submission; query cancelled or replaced | Submitted buffers and reservations stay owned until I/O drains; old results cannot attach to the new query; unrelated queries progress; reservations return to zero. |
| One-page read batches with cancellation, a relative deadline or best-effort stopping | No second batch is submitted. A deadline lets demand recompute while the first batch drains with its budget retained. Unread pages are not classified as HLL misses. |
| One owner cancels a shared preparation read | Another demand owner completes from the shared read; cancellation cannot revoke its buffers. |
| Prepared result expires without another poll | The Manager releases the undelivered lease; a matching claim before expiry keeps its bytes owned through GPU completion. |
| Restore completion delayed and eventfd notification dropped | A wait deadline returns no ownership of destination pages. Polling the same handle discovers terminal completion; restored bytes match, and reservations drain. |
| cuFile worker paused before reading, query cancelled and notification dropped | The SSD extent remains pinned through restore completion. A concurrent DRAM restore completes; polling recovers completion, bytes match and ownership counters drain. |
| cuFile write paused, completion failed or Manager killed | Unfinished objects stay invisible; GPU pages remain owned; DRAM restores progress; failed reservations can be retried and staging is released on unregister. |
| GDS hot-copy completion held while SSD work continues | Publish and unregister keep engine mappings; unrelated SSD demand restores complete with exact GPU bytes. |
| FP8 read canceled, mixed raw/encoded prefix or corrupted file | Scratch survives submitted I/O, FP8-representable inputs and raw fallbacks restore correctly, damaged decodes become misses and scratch returns to zero. |
| Publish delayed beyond the call deadline | The publisher retains source pages while other query sessions progress. Releasing the barrier completes the save. Killing the Manager terminates the wait safely. |
| Publish acknowledgement malformed | The session is poisoned and the publisher remains fenced until Manager death. Descriptor corruption cannot be mistaken for DMA completion. |
| Manager restart with old clients, leases and a pending restore | A fresh service incarnation starts behind the same UDS address; old handles/leases are rejected. A newly registered engine can publish and restore. Both default and configured service prefixes are exercised. |
| Engine process killed with registered CUDA IPC buffers | Session watching drains work and drops the old registration. Existing engine-restart E2Es verify subsequent reuse against output controls. |

Publish logs a warning after the configured ordinary call deadline, then at
most once per minute. It never frees sources merely because a timer expired.
A permanently stuck live Manager requires operational restart; this gate does
not install an automatic process killer.

## Reproduce

Build before starting any native/GPU tests: Mooncake shared libraries are
restaged by Cargo and must not be replaced under running processes.

```bash
PYO3_PYTHON=$PWD/.venv/sglang-release/bin/python \
  cargo build --release -p orbitkv-server -p orbitkv-py \
  --no-default-features --features cuda-13,mooncake,orbitkv-server/test-hooks
# Install the matching extension using the development/build instructions.
cd python
ORBITKV_CACHE_MANAGER_BINARY=../target/release/orbitkv-cache-manager-py \
ORBITKV_FAULT_TESTS=1 ../.venv/sglang-release/bin/python -m pytest -m integration \
  tests/integration/test_cache_faults.py tests/integration/test_session_watcher.py
```

The test fixture alone sets `ORBITKV_TEST_FAULTS` for its private process. Do not
ship `test-hooks` binaries. Run the [hybrid recovery gates](hybrid-recovery.md#reproducible-gates)
with normal builds as well. The H20 container verifies functional SSD I/O on its
mounted filesystem; it does not measure physical NVMe or RDMA behavior.

The cuFile cases additionally requires `--ssd-backend cufile` and a writable
`--basetemp` on a cuFile-compatible mount. Use the environment settings in
[GPU storage recovery](gds.md) and record whether compatibility mode was enabled.
Without this option, the ordinary fault gate skips the cuFile-only cases.

## Concurrent Qwen3 serving

The explicit stress gate uses Qwen3-8B with deterministic inference in each
pinned engine. It restarts the engine while retaining the Manager, pauses SSD
reads and abandons a streaming request while an unrelated request progresses,
drops completion notifications, and kills the Manager during a GPU restore.
It checks reference outputs, positive restore bytes, cancellation observations
and final ownership counters. A restarted Manager starts cold; this is not an
SSD index persistence test.

```bash
cd python
ORBITKV_FAULT_TESTS=1 ORBITKV_PREPARE_REQUESTS=1 \
ORBITKV_CACHE_MANAGER_BINARY=/absolute/path/to/test-hooks-manager \
../.venv/vllm-release/bin/python -m pytest -m stress \
  tests/stress/test_recovery_faults.py -k vllm --model /workspace/models/qwen3-8b \
  --basetemp=/workspace/orbitkv/benches/results/runs/serving-fault-vllm
```

Use the SGLang interpreter and `-k sglang` for its gate. Set
`ORBITKV_PREPARE_REQUESTS=0` for ordinary demand. Keep the two GPU runs sequential.
The workspace test directory retains every engine/Manager incarnation log and
`fault-results.json`. Manager sockets use short temporary paths independently
of the evidence directory.

Multi-rank serving, long-running injected-fault traffic, hardware hangs and remote
failover have separate qualification gates. Page-generation references in a
future region protocol are not replaced by process/session fencing.
