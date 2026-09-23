# Native client control-path measurements

On 2026-09-22, the final control path reduced median polling time for a
1024-hash, budget-waiting query from 123.50 to 106.39 microseconds on one H20
host. This is a narrow control-path measurement, not model TTFT, SSD throughput,
or a comparison with another KV cache.

## Workload and controls

The [harness](../benches/client.py) starts a real Manager, registers CUDA pages,
publishes 1024 64-KiB pages and holds their lease to occupy the 64-MiB instance
query budget. Measured requests must remain admitted `Loading`; they cannot
silently become a miss, finish an I/O, or acquire another lease. Each size uses
three batches of 1000 calls. Setup, hash-batch construction and publication are
outside the timed loop. No other GPU workload runs concurrently.

The baseline uses the Python client from main `31fa1f6b` and its archived native
extension, with the Manager before the deferred-copy optimization. The final
client uses immutable Rust `BlockHashes`. A control connects the archived Python
client to the final Manager, separating client changes from Manager changes.
All runs use Python 3.11.2, the same H20, a 256-MB pinned pool, and the same
query budgets. [Archived per-batch measurements](https://github.com/feichai0017/orbitkv/tree/44c1e5f9a253aa7378c6187b2aeea9bff93df304/benches/results)
include commands, percentiles and caller thread CPU time.

## Results

Each cell is the median of the three batch p50 values, in microseconds.

| Hashes per waiting query | Python client + original Manager | Rust client + final Manager | Python client + final Manager |
| --- | ---: | ---: | ---: |
| 64 | 107.12 | 106.34 | 106.34 |
| 256 | 109.67 | 106.38 | 106.32 |
| 1024 | 123.50 | 106.39 | 106.33 |

The 1024-hash reduction is about 14%. The control attributes the observed
wall-time improvement primarily to the Manager: it no longer clones the full
query on each poll before byte admission. Final polling times are almost flat
over the tested sizes. On the same final Manager, Python and Rust client times
are indistinguishable at this experiment's resolution; moving code to Rust
alone does not establish a latency improvement.

Rust ownership removes the Python query map, locks, ticket counters and restore
waiting loop. Both adapters retain an immutable native hash batch and share
prefix views, avoiding repeated per-page PyO3 conversion; SGLang also stops
rehashing unchanged lookup keys. Engine request-drift checks and GPU allocation
remain in the adapters. These structural changes require separate serving
profiles before claiming CPU or TTFT gains for a model workload.

## Reproduce the current path

Build the CUDA-matched extension and `target/release/orbitkv-cache-manager`, then
run from the repository root in a Torch/CUDA environment:

```bash
PYTHONPATH=python .venv/sglang-release/bin/python -m benches.client \
  --label rust-client --output benches/results/runs/client-poll
```

The output directory must be new. Do not run Cargo builds or other GPU workloads
during the experiment. Comparing older clients requires their matching native
extension and Python API in a separate process; passing fresh hash lists on
every callback would measure a different workload from the native batch path.

The harness does not measure first submission, successful-hit lease creation,
model execution, multi-rank traffic, or throughput under concurrent serving.
The remaining first-use and conversion costs belong in those next profiles.
