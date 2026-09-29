# Benchmark evidence and artifact storage

Historical run labels below refer to the [evidence inventory and availability limits](benchmark-evidence.md).

Keep benchmark programs, reusable workloads, reproduction scripts and small
input fixtures in the repository. Raw responses, JSON/CSV measurements, manifests,
logs, traces, generated plots and per-run reports belong in an explicitly chosen
external directory or CI artifacts. Preserve unsuccessful runs and controls.
Public documentation contains reviewed conclusions, their limits and links to
versioned evidence; passing a correctness gate does not establish a speedup.

## Run and retain an experiment

Choose a new directory outside the source checkout for each run. Python harnesses
validate `--output` before starting services, including resolution of symlinks.
`benches.sharegpt` requires `--output-dir`; `benches/serving.sh` requires
`RESULT_DIR`. The GDS qualification uses a private directory on the explicitly
selected external `--ssd-dir` mount. Criterion measurements require an external
`CARGO_TARGET_DIR`. Build native libraries before starting any runtime gate.

```bash
.venv/vllm-release/bin/python -m benches.single_node \
  --engine vllm --backend orbitkv --model /path/to/immutable-model \
  --output /var/tmp/orbitkv-bench/vllm-dram-001

python -m benches.report /var/tmp/orbitkv-bench/vllm-dram-001 \
  --output /var/tmp/orbitkv-bench/vllm-report-001
```

The [benchmark guide](../benches/README.md) defines workloads and matched controls.
The [preparation reproduction script](../benches/reproduce_preparation.sh) accepts
an external output root as its first argument. Preserve source/engine/model
revisions, binary hashes, hardware and transport, budgets, commands, correctness
and drain evidence, uncertainty and all failed preparations with the result.
Use at least three order-alternated matched runs and declare thresholds before
claiming performance gains. Compare equal state coverage and completion targets.

In CI, select a directory such as `$RUNNER_TEMP/orbitkv-evidence`, write logs and
reports there, and upload it with `actions/upload-artifact@v4` using
`if: always()` so failures survive the job. Release qualification must identify
the exact installed artifact. CI retention expiry is not permanent publication;
copy evidence referenced by a release into versioned durable storage before expiry.

`python3 scripts/check-experiment-output.py` inspects the Git index in local
checks and CI, including force-added files. It reserves `benches/results/`,
`benches/runs/`, `results/` and `runs/` for generated output and rejects tracked
entries there. It deliberately allows deterministic fixtures under `tests/` and
source files whose names contain `result`; it does not infer content from such names.

## Historical tracked evidence

The S1 migration starts at
[`9fe1441c0d7d4c47b1914c303f837bba9f4a758f`](https://github.com/feichai0017/orbitkv/tree/9fe1441c0d7d4c47b1914c303f837bba9f4a758f/benches/results).
That immutable revision retains all 11 formerly tracked result files, including
preparation, offload, cache-policy and shared-cache controls. Git history is not
rewritten. Older collections remain at their existing immutable
[September 21 snapshot](https://github.com/feichai0017/orbitkv/tree/44c1e5f9a253aa7378c6187b2aeea9bff93df304/benches/results)
and [recovery snapshot](https://github.com/feichai0017/orbitkv/tree/4712f780c900120719f178b2ea36c9e0ac7c135f/benches/results/20260922-recovery-baseline).
The [pre-migration documents](https://github.com/feichai0017/orbitkv/tree/9fe1441c0d7d4c47b1914c303f837bba9f4a758f/docs)
retain the original report tables, measured revisions and historical locations.
These are measurements of those revisions, not fresh evidence for current HEAD.
Website link tests validate historical file targets with `git cat-file`; a shallow
checkout needs `git fetch --unshallow` before running `npm test`. Website CI fetches
full history for this check.

## Local archive and verification

The selected S1 archive on the measurement host is
`/root/orbitkv-artifacts/s1-evidence-20260929/`:

- `legacy-results/` retains the complete local results tree, including ignored
  run data, commands, failed attempts and controls.
- `archive-manifest.json` records the source/base commit, relative paths, sizes,
  SHA-256 values, symlink targets and whether each entry was tracked.
- `SHA256SUMS` verifies archived regular files; `archive-verified.json` records
  the inventory hash and totals. The migration hashes the source before copying,
  hashes the archive, and rehashes the source before removing any originals.
- `symlink-relocations.json` preserves original absolute targets and their new
  relative targets inside the archive. All 43 symlinks resolve after migration;
  35 pytest convenience links required relocation. Regular-file hashes are unchanged.
- `historical-docs/` is a convenience copy of the documents at the starting
  revision. Immutable Git links above remain the public reference.

```bash
cd /root/orbitkv-artifacts/s1-evidence-20260929
sha256sum --check SHA256SUMS
```

The verified archive contains **2,345 regular files, 85,355,300,572 logical bytes**
(including sparse files), plus symlinks and directories. The manifest SHA-256 is
`55d5155a24a730c6053db91870e0a9515d35e026e18565989b9f0e0bd2f66db3`.

The local archive covers the September 28–29 `cache-e2e`, `completion-evidence`,
`layered-restore`, `partitioned-restore`, `strided-dma`, `same-host-a100`,
`two-host` and `two-host-natural` run directories present at migration. Their
original `benches/results/runs/<name>` paths now map to
`legacy-results/runs/<name>` under that archive root. The manifest, not a path
mentioned in prose, establishes which bytes were archived.

Earlier ignored runs (including P4 cost/route experiments, queued warming and
query-budget controls) were absent from this checkout at migration. Their
historical documents and tracked aggregates remain accessible at the fixed Git
revisions above; this migration does not claim to have recovered or verified
missing raw logs. A `historical run label` in a document identifies that earlier
record, not a readable local directory. The local archive is not a public download
service or an off-host backup. Publishing new charts requires durable versioned
artifacts; new performance or hardware claims need their own qualification.
