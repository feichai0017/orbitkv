---
name: engine-integration
description: Implement or review OrbitKV vLLM/SGLang connectors, hybrid recovery, P/D composition, release upgrades and upstream contributions; diagnose engine page-lifecycle or async restore failures.
---

# Released-engine integration

Read `AGENTS.md`, `docs/adapters.md` and S5 in `docs/completion-plan.md` from the
Git root. The completion plan is the only work queue. Use its current support
status; do not treat an upgrade target or upstream feature as qualified behavior.

## Select the actual contract

For an upgrade, check the latest official non-prerelease tag and record its commit.
Inspect that source and the corresponding released LMCache integration. Use main
and related issues/PRs to find fixes; distinguish unreleased fixes explicitly.
Freeze the selected release during review and update pins, lockfile and submodule
together after the consumed adapter passes. Do not add old/new API fallbacks.

## Keep one owner per boundary

- Engines own request scheduling, GPU allocation, native prefix trees, rank
  collectives, model execution and final request readiness.
- Python translates engine layout, state boundaries, callbacks and CUDA readiness.
  Existing Rust owners manage shared cache policy, transfers, admission and drain.
- vLLM uses the KV Connector contract; SGLang uses UnifiedRadixCache and the
  external linker. Preserve valid native HBM hits without external synchronous
  lookup. Keep unselected backend imports free of native initialization.
- vLLM 0.30.0 workers return `KVConnectorTransferResults` directly. P/D failed
  receives must also be finished receives in the same poll; do not restore a
  custom worker-metadata failure queue. Ordinary cache delivery is best effort,
  while a P/D producer retains the native reliable-delivery requirement.
- SGLang 0.5.20 cancellation already reaches the external linker through
  `BasePrefixCache.finish(ABORT)` and `UnifiedRadixCache.release_aborted_request`.
  Keep cancellation/drain in that consumed lifecycle; do not restore a duplicate
  private Scheduler abort Hook. Queue preparation and P/D observation Hooks are
  registered only when their startup options are enabled.
- P/D uses the native lifecycle plus explicit TENT factories at the exact
  experimental revisions in `docs/pd.md`; official pins lack these APIs.
  The custom vLLM P/D package, handshake/proxy and partial-tail cache extension
  are removed. Keep ordinary cache adapters independent; do not recreate them.
- Native MultiConnector selects one load owner. Release unselected cache query
  leases; an unselected P/D cleanup pull must never write or emit decode
  completion. SGLang P restores historical state; D only saves while native
  P/D owns incoming destinations. Preserve reliable native P/D delivery.
- SGLang P/D telemetry consumes public immutable `PDTransferEvent` callbacks;
  all six private P/D observation Hooks are removed. Callbacks never authorize
  release. Timeout quarantines pages; per-attempt writer ACKs prove drain.
  Retain source pages and all submitted layer futures until native quiescence.
- Run native GPU lifetime and model composition gates for cancellation, partial
  write, delayed ACK, shutdown, restart and observed preemption/retraction.
  Permanent peer loss, cross-host revocation, RDMA and hybrid/rank combinations
  remain open. A TENT call alone does not prove GPUDirect RDMA. Use batch status
  for additive byte accounting and per-task status for terminal drain.

## Replace internal coupling safely

Inventory the concrete callers before editing `vllm/runtime.py`, the multi-group
native-prefix bypass, `sglang/recovery.py`, private lifecycle Hooks or the SGLang
TENT factory registration. Identify a released replacement or propose a minimal
engine-generic upstream fix. Preserve safety until the replacement is consumed;
then remove the old path in the same change. Do not add a generic facade merely
to hide internal dependencies. A public adapter owns an interface and may remain.

For a stuck or incorrect request, read [the boundary tracing guide](references/lifecycle.md).
Enqueue, layer readiness, final native drain and source retirement are distinct.
Timeout, process disappearance and metadata expiry cannot prove physical drain.

## Verify the claimed profile

Run the relevant gates from `AGENTS.md`: source-only tests for local contracts;
actual engine/GPU tests for changed callbacks, preemption, page reuse and graph
ordering. Include cold/partial/full hits, native HBM, restart and failure controls.
Preserve adapter/salt/model identity and effective committed lengths. Record TP,
PP, attention DP, EP, P/D topology and transport separately; a one-GPU process test
does not qualify their distributed combination. Keep evidence outside checkout.

For upstream delivery, separate baseline registration/configuration/tests/docs,
generic lifecycle fixes and TENT transport integration into reviewable PRs.
Use a clean engine checkout plus installed wheel as the acceptance environment.
Report local qualification, upstream merge and released support separately.
