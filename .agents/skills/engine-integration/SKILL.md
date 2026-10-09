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
- vLLM 0.31.0 workers return `KVConnectorTransferResults` directly. P/D failed
  receives must also be finished receives in the same poll; do not restore a
  custom worker-metadata failure queue. Ordinary cache delivery is best effort,
  while a P/D producer retains the native reliable-delivery requirement.
- SGLang 0.5.21 cancellation already reaches the external linker through
  `BasePrefixCache.finish(ABORT)` and `UnifiedRadixCache.release_aborted_request`.
  Keep cancellation/drain in that consumed lifecycle; do not restore a duplicate
  private Scheduler abort Hook. Queue preparation is opt-in; fork P/D observation
  Hooks are removed.
- P/D uses official vLLM NIXL/MultiConnector and SGLang native disaggregation;
  Manager shared-cache traffic still uses TENT. Do not reintroduce fork factories,
  callbacks, custom connectors, handshake/proxy or partial-tail cache modes.
- The candidate profile reads/writes cache on P and only saves on D. Native P/D
  owns incoming destinations; cache retains completed state until its own drain.
  Preserve reliable P/D delivery and release unselected cache query leases.
- Official vLLM 0.31.0 requires V1 (`VLLM_USE_V2_MODEL_RUNNER=0`) and one attention
  cache group. Reject V2/recurrent profiles before connection; the runner and
  native-prefix monkey patches are removed. Reopen only after consumed released
  ordering and atomic state hand-off pass GPU/model gates.
- SGLang native P/D candidate enables released deferred KV release on both P
  and D, but 0.5.21 still releases on timeout without a full drain ACK. Keep
  transfer cancellation, peer-loss and delayed-ACK page reuse unqualified;
  ordinary linker cancellation is a separate cache-owned contract.
- vLLM's public `reset_cache()` now invalidates future external reuse through
  a fresh prefix-key generation only after all tracked request/query/load/save
  owners end. HBM-only reset preserves the current generation. Test idle
  `reset_external=true` followed by a cold miss and reuse of newly saved state;
  do not advertise active-request reset, persistent cluster-wide invalidation
  or live weight changes that bypass the callback. SGLang ordinary flush keeps
  same-weight external reuse and is not a universal weight-update notification.
- Fixed local PEFT adapters require startup content fingerprints and effective
  LoRA settings in the deployment namespace. Keep native base/adapter hashes
  intact; reject resumable/session paths and caller extra-key aliases that bypass
  safe matching. This immutable profile does not intercept engine SDK/RPC
  add/remove/reload or live-weight updates; change artifacts only after stopping
  the engine.
- SGLang 0.5.21 preserves raw `ServerArgs` and publishes resolved configuration
  separately. Consume `runtime_context.get_lora()` for effective enable, startup
  LoRARef UIDs, backend and target modules, as its native loader/registry do.
  `--lora-paths` does not make the raw `enable_lora` field true.
- Run native model composition and lifetime gates separately. Output/restart
  does not prove cancellation, partial-submit, delayed-ACK or page-reuse safety.
  Earlier fork passes remain upstream contribution evidence, not release support.

## Replace internal coupling safely

Audit every remaining Hook against `docs/engine-release-audit.md`. SGLang 0.5.21
captures graphs before the tree-cache factory; preserve pre-capture events until
a safe released callback exists. Its linker lookup lacks pending tickets, so
pending-query admission remains a guarded internal Hook; queue preparation is
opt-in. Do not describe these as public lifecycle integration. Keep native abort
finish, cancellation/drain and page ownership until a replacement is consumed.
Do not add a facade merely to hide internal coupling.

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
