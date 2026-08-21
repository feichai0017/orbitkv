#!/usr/bin/env bash
set -euo pipefail

# Breaking-only compact ABI5 qualification runner for official SGLang v0.5.17.
#
# Default action is host-only preflight. GPU execution is split into two
# explicit phases so B4 cannot run before both B1 manager/stock pairs pass.
# The compact source closure and rebuilt qualification release are hash-locked
# below; any later source or artifact edit makes preflight fail closed.

readonly QUAL_ROOT="/workspace/orbitkv"
readonly QUAL_PYTHON="${QUAL_ROOT}/.venv-sglang-v0517/bin/python"
readonly QUAL_BENCH="${QUAL_ROOT}/integrations/sglang/bench_canonical_manager.py"
readonly QUAL_STOCK_ROOT="${QUAL_ROOT}/.qualification/sglang-v0517-stock"
readonly QUAL_MANAGER_ROOT="${QUAL_ROOT}/.qualification/sglang-v0517-manager"
readonly QUAL_LIBRARY="${QUAL_ROOT}/.qualification/build/ffi-target/release/liborbitkv_ffi.so"
readonly QUAL_QWEN_MODEL="/workspace/models/qwen2.5-7b-instruct"
readonly QUAL_GPT_MODEL="/workspace/models/gpt-oss-20b"
readonly QUAL_QWEN_PLAN="${QUAL_ROOT}/.qualification/plans/qwen2.5-7b-full-page16-bf16.json"
readonly QUAL_GPT_PLAN="${QUAL_ROOT}/.qualification/plans/gpt-oss-20b-hybrid-page16-bf16.json"
readonly QUAL_REQUIREMENTS_LOCK="${QUAL_ROOT}/.qualification/requirements-v0.5.17.lock.txt"
readonly QUAL_RECORD_ROOT="${QUAL_ROOT}/.qualification/records/abi5-v5-current"
readonly QUAL_PATH="${QUAL_ROOT}/.venv-sglang-v0517/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

readonly QUAL_SGLANG_REVISION="29481685462732237d80d86076d6563e1f658102"
readonly QUAL_REQUIREMENTS_SHA256="472d8f63cad22cd7ac4908059562bebde5e54b8d2432f750640a14d525d2fa97"
readonly QUAL_QWEN_PLAN_SHA256="415db5596d4bb6943c930d3cc159471e0f8911ed5572707d527152460acca130"
readonly QUAL_GPT_PLAN_SHA256="cf870e5f8191f8bffd9b7bc4eac5d8c3aa8c6bc1b46f04b35c3c738c3d12e5e5"
readonly QUAL_QWEN_CONFIG_SHA256="7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c"
readonly QUAL_QWEN_INDEX_SHA256="624bf7c47cd12468fdc16e38a47cf4f19e0415b859a223ba3c027eed2f0e1028"
readonly QUAL_GPT_CONFIG_SHA256="3a2a26ded679375b7928ddeca59764df7cea83220c1961035f6d6e232659e9ce"
readonly QUAL_GPT_INDEX_SHA256="0e085b977c4c9942f85938828e8c989ed7d5cdabf852e4da6a67c116cd502cd1"

# Frozen compact ABI5 source closure and reproducible qualification release.
readonly QUAL_SOURCE_CLOSURE_SHA256="9233c06d40ffa19eb08b88cc1cb6fa3b72cceffcaecc96c830950f93bd5c70bc"
readonly QUAL_LIBRARY_SHA256="3917bf0ae32239c11f913b5322b1679a71a57b3990004e55c04ea8097181dc5f"

readonly -a QUAL_SOURCE_FILES=(
  "Cargo.toml"
  "Cargo.lock"
  "src/lib.rs"
  "src/kv_manager.rs"
  "src/hf_config.rs"
  "src/plan.rs"
  "src/retention.rs"
  "crates/orbitkv-ffi/Cargo.toml"
  "crates/orbitkv-ffi/Cargo.lock"
  "crates/orbitkv-ffi/include/orbitkv.h"
  "crates/orbitkv-ffi/src/lib.rs"
  "crates/orbitkv-ffi/src/manager.rs"
  "integrations/sglang/bench_canonical_manager.py"
  "integrations/sglang/checkpoint_identity.py"
  "integrations/sglang/pyproject.toml"
  "integrations/sglang/prepare_pinned_checkout.py"
  "integrations/sglang/patches/v0.5.17-orbitkv-fail-closed.patch"
  "integrations/sglang/src/orbitkv_sglang/__init__.py"
  "integrations/sglang/src/orbitkv_sglang/config.py"
  "integrations/sglang/src/orbitkv_sglang/ffi.py"
  "integrations/sglang/src/orbitkv_sglang/pinned.py"
  "integrations/sglang/src/orbitkv_sglang/plugin.py"
  "integrations/sglang/src/orbitkv_sglang/runtime.py"
)

readonly -a QUAL_ABI5_SYMBOLS=(
  "orbitkv_abi_version"
  "orbitkv_manager_abort_steps"
  "orbitkv_manager_acknowledge_reclamations"
  "orbitkv_manager_arena_identities"
  "orbitkv_manager_arena_stats"
  "orbitkv_manager_complete_batch"
  "orbitkv_manager_create"
  "orbitkv_manager_destroy"
  "orbitkv_manager_prepare_batch"
  "orbitkv_manager_quarantine_steps"
  "orbitkv_manager_quarantine_submissions"
  "orbitkv_manager_recycle_requests"
  "orbitkv_manager_release_batch"
  "orbitkv_manager_request_acquire_batch"
  "orbitkv_manager_stats"
  "orbitkv_manager_submit_batch"
)

die() {
  printf 'qualification error: %s\n' "$*" >&2
  exit 1
}

require_regular_file() {
  [[ -f "$1" && ! -L "$1" ]] || die "required regular file is missing or symlinked: $1"
}

require_executable_file() {
  [[ -f "$1" && -x "$1" ]] || die "required executable file is missing: $1"
}

require_directory() {
  [[ -d "$1" && ! -L "$1" ]] || die "required directory is missing or symlinked: $1"
}

check_sha256() {
  local expected_sha="$1"
  local artifact_file="$2"
  local actual_sha
  require_regular_file "$artifact_file"
  actual_sha="$(sha256sum "$artifact_file" | awk '{print $1}')"
  [[ "$actual_sha" == "$expected_sha" ]] || die "SHA-256 mismatch for $artifact_file: $actual_sha"
}

source_closure_sha256() {
  local relative_file
  local file_sha
  for relative_file in "${QUAL_SOURCE_FILES[@]}"; do
    require_regular_file "${QUAL_ROOT}/${relative_file}"
    file_sha="$(sha256sum "${QUAL_ROOT}/${relative_file}" | awk '{print $1}')"
    printf '%s  %s\n' "$file_sha" "$relative_file"
  done | sha256sum | awk '{print $1}'
}

verify_crate_direct_modules_covered() {
  local crate_lib="$1"
  local crate_source_dir="${crate_lib%/lib.rs}"
  local module_name
  local file_candidate
  local directory_candidate
  local module_file
  local filesystem_matches
  local closure_matches
  local closure_entry

  while IFS= read -r module_name; do
    file_candidate="${crate_source_dir}/${module_name}.rs"
    directory_candidate="${crate_source_dir}/${module_name}/mod.rs"
    module_file=""
    filesystem_matches=0
    if [[ -f "${QUAL_ROOT}/${file_candidate}" ]]; then
      module_file="$file_candidate"
      filesystem_matches=$((filesystem_matches + 1))
    fi
    if [[ -f "${QUAL_ROOT}/${directory_candidate}" ]]; then
      module_file="$directory_candidate"
      filesystem_matches=$((filesystem_matches + 1))
    fi
    [[ "$filesystem_matches" -eq 1 ]] || \
      die "direct Rust module $module_name from $crate_lib is missing or ambiguous"

    closure_matches=0
    for closure_entry in "${QUAL_SOURCE_FILES[@]}"; do
      if [[ "$closure_entry" == "$module_file" ]]; then
        closure_matches=$((closure_matches + 1))
      fi
    done
    [[ "$closure_matches" -eq 1 ]] || \
      die "direct Rust module $module_file is not covered exactly once by the source closure"
  done < <(
    sed -nE \
      's/^[[:space:]]*(pub(\([^)]*\))?[[:space:]]+)?mod[[:space:]]+([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*;.*/\3/p' \
      "${QUAL_ROOT}/${crate_lib}"
  )
}

verify_source_manifest() {
  local relative_file
  local -A seen_files=()

  for relative_file in "${QUAL_SOURCE_FILES[@]}"; do
    [[ -z "${seen_files[$relative_file]+present}" ]] || \
      die "duplicate source-closure entry: $relative_file"
    seen_files["$relative_file"]=1
  done
  verify_crate_direct_modules_covered "src/lib.rs"
  verify_crate_direct_modules_covered "crates/orbitkv-ffi/src/lib.rs"
}

require_final_hash_locks() {
  [[ "$QUAL_SOURCE_CLOSURE_SHA256" =~ ^[0-9a-f]{64}$ ]] || \
    die "final ABI5 source-closure hash is not locked"
  [[ "$QUAL_LIBRARY_SHA256" =~ ^[0-9a-f]{64}$ ]] || \
    die "final ABI5 release-library hash is not locked"
}

verify_release_surface() {
  local actual_closure
  local -a actual_symbols
  local symbol_index

  require_final_hash_locks
  verify_source_manifest
  actual_closure="$(source_closure_sha256)"
  [[ "$actual_closure" == "$QUAL_SOURCE_CLOSURE_SHA256" ]] || \
    die "source closure changed: $actual_closure"
  check_sha256 "$QUAL_LIBRARY_SHA256" "$QUAL_LIBRARY"

  mapfile -t actual_symbols < <(
    nm -D --defined-only "$QUAL_LIBRARY" |
      awk '$3 ~ /^orbitkv_/ {print $3}' |
      LC_ALL=C sort
  )
  [[ "${#actual_symbols[@]}" -eq "${#QUAL_ABI5_SYMBOLS[@]}" ]] || \
    die "release library does not export exactly 16 orbitkv_* symbols"
  for symbol_index in "${!QUAL_ABI5_SYMBOLS[@]}"; do
    [[ "${actual_symbols[$symbol_index]}" == "${QUAL_ABI5_SYMBOLS[$symbol_index]}" ]] || \
      die "release symbol mismatch at index $symbol_index"
  done
}

verify_static_inputs() {
  # A standard venv Python is a symlink to the base interpreter; its runtime
  # prefix and executable identity are validated again in Python below.
  require_executable_file "$QUAL_PYTHON"
  require_regular_file "$QUAL_BENCH"
  require_directory "$QUAL_STOCK_ROOT"
  require_directory "$QUAL_MANAGER_ROOT"
  require_directory "$QUAL_QWEN_MODEL"
  require_directory "$QUAL_GPT_MODEL"

  check_sha256 "$QUAL_REQUIREMENTS_SHA256" "$QUAL_REQUIREMENTS_LOCK"
  check_sha256 "$QUAL_QWEN_PLAN_SHA256" "$QUAL_QWEN_PLAN"
  check_sha256 "$QUAL_GPT_PLAN_SHA256" "$QUAL_GPT_PLAN"
  check_sha256 "$QUAL_QWEN_CONFIG_SHA256" "${QUAL_QWEN_MODEL}/config.json"
  check_sha256 "$QUAL_QWEN_INDEX_SHA256" "${QUAL_QWEN_MODEL}/model.safetensors.index.json"
  check_sha256 "$QUAL_GPT_CONFIG_SHA256" "${QUAL_GPT_MODEL}/config.json"
  check_sha256 "$QUAL_GPT_INDEX_SHA256" "${QUAL_GPT_MODEL}/model.safetensors.index.json"

  [[ "$(git -C "$QUAL_STOCK_ROOT" rev-parse HEAD)" == "$QUAL_SGLANG_REVISION" ]] || \
    die "stock checkout is not official SGLang v0.5.17"
  [[ "$(git -C "$QUAL_MANAGER_ROOT" rev-parse HEAD)" == "$QUAL_SGLANG_REVISION" ]] || \
    die "manager checkout is not official SGLang v0.5.17"
  [[ "$(git -C "$QUAL_STOCK_ROOT" describe --tags --exact-match HEAD)" == "v0.5.17" ]] || \
    die "stock checkout does not peel to tag v0.5.17"
  [[ "$(git -C "$QUAL_MANAGER_ROOT" describe --tags --exact-match HEAD)" == "v0.5.17" ]] || \
    die "manager checkout does not peel to tag v0.5.17"
  [[ "$(git -C "$QUAL_STOCK_ROOT" remote get-url origin)" == "https://github.com/sgl-project/sglang.git" ]] || \
    die "stock checkout does not use the official SGLang remote"
  [[ "$(git -C "$QUAL_MANAGER_ROOT" remote get-url origin)" == "https://github.com/sgl-project/sglang.git" ]] || \
    die "manager checkout does not use the official SGLang remote"

  env -u PYTHONPATH PATH="$QUAL_PATH" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    CUDA_VISIBLE_DEVICES= "$QUAL_PYTHON" \
    "${QUAL_ROOT}/integrations/sglang/prepare_pinned_checkout.py" \
    check-base --sglang-root "$QUAL_STOCK_ROOT"
  env -u PYTHONPATH PATH="$QUAL_PATH" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    CUDA_VISIBLE_DEVICES= "$QUAL_PYTHON" \
    "${QUAL_ROOT}/integrations/sglang/prepare_pinned_checkout.py" \
    verify --sglang-root "$QUAL_MANAGER_ROOT"

  env -u PYTHONPATH PATH="$QUAL_PATH" PYTHONNOUSERSITE=1 \
    "$QUAL_PYTHON" -m pip check
  diff -q "$QUAL_REQUIREMENTS_LOCK" <(
    env -u PYTHONPATH PATH="$QUAL_PATH" PYTHONNOUSERSITE=1 \
      "$QUAL_PYTHON" -m pip freeze --all
  ) >/dev/null || die "release venv differs from the frozen requirements lock"
}

verify_python_contracts() {
  env -u PYTHONPATH \
    PATH="$QUAL_PATH" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    CUDA_VISIBLE_DEVICES= QUAL_ROOT="$QUAL_ROOT" QUAL_LIBRARY="$QUAL_LIBRARY" \
    QUAL_STOCK_ROOT="$QUAL_STOCK_ROOT" QUAL_MANAGER_ROOT="$QUAL_MANAGER_ROOT" \
    QUAL_QWEN_MODEL="$QUAL_QWEN_MODEL" QUAL_GPT_MODEL="$QUAL_GPT_MODEL" \
    QUAL_QWEN_PLAN="$QUAL_QWEN_PLAN" QUAL_GPT_PLAN="$QUAL_GPT_PLAN" \
    "$QUAL_PYTHON" - <<'PY'
from __future__ import annotations

import ctypes
import os
import sys
from argparse import Namespace
from pathlib import Path

root = Path(os.environ["QUAL_ROOT"])
integration = root / "integrations/sglang"
sys.path.insert(0, str(integration))
sys.path.insert(0, str(integration / "src"))
assert Path(sys.prefix).resolve() == (root / ".venv-sglang-v0517").resolve()
assert Path(sys.executable).absolute() == root / ".venv-sglang-v0517/bin/python"

import bench_canonical_manager as bench
from orbitkv_sglang.config import load_config
from orbitkv_sglang.ffi import CtypesManagerFactory
from orbitkv_sglang.runtime import ArenaRegistration, ManagerCreateSettings

expected_counters = (
    "request_acquire_batch_calls",
    "prepare_batch_calls",
    "submit_batch_calls",
    "complete_batch_calls",
    "release_batch_calls",
    "acknowledge_reclamations_calls",
    "recycle_requests_calls",
    "abort_steps_calls",
    "quarantine_steps_calls",
    "quarantine_submissions_calls",
    "acquired_items",
    "prepared_items",
    "submitted_items",
    "completed_items",
    "released_items",
    "hot_workspace_allocations",
    "capacity_memset_bytes",
    "root_entries_crossed",
    "materialized_page_objects",
    "buffer_too_small_failures",
    "forward_events",
    "completion_values",
    "event_queries",
    "event_waits",
    "quarantine_count",
    "fail_stop_count",
)
assert bench.RECORD_SCHEMA == "orbitkv.sglang-v0517-full-hybrid-single-run.v5"
assert bench._BATCH_COUNTER_FIELDS == expected_counters
assert bench.verify_pinned_module_constants()["revision"] == bench.SUPPORTED_SGLANG_REVISION
bench.verify_sglang_source(Path(os.environ["QUAL_STOCK_ROOT"]), "stock")
bench.verify_sglang_source(Path(os.environ["QUAL_MANAGER_ROOT"]), "manager")
bench.build_tool_identity()
bench.verify_manager_entrypoint()
bench.verify_stock_plugin_selection()

library_path = Path(os.environ["QUAL_LIBRARY"])
library = ctypes.CDLL(str(library_path))
library.orbitkv_abi_version.argtypes = []
library.orbitkv_abi_version.restype = ctypes.c_uint32
assert library.orbitkv_abi_version() == 5

cases = (
    ("QUAL_QWEN_MODEL", "QUAL_QWEN_PLAN", "flashinfer", 1, 528, 1024, 1, "full"),
    ("QUAL_QWEN_MODEL", "QUAL_QWEN_PLAN", "flashinfer", 4, 2112, 4096, 5, "full"),
    ("QUAL_GPT_MODEL", "QUAL_GPT_PLAN", "fa3", 1, 528, 1024, 1, "hybrid_full_swa"),
    ("QUAL_GPT_MODEL", "QUAL_GPT_PLAN", "fa3", 4, 2112, 4096, 5, "hybrid_full_swa"),
)
factory = CtypesManagerFactory()
for model_env, plan_env, backend, batch, chunk, capacity, iterations, profile in cases:
    model = Path(os.environ[model_env])
    plan = Path(os.environ[plan_env])
    aligned_prompt_batch = batch * ((513 + 15) // 16) * 16
    assert chunk == aligned_prompt_batch
    config = load_config(
        {"ORBITKV_PLAN": str(plan), "ORBITKV_LIBRARY": str(library_path)}
    )
    contract, checkpoint = bench.checkpoint_contract(model, config)
    assert contract["attention_profile"] == profile
    assert checkpoint["indexed_weights_complete"] is True
    arguments = Namespace(
        mode="manager",
        sglang_root=os.environ["QUAL_MANAGER_ROOT"],
        model=str(model),
        plan=str(plan),
        library=str(library_path),
        requests=batch,
        max_running_requests=batch,
        prompt_tokens=513,
        decode_tokens=33,
        iterations=iterations,
        chunked_prefill_size=chunk,
        context_length=1024,
        max_total_tokens=capacity,
        mem_fraction_static=None,
        attention_backend=backend,
        seed=20260820,
    )
    bench.validate_arguments(arguments)
    engine = bench.engine_arguments(arguments, model, contract)
    assert engine["attention_backend"] == backend
    assert engine["chunked_prefill_size"] == chunk
    assert engine["max_running_requests"] == batch
    assert engine["max_total_tokens"] == capacity
    assert engine["enable_deterministic_inference"] is True
    assert engine["sampling_backend"] == "pytorch"
    if profile == "hybrid_full_swa":
        assert engine["moe_runner_backend"] == "triton"
    else:
        assert "moe_runner_backend" not in engine
    arena_tokens = []
    if config.sliding_class is not None:
        minimum = config.sliding_class.minimum_sliding_pool_tokens(
            maximum_running_requests=batch,
            chunked_prefill_tokens=chunk,
        )
        derived_swa_cap = min(
            ((288 * batch + chunk + 16 + 15) // 16) * 16,
            capacity,
        )
        assert derived_swa_cap >= minimum
    else:
        derived_swa_cap = None
    for class_config in config.classes:
        arena_tokens.append(
            capacity if class_config.retention == "full" else derived_swa_cap
        )
    assert all(value is not None and value % 16 == 0 for value in arena_tokens)
    resolved_state = dict(engine)
    resolved_state["memory_usage"] = {
        "token_capacity": capacity,
        "token_capacity_swa": derived_swa_cap,
    }
    readback = bench.verify_runtime_contract(
        arguments,
        {"internal_states": [resolved_state]},
        contract,
    )
    assert readback["requested_max_total_tokens"] == capacity
    assert readback["full_tokens"] == capacity
    assert readback["swa_tokens"] == derived_swa_cap
    page_counts = tuple(int(value) // 16 for value in arena_tokens)
    registrations = tuple(
        ArenaRegistration(index, index + 1, index + 1, page_count)
        for index, page_count in enumerate(page_counts)
    )
    manager = factory.create(
        config,
        ManagerCreateSettings(batch, batch, sum(page_counts), chunk),
        registrations,
    )
    try:
        assert tuple(item.page_count for item in manager.arenas) == page_counts
        assert manager.stats().free_pages == sum(page_counts)
        assert all(item.free_pages == item.page_count for item in manager.arena_stats())
    finally:
        manager.destroy()

print("ABI5 v5 Python/source/plan/checkpoint contracts passed")
PY
}

verify_cuda_hidden_imports() {
  env -u PYTHONPATH \
    PATH="$QUAL_PATH" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    CUDA_VISIBLE_DEVICES= SGLANG_PLUGINS=orbitkv_stock_baseline_no_plugins \
    PYTHONPATH="${QUAL_STOCK_ROOT}/python:${QUAL_ROOT}/integrations/sglang/src" \
    "$QUAL_PYTHON" - <<'PY'
import sglang
from sglang.srt.plugins import HookRegistry, load_plugins

load_plugins()
assert sglang.__version__ == "0.5.17"
assert not HookRegistry._patched
print("CUDA-hidden stock import passed with zero OrbitKV hooks")
PY

  env -u PYTHONPATH \
    PATH="$QUAL_PATH" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    CUDA_VISIBLE_DEVICES= SGLANG_PLUGINS=orbitkv_manager \
    ORBITKV_SGLANG_ROOT="$QUAL_MANAGER_ROOT" ORBITKV_PLAN="$QUAL_QWEN_PLAN" \
    ORBITKV_LIBRARY="$QUAL_LIBRARY" \
    PYTHONPATH="${QUAL_MANAGER_ROOT}/python:${QUAL_ROOT}/integrations/sglang/src" \
    "$QUAL_PYTHON" - <<'PY'
import sglang
from sglang.srt.plugins import load_plugins

load_plugins()

from orbitkv_sglang import plugin
from sglang.srt.mem_cache import allocation, common
from sglang.srt.managers import schedule_batch, scheduler
from sglang.srt.managers.scheduler_components import batch_result_processor
from sglang.srt.plugins.hook_registry import HookRegistry

assert sglang.__version__ == "0.5.17"
assert len(plugin.HOOK_TARGETS) == 9
assert all(target in HookRegistry._patched for target in plugin.HOOK_TARGETS)
assert schedule_batch.alloc_for_extend is allocation.alloc_for_extend
assert schedule_batch.alloc_for_decode is allocation.alloc_for_decode
assert schedule_batch.release_kv_cache is common.release_kv_cache
assert scheduler.release_kv_cache is common.release_kv_cache
assert batch_result_processor.release_kv_cache is common.release_kv_cache
print("CUDA-hidden manager activation passed with 9 hooks and 5 aliases")
PY
}

host_preflight() {
  verify_static_inputs
  verify_release_surface
  verify_python_contracts
  verify_cuda_hidden_imports
  printf 'ABI5 v5 H20 host preflight passed; no GPU was enumerated or initialized.\n'
}

require_execution_gate() {
  local expected_gate
  expected_gate="GO:${QUAL_SOURCE_CLOSURE_SHA256}:${QUAL_LIBRARY_SHA256}"
  [[ "${ORBITKV_H20_EXECUTE-}" == "ABI5_V5_H20_QUALIFICATION" ]] || \
    die "GPU phase requires ORBITKV_H20_EXECUTE=ABI5_V5_H20_QUALIFICATION"
  [[ "${ORBITKV_ABI5_HOST_GATE-}" == "$expected_gate" ]] || \
    die "GPU phase requires the final source/library-bound host-performance GO token"
}

require_idle_h20() {
  local gpu_line
  local compute_pids
  gpu_line="$(
    nvidia-smi --id=0 \
      --query-gpu=name,uuid,memory.used,utilization.gpu \
      --format=csv,noheader,nounits
  )"
  [[ "$(printf '%s\n' "$gpu_line" | wc -l)" -eq 1 ]] || \
    die "device 0 did not resolve to exactly one GPU"
  [[ "$gpu_line" == *H20* ]] || die "device 0 is not an H20: $gpu_line"
  compute_pids="$(
    nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader,nounits
  )"
  [[ -z "${compute_pids//[[:space:]]/}" ]] || \
    die "H20 device 0 has active compute processes: $compute_pids"
  printf 'Dedicated H20 accepted: %s\n' "$gpu_line"
}

readonly -a QUAL_RUN_ENV=(
  /usr/bin/env -i
  "HOME=${HOME}"
  "USER=${USER-root}"
  "LOGNAME=${LOGNAME-${USER-root}}"
  "LANG=C.UTF-8"
  "LC_ALL=C.UTF-8"
  "PATH=${QUAL_PATH}"
  "LD_LIBRARY_PATH=${LD_LIBRARY_PATH-}"
  "CUDA_HOME=/usr/local/cuda"
  "CUDA_VISIBLE_DEVICES=0"
  "PYTHONNOUSERSITE=1"
  "PYTHONDONTWRITEBYTECODE=1"
  "PYTHONHASHSEED=0"
  "HF_HUB_OFFLINE=1"
  "TRANSFORMERS_OFFLINE=1"
  "TOKENIZERS_PARALLELISM=false"
  "SGLANG_SWA_EVICTION_INTERVAL=128"
)

run_case() {
  local slug="$1"
  local mode="$2"
  local sglang_root="$3"
  local model="$4"
  local plan="$5"
  local requests="$6"
  local chunk_tokens="$7"
  local capacity_tokens="$8"
  local iterations="$9"
  local backend="${10}"
  local output_file="${QUAL_RECORD_ROOT}/${slug}.json"
  local partial_file="${output_file}.partial"
  local stderr_file="${QUAL_RECORD_ROOT}/${slug}.stderr.log"
  local -a manager_arguments=()

  [[ ! -e "$output_file" && ! -e "$partial_file" && ! -e "$stderr_file" ]] || \
    die "refusing to overwrite an existing qualification record for $slug"
  if [[ "$mode" == "manager" ]]; then
    manager_arguments=(--plan "$plan" --library "$QUAL_LIBRARY")
  elif [[ "$mode" != "stock" ]]; then
    die "unknown qualification mode: $mode"
  fi

  "${QUAL_RUN_ENV[@]}" "$QUAL_PYTHON" "$QUAL_BENCH" \
    --mode "$mode" \
    --sglang-root "$sglang_root" \
    --model "$model" \
    "${manager_arguments[@]}" \
    --requests "$requests" \
    --max-running-requests "$requests" \
    --prompt-tokens 513 \
    --decode-tokens 33 \
    --iterations "$iterations" \
    --chunked-prefill-size "$chunk_tokens" \
    --context-length 1024 \
    --max-total-tokens "$capacity_tokens" \
    --attention-backend "$backend" \
    --seed 20260820 \
    >"$partial_file" 2>"$stderr_file"

  env -u PYTHONPATH PATH="$QUAL_PATH" PYTHONNOUSERSITE=1 \
    "$QUAL_PYTHON" -m json.tool "$partial_file" >/dev/null
  mv -- "$partial_file" "$output_file"
  printf 'Completed %s\n' "$output_file"
}

check_pair() {
  local stock_file="$1"
  local manager_file="$2"
  local batch_size="$3"
  local iterations="$4"
  local backend="$5"
  local chunk_tokens="$6"
  local capacity_tokens="$7"
  local profile="$8"

  require_regular_file "$stock_file"
  require_regular_file "$manager_file"
  env -u PYTHONPATH \
    PATH="$QUAL_PATH" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    QUAL_ROOT="$QUAL_ROOT" QUAL_LIBRARY="$QUAL_LIBRARY" \
    QUAL_LIBRARY_SHA256="$QUAL_LIBRARY_SHA256" \
    "$QUAL_PYTHON" - \
    "$stock_file" "$manager_file" "$batch_size" "$iterations" \
    "$backend" "$chunk_tokens" "$capacity_tokens" "$profile" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(os.environ["QUAL_ROOT"])
integration = root / "integrations/sglang"
sys.path.insert(0, str(integration))
sys.path.insert(0, str(integration / "src"))
import bench_canonical_manager as bench

stock_path = Path(sys.argv[1])
manager_path = Path(sys.argv[2])
batch = int(sys.argv[3])
iterations = int(sys.argv[4])
backend = sys.argv[5]
chunk = int(sys.argv[6])
capacity = int(sys.argv[7])
profile = sys.argv[8]
stock = json.loads(stock_path.read_text(encoding="utf-8"))
manager = json.loads(manager_path.read_text(encoding="utf-8"))

assert stock["schema"] == manager["schema"] == bench.RECORD_SCHEMA
assert stock["mode"] == "stock" and manager["mode"] == "manager"
assert stock["manager"] is None and isinstance(manager["manager"], dict)
assert stock["pairing"]["pair_key_sha256"] == manager["pairing"]["pair_key_sha256"]
assert stock["output_token_digest_sha256"] == manager["output_token_digest_sha256"]
assert stock["output_request_digests_sha256"] == manager["output_request_digests_sha256"]
assert stock["request_traces"] == manager["request_traces"]
assert stock["checkpoint_identity_sha256"] == manager["checkpoint_identity_sha256"]
assert stock["workload"] == manager["workload"]
assert stock["engine_args"] == manager["engine_args"]
assert stock["capacity_readback"] == manager["capacity_readback"]
assert stock["completed_requests"] == manager["completed_requests"] == batch * iterations
assert stock["completion_tokens"] == manager["completion_tokens"] == batch * iterations * 33
assert len(stock["iteration_seconds"]) == len(manager["iteration_seconds"]) == iterations
assert all(value > 0 for value in stock["iteration_seconds"] + manager["iteration_seconds"])
for record in (stock, manager):
    request_digests = record["output_request_digests_sha256"]
    assert isinstance(request_digests, list) and len(request_digests) == iterations
    assert all(isinstance(item, list) and len(item) == batch for item in request_digests)
    assert all(
        isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
        for item in request_digests
        for digest in item
    )

for record in (stock, manager):
    assert record["source_identity"]["release"] == "v0.5.17"
    assert record["source_identity"]["revision"] == "29481685462732237d80d86076d6563e1f658102"
    assert record["runtime_identity"]["sglang_version"] == "0.5.17"
    assert record["runtime_identity"]["attention_backend"] == backend
    assert record["runtime_identity"]["deterministic_inference"] is True
    assert record["runtime_identity"]["sampling_backend"] == "pytorch"
    assert record["runtime_identity"]["cache"] == "ChunkCache"
    assert record["runtime_identity"]["execution"] == "eager"
    assert record["checkpoint_contract"]["attention_profile"] == profile
    assert record["workload"]["requests"] == batch
    assert record["workload"]["max_running_requests"] == batch
    assert record["workload"]["iterations"] == iterations
    assert record["workload"]["decode_tokens"] == 33
    assert record["engine_args"]["chunked_prefill_size"] == chunk
    assert record["engine_args"]["max_total_tokens"] == capacity
    assert record["engine_args"]["enable_deterministic_inference"] is True
    assert record["engine_args"]["sampling_backend"] == "pytorch"
    expected_sampling = {
        "temperature": 0,
        "max_new_tokens": 33,
        "min_new_tokens": 33,
        "ignore_eos": True,
        "sampling_seed": record["workload"]["seed"],
    }
    assert record["sampling_params"] == expected_sampling
    assert record["pairing"]["contract"]["engine_args"] == record["engine_args"]
    assert record["pairing"]["contract"]["sampling_params"] == expected_sampling
    assert record["capacity_readback"]["requested_max_total_tokens"] == capacity
    assert record["capacity_readback"]["full_tokens"] == capacity

    expected_moe = "triton" if profile == "hybrid_full_swa" else None
    if expected_moe is None:
        assert "moe_runner_backend" not in record["engine_args"]
        assert record["runtime_identity"]["moe_runner_backend"] in (None, "auto")
    else:
        assert record["engine_args"]["moe_runner_backend"] == expected_moe
        assert record["runtime_identity"]["moe_runner_backend"] == expected_moe

    expected_prompts = bench.deterministic_input_ids(
        requests=batch,
        prompt_tokens=record["workload"]["prompt_tokens"],
        vocab_size=record["checkpoint_contract"]["vocab_size"],
        seed=record["workload"]["seed"],
    )
    expected_input_digests = [
        bench.canonical_digest(prompt) for prompt in expected_prompts
    ]
    traces = record["request_traces"]
    assert isinstance(traces, list) and len(traces) == iterations
    stable_outputs: list[list[int]] | None = None
    for iteration_index, trace_row in enumerate(traces):
        assert isinstance(trace_row, list) and len(trace_row) == batch
        row_outputs: list[list[int]] = []
        for request_index, trace in enumerate(trace_row):
            assert set(trace) == {
                "request_index",
                "submitted_rid",
                "submitted_input_ids_sha256",
                "returned_rid",
                "output_ids",
                "output_ids_sha256",
            }
            expected_rid = (
                f"orbitkv-canonical-{record['workload']['seed']}-"
                f"{iteration_index}-{request_index}"
            )
            assert trace["request_index"] == request_index
            assert trace["submitted_rid"] == trace["returned_rid"] == expected_rid
            assert (
                trace["submitted_input_ids_sha256"]
                == expected_input_digests[request_index]
            )
            output_ids = trace["output_ids"]
            assert isinstance(output_ids, list) and len(output_ids) == 33
            assert all(isinstance(token, int) and token >= 0 for token in output_ids)
            assert trace["output_ids_sha256"] == bench.canonical_digest(output_ids)
            assert (
                record["output_request_digests_sha256"][iteration_index][
                    request_index
                ]
                == trace["output_ids_sha256"]
            )
            row_outputs.append(output_ids)
        if stable_outputs is None:
            stable_outputs = row_outputs
        else:
            assert row_outputs == stable_outputs

assert stock["runtime_identity"]["moe_runner_backend"] == manager["runtime_identity"][
    "moe_runner_backend"
]

current_harness_sha = hashlib.sha256(Path(bench.__file__).read_bytes()).hexdigest()
current_adapter = bench._adapter_identity()
for record in (stock, manager):
    assert record["source_identity"]["harness_sha256"] == current_harness_sha
    assert record["source_identity"]["adapter"] == current_adapter

assert stock["source_identity"]["python_source_contract"] == "clean_pinned_head"
assert manager["source_identity"]["python_source_contract"] == "pinned_head_plus_canonical_loader_patch"
assert stock["source_identity"]["library"] is None
assert stock["source_identity"]["plan"] is None
library_identity = manager["source_identity"]["library"]
assert library_identity["path"] == os.environ["QUAL_LIBRARY"]
assert library_identity["sha256"] == os.environ["QUAL_LIBRARY_SHA256"]
assert manager["manager"]["library"] == library_identity
assert manager["manager"]["after_load"]["abi_version"] == 5
assert manager["manager"]["final_census"]["abi_version"] == 5

counters = manager["manager"]["final_census"]["batch_counters"]
expected_batch_calls = iterations * manager["workload"]["decode_tokens"]
expected_batch_items = expected_batch_calls * batch
for field in (
    "prepare_batch_calls",
    "submit_batch_calls",
    "complete_batch_calls",
    "forward_events",
    "completion_values",
):
    assert counters[field] == expected_batch_calls
for field in ("prepared_items", "submitted_items", "completed_items"):
    assert counters[field] == expected_batch_items
assert counters["request_acquire_batch_calls"] == iterations
assert counters["acquired_items"] == batch * iterations
assert counters["release_batch_calls"] == iterations
assert counters["released_items"] == batch * iterations
assert counters["recycle_requests_calls"] == iterations
for field in (
    "hot_workspace_allocations",
    "capacity_memset_bytes",
    "root_entries_crossed",
    "materialized_page_objects",
    "buffer_too_small_failures",
    "abort_steps_calls",
    "quarantine_steps_calls",
    "quarantine_submissions_calls",
    "quarantine_count",
    "fail_stop_count",
):
    assert counters[field] == 0

swa = manager["manager"]["final_census"]["swa_activity"]
if profile == "full":
    assert swa["status"] == "not_applicable"
    assert not swa["applicable"]
    assert not any(
        swa[field]
        for field in (
            "swa_retirement_certificates",
            "swa_pages_reclaimed",
            "swa_wrap_events",
        )
    )
    assert counters["acknowledge_reclamations_calls"] == iterations
else:
    assert swa["status"] == "exposed" and swa["applicable"]
    assert all(
        swa[field] > 0
        for field in (
            "swa_retirement_certificates",
            "swa_pages_reclaimed",
            "swa_wrap_events",
        )
    )
    assert counters["acknowledge_reclamations_calls"] >= iterations

print(f"paired ABI5 v5 records passed: {stock_path.name} / {manager_path.name}")
PY
}

check_b1_pairs() {
  check_pair \
    "${QUAL_RECORD_ROOT}/qwen2.5-7b-b1-stock.json" \
    "${QUAL_RECORD_ROOT}/qwen2.5-7b-b1-manager.json" \
    1 1 flashinfer 528 1024 full
  check_pair \
    "${QUAL_RECORD_ROOT}/gpt-oss-20b-b1-stock.json" \
    "${QUAL_RECORD_ROOT}/gpt-oss-20b-b1-manager.json" \
    1 1 fa3 528 1024 hybrid_full_swa
}

check_b4_pairs() {
  check_pair \
    "${QUAL_RECORD_ROOT}/qwen2.5-7b-b4-stock.json" \
    "${QUAL_RECORD_ROOT}/qwen2.5-7b-b4-manager.json" \
    4 5 flashinfer 2112 4096 full
  check_pair \
    "${QUAL_RECORD_ROOT}/gpt-oss-20b-b4-stock.json" \
    "${QUAL_RECORD_ROOT}/gpt-oss-20b-b4-manager.json" \
    4 5 fa3 2112 4096 hybrid_full_swa
}

run_b1() {
  run_case qwen2.5-7b-b1-stock stock "$QUAL_STOCK_ROOT" "$QUAL_QWEN_MODEL" "" 1 528 1024 1 flashinfer
  run_case qwen2.5-7b-b1-manager manager "$QUAL_MANAGER_ROOT" "$QUAL_QWEN_MODEL" "$QUAL_QWEN_PLAN" 1 528 1024 1 flashinfer
  run_case gpt-oss-20b-b1-stock stock "$QUAL_STOCK_ROOT" "$QUAL_GPT_MODEL" "" 1 528 1024 1 fa3
  run_case gpt-oss-20b-b1-manager manager "$QUAL_MANAGER_ROOT" "$QUAL_GPT_MODEL" "$QUAL_GPT_PLAN" 1 528 1024 1 fa3
  check_b1_pairs
}

run_b4() {
  check_b1_pairs
  run_case qwen2.5-7b-b4-stock stock "$QUAL_STOCK_ROOT" "$QUAL_QWEN_MODEL" "" 4 2112 4096 5 flashinfer
  run_case qwen2.5-7b-b4-manager manager "$QUAL_MANAGER_ROOT" "$QUAL_QWEN_MODEL" "$QUAL_QWEN_PLAN" 4 2112 4096 5 flashinfer
  run_case gpt-oss-20b-b4-stock stock "$QUAL_STOCK_ROOT" "$QUAL_GPT_MODEL" "" 4 2112 4096 5 fa3
  run_case gpt-oss-20b-b4-manager manager "$QUAL_MANAGER_ROOT" "$QUAL_GPT_MODEL" "$QUAL_GPT_PLAN" 4 2112 4096 5 fa3
  check_b4_pairs
}

action="${1-preflight}"
[[ "$#" -le 1 ]] || die "usage: $0 [preflight|b1|b4]"
case "$action" in
  preflight)
    host_preflight
    ;;
  b1)
    host_preflight
    require_execution_gate
    require_idle_h20
    mkdir -p "$QUAL_RECORD_ROOT"
    require_directory "$QUAL_RECORD_ROOT"
    run_b1
    ;;
  b4)
    host_preflight
    require_execution_gate
    require_idle_h20
    require_directory "$QUAL_RECORD_ROOT"
    run_b4
    ;;
  *)
    die "usage: $0 [preflight|b1|b4]"
    ;;
esac
