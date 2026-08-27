#!/usr/bin/env python3
"""Clean-source sealed ABI8 H20 token-relocation qualification.

The default action is a host-only preflight.  GPU work is available only via
an explicit execution token.  A successful seal qualifies correctness and
lifecycle behavior for the frozen B1/B4 Full-attention matrix; it never
attests the hardware identity and never opens a performance gate.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence


INTEGRATION_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = INTEGRATION_ROOT.parents[1]
SOURCE_ROOT = INTEGRATION_ROOT / "src"
TOOLS_ROOT = REPOSITORY_ROOT / "tools"
sys.dont_write_bytecode = True
for candidate in (INTEGRATION_ROOT, SOURCE_ROOT, TOOLS_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import bench_token_relocation as benchmark  # noqa: E402
import qualify_abi8_h20 as abi8  # noqa: E402
import verify_token_relocation_h20_evidence as evidence  # noqa: E402
from checkpoint_identity import checkpoint_identity  # noqa: E402
from orbitkv_sglang import pinned, qualification_primitives  # noqa: E402


PREFLIGHT_SCHEMA = "orbitkv.abi8-h20-token-relocation-preflight.v1"
PAIR_SCHEMA = "orbitkv.abi8-h20-token-relocation-pair-verification.v1"
SUMMARY_SCHEMA = "orbitkv.abi8-h20-token-relocation-multi-epoch-summary.v1"
MANIFEST_SCHEMA = "orbitkv.abi8-h20-token-relocation-sealed-manifest.v1"
SEAL_VERIFICATION_SCHEMA = (
    "orbitkv.abi8-h20-token-relocation-seal-verification.v1"
)
MODEL_PROVENANCE_SCHEMA = "orbitkv.qwen25-0.5b-model-provenance.v1"
EXECUTION_TOKEN = "ABI8_H20_TOKEN_RELOCATION_QUALIFICATION"
QUALIFICATION_SCOPE = "token_relocation_scoped_correctness_lifecycle"
QUALIFICATION_STATUS = (
    "abi8_sglang_full_token_relocation_correctness_lifecycle_"
    "qualified_performance_pending"
)
QUALIFICATION_CLAIM = "scoped_correctness_and_lifecycle_only"
EVIDENCE_CLASS = "sealed_clean_source_scoped_qualification"
SGLANG_REVISION = "29481685462732237d80d86076d6563e1f658102"
REQUIREMENTS_SHA256 = (
    "472d8f63cad22cd7ac4908059562bebde5e54b8d2432f750640a14d525d2fa97"
)
ORBITKV_EDITABLE_TEMPLATE = (
    "-e git+https://github.com/feichai0017/orbitkv.git@{commit}"
    "#egg=orbitkv_sglang&subdirectory=integrations/sglang"
)
PLAN_SHA256 = (
    "6e51963cc09c45a26446621a2d164386ed2e1aed19de585a168db4267ea6cb8a"
)
MODEL_CONFIG_SHA256 = (
    "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45"
)
MODEL_WEIGHT_SHA256 = (
    "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
)
MODEL_WEIGHT_BYTES = 988_097_824
MODEL_NAME = "qwen2.5-0.5b"
MODEL_DIRECTORY_NAME = "qwen2.5-0.5b-instruct"
PLAN_NAME = "qwen2.5-0.5b-full-page16-bf16.json"
EPOCH_COUNT = 4
ITERATIONS = 5
MODES = ("naive", "relocate")
COMPONENT_CASES = (
    "test_opaque_payload_coordinates_are_embedded_without_collisions",
    *(
        "test_stale_member_makes_append_and_relocation_prepare_"
        f"failure_atomic[{batch}]"
        for batch in (1, 4, 32)
    ),
    *(
        "test_real_cuda_opaque_token_relocation_conformance" f"[{batch}]"
        for batch in (1, 4, 32)
    ),
)
SOURCE_REQUIRED_PATHS = frozenset(
    {
        "integrations/sglang/qualify_token_relocation_h20.py",
        "integrations/sglang/qualify_abi8_h20.py",
        "integrations/sglang/bench_token_relocation.py",
        "integrations/sglang/bench_canonical_manager.py",
        "integrations/sglang/checkpoint_identity.py",
        "integrations/sglang/prepare_pinned_checkout.py",
        "integrations/sglang/pyproject.toml",
        "integrations/sglang/patches/v0.5.17-orbitkv-fail-closed.patch",
        "tools/verify_token_relocation_h20_evidence.py",
        "tools/verify_token_relocation_h20_seal.py",
    }
)


@dataclass(frozen=True)
class Case:
    batch: int
    capacity_tokens: int

    @property
    def slug(self) -> str:
        return f"{MODEL_NAME}-b{self.batch}"


CASES = (Case(1, 128), Case(4, 512))


def execution_order(epoch: int) -> tuple[str, str]:
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
        raise ValueError("epoch must be a positive integer")
    return MODES if epoch % 2 else tuple(reversed(MODES))


def sha256_file(path: Path) -> str:
    return qualification_primitives.sha256_file(path)


def canonical_digest(value: Any) -> str:
    try:
        return qualification_primitives.canonical_json_sha256(value)
    except ValueError as error:
        if isinstance(error.__cause__, (TypeError, ValueError)):
            raise error.__cause__
        raise


def _run(
    arguments: Sequence[str], *, cwd: Path = REPOSITORY_ROOT,
    env: dict[str, str] | None = None, timeout: int = 600,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(arguments), cwd=cwd, env=env, check=True,
            capture_output=True, text=True, timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as error:
        detail = getattr(error, "stderr", None) or str(error)
        raise RuntimeError(
            f"command failed: {' '.join(str(value) for value in arguments)}: "
            f"{detail}"
        ) from error


def _strict_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"JSON input is not a regular file: {path}")
    try:
        value = qualification_primitives.parse_strict_json_object(
            path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, ValueError) as error:
        if (
            isinstance(error, ValueError)
            and error.__cause__ is None
            and str(error) == "strict JSON value must be an object"
        ):
            raise RuntimeError(f"JSON value is not an object: {path}") from None
        detail = error.__cause__ if isinstance(error, ValueError) else None
        message = str(detail or error)
        message = message.replace(
            "duplicate JSON object key ", "duplicate key ", 1
        ).replace("non-finite JSON number ", "non-finite number ", 1)
        raise RuntimeError(
            f"cannot load strict JSON {path}: {message}"
        ) from (detail or error)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON value is not an object: {path}")
    return value


def _write_new(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _require_hash(path: Path, expected: str, label: str) -> dict[str, Any]:
    requested = path.expanduser().absolute()
    if requested.is_symlink():
        raise RuntimeError(f"{label} is not a regular file")
    resolved = requested.resolve(strict=True)
    if not resolved.is_file():
        raise RuntimeError(f"{label} is not a regular file")
    observed = sha256_file(resolved)
    if observed != expected:
        raise RuntimeError(f"{label} SHA-256 mismatch: {observed}")
    return {
        "path": str(resolved), "bytes": resolved.stat().st_size,
        "sha256": observed,
    }


def _source_closure(source: dict[str, Any]) -> list[dict[str, str]]:
    inventory = source.get("inventory")
    if not isinstance(inventory, list):
        raise RuntimeError("source inventory is missing")
    indexed = {
        item.get("path"): item.get("sha256")
        for item in inventory if isinstance(item, dict)
    }
    adapter_paths = {
        path for path in indexed
        if isinstance(path, str)
        and path.startswith("integrations/sglang/src/orbitkv_sglang/")
        and path.endswith((".py", ".json"))
    }
    paths = SOURCE_REQUIRED_PATHS | adapter_paths
    missing = sorted(path for path in paths if path not in indexed)
    if missing:
        raise RuntimeError(
            "tracked source omits qualification closure: " + ", ".join(missing)
        )
    return [{"path": path, "sha256": indexed[path]} for path in sorted(paths)]


def _model_identity(path: Path) -> dict[str, Any]:
    root = path.expanduser().resolve(strict=True)
    if not root.is_dir() or root.name != MODEL_DIRECTORY_NAME:
        raise RuntimeError(
            f"model must be the pinned {MODEL_DIRECTORY_NAME} checkpoint"
        )
    identity = checkpoint_identity(root, "auto")
    expected_weight = {
        "name": "model.safetensors",
        "bytes": MODEL_WEIGHT_BYTES,
        "sha256": MODEL_WEIGHT_SHA256,
    }
    if (
        identity.get("config_sha256") != MODEL_CONFIG_SHA256
        or identity.get("index_files") != []
        or identity.get("weight_files") != [expected_weight]
        or identity.get("weight_bytes") != MODEL_WEIGHT_BYTES
        or identity.get("indexed_weights_complete") is not True
        or identity.get("missing_indexed_weights") != []
    ):
        raise RuntimeError("model identity is not the pinned Qwen2.5-0.5B checkpoint")
    return {
        "name": MODEL_NAME, "root": str(root),
        "checkpoint": identity,
        "identity_sha256": canonical_digest(identity),
    }


def _manager_checkout_identity(root: Path) -> dict[str, Any]:
    checkout = pinned.validate_patched_checkout(root)
    identity = abi8._sglang_checkout_identity(checkout, "manager")
    status = _run(
        (
            "git", "-C", str(checkout), "status", "--porcelain=v1",
            "-z", "--untracked-files=all",
        )
    ).stdout
    expected_status = f" M {pinned.PATCHED_SOURCE_PATH}\0"
    if (
        identity.get("revision") != SGLANG_REVISION
        or identity.get("tag") != "v0.5.17"
        or identity.get("remote") != "https://github.com/sgl-project/sglang.git"
        or status != expected_status
    ):
        raise RuntimeError(
            "manager checkout is not exact official patched SGLang v0.5.17"
        )
    return identity


def _require_active_editable(
    python_identity: dict[str, Any], source_commit: str
) -> str:
    expected = ORBITKV_EDITABLE_TEMPLATE.format(commit=source_commit)
    if python_identity.get("active_editable") != expected:
        raise RuntimeError(
            "active OrbitKV editable does not point at the clean source commit"
        )
    return expected


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    work_dir = args.work_dir.expanduser().resolve()
    if work_dir.exists() or work_dir.is_symlink():
        raise RuntimeError(f"refusing to overwrite existing work directory: {work_dir}")
    if work_dir.is_relative_to(REPOSITORY_ROOT):
        relative_work_dir = work_dir.relative_to(REPOSITORY_ROOT)
        ignored = subprocess.run(
            ("git", "check-ignore", "-q", "--no-index", "--",
             relative_work_dir.as_posix()),
            cwd=REPOSITORY_ROOT, check=False, capture_output=True, timeout=10,
        )
        if ignored.returncode != 0:
            raise RuntimeError(
                "work directory inside the source repository must be Git-ignored"
            )
    source = abi8.source_identity()
    closure = _source_closure(source)
    if pinned.SUPPORTED_SGLANG_REVISION != SGLANG_REVISION:
        raise RuntimeError("adapter pinned SGLang revision drifted")
    manager = pinned.validate_patched_checkout(args.manager_root)
    manager_identity = _manager_checkout_identity(manager)
    pinned_contract = abi8.benchmark.verify_pinned_module_constants()
    manager_entrypoint = abi8.benchmark.verify_manager_entrypoint()
    requirements = _require_hash(
        args.requirements, REQUIREMENTS_SHA256, "requirements lock"
    )
    plan = _require_hash(args.plan, PLAN_SHA256, "token-manager plan")
    model = _model_identity(args.model)
    python_identity = abi8._verify_python_environment(
        args.python, Path(requirements["path"])
    )
    _require_active_editable(python_identity, source["commit"])
    cargo_executable = shutil.which(args.cargo)
    if cargo_executable is None:
        raise RuntimeError(f"Cargo executable is unavailable: {args.cargo}")
    # Keep the cargo proxy basename: resolving the rustup-managed symlink would
    # invoke `rustup build` instead of dispatching the `cargo` tool.
    cargo_executable = str(Path(cargo_executable).absolute())
    work_dir.mkdir(parents=True)
    library_path, build = abi8._build_library(work_dir, cargo_executable)
    library = abi8.library_identity(library_path)
    record = {
        "schema": PREFLIGHT_SCHEMA,
        "qualification_scope": QUALIFICATION_SCOPE,
        "status": "host_preflight_passed_gpu_not_initialized",
        "source": source,
        "source_closure": closure,
        "benchmark": {
            "path": str(Path(benchmark.__file__).resolve()),
            "sha256": sha256_file(Path(benchmark.__file__).resolve()),
            "record_schema": benchmark.RECORD_SCHEMA,
        },
        "verifier": {
            "path": str(Path(evidence.__file__).resolve()),
            "sha256": sha256_file(Path(evidence.__file__).resolve()),
        },
        "library": library,
        "build": build,
        "sglang": {
            "release": pinned.SUPPORTED_SGLANG_RELEASE,
            "revision": SGLANG_REVISION,
            "manager_root": str(manager),
            "manager": manager_identity,
            "pinned_contract": pinned_contract,
            "manager_entrypoint": manager_entrypoint,
        },
        "inputs": {
            "requirements": requirements, "model": model, "plan": plan
        },
        "python": python_identity,
    }
    _write_new(work_dir / "preflight.json", record)
    return record


def _validate_preflight(record: dict[str, Any]) -> None:
    if (
        record.get("schema") != PREFLIGHT_SCHEMA
        or record.get("qualification_scope") != QUALIFICATION_SCOPE
        or record.get("status")
        != "host_preflight_passed_gpu_not_initialized"
    ):
        raise RuntimeError("work directory has no token-relocation preflight")
    current_source = abi8.source_identity()
    if current_source != record.get("source"):
        raise RuntimeError("active source differs from clean preflight identity")
    if _source_closure(current_source) != record.get("source_closure"):
        raise RuntimeError("qualification source closure differs from preflight")
    benchmark_identity = record.get("benchmark", {})
    if (
        benchmark_identity.get("record_schema") != benchmark.RECORD_SCHEMA
        or sha256_file(Path(benchmark_identity.get("path", "")))
        != benchmark_identity.get("sha256")
    ):
        raise RuntimeError("relocation benchmark differs from preflight")
    verifier_identity = record.get("verifier", {})
    if sha256_file(Path(verifier_identity.get("path", ""))) != verifier_identity.get("sha256"):
        raise RuntimeError("trusted relocation verifier differs from preflight")
    if abi8.library_identity(Path(record["library"]["path"])) != record["library"]:
        raise RuntimeError("ABI8 library differs from preflight")
    build = record.get("build")
    command = build.get("command") if isinstance(build, dict) else None
    if (
        not isinstance(command, list)
        or len(command) < 5
        or not isinstance(command[0], str)
        or not Path(command[0]).is_absolute()
    ):
        raise RuntimeError("preflight build has no absolute Cargo executable")
    cargo = Path(command[0]).absolute()
    if not cargo.is_file():
        raise RuntimeError("preflight Cargo executable is not a regular file")
    if _run((str(cargo), "--version")).stdout.strip() != build.get(
        "cargo_version"
    ):
        raise RuntimeError("Cargo toolchain differs from preflight")
    target = Path(build.get("cargo_target_dir", "")).resolve(strict=True)
    if Path(record["library"]["path"]).resolve(strict=True) != (
        target / "release/liborbitkv_ffi.so"
    ).resolve(strict=True):
        raise RuntimeError("preflight library is outside its Cargo target")
    requirements = record["inputs"]["requirements"]
    _require_hash(Path(requirements["path"]), requirements["sha256"], "requirements lock")
    plan = record["inputs"]["plan"]
    _require_hash(Path(plan["path"]), plan["sha256"], "token-manager plan")
    if _model_identity(Path(record["inputs"]["model"]["root"])) != record["inputs"]["model"]:
        raise RuntimeError("model differs from preflight identity")
    python = abi8._verify_python_environment(
        Path(record["python"]["executable"]), Path(requirements["path"])
    )
    if python != record["python"]:
        raise RuntimeError("Python environment differs from preflight")
    _require_active_editable(python, current_source["commit"])
    manager = pinned.validate_patched_checkout(record["sglang"]["manager_root"])
    if _manager_checkout_identity(manager) != record["sglang"]["manager"]:
        raise RuntimeError("manager SGLang checkout differs from preflight")
    if (
        record["sglang"].get("release") != "v0.5.17"
        or record["sglang"].get("revision") != SGLANG_REVISION
        or record["sglang"].get("pinned_contract")
        != abi8.benchmark.verify_pinned_module_constants()
        or record["sglang"].get("manager_entrypoint")
        != abi8.benchmark.verify_manager_entrypoint()
    ):
        raise RuntimeError("pinned SGLang or plugin identity differs from preflight")


def _fresh_environment(
    python: str, *, cargo: str | None = None, cargo_target_dir: str | None = None
) -> dict[str, str]:
    python_bin = str(Path(python).absolute().parent)
    cargo_bin = (
        str(Path(cargo).absolute().parent)
        if cargo is not None
        else None
    )
    path_entries = [python_bin]
    if cargo_bin is not None and cargo_bin not in path_entries:
        path_entries.append(cargo_bin)
    path_entries.extend(
        (
            "/usr/local/cuda/bin", "/usr/local/sbin", "/usr/local/bin",
            "/usr/sbin", "/usr/bin", "/sbin", "/bin",
        )
    )
    environment = {
        "HOME": os.environ.get("HOME", "/root"),
        "USER": os.environ.get("USER", "root"),
        "LOGNAME": os.environ.get("LOGNAME", os.environ.get("USER", "root")),
        "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
        "PATH": os.pathsep.join(path_entries),
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
        "CUDA_HOME": "/usr/local/cuda", "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0", "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    if cargo_target_dir is not None:
        target = Path(cargo_target_dir).resolve(strict=True)
        environment["CARGO_TARGET_DIR"] = str(target)
    return environment


def _assert_idle_h20() -> None:
    name = _run(
        ("nvidia-smi", "--id=0", "--query-gpu=name",
         "--format=csv,noheader,nounits")
    ).stdout.strip()
    if name != "NVIDIA H20":
        raise RuntimeError(f"CUDA device 0 is not exactly one H20: {name!r}")
    processes = _run(
        ("nvidia-smi", "--id=0", "--query-compute-apps=pid",
         "--format=csv,noheader,nounits")
    ).stdout.strip()
    if processes:
        raise RuntimeError(f"H20 has active compute processes: {processes}")


def _case_command(
    preflight_record: dict[str, Any], case: Case, mode: str,
    output: Path, *, seed: int,
) -> list[str]:
    return [
        preflight_record["python"]["executable"],
        preflight_record["benchmark"]["path"],
        "--mode", mode,
        "--sglang-root", preflight_record["sglang"]["manager_root"],
        "--model", preflight_record["inputs"]["model"]["root"],
        "--plan", preflight_record["inputs"]["plan"]["path"],
        "--library", preflight_record["library"]["path"],
        "--requests", str(case.batch),
        "--iterations", str(ITERATIONS),
        "--max-total-tokens", str(case.capacity_tokens),
        "--context-length", "128",
        "--seed", str(seed),
        "--attention-backend", "flashinfer",
        "--output", str(output),
    ]


def _expected_adapter_root() -> Path:
    return SOURCE_ROOT.resolve()


def _validate_raw_record(
    record: dict[str, Any], epoch: int, case: Case, mode: str,
    preflight_record: dict[str, Any], label: str,
) -> dict[str, Any]:
    validator = getattr(evidence, "validate_record", evidence._validate_record)
    keywords: dict[str, Any] = {
        "expected_harness_sha256": preflight_record["benchmark"]["sha256"],
        "expected_adapter": abi8.benchmark._adapter_identity(),
        "expected_iterations": ITERATIONS,
    }
    # New verifiers accept an explicit source root; the fallback keeps this
    # runner compatible while that public interface is imported.
    if getattr(evidence, "validate_record", None) is not None:
        keywords["expected_adapter_source_root"] = _expected_adapter_root()
    checked = validator(record, epoch, case.batch, mode, label, **keywords)
    expected_library = preflight_record["library"]
    expected_plan = preflight_record["inputs"]["plan"]
    for observed, expected, artifact in (
        (checked["library"], expected_library, "library"),
        (checked["plan"], expected_plan, "plan"),
    ):
        if any(observed.get(name) != expected.get(name) for name in ("path", "bytes", "sha256")):
            raise RuntimeError(f"{label} {artifact} differs from preflight")
    source = checked["source"]
    if Path(source["root"]).resolve() != Path(
        preflight_record["sglang"]["manager_root"]
    ).resolve():
        raise RuntimeError(f"{label} SGLang root differs from preflight")
    if (
        source.get("plugin_selection")
        != preflight_record["sglang"]["manager_entrypoint"]
        or record.get("checkpoint")
        != preflight_record["inputs"]["model"]["checkpoint"]
        or record.get("runtime_identity", {}).get("python")
        != preflight_record["python"]["executable"]
    ):
        raise RuntimeError(f"{label} execution identity differs from preflight")
    return checked


def _run_record(
    command: Sequence[str], output: Path, stderr_path: Path,
    *, epoch: int, case: Case, mode: str, preflight_record: dict[str, Any],
) -> dict[str, Any]:
    partial_dir = output.parent / ".partial"
    stderr_partial_dir = stderr_path.parent / ".partial"
    partial = partial_dir / output.name
    stderr_partial = stderr_partial_dir / stderr_path.name
    for candidate in (
        output, partial, stderr_path, stderr_partial,
        partial_dir, stderr_partial_dir,
    ):
        if candidate.exists() or candidate.is_symlink():
            raise RuntimeError(f"refusing to overwrite record artifact: {candidate}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    partial_dir.mkdir()
    stderr_partial_dir.mkdir()
    invoked = list(command)
    invoked[invoked.index("--output") + 1] = str(partial)
    completed = _run(
        invoked, env=_fresh_environment(invoked[0]), timeout=3600
    )
    stderr_partial.write_text(completed.stderr, encoding="utf-8")
    record = _strict_json(partial)
    _validate_raw_record(
        record, epoch, case, mode, preflight_record, output.name
    )
    partial.rename(output)
    stderr_partial.rename(stderr_path)
    partial_dir.rmdir()
    stderr_partial_dir.rmdir()
    return record


def _pair_records(
    naive_path: Path, relocate_path: Path, *, epoch: int, case: Case,
    order: Sequence[str], preflight_record: dict[str, Any],
) -> dict[str, Any]:
    naive = _strict_json(naive_path)
    relocate = _strict_json(relocate_path)
    naive_checked = _validate_raw_record(
        naive, epoch, case, "naive", preflight_record, naive_path.name
    )
    relocate_checked = _validate_raw_record(
        relocate, epoch, case, "relocate", preflight_record, relocate_path.name
    )
    equal_fields = (
        "pairing", "request_output_ids", "output_token_digest_sha256",
        "checkpoint", "checkpoint_contract", "sampling_params",
        "workload", "runtime_identity",
    )
    for field in equal_fields:
        if naive.get(field) != relocate.get(field):
            raise RuntimeError(f"epoch {epoch} B{case.batch} pair differs: {field}")
    for field in ("source", "library", "plan", "gpu_uuid"):
        if naive_checked[field] != relocate_checked[field]:
            raise RuntimeError(f"epoch {epoch} B{case.batch} identity differs: {field}")
    left_start, left_end = naive_checked["observed_interval_ns"]
    right_start, right_end = relocate_checked["observed_interval_ns"]
    if max(left_start, right_start) < min(left_end, right_end):
        raise RuntimeError(f"epoch {epoch} B{case.batch} record processes overlap")
    observed_order = (
        ["naive", "relocate"]
        if naive_checked["started_at"] < relocate_checked["started_at"]
        else ["relocate", "naive"]
    )
    if list(order) != observed_order:
        raise RuntimeError(f"epoch {epoch} B{case.batch} execution order differs")
    return {
        "schema": PAIR_SCHEMA, "status": "passed",
        "qualification_claim": QUALIFICATION_CLAIM,
        "preflight_bound": True, "hardware_attested": False,
        "qualified": True, "performance_go": False,
        "epoch": epoch, "batch_size": case.batch,
        "execution_order": list(order),
        "naive_record": naive_path.name,
        "relocate_record": relocate_path.name,
        "naive_record_sha256": sha256_file(naive_path),
        "relocate_record_sha256": sha256_file(relocate_path),
        "pair_key_sha256": naive["pairing"]["pair_key_sha256"],
        "output_token_digest_sha256": naive["output_token_digest_sha256"],
        "exact_token_equality": True,
        "manager_census_fully_drained": True,
        "failure_and_quarantine_counters_zero": True,
    }


def _sealed_summary(summary: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(summary)
    result.update(
        schema=SUMMARY_SCHEMA,
        status="scoped_correctness_and_lifecycle_qualification_passed",
        evidence_class=EVIDENCE_CLASS, diagnostic_only=False,
        qualification_status=QUALIFICATION_STATUS,
        qualification_claim=QUALIFICATION_CLAIM,
        sealed=True, source_clean=True, source_dirty=False,
        preflight_bound=True, hardware_attested=False, qualified=True,
        performance_go=False,
    )
    return result


def _verify_work_matrix(
    work_dir: Path, preflight_record: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for epoch in range(1, EPOCH_COUNT + 1):
        order = execution_order(epoch)
        record_dir = work_dir / "records" / f"epoch-{epoch:03d}"
        pair_dir = work_dir / "pairs" / f"epoch-{epoch:03d}"
        log_dir = work_dir / "logs" / f"epoch-{epoch:03d}"
        for case in CASES:
            paths = {
                mode: record_dir / f"{case.slug}-{mode}.json" for mode in MODES
            }
            pair = _pair_records(
                paths["naive"], paths["relocate"], epoch=epoch,
                case=case, order=order, preflight_record=preflight_record,
            )
            pair_path = pair_dir / f"{case.slug}-pair.json"
            if _strict_json(pair_path) != pair:
                raise RuntimeError(f"stored pair differs from derivation: {pair_path}")
            for mode in MODES:
                log = log_dir / f"{case.slug}-{mode}.stderr.log"
                if not log.is_file() or log.is_symlink():
                    raise RuntimeError(f"missing regular stderr log: {log}")
            pairs.append(pair)
    verifier = getattr(evidence, "verify_evidence")
    keywords: dict[str, Any] = {
        "expected_harness_sha256": preflight_record["benchmark"]["sha256"],
        "expected_adapter": abi8.benchmark._adapter_identity(),
    }
    # The sealed-aware verifier accepts the root explicitly.
    try:
        diagnostic_summary = verifier(
            work_dir, expected_adapter_source_root=_expected_adapter_root(),
            **keywords,
        )
    except TypeError as error:
        if "expected_" not in str(error):
            raise
        diagnostic_summary = verifier(work_dir)
    return pairs, _sealed_summary(diagnostic_summary)


def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    if args.execute != EXECUTION_TOKEN:
        raise RuntimeError(f"run requires --execute {EXECUTION_TOKEN}")
    if args.epochs != EPOCH_COUNT:
        raise RuntimeError(f"sealed qualification requires exactly {EPOCH_COUNT} epochs")
    if args.phase != "all":
        raise RuntimeError(
            "sealed qualification must run B1 and B4 together to preserve "
            "the four chronological alternating epochs"
        )
    work_dir = args.work_dir.expanduser().resolve(strict=True)
    preflight_record = _strict_json(work_dir / "preflight.json")
    _validate_preflight(preflight_record)
    selected = list(CASES)
    written_pairs: list[dict[str, Any]] = []
    _assert_idle_h20()
    for epoch in range(1, EPOCH_COUNT + 1):
        order = execution_order(epoch)
        for case in selected:
            record_dir = work_dir / "records" / f"epoch-{epoch:03d}"
            log_dir = work_dir / "logs" / f"epoch-{epoch:03d}"
            paths = {
                mode: record_dir / f"{case.slug}-{mode}.json" for mode in MODES
            }
            for mode in order:
                _validate_preflight(preflight_record)
                command = _case_command(
                    preflight_record, case, mode, paths[mode], seed=args.seed
                )
                _run_record(
                    command, paths[mode],
                    log_dir / f"{case.slug}-{mode}.stderr.log",
                    epoch=epoch, case=case, mode=mode,
                    preflight_record=preflight_record,
                )
                _validate_preflight(preflight_record)
                _assert_idle_h20()
            pair = _pair_records(
                paths["naive"], paths["relocate"], epoch=epoch,
                case=case, order=order, preflight_record=preflight_record,
            )
            _write_new(
                work_dir / "pairs" / f"epoch-{epoch:03d}"
                / f"{case.slug}-pair.json", pair,
            )
            written_pairs.append(pair)
    complete = all(
        (
            work_dir / "pairs" / f"epoch-{epoch:03d}"
            / f"{case.slug}-pair.json"
        ).is_file()
        for epoch in range(1, EPOCH_COUNT + 1) for case in CASES
    )
    if complete:
        _, summary = _verify_work_matrix(work_dir, preflight_record)
        _write_new(work_dir / "summary.json", summary)
        return summary
    return {
        "schema": PAIR_SCHEMA, "status": "phase_completed",
        "phase": args.phase, "pair_count": len(written_pairs),
        "qualified": False, "hardware_attested": False,
        "performance_go": False,
    }


def _component_properties(path: Path) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"component JUnit is not a regular file: {path}")
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as error:
        raise RuntimeError(f"invalid component JUnit XML: {path}") from error
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    if len(suites) != 1:
        raise RuntimeError("component JUnit must contain exactly one test suite")
    suite = suites[0]
    counts = {
        "tests": str(len(COMPONENT_CASES)), "errors": "0",
        "failures": "0", "skipped": "0",
    }
    if any(suite.get(name) != value for name, value in counts.items()):
        raise RuntimeError("component JUnit result counts differ")
    cases = [node.get("name") for node in suite.findall("testcase")]
    if len(cases) != len(set(cases)) or set(cases) != set(COMPONENT_CASES):
        raise RuntimeError("component JUnit cases differ from the qualification set")
    nodes = suite.findall("./properties/property")
    properties = {node.get("name"): node.get("value") for node in nodes}
    if len(properties) != len(nodes):
        raise RuntimeError("component JUnit has duplicate properties")
    required = {
        "orbitkv.cuda.available": "true",
        "orbitkv.cuda.device_name": "NVIDIA H20",
    }
    if any(properties.get(name) != value for name, value in required.items()):
        raise RuntimeError("component JUnit was not recorded on one H20")
    for name in (
        "orbitkv.cuda.device_uuid", "orbitkv.cuda.runtime_version",
        "orbitkv.torch.version",
    ):
        if not properties.get(name):
            raise RuntimeError(f"component JUnit property is missing: {name}")
    return properties  # type: ignore[return-value]


def run_component(args: argparse.Namespace) -> dict[str, Any]:
    if args.execute != EXECUTION_TOKEN:
        raise RuntimeError(f"component requires --execute {EXECUTION_TOKEN}")
    work_dir = args.work_dir.expanduser().resolve(strict=True)
    preflight_record = _strict_json(work_dir / "preflight.json")
    _validate_preflight(preflight_record)
    output = work_dir / "component-conformance.xml"
    partial = output.with_suffix(".xml.partial")
    if output.exists() or partial.exists():
        raise RuntimeError("refusing to overwrite component JUnit evidence")
    _assert_idle_h20()
    build = preflight_record["build"]
    environment = _fresh_environment(
        preflight_record["python"]["executable"],
        cargo=build["command"][0],
        cargo_target_dir=build["cargo_target_dir"],
    )
    environment["ORBITKV_CUDA_RELOCATION_CYCLES"] = "2"
    command = (
        preflight_record["python"]["executable"], "-m", "pytest",
        "tests/test_cuda_relocation_conformance.py", "-q",
        f"--junitxml={partial}",
    )
    _run(command, cwd=INTEGRATION_ROOT, env=environment, timeout=3600)
    properties = _component_properties(partial)
    component_library = Path(preflight_record["library"]["path"])
    observed_library = {
        "path": str(component_library.resolve(strict=True)),
        "sha256": sha256_file(component_library),
        "bytes": component_library.stat().st_size,
        "symbols": abi8._symbols(component_library),
    }
    library = ctypes.CDLL(str(component_library.resolve(strict=True)))
    library.orbitkv_abi_version.restype = ctypes.c_uint32
    observed_library["abi_version"] = int(library.orbitkv_abi_version())
    expected_library = preflight_record["library"]
    for name in ("sha256", "bytes", "abi_version", "symbols"):
        if observed_library.get(name) != expected_library.get(name):
            raise RuntimeError(
                "component suite did not exercise the exact preflight ABI8 library"
            )
    _validate_preflight(preflight_record)
    partial.rename(output)
    _assert_idle_h20()
    return {
        "status": "passed", "test_count": len(COMPONENT_CASES),
        "device": {
            "name": properties["orbitkv.cuda.device_name"],
            "uuid": properties["orbitkv.cuda.device_uuid"],
        },
        "hardware_attested": False, "performance_go": False,
    }


def _copy_source_closure(
    destination: Path, closure: Iterable[dict[str, str]]
) -> None:
    for item in closure:
        relative = PurePosixPath(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("preflight source closure has an unsafe path")
        source = REPOSITORY_ROOT / Path(relative)
        if (
            not source.is_file() or source.is_symlink()
            or sha256_file(source) != item["sha256"]
        ):
            raise RuntimeError(f"source closure differs from preflight: {relative}")
        target = destination / Path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _verify_source_commit_closure(
    source: dict[str, Any], closure: Iterable[dict[str, str]]
) -> None:
    commit = source.get("commit")
    if not isinstance(commit, str):
        raise RuntimeError("preflight source commit is missing")
    for item in closure:
        relative = item["path"]
        try:
            blob = subprocess.run(
                ("git", "show", f"{commit}:{relative}"),
                cwd=REPOSITORY_ROOT, check=True, capture_output=True, timeout=30,
            ).stdout
        except (OSError, subprocess.SubprocessError) as error:
            raise RuntimeError(
                f"cannot resolve qualification source at commit: {relative}"
            ) from error
        if hashlib.sha256(blob).hexdigest() != item["sha256"]:
            raise RuntimeError(
                f"qualification source is not committed at preflight HEAD: {relative}"
            )


def _artifact_hashes(root: Path, excluded: frozenset[str]) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
        and path.relative_to(root).as_posix() not in excluded
    }


def _create_source_bundle(path: Path, commit: str) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError("refusing to overwrite source bundle")
    current = _run(("git", "rev-parse", "HEAD")).stdout.strip()
    if current != commit:
        raise RuntimeError("source commit changed since preflight")
    _run(("git", "bundle", "create", str(path), "HEAD"))
    heads = _run(("git", "bundle", "list-heads", str(path))).stdout.splitlines()
    if heads != [f"{commit} HEAD"]:
        raise RuntimeError("source bundle does not contain exactly preflight HEAD")


def _model_provenance(model: dict[str, Any]) -> dict[str, Any]:
    checkpoint = model["checkpoint"]
    return {
        "schema": MODEL_PROVENANCE_SCHEMA, "name": MODEL_NAME,
        "directory_name": MODEL_DIRECTORY_NAME,
        "config_sha256": checkpoint["config_sha256"],
        "weight_files": checkpoint["weight_files"],
        "checkpoint_identity_sha256": model["identity_sha256"],
    }


def _scope() -> dict[str, Any]:
    return {
        "cases": [
            {
                "model": MODEL_NAME, "profile": "full",
                "attention_backend": "flashinfer",
                "batch_size": case.batch, "modes": list(MODES),
            }
            for case in CASES
        ],
        "execution": {
            "page_tokens": 16, "kv_dtype": "bf16",
            "kv_layout": "nhd", "execution": "eager",
            "gpu_count": 1, "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1, "data_parallel_size": 1,
            "decode_context_parallel_size": 1,
        },
        "excluded": [
            "performance_qualification", "capacity_or_memory_saving_claims",
            "general_speedup_claims", "prefix_or_radix_sharing",
            "hybrid_or_swa_attention", "mla", "cuda_graphs",
            "overlap_scheduling", "speculation",
            "distributed_execution", "multi_gpu", "production_readiness",
        ],
    }


def seal(args: argparse.Namespace) -> dict[str, Any]:
    work_dir = args.work_dir.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise RuntimeError(f"refusing to overwrite existing seal directory: {output_dir}")
    preflight_record = _strict_json(work_dir / "preflight.json")
    _validate_preflight(preflight_record)
    if output_dir.is_relative_to(REPOSITORY_ROOT):
        raise RuntimeError(
            "seal output must be outside the clean qualification repository"
        )
    _verify_source_commit_closure(
        preflight_record["source"], preflight_record["source_closure"]
    )
    pairs, summary = _verify_work_matrix(work_dir, preflight_record)
    if len(pairs) != EPOCH_COUNT * len(CASES):
        raise RuntimeError("seal requires exactly eight verified pairs")
    if _strict_json(work_dir / "summary.json") != summary:
        raise RuntimeError("stored summary differs from trusted recomputation")
    component = work_dir / "component-conformance.xml"
    component_properties = _component_properties(component)
    summary_hardware = summary.get("hardware", {})
    if (
        component_properties.get("orbitkv.cuda.device_name")
        != summary_hardware.get("observed_name")
        or component_properties.get("orbitkv.cuda.device_uuid")
        != summary_hardware.get("observed_uuid")
    ):
        raise RuntimeError("component and model evidence GPU identities differ")

    output_dir.mkdir(parents=True)
    shutil.copy2(work_dir / "preflight.json", output_dir / "preflight.json")
    shutil.copy2(work_dir / "summary.json", output_dir / "summary.json")
    shutil.copy2(component, output_dir / "component-conformance.xml")
    for directory in ("records", "pairs", "logs"):
        shutil.copytree(work_dir / directory, output_dir / directory)

    qualification = output_dir / "qualification"
    (qualification / "build").mkdir(parents=True)
    (qualification / "plans").mkdir()
    (qualification / "model").mkdir()
    shutil.copy2(
        preflight_record["library"]["path"],
        qualification / "build/liborbitkv_ffi.so",
    )
    shutil.copy2(
        preflight_record["inputs"]["plan"]["path"],
        qualification / "plans" / PLAN_NAME,
    )
    model = preflight_record["inputs"]["model"]
    shutil.copy2(
        Path(model["root"]) / "config.json",
        qualification / "model/config.json",
    )
    _write_new(qualification / "model-provenance.json", _model_provenance(model))
    original_lock = qualification / "requirements.input.lock.txt"
    shutil.copy2(preflight_record["inputs"]["requirements"]["path"], original_lock)
    materialized_lock = qualification / "requirements.lock.txt"
    materialized_lock.write_text(
        abi8._materialize_requirements_lock(
            original_lock, preflight_record["python"]["active_editable"]
        ),
        encoding="utf-8",
    )
    _copy_source_closure(qualification / "source", preflight_record["source_closure"])
    bundle = qualification / "source.bundle"
    _create_source_bundle(bundle, preflight_record["source"]["commit"])
    _validate_preflight(preflight_record)

    readme = (
        "# OrbitKV ABI8 H20 token-relocation qualification\n\n"
        "Status: scoped Full-attention token-relocation correctness and lifecycle "
        "qualified; performance pending. `hardware_attested=false` and "
        "`performance_go=false`. No capacity, memory-saving, or general speedup "
        "claim is made.\n\n"
        "Verify first from a trusted OrbitKV checkout with:\n\n"
        "```sh\nPYTHONDONTWRITEBYTECODE=1 python3 tools/verify_manifests.py "
        "/path/to/seal/manifest.json\n```\n\n"
        "After that succeeds, the bundled portable consistency check is:\n\n"
        "```sh\nPYTHONDONTWRITEBYTECODE=1 python3 -S "
        "qualification/source/integrations/sglang/"
        "qualify_token_relocation_h20.py verify-seal .\n```\n"
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    artifacts = _artifact_hashes(
        output_dir, frozenset({"manifest.json", "SHA256SUMS"})
    )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "qualification_status": QUALIFICATION_STATUS,
        "qualification_claim": QUALIFICATION_CLAIM,
        "evidence_class": EVIDENCE_CLASS,
        "sealed": True, "source_clean": True, "source_dirty": False,
        "preflight_bound": True, "qualified": True,
        "hardware_attested": False, "performance_go": False,
        "abi_version": 8, "exact_symbol_count": 40,
        "record_schema": benchmark.RECORD_SCHEMA,
        "summary_schema": SUMMARY_SCHEMA, "pair_schema": PAIR_SCHEMA,
        "epoch_count": EPOCH_COUNT, "batch_sizes": [1, 4],
        "record_count": EPOCH_COUNT * len(CASES) * len(MODES),
        "pair_count": len(pairs), "scope": _scope(),
        "source_commit": preflight_record["source"]["commit"],
        "source_inventory_sha256": preflight_record["source"]["inventory_sha256"],
        "source_provenance": {
            "kind": "git_bundle",
            "path": "qualification/source.bundle",
            "sha256": sha256_file(bundle),
            "commit": preflight_record["source"]["commit"],
            "reference": "HEAD",
        },
        "sglang_release": "v0.5.17",
        "sglang_revision": SGLANG_REVISION,
        "library_sha256": preflight_record["library"]["sha256"],
        "model_identity_sha256": model["identity_sha256"],
        "input_hashes": {
            "requirements_input_sha256": REQUIREMENTS_SHA256,
            "requirements_sha256": sha256_file(materialized_lock),
            "plan_sha256": PLAN_SHA256,
            "model": {
                "config_sha256": MODEL_CONFIG_SHA256,
                "weight_sha256": MODEL_WEIGHT_SHA256,
                "weight_bytes": MODEL_WEIGHT_BYTES,
            },
        },
        "observed_hardware": {
            "name": summary_hardware["observed_name"],
            "uuid": summary_hardware["observed_uuid"],
            "snapshot_count": summary_hardware["snapshot_count"],
            "attestation": summary_hardware["attestation"],
        },
        "artifacts": artifacts,
    }
    _write_new(output_dir / "manifest.json", manifest)
    sums = dict(artifacts)
    sums["manifest.json"] = sha256_file(output_dir / "manifest.json")
    (output_dir / "SHA256SUMS").write_text(
        "".join(
            f"{digest}  {name}\n" for name, digest in sorted(sums.items())
        ),
        encoding="utf-8",
    )
    verify_seal(output_dir)
    return manifest


def verify_seal(path: Path) -> dict[str, Any]:
    try:
        import verify_token_relocation_h20_seal as seal_verifier
    except ImportError as error:
        raise RuntimeError(
            "trusted token-relocation seal verifier is unavailable"
        ) from error
    result = seal_verifier.verify_sealed_archive(path)
    expected = {
        "schema": SEAL_VERIFICATION_SCHEMA, "status": "passed",
        "qualification_status": QUALIFICATION_STATUS,
        "qualification_claim": QUALIFICATION_CLAIM,
        "sealed": True, "source_clean": True,
        "preflight_bound": True, "qualified": True,
        "hardware_attested": False, "performance_go": False,
        "epoch_count": 4, "record_count": 16, "pair_count": 8,
        "abi_version": 8, "exact_symbol_count": 40,
        "all_pairs_passed": True, "exact_token_equality": True,
        "manager_census_fully_drained": True,
        "failure_and_quarantine_counters_zero": True,
    }
    for name, value in expected.items():
        if result.get(name) != value or type(result.get(name)) is not type(value):
            raise RuntimeError(f"trusted seal verification differs at {name}")
    return result


def _common_work_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--work-dir", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action")
    pre = subparsers.add_parser("preflight")
    _common_work_dir(pre)
    pre.add_argument(
        "--python", type=Path,
        default=REPOSITORY_ROOT / ".venv-sglang-v0517/bin/python",
    )
    pre.add_argument("--cargo", default="cargo")
    pre.add_argument(
        "--manager-root", type=Path,
        default=REPOSITORY_ROOT / ".qualification/sglang-v0517-manager",
    )
    pre.add_argument(
        "--requirements", type=Path,
        default=REPOSITORY_ROOT / ".qualification/requirements-v0.5.17.lock.txt",
    )
    pre.add_argument(
        "--model", type=Path,
        default=Path("/workspace/models/qwen2.5-0.5b-instruct"),
    )
    pre.add_argument(
        "--plan", type=Path,
        default=(
            REPOSITORY_ROOT
            / ".qualification/plans/qwen2.5-0.5b-full-page16-bf16.json"
        ),
    )
    run = subparsers.add_parser("run")
    _common_work_dir(run)
    run.add_argument("--execute", required=True)
    run.add_argument("--phase", choices=("b1", "b4", "all"), default="all")
    run.add_argument("--epochs", type=int, default=EPOCH_COUNT)
    run.add_argument("--seed", type=int, default=20260821)
    component = subparsers.add_parser("component")
    _common_work_dir(component)
    component.add_argument("--execute", required=True)
    seal_parser = subparsers.add_parser("seal")
    _common_work_dir(seal_parser)
    seal_parser.add_argument("--output-dir", type=Path, required=True)
    verify = subparsers.add_parser("verify-seal")
    verify.add_argument("seal_dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw or raw[0] not in {
        "preflight", "run", "component", "seal", "verify-seal"
    }:
        raw.insert(0, "preflight")
    args = parser.parse_args(raw)
    try:
        if args.action == "preflight":
            result = preflight(args)
        elif args.action == "run":
            result = run_matrix(args)
        elif args.action == "component":
            result = run_component(args)
        elif args.action == "seal":
            result = seal(args)
        elif args.action == "verify-seal":
            result = verify_seal(args.seal_dir)
        else:
            raise RuntimeError(f"unknown action: {args.action}")
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
