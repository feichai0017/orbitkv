#!/usr/bin/env python3
"""Validate and summarize dense-linear shape census records."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from math import isfinite, prod
from pathlib import Path
from typing import Any


SCHEMA = "oxide.gemm-shape-census.v1"
CAPTURE_BOUNDARY = "resolved_dense_linear_before_backend_selection"
COUNT_SEMANTICS = "successful_linear_dispatches"
PHASES = {"prefill", "decode"}
OBSERVED_PATHS = {
    "mistral_cuda_gemv",
    "candle_cuda_flattened_matmul",
    "candle_matmul",
}
DTYPES = {"bf16", "f16", "f32"}
LAYOUTS = {
    "row_major_contiguous",
    "last_dimension_contiguous",
    "strided",
}
WEIGHT_LAYOUTS = {
    "row_major_contiguous_nk",
    "last_dimension_contiguous_nk",
    "strided_nk",
}
POST_OPS = {"bias"}
MAX_USIZE_64 = (1 << 64) - 1
HEX_40 = re.compile(r"^[0-9a-f]{40}$")
HEX_64 = re.compile(r"^[0-9a-f]{64}$")
ACCEPTED_CLAIMS = [
    "exact dense-linear host dispatch counts for the pinned model and workload"
]
EXCLUDED_CLAIMS = [
    "latency",
    "throughput",
    "CUDA kernel launch counts",
    "general model coverage",
    "production workload frequency",
]

SOURCE_FIELDS = (
    "producer",
    "repository",
    "commit",
    "worktree_clean",
    "cargo_lock_sha256",
    "binary_sha256",
    "oxide_schema_commit",
)
HARDWARE_FIELDS = ("gpu", "compute_capability")
ENVIRONMENT_FIELDS = (
    "host_os",
    "host_arch",
    "rustc_version",
    "cuda_toolkit_version",
    "driver_version",
    "cuda_arch",
)
MODEL_FIELDS = (
    "name",
    "weights_sha256",
    "config_sha256",
    "tokenizer_sha256",
    "tensor_parallel_size",
)
WORKLOAD_FIELDS = (
    "scheduler",
    "request_count",
    "prompt_tokens",
    "completion_tokens",
    "prefill_forward_steps",
    "decode_forward_steps",
    "canonical_request",
    "canonical_request_sha256",
    "cuda_graph_enabled",
)
CAPTURE_FIELDS = (
    "boundary",
    "count_semantics",
    "sample_rate",
    "complete",
    "timed",
    "mistral_gemv_enabled",
)
ENTRY_FIELDS = (
    "phase",
    "site",
    "observed_path",
    "device_kind",
    "device_ordinal",
    "m",
    "n",
    "k",
    "a_shape",
    "a_stride",
    "a_offset_elements",
    "weight_shape",
    "weight_stride",
    "weight_offset_elements",
    "a_dtype",
    "weight_dtype",
    "a_layout",
    "weight_layout",
    "transpose_a",
    "transpose_weight",
    "post_ops",
    "host_calls",
)


def _require_object(value: Any, source: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{source}: must be a JSON object")
    return value


def _require_fields(
    value: dict[str, Any], fields: tuple[str, ...], source: str
) -> None:
    missing = [field for field in fields if field not in value]
    if missing:
        raise ValueError(f"{source}: missing required fields: {', '.join(missing)}")


def _require_exact_fields(
    value: dict[str, Any], fields: tuple[str, ...], source: str
) -> None:
    _require_fields(value, fields, source)
    unexpected = sorted(set(value) - set(fields))
    if unexpected:
        raise ValueError(f"{source}: unexpected fields: {', '.join(unexpected)}")


def _require_string(value: Any, source: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source}: must be a nonempty string")
    return value


def _require_bool(value: Any, source: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{source}: must be a boolean")
    return value


def _require_int(
    value: Any, source: str, *, minimum: int, maximum: int = MAX_USIZE_64
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > maximum
    ):
        raise ValueError(
            f"{source}: must be an integer from {minimum} through {maximum}"
        )
    return value


def _require_number(value: Any, source: str, *, minimum: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or value < minimum
    ):
        raise ValueError(f"{source}: must be a finite number at least {minimum}")
    return float(value)


def _require_hex(value: Any, pattern: re.Pattern[str], source: str) -> str:
    text = _require_string(value, source)
    if pattern.fullmatch(text) is None:
        raise ValueError(f"{source}: has an invalid hexadecimal digest")
    return text


def _require_enum(value: Any, admitted: set[str], source: str) -> str:
    text = _require_string(value, source)
    if text not in admitted:
        raise ValueError(f"{source}: unsupported value {text!r}")
    return text


def _require_int_list(
    value: Any, source: str, *, minimum: int, nonempty: bool = True
) -> list[int]:
    if not isinstance(value, list) or (nonempty and not value):
        raise ValueError(f"{source}: must be a nonempty integer list")
    return [
        _require_int(item, f"{source}[{index}]", minimum=minimum)
        for index, item in enumerate(value)
    ]


def _is_contiguous(shape: list[int], stride: list[int]) -> bool:
    expected = 1
    for dimension, actual_stride in zip(reversed(shape), reversed(stride)):
        if dimension != 1 and actual_stride != expected:
            return False
        expected *= dimension
    return True


def _required_span(
    shape: list[int], stride: list[int], offset: int, source: str
) -> int:
    last_index = 0
    for dimension, actual_stride in zip(shape, stride):
        extent = dimension - 1
        if extent != 0 and actual_stride > MAX_USIZE_64 // extent:
            raise ValueError(f"{source}: tensor span overflows 64-bit usize")
        term = extent * actual_stride
        if last_index > MAX_USIZE_64 - term:
            raise ValueError(f"{source}: tensor span overflows 64-bit usize")
        last_index += term
    if last_index == MAX_USIZE_64:
        raise ValueError(f"{source}: tensor span overflows 64-bit usize")
    if offset > MAX_USIZE_64 - 1 - last_index:
        raise ValueError(f"{source}: tensor offset and span overflow 64-bit usize")
    return last_index + 1


def _expected_layout(shape: list[int], stride: list[int], *, weight: bool) -> str:
    if _is_contiguous(shape, stride):
        return "row_major_contiguous_nk" if weight else "row_major_contiguous"
    if stride[-1] == 1:
        return (
            "last_dimension_contiguous_nk"
            if weight
            else "last_dimension_contiguous"
        )
    return "strided_nk" if weight else "strided"


def _validate_source(record: dict[str, Any], source: str) -> None:
    _require_exact_fields(record, SOURCE_FIELDS, source)
    _require_string(record["producer"], f"{source}.producer")
    _require_string(record["repository"], f"{source}.repository")
    _require_hex(record["commit"], HEX_40, f"{source}.commit")
    if not _require_bool(record["worktree_clean"], f"{source}.worktree_clean"):
        raise ValueError(f"{source}.worktree_clean: dirty source is not admitted")
    _require_hex(
        record["cargo_lock_sha256"], HEX_64, f"{source}.cargo_lock_sha256"
    )
    _require_hex(record["binary_sha256"], HEX_64, f"{source}.binary_sha256")
    _require_hex(record["oxide_schema_commit"], HEX_40, f"{source}.oxide_schema_commit")


def _validate_hardware(record: dict[str, Any], source: str) -> None:
    _require_exact_fields(record, HARDWARE_FIELDS, source)
    _require_string(record["gpu"], f"{source}.gpu")
    capability = _require_string(
        record["compute_capability"], f"{source}.compute_capability"
    )
    if re.fullmatch(r"[0-9]+\.[0-9]+", capability) is None:
        raise ValueError(f"{source}.compute_capability: expected major.minor")


def _validate_environment(record: dict[str, Any], source: str) -> None:
    _require_exact_fields(record, ENVIRONMENT_FIELDS, source)
    for field in (
        "host_os",
        "host_arch",
        "rustc_version",
        "cuda_toolkit_version",
        "driver_version",
        "cuda_arch",
    ):
        _require_string(record[field], f"{source}.{field}")


def _validate_model(record: dict[str, Any], source: str) -> None:
    _require_exact_fields(record, MODEL_FIELDS, source)
    _require_string(record["name"], f"{source}.name")
    for field in ("weights_sha256", "config_sha256", "tokenizer_sha256"):
        _require_hex(record[field], HEX_64, f"{source}.{field}")
    if _require_int(
        record["tensor_parallel_size"], f"{source}.tensor_parallel_size", minimum=1
    ) != 1:
        raise ValueError(
            f"{source}.tensor_parallel_size: version 1 requires tensor parallel size one"
        )


def _validate_workload(record: dict[str, Any], source: str) -> None:
    _require_exact_fields(record, WORKLOAD_FIELDS, source)
    if record["scheduler"] != "single_request":
        raise ValueError(f"{source}.scheduler: version 1 requires single_request")
    if _require_int(
        record["request_count"], f"{source}.request_count", minimum=1
    ) != 1:
        raise ValueError(f"{source}.request_count: version 1 requires one request")
    _require_int(record["prompt_tokens"], f"{source}.prompt_tokens", minimum=1)
    completion_tokens = _require_int(
        record["completion_tokens"], f"{source}.completion_tokens", minimum=1
    )
    if _require_int(
        record["prefill_forward_steps"],
        f"{source}.prefill_forward_steps",
        minimum=1,
    ) != 1:
        raise ValueError(
            f"{source}.prefill_forward_steps: version 1 requires one prefill forward"
        )
    decode_forward_steps = _require_int(
        record["decode_forward_steps"],
        f"{source}.decode_forward_steps",
        minimum=0,
    )
    if decode_forward_steps != completion_tokens - 1:
        raise ValueError(
            f"{source}.decode_forward_steps: expected completion_tokens minus one"
        )
    request = _require_object(
        record["canonical_request"], f"{source}.canonical_request"
    )
    _require_exact_fields(
        request, ("messages", "sampler"), f"{source}.canonical_request"
    )
    messages = request["messages"]
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{source}.canonical_request.messages: must be nonempty")
    for index, raw_message in enumerate(messages):
        message = _require_object(
            raw_message, f"{source}.canonical_request.messages[{index}]"
        )
        _require_exact_fields(
            message,
            ("role", "content"),
            f"{source}.canonical_request.messages[{index}]",
        )
        _require_enum(
            message["role"],
            {"system", "user", "assistant"},
            f"{source}.canonical_request.messages[{index}].role",
        )
        _require_string(
            message["content"],
            f"{source}.canonical_request.messages[{index}].content",
        )
    sampler = _require_object(
        request["sampler"], f"{source}.canonical_request.sampler"
    )
    _require_exact_fields(
        sampler,
        ("temperature", "max_output_tokens", "top_logprobs", "return_logprobs"),
        f"{source}.canonical_request.sampler",
    )
    _require_number(
        sampler["temperature"],
        f"{source}.canonical_request.sampler.temperature",
        minimum=0.0,
    )
    max_output_tokens = _require_int(
        sampler["max_output_tokens"],
        f"{source}.canonical_request.sampler.max_output_tokens",
        minimum=1,
    )
    if completion_tokens > max_output_tokens:
        raise ValueError(
            f"{source}.completion_tokens: exceeds the request output limit"
        )
    _require_int(
        sampler["top_logprobs"],
        f"{source}.canonical_request.sampler.top_logprobs",
        minimum=0,
    )
    _require_bool(
        sampler["return_logprobs"],
        f"{source}.canonical_request.sampler.return_logprobs",
    )
    canonical = json.dumps(
        request, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    request_digest = _require_hex(
        record["canonical_request_sha256"],
        HEX_64,
        f"{source}.canonical_request_sha256",
    )
    if hashlib.sha256(canonical).hexdigest() != request_digest:
        raise ValueError(
            f"{source}.canonical_request_sha256: does not match canonical request JSON"
        )
    if _require_bool(
        record["cuda_graph_enabled"], f"{source}.cuda_graph_enabled"
    ):
        raise ValueError(
            f"{source}.cuda_graph_enabled: Graph replay is not admitted by this schema"
        )


def _validate_capture(record: dict[str, Any], source: str) -> None:
    _require_exact_fields(record, CAPTURE_FIELDS, source)
    if record["boundary"] != CAPTURE_BOUNDARY:
        raise ValueError(f"{source}.boundary: unsupported capture boundary")
    if record["count_semantics"] != COUNT_SEMANTICS:
        raise ValueError(f"{source}.count_semantics: unsupported count semantics")
    if _require_int(record["sample_rate"], f"{source}.sample_rate", minimum=1) != 1:
        raise ValueError(f"{source}.sample_rate: sampled census is not admitted")
    if not _require_bool(record["complete"], f"{source}.complete"):
        raise ValueError(f"{source}.complete: incomplete census is not admitted")
    if _require_bool(record["timed"], f"{source}.timed"):
        raise ValueError(f"{source}.timed: timing records use a different schema")
    if not _require_bool(
        record["mistral_gemv_enabled"], f"{source}.mistral_gemv_enabled"
    ):
        raise ValueError(
            f"{source}.mistral_gemv_enabled: version 1 requires the engine baseline"
        )


def _entry_identity(entry: dict[str, Any]) -> str:
    identity = {key: value for key, value in entry.items() if key != "host_calls"}
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


def _validate_entry(entry: dict[str, Any], source: str) -> None:
    _require_exact_fields(entry, ENTRY_FIELDS, source)
    _require_enum(entry["phase"], PHASES, f"{source}.phase")
    _require_string(entry["site"], f"{source}.site")
    observed_path = _require_enum(
        entry["observed_path"], OBSERVED_PATHS, f"{source}.observed_path"
    )
    if entry["device_kind"] != "cuda":
        raise ValueError(f"{source}.device_kind: version 1 requires CUDA")
    if _require_int(
        entry["device_ordinal"], f"{source}.device_ordinal", minimum=0
    ) != 0:
        raise ValueError(f"{source}.device_ordinal: version 1 requires GPU zero")
    m = _require_int(entry["m"], f"{source}.m", minimum=1)
    n = _require_int(entry["n"], f"{source}.n", minimum=1)
    k = _require_int(entry["k"], f"{source}.k", minimum=1)
    a_shape = _require_int_list(entry["a_shape"], f"{source}.a_shape", minimum=1)
    a_stride = _require_int_list(
        entry["a_stride"], f"{source}.a_stride", minimum=0
    )
    a_offset = _require_int(
        entry["a_offset_elements"], f"{source}.a_offset_elements", minimum=0
    )
    weight_shape = _require_int_list(
        entry["weight_shape"], f"{source}.weight_shape", minimum=1
    )
    weight_stride = _require_int_list(
        entry["weight_stride"], f"{source}.weight_stride", minimum=0
    )
    weight_offset = _require_int(
        entry["weight_offset_elements"],
        f"{source}.weight_offset_elements",
        minimum=0,
    )
    if len(a_shape) != len(a_stride):
        raise ValueError(f"{source}: activation shape and stride ranks differ")
    if len(weight_shape) != 2 or len(weight_stride) != 2:
        raise ValueError(f"{source}: weight shape and stride must have rank two")
    _required_span(a_shape, a_stride, a_offset, f"{source}.activation")
    _required_span(
        weight_shape, weight_stride, weight_offset, f"{source}.weight"
    )
    if a_shape[-1] != k or prod(a_shape[:-1]) != m:
        raise ValueError(f"{source}: activation shape does not match M and K")
    if weight_shape != [n, k]:
        raise ValueError(f"{source}: weight shape does not match N and K")
    for name, left, right in (("A", m, k), ("weight", n, k), ("output", m, n)):
        if left > MAX_USIZE_64 // right:
            raise ValueError(f"{source}: {name} element count overflows 64-bit usize")
    a_layout = _require_enum(entry["a_layout"], LAYOUTS, f"{source}.a_layout")
    weight_layout = _require_enum(
        entry["weight_layout"], WEIGHT_LAYOUTS, f"{source}.weight_layout"
    )
    if a_layout != _expected_layout(a_shape, a_stride, weight=False):
        raise ValueError(f"{source}.a_layout: category does not match shape and stride")
    if weight_layout != _expected_layout(weight_shape, weight_stride, weight=True):
        raise ValueError(
            f"{source}.weight_layout: category does not match shape and stride"
        )
    a_dtype = _require_enum(entry["a_dtype"], DTYPES, f"{source}.a_dtype")
    weight_dtype = _require_enum(
        entry["weight_dtype"], DTYPES, f"{source}.weight_dtype"
    )
    if a_dtype != weight_dtype:
        raise ValueError(f"{source}: activation and weight dtypes must match")
    if _require_bool(entry["transpose_a"], f"{source}.transpose_a"):
        raise ValueError(f"{source}.transpose_a: dense-linear A must not be transposed")
    if not _require_bool(entry["transpose_weight"], f"{source}.transpose_weight"):
        raise ValueError(f"{source}.transpose_weight: dense-linear weight must be transposed")
    post_ops = entry["post_ops"]
    if not isinstance(post_ops, list):
        raise ValueError(f"{source}.post_ops: must be a string list")
    for index, item in enumerate(post_ops):
        _require_enum(item, POST_OPS, f"{source}.post_ops[{index}]")
    if len(set(post_ops)) != len(post_ops):
        raise ValueError(f"{source}.post_ops: duplicate operation")
    _require_int(entry["host_calls"], f"{source}.host_calls", minimum=1)
    gemv_admitted = m <= 8 and k % 2 == 0
    if observed_path == "mistral_cuda_gemv" and not gemv_admitted:
        raise ValueError(f"{source}.observed_path: shape does not admit Mistral GEMV")
    if observed_path == "candle_cuda_flattened_matmul" and (
        gemv_admitted or len(a_shape) <= 2
    ):
        raise ValueError(
            f"{source}.observed_path: shape does not admit flattened CUDA matmul"
        )
    if observed_path == "candle_matmul" and (
        gemv_admitted or len(a_shape) > 2
    ):
        raise ValueError(f"{source}.observed_path: shape does not admit Candle matmul")


def validate_run(record: dict[str, Any], source: str) -> None:
    required = (
        "schema",
        "run_id",
        "source",
        "hardware",
        "environment",
        "model",
        "workload",
        "capture",
        "entries",
        "accepted_claims",
        "excluded_claims",
    )
    _require_exact_fields(record, required, source)
    if record["schema"] != SCHEMA:
        raise ValueError(f"{source}.schema: expected {SCHEMA}")
    _require_string(record["run_id"], f"{source}.run_id")
    _validate_source(_require_object(record["source"], f"{source}.source"), f"{source}.source")
    _validate_hardware(
        _require_object(record["hardware"], f"{source}.hardware"),
        f"{source}.hardware",
    )
    _validate_environment(
        _require_object(record["environment"], f"{source}.environment"),
        f"{source}.environment",
    )
    _validate_model(_require_object(record["model"], f"{source}.model"), f"{source}.model")
    _validate_workload(
        _require_object(record["workload"], f"{source}.workload"),
        f"{source}.workload",
    )
    _validate_capture(
        _require_object(record["capture"], f"{source}.capture"),
        f"{source}.capture",
    )
    entries = record["entries"]
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{source}.entries: must be a nonempty list")
    identities: set[str] = set()
    phase_calls = defaultdict(int)
    for index, raw_entry in enumerate(entries):
        entry = _require_object(raw_entry, f"{source}.entries[{index}]")
        _validate_entry(entry, f"{source}.entries[{index}]")
        identity = _entry_identity(entry)
        if identity in identities:
            raise ValueError(f"{source}.entries[{index}]: duplicate census entry")
        identities.add(identity)
        phase_calls[entry["phase"]] += entry["host_calls"]
    workload = record["workload"]
    if workload["prompt_tokens"] > 0 and phase_calls["prefill"] == 0:
        raise ValueError(f"{source}.entries: prompt workload has no prefill calls")
    if workload["decode_forward_steps"] > 0 and phase_calls["decode"] == 0:
        raise ValueError(f"{source}.entries: decode workload has no decode calls")
    if workload["decode_forward_steps"] == 0 and phase_calls["decode"] != 0:
        raise ValueError(f"{source}.entries: decode calls exist without decode forwards")
    for field in ("accepted_claims", "excluded_claims"):
        values = record[field]
        if not isinstance(values, list) or not values or any(
            not isinstance(item, str) or not item for item in values
        ):
            raise ValueError(f"{source}.{field}: must be a nonempty string list")
    if record["accepted_claims"] != ACCEPTED_CLAIMS:
        raise ValueError(f"{source}.accepted_claims: unsupported claim boundary")
    if record["excluded_claims"] != EXCLUDED_CLAIMS:
        raise ValueError(f"{source}.excluded_claims: unsupported claim boundary")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def load_runs(paths: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    records: list[dict[str, Any]] = []
    inputs: list[dict[str, str]] = []
    run_ids: set[str] = set()
    for path_string in paths:
        path = Path(path_string)
        contents = path.read_bytes()
        inputs.append(
            {"path": path_string, "sha256": hashlib.sha256(contents).hexdigest()}
        )
        text = contents.decode("utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            raw = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
            source = f"{path}:{line_number}"
            record = _require_object(raw, source)
            validate_run(record, source)
            run_id = record["run_id"]
            if run_id in run_ids:
                raise ValueError(f"{source}: duplicate run_id {run_id!r}")
            run_ids.add(run_id)
            records.append(record)
    if not records:
        raise ValueError("no census records found")
    return records, inputs


def _frozen(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _require_one_contract(records: list[dict[str, Any]]) -> None:
    fields = (
        "source",
        "hardware",
        "environment",
        "model",
        "workload",
        "capture",
        "accepted_claims",
        "excluded_claims",
    )
    baseline = records[0]
    for index, record in enumerate(records[1:], start=2):
        changed = [
            field for field in fields if _frozen(record[field]) != _frozen(baseline[field])
        ]
        if changed:
            raise ValueError(
                f"record {index}: incompatible census contract fields: {', '.join(changed)}"
            )


def _oxide_contract_compatible(entry: dict[str, Any]) -> bool:
    return (
        entry["a_dtype"] == "bf16"
        and entry["weight_dtype"] == "bf16"
        and entry["a_layout"] == "row_major_contiguous"
        and entry["weight_layout"] == "row_major_contiguous_nk"
        and entry["transpose_a"] is False
        and entry["transpose_weight"] is True
        and not entry["post_ops"]
    )


def _shape_key(entry: dict[str, Any]) -> tuple[Any, ...]:
    return (
        entry["m"],
        entry["n"],
        entry["k"],
        entry["device_kind"],
        entry["device_ordinal"],
        tuple(entry["a_shape"]),
        tuple(entry["a_stride"]),
        entry["a_offset_elements"],
        tuple(entry["weight_shape"]),
        tuple(entry["weight_stride"]),
        entry["weight_offset_elements"],
        entry["a_dtype"],
        entry["weight_dtype"],
        entry["a_layout"],
        entry["weight_layout"],
        entry["transpose_a"],
        entry["transpose_weight"],
        tuple(entry["post_ops"]),
    )


def summarize_runs(
    records: list[dict[str, Any]], inputs: list[dict[str, str]]
) -> dict[str, Any]:
    run_ids: set[str] = set()
    for index, record in enumerate(records, start=1):
        validate_run(record, f"record {index}")
        run_id = record["run_id"]
        if run_id in run_ids:
            raise ValueError(f"record {index}: duplicate run_id {run_id!r}")
        run_ids.add(run_id)
    _require_one_contract(records)
    run_count = len(records)
    total_calls = sum(
        entry["host_calls"] for record in records for entry in record["entries"]
    )
    if total_calls <= 0:
        raise ValueError("census has no host calls")
    total_flops = sum(
        2 * entry["m"] * entry["n"] * entry["k"] * entry["host_calls"]
        for record in records
        for entry in record["entries"]
    )
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        for entry in record["entries"]:
            key = _shape_key(entry)
            group = grouped.setdefault(
                key,
                {
                    "m": entry["m"],
                    "n": entry["n"],
                    "k": entry["k"],
                    "device_kind": entry["device_kind"],
                    "device_ordinal": entry["device_ordinal"],
                    "a_shape": entry["a_shape"],
                    "a_stride": entry["a_stride"],
                    "a_offset_elements": entry["a_offset_elements"],
                    "weight_shape": entry["weight_shape"],
                    "weight_stride": entry["weight_stride"],
                    "weight_offset_elements": entry["weight_offset_elements"],
                    "a_dtype": entry["a_dtype"],
                    "weight_dtype": entry["weight_dtype"],
                    "a_layout": entry["a_layout"],
                    "weight_layout": entry["weight_layout"],
                    "transpose_a": entry["transpose_a"],
                    "transpose_weight": entry["transpose_weight"],
                    "post_ops": entry["post_ops"],
                    "oxide_logical_contract_compatible": _oxide_contract_compatible(entry),
                    "host_calls": 0,
                    "flops": 0,
                    "phase_calls": defaultdict(int),
                    "site_calls": defaultdict(int),
                    "observed_path_calls": defaultdict(int),
                },
            )
            calls = entry["host_calls"]
            flops = 2 * entry["m"] * entry["n"] * entry["k"] * calls
            group["host_calls"] += calls
            group["flops"] += flops
            group["phase_calls"][entry["phase"]] += calls
            group["site_calls"][entry["site"]] += calls
            group["observed_path_calls"][entry["observed_path"]] += calls
    ordered = sorted(
        grouped.values(),
        key=lambda group: (
            -group["host_calls"],
            -group["flops"],
            group["m"],
            group["n"],
            group["k"],
            group["a_dtype"],
            group["weight_dtype"],
            _shape_key(group),
        ),
    )
    cumulative_calls = 0
    shapes = []
    for group in ordered:
        cumulative_calls += group["host_calls"]
        shapes.append(
            {
                **{
                    key: value
                    for key, value in group.items()
                    if key
                    not in {"phase_calls", "site_calls", "observed_path_calls"}
                },
                "host_calls_per_run": group["host_calls"] / run_count,
                "call_share": group["host_calls"] / total_calls,
                "cumulative_call_share": cumulative_calls / total_calls,
                "flop_share": group["flops"] / total_flops,
                "phase_calls": dict(sorted(group["phase_calls"].items())),
                "site_calls": dict(sorted(group["site_calls"].items())),
                "observed_path_calls": dict(
                    sorted(group["observed_path_calls"].items())
                ),
            }
        )
    contract = {
        field: records[0][field]
        for field in (
            "source",
            "hardware",
            "environment",
            "model",
            "workload",
            "capture",
            "accepted_claims",
            "excluded_claims",
        )
    }
    decode_steps = records[0]["workload"]["decode_forward_steps"] * run_count
    decode_calls = sum(
        entry["host_calls"]
        for record in records
        for entry in record["entries"]
        if entry["phase"] == "decode"
    )
    return {
        "schema": "oxide.gemm-shape-census-summary.v1",
        "inputs": inputs,
        "run_ids": [record["run_id"] for record in records],
        "contract": contract,
        "totals": {
            "runs": run_count,
            "host_calls": total_calls,
            "host_calls_per_run": total_calls / run_count,
            "prefill_host_calls": total_calls - decode_calls,
            "decode_host_calls": decode_calls,
            "decode_host_calls_per_forward": (
                decode_calls / decode_steps if decode_steps else None
            ),
            "flops": total_flops,
        },
        "shapes": shapes,
    }


def _parse_args(args: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "summarize"):
        child = subparsers.add_parser(command)
        child.add_argument("paths", nargs="+")
    return parser.parse_args(args)


def main(args: list[str]) -> None:
    options = _parse_args(args)
    records, inputs = load_runs(options.paths)
    if options.command == "validate":
        print(json.dumps({"schema": SCHEMA, "runs": len(records), "status": "valid"}))
    else:
        print(json.dumps(summarize_runs(records, inputs), indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise SystemExit(str(error)) from error
