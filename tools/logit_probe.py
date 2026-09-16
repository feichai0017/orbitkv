#!/usr/bin/env python3
"""Export independent decoder logits or compare teacher-forced traces.

Inputs are local JSON manifests. This tool never searches schedules, downloads a
checkpoint, changes numerical thresholds, or promotes diagnostic output.
"""
from __future__ import annotations

import argparse
import array
import hashlib
import heapq
import importlib
import inspect
import json
import math
from pathlib import Path
import sys
import time

SCHEMA = 1
FLOAT_BYTES = 4


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read_row(path, vocabulary):
    raw = Path(path).read_bytes()
    if len(raw) != vocabulary * FLOAT_BYTES:
        raise ValueError(f"invalid vocabulary row size: {path}")
    row = array.array("f")
    row.frombytes(raw)
    if sys.byteorder != "little":
        row.byteswap()
    if not all(math.isfinite(value) for value in row):
        raise ValueError(f"non-finite logits: {path}")
    return row


def top_tokens(row, count):
    return [{"token_id": index, "logit": row[index]}
            for index in heapq.nsmallest(count, range(len(row)),
                                        key=lambda index: (-row[index], index))]


def compare_rows(reference, candidate, top_count):
    if len(reference) != len(candidate) or len(reference) < 2:
        raise ValueError("rows must have matching vocabularies of at least two tokens")
    if not all(math.isfinite(value) for row in (reference, candidate) for value in row):
        raise ValueError("comparison requires finite logits")
    if top_count < 2:
        raise ValueError("top_count must include at least two tokens")
    reference_top = top_tokens(reference, top_count)
    candidate_top = top_tokens(candidate, top_count)
    reference_id = reference_top[0]["token_id"]
    candidate_id = candidate_top[0]["token_id"]
    reference_maxima = [index for index, value in enumerate(reference) if value == reference[reference_id]]
    candidate_maxima = [index for index, value in enumerate(candidate) if value == candidate[candidate_id]]
    errors = [abs(actual - expected) for expected, actual in zip(reference, candidate)]
    worst = max(range(len(errors)), key=errors.__getitem__)
    return {"argmax_equal": reference_id == candidate_id,
            "canonical_argmax_policy": "lowest-token-id",
            "reference_top": reference_top, "candidate_top": candidate_top,
            "reference_maximum_tie_count": len(reference_maxima),
            "candidate_maximum_tie_count": len(candidate_maxima),
            "reference_maximum_token_ids": reference_maxima,
            "candidate_maximum_token_ids": candidate_maxima,
            "highest_token_id_argmax_equal": reference_maxima[-1] == candidate_maxima[-1],
            "reference_top2_margin": reference_top[0]["logit"] - reference_top[1]["logit"],
            "candidate_top2_margin": candidate_top[0]["logit"] - candidate_top[1]["logit"],
            "reference_preference_for_its_winner": reference[reference_id] - reference[candidate_id],
            "candidate_preference_for_its_winner": candidate[candidate_id] - candidate[reference_id],
            "maximum_absolute_error": errors[worst], "worst_error_token_id": worst,
            "root_mean_square_error": math.sqrt(math.fsum(error * error for error in errors) / len(errors))}


def compare_traces(reference_dir, candidate_dir, top_count):
    reference_dir, candidate_dir = Path(reference_dir), Path(candidate_dir)
    reference, candidate = (read_json(directory / "trace.json") for directory in (reference_dir, candidate_dir))
    for trace in (reference, candidate):
        if trace["schema"] != SCHEMA or not trace["teacher_forced"]:
            raise ValueError("comparison requires versioned teacher-forced traces")
    vocabulary = reference["vocabulary_size"]
    if vocabulary != candidate["vocabulary_size"] or len(reference["cases"]) != len(candidate["cases"]):
        raise ValueError("trace vocabulary or case count mismatch")
    cases = []
    files = [identity(directory / "trace.json") for directory in (reference_dir, candidate_dir)]
    for expected, actual in zip(reference["cases"], candidate["cases"]):
        if (expected["id"], expected["prompt_token_ids"]) != (actual["id"], actual["prompt_token_ids"]):
            raise ValueError("case identity or prompt mismatch")
        if len(expected["steps"]) != len(actual["steps"]):
            raise ValueError("step count mismatch")
        steps = []
        for index, (left, right) in enumerate(zip(expected["steps"], actual["steps"])):
            if left["step"] != index or right["step"] != index or left["input_token_ids"] != right["input_token_ids"]:
                raise ValueError("step numbering or teacher-forced input mismatch")
            paths = [reference_dir / left["logits_file"], candidate_dir / right["logits_file"]]
            rows = [read_row(path, vocabulary) for path in paths]
            files.extend(identity(path) for path in paths)
            metrics = compare_rows(*rows, top_count)
            for row, item in zip(rows, (left, right)):
                if not 0 <= item["selected_token_id"] < len(row):
                    raise ValueError("selected token is outside the vocabulary")
            metrics.update(step=index, input_token_ids=left["input_token_ids"],
                           reference_selected_token_id=left["selected_token_id"],
                           candidate_selected_token_id=right["selected_token_id"],
                           selected_token_equal=left["selected_token_id"] == right["selected_token_id"])
            metrics["selected_token_is_logit_maximum"] = rows[1][right["selected_token_id"]] == max(rows[1])
            metrics["reference_selected_token_is_logit_maximum"] = rows[0][left["selected_token_id"]] == max(rows[0])
            metrics["candidate_selected_matches_highest_maximum_token_id"] = right["selected_token_id"] == metrics["candidate_maximum_token_ids"][-1]
            metrics["reference_logits_at_selected_tokens"] = {
                "reference_selected": rows[0][left["selected_token_id"]],
                "candidate_selected": rows[0][right["selected_token_id"]],
            }
            metrics["candidate_logits_at_selected_tokens"] = {
                "reference_selected": rows[1][left["selected_token_id"]],
                "candidate_selected": rows[1][right["selected_token_id"]],
            }
            steps.append(metrics)
        cases.append({"id": expected["id"], "prompt_token_ids": expected["prompt_token_ids"],
                      "first_argmax_difference": next((step["step"] for step in steps if not step["argmax_equal"]), None),
                      "first_selected_token_difference": next((step["step"] for step in steps if not step["selected_token_equal"]), None),
                      "steps": steps, "candidate_drain_passed": actual.get("drain_passed")})
    return {"schema": SCHEMA, "scope": "Teacher-forced numerical diagnostic; no tolerance or performance qualification is implied.",
            "reference": str(reference_dir.resolve()), "candidate": str(candidate_dir.resolve()),
            "vocabulary_size": vocabulary, "cases": cases, "files": files}


def validate_manifest(manifest):
    if manifest["schema"] != SCHEMA or not manifest["cases"]:
        raise ValueError("invalid or empty probe manifest")
    ids = [case["id"] for case in manifest["cases"]]
    if len(set(ids)) != len(ids):
        raise ValueError("case IDs must be unique")
    for case in manifest["cases"]:
        if not case["prompt_token_ids"] or not case["continuation_token_ids"]:
            raise ValueError("each case requires a prompt and a continuation length")
        if any(type(token) is not int or token < 0 for token in case["prompt_token_ids"] + case["continuation_token_ids"]):
            raise ValueError("token IDs must be nonnegative integers")


def export_oracle(args):
    # Heavy dependencies stay confined to oracle export, not trace comparison.
    import torch
    import transformers
    manifest = read_json(args.manifest)
    validate_manifest(manifest)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    model_directory = Path(manifest["model_directory"]).resolve()
    config = transformers.AutoConfig.from_pretrained(model_directory, local_files_only=True)
    overrides = read_json(args.config_overrides) if args.config_overrides else {}
    for key, value in overrides.items():
        if not hasattr(config, key):
            raise ValueError(f"unknown model config override: {key}")
        setattr(config, key, value)
    adapter = None
    if args.linear_backend == "deepgemm-fp8-block":
        from logit_probe_backends import install_deepgemm_fp8
        adapter = install_deepgemm_fp8()
    model_class = getattr(transformers, args.model_class)
    device = f"cuda:{manifest['device_index']}"
    started = time.monotonic()
    model = model_class.from_pretrained(model_directory, config=config, dtype=getattr(torch, args.dtype),
                                        device_map=device, low_cpu_mem_usage=True, local_files_only=True)
    model.eval()
    loaded_seconds = time.monotonic() - started
    traces = []
    with torch.inference_mode():
        for case_index, case in enumerate(manifest["cases"]):
            cache = None
            tokens = case["prompt_token_ids"]
            generated = []
            steps = []
            for step in range(len(case["continuation_token_ids"])):
                boundary = len(case["prompt_token_ids"]) + step
                positions = list(range(boundary - len(tokens), boundary))
                result = model(input_ids=torch.tensor([tokens], device=device),
                               position_ids=torch.tensor([positions], device=device),
                               past_key_values=cache, use_cache=True)
                row = result.logits[0, -1].float().cpu()
                if not bool(torch.isfinite(row).all()):
                    raise ValueError("oracle produced non-finite logits")
                selected = int(row.argmax().item())
                logits_file = f"case-{case_index}-step-{step}.f32"
                row.numpy().astype("<f4").tofile(output / logits_file)
                steps.append({"step": step, "input_token_ids": tokens, "selected_token_id": selected, "logits_file": logits_file})
                generated.append(selected)
                cache = result.past_key_values
                tokens = [selected if args.continuation == "greedy" else case["continuation_token_ids"][step]]
            if args.continuation == "greedy":
                case["continuation_token_ids"] = generated
            traces.append({"id": case["id"], "prompt_token_ids": case["prompt_token_ids"], "steps": steps})
    modules = {type(module).__module__ for module in model.modules()}
    modules.update(("torch", "transformers", "transformers.integrations.finegrained_fp8"))
    if adapter:
        modules.add("deep_gemm")
    module_files = sorted({inspect.getfile(importlib.import_module(name)) for name in modules})
    write_json(output / "trace.json", {"schema": SCHEMA, "teacher_forced": True,
                "vocabulary_size": row.numel(), "cases": traces})
    write_json(output / "probe.json", manifest)
    metadata = {"schema": SCHEMA, "model_class": args.model_class, "dtype": args.dtype,
                "continuation": args.continuation, "linear_backend": args.linear_backend,
                "adapter": adapter, "loaded_seconds": loaded_seconds,
                "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
                "inputs": [identity(args.manifest), identity(model_directory / "config.json"), identity(Path(__file__))],
                "source_modules": [identity(path) for path in module_files],
                "config_overrides": overrides,
                "limitations": ["Independent model implementation; the optional DeepGEMM backend shares the external GEMM library with OrbitKV.",
                                "Teacher-forced steps do not assert free-running text equality or serving performance."]}
    if args.config_overrides:
        metadata["inputs"].append(identity(args.config_overrides))
    if adapter:
        metadata["inputs"].append(identity(Path(__file__).with_name("logit_probe_backends.py")))
    for name in ("model.safetensors.index.json", "README.md"):
        if (model_directory / name).is_file():
            metadata["inputs"].append(identity(model_directory / name))
    write_json(output / "metadata.json", metadata)
    print(json.dumps({"output": str(output), "cases": len(traces), "loaded_seconds": loaded_seconds}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    oracle = commands.add_parser("oracle")
    oracle.add_argument("--manifest", required=True, type=Path)
    oracle.add_argument("--output-dir", required=True, type=Path)
    oracle.add_argument("--model-class", required=True)
    oracle.add_argument("--dtype", required=True, choices=("float32", "float16", "bfloat16"))
    oracle.add_argument("--linear-backend", choices=("transformers", "deepgemm-fp8-block"), default="transformers")
    oracle.add_argument("--config-overrides", type=Path)
    oracle.add_argument("--continuation", choices=("greedy", "manifest"), default="greedy")
    compare = commands.add_parser("compare")
    compare.add_argument("--reference-dir", required=True, type=Path)
    compare.add_argument("--candidate-dir", required=True, type=Path)
    compare.add_argument("--top-count", required=True, type=int)
    compare.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "oracle":
        export_oracle(args)
    else:
        if args.output.exists():
            raise ValueError("comparison output must be fresh")
        write_json(args.output, compare_traces(args.reference_dir, args.candidate_dir, args.top_count))


if __name__ == "__main__":
    main()
