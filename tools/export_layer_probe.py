#!/usr/bin/env python3
"""Capture unchanged Transformers boundaries for the executor layer probe."""

from __future__ import annotations

import argparse
import functools
import importlib
import inspect
import json
from pathlib import Path
import re

from logit_probe import identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--oracle-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--module-pattern", required=True, help="Regex selecting named modules to observe.")
    parser.add_argument("--function", action="append", default=[], help="Optional module:function to observe inside selected modules.")
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    probe = json.loads(args.manifest.read_text())
    metadata = json.loads(args.oracle_metadata.read_text())
    if not 0 <= args.case_index < len(probe["cases"]) or args.steps <= 0:
        parser.error("case index must exist and steps must be positive")
    case = probe["cases"][args.case_index]
    if args.steps > len(case["continuation_token_ids"]):
        parser.error("steps exceed the supplied teacher-forced history")
    pattern = re.compile(args.module_pattern)

    import torch
    import transformers
    from safetensors.torch import save_file

    backend = metadata["linear_backend"]
    if backend == "deepgemm-fp8-block":
        from logit_probe_backends import install_deepgemm_fp8
        install_deepgemm_fp8()
    elif backend != "transformers":
        parser.error(f"unsupported reference backend: {backend}")
    config = transformers.AutoConfig.from_pretrained(probe["model_directory"], local_files_only=True)
    for key, value in metadata.get("config_overrides", {}).items():
        setattr(config, key, value)
    device = f"cuda:{probe['device_index']}"
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records, stack, handles, originals = {}, [], [], []

    def keep(name, value):
        if isinstance(value, torch.Tensor):
            records[name] = value.detach().contiguous().cpu().clone()
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                keep(f"{name}.{index}", item)

    def before(name, module, inputs):
        stack.append(name)
        keep(name + ".input", inputs)

    def after(name, module, inputs, result):
        keep(name + ".output", result)
        assert stack.pop() == name

    def wrap(original, operation):
        @functools.wraps(original)
        def invoke(*inputs, **kwargs):
            scope = stack[-1] if stack else None
            if scope:
                for key, value in inspect.signature(original).bind_partial(*inputs, **kwargs).arguments.items():
                    keep(f"{scope}.{operation}.input.{key}", value)
            result = original(*inputs, **kwargs)
            if scope:
                keep(f"{scope}.{operation}.output", result)
            return result
        return invoke

    try:
        for target in args.function:
            module_name, operation = target.rsplit(":", 1)
            module = importlib.import_module(module_name)
            original = getattr(module, operation)
            if not callable(original):
                raise ValueError(f"reference function is not callable: {target}")
            originals.append((module, operation, original))
            setattr(module, operation, wrap(original, operation))
        model = getattr(transformers, metadata["model_class"]).from_pretrained(
            probe["model_directory"], config=config, dtype=getattr(torch, metadata["dtype"]),
            device_map=device, low_cpu_mem_usage=True, local_files_only=True,
        ).eval()
        for name, module in model.named_modules():
            if pattern.search(name):
                handles.extend((module.register_forward_pre_hook(functools.partial(before, name)),
                                module.register_forward_hook(functools.partial(after, name))))
        if not handles:
            raise ValueError("module pattern matched no reference modules")
        tokens, cache = case["prompt_token_ids"], None
        with torch.inference_mode():
            for step in range(args.steps):
                records.clear()
                result = model(input_ids=torch.tensor([tokens], device=device), past_key_values=cache, use_cache=True)
                keep("logits", result.logits)
                save_file(records, args.output_dir / f"step-{step}.safetensors")
                print(json.dumps({"step": step, "tensors": len(records)}), flush=True)
                cache = result.past_key_values
                tokens = [case["continuation_token_ids"][step]]
    finally:
        for handle in handles:
            handle.remove()
        for module, operation, original in reversed(originals):
            setattr(module, operation, original)
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "model_directory": probe["model_directory"], "case": case,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "module_pattern": args.module_pattern, "functions": args.function,
        "steps": args.steps, "inputs": [identity(args.manifest), identity(args.oracle_metadata), identity(Path(__file__))],
        "scope": "Hooks retain original reference computations; an optional DeepGEMM adapter shares the external GEMM library.",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
