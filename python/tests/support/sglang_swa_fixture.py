"""Build a tiny deterministic Full + SWA model using SGLang's native Mellum path.

This is a numerical cache-recovery fixture, not a pretrained quality benchmark.
Generate it with the SGLang release environment; no downloads are required.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    import torch
    from transformers import AutoTokenizer, Qwen3MoeConfig, Qwen3MoeForCausalLM

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must not already exist")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    config = Qwen3MoeConfig(
        vocab_size=len(tokenizer),
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=2048,
        num_experts=4,
        num_experts_per_tok=2,
        mlp_only_layers=[0, 1, 2, 3],
        # SGLang's Mellum loader expects a separate lm_head tensor.
        tie_word_embeddings=False,
    )
    torch.manual_seed(42)
    model = Qwen3MoeForCausalLM(config).to(torch.bfloat16)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    path = args.output / "config.json"
    config = json.loads(path.read_text())
    config.update(
        architectures=["MellumForCausalLM"],
        layer_types=["sliding_attention", "full_attention"] * 2,
        mlp_layer_types=["dense"] * 4,
        sliding_window=256,
        use_sliding_window=True,
        rope_parameters={
            name: {"rope_type": "default", "rope_theta": 10000.0}
            for name in ("sliding_attention", "full_attention")
        },
    )
    path.write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
