"""Create a small local Qwen2-MoE checkpoint for offline engine gates."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--moe-intermediate-size", type=int, default=256)
    parser.add_argument("--shared-expert-intermediate-size", type=int, default=256)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--experts-per-token", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--max-position-embeddings", type=int, default=256)
    parser.add_argument("--seed", type=int, default=83)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(
        args.layers,
        args.hidden_size,
        args.intermediate_size,
        args.moe_intermediate_size,
        args.shared_expert_intermediate_size,
        args.attention_heads,
        args.kv_heads,
        args.experts,
        args.experts_per_token,
        args.vocab_size,
        args.max_position_embeddings,
    ) <= 0:
        raise ValueError("all synthetic model dimensions must be positive")
    if args.hidden_size % args.attention_heads:
        raise ValueError("hidden size must be divisible by attention heads")
    if args.experts_per_token > args.experts:
        raise ValueError("experts per token must not exceed experts")

    import torch
    from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM

    torch.manual_seed(args.seed)
    config = Qwen2MoeConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.layers,
        num_attention_heads=args.attention_heads,
        num_key_value_heads=args.kv_heads,
        max_position_embeddings=args.max_position_embeddings,
        rms_norm_eps=1.0e-6,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
        decoder_sparse_step=1,
        moe_intermediate_size=args.moe_intermediate_size,
        shared_expert_intermediate_size=args.shared_expert_intermediate_size,
        num_experts_per_tok=args.experts_per_token,
        num_experts=args.experts,
        norm_topk_prob=True,
        mlp_only_layers=[],
    )
    config.loom_fixture = "synthetic-random-qwen2-moe"
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = Qwen2MoeForCausalLM(config)
    finally:
        torch.set_default_dtype(original_dtype)
    model.eval()
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True, max_shard_size="2GB")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        {
            "output": str(args.output.resolve()),
            "parameters": parameters,
            "storage_dtype": str(next(model.parameters()).dtype),
            "layers": args.layers,
            "experts": args.experts,
            "experts_per_token": args.experts_per_token,
        }
    )


if __name__ == "__main__":
    main()
