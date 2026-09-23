"""Create a small native Inkling Full + SWA + convolution configuration.

Use --sglang-load-format dummy for the serving gate and a fixed explicit
ORBITKV_MODEL_FINGERPRINT identifying this configuration and initialization seed.
This exercises state recovery, not pretrained model quality.
"""

import argparse
from pathlib import Path


def main():
    from sglang.srt.configs.inkling import InklingMMConfig, InklingModelConfig
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must not already exist")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    config = InklingMMConfig(
        text_config=InklingModelConfig(
            vocab_size=len(tokenizer),
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=4,
            dense_mlp_idx=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            local_layer_ids=[1, 3],
            sliding_window_size=256,
            use_sconv=True,
            dtype="bfloat16",
            num_nextn_predict_layers=0,
            max_position_embeddings=2048,
        ),
        architectures=["InklingForConditionalGeneration"],
        dtype="bfloat16",
        max_position_embeddings=2048,
    )
    config.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()
