"""Explicitly regenerate the fixed, model-independent Torch/DeepGEMM fixture."""

import json
from pathlib import Path

import deep_gemm
import torch


def main() -> None:
    seed = 281
    inputs = torch.randn(3, 256, generator=torch.Generator().manual_seed(seed))
    inputs = inputs.to(torch.bfloat16)
    inputs[0, :128] = (inputs[0, :128] * 0.1).clamp(-1, 1)
    # The old exact quotient is 7/2048; 217/512 then lands exactly at
    # the FP8 midpoint 124. Rounded reciprocal scaling changes that decision.
    inputs[0, 0] = 1.53125
    inputs[0, 1] = 0.423828125
    inputs[0, 128:] *= 1e-6
    inputs[1, 0] = 0.0
    inputs[1, 1] = -0.0
    inputs = inputs.cuda()
    quantized, scales = deep_gemm.per_token_cast_to_fp8(inputs, False)
    packed_scales = torch.zeros(2, 4, dtype=torch.float32, device=inputs.device)
    packed_scales[:, :3] = scales.T

    # Independent old CUDA semantics: rounded division, rather than Torch's
    # scalar division lowering. F64 prevents accidental reciprocal strength
    # reduction at the F32 boundary that this fixture needs to distinguish.
    values = inputs.reshape(3, 2, 128).float()
    maximum = values.abs().amax(dim=-1).clamp(min=1e-4)
    old_scales = (maximum.double() / 448).float()
    old_values = (values.double() / old_scales.unsqueeze(-1).double()).float()
    old_quantized = old_values.to(torch.float8_e4m3fn).reshape_as(quantized)

    payload = {
        "producer": {
            "torch": torch.__version__,
            "deepgemm": deep_gemm.__version__,
            "operation": "per_token_cast_to_fp8(x, use_ue8m0=False)",
            "seed": seed,
        },
        "m": 3,
        "k": 256,
        "input_bf16_bits": inputs.view(torch.int16).cpu().reshape(-1).numpy().view("uint16").tolist(),
        "fp8_bytes": quantized.view(torch.uint8).cpu().reshape(-1).tolist(),
        "scale_f32_bits": packed_scales.view(torch.int32).cpu().reshape(-1).numpy().view("uint32").tolist(),
        "old_division_scale_mismatches": int((old_scales.view(torch.int32) != scales.view(torch.int32)).sum()),
        "old_division_quantized_byte_mismatches": int((old_quantized.view(torch.uint8) != quantized.view(torch.uint8)).sum()),
    }
    assert payload["old_division_scale_mismatches"] > 0
    assert payload["old_division_quantized_byte_mismatches"] > 0
    path = Path(__file__).resolve().with_name("row128-rne.json")
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print({key: value for key, value in payload.items() if not isinstance(value, list)})


if __name__ == "__main__":
    main()
