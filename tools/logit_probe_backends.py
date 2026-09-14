"""Explicit adapters for independent oracle providers with restricted numeric ABIs."""

# DeepGEMM fp8_gemm_nt accepts 128 x 128 weight blocks with E4M3 data and
# float32 inverse scales; this is the provider ABI, not a model tuning choice.
DEEPGEMM_WEIGHT_BLOCK = (128, 128)


def install_deepgemm_fp8():
    import deep_gemm
    import torch
    import transformers.integrations.finegrained_fp8 as integration

    def linear(input, weight, weight_scale_inv, block_size=None, bias=None,
               activation_scale=None, output_dtype=None):
        if tuple(block_size or ()) != DEEPGEMM_WEIGHT_BLOCK or activation_scale is not None:
            raise ValueError("oracle DeepGEMM adapter requires dynamic activation and 128 x 128 FP8 weight blocks")
        if weight.dtype != torch.float8_e4m3fn or weight_scale_inv.dtype != torch.float32:
            raise ValueError("oracle DeepGEMM adapter requires E4M3 weights and float32 scales")
        flattened = input.reshape(-1, input.shape[-1])
        quantized, scales = deep_gemm.per_token_cast_to_fp8(flattened, False, block_size[-1])
        output = torch.empty((flattened.shape[0], weight.shape[0]), device=flattened.device,
                             dtype=output_dtype or input.dtype)
        deep_gemm.fp8_gemm_nt((quantized, scales), (weight, weight_scale_inv), output)
        output = output.reshape(input.shape[:-1] + (weight.shape[0],))
        return output if bias is None else output + bias

    integration.deepgemm_fp8_fp4_linear = linear
    return {"name": "deepgemm-fp8-block", "version": deep_gemm.__version__,
            "weight_block": list(DEEPGEMM_WEIGHT_BLOCK), "scales": "float32",
            "quantization": "deep_gemm.per_token_cast_to_fp8(use_ue8m0=False)"}
