from __future__ import annotations

import importlib.util

import pytest

torch = pytest.importorskip("torch")

from loom_kernels.torch_ops import (
    Operator,
    launch_count,
    reset_launch_count,
    rms_norm_dynamic_int8,
    rms_norm_dynamic_int8_out,
)


def assert_dynamic_int8_close(
    actual_output: torch.Tensor,
    actual_scales: torch.Tensor,
    expected_output: torch.Tensor,
    expected_scales: torch.Tensor,
) -> None:
    torch.testing.assert_close(
        actual_scales, expected_scales, rtol=2.0e-6, atol=1.0e-8
    )
    integer_delta = (
        actual_output.to(torch.int16) - expected_output.to(torch.int16)
    ).abs()
    assert integer_delta.max().item() <= 1
    assert torch.count_nonzero(integer_delta).item() <= actual_output.shape[0]

    actual_dequantized = actual_output.float() * actual_scales
    expected_dequantized = expected_output.float() * expected_scales
    quantization_step = max(
        actual_scales.abs().max().item(),
        expected_scales.abs().max().item(),
    )
    torch.testing.assert_close(
        actual_dequantized,
        expected_dequantized,
        rtol=2.0e-6,
        atol=quantization_step + 1.0e-8,
    )


def dynamic_int8_reference(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    residual: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if residual is None:
        summed = input_tensor.float()
    else:
        summed = input_tensor.float() + residual.float()
        residual.copy_(summed.to(residual.dtype))

    inverse_rms = torch.rsqrt(
        summed.square().mean(dim=-1, keepdim=True) + epsilon
    )
    normalized = (summed * inverse_rms).to(input_tensor.dtype)
    weighted = (normalized * weight).to(input_tensor.dtype)
    absolute_maximum = weighted.float().abs().amax(dim=-1, keepdim=True)
    scales = absolute_maximum / 127.0
    inverse_scale = torch.where(
        absolute_maximum == 0,
        torch.zeros_like(absolute_maximum),
        127.0 / absolute_maximum,
    )
    output = (
        (weighted.float() * inverse_scale)
        .round()
        .clamp(-128.0, 127.0)
        .to(torch.int8)
    )
    return output, scales


def vllm_ir_dynamic_int8_reference(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    residual: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm import ir

    output = torch.empty_like(input_tensor, dtype=torch.int8)
    rows = input_tensor.numel() // input_tensor.shape[-1]
    scales = torch.empty(
        (rows, 1), device=input_tensor.device, dtype=torch.float32
    )
    if residual is None:
        normalized = ir.ops.rms_norm(input_tensor, weight, epsilon)
    else:
        normalized, updated_residual = ir.ops.fused_add_rms_norm(
            input_tensor, residual, weight, epsilon
        )
        residual.copy_(updated_residual)
    torch.ops._C.dynamic_scaled_int8_quant(
        output, normalized, scales, None
    )
    return output, scales


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(8, 896), (3, 127)])
@pytest.mark.parametrize("with_residual", [False, True])
def test_rms_norm_dynamic_int8_matches_reference(
    dtype, shape, with_residual
):
    torch.manual_seed(43)
    epsilon = 1.0e-6
    input_tensor = torch.randn(shape, device="cuda", dtype=dtype)
    weight = torch.randn(shape[-1], device="cuda", dtype=dtype)
    residual = (
        torch.randn(shape, device="cuda", dtype=dtype)
        if with_residual
        else None
    )
    expected_residual = residual.clone() if residual is not None else None
    expected_output, expected_scales = dynamic_int8_reference(
        input_tensor, weight, epsilon, expected_residual
    )
    if importlib.util.find_spec("vllm") is not None:
        vllm_residual = residual.clone() if residual is not None else None
        vllm_output, vllm_scales = vllm_ir_dynamic_int8_reference(
            input_tensor, weight, epsilon, vllm_residual
        )
        assert_dynamic_int8_close(
            expected_output, expected_scales, vllm_output, vllm_scales
        )
        if expected_residual is not None:
            assert vllm_residual is not None
            assert torch.equal(expected_residual, vllm_residual)

    reset_launch_count(Operator.RMS_NORM_DYNAMIC_INT8)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        output, scales = rms_norm_dynamic_int8(
            input_tensor, weight, epsilon, residual
        )
    stream.synchronize()

    assert launch_count(Operator.RMS_NORM_DYNAMIC_INT8) == 1
    assert output.dtype == torch.int8
    assert scales.shape == (input_tensor.numel() // shape[-1], 1)
    assert_dynamic_int8_close(
        output, scales, expected_output, expected_scales
    )
    if residual is not None:
        assert expected_residual is not None
        assert torch.equal(residual, expected_residual)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rms_norm_dynamic_int8_preserves_zero_row_scale():
    input_tensor = torch.zeros(2, 128, device="cuda", dtype=torch.bfloat16)
    weight = torch.ones(128, device="cuda", dtype=torch.bfloat16)

    output, scales = rms_norm_dynamic_int8(
        input_tensor, weight, 1.0e-6
    )
    torch.cuda.synchronize()

    assert torch.count_nonzero(output).item() == 0
    assert torch.equal(scales, torch.zeros_like(scales))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rms_norm_dynamic_int8_out_reuses_caller_buffers():
    input_tensor = torch.randn(4, 896, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(896, device="cuda", dtype=torch.bfloat16)
    output = torch.empty_like(input_tensor, dtype=torch.int8)
    scales = torch.empty(4, 1, device="cuda", dtype=torch.float32)
    output_pointer = output.data_ptr()
    scales_pointer = scales.data_ptr()

    returned_output, returned_scales = rms_norm_dynamic_int8_out(
        input_tensor, weight, output, scales, 1.0e-6
    )
    torch.cuda.synchronize()

    assert returned_output is output
    assert returned_scales is scales
    assert output.data_ptr() == output_pointer
    assert scales.data_ptr() == scales_pointer


def test_rms_norm_dynamic_int8_schema_declares_all_mutations():
    schema = str(
        torch.ops.loom_kernels.rms_norm_dynamic_per_token_int8.default._schema
    )
    assert "Tensor(a!) result" in schema
    assert "Tensor(b!) scale" in schema
    assert "Tensor(c!)? residual=None" in schema


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rms_norm_dynamic_int8_survives_torch_compile():
    def target(input_tensor, residual, weight, output, scales):
        torch.ops.loom_kernels.rms_norm_dynamic_per_token_int8(
            output,
            input_tensor,
            weight,
            scales,
            1.0e-6,
            residual,
        )
        return output, scales, residual

    compiled = torch.compile(target, fullgraph=True)
    input_tensor = torch.randn(2, 896, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(input_tensor)
    expected_residual = residual.clone()
    weight = torch.randn(896, device="cuda", dtype=torch.bfloat16)
    output = torch.empty_like(input_tensor, dtype=torch.int8)
    scales = torch.empty(2, 1, device="cuda", dtype=torch.float32)
    expected_output, expected_scales = dynamic_int8_reference(
        input_tensor, weight, 1.0e-6, expected_residual
    )

    actual_output, actual_scales, actual_residual = compiled(
        input_tensor, residual, weight, output, scales
    )
    torch.cuda.synchronize()

    assert_dynamic_int8_close(
        actual_output,
        actual_scales,
        expected_output,
        expected_scales,
    )
    assert torch.equal(actual_residual, expected_residual)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rms_norm_dynamic_int8_can_be_captured_and_replayed():
    input_tensor = torch.randn(2, 896, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(896, device="cuda", dtype=torch.bfloat16)
    expected_output, expected_scales = dynamic_int8_reference(
        input_tensor, weight, 1.0e-6
    )
    output = torch.empty_like(input_tensor, dtype=torch.int8)
    scales = torch.empty(2, 1, device="cuda", dtype=torch.float32)

    reset_launch_count(Operator.RMS_NORM_DYNAMIC_INT8)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.ops.loom_kernels.rms_norm_dynamic_per_token_int8(
            output, input_tensor, weight, scales, 1.0e-6, None
        )
    output.zero_()
    scales.zero_()
    graph.replay()
    torch.cuda.synchronize()

    assert launch_count(Operator.RMS_NORM_DYNAMIC_INT8) == 1
    assert_dynamic_int8_close(
        output, scales, expected_output, expected_scales
    )
