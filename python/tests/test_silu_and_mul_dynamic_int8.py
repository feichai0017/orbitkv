from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from loom_kernels.torch_ops import (
    Operator,
    launch_count,
    reset_launch_count,
    silu_and_mul_dynamic_int8,
    silu_and_mul_dynamic_int8_out,
)


def compiled_native_reference(
    input_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pytest.importorskip("vllm")
    import vllm._custom_ops  # noqa: F401 - registers vLLM dispatcher ops

    width = input_tensor.shape[-1] // 2
    rows = input_tensor.numel() // input_tensor.shape[-1]
    gate = input_tensor[..., :width].float()
    up = input_tensor[..., width:].float()
    activated = (gate / (1.0 + torch.exp(-gate)) * up).to(input_tensor.dtype)
    output = torch.empty_like(activated, dtype=torch.int8)
    scales = torch.empty(
        (rows, 1), device=input_tensor.device, dtype=torch.float32
    )
    torch.ops._C.dynamic_scaled_int8_quant(output, activated, scales, None)
    return output, scales


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows,width", [(1, 4864), (32, 4864), (3, 127)])
def test_silu_and_mul_dynamic_int8_matches_compiled_native_on_external_stream(
    dtype, rows, width
):
    torch.manual_seed(83)
    input_tensor = torch.randn(rows, width * 2, device="cuda", dtype=dtype)
    expected_output, expected_scales = compiled_native_reference(input_tensor)

    reset_launch_count(Operator.SILU_AND_MUL_DYNAMIC_INT8)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        output, scales = silu_and_mul_dynamic_int8(input_tensor)
    stream.synchronize()

    assert launch_count(Operator.SILU_AND_MUL_DYNAMIC_INT8) == 1
    assert output.shape == (rows, width)
    assert output.dtype == torch.int8
    assert scales.shape == (rows, 1)
    assert torch.equal(output, expected_output)
    assert torch.equal(scales, expected_scales)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_silu_and_mul_dynamic_int8_preserves_prefix_and_reuses_buffers():
    input_tensor = torch.randn(2, 3, 256, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(2, 3, 128, device="cuda", dtype=torch.int8)
    scales = torch.empty(6, 1, device="cuda", dtype=torch.float32)
    output_pointer = output.data_ptr()
    scales_pointer = scales.data_ptr()
    expected_output, expected_scales = compiled_native_reference(input_tensor)

    returned_output, returned_scales = silu_and_mul_dynamic_int8_out(
        input_tensor, output, scales
    )
    torch.cuda.synchronize()

    assert returned_output is output
    assert returned_scales is scales
    assert output.data_ptr() == output_pointer
    assert scales.data_ptr() == scales_pointer
    assert torch.equal(output, expected_output)
    assert torch.equal(scales, expected_scales)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_silu_and_mul_dynamic_int8_preserves_zero_row_contract():
    input_tensor = torch.zeros(2, 256, device="cuda", dtype=torch.bfloat16)
    input_tensor[0] = torch.randn_like(input_tensor[0])
    expected_output, expected_scales = compiled_native_reference(input_tensor)

    output, scales = silu_and_mul_dynamic_int8(input_tensor)
    torch.cuda.synchronize()

    assert torch.equal(output, expected_output)
    assert torch.equal(scales, expected_scales)
    assert scales[1].item() == 0.0
    assert torch.count_nonzero(output[1]).item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_silu_and_mul_dynamic_int8_rejects_invalid_contracts_before_launch():
    input_tensor = torch.randn(2, 256, device="cuda", dtype=torch.bfloat16)
    scales = torch.empty(2, 1, device="cuda", dtype=torch.float32)
    reset_launch_count(Operator.SILU_AND_MUL_DYNAMIC_INT8)

    with pytest.raises(RuntimeError, match="output must use signed INT8"):
        silu_and_mul_dynamic_int8_out(
            input_tensor,
            torch.empty(2, 128, device="cuda", dtype=torch.bfloat16),
            scales,
        )
    with pytest.raises(RuntimeError, match=r"scales must have shape \[rows, 1\]"):
        silu_and_mul_dynamic_int8_out(
            input_tensor,
            torch.empty(2, 128, device="cuda", dtype=torch.int8),
            torch.empty(1, 2, device="cuda", dtype=torch.float32),
        )
    overlapping_output = input_tensor.view(torch.int8).flatten()[:256].view(2, 128)
    with pytest.raises(RuntimeError, match="storage must not overlap"):
        silu_and_mul_dynamic_int8_out(
            input_tensor,
            overlapping_output,
            scales,
        )

    assert launch_count(Operator.SILU_AND_MUL_DYNAMIC_INT8) == 0


def test_silu_and_mul_dynamic_int8_schema_declares_both_mutations():
    schema = str(
        torch.ops.loom_kernels.silu_and_mul_dynamic_per_token_int8.default._schema
    )
    assert "Tensor(a!) result" in schema
    assert "Tensor(b!) scale" in schema


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_silu_and_mul_dynamic_int8_survives_torch_compile():
    def target(input_tensor, output, scales):
        torch.ops.loom_kernels.silu_and_mul_dynamic_per_token_int8(
            output, input_tensor, scales
        )
        return output, scales

    input_tensor = torch.randn(4, 512, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(4, 256, device="cuda", dtype=torch.int8)
    scales = torch.empty(4, 1, device="cuda", dtype=torch.float32)
    expected_output, expected_scales = compiled_native_reference(input_tensor)

    actual_output, actual_scales = torch.compile(target, fullgraph=True)(
        input_tensor, output, scales
    )
    torch.cuda.synchronize()

    assert torch.equal(actual_output, expected_output)
    assert torch.equal(actual_scales, expected_scales)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_silu_and_mul_dynamic_int8_can_be_captured_and_replayed():
    input_tensor = torch.randn(4, 512, device="cuda", dtype=torch.float16)
    output = torch.empty(4, 256, device="cuda", dtype=torch.int8)
    scales = torch.empty(4, 1, device="cuda", dtype=torch.float32)
    expected_output, expected_scales = compiled_native_reference(input_tensor)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.ops.loom_kernels.silu_and_mul_dynamic_per_token_int8(
            output, input_tensor, scales
        )
    output.zero_()
    scales.zero_()
    graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(output, expected_output)
    assert torch.equal(scales, expected_scales)
