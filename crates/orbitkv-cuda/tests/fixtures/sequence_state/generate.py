import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def bits(tensor):
    return tensor.contiguous().view(torch.uint16).cpu().reshape(-1).tolist()


def f32_bits(tensor):
    return tensor.contiguous().view(torch.int32).cpu().reshape(-1).tolist()


def convolution_fixture():
    channels, width = 17, 4
    weights = torch.randn(channels, width, device="cuda", dtype=torch.bfloat16)
    history = torch.randn(3, channels, width - 1, device="cuda", dtype=torch.bfloat16)
    indptr = [0, 4, 4, 5]
    inputs = torch.randn(indptr[-1], channels, device="cuda", dtype=torch.bfloat16)
    outputs, states = [], []
    for row, (begin, end) in enumerate(zip(indptr, indptr[1:])):
        joined = torch.cat((history[row], inputs[begin:end].T), dim=1)
        if end > begin:
            conv = F.conv1d(joined[None], weights[:, None], groups=channels)
            outputs.append(F.silu(conv)[0].T.contiguous())
        states.append(joined[:, -(width - 1):].contiguous())
    return {
        "source": "Torch conv1d BF16 output followed by SiLU; unchanged standalone reference",
        "torch": torch.__version__, "seed": 319, "channels": channels,
        "kernel_width": width, "indptr": indptr, "input": bits(inputs),
        "weights": bits(weights), "history": bits(history),
        "output": bits(torch.cat(outputs)), "next_history": bits(torch.stack(states)),
    }


def delta_scan_fixture():
    key_width, value_width = 128, 8
    query = torch.randn(2, 1, key_width, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(2, 1, key_width, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(2, 1, value_width, device="cuda", dtype=torch.bfloat16)
    initial_state = torch.randn(1, 1, key_width, value_width, device="cuda", dtype=torch.bfloat16)
    log_decay = torch.tensor([[-0.125], [-0.5]], device="cuda", dtype=torch.float32)
    beta = torch.tensor([[0.625], [0.375]], device="cuda", dtype=torch.bfloat16)
    normalized_query = query * torch.rsqrt((query * query).sum(-1, keepdim=True) + 1e-6)
    normalized_key = key * torch.rsqrt((key * key).sum(-1, keepdim=True) + 1e-6)
    state = initial_state.float()
    scan_output = torch.empty_like(value)
    for token in range(2):
        state = state * log_decay[token].exp()[..., None, None]
        memory = (state * normalized_key[token].float().unsqueeze(-1)).sum(-2)
        delta = (value[token].float() - memory) * beta[token].float().unsqueeze(-1)
        state = state + normalized_key[token].float().unsqueeze(-1) * delta.unsqueeze(-2)
        scan_output[token] = (
            (state * normalized_query[token].float().unsqueeze(-1)).sum(-2)
            / key_width**0.5
        )
    return {
        "source": "Torch 2.11 CUDA BF16 Q/K normalization and F32 recurrent delta rule",
        "torch": torch.__version__, "seed": 319, "key_width": key_width,
        "value_width": value_width, "indptr": [0, 2], "query": bits(query),
        "key": bits(key), "value": bits(value),
        "log_decay_f32_bits": f32_bits(log_decay), "beta": bits(beta),
        "initial_state": bits(initial_state), "output": bits(scan_output),
        "final_state": bits(state.bfloat16()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", choices=("convolution", "delta_scan"))
    args = parser.parse_args()
    torch.manual_seed(319)
    fixture = convolution_fixture() if args.fixture == "convolution" else delta_scan_fixture()
    with Path(__file__).with_name(f"{args.fixture}.json").open("x") as output:
        json.dump(fixture, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
