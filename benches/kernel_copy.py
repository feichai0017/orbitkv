"""Validate and measure the consumed mapped-host kernel against one KDA candidate.

Run on an idle CUDA GPU. Output directories must be new and outside the checkout.
This driver measures copy mechanics; it cannot qualify inference latency or
promote a kernel into the production transfer backend.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import random
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

# Preserve the production descriptor ABI. Extra CTAs divide each fragment's
# vector and tail loops without sharing writable state or changing stream order.
CANDIDATE = r"""
extern "C" __global__ void transfer_blocks(const unsigned long long* __restrict__ desc,
                                           int n, int ctas) {
    const int tid = threadIdx.x;
    const long long work_count = (long long)n * ctas;
    for (long long work = blockIdx.x; work < work_count; work += gridDim.x) {
        const int i = work / ctas;
        const int shard = work % ctas;
        char* dst = (char*)desc[3 * i];
        const char* src = (const char*)desc[3 * i + 1];
        const unsigned long long size = desc[3 * i + 2];
        const unsigned long long start = (unsigned long long)shard * blockDim.x + tid;
        const unsigned long long stride = (unsigned long long)blockDim.x * ctas;
        if ((((unsigned long long)dst | (unsigned long long)src) & 15ULL) == 0) {
            int4* dst4 = (int4*)dst;
            const int4* src4 = (const int4*)src;
            const unsigned long long n4 = size / 16;
            for (unsigned long long j = start; j < n4; j += stride) {
                dst4[j] = src4[j];
            }
            for (unsigned long long j = n4 * 16 + start; j < size; j += stride) {
                dst[j] = src[j];
            }
        } else {
            for (unsigned long long j = start; j < size; j += stride) {
                dst[j] = src[j];
            }
        }
    }
}
"""


@dataclass(frozen=True)
class Case:
    name: str
    sizes: tuple[int, ...]
    device_offset: int = 16
    host_offset: int = 16
    reverse: bool = False

    @property
    def stride(self) -> int:
        return ((max(self.sizes, default=0) + 31) // 16 + 1) * 16

    @property
    def length(self) -> int:
        return max(self.device_offset, self.host_offset) + len(self.sizes) * self.stride + 32


def cases() -> list[Case]:
    return [
        Case("empty", ()),
        Case("zero", (0, 0, 0)),
        Case("scalar-tails", (1, 7, 15, 16, 17, 31, 4097, 65543)),
        Case("unaligned-both", (1, 17, 4097, 1048583), 17, 19),
        Case("unaligned-device", (17, 4097, 1048583), 17, 16),
        Case("unaligned-host", (17, 4097, 1048583), 16, 17),
        Case("mixed-reordered", (0, 17, 4096, 65543, 1048583), reverse=True),
        Case("grid-stride", (1,) * 65536, reverse=True),
        *[Case(f"large-{n}", (1048576,) * n) for n in (1, 2, 8, 32)],
        Case("one-4mib", (4194304,)),
        *[Case(f"fragmented-4k-{n}", (4096,) * n, reverse=True) for n in (1024, 4096)],
    ]


def ctas_per_fragment(case: Case, multiprocessors: int) -> int:
    if not case.sizes:
        return 1
    occupancy = max(1, multiprocessors * 4 // len(case.sizes))
    size_limit = max(1, (max(case.sizes) + 65535) // 65536)
    return min(16, occupancy, size_limit)


def checked(result: int, operation: str) -> None:
    if result:
        raise RuntimeError(f"{operation} failed with CUDA/NVRTC status {result}")


class CudaKernel:
    def __init__(self, source: str, architecture: str, library: str):
        self.driver = ctypes.CDLL("libcuda.so.1")
        self.nvrtc = ctypes.CDLL(library)
        self.module = ctypes.c_void_p()
        self.function = ctypes.c_void_p()
        functions = re.findall(r'extern "C" __global__ void (\w+)\(', source)
        if len(functions) != 1:
            raise ValueError("expected one global copy kernel")
        program = ctypes.c_void_p()
        checked(
            self.nvrtc.nvrtcCreateProgram(
                ctypes.byref(program), source.encode(), b"copy.cu", 0, None, None
            ),
            "nvrtcCreateProgram",
        )
        try:
            options = (ctypes.c_char_p * 2)(b"--std=c++17", architecture.encode())
            result = self.nvrtc.nvrtcCompileProgram(program, len(options), options)
            log_size = ctypes.c_size_t()
            checked(self.nvrtc.nvrtcGetProgramLogSize(program, ctypes.byref(log_size)), "log size")
            log = ctypes.create_string_buffer(log_size.value)
            checked(self.nvrtc.nvrtcGetProgramLog(program, log), "compile log")
            self.compile_log = log.value.decode()
            checked(result, f"nvrtcCompileProgram: {self.compile_log}")
            size = ctypes.c_size_t()
            checked(self.nvrtc.nvrtcGetPTXSize(program, ctypes.byref(size)), "PTX size")
            ptx = ctypes.create_string_buffer(size.value)
            checked(self.nvrtc.nvrtcGetPTX(program, ptx), "PTX")
            checked(self.driver.cuModuleLoadData(ctypes.byref(self.module), ptx), "module load")
            checked(
                self.driver.cuModuleGetFunction(
                    ctypes.byref(self.function), self.module, functions[0].encode()
                ),
                "function lookup",
            )
        finally:
            self.nvrtc.nvrtcDestroyProgram(ctypes.byref(program))

    def launch(self, descriptor, count: int, ctas: int, stream: int, candidate: bool) -> None:
        if not count:
            return
        values = [ctypes.c_uint64(descriptor.data_ptr()), ctypes.c_int(count)]
        if candidate:
            values.append(ctypes.c_int(ctas))
        arguments = (ctypes.c_void_p * len(values))(
            *(ctypes.cast(ctypes.byref(value), ctypes.c_void_p) for value in values)
        )
        checked(
            self.driver.cuLaunchKernel(
                self.function,
                ctypes.c_uint(min(65535, count * ctas)),
                ctypes.c_uint(1),
                ctypes.c_uint(1),
                ctypes.c_uint(256),
                ctypes.c_uint(1),
                ctypes.c_uint(1),
                ctypes.c_uint(0),
                ctypes.c_void_p(stream),
                arguments,
                None,
            ),
            "kernel launch",
        )

    def close(self) -> None:
        if self.module.value:
            checked(self.driver.cuModuleUnload(self.module), "module unload")
            self.module = ctypes.c_void_p()


def mapped_pointer(driver, tensor) -> int:
    pointer = ctypes.c_uint64()
    checked(
        driver.cuMemHostGetDevicePointer_v2(
            ctypes.byref(pointer), ctypes.c_void_p(tensor.data_ptr()), ctypes.c_uint(0)
        ),
        "mapped host pointer",
    )
    return pointer.value


def run_case(torch, case, direction, kernels, properties, warmup, samples, seed):
    host = torch.empty(case.length, dtype=torch.uint8, pin_memory=True)
    device = torch.empty(case.length, dtype=torch.uint8, device="cuda")
    mapped = mapped_pointer(kernels["baseline"].driver, host)
    stride = case.stride
    indices = range(len(case.sizes) - 1, -1, -1) if case.reverse else range(len(case.sizes))
    rows = []
    for i in indices:
        gpu_pointer = device.data_ptr() + case.device_offset + i * stride
        host_pointer = mapped + case.host_offset + i * stride
        dst, src = (
            (gpu_pointer, host_pointer) if direction == "h2d" else (host_pointer, gpu_pointer)
        )
        rows.append((dst, src, case.sizes[i]))
    cpu_descriptors = torch.tensor(rows, dtype=torch.int64).reshape(-1).pin_memory()
    gpu_descriptors = cpu_descriptors.to("cuda", non_blocking=True)
    stream = torch.cuda.current_stream()
    candidate_ctas = ctas_per_fragment(case, properties.multi_processor_count)
    positions = torch.arange(case.length, dtype=torch.int64)
    source = (positions * 31) ^ (positions >> 8) ^ ((positions >> 12) * 17) ^ (seed * 67)
    source = (source & 255).to(torch.uint8)
    expected = torch.full((case.length,), 0xA7, dtype=torch.uint8)
    for i, size in enumerate(case.sizes):
        src_offset = (case.host_offset if direction == "h2d" else case.device_offset) + i * stride
        dst_offset = (case.device_offset if direction == "h2d" else case.host_offset) + i * stride
        expected[dst_offset : dst_offset + size] = source[src_offset : src_offset + size]

    # Validate before timing; source and both sides' guard bytes are independent.
    for name, kernel in kernels.items():
        if direction == "h2d":
            host.copy_(source)
            device.fill_(0xA7)
        else:
            device.copy_(source)
            host.fill_(0xA7)
        kernel.launch(
            gpu_descriptors,
            len(rows),
            candidate_ctas if name == "candidate" else 1,
            stream.cuda_stream,
            name == "candidate",
        )
        stream.synchronize()
        actual = device.cpu() if direction == "h2d" else host
        if not torch.equal(actual, expected):
            first = int(torch.nonzero(actual != expected)[0])
            raise AssertionError(
                f"{case.name}/{direction}/{name}: destination or guard mismatch at {first}"
            )
        actual_source = host if direction == "h2d" else device.cpu()
        if not torch.equal(actual_source, source):
            raise AssertionError(f"{case.name}/{direction}/{name}: source overwritten")

    timing = {name: [] for name in kernels}
    for index in range(warmup + samples):
        order = ("baseline", "candidate") if index % 4 in (0, 3) else ("candidate", "baseline")
        for name in order:
            upload_start, kernel_start, end = [
                torch.cuda.Event(enable_timing=True) for _ in range(3)
            ]
            start = time.perf_counter_ns()
            upload_start.record(stream)
            gpu_descriptors.copy_(cpu_descriptors, non_blocking=True)
            kernel_start.record(stream)
            kernels[name].launch(
                gpu_descriptors,
                len(rows),
                candidate_ctas if name == "candidate" else 1,
                stream.cuda_stream,
                name == "candidate",
            )
            end.record(stream)
            end.synchronize()
            if index >= warmup:
                timing[name].append(
                    {
                        "ordinal": index - warmup,
                        "order": order.index(name),
                        "cuda_kernel_ms": kernel_start.elapsed_time(end),
                        "cuda_upload_and_kernel_ms": upload_start.elapsed_time(end),
                        "python_submit_to_drain_ms": (time.perf_counter_ns() - start) / 1e6,
                    }
                )
    return {
        "case": case.name,
        "direction": direction,
        "descriptors": len(rows),
        "bytes": sum(case.sizes),
        "candidate_ctas": candidate_ctas,
        "correctness": "pass",
        "samples": timing,
        "median_upload_and_kernel_ratio": (
            statistics.median(s["cuda_upload_and_kernel_ms"] for s in timing["candidate"])
            / statistics.median(s["cuda_upload_and_kernel_ms"] for s in timing["baseline"])
        )
        if samples
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nvrtc", default="/usr/local/cuda/lib64/libnvrtc.so")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    output = args.output.resolve()
    if output.is_relative_to(source_root):
        parser.error("kernel evidence must be outside the source checkout")
    output.mkdir(parents=True, exist_ok=False)
    import torch

    torch.set_num_threads(args.cpu_threads)
    torch.cuda.init()
    # Materialize the runtime primary context before loading driver modules.
    _context_guard = torch.empty(1, dtype=torch.uint8, device="cuda")
    properties = torch.cuda.get_device_properties(0)
    source_path = source_root / "crates/orbitkv-core/src/transfer/kernel.rs"
    rust_source = source_path.read_text()
    match = re.search(r'const KERNEL_SRC: &str = r#"(.*?)"#;', rust_source, flags=re.DOTALL)
    if match is None:
        raise ValueError("cannot extract the consumed NVRTC baseline")
    baseline = match[1]
    architecture = f"--gpu-architecture=compute_{properties.major}{properties.minor}"
    inputs = {
        "gpu": properties.name,
        "compute": [properties.major, properties.minor],
        "multiprocessors": properties.multi_processor_count,
        "torch": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
        "cuda": torch.version.cuda,
        "nvrtc_library": str(Path(args.nvrtc).resolve()),
        "source": str(source_path),
        "source_sha256": hashlib.sha256(rust_source.encode()).hexdigest(),
        "baseline_sha256": hashlib.sha256(baseline.encode()).hexdigest(),
        "candidate_sha256": hashlib.sha256(CANDIDATE.encode()).hexdigest(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": args.seed,
        "warmup": 0 if args.validate_only else 5,
        "measured": 0 if args.validate_only else 30,
        "scope": "copy mechanics; Python wall time is not native restore latency",
        "promotion": "requires independent reproduction and consumed application gates",
    }
    (output / "inputs.json").write_text(json.dumps(inputs, indent=2) + "\n")
    kernels = {}
    completed = []
    try:
        for name, source in (("baseline", baseline), ("candidate", CANDIDATE)):
            kernels[name] = CudaKernel(source, architecture, args.nvrtc)
            (output / f"{name}.compile.log").write_text(kernels[name].compile_log)
        selected = cases()
        random.Random(args.seed).shuffle(selected)
        for case in selected:
            for direction in ("h2d", "d2h"):
                result = run_case(
                    torch,
                    case,
                    direction,
                    kernels,
                    properties,
                    inputs["warmup"],
                    inputs["measured"],
                    args.seed,
                )
                completed.append(result)
                (output / "results.json").write_text(json.dumps(completed, indent=2) + "\n")
                print(json.dumps({k: v for k, v in result.items() if k != "samples"}), flush=True)
        (output / "status.json").write_text(
            json.dumps(
                {
                    "status": "MECHANICS_PASSED",
                    "cells": len(completed),
                    "production_promoted": False,
                }
            )
            + "\n"
        )
    except Exception as error:
        (output / "status.json").write_text(
            json.dumps({"status": "FAILED", "cells": len(completed), "error": repr(error)}) + "\n"
        )
        raise
    finally:
        torch.cuda.synchronize()
        for kernel in kernels.values():
            kernel.close()


if __name__ == "__main__":
    main()
