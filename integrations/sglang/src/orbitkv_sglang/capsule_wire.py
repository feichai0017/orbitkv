from __future__ import annotations

import io
import struct
from typing import Any

import torch


MAGIC = b"ORBITKVSGWIRE1\0"
_NONE = 0
_TENSOR = 1
_LIST = 2
_DICT = 3
_BOOL = 4

_DTYPE_TO_CODE = {
    torch.bool: 1,
    torch.uint8: 2,
    torch.int8: 3,
    torch.int16: 4,
    torch.int32: 5,
    torch.int64: 6,
    torch.float16: 7,
    torch.float32: 8,
    torch.float64: 9,
    torch.bfloat16: 10,
}
_CODE_TO_DTYPE = {code: dtype for dtype, code in _DTYPE_TO_CODE.items()}


def encode_cpu_tensors(value: Any) -> bytes:
    stream = io.BytesIO()
    stream.write(MAGIC)
    _encode_value(stream, value)
    return stream.getvalue()


def decode_cpu_tensors(payload: bytes) -> Any:
    stream = io.BytesIO(payload)
    if stream.read(len(MAGIC)) != MAGIC:
        raise ValueError("unsupported OrbitKV SGLang tensor wire payload")
    value = _decode_value(stream)
    if stream.read(1):
        raise ValueError("trailing bytes in OrbitKV SGLang tensor wire payload")
    return value


def _encode_value(stream: io.BytesIO, value: Any) -> None:
    if value is None:
        stream.write(bytes([_NONE]))
        return
    if isinstance(value, torch.Tensor):
        _encode_tensor(stream, value)
        return
    if isinstance(value, (list, tuple)):
        stream.write(bytes([_LIST]))
        stream.write(struct.pack("<Q", len(value)))
        for item in value:
            _encode_value(stream, item)
        return
    if isinstance(value, dict):
        stream.write(bytes([_DICT]))
        keys = sorted(value)
        if not all(isinstance(key, str) for key in keys):
            raise TypeError("OrbitKV tensor wire dict keys must be strings")
        stream.write(struct.pack("<Q", len(keys)))
        for key in keys:
            encoded = key.encode("utf-8")
            stream.write(struct.pack("<Q", len(encoded)))
            stream.write(encoded)
            _encode_value(stream, value[key])
        return
    if isinstance(value, bool):
        stream.write(bytes([_BOOL, int(value)]))
        return
    raise TypeError(f"unsupported OrbitKV tensor wire value: {type(value)!r}")


def _encode_tensor(stream: io.BytesIO, tensor: torch.Tensor) -> None:
    if tensor.device.type != "cpu":
        raise ValueError("OrbitKV tensor wire requires CPU tensors")
    tensor = tensor.detach().contiguous()
    dtype_code = _DTYPE_TO_CODE.get(tensor.dtype)
    if dtype_code is None:
        raise TypeError(f"unsupported OrbitKV tensor dtype: {tensor.dtype}")
    raw = tensor.view(torch.uint8).numpy().tobytes()
    stream.write(bytes([_TENSOR, dtype_code]))
    stream.write(struct.pack("<Q", tensor.ndim))
    for dimension in tensor.shape:
        stream.write(struct.pack("<Q", int(dimension)))
    stream.write(struct.pack("<Q", len(raw)))
    stream.write(raw)


def _decode_value(stream: io.BytesIO) -> Any:
    tag = _read_exact(stream, 1)[0]
    if tag == _NONE:
        return None
    if tag == _TENSOR:
        return _decode_tensor(stream)
    if tag == _LIST:
        return [_decode_value(stream) for _ in range(_read_u64(stream))]
    if tag == _DICT:
        result = {}
        for _ in range(_read_u64(stream)):
            key = _read_exact(stream, _read_u64(stream)).decode("utf-8")
            result[key] = _decode_value(stream)
        return result
    if tag == _BOOL:
        raw = _read_exact(stream, 1)[0]
        if raw not in (0, 1):
            raise ValueError("invalid boolean in OrbitKV tensor wire payload")
        return bool(raw)
    raise ValueError(f"unknown OrbitKV tensor wire tag: {tag}")


def _decode_tensor(stream: io.BytesIO) -> torch.Tensor:
    dtype_code = _read_exact(stream, 1)[0]
    dtype = _CODE_TO_DTYPE.get(dtype_code)
    if dtype is None:
        raise ValueError(f"unknown OrbitKV tensor dtype code: {dtype_code}")
    shape = tuple(_read_u64(stream) for _ in range(_read_u64(stream)))
    raw = bytearray(_read_exact(stream, _read_u64(stream)))
    tensor = torch.frombuffer(raw, dtype=dtype).clone()
    expected = 1
    for dimension in shape:
        expected *= dimension
    if tensor.numel() != expected:
        raise ValueError("OrbitKV tensor wire shape does not match payload bytes")
    return tensor.reshape(shape)


def _read_u64(stream: io.BytesIO) -> int:
    return struct.unpack("<Q", _read_exact(stream, 8))[0]


def _read_exact(stream: io.BytesIO, length: int) -> bytes:
    value = stream.read(length)
    if len(value) != length:
        raise ValueError("truncated OrbitKV SGLang tensor wire payload")
    return value
