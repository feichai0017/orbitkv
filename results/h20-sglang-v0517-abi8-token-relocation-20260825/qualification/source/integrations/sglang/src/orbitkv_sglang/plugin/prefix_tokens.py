from __future__ import annotations

from numbers import Integral
from typing import Any


def tokens_from_radix_key(key: Any, page_size: int) -> tuple[int, ...]:
    if bool(getattr(key, "is_bigram", False)):
        raise RuntimeError("OrbitKV does not support EAGLE/bigram prefix keys")
    if getattr(key, "extra_key", None) is not None:
        raise RuntimeError("OrbitKV does not support LoRA or namespaced prefix keys")
    try:
        values = tuple(key)
    except Exception as error:
        raise RuntimeError("SGLang prefix key is not readable") from error
    result = tuple(_token(value) for value in values)
    aligned = len(result) // page_size * page_size
    return result[:aligned]


def request_tokens(req: Any, boundary: int) -> tuple[int, ...]:
    if getattr(req, "extra_key", None) is not None:
        raise RuntimeError("OrbitKV does not support LoRA or namespaced requests")
    try:
        values = tuple(req.origin_input_ids) + tuple(req.output_ids)
    except Exception as error:
        raise RuntimeError("SGLang request token history is not readable") from error
    if boundary > len(values):
        raise RuntimeError("request KV boundary exceeds its token history")
    return tuple(_token(value) for value in values[:boundary])


def _token(value: Any) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or not 0 <= int(value) < 2**63
    ):
        raise RuntimeError("SGLang prefix tokens must be nonnegative int64 values")
    return int(value)
