"""Quantized state must not cross scale or dtype identities."""

import sys
from types import ModuleType, SimpleNamespace

from orbitkv.sglang.config import derive_namespace


def test_external_scale_contents_and_kv_dtype_isolate_cache(monkeypatch, tmp_path):
    runtime = ModuleType("sglang.srt.runtime_context")
    runtime.get_parallel = lambda: SimpleNamespace(tp_rank=0, tp_size=1)
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    monkeypatch.setattr("orbitkv.sglang.config.version", lambda _: "0.5.20")
    monkeypatch.setenv("ORBITKV_MODEL_FINGERPRINT", "a" * 64)
    scales = tmp_path / "scales.json"
    scales.write_text('{"k": 1.0, "v": 1.0}')
    args = SimpleNamespace(
        enable_lora=False,
        dtype="bfloat16",
        attention_backend="flashinfer",
        prefill_attention_backend=None,
        decode_attention_backend=None,
        model_path="unused-deployment-fingerprint",
        revision=None,
        tokenizer_path=None,
        kv_cache_dtype="fp8_e4m3",
        quantization_param_path=str(scales),
    )
    params = SimpleNamespace(pp_rank=0, pp_size=1, attn_cp_rank=0, attn_cp_size=1)
    layout = SimpleNamespace(page_size=64, pools={})
    original = derive_namespace(args, params, layout)
    assert derive_namespace(args, params, layout) == original
    # Same filename and size; the model override must not hide changed scales.
    scales.write_text('{"k": 2.0, "v": 1.0}')
    assert derive_namespace(args, params, layout) != original
    scales.write_text('{"k": 1.0, "v": 1.0}')
    args.kv_cache_dtype = "auto"
    assert derive_namespace(args, params, layout) != original


def test_static_adapter_namespace_consumes_content_uid_and_lora_backend(monkeypatch, tmp_path):
    runtime = ModuleType("sglang.srt.runtime_context")
    runtime.get_parallel = lambda: SimpleNamespace(tp_rank=0, tp_size=1)
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    monkeypatch.setattr("orbitkv.sglang.config.version", lambda _: "0.5.21")
    monkeypatch.setenv("ORBITKV_MODEL_FINGERPRINT", "a" * 64)
    monkeypatch.setenv("ORBITKV_STATIC_LORA", "1")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"r":8}')
    weights = adapter / "adapter_model.safetensors"
    weights.write_bytes(b"original")
    ref = SimpleNamespace(lora_name="fixed", lora_path=str(adapter), lora_id="native-uid")
    args = SimpleNamespace(
        enable_lora=True,
        lora_paths=[ref],
        lora_backend="triton",
        dtype="bfloat16",
        attention_backend="flashinfer",
        prefill_attention_backend=None,
        decode_attention_backend=None,
        model_path="deployment",
        revision=None,
        tokenizer_path=None,
    )
    params = SimpleNamespace(pp_rank=0, pp_size=1, attn_cp_rank=0, attn_cp_size=1)
    layout = SimpleNamespace(page_size=64, pools={})
    original = derive_namespace(args, params, layout)
    weights.write_bytes(b"replaced")
    assert derive_namespace(args, params, layout) != original
    weights.write_bytes(b"original")
    args.lora_backend = "torch_native"
    assert derive_namespace(args, params, layout) != original
    args.lora_backend = "triton"
    ref.lora_id = "different-native-uid"
    assert derive_namespace(args, params, layout) != original
