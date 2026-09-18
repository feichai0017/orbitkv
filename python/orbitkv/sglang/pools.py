from __future__ import annotations

_POOL_COMPONENTS = {
    "kv": "attention_kv",
    "mamba": "recurrent_checkpoint",
    "swa": "sliding_window_kv",
    "draft": "draft_kv",
    "draft_swa": "sliding_window_kv",
    "indexer": "indexer_state",
    "draft_indexer": "indexer_state",
    "deepseek_v4_c4": "mla_kv",
    "deepseek_v4_c128": "mla_kv",
    "deepseek_v4_c4_state": "recurrent_checkpoint",
    "deepseek_v4_c128_state": "recurrent_checkpoint",
}


def component_for_pool(pool: object) -> str:
    """Map an SGLang PoolName-like value to an OrbitKV component name.

    The adapter accepts strings and enum-like objects so this module remains
    importable without installing SGLang. Unknown pools remain explicit and
    namespace-scoped rather than being silently treated as attention KV.
    """

    value = getattr(pool, "value", pool)
    name = str(value)
    return _POOL_COMPONENTS.get(name, f"opaque:{name}")
