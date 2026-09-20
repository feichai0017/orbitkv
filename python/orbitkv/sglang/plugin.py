"""Register OrbitKV as an SGLang RadixCache backend in every SGLang process."""


def register() -> None:
    from sglang.srt.mem_cache.registry import register_radix_cache_backend

    from orbitkv.sglang.linker import create_cache

    register_radix_cache_backend("orbitkv", create_cache)
