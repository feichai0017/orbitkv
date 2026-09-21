"""Register and construct the OrbitKV SGLang RadixCache backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache


def register() -> None:
    from sglang.srt.mem_cache.registry import register_radix_cache_backend
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    from .admission import abort_request, admit_request, enqueue_request

    register_radix_cache_backend("orbitkv", create_cache)
    HookRegistry.register(
        "sglang.srt.managers.schedule_policy.PrefillAdder.add_one_req",
        admit_request,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.managers.scheduler.Scheduler._add_request_to_queue",
        enqueue_request,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.managers.scheduler.Scheduler._release_aborted_request",
        abort_request,
        HookType.AROUND,
    )


def create_cache(ctx: Any) -> UnifiedRadixCache:
    """Factory selected by SGLang's ``--radix-cache-backend orbitkv``."""
    from sglang.srt.mem_cache.unified_cache.components import ComponentType
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    from .linker import OrbitKVLinker

    if ctx.disable_radix_cache:
        raise ValueError("OrbitKV direct GPU linker requires RadixCache")
    if ctx.enable_hierarchical_cache:
        raise ValueError("OrbitKV direct GPU linker does not support hierarchical cache")
    if ctx.is_hybrid_swa or ctx.is_hybrid_ssm:
        raise ValueError("OrbitKV direct GPU linker currently supports full-attention KV only")
    if (
        ctx.is_dsa
        or ctx.params.is_eagle
        or ctx.params.mtp_draft_device_pools
        or ctx.params.component_registry_override
        or hasattr(ctx.params.req_to_token_pool, "req_to_c128_sidecar")
    ):
        raise ValueError(
            "OrbitKV direct GPU linker cannot restore DSA, draft, or auxiliary GPU state; "
            "select a backend with a complete recovery contract for that model"
        )
    from sglang.srt.runtime_context import get_disagg, get_memory

    if not get_memory().enable_unified_cache_external_linker:
        raise ValueError(
            "OrbitKV direct GPU linker requires --enable-unified-cache-external-linker "
            "so SGLang schedules GPU KV restores"
        )
    if get_disagg().disaggregation_decode_retraction_backup == "host_pool":
        raise ValueError("OrbitKV direct GPU linker does not support host-pool retraction")

    # SGLang's built-in unified-cache factory hardcodes Mooncake/Mori when the
    # external-linker flag is set. Construct its public RadixCache component
    # directly and attach OrbitKV through the public linker interface.
    ctx.params.tree_components = (ComponentType.FULL,)
    cache = UnifiedRadixCache(ctx.params)
    linker = OrbitKVLinker(ctx.server_args, ctx.params, components=set(cache.components))
    try:
        cache.init_cache_linker(linker)
    except Exception:
        linker.close()
        raise
    counter = linker.layer_done_counter
    kvcache = ctx.params.token_to_kv_pool_allocator.get_kvcache()
    kvcache.register_layer_transfer_counter(counter)
    ctx.tp_worker.register_hicache_layer_transfer_counter(counter)
    return cache
