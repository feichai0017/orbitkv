from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from mla_helpers import load_mla_plan, write_mla_plan
from orbitkv_sglang.ffi import CtypesManagerFactory
from orbitkv_sglang.runtime import (
    ArenaRegistration,
    CanonicalRuntime,
    ManagerCreateSettings,
    RelocationCopyReceipt,
    RelocationPolicy,
)
from test_runtime_lifecycle import _policy_updates, _step_batch, ffi_library


def test_latent_manager_plan_preserves_component_geometry(
    tmp_path: Path, ffi_library: Path
) -> None:
    plan = tmp_path / "mla-plan.json"
    write_mla_plan(plan)
    config = load_mla_plan(plan, ffi_library)
    assert config.full_class is not None
    assert config.full_class.name == "latent_mla"
    assert config.full_class.storage == "latent_kv"
    assert config.full_class.components_by_name == {"latent": 1024, "rope": 128}

    write_mla_plan(plan, rope=64)
    value = plan.read_text().replace(
        '"bytes_per_token_per_layer": 1088',
        '"bytes_per_token_per_layer": 1152',
    )
    plan.write_text(value)
    with pytest.raises(ValueError, match="component bytes"):
        load_mla_plan(plan, ffi_library)


def test_runtime_mla_transaction_copies_real_latent_rows_and_drains(
    tmp_path: Path, ffi_library: Path
) -> None:
    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

    plan = tmp_path / "mla-runtime-plan.json"
    write_mla_plan(plan, latent=16, rope=8)
    config = load_mla_plan(plan, ffi_library)
    manager = CtypesManagerFactory().create(
        config,
        ManagerCreateSettings(1, 4, 1, 64, 64),
        (ArenaRegistration(0, 1, 1, 64, 0),),
    )
    runtime = CanonicalRuntime(config, manager)
    _step_batch(runtime, (("mla-request", 48),))
    pool = MLATokenToKVPool(
        size=1024,
        page_size=16,
        dtype=torch.bfloat16,
        kv_lora_rank=8,
        qk_rope_head_dim=4,
        layer_num=2,
        device="cpu",
        enable_memory_saver=False,
        start_layer=0,
        end_layer=2,
    )
    before = runtime.token_view("mla-request", 0)
    arena = runtime.arenas_by_class[0]
    for placement in before.placements:
        location = placement.location
        assert location is not None
        page = location.backend_index - arena.backend_base_index + 1
        slot = page * 16 + location.offset
        for layer, tensor in enumerate(pool.kv_buffer):
            tensor[slot].fill_(placement.token_id + layer * 100)

    def copied(prepared: Any) -> tuple[RelocationCopyReceipt, ...]:
        sources = []
        destinations = []
        for movement in prepared.moves:
            source_page = movement.source.backend_index - arena.backend_base_index + 1
            target_page = (
                movement.destination.backend_index - arena.backend_base_index + 1
            )
            sources.append(source_page * 16 + movement.source.offset)
            destinations.append(target_page * 16 + movement.destination.offset)
        pool.move_kv_cache(torch.tensor(destinations), torch.tensor(sources))
        for movement, destination in zip(prepared.moves, destinations, strict=True):
            for layer, tensor in enumerate(pool.kv_buffer):
                assert torch.all(tensor[destination] == movement.token_id + layer * 100)
        return tuple(
            RelocationCopyReceipt(
                prepared.relocation,
                movement.token_id,
                movement.source,
                movement.destination,
            )
            for movement in prepared.moves
        )

    output = runtime.relocate_tokens(
        "mla-request",
        0,
        _policy_updates(),
        RelocationPolicy(3, 2, 250, True),
        copied,
        13,
        1,
    )
    assert len(output.retained_locations) == 24
    runtime.acknowledge_relocation(output)
    _step_batch(runtime, (("mla-request", 49),), domain=14)
    runtime.release_batch(("mla-request",))
    assert runtime.stats().free_pages == 64
    runtime.close()


__all__ = ["ffi_library"]
