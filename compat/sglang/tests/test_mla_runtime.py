from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest
import torch

from mla_helpers import load_mla_plan, write_mla_plan
from orbitkv_sglang.execution_plan import (
    BindingResult,
    StepExecutionResult,
    confirm_execution,
    expected_bindings,
    lower_batch_plan,
)
from orbitkv_sglang.ffi import CtypesRuntimeSession
from orbitkv_sglang.ffi.session_types import EnginePrefixId, EnginePrefixLookup
from orbitkv_sglang.runtime import (
    ArenaRegistration,
    CacheSharingPolicy,
    ManagerCreateSettings,
    ManagerError,
    PrefixSemanticKey,
    SessionCreateSettings,
)
from orbitkv_sglang.session_runtime import SessionRuntime
from ffi_test_support import ffi_library


PAGE_TOKENS = 16
PAGE_COUNT = 8
LATENT_ELEMENTS = 8
ROPE_ELEMENTS = 4


class _ReadyEvent:
    @staticmethod
    def query() -> bool:
        return True

    @staticmethod
    def synchronize() -> None:
        return None


def _step_locations(step: Any) -> tuple[int, ...]:
    spec = step.by_class[0]
    previous = int(spec.previous_layout_boundary)
    target = int(spec.target_layout_boundary)
    remaining = target - previous
    locations: list[int] = []
    if previous % PAGE_TOKENS:
        count = min(remaining, PAGE_TOKENS - previous % PAGE_TOKENS)
        locations.extend(
            range(
                int(spec.last_location) + 1,
                int(spec.last_location) + 1 + count,
            )
        )
        remaining -= count
    for page_id in spec.exact_new_pages:
        count = min(remaining, PAGE_TOKENS)
        locations.extend(
            range(
                int(page_id) * PAGE_TOKENS,
                int(page_id) * PAGE_TOKENS + count,
            )
        )
        remaining -= count
    assert remaining == 0
    return tuple(locations)


def _stage_same_prompt_rows(
    pool: Any, source_rows: dict[str, tuple[int, ...]], token_begin: int
) -> None:
    for rows in source_rows.values():
        for token_offset, row in enumerate(rows):
            token = token_begin + token_offset
            for layer, tensor in enumerate(pool.kv_buffer):
                latent = (
                    torch.arange(LATENT_ELEMENTS, dtype=torch.bfloat16)
                    + token
                    + layer * 32
                )
                rope = -(
                    torch.arange(ROPE_ELEMENTS, dtype=torch.bfloat16)
                    + token
                    + layer * 32
                    + 1
                )
                tensor[row].copy_(torch.cat((latent, rope)).reshape(1, -1))

    left, right = tuple(source_rows.values())
    for left_row, right_row in zip(left, right, strict=True):
        for tensor in pool.kv_buffer:
            assert torch.equal(tensor[left_row], tensor[right_row])


def _append_real_mla_rows(
    runtime: SessionRuntime,
    config: Any,
    pool: Any,
    appends: tuple[tuple[str, int], ...],
    source_rows: dict[str, tuple[int, ...]],
    completion_domain: int,
) -> tuple[Any, dict[str, tuple[int, ...]]]:
    plan = runtime.prepare(appends)
    keys_by_request = {
        runtime.binding_for(key).request_id: key for key, _target in appends
    }
    lowered = lower_batch_plan(
        plan,
        config,
        runtime.arenas,
        {
            request_id: runtime.view_for(key)
            for request_id, key in keys_by_request.items()
        },
    )
    bindings = expected_bindings(plan, lowered, runtime.arenas)
    locations_by_key: dict[str, tuple[int, ...]] = {}
    results = []
    for lowered_step, selected in zip(lowered.steps, bindings, strict=True):
        key = keys_by_request[lowered_step.request_id]
        destinations = _step_locations(lowered_step)
        sources = source_rows[key]
        assert len(destinations) == len(sources)
        assert all(
            0 <= location < pool.size + pool.page_size
            for location in destinations
        )
        assert set(destinations).isdisjoint(sources)

        copies = tuple(
            intent
            for class_spec in lowered_step.class_specs
            for intent in class_spec.copy_intents
        )
        assert copies == ()
        pool.move_kv_cache(
            torch.tensor(destinations, dtype=torch.int64, device="cpu"),
            torch.tensor(sources, dtype=torch.int64, device="cpu"),
        )
        for destination, source in zip(destinations, sources, strict=True):
            for tensor in pool.kv_buffer:
                assert torch.equal(
                    tensor[destination, :, :LATENT_ELEMENTS],
                    tensor[source, :, :LATENT_ELEMENTS],
                )
                assert torch.equal(
                    tensor[destination, :, LATENT_ELEMENTS:],
                    tensor[source, :, LATENT_ELEMENTS:],
                )
        locations_by_key[key] = destinations
        results.append(
            StepExecutionResult(
                lowered_step.request_id,
                tuple(
                    BindingResult(
                        item.page,
                        item.backend_domain,
                        item.backend_index,
                        True,
                        True,
                    )
                    for item in selected
                ),
                copies,
                True,
            )
        )

    evidence = confirm_execution(
        plan, lowered, runtime.arenas, tuple(results)
    )
    ticket = runtime.submit(evidence)
    runtime.register_event(ticket, _ReadyEvent(), completion_domain)
    publications = runtime.poll()
    assert len(publications) == 1
    return publications[0], locations_by_key


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


def test_native_request_private_mla_rows_are_independent_and_drain(
    tmp_path: Path, ffi_library: Path
) -> None:
    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

    plan_path = tmp_path / "mla-native-runtime-plan.json"
    write_mla_plan(plan_path, latent=16, rope=8)
    config = load_mla_plan(plan_path, ffi_library)
    pool = MLATokenToKVPool(
        size=256,
        page_size=PAGE_TOKENS,
        dtype=torch.bfloat16,
        kv_lora_rank=LATENT_ELEMENTS,
        qk_rope_head_dim=ROPE_ELEMENTS,
        layer_num=2,
        device="cpu",
        enable_memory_saver=False,
        start_layer=0,
        end_layer=2,
    )
    settings = SessionCreateSettings(
        manager=ManagerCreateSettings(
            maximum_requests=2,
            maximum_operations=2,
            maximum_prefixes=2,
            maximum_reclamations=PAGE_COUNT,
            maximum_step_tokens=64,
        ),
        cache_sharing_policy=CacheSharingPolicy.REQUEST_PRIVATE,
    )
    native = CtypesRuntimeSession.create(
        config,
        settings,
        (ArenaRegistration(0, 1, 1, PAGE_COUNT, 0),),
    )
    assert native.cache_sharing_policy is CacheSharingPolicy.REQUEST_PRIVATE

    managed_rows: dict[str, set[int]] = {"request-a": set(), "request-b": set()}
    release_cleanup_calls = 0

    def cleanup(updates: tuple[Any, ...], retirements: tuple[Any, ...]) -> bool:
        nonlocal release_cleanup_calls
        if not any(update.releasing for update in updates):
            assert all(not update.detached for update in updates)
            assert retirements == ()
            return True

        release_cleanup_calls += 1
        assert {update.key for update in updates} == set(managed_rows)
        assert all(update.releasing and update.boundary == 18 for update in updates)
        arena = runtime.arenas_by_class[0]
        cleared: set[int] = set()
        for update in updates:
            request_rows: set[int] = set()
            for detached in update.detached:
                assert detached.class_id == 0
                page = detached.old_backend_index - arena.backend_base_index + 1
                offset = (
                    detached.token_begin
                    - detached.logical_ordinal * PAGE_TOKENS
                )
                count = detached.token_end_exclusive - detached.token_begin
                request_rows.update(
                    range(
                        page * PAGE_TOKENS + offset,
                        page * PAGE_TOKENS + offset + count,
                    )
                )
            assert request_rows == managed_rows[update.key]
            cleared.update(request_rows)
        assert len(retirements) == 4
        cleared_indices = torch.tensor(sorted(cleared), dtype=torch.int64)
        for tensor in pool.kv_buffer:
            tensor.index_fill_(0, cleared_indices, 0)
        return True

    runtime = SessionRuntime(native, mirror_cleanup=cleanup)
    assert runtime.cache_sharing_policy is CacheSharingPolicy.REQUEST_PRIVATE
    drained = False
    try:
        request_keys = ("request-a", "request-b")
        runtime.acquire_unbound(request_keys)
        runtime.bind_request_rows(((request_keys[0], 1), (request_keys[1], 2)))

        prefill_sources = {
            request_keys[0]: tuple(range(160, 177)),
            request_keys[1]: tuple(range(192, 209)),
        }
        _stage_same_prompt_rows(pool, prefill_sources, 0)
        prefill, prefill_locations = _append_real_mla_rows(
            runtime,
            config,
            pool,
            ((request_keys[0], 17), (request_keys[1], 17)),
            prefill_sources,
            11,
        )
        assert tuple(step.boundary for step in prefill.steps) == (17, 17)
        assert tuple(step.resident_count for step in prefill.steps) == (2, 2)
        assert set(prefill_locations[request_keys[0]]).isdisjoint(
            prefill_locations[request_keys[1]]
        )
        for key in request_keys:
            managed_rows[key].update(prefill_locations[key])

        decode_sources = {request_keys[0]: (224,), request_keys[1]: (225,)}
        _stage_same_prompt_rows(pool, decode_sources, 17)
        decode, decode_locations = _append_real_mla_rows(
            runtime,
            config,
            pool,
            ((request_keys[0], 18), (request_keys[1], 18)),
            decode_sources,
            12,
        )
        assert tuple(step.boundary for step in decode.steps) == (18, 18)
        assert tuple(step.resident_count for step in decode.steps) == (2, 2)
        assert decode_locations[request_keys[0]] != decode_locations[request_keys[1]]
        for key in request_keys:
            managed_rows[key].update(decode_locations[key])
            assert len(managed_rows[key]) == 18
        assert managed_rows[request_keys[0]].isdisjoint(
            managed_rows[request_keys[1]]
        )

        stats = runtime.stats()
        assert stats.active_requests == 2
        assert stats.active_prefixes == 0
        assert stats.total_request_page_refs == 4
        assert stats.total_prefix_page_refs == 0

        semantic = PrefixSemanticKey(b"m" * 32, b"p" * 32, 18)
        fake_prefix = EnginePrefixId(runtime.engine_epoch, 1)
        fake_lookup = EnginePrefixLookup(semantic, fake_prefix, 2)
        prefix_operations: tuple[tuple[str, Callable[[], object]], ...] = (
            ("lookup", lambda: runtime.prefix_lookup((semantic,))),
            ("publish", lambda: runtime.prefix_publish(((request_keys[0], semantic),))),
            (
                "attach",
                lambda: runtime.prepare_prefix_attach(
                    ((request_keys[1], fake_lookup),)
                ),
            ),
            (
                "request fork",
                lambda: runtime.prepare_request_fork(
                    ((request_keys[0], request_keys[1]),)
                ),
            ),
            ("evict", lambda: runtime.prepare_prefix_evict((fake_prefix,))),
            (
                "publish-release",
                lambda: runtime.prepare_prefix_publish_release(
                    ((request_keys[0], semantic),)
                ),
            ),
        )
        before_prefix_probes = runtime.stats()
        for operation, invoke in prefix_operations:
            with pytest.raises(
                ManagerError, match="request-private cache sharing policy"
            ):
                invoke()
            assert runtime.stats() == before_prefix_probes, operation

        release = runtime.prepare_release(request_keys)
        assert len(release.releases) == 2
        assert len(release.retirements) == 4
        runtime.confirm_release(release)
        stats = runtime.stats()
        assert stats.active_requests == 0
        assert stats.active_snapshots == 0
        assert stats.active_prefixes == 0
        assert stats.prepared_steps == 0
        assert stats.submitted_steps == 0
        assert stats.reserved_pages == 0
        assert stats.writing_pages == 0
        assert stats.active_pages == 0
        assert stats.retiring_pages == 0
        assert stats.pending_reclamations == 0
        assert stats.total_request_page_refs == 0
        assert stats.total_prefix_page_refs == 0
        assert stats.free_pages == PAGE_COUNT
        assert release_cleanup_calls == 1
        for rows in managed_rows.values():
            for tensor in pool.kv_buffer:
                assert not torch.count_nonzero(
                    tensor[torch.tensor(sorted(rows), dtype=torch.int64)]
                )
        drained = True
    finally:
        if not drained and runtime.failure_reason is None:
            runtime.fail_stop("test cleanup after incomplete MLA lifecycle")
        else:
            runtime.close()


__all__ = ["ffi_library"]
