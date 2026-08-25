from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

from orbitkv_runtime import (
    ArenaRegistration as NeutralArenaRegistration,
    BackendPageAddress,
    DataPlaneOperation,
    ExternalTokenWrite,
)
from orbitkv_sglang.config import ClassConfig
from orbitkv_sglang.plugin.structured_arena import (
    ExternalAppendManifest,
    SglangArenaComponent,
    build_external_append_manifest,
    build_sglang_structured_arenas,
)
from orbitkv_sglang.runtime import (
    ArenaIdentity,
    PageLease,
    PageShadow,
    RequestLease,
    StepLease,
)


PAGE_TOKENS = 16
DTYPE = object()


class _FakeTensor:
    _next_pointer = 4096

    def __init__(
        self,
        shape,
        *,
        dtype=DTYPE,
        device="cuda:0",
        element_size=2,
        contiguous=True,
        pointer=None,
    ):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self.requires_grad = False
        self._element_size = element_size
        self._contiguous = contiguous
        if pointer is None:
            pointer = type(self)._next_pointer
            type(self)._next_pointer += self.numel() * element_size + 4096
        self._pointer = pointer

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def nbytes(self):
        return self.numel() * self.element_size()

    def numel(self):
        result = 1
        for item in self.shape:
            result *= item
        return result

    def element_size(self):
        return self._element_size

    def is_contiguous(self):
        return self._contiguous

    def data_ptr(self):
        return self._pointer


class _FakeLocations:
    def __init__(self, values):
        self._values = tuple(values)
        self.ndim = 1

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self._values)


def _class(
    class_id,
    retention,
    layers,
    *,
    key_bytes=16,
    value_bytes=24,
    storage="token_kv",
):
    return ClassConfig(
        class_id=class_id,
        pool_id=class_id + 1,
        backend_domain=class_id + 11,
        name="full" if retention == "full" else "swa",
        layers=tuple(layers),
        retention=retention,
        bytes_per_token_per_layer=key_bytes + value_bytes,
        window_tokens=None if retention == "full" else 32,
        period_blocks=None if retention == "full" else 3,
        storage=storage,
        components=(("key", key_bytes), ("value", value_bytes)),
    )


def _config(*classes):
    return SimpleNamespace(page_tokens=PAGE_TOKENS, classes=tuple(classes))


def _identity(class_config, *, pages, base, first, engine_epoch=7):
    return ArenaIdentity(
        engine_epoch=engine_epoch,
        pool_epoch=19 + class_config.class_id,
        pool_id=class_config.pool_id,
        class_id=class_config.class_id,
        backend_domain=class_config.backend_domain,
        page_count=pages,
        page_tokens=PAGE_TOKENS,
        backend_base_index=base,
        first_page_id=first,
    )


def _pool(class_config, *, size, key_dim=4, value_dim=6, **overrides):
    rows = size + PAGE_TOKENS
    values = dict(
        size=size,
        page_size=PAGE_TOKENS,
        dtype=DTYPE,
        store_dtype=DTYPE,
        kv_cache_layout="nhd",
        use_hnd=False,
        use_mla=False,
        dsa_kv_cache_store_fp8=False,
        is_quantized_kv_cache=False,
        quant_method=SimpleNamespace(name="unquantized"),
        k_scale_buffer=None,
        v_scale_buffer=None,
        layer_num=len(class_config.layers),
        head_num=2,
        head_dim=key_dim,
        v_head_dim=value_dim,
        k_buffer=[
            _FakeTensor((rows, 2, key_dim)) for _ in class_config.layers
        ],
        v_buffer=[
            _FakeTensor((rows, 2, value_dim)) for _ in class_config.layers
        ],
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class _Runtime:
    def __init__(self, identities):
        self.arenas = tuple(identities)
        self.arenas_by_class = {item.class_id: item for item in identities}
        self._records = {}

    def record_for(self, key):
        return self._records[key]


def _structured_fixture(*, hybrid=False):
    full = _class(0, "full", (0, 2))
    full_identity = _identity(full, pages=4, base=7, first=101)
    if not hybrid:
        config = _config(full)
        runtime = _Runtime((full_identity,))
        pool = _pool(full, size=64)
        return config, runtime, pool

    sliding = _class(1, "sliding", (1, 3), key_bytes=8, value_bytes=8)
    sliding_identity = _identity(sliding, pages=3, base=21, first=105)
    config = _config(full, sliding)
    runtime = _Runtime((full_identity, sliding_identity))
    pool = SimpleNamespace(
        full_kv_pool=_pool(full, size=64),
        swa_kv_pool=_pool(
            sliding, size=48, key_dim=2, value_dim=2
        ),
    )
    return config, runtime, pool


def _page(request, identity, ordinal, page_delta):
    lease = PageLease(
        identity.engine_epoch,
        identity.pool_epoch,
        3 + page_delta,
        identity.first_page_id + page_delta,
        identity.pool_id,
    )
    return PageShadow(
        request=request,
        class_id=identity.class_id,
        logical_ordinal=ordinal,
        page=lease,
        backend_index=identity.backend_base_index + page_delta,
    )


def _manifest_inputs(
    config,
    runtime,
    *,
    previous,
    target,
    physical_previous=None,
):
    request = RequestLease(runtime.arenas[0].engine_epoch, 2, 5)
    step = StepLease(runtime.arenas[0].engine_epoch, 4, 6)
    physical_previous = previous if physical_previous is None else physical_previous
    append_count = target - previous
    current = {}
    new_pages = []
    specs = []
    locations = {}
    for class_config, identity in zip(config.classes, runtime.arenas, strict=True):
        begin_page, begin_offset = divmod(physical_previous, PAGE_TOKENS)
        physical_target = physical_previous + append_count
        final_page = (physical_target - 1) // PAGE_TOKENS
        if begin_offset:
            current[(class_config.class_id, begin_page)] = _page(
                request, identity, begin_page, begin_page
            )
            first_new_page = begin_page + 1
        else:
            first_new_page = begin_page
        for ordinal in range(first_new_page, final_page + 1):
            new_pages.append(_page(request, identity, ordinal, ordinal))
        tail = current.get((class_config.class_id, begin_page))
        if tail is None and first_new_page <= final_page:
            tail = next(
                item
                for item in new_pages
                if item.class_id == class_config.class_id
                and item.logical_ordinal == begin_page
            )
        specs.append(
            SimpleNamespace(
                class_id=class_config.class_id,
                pool_id=class_config.pool_id,
                previous_layout_boundary=physical_previous,
                target_layout_boundary=physical_target,
                tail_action=SimpleNamespace(
                    destination=None if tail is None else tail.page
                ),
            )
        )
        locations[class_config.class_id] = _FakeLocations(
            range(
                PAGE_TOKENS + physical_previous,
                PAGE_TOKENS + physical_target,
            )
        )

    prepared = SimpleNamespace(
        request=request,
        step=step,
        previous_boundary=previous,
        target_boundary=target,
    )
    record = SimpleNamespace(
        key="request-a", prepared=prepared, new_pages=tuple(new_pages)
    )
    batch_record = SimpleNamespace(keys=("request-a",), records=(record,))
    plan = SimpleNamespace(
        request=request,
        previous_boundary=previous,
        target_boundary=target,
        class_specs=tuple(specs),
    )
    cursor = SimpleNamespace(lease=request, pages=current)
    runtime._records["request-a"] = SimpleNamespace(
        cursor=cursor, pending=record
    )
    return batch_record, (plan,), locations


def test_full_structured_arena_describes_every_layer_component():
    config, runtime, pool = _structured_fixture()

    arenas = build_sglang_structured_arenas(config, runtime, pool)

    assert len(arenas) == 1
    arena = arenas[0]
    assert arena.registration == NeutralArenaRegistration(
        7, 19, 1, 0, 11, 4, 16, 80, 7, 101
    )
    assert (arena.retention, arena.storage, arena.storage_token_bias) == (
        "full",
        "token_kv",
        16,
    )
    assert [
        (
            item.global_layer_id,
            item.local_layer_id,
            item.name,
            item.token_byte_offset,
            item.token_byte_count,
        )
        for item in arena.components
    ] == [
        (0, 0, "key", 0, 16),
        (0, 0, "value", 16, 24),
        (2, 1, "key", 40, 16),
        (2, 1, "value", 56, 24),
    ]
    assert [item.tensor for item in arena.components] == [
        pool.k_buffer[0],
        pool.v_buffer[0],
        pool.k_buffer[1],
        pool.v_buffer[1],
    ]


def test_full_swa_arenas_bind_distinct_leaf_pools_and_widths():
    config, runtime, pool = _structured_fixture(hybrid=True)

    full, sliding = build_sglang_structured_arenas(config, runtime, pool)

    assert (full.class_id, full.registration.page_count, full.token_bytes) == (
        0,
        4,
        80,
    )
    assert (sliding.class_id, sliding.registration.page_count, sliding.token_bytes) == (
        1,
        3,
        32,
    )
    assert sliding.retention == "sliding"
    assert full.components[0].tensor is pool.full_kv_pool.k_buffer[0]
    assert sliding.components[0].tensor is pool.swa_kv_pool.k_buffer[0]


def test_external_manifest_maps_multi_page_full_append_without_dummy_bias():
    config, runtime, pool = _structured_fixture()
    arenas = build_sglang_structured_arenas(config, runtime, pool)
    batch_record, plans, locations = _manifest_inputs(
        config, runtime, previous=14, target=35
    )

    manifest = build_external_append_manifest(
        batch_record, plans, locations, runtime, config, arenas
    )

    assert isinstance(manifest, ExternalAppendManifest)
    assert len(manifest.writes) == 21
    assert tuple(item.token_id for item in manifest.writes) == tuple(range(14, 35))
    assert tuple(
        (item.destination.backend_index, item.destination.token_offset)
        for item in manifest.writes
    ) == (
        (7, 14),
        (7, 15),
        *((8, offset) for offset in range(16)),
        (9, 0),
        (9, 1),
        (9, 2),
    )
    assert all(item.byte_count == 80 for item in manifest.writes)
    assert all(
        item.context.operation is DataPlaneOperation.APPEND
        for item in manifest.writes
    )
    assert manifest.expected_locations_by_class == ((0, tuple(range(30, 51))),)
    assert tuple(
        (item.backend_index, item.page.page_id) for item in manifest.last_use_pages
    ) == ((7, 101), (8, 102), (9, 103))


def test_external_manifest_orders_request_token_then_full_and_swa_class():
    config, runtime, pool = _structured_fixture(hybrid=True)
    arenas = build_sglang_structured_arenas(config, runtime, pool)
    batch_record, plans, locations = _manifest_inputs(
        config, runtime, previous=0, target=2
    )

    manifest = build_external_append_manifest(
        batch_record, plans, locations, runtime, config, arenas
    )

    assert [
        (item.token_id, item.destination.class_id, item.byte_count)
        for item in manifest.writes
    ] == [(0, 0, 80), (0, 1, 32), (1, 0, 80), (1, 1, 32)]
    assert manifest.expected_locations_by_class == (
        (0, (16, 17)),
        (1, (16, 17)),
    )
    assert len(manifest.last_use_pages) == 2
    assert {item.class_id for item in manifest.last_use_pages} == {0, 1}


def test_external_manifest_keeps_logical_ids_separate_from_packed_positions():
    config, runtime, pool = _structured_fixture()
    arenas = build_sglang_structured_arenas(config, runtime, pool)
    batch_record, plans, locations = _manifest_inputs(
        config, runtime, previous=100, target=103, physical_previous=15
    )

    manifest = build_external_append_manifest(
        batch_record, plans, locations, runtime, config, arenas
    )

    assert tuple(item.token_id for item in manifest.writes) == (100, 101, 102)
    assert tuple(
        (item.destination.backend_index, item.destination.token_offset)
        for item in manifest.writes
    ) == ((7, 15), (8, 0), (8, 1))
    assert manifest.expected_locations_by_class == ((0, (31, 32, 33)),)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda pool: setattr(pool, "kv_cache_layout", "hnd"), "not NHD"),
        (lambda pool: setattr(pool, "store_dtype", object()), "packed or quantized"),
        (
            lambda pool: pool.k_buffer.__setitem__(
                0, _FakeTensor((79, 2, 4))
            ),
            "not exact NHD",
        ),
        (
            lambda pool: pool.v_buffer.__setitem__(
                0,
                _FakeTensor(
                    (80, 2, 6), pointer=pool.k_buffer[0].data_ptr()
                ),
            ),
            "component buffers alias",
        ),
    ),
)
def test_structured_arena_rejects_unsupported_or_drifted_geometry(
    mutation, message
):
    config, runtime, pool = _structured_fixture()
    mutation(pool)

    with pytest.raises(RuntimeError, match=message):
        build_sglang_structured_arenas(config, runtime, pool)


def test_structured_arena_rejects_mla_and_pure_swa_profiles():
    config, runtime, pool = _structured_fixture()
    latent = _class(0, "full", (0,), storage="latent_kv")
    latent_config = _config(latent)
    latent_runtime = _Runtime((_identity(latent, pages=4, base=7, first=101),))
    with pytest.raises(RuntimeError, match="MLA"):
        build_sglang_structured_arenas(latent_config, latent_runtime, pool)

    sliding = _class(0, "sliding", (0,))
    sliding_config = _config(sliding)
    sliding_runtime = _Runtime(
        (_identity(sliding, pages=4, base=7, first=101),)
    )
    with pytest.raises(RuntimeError, match=r"Full or ordered Full\+SWA"):
        build_sglang_structured_arenas(sliding_config, sliding_runtime, pool)


@pytest.mark.parametrize(
    "replacement",
    (
        (29,),
        tuple(range(30, 50)),
        (0,) + tuple(range(31, 51)),
        tuple(range(30, 50)) + (999,),
    ),
)
def test_external_manifest_rejects_slot_cardinality_or_mapping_mismatch(
    replacement,
):
    config, runtime, pool = _structured_fixture()
    arenas = build_sglang_structured_arenas(config, runtime, pool)
    batch_record, plans, locations = _manifest_inputs(
        config, runtime, previous=14, target=35
    )
    locations[0] = _FakeLocations(replacement)

    with pytest.raises(RuntimeError, match="cardinality|slot vector"):
        build_external_append_manifest(
            batch_record, plans, locations, runtime, config, arenas
        )


def test_manifest_and_descriptors_are_frozen_and_module_has_no_torch_import():
    config, runtime, pool = _structured_fixture()
    arenas = build_sglang_structured_arenas(config, runtime, pool)
    batch_record, plans, locations = _manifest_inputs(
        config, runtime, previous=0, target=1
    )
    manifest = build_external_append_manifest(
        batch_record, plans, locations, runtime, config, arenas
    )

    assert isinstance(arenas[0].components[0], SglangArenaComponent)
    assert isinstance(manifest.writes[0], ExternalTokenWrite)
    assert isinstance(manifest.last_use_pages[0], BackendPageAddress)
    with pytest.raises(FrozenInstanceError):
        arenas[0].retention = "sliding"
    with pytest.raises(FrozenInstanceError):
        manifest.writes = ()

    module = (
        Path(__file__).resolve().parents[1]
        / "src/orbitkv_sglang/plugin/structured_arena.py"
    )
    tree = ast.parse(module.read_text(encoding="utf-8"))
    assert all(
        not (
            isinstance(node, ast.Import)
            and any(alias.name == "torch" for alias in node.names)
        )
        and not (isinstance(node, ast.ImportFrom) and node.module == "torch")
        for node in tree.body
    )


def test_external_manifest_rejects_stale_page_geometry():
    config, runtime, pool = _structured_fixture()
    arenas = build_sglang_structured_arenas(config, runtime, pool)
    batch_record, plans, locations = _manifest_inputs(
        config, runtime, previous=0, target=1
    )
    page = batch_record.records[0].new_pages[0]
    stale = PageLease(
        page.page.engine_epoch,
        page.page.pool_epoch + 1,
        page.page.generation,
        page.page.page_id,
        page.page.pool_id,
    )
    batch_record.records[0].new_pages = (
        PageShadow(
            page.request,
            page.class_id,
            page.logical_ordinal,
            stale,
            page.backend_index,
        ),
    )

    with pytest.raises(RuntimeError, match="differs from its structured arena"):
        build_external_append_manifest(
            batch_record, plans, locations, runtime, config, arenas
        )
