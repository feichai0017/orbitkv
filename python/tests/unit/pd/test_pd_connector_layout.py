from __future__ import annotations

from orbitkv.vllm.pd.layout_mapping import (
    HeadSlice,
    build_push_layout_plan,
    decode_rank_source_counts,
)

# ruff: noqa: F403,F405,I001
from .pd_connector_test_utils import *


def test_bhnc_layout_offsets_and_head_slices() -> None:
    tensor = FakeTensor(shape=(8, 4, 16, 64), stride=(4096, 1024, 64, 1))
    layout = KvCacheLayout.from_tensor(
        "layer.0", tensor, logical_block_size=16, layer_spec=fake_cache_spec()
    )
    assert layout.block_bytes == 8192
    assert layout.block_slices(3).regions == (
        BlockRegionSlice(block_id=3, src_offset_bytes=24576, bytes=8192),
    )
    assert layout.block_slices(3, 1, 3).regions == (
        BlockRegionSlice(block_id=3, src_offset_bytes=26624, bytes=4096),
    )
    assert layout.remote_layout(0, (3, 4)).regions == (
        TransferRegionLayout(region_idx=0, base_addr=0x1000, block_len=8192),
    )


@pytest.mark.parametrize(
    ("shape", "stride", "storage_bytes", "spec", "error"),
    [
        ((8, 4, 16, 64), (4096, 64, 256, 1), None, None, "dense HNC"),
        ((8, 4, 16, 64), (4096, 1024, 64, 2), None, None, "dense HNC"),
        ((8, 4, 16, 64), (1024, 1024, 64, 1), None, None, "overlapping"),
        ((8, 4, 16, 64), (4096, 1024, 64, 1), 65535, None, "backing storage"),
        ((8, 4, 16, 64), (4096, 1024, 64, 1), None, fake_cache_spec(heads=8), "KVCacheSpec"),
        (
            (8, 4, 16, 64),
            (4096, 1024, 64, 1),
            None,
            fake_cache_spec(content_bytes=64),
            "KVCacheSpec",
        ),
        ((8, 16, 64), (1024, 64, 1), None, None, "4D BHNC"),
    ],
)
def test_bhnc_rejects_invalid_views(shape, stride, storage_bytes, spec, error):
    with pytest.raises(AssertionError, match=error):
        KvCacheLayout.from_tensor(
            "layer.0",
            FakeTensor(shape, stride, storage_bytes=storage_bytes),
            logical_block_size=16,
            layer_spec=spec,
        )


def test_bhnc_preserves_page_gaps_and_storage_offset():
    tensor = FakeTensor((8, 4, 16, 64), (8192, 1024, 64, 1), ptr=0x4000, storage_offset=4096)
    spec = fake_cache_spec()
    spec.page_size_bytes = 16384
    layout = KvCacheLayout.from_tensor("layer.0", tensor, logical_block_size=16, layer_spec=spec)
    assert layout.block_slices(7).regions == (
        BlockRegionSlice(block_id=7, src_offset_bytes=7 * 16384, bytes=8192),
    )
    assert layout.remote_layout(0).regions == (
        TransferRegionLayout(region_idx=0, base_addr=0x4000, block_len=8192, block_stride=16384),
    )
    with pytest.raises(AssertionError, match="out of range"):
        layout.block_slices(8)
    with pytest.raises(AssertionError, match="invalid remote block"):
        layout.remote_layout(0, (8,))


def test_pd_worker_registers_mla_and_indexer_layouts_from_layer_specs() -> None:
    main_tensor = FakeTensor(
        shape=(8, 1, 64, 656),
        stride=(64 * 656, 64 * 656, 656, 1),
        ptr=0x1000,
        element_size=1,
    )
    indexer_tensor = FakeTensor(
        shape=(8, 1, 64, 128),
        stride=(64 * 128, 64 * 128, 128, 1),
        ptr=0x200000,
        element_size=1,
    )
    kv_cache_config = fake_kv_cache_config(
        num_blocks=8,
        specs={
            "layer.0": fake_cache_spec(block_size=64, heads=1, content_bytes=656),
            "indexer.0": fake_cache_spec(block_size=64, heads=1, content_bytes=128),
        },
    )
    worker = DecodeWorker(
        fake_mla_config(),
        kv_cache_config=kv_cache_config,
        transfer=MockMooncakePort(),
    )

    worker.register_kv_caches({"layer.0": main_tensor, "indexer.0": indexer_tensor})

    assert worker.layouts["layer.0"].block_size == 64
    assert worker.layouts["layer.0"].remote_layout(0).regions == (
        TransferRegionLayout(region_idx=0, base_addr=0x1000, block_len=64 * 656),
    )
    assert worker.layouts["indexer.0"].remote_layout(1).regions == (
        TransferRegionLayout(region_idx=0, base_addr=0x200000, block_len=64 * 128),
    )
    assert (
        worker.transfer.local_layers[0].regions[0].block_len
        != worker.transfer.local_layers[1].regions[0].block_len
    )


def test_pd_connector_mla_returns_default_layout_and_allows_128_block_size() -> None:
    assert PdDecodeConnector.get_required_kvcache_layout(fake_mla_config(block_size=64)) is None
    PdDecodeConnector(fake_mla_config(block_size=128), KVConnectorRole.WORKER)


def test_pd_connectors_allow_mtp_layout() -> None:
    PdDecodeConnector(fake_mtp_config(), KVConnectorRole.WORKER)
    PdPrefillConnector(fake_mtp_config(), KVConnectorRole.WORKER)


def test_vllm_plugin_registers_framework_connectors(monkeypatch) -> None:
    import sys

    import orbitkv.vllm.plugin as vllm_plugin

    registered = []

    class FakeFactory:
        @staticmethod
        def register_connector(name: str, module: str, class_name: str) -> None:
            registered.append((name, module, class_name))

    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.kv_transfer.kv_connector.factory",
        SimpleNamespace(KVConnectorFactory=FakeFactory),
    )

    vllm_plugin.register()

    assert (
        "PdDecodeConnector",
        "orbitkv.vllm.pd",
        "PdDecodeConnector",
    ) in registered
    assert "PdConnector" not in {name for name, _, _ in registered}
    assert "NoopKVConnector" not in {name for name, _, _ in registered}
    assert (
        "PdPrefillConnector",
        "orbitkv.vllm.pd",
        "PdPrefillConnector",
    ) in registered
    assert not any(name.startswith("Nixl") for name, _, _ in registered)


def test_pd_worker_rejects_mla_physical_logical_block_split() -> None:
    tensor = FakeTensor(
        shape=(16, 1, 32, 128),
        stride=(32 * 128, 32 * 128, 128, 1),
        ptr=0x1000,
        element_size=1,
    )
    kv_cache_config = fake_kv_cache_config(
        num_blocks=8,
        specs={
            "layer.0": fake_cache_spec(block_size=64, heads=1, content_bytes=128),
        },
    )
    worker = DecodeWorker(
        fake_mla_config(block_size=64),
        kv_cache_config=kv_cache_config,
        transfer=MockMooncakePort(),
    )

    with pytest.raises(AssertionError, match="physical/logical block split"):
        worker.register_kv_caches({"layer.0": tensor})


def test_p_worker_maps_mla_prefill_tp_greater_than_decode_tp() -> None:
    handshakes = tuple(
        PdHandshake(
            request_id=f"decode-r{rank}",
            engine_id="decode",
            tp_rank=rank,
            tp_size=4,
            block_size=64,
            layers=(),
        )
        for rank in range(4)
    )
    worker = PrefillWorker(
        fake_mla_config(tp_rank=2, tp_size=8),
        transfer=MockMooncakePort(),
    )

    worker.prepare_pushes(
        PdConnectorMetadata(
            reqs_to_push={
                "prefill-r2": PushReqMeta(
                    local_block_ids=([1],),
                    target_request_id="decode",
                    handshakes=handshakes,
                )
            }
        )
    )

    assert worker.transfer.peer_handshakes["prefill-r2"] is handshakes[1]


def test_p_worker_skips_non_representative_mla_prefill_rank() -> None:
    handshakes = tuple(
        PdHandshake(
            request_id=f"decode-r{rank}",
            engine_id="decode",
            tp_rank=rank,
            tp_size=4,
            block_size=64,
            layers=(),
        )
        for rank in range(4)
    )
    worker = PrefillWorker(
        fake_mla_config(tp_rank=3, tp_size=8),
        transfer=MockMooncakePort(),
    )

    worker.prepare_pushes(
        PdConnectorMetadata(
            reqs_to_push={
                "prefill-r3": PushReqMeta(
                    local_block_ids=([1],),
                    target_request_id="decode",
                    handshakes=handshakes,
                )
            }
        )
    )

    assert "prefill-r3" not in worker.transfer.peer_handshakes
    assert worker.get_finished({"prefill-r3"}) == ({"prefill-r3"}, None)


def test_layout_mapping_homogeneous_tp_reads_same_remote_rank() -> None:
    handshakes = decode_handshakes(tp_size=4)

    plan = build_push_layout_plan(
        prefill_tp_rank=2,
        prefill_tp_size=4,
        decode_handshakes=handshakes,
        local_num_kv_heads=2,
        remote_num_kv_heads=2,
        total_num_kv_heads=8,
        use_mla=False,
    )

    assert [(target.handshake.tp_rank, target.head_slices) for target in plan.targets] == [(2, ())]


def test_layout_mapping_decode_tp_greater_than_prefill_tp_splits_local_heads() -> None:
    handshakes = decode_handshakes(tp_size=4)

    plan = build_push_layout_plan(
        prefill_tp_rank=1,
        prefill_tp_size=2,
        decode_handshakes=handshakes,
        local_num_kv_heads=4,
        remote_num_kv_heads=2,
        total_num_kv_heads=8,
        use_mla=False,
    )

    assert [(target.handshake.tp_rank, target.head_slices) for target in plan.targets] == [
        (
            2,
            (
                HeadSlice(
                    local_start=0,
                    local_end=2,
                    remote_start=0,
                    remote_end=2,
                    global_heads=(4, 5),
                ),
            ),
        ),
        (
            3,
            (
                HeadSlice(
                    local_start=2,
                    local_end=4,
                    remote_start=0,
                    remote_end=2,
                    global_heads=(6, 7),
                ),
            ),
        ),
    ]


def test_layout_mapping_prefill_tp_greater_than_decode_tp_offsets_remote_heads() -> None:
    handshakes = decode_handshakes(tp_size=2)

    plan = build_push_layout_plan(
        prefill_tp_rank=3,
        prefill_tp_size=4,
        decode_handshakes=handshakes,
        local_num_kv_heads=2,
        remote_num_kv_heads=4,
        total_num_kv_heads=8,
        use_mla=False,
    )

    assert [(target.handshake.tp_rank, target.head_slices) for target in plan.targets] == [
        (
            1,
            (
                HeadSlice(
                    local_start=0,
                    local_end=2,
                    remote_start=2,
                    remote_end=4,
                    global_heads=(6, 7),
                ),
            ),
        )
    ]


def test_p_worker_prefill_tp_greater_than_decode_tp_registers_remote_head_slices() -> None:
    tensor = FakeTensor(
        shape=(8, 4, 16, 64),
        stride=(4 * 16 * 64, 16 * 64, 64, 1),
        ptr=0x1000,
    )
    decode_tensor = FakeTensor(
        shape=(8, 8, 16, 64),
        stride=(8 * 16 * 64, 16 * 64, 64, 1),
        ptr=0x2000,
    )
    kv_cache_config = fake_kv_cache_config(
        num_blocks=8,
        specs={
            "layer.0": fake_cache_spec(),
        },
    )
    decode_layer = KvCacheLayout.from_tensor(
        "layer.0", decode_tensor, logical_block_size=16
    ).remote_layout(
        0,
        (1, 2),
    )
    decode_handshake = PdHandshake(
        request_id="decode",
        engine_id="decode",
        tp_rank=0,
        tp_size=1,
        block_size=16,
        layers=(decode_layer,),
    )

    def build_worker(rank: int) -> PrefillWorker:
        worker = PrefillWorker(
            SimpleNamespace(
                kv_transfer_config=FakeKVTransferConfig(engine_id="prefill"),
                model_config=SimpleNamespace(
                    use_mla=False,
                    get_total_num_kv_heads=lambda: 8,
                ),
                cache_config=SimpleNamespace(block_size=16),
                parallel_config=SimpleNamespace(
                    tensor_parallel_rank=rank,
                    tensor_parallel_size=2,
                    decode_context_parallel_size=1,
                    prefill_context_parallel_size=1,
                ),
            ),
            kv_cache_config=kv_cache_config,
            transfer=MockMooncakePort(),
        )
        worker.register_kv_caches({"layer.0": tensor})
        return worker

    rank0 = build_worker(0)
    rank1 = build_worker(1)

    for worker in (rank0, rank1):
        worker.prepare_pushes(
            PdConnectorMetadata(
                reqs_to_push={
                    f"prefill-r{worker.tp_rank}": PushReqMeta(
                        local_block_ids=([1, 2],),
                        target_request_id="decode",
                        handshakes=(decode_handshake,),
                    )
                }
            )
        )

    rank0_remote = rank0.transfer.peer_handshakes["prefill-r0"].layers[0]
    rank1_remote = rank1.transfer.peer_handshakes["prefill-r1"].layers[0]

    assert rank0_remote.regions == (
        TransferRegionLayout(region_idx=0, base_addr=0x2000, block_len=8192, block_stride=16384),
    )
    assert rank1_remote.regions == (
        TransferRegionLayout(region_idx=0, base_addr=0x4000, block_len=8192, block_stride=16384),
    )


def test_layout_mapping_counts_prefill_sources_per_decode_rank() -> None:
    assert decode_rank_source_counts(
        prefill_tp_size=2,
        decode_tp_size=1,
        local_num_kv_heads=2,
        remote_num_kv_heads=4,
        total_num_kv_heads=4,
        use_mla=False,
    ) == {0: 2}

    assert decode_rank_source_counts(
        prefill_tp_size=1,
        decode_tp_size=2,
        local_num_kv_heads=4,
        remote_num_kv_heads=2,
        total_num_kv_heads=4,
        use_mla=False,
    ) == {0: 1, 1: 1}


def test_layout_mapping_mla_uses_one_remote_rank_and_skips_duplicates() -> None:
    handshakes = decode_handshakes(tp_size=4)

    selected = build_push_layout_plan(
        prefill_tp_rank=2,
        prefill_tp_size=8,
        decode_handshakes=handshakes,
        local_num_kv_heads=1,
        remote_num_kv_heads=1,
        total_num_kv_heads=1,
        use_mla=True,
    )
    skipped = build_push_layout_plan(
        prefill_tp_rank=3,
        prefill_tp_size=8,
        decode_handshakes=handshakes,
        local_num_kv_heads=1,
        remote_num_kv_heads=1,
        total_num_kv_heads=1,
        use_mla=True,
    )

    assert [(target.handshake.tp_rank, target.head_slices) for target in selected.targets] == [
        (1, ())
    ]
    assert skipped.targets == ()


def test_layout_mapping_mla_prefill_tp_less_than_decode_tp_fans_out() -> None:
    handshakes = decode_handshakes(tp_size=4)

    rank0 = build_push_layout_plan(
        prefill_tp_rank=0,
        prefill_tp_size=2,
        decode_handshakes=handshakes,
        local_num_kv_heads=1,
        remote_num_kv_heads=1,
        total_num_kv_heads=1,
        use_mla=True,
    )
    rank1 = build_push_layout_plan(
        prefill_tp_rank=1,
        prefill_tp_size=2,
        decode_handshakes=handshakes,
        local_num_kv_heads=1,
        remote_num_kv_heads=1,
        total_num_kv_heads=1,
        use_mla=True,
    )

    assert [(target.handshake.tp_rank, target.head_slices) for target in rank0.targets] == [
        (0, ()),
        (1, ()),
    ]
    assert [(target.handshake.tp_rank, target.head_slices) for target in rank1.targets] == [
        (2, ()),
        (3, ()),
    ]
    assert decode_rank_source_counts(
        prefill_tp_size=2,
        decode_tp_size=4,
        local_num_kv_heads=1,
        remote_num_kv_heads=1,
        total_num_kv_heads=1,
        use_mla=True,
    ) == {0: 1, 1: 1, 2: 1, 3: 1}


def test_layout_mapping_gqa_dedup_skips_redundant_prefill_rank() -> None:
    handshakes = decode_handshakes(tp_size=1)

    owner = build_push_layout_plan(
        prefill_tp_rank=0,
        prefill_tp_size=4,
        decode_handshakes=handshakes,
        local_num_kv_heads=1,
        remote_num_kv_heads=2,
        total_num_kv_heads=2,
        use_mla=False,
    )
    duplicate = build_push_layout_plan(
        prefill_tp_rank=2,
        prefill_tp_size=4,
        decode_handshakes=handshakes,
        local_num_kv_heads=1,
        remote_num_kv_heads=2,
        total_num_kv_heads=2,
        use_mla=False,
    )

    assert [(target.handshake.tp_rank, target.head_slices) for target in owner.targets] == [
        (
            0,
            (
                HeadSlice(
                    local_start=0,
                    local_end=1,
                    remote_start=0,
                    remote_end=1,
                    global_heads=(0,),
                ),
            ),
        )
    ]
    assert duplicate.targets == ()


def test_layout_mapping_rejects_non_divisible_tp_ratios() -> None:
    with pytest.raises(AssertionError, match="requires divisible TP sizes"):
        build_push_layout_plan(
            prefill_tp_rank=0,
            prefill_tp_size=2,
            decode_handshakes=decode_handshakes(tp_size=3),
            local_num_kv_heads=3,
            remote_num_kv_heads=2,
            total_num_kv_heads=6,
            use_mla=False,
        )


def test_real_mooncake_port_maps_pd_push_to_mooncake_ranges() -> None:
    native_engine = FakeMooncakeTransferEngine()
    transfer = RealMooncakePort(native_engine)
    layer = split_remote_layer(
        block_ids=(0, 1),
        k_base=0x1000,
        v_base=0x9000,
        block_len=4096,
    )

    registered = transfer.register_local_layers((layer,))

    assert registered[0].block_ids == (0, 1)
    assert registered[0].regions == layer.regions
    assert native_engine.registered_regions == [
        {"addr": 0x1000, "len": 8192, "location": "*"},
        {"addr": 0x9000, "len": 8192, "location": "*"},
    ]

    handshake = PdHandshake(
        request_id="req-1",
        engine_id="decode",
        transfer_endpoint=native_engine.endpoint,
        tp_rank=0,
        tp_size=1,
        block_size=16,
        layers=registered,
    )
    generation = transfer.open_request("req-1", handshake)

    assert transfer.peer_handshakes["req-1"] == handshake

    transfer.push_layer(
        "req-1",
        0,
        [
            LayerBlockSlices(
                regions=(
                    BlockRegionSlice(block_id=1, src_offset_bytes=4096, bytes=4096),
                    BlockRegionSlice(block_id=1, src_offset_bytes=36864, bytes=4096),
                )
            )
        ],
        request_generation=generation,
    )
    transfer.push_done("req-1")
    transfer.wait_done("req-1")

    assert native_engine.writes == [
        (
            native_engine.endpoint,
            [(0x2000, 0x2000, 4096), (0xA000, 0xA000, 4096)],
            30.0,
        )
    ]
    assert transfer.pop_finished_sending() == {"req-1"}
    assert transfer.pop_finished_sending() == set()


def test_real_mooncake_port_rejects_missing_mooncake_endpoint() -> None:
    native_engine = FakeMooncakeTransferEngine()
    transfer = RealMooncakePort(native_engine)
    layer = split_remote_layer(block_ids=(0,), block_len=1024)
    handshake = PdHandshake(
        request_id="req-1",
        engine_id="decode",
        tp_rank=0,
        tp_size=1,
        block_size=16,
        layers=(layer,),
    )

    with pytest.raises(ValueError, match="transfer_endpoint"):
        transfer.open_request("req-1", handshake)


def test_real_mooncake_port_uses_tent_nic_bandwidth_evidence() -> None:
    native_engine = FakeMooncakeTransferEngine()
    native_engine.nic_stats = [
        ("mlx5_0", 4096, 100_000_000_000.0),
        ("mlx5_1", 0, 80_000_000_000.0),
    ]
    transfer = RealMooncakePort(native_engine)

    assert transfer.aggregated_link_speed() == 1_440_000_000_000


def test_pd_handshake_serializes_regions_layout() -> None:
    layer = split_remote_layer(
        block_ids=(8, 9, 10),
        k_base=0x10_000,
        v_base=0x20_000,
        block_len=0x400,
    )
    handshake = PdHandshake(
        request_id="req-1",
        engine_id="decode",
        transfer_endpoint="10.0.0.2:15290",
        tp_rank=0,
        tp_size=1,
        block_size=16,
        layers=(layer,),
    )

    data = handshake_to_dict(handshake)
    assert data is not None
    layer_data = data["layers"][0]
    assert layer_data["block_ids"] == [8, 9, 10]
    assert layer_data["regions"] == [
        {"region_idx": 0, "base_addr": 0x10_000, "block_len": 0x400},
        {"region_idx": 1, "base_addr": 0x20_000, "block_len": 0x400},
    ]
    assert "k_block_addrs" not in layer_data
    assert "v_block_addrs" not in layer_data
    assert "linear" not in layer_data

    restored = handshake_from_dict(data)
    assert restored is not None
    assert restored.layers[0].block_ids == layer.block_ids
    assert restored.layers[0].regions == layer.regions


def test_pd_handshake_serializes_strided_regions_layout() -> None:
    layer = LayerRemoteLayout(
        layer_name="layer.0",
        layer_idx=0,
        block_ids=(8, 9, 10),
        regions=(
            TransferRegionLayout(
                region_idx=0,
                base_addr=0x10_400,
                block_len=0x400,
                block_stride=0x1000,
            ),
            TransferRegionLayout(
                region_idx=1,
                base_addr=0x20_400,
                block_len=0x400,
                block_stride=0x1000,
            ),
        ),
    )
    handshake = PdHandshake(
        request_id="req-1",
        engine_id="decode",
        transfer_endpoint="10.0.0.2:15290",
        tp_rank=0,
        tp_size=1,
        block_size=16,
        layers=(layer,),
    )

    data = handshake_to_dict(handshake)

    assert data["layers"][0]["regions"] == [
        {
            "region_idx": 0,
            "base_addr": 0x10_400,
            "block_len": 0x400,
            "block_stride": 0x1000,
        },
        {
            "region_idx": 1,
            "base_addr": 0x20_400,
            "block_len": 0x400,
            "block_stride": 0x1000,
        },
    ]
    restored = handshake_from_dict(data)
    assert restored is not None
    assert restored.layers[0].regions == layer.regions


def test_pd_handshake_compact_serializes_shared_block_ids_once() -> None:
    layers = (
        split_remote_layer(layer_name="layer.0", layer_idx=0, block_ids=(8, 9, 10)),
        split_remote_layer(layer_name="layer.1", layer_idx=1, block_ids=(8, 9, 10)),
    )
    handshake = PdHandshake(
        request_id="req-1",
        engine_id="decode",
        transfer_endpoint="10.0.0.2:15290",
        tp_rank=0,
        tp_size=1,
        block_size=16,
        layers=layers,
    )

    data = handshake_to_compact_dict(handshake)

    assert data["block_ids"] == [8, 9, 10]
    assert "block_ids" not in data["layers"][0]
    assert "block_ids" not in data["layers"][1]
    restored = handshake_from_dict(data)
    assert restored is not None
    assert restored.layers[0].block_ids == (8, 9, 10)
    assert restored.layers[1].block_ids == (8, 9, 10)


def test_pd_worker_builds_mooncake_by_default_when_extension_exists(monkeypatch) -> None:
    monkeypatch.setattr(
        native, "MooncakeTransferEngine", FakeMooncakeTransferEngineCtor, raising=False
    )
    tensor = FakeTensor(
        shape=(8, 4, 16, 64),
        stride=(4 * 16 * 64, 16 * 64, 64, 1),
        device_index=2,
    )
    config = SimpleNamespace(
        kv_transfer_config=FakeKVTransferConfig(
            engine_id="decode",
            kv_connector_extra_config={
                "orbitkv.pd.mooncake.bind_host": "10.0.0.2",
                "orbitkv.pd.mooncake.rank_map": {"0": {"nic": "mlx5_2"}},
            },
        ),
        parallel_config=SimpleNamespace(tensor_parallel_rank=0),
    )

    worker = DecodeWorker(config)
    worker.register_kv_caches({"layer.0": tensor})

    assert isinstance(worker.transfer, RealMooncakePort)
    assert FakeMooncakeTransferEngineCtor.last_kwargs == {
        "bind_host": "10.0.0.2",
        "nics": ["mlx5_2"],
    }


def test_pd_worker_allows_mooncake_transport_autoselection(monkeypatch) -> None:
    monkeypatch.setattr(
        native, "MooncakeTransferEngine", FakeMooncakeTransferEngineCtor, raising=False
    )
    tensor = FakeTensor(
        shape=(8, 4, 16, 64),
        stride=(4 * 16 * 64, 16 * 64, 64, 1),
        device_index=2,
    )
    config = SimpleNamespace(
        kv_transfer_config=FakeKVTransferConfig(
            engine_id="decode",
            kv_connector_extra_config={},
        ),
        parallel_config=SimpleNamespace(tensor_parallel_rank=0),
    )

    worker = DecodeWorker(config)
    worker.register_kv_caches({"layer.0": tensor})

    assert FakeMooncakeTransferEngineCtor.last_kwargs == {
        "bind_host": "127.0.0.1",
        "nics": [],
    }


def test_pd_worker_uses_runtime_tp_rank_for_mooncake_rank_map(monkeypatch) -> None:
    monkeypatch.setattr(
        native, "MooncakeTransferEngine", FakeMooncakeTransferEngineCtor, raising=False
    )
    monkeypatch.setattr(worker_mod, "get_tensor_model_parallel_rank", lambda: 2)
    monkeypatch.setattr(worker_mod, "get_tensor_model_parallel_world_size", lambda: 8)
    tensor = FakeTensor(
        shape=(8, 4, 16, 64),
        stride=(4 * 16 * 64, 16 * 64, 64, 1),
        device_index=2,
    )
    config = SimpleNamespace(
        kv_transfer_config=FakeKVTransferConfig(
            engine_id="decode",
            kv_connector_extra_config={
                "orbitkv.pd.mooncake.rank_map": {"0": {"nic": "mlx5_1"}, "2": {"nic": "mlx5_2"}}
            },
        ),
        parallel_config=SimpleNamespace(tensor_parallel_rank=0, tensor_parallel_size=8),
    )

    worker = DecodeWorker(config)
    worker.register_kv_caches({"layer.0": tensor})

    assert FakeMooncakeTransferEngineCtor.last_kwargs["nics"] == ["mlx5_2"]


def test_pd_worker_rank_map_uses_tp_rank_even_when_cuda_ordinal_differs(monkeypatch) -> None:
    monkeypatch.setattr(
        native, "MooncakeTransferEngine", FakeMooncakeTransferEngineCtor, raising=False
    )
    monkeypatch.setattr(worker_mod, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(worker_mod, "get_tensor_model_parallel_world_size", lambda: 1)
    tensor = FakeTensor(
        shape=(8, 4, 16, 64),
        stride=(4 * 16 * 64, 16 * 64, 64, 1),
        device_index=4,
    )
    config = SimpleNamespace(
        kv_transfer_config=FakeKVTransferConfig(
            engine_id="decode",
            kv_connector_extra_config={
                "orbitkv.pd.mooncake.rank_map": {"0": {"nic": "mlx5_0"}, "4": {"nic": "mlx5_4"}}
            },
        ),
        parallel_config=SimpleNamespace(tensor_parallel_rank=0, tensor_parallel_size=1),
    )

    worker = DecodeWorker(config)
    worker.register_kv_caches({"layer.0": tensor})

    assert FakeMooncakeTransferEngineCtor.last_kwargs["nics"] == ["mlx5_0"]


@pytest.mark.parametrize("rank_map", [{"4": {"nic": "mlx5_4"}}, {"0": {}}, [], {"0": {"nic": ""}}])
def test_pd_rank_map_rejects_missing_rank_or_nic(monkeypatch, rank_map):
    from orbitkv.vllm.pd.mooncake import build_mooncake_port

    monkeypatch.setattr(
        native, "MooncakeTransferEngine", FakeMooncakeTransferEngineCtor, raising=False
    )
    config = SimpleNamespace(
        kv_transfer_config=FakeKVTransferConfig(
            kv_connector_extra_config={"orbitkv.pd.mooncake.rank_map": rank_map}
        )
    )
    with pytest.raises(ValueError, match="rank_map"):
        build_mooncake_port(config, 4, tp_rank=0)


def test_pd_write_preserves_different_source_and_destination_page_gaps():
    engine = FakeMooncakeTransferEngine()
    port = RealMooncakePort(engine)
    tensor = FakeTensor((4, 4, 16, 64), (8192, 1024, 64, 1))
    layout = KvCacheLayout.from_tensor("layer.0", tensor, logical_block_size=16)
    port.register_local_layers((layout.remote_layout(0),))
    remote = LayerRemoteLayout(
        layer_name="layer.0",
        layer_idx=0,
        block_ids=(2, 3),
        regions=(
            TransferRegionLayout(
                region_idx=0, base_addr=0x80000, block_len=8192, block_stride=32768
            ),
        ),
    )
    handshake = PdHandshake(
        request_id="remote",
        engine_id="decode",
        transfer_endpoint="peer:1",
        tp_rank=0,
        tp_size=1,
        block_size=16,
        layers=(remote,),
    )
    generation = port.open_request("req", handshake)
    port.push_layer(
        "req",
        0,
        [
            LayerBlockSlices(
                regions=(BlockRegionSlice(block_id=3, src_offset_bytes=16384, bytes=8192),)
            ),
            LayerBlockSlices(
                regions=(BlockRegionSlice(block_id=2, src_offset_bytes=49152, bytes=8192),)
            ),
        ],
        request_generation=generation,
    )
    assert engine.writes == [
        (
            "peer:1",
            [
                (0x5000, 0x98000, 8192),
                (0xD000, 0x90000, 8192),
            ],
            30.0,
        )
    ]
    assert engine.registered_regions == [{"addr": 0x1000, "len": 57344, "location": "*"}]
    for block, error in [
        (BlockRegionSlice(block_id=2, src_offset_bytes=57344, bytes=8192), "registered layer"),
        (BlockRegionSlice(block_id=1, src_offset_bytes=0, bytes=8192), "not authorized"),
        (BlockRegionSlice(block_id=2, src_offset_bytes=0, bytes=16384), "must equal"),
    ]:
        with pytest.raises((ValueError, RuntimeError), match=error):
            port.push_layer(
                "req", 0, [LayerBlockSlices(regions=(block,))], request_generation=generation
            )
    assert len(engine.writes) == 1


def test_pinned_mla_callback_squeezes_only_the_registered_single_head_axis():
    tensor = FakeTensor((8, 1, 64, 656), (41984, 41984, 656, 1), element_size=1)
    layout = KvCacheLayout.from_tensor(
        "mla.0",
        tensor,
        logical_block_size=64,
        layer_spec=fake_cache_spec(block_size=64, heads=1, content_bytes=656),
    )
    callback = FakeTensor((8, 64, 656), (41984, 656, 1), element_size=1)
    prefill_worker_mod._assert_runtime_layout_matches("mla.0", callback, layout)
    changed = FakeTensor((8, 64, 656), (41984, 656, 1), ptr=0x2000, element_size=1)
    with pytest.raises(AssertionError, match="base address changed"):
        prefill_worker_mod._assert_runtime_layout_matches("mla.0", changed, layout)
