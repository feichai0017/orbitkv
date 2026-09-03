from __future__ import annotations

import copy
import json
import os
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = INTEGRATION_ROOT / "bridge/src"
TOOLS_ROOT = INTEGRATION_ROOT / "tools"
sys.path.insert(0, str(TOOLS_ROOT))
sys.path.insert(0, str(SOURCE_ROOT))

import engine_e2e as runner  # noqa: E402
from orbitkv_sglang import observability  # noqa: E402
from orbitkv_sglang._retention_ir import (  # noqa: E402
    layout_plan_fingerprint,
)
from orbitkv_sglang.runtime_admission import (  # noqa: E402
    runtime_binding_from_manifest,
)
from orbitkv_sglang.runtime import CacheSharingPolicy  # noqa: E402
from orbitkv_sglang.session_activity import SessionSwaActivity  # noqa: E402
from manifest_test_support import compile_runtime_manifest  # noqa: E402


_SHARED_SESSION_EXPECTED = {
    "expected_lifecycle_route": "native_session",
    "expected_cache_policy": "shared_prefix",
    "expected_class_ids": (0,),
}


def _args(mode: str, paths: dict[str, Path | None]) -> Namespace:
    return Namespace(
        mode=mode,
        sglang_root=str(paths["sglang_root"]),
        model=str(paths["model"]),
        manifest=(None if paths["manifest"] is None else str(paths["manifest"])),
        library=(None if paths["library"] is None else str(paths["library"])),
        prompt_tokens=5,
        decode_tokens=3,
        warmups=1,
        iterations=3,
        context_length=64,
        max_total_tokens=64,
        chunked_prefill_size=16,
        page_size=16,
        attention_backend="fa3",
        mem_fraction_static=None,
        seed=7,
    )


def _seal(value: dict[str, object]) -> None:
    payload = {name: item for name, item in value.items() if name != "fingerprint"}
    value["fingerprint"] = "sha256:" + runner.hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def _chunked_manifest(chunk_tokens: int = 64) -> dict[str, object]:
    blocks = chunk_tokens // 16
    program = {
        "schema": "orbitkv.retention-ir.v1",
        "page_tokens": 16,
        "states": [
            {
                "name": "chunked",
                "layers": [0, 1],
                "bytes_per_token_per_layer": 96,
                "may_read": {
                    "op": "equal",
                    "lhs": {
                        "op": "floor_div",
                        "value": {"op": "query_position"},
                        "divisor": chunk_tokens,
                    },
                    "rhs": {
                        "op": "floor_div",
                        "value": {"op": "key_position"},
                        "divisor": chunk_tokens,
                    },
                },
            }
        ],
    }
    layout = {
        "schema": "orbitkv.layout-program.v1",
        "plan_fingerprint": "",
        "page_tokens": 16,
        "classes": [
            {
                "name": "chunked",
                "layers": [0, 1],
                "bytes_per_token_per_layer": 96,
                "address": {
                    "kind": "resettable_arena",
                    "blocks_per_epoch": blocks,
                },
                "retirement": {
                    "kind": "epoch_end",
                    "blocks_per_epoch": blocks,
                },
                "minimum_slots_per_request": blocks,
            }
        ],
    }
    layout["plan_fingerprint"] = layout_plan_fingerprint(layout)
    manifest: dict[str, object] = {
        "schema": "orbitkv.runtime-manifest",
        "version": 3,
        "fingerprint": "",
        "source": {"kind": "retention_ir", "program": program},
        "token_manager_plan": {"layout": layout},
        "attention_state_plan": None,
        "capability_requirements": [
            "resettable_arena_addressing",
            "semantic_retirement",
            "token_manager",
        ],
    }
    _seal(manifest)
    return manifest


@pytest.fixture
def paths(tmp_path: Path) -> dict[str, Path | None]:
    root = tmp_path / "sglang"
    package = root / "python/sglang"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "vocab_size": 128,
                "max_position_embeddings": 128,
                "eos_token_id": 2,
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    library = tmp_path / "liborbitkv.so"
    library.write_bytes(b"library")
    return {
        "sglang_root": root,
        "model": model,
        "manifest": manifest,
        "library": library,
    }


def _engine_state(engine_args: dict[str, object]) -> dict[str, object]:
    fields = (
        "page_size",
        "max_total_tokens",
        "attention_backend",
        "dtype",
        "kv_cache_dtype",
        "chunked_prefill_size",
        "max_prefill_tokens",
        "prefill_max_requests",
        "max_running_requests",
        "disable_overlap_schedule",
        "disable_radix_cache",
        "disable_cuda_graph",
        "enable_torch_compile",
        "enable_dynamic_chunking",
        "enable_mixed_chunk",
        "tp_size",
        "pp_size",
        "dp_size",
        "dcp_size",
    )
    state = {name: engine_args[name] for name in fields}
    state["effective_max_running_requests_per_dp"] = 1
    return state


def _manager_state(
    *,
    active: bool,
    activity_scale: int = 1,
    lifecycle_route: str = "native_session",
    cache_policy: str = "shared_prefix",
    hybrid: bool = False,
) -> dict[str, object]:
    counters = {name: 0 for name in runner._SESSION_COUNTER_FIELDS}
    if active:
        counters.update(
            forward_events=8 * activity_scale,
            completion_values=8 * activity_scale,
            event_queries=8 * activity_scale,
        )
    identities = [
        {
            "engine_epoch": 1,
            "pool_epoch": 2 + index,
            "pool_id": 1 + index,
            "class_id": index,
            "backend_domain": 1 + index,
            "page_count": 8,
            "page_tokens": 16,
            "backend_base_index": index * 8,
            "first_page_id": 1 + index * 8,
        }
        for index in range(2 if hybrid else 1)
    ]
    arenas = [
        {
            **{
                name: identity[name]
                for name in (
                    "engine_epoch",
                    "pool_epoch",
                    "pool_id",
                    "class_id",
                    "backend_domain",
                    "page_count",
                    "first_page_id",
                )
            },
            "free_pages": 8,
            "reserved_pages": 0,
            "writing_pages": 0,
            "active_pages": 0,
            "retiring_pages": 0,
            "quarantined_pages": 0,
            "exhausted_pages": 0,
            "request_page_refs": 0,
            "prefix_page_refs": 0,
            "reader_pins": 0,
        }
        for identity in identities
    ]
    total_pages = sum(item["page_count"] for item in arenas)
    return {
        "wire_version": runner.WIRE_VERSION,
        "manager_input_fingerprint": "sha256:manager-input",
        "runtime_manifest_fingerprint": "sha256:manifest",
        "runtime_binding_fingerprint": "sha256:binding",
        "lifecycle_route": lifecycle_route,
        "cache_policy": cache_policy,
        "direct_source_owner": {
            "module": "orbitkv_sglang.engine",
            "type": "OrbitKvLifecycleOwner",
            "owner_is_process_singleton": True,
            "config_is_canonical": True,
            "allocator_owned": True,
            "tree_cache_owned": True,
        },
        "completion_evidence": {
            "event_backend": "cuda_event_current_forward_stream",
            "pending_events": 0,
            "completion_high_water": (
                [{"domain": 1, "value": 8 * activity_scale}]
                if active
                else []
            ),
        },
        "runtime_proof": None,
        "fixed_state_byte_count": 0,
        "fixed_state_descriptors": [],
        "tree_cache_type": {"module": "test", "qualname": "Cache"},
        "identities": identities,
        "manager_stats": {
            "active_requests": 0,
            "active_snapshots": 0,
            "active_prefixes": 0,
            "evicted_prefixes": 0,
            "prepared_steps": 0,
            "submitted_steps": 0,
            "free_pages": total_pages,
            "reserved_pages": 0,
            "writing_pages": 0,
            "active_pages": 0,
            "retiring_pages": 0,
            "quarantined_pages": 0,
            "exhausted_pages": 0,
            "pending_reclamations": 0,
            "total_request_page_refs": 0,
            "total_prefix_page_refs": 0,
            "total_reader_pins": 0,
        },
        "arena_stats": arenas,
        "batch_counters": counters,
        "swa_activity": {
            "status": "not_applicable",
            "applicable": False,
            "source": "native_runtime_session",
            "derived": False,
            "swa_retirement_certificates": 0,
            "swa_pages_reclaimed": 0,
            "swa_wrap_events": 0,
            "swa_page_reuse_events": 0,
        },
        "pressure": {
            "schema": "orbitkv.runtime-pressure.v1",
            "enabled": False,
            "mode": "event_driven_high_water",
            "sample_count": 0,
        },
    }


def test_validate_arguments_separates_manager_and_stock_artifacts(
    paths: dict[str, Path | None],
) -> None:
    manager = _args("manager", paths)
    assert runner.validate_arguments(manager) == paths

    stock = _args("stock", paths)
    stock.manifest = None
    stock.library = None
    resolved = runner.validate_arguments(stock)
    assert resolved["manifest"] is None
    assert resolved["library"] is None

    stock.manifest = str(paths["manifest"])
    with pytest.raises(ValueError, match="stock mode forbids"):
        runner.validate_arguments(stock)

    manager.library = None
    with pytest.raises(ValueError, match="manager mode requires"):
        runner.validate_arguments(manager)


@pytest.mark.parametrize(
    ("topology", "classes", "cache_policy"),
    (
        (
            "whole_domain_full_token_kv",
            [
                {
                    "name": "full",
                    "layers": [0, 1],
                    "retention": "full",
                    "bytes_per_token_per_layer": 96,
                    "window_tokens": None,
                }
            ],
            "shared_prefix",
        ),
        (
            "whole_domain_full_latent_kv",
            [
                {
                    "name": "latent",
                    "layers": [0, 1],
                    "retention": "full",
                    "bytes_per_token_per_layer": 96,
                    "window_tokens": None,
                    "storage": "latent_kv",
                    "components": [
                        {
                            "name": "latent",
                            "bytes_per_token_per_layer": 80,
                        },
                        {
                            "name": "rope",
                            "bytes_per_token_per_layer": 16,
                        },
                    ],
                }
            ],
            "request_private",
        ),
        (
            "whole_domain_full_sliding_token_kv",
            [
                {
                    "name": "full",
                    "layers": [0],
                    "retention": "full",
                    "bytes_per_token_per_layer": 96,
                    "window_tokens": None,
                },
                {
                    "name": "sliding",
                    "layers": [1],
                    "retention": "sliding",
                    "bytes_per_token_per_layer": 96,
                    "window_tokens": 18,
                },
            ],
            "shared_prefix",
        ),
        (
            "whole_domain_sliding_token_kv",
            [
                {
                    "name": "sliding",
                    "layers": [0, 1],
                    "retention": "sliding",
                    "bytes_per_token_per_layer": 96,
                    "window_tokens": 18,
                }
            ],
            "request_private",
        ),
    ),
)
def test_admission_accepts_attention_state_native_session_profiles(
    tmp_path: Path,
    topology: str,
    classes: list[dict[str, object]],
    cache_policy: str,
) -> None:
    manifest = compile_runtime_manifest(
        tmp_path,
        manager_plan={"page_tokens": 16, "classes": classes},
        stem="native-session-profile",
    )
    library = tmp_path / "liborbitkv.so"
    library.touch()

    admission = runner.load_native_session_admission(manifest, library)

    assert admission["runtime_binding"]["execution_topology"] == topology
    assert admission["runtime_manifest"]["source"]["kind"] == "attention_state"
    signature = admission["runtime_binding"]["execution_signature"]
    assert signature["fixed_states"] == []
    assert admission["cache_policy"] == cache_policy
    assert admission["lifecycle_route"] == "native_session"
    assert admission["class_ids"] == tuple(range(len(classes)))
    assert admission["chunk_geometry"] is None
    args = _args(
        "manager",
        {
            "sglang_root": tmp_path,
            "model": tmp_path,
            "manifest": manifest,
            "library": library,
        },
    )
    engine = runner.engine_arguments(
        args,
        tmp_path,
        cache_policy=cache_policy,
    )
    assert engine["disable_radix_cache"] is (
        cache_policy == "request_private"
    )


def test_admission_accepts_exact_chunked_and_derives_geometry(
    tmp_path: Path,
) -> None:
    manifest = _chunked_manifest()
    path = tmp_path / "runtime-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    library = tmp_path / "liborbitkv.so"
    library.touch()

    admission = runner.load_native_session_admission(path, library)

    assert admission["runtime_binding"]["execution_topology"] == (
        "whole_domain_chunked_token_kv"
    )
    assert admission["cache_policy"] == "request_private"
    assert admission["class_ids"] == (0,)
    assert admission["chunk_geometry"] == {
        "page_tokens": 16,
        "blocks_per_epoch": 4,
        "chunk_tokens": 64,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda item: item["address"].update(kind="periodic"),
            "resettable-arena",
        ),
        (
            lambda item: item["retirement"].update(blocks_per_epoch=5),
            "EpochEnd",
        ),
        (
            lambda item: item.update(minimum_slots_per_request=5),
            "minimum_slots_per_request",
        ),
    ),
)
def test_chunked_geometry_rejects_address_retirement_or_slot_drift(
    mutation: object, message: str
) -> None:
    binding = runtime_binding_from_manifest(_chunked_manifest())
    signature = binding["execution_signature"]
    item = copy.deepcopy(signature["token_classes"][0])
    assert callable(mutation)
    mutation(item)

    with pytest.raises(ValueError, match=message):
        runner._chunked_geometry(signature, item)


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ({"attention_backend": "flashinfer"}, "attention-backend=fa3"),
        ({"chunked_prefill_size": 32}, "chunked-prefill-size"),
        ({"checkpoint_chunk_tokens": 32}, "checkpoint attention_chunk_size"),
        (
            {"prompt_tokens": 32, "decode_tokens": 33},
            "cross an epoch boundary",
        ),
    ),
)
def test_chunked_execution_binds_engine_checkpoint_and_cross_epoch_workload(
    paths: dict[str, Path | None], change: dict[str, object], message: str
) -> None:
    args = _args("manager", paths)
    args.prompt_tokens = 48
    args.decode_tokens = 18
    args.context_length = 128
    args.chunked_prefill_size = 64
    checkpoint_chunk_tokens = int(change.get("checkpoint_chunk_tokens", 64))
    for name, value in change.items():
        if name != "checkpoint_chunk_tokens":
            setattr(args, name, value)
    geometry = {
        "page_tokens": 16,
        "blocks_per_epoch": 4,
        "chunk_tokens": 64,
    }

    with pytest.raises(RuntimeError, match=message):
        runner.validate_chunked_execution(
            args, geometry, checkpoint_chunk_tokens
        )


def test_chunked_execution_reports_derived_cross_epoch_geometry(
    paths: dict[str, Path | None],
) -> None:
    args = _args("manager", paths)
    args.prompt_tokens = 48
    args.decode_tokens = 18
    args.context_length = 128
    args.chunked_prefill_size = 64

    assert runner.validate_chunked_execution(
        args,
        {
            "page_tokens": 16,
            "blocks_per_epoch": 4,
            "chunk_tokens": 64,
        },
        64,
    ) == {
        "page_tokens": 16,
        "blocks_per_epoch": 4,
        "chunk_tokens": 64,
        "final_kv_tokens": 65,
        "chunk_epoch_count": 2,
    }


def test_verify_source_selects_exact_checkout_validator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    patch = tmp_path / "adapter.patch"
    patch.write_bytes(b"reviewed")
    monkeypatch.setattr(runner, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        runner.pinned,
        "pinned_source_contract",
        lambda: {
            "release": "v1.2.3",
            "revision": "abc",
            "patch_path": "adapter.patch",
        },
    )
    calls = []
    monkeypatch.setattr(
        runner.pinned, "validate_base_checkout", lambda root: calls.append("base") or root
    )
    monkeypatch.setattr(
        runner.pinned,
        "validate_patched_checkout",
        lambda root: calls.append("patched") or root,
    )

    assert runner.verify_source(tmp_path, "stock")["patch"]["status"] == "absent"
    assert runner.verify_source(tmp_path, "manager")["patch"]["status"] == "applied"
    assert calls == ["base", "patched"]


def test_configure_environment_rejects_plugins_and_clears_stale_orbitkv(
    monkeypatch: pytest.MonkeyPatch, paths: dict[str, Path | None]
) -> None:
    for name in tuple(sys.modules):
        if name == "sglang" or name.startswith("sglang."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(os, "environ", os.environ.copy())
    monkeypatch.setenv("SGLANG_PLUGINS", "anything")
    with pytest.raises(RuntimeError, match="SGLANG_PLUGINS must be unset"):
        runner.configure_environment(_args("manager", paths), paths)

    monkeypatch.delenv("SGLANG_PLUGINS")
    monkeypatch.setenv("ORBITKV_STALE", "yes")
    environment = runner.configure_environment(_args("manager", paths), paths)
    assert "ORBITKV_STALE" not in os.environ
    assert environment["ORBITKV_RUNTIME_MANIFEST"] == str(paths["manifest"])
    assert "ORBITKV_STRUCTURED_DATA_PLANE" not in environment
    assert "ORBITKV_PRESSURE_TELEMETRY" not in environment
    assert "SGLANG_PLUGINS" not in environment

    stock_args = _args("stock", paths)
    stock_args.manifest = None
    stock_args.library = None
    stock_environment = runner.configure_environment(
        stock_args, dict(paths, manifest=None, library=None)
    )
    assert not any(name.startswith("ORBITKV_") for name in os.environ)
    assert not any(name.startswith("ORBITKV_") for name in stock_environment)


def test_reject_installed_sglang_plugins_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner.importlib.metadata, "entry_points", lambda **_kwargs: []
    )
    runner.reject_installed_sglang_plugins()

    monkeypatch.setattr(
        runner.importlib.metadata,
        "entry_points",
        lambda **_kwargs: [SimpleNamespace(name="unrelated")],
    )
    with pytest.raises(RuntimeError, match="forbids installed SGLang plugins: unrelated"):
        runner.reject_installed_sglang_plugins()

    def unavailable(**_kwargs: object) -> object:
        raise OSError("metadata unavailable")

    monkeypatch.setattr(runner.importlib.metadata, "entry_points", unavailable)
    with pytest.raises(RuntimeError, match="cannot prove.*group is empty"):
        runner.reject_installed_sglang_plugins()


def test_deterministic_inputs_are_repeatable_unique_and_exclude_controls() -> None:
    arguments = {
        "prompt_tokens": 32,
        "vocab_size": 128,
        "seed": 17,
        "forbidden": (0, 1, 2, 17, 99),
    }
    first = runner.deterministic_input_ids(**arguments, iteration=0)
    assert first == runner.deterministic_input_ids(**arguments, iteration=0)
    second = runner.deterministic_input_ids(**arguments, iteration=1)
    assert first != second
    assert not set(first + second) & set(arguments["forbidden"])


def test_timing_summary_uses_measured_iterations_only() -> None:
    result = runner.summarize_timings(
        [1.0, 2.0, 3.0, 4.0], prompt_tokens=6, decode_tokens=2
    )
    assert result["iteration_seconds"] == [1.0, 2.0, 3.0, 4.0]
    assert result["median_seconds"] == 2.5
    assert result["p95_seconds"] == pytest.approx(3.85)
    assert result["output_tokens_per_second"] == pytest.approx(0.8)
    assert result["total_tokens_per_second"] == pytest.approx(3.2)


def test_accelerator_provenance_is_complete_and_generic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch_module = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            get_device_properties=lambda index: SimpleNamespace(
                name="Generic Accelerator", total_memory=1024 * 1024
            ),
            get_device_capability=lambda index: (9, 0),
        ),
        version=SimpleNamespace(cuda="13.0"),
    )
    monkeypatch.setattr(runner, "_cuda_driver_version", lambda: "13.1")

    assert runner.accelerator_provenance(torch_module) == {
        "device_type": "cuda",
        "device_name": "Generic Accelerator",
        "compute_capability": {"major": 9, "minor": 0},
        "total_memory_bytes": 1024 * 1024,
        "runtime_version": "13.0",
        "driver_version": "13.1",
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("name", "", "device name"),
        ("total_memory", True, "total memory"),
    ),
)
def test_accelerator_provenance_rejects_invalid_properties(
    monkeypatch: pytest.MonkeyPatch, field: str, value: object, message: str
) -> None:
    properties = SimpleNamespace(name="Accelerator", total_memory=1024)
    setattr(properties, field, value)
    torch_module = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            get_device_properties=lambda index: properties,
            get_device_capability=lambda index: (9, 0),
        ),
        version=SimpleNamespace(cuda="13.0"),
    )
    monkeypatch.setattr(runner, "_cuda_driver_version", lambda: "13.1")
    with pytest.raises(RuntimeError, match=message):
        runner.accelerator_provenance(torch_module)


@pytest.mark.parametrize("hybrid", (False, True), ids=("full", "full-swa"))
def test_observability_emits_shared_prefix_native_session_contract(
    monkeypatch: pytest.MonkeyPatch, hybrid: bool,
) -> None:
    monkeypatch.delenv("ORBITKV_PRESSURE_TELEMETRY", raising=False)
    identities = tuple(
        SimpleNamespace(
            engine_epoch=1,
            pool_epoch=2 + index,
            pool_id=3 + index,
            class_id=index,
            backend_domain=4 + index,
            page_count=8,
            page_tokens=16,
            backend_base_index=index * 8,
            first_page_id=1 + index * 8,
        )
        for index in range(2 if hybrid else 1)
    )
    arenas = tuple(
        SimpleNamespace(
            **{
                name: getattr(identity, name)
                for name in (
                    "engine_epoch", "pool_epoch", "pool_id", "class_id",
                    "backend_domain", "page_count", "first_page_id",
                )
            },
            free_pages=8, reserved_pages=0, writing_pages=0, active_pages=0,
            retiring_pages=0, quarantined_pages=0, exhausted_pages=0,
            request_page_refs=0, prefix_page_refs=0, reader_pins=0,
        )
        for identity in identities
    )
    stats = SimpleNamespace(
        active_requests=0,
        active_snapshots=0,
        active_prefixes=0,
        evicted_prefixes=0,
        prepared_steps=0,
        submitted_steps=0,
        free_pages=8 * len(identities),
        reserved_pages=0,
        writing_pages=0,
        active_pages=0,
        retiring_pages=0,
        quarantined_pages=0,
        exhausted_pages=0,
        pending_reclamations=0,
        total_request_page_refs=0,
        total_prefix_page_refs=0,
        total_reader_pins=0,
    )

    class RuntimeSession:
        arenas = identities
        polls = 0
        mirror_cleanup_bound = True
        cache_sharing_policy = CacheSharingPolicy.SHARED_PREFIX
        activity_calls = 0

        def poll(self) -> tuple[object, ...]:
            self.polls += 1
            return ()

        def census(self) -> tuple[object, tuple[object, ...]]:
            return stats, arenas

        def completion_evidence(self, *, external: bool) -> dict[str, object]:
            assert external is False
            return {
                "event_backend": "cuda_event_current_forward_stream",
                "pending_events": 0,
                "completion_high_water": [{"domain": 4, "value": 2}],
            }

        def performance_counters(self) -> dict[str, int]:
            return {
                "forward_events": 2,
                "completion_values": 2,
                "event_queries": 1,
                "event_waits": 1,
                "fail_stop_count": 0,
            }

        def swa_activity(self, classes: object) -> SessionSwaActivity:
            self.activity_calls += 1
            assert classes is config.classes
            return SessionSwaActivity(
                hybrid,
                3 if hybrid else 0,
                3 if hybrid else 0,
                1 if hybrid else 0,
                2 if hybrid else 0,
            )

    runtime = RuntimeSession()
    config = SimpleNamespace(
        runtime_binding={"fingerprint": "sha256:binding"},
        runtime_manifest_fingerprint="sha256:manifest",
        plan_fingerprint="sha256:manager-input",
        fixed_state_byte_count=0,
        classes=tuple(
            SimpleNamespace(
                class_id=index,
                retention=("sliding" if hybrid and index == 1 else "full"),
                period_blocks=(3 if hybrid and index == 1 else None),
            )
            for index in range(2 if hybrid else 1)
        ),
    )
    monkeypatch.setattr(observability, "_runtime", lambda: runtime)
    monkeypatch.setattr(observability, "SessionRuntime", RuntimeSession)
    monkeypatch.setattr(observability, "_config", lambda: config)
    monkeypatch.setattr(
        observability._state, "_requires_disabled_radix_cache", lambda: False
    )
    monkeypatch.setattr(observability._state, "_runtime_backend_proof", lambda: None)
    monkeypatch.setattr(observability._state, "_fixed_state_descriptors", lambda: [])
    monkeypatch.setattr(
        observability._state,
        "_activity_counters",
        lambda: pytest.fail("session observability used Python-runtime counters"),
    )
    monkeypatch.setattr(observability._state, "_FIXED_STATE", None)

    cache = SimpleNamespace(_no_prefix=False, disable_finished_insert=False)
    result = SimpleNamespace(internal_state={})
    returned = observability.augment_internal_state(
        lambda _scheduler: result, SimpleNamespace(tree_cache=cache)
    )

    assert returned is result
    assert runtime.polls == 1
    assert runtime.activity_calls == 1
    manager = result.internal_state["orbitkv_manager"]
    assert manager["lifecycle_route"] == "native_session"
    assert "lifecycle_authority" not in manager
    assert manager["cache_policy"] == "shared_prefix"
    assert len(manager["identities"]) == (2 if hybrid else 1)
    assert manager["batch_counters"] == {
        "forward_events": 2,
        "completion_values": 2,
        "event_queries": 1,
        "event_waits": 1,
        "fail_stop_count": 0,
    }
    assert manager["swa_activity"] == {
        "status": "exposed" if hybrid else "not_applicable",
        "applicable": hybrid,
        "source": "native_runtime_session",
        "derived": False,
        "swa_retirement_certificates": 3 if hybrid else 0,
        "swa_pages_reclaimed": 3 if hybrid else 0,
        "swa_wrap_events": 1 if hybrid else 0,
        "swa_page_reuse_events": 2 if hybrid else 0,
    }
    assert manager["pressure"] == {
        "schema": "orbitkv.runtime-pressure.v1",
        "enabled": False,
        "mode": "event_driven_high_water",
        "sample_count": 0,
    }


def test_observability_rejects_non_session_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NativeSession:
        pass

    monkeypatch.setattr(observability, "SessionRuntime", NativeSession)
    monkeypatch.setattr(observability, "_runtime", lambda: object())

    with pytest.raises(RuntimeError, match="not a native SessionRuntime"):
        observability.augment_internal_state(
            lambda _scheduler: SimpleNamespace(internal_state={}),
            SimpleNamespace(tree_cache=object()),
        )


def test_session_baseline_and_monotonic_evidence_are_required() -> None:
    baseline = {
        "identities": [{"pool_id": 1}],
        "batch_counters": {name: 0 for name in runner._SESSION_COUNTER_FIELDS},
        "completion_evidence": {"completion_high_water": []},
    }
    runner.require_session_baseline(baseline)
    active = copy.deepcopy(baseline)
    active["batch_counters"].update(
        forward_events=1, completion_values=1, event_queries=1
    )
    active["completion_evidence"]["completion_high_water"] = [
        {"domain": 1, "value": 1}
    ]
    runner.require_session_monotonic(baseline, active)

    dirty = copy.deepcopy(baseline)
    dirty["batch_counters"]["event_queries"] = 1
    with pytest.raises(RuntimeError, match="not empty immediately after load"):
        runner.require_session_baseline(dirty)

    regressed = copy.deepcopy(active)
    regressed["batch_counters"]["event_queries"] = 0
    with pytest.raises(RuntimeError, match="counters decreased"):
        runner.require_session_monotonic(active, regressed)

    foreign = copy.deepcopy(active)
    foreign["identities"][0]["pool_id"] = 2
    with pytest.raises(RuntimeError, match="arena identity changed"):
        runner.require_session_monotonic(active, foreign)


@pytest.mark.parametrize(
    (
        "runtime_is_session",
        "cleanup_bound",
        "request_private",
        "no_prefix",
        "disable_finished_insert",
        "message",
    ),
    (
        (False, True, False, False, False, "not a native SessionRuntime"),
        (True, False, False, False, False, "cleanup authority"),
        (True, True, True, False, False, "request_private policy"),
        (True, True, False, True, True, "shared_prefix policy"),
        (
            True,
            True,
            True,
            True,
            True,
            "native runtime-session cache policy",
        ),
    ),
)
def test_observability_fails_closed_on_session_wiring_drift(
    monkeypatch: pytest.MonkeyPatch,
    runtime_is_session: bool,
    cleanup_bound: bool,
    request_private: bool,
    no_prefix: bool,
    disable_finished_insert: bool,
    message: str,
) -> None:
    class ExpectedRuntime:
        mirror_cleanup_bound = cleanup_bound
        cache_sharing_policy = CacheSharingPolicy.SHARED_PREFIX

        def poll(self) -> tuple[object, ...]:
            return ()

        def census(self) -> tuple[object, tuple[object, ...]]:
            stats = SimpleNamespace()
            identity = SimpleNamespace(class_id=0)
            return stats, (identity,)

    class OtherRuntime(ExpectedRuntime):
        pass

    runtime = ExpectedRuntime() if runtime_is_session else OtherRuntime()
    if not runtime_is_session:
        monkeypatch.setattr(
            observability,
            "SessionRuntime",
            type("DifferentRuntime", (), {}),
        )
    else:
        monkeypatch.setattr(observability, "SessionRuntime", ExpectedRuntime)
    monkeypatch.setattr(observability, "_runtime", lambda: runtime)
    monkeypatch.setattr(
        observability._state,
        "_requires_disabled_radix_cache",
        lambda: request_private,
    )
    monkeypatch.setattr(observability._state, "_FIXED_STATE", None)
    scheduler = SimpleNamespace(
        tree_cache=SimpleNamespace(
            _no_prefix=no_prefix,
            disable_finished_insert=disable_finished_insert,
        )
    )
    with pytest.raises(RuntimeError, match=message):
        observability.augment_internal_state(
            lambda _scheduler: SimpleNamespace(internal_state={}), scheduler
        )


def test_observability_rejects_session_pressure_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RuntimeSession:
        arenas = (SimpleNamespace(class_id=0),)
        mirror_cleanup_bound = True
        cache_sharing_policy = CacheSharingPolicy.SHARED_PREFIX

        def poll(self) -> tuple[object, ...]:
            return ()

        def census(self) -> tuple[object, tuple[object, ...]]:
            return SimpleNamespace(), self.arenas

    runtime = RuntimeSession()
    monkeypatch.setattr(observability, "SessionRuntime", RuntimeSession)
    monkeypatch.setattr(observability, "_runtime", lambda: runtime)
    monkeypatch.setattr(
        observability._state, "_requires_disabled_radix_cache", lambda: False
    )
    monkeypatch.setattr(observability._state, "_FIXED_STATE", None)
    monkeypatch.setenv("ORBITKV_PRESSURE_TELEMETRY", "1")
    scheduler = SimpleNamespace(
        tree_cache=SimpleNamespace(
            _no_prefix=False, disable_finished_insert=False
        )
    )
    with pytest.raises(RuntimeError, match="do not support pressure telemetry"):
        observability.augment_internal_state(
            lambda _scheduler: SimpleNamespace(internal_state={}), scheduler
        )


def test_manager_snapshot_requires_owner_activity_and_final_drain() -> None:
    info = {
        "internal_states": [
            {
                "page_size": 16,
                "orbitkv_manager": _manager_state(active=True),
            }
        ]
    }
    snapshot = runner.manager_snapshot(
        info,
        "after_workload",
        manifest_fingerprint="sha256:manifest",
        binding_fingerprint="sha256:binding",
        manager_input_fingerprint="sha256:manager-input",
        **_SHARED_SESSION_EXPECTED,
        require_activity=True,
    )
    runner.require_manager_drained(snapshot)
    assert runner.post_workload_residency_for_policy(snapshot) == {
        name: 0 for name in runner._POST_WORKLOAD_RESIDENCY_FIELDS
    }
    assert snapshot["lifecycle_route"] == "native_session"
    assert snapshot["cache_policy"] == "shared_prefix"

    final = copy.deepcopy(info)
    final["internal_states"][0]["orbitkv_manager"] = _manager_state(
        active=True
    )
    runner.require_manager_drained(
        runner.manager_snapshot(
            final,
            "final",
            manifest_fingerprint="sha256:manifest",
            binding_fingerprint="sha256:binding",
            manager_input_fingerprint="sha256:manager-input",
            **_SHARED_SESSION_EXPECTED,
            require_activity=True,
        )
    )

    foreign = copy.deepcopy(final)
    foreign["internal_states"][0]["orbitkv_manager"][
        "direct_source_owner"
    ]["allocator_owned"] = False
    with pytest.raises(RuntimeError, match="owner proof failed"):
        runner.manager_snapshot(
            foreign,
            "final",
            manifest_fingerprint="sha256:manifest",
            binding_fingerprint="sha256:binding",
            manager_input_fingerprint="sha256:manager-input",
            **_SHARED_SESSION_EXPECTED,
            require_activity=True,
        )


def test_counter_progress_requires_measured_session_lifecycle() -> None:
    before_info = {
        "internal_states": [
            {
                "page_size": 16,
                "orbitkv_manager": _manager_state(
                    active=True, activity_scale=1
                ),
            }
        ]
    }
    after_info = copy.deepcopy(before_info)
    before = runner.manager_snapshot(
        before_info,
        "after_warmup",
        manifest_fingerprint="sha256:manifest",
        binding_fingerprint="sha256:binding",
        manager_input_fingerprint="sha256:manager-input",
        **_SHARED_SESSION_EXPECTED,
        require_activity=True,
    )
    after = runner.manager_snapshot(
        after_info,
        "after_workload",
        manifest_fingerprint="sha256:manifest",
        binding_fingerprint="sha256:binding",
        manager_input_fingerprint="sha256:manager-input",
        **_SHARED_SESSION_EXPECTED,
        require_activity=True,
    )
    with pytest.raises(RuntimeError, match="did not advance"):
        runner.require_counter_progress(before, after)

    progressed_info = copy.deepcopy(after_info)
    progressed_info["internal_states"][0]["orbitkv_manager"] = _manager_state(
        active=True, activity_scale=2
    )
    progressed = runner.manager_snapshot(
        progressed_info,
        "after_workload",
        manifest_fingerprint="sha256:manifest",
        binding_fingerprint="sha256:binding",
        manager_input_fingerprint="sha256:manager-input",
        **_SHARED_SESSION_EXPECTED,
        require_activity=True,
    )
    runner.require_counter_progress(before, progressed)

    no_fence_info = copy.deepcopy(progressed_info)
    no_fence_info["internal_states"][0]["orbitkv_manager"][
        "batch_counters"
    ]["forward_events"] = 0
    with pytest.raises(RuntimeError, match="session completion activity"):
        runner.manager_snapshot(
            no_fence_info,
            "after_workload",
            manifest_fingerprint="sha256:manifest",
            binding_fingerprint="sha256:binding",
            manager_input_fingerprint="sha256:manager-input",
            **_SHARED_SESSION_EXPECTED,
            require_activity=True,
        )


@pytest.mark.parametrize(
    ("sliding", "field", "alias"),
    (
        (True, "applicable", 1),
        (True, "derived", 0),
        (False, "applicable", 0),
        (False, "derived", 0),
    ),
)
def test_session_swa_activity_rejects_integer_boolean_aliases(
    sliding: bool, field: str, alias: int
) -> None:
    activity = {
        "status": "exposed" if sliding else "not_applicable",
        "applicable": sliding,
        "source": "native_runtime_session",
        "derived": False,
        "swa_retirement_certificates": 0,
        "swa_pages_reclaimed": 0,
        "swa_wrap_events": 0,
        "swa_page_reuse_events": 0,
    }
    activity[field] = alias

    with pytest.raises(RuntimeError, match="SWA activity changed"):
        runner._session_swa_activity(
            activity, "after_workload", sliding=sliding
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda value: value.update(lifecycle_route="canonical_manager"),
            "lifecycle route",
        ),
        (
            lambda value: value.update(cache_policy="request_private"),
            "cache policy",
        ),
        (
            lambda value: value.update(
                lifecycle_authority=value.pop("lifecycle_route")
            ),
            "fields changed",
        ),
        (
            lambda value: value["swa_activity"].update(status="exposed"),
            "SWA activity",
        ),
        (
            lambda value: value["pressure"].update(enabled=True),
            "pressure",
        ),
        (
            lambda value: value["batch_counters"].update(
                prepare_batch_calls=1
            ),
            "counter keys differ",
        ),
    ),
)
def test_manager_snapshot_rejects_session_policy_drift(
    mutate: object, message: str
) -> None:
    manager = _manager_state(active=True)
    assert callable(mutate)
    mutate(manager)
    info = {
        "internal_states": [
            {"page_size": 16, "orbitkv_manager": manager}
        ]
    }
    with pytest.raises(RuntimeError, match=message):
        runner.manager_snapshot(
            info,
            "after_workload",
            manifest_fingerprint="sha256:manifest",
            binding_fingerprint="sha256:binding",
            manager_input_fingerprint="sha256:manager-input",
            **_SHARED_SESSION_EXPECTED,
            require_activity=True,
        )


@pytest.mark.parametrize("mode", ("stock", "manager"))
def test_run_uses_synchronous_engine_and_emits_complete_record(
    monkeypatch: pytest.MonkeyPatch,
    paths: dict[str, Path | None],
    mode: str,
) -> None:
    args = _args(mode, paths)
    if mode == "stock":
        args.manifest = None
        args.library = None
        paths = dict(paths, manifest=None, library=None)
    engine_events = []
    generated = []
    engine_kwargs = []
    readbacks = [0]

    class FakeEngine:
        def __init__(self, **kwargs: object) -> None:
            engine_kwargs.append(dict(kwargs))
            engine_events.append("init")

        def __enter__(self):
            engine_events.append("enter")
            return self

        def __exit__(self, *_args: object) -> None:
            engine_events.append("exit")

        def generate(self, **kwargs: object) -> dict[str, object]:
            call = copy.deepcopy(kwargs)
            generated.append(call)
            rid = call["rid"][0]
            prompt = call["input_ids"][0]
            return {
                "output_ids": [prompt[0] + offset for offset in range(3)],
                "meta_info": {"id": rid, "cached_tokens": 0},
            }

        def get_server_info(self) -> dict[str, object]:
            engine_events.append("readback")
            state = _engine_state(engine_kwargs[0])
            if mode == "manager":
                active = readbacks[0] > 0
                state["orbitkv_manager"] = _manager_state(
                    active=active,
                    activity_scale=max(1, readbacks[0]),
                )
            readbacks[0] += 1
            return {"internal_states": [state]}

        def flush_cache(self) -> SimpleNamespace:
            engine_events.append("flush")
            return SimpleNamespace(success=True)

    root = paths["sglang_root"]
    assert isinstance(root, Path)
    fake_sglang = ModuleType("sglang")
    fake_sglang.Engine = FakeEngine
    fake_sglang.__version__ = "1.2.3"
    fake_sglang.__file__ = str(root / "python/sglang/__init__.py")
    fake_sglang.__path__ = [str(root / "python/sglang")]
    fake_torch = ModuleType("torch")
    fake_srt = ModuleType("sglang.srt")
    fake_srt.__path__ = []
    fake_environ = ModuleType("sglang.srt.environ")
    fake_environ.envs = SimpleNamespace(
        SGLANG_USE_HND_KVCACHE=SimpleNamespace(get=lambda: False)
    )

    for name in tuple(sys.modules):
        if name == "sglang" or name.startswith("sglang."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(os, "environ", os.environ.copy())
    monkeypatch.delenv("SGLANG_PLUGINS", raising=False)
    monkeypatch.setitem(sys.modules, "sglang", fake_sglang)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "sglang.srt", fake_srt)
    monkeypatch.setitem(sys.modules, "sglang.srt.environ", fake_environ)

    monkeypatch.setattr(
        runner,
        "configure_environment",
        lambda _args, _paths: {"SGLANG_USE_HND_KVCACHE": "0"},
    )
    legacy_checks = []
    monkeypatch.setattr(
        runner.runtime_support,
        "reject_legacy_manager_entrypoint",
        lambda: legacy_checks.append(True),
    )
    plugin_checks = []
    monkeypatch.setattr(
        runner, "reject_installed_sglang_plugins", lambda: plugin_checks.append(True)
    )
    monkeypatch.setattr(
        runner,
        "accelerator_provenance",
        lambda module: {
            "device_type": "cuda",
            "device_name": "Generic Accelerator",
            "compute_capability": {"major": 9, "minor": 0},
            "total_memory_bytes": 1024 * 1024 * 1024,
            "runtime_version": "13.0",
            "driver_version": "13.1",
        },
    )
    monkeypatch.setattr(
        runner,
        "verify_source",
        lambda _root, selected: {
            "root": str(_root),
            "release": "v1.2.3",
            "revision": "abc",
            "patch": {
                "status": "applied" if selected == "manager" else "absent",
                "sha256": None,
            },
        },
    )
    monkeypatch.setattr(
        runner,
        "checkpoint_identity",
        lambda _model, load_format: {
            "load_format": load_format,
            "config_sha256": "config",
            "weight_files": [{"name": "weights", "bytes": 1}],
            "weight_bytes": 1,
            "indexed_weights_complete": True,
        },
    )
    if mode == "manager":
        monkeypatch.setattr(
            runner,
            "load_native_session_admission",
            lambda _manifest, _library: {
                "runtime_manifest": {
                    "fingerprint": "sha256:manifest",
                    "token_manager_plan": {"layout": {}},
                },
                "runtime_binding": {
                    "fingerprint": "sha256:binding",
                    "execution_signature": {"page_tokens": 16},
                },
                "manager_input_fingerprint": "sha256:manager-input",
                "lifecycle_route": "native_session",
                "cache_policy": "shared_prefix",
                "class_ids": (0,),
                    "chunk_geometry": None,
            },
        )
        loaded = []
        monkeypatch.setattr(runner, "LoadedLibrary", lambda path: loaded.append(path))

    clock = iter((10.0, 11.0, 20.0, 22.0, 30.0, 34.0))
    monkeypatch.setattr(runner.time, "perf_counter", lambda: next(clock))
    record = runner.run(args, paths)

    assert legacy_checks == [True]
    assert plugin_checks == [True]
    assert engine_events == [
        "init",
        "enter",
        "readback",
        "readback",
        "readback",
        "flush",
        "readback",
        "exit",
    ]
    assert len(generated) == 4
    assert all(set(call) == {"input_ids", "rid", "sampling_params"} for call in generated)
    assert len({tuple(call["input_ids"][0]) for call in generated}) == 4
    assert len({call["rid"][0] for call in generated}) == 4
    assert record["timings"]["iteration_seconds"] == [1.0, 2.0, 4.0]
    assert len(record["outputs"]["warmups"]) == 1
    assert len(record["outputs"]["iterations"]) == 3
    assert record["checkpoint"]["weight_bytes"] == 1
    assert record["accelerator"]["device_type"] == "cuda"
    assert record["outputs"]["aggregate_sha256"] == runner.canonical_json_sha256(
        {
            "warmups": record["outputs"]["warmups"],
            "iterations": record["outputs"]["iterations"],
        }
    )
    assert engine_kwargs[0]["disable_overlap_schedule"] is True
    assert engine_kwargs[0]["disable_cuda_graph"] is True
    assert engine_kwargs[0]["speculative_algorithm"] is None
    assert engine_kwargs[0]["tp_size"] == 1
    if mode == "manager":
        assert engine_kwargs[0]["disable_radix_cache"] is False
        assert engine_kwargs[0]["radix_cache_backend"] == "orbitkv"
        assert record["runtime_manifest"] is not None
        assert record["runtime_binding"] is not None
        assert record["manager"]["post_workload_residency"] == {
            name: 0 for name in runner._POST_WORKLOAD_RESIDENCY_FIELDS
        }
        assert record["manager"]["snapshots"][-1][
            "direct_source_owner"
        ]["owner_is_process_singleton"] is True
    else:
        assert engine_kwargs[0]["disable_radix_cache"] is False
        assert "radix_cache_backend" not in engine_kwargs[0]
        assert record["runtime_manifest"] is None
        assert record["runtime_binding"] is None
        assert record["manager"] is None


def test_stock_rejects_manager_namespace() -> None:
    with pytest.raises(RuntimeError, match="stock run loaded OrbitKV"):
        runner.require_stock_absent(
            {"internal_states": [{"orbitkv_manager": {}}]}, "final"
        )
