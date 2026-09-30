"""Plugin discovery must work without loading an unselected cache runtime."""

import os
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_unselected_plugins_do_not_load_native_or_gpu_modules(engine):
    program = textwrap.dedent(
        """
        import importlib.abc
        import sys
        from types import ModuleType, SimpleNamespace

        class NoRuntime(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in {
                    "torch", "orbitkv.orbitkv", "orbitkv.sglang.linker",
                    "orbitkv.sglang.layout", "orbitkv.vllm.connector",
                    "orbitkv.vllm.pd",
                    "orbitkv.sglang.completion", "sglang.srt.disaggregation",
                }:
                    raise AssertionError(f"unselected runtime imported: {fullname}")

        sys.meta_path.insert(0, NoRuntime())

        def module(name, **attrs):
            parts = name.split(".")
            for end in range(1, len(parts) + 1):
                prefix = ".".join(parts[:end])
                if prefix not in sys.modules:
                    item = ModuleType(prefix)
                    item.__path__ = []
                    sys.modules[prefix] = item
            vars(sys.modules[name]).update(attrs)

        if sys.argv[1] == "vllm":
            registered = {}

            class Factory:
                @staticmethod
                def register_connector(name, path, cls):
                    if name in registered:
                        raise ValueError("duplicate connector")
                    registered[name] = (path, cls)

            module("vllm.distributed.kv_transfer.kv_connector.factory",
                   KVConnectorFactory=Factory)
            from orbitkv.vllm.plugin import register
            register()
            assert set(registered) == {
                "OrbitKVConnector", "PdDecodeConnector", "PdPrefillConnector"
            }
            try:
                register()
            except ValueError as error:
                assert str(error) == "duplicate connector"
            else:
                raise AssertionError("registration conflict was swallowed")
        else:
            backends = {}
            hooks = {}
            module("sglang.srt.mem_cache.registry",
                   register_radix_cache_backend=lambda name, factory:
                       backends.update({name: factory}))
            module("sglang.srt.plugins.hook_registry",
                   HookRegistry=SimpleNamespace(register=lambda name, fn, kind:
                       hooks.update({name: (fn, kind)})),
                   HookType=SimpleNamespace(BEFORE="before", AFTER="after", AROUND="around"))
            module("sglang.srt.runtime_context",
                   get_memory=lambda: SimpleNamespace(radix_cache_backend="native"))
            from orbitkv.sglang.plugin import register
            register()
            assert set(backends) == {"orbitkv"}
            assert set(hooks) == {
                "sglang.srt.managers.tp_worker.TpModelWorker.init_cuda_graphs",
                "sglang.srt.managers.schedule_policy.PrefillAdder.add_one_req",
            }

            from orbitkv.sglang.events import initialize_layer_counter
            from orbitkv.sglang.admission import admit_request, enqueue_request
            initialize_layer_counter(object())
            req = object()
            scheduler = SimpleNamespace(waiting_queue=[])
            calls = []

            def enqueue(owner, request):
                calls.append(request)
                owner.waiting_queue.append(request)
                return "queued"

            assert enqueue_request(enqueue, scheduler, req) == "queued"
            assert calls == [req]
            assert admit_request(lambda *args: "admitted", object(), req) == "admitted"
        """
    )
    environment = dict(
        os.environ, ORBITKV_SGLANG_TENT="0", ORBITKV_PREPARE_REQUESTS="0", ORBITKV_QUEUE_WARMUP="0"
    )
    result = subprocess.run(
        [sys.executable, "-c", program, engine],
        env=environment,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
