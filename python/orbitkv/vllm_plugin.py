"""
vLLM general plugin for OrbitKV.

Registered via the ``vllm.general_plugins`` entry-point so that
``load_general_plugins()`` executes it in **every** vLLM process
(API-server, engine-core, workers).

The sole purpose is to register OrbitKV KV connectors in the
``KVConnectorFactory`` registry.  The registration is lazy —
connector modules are only imported when the class is actually
looked up — so this module adds negligible overhead.
"""

import contextlib


def register() -> None:
    from vllm.distributed.kv_transfer.kv_connector.factory import (
        KVConnectorFactory,
    )

    with contextlib.suppress(ValueError):
        KVConnectorFactory.register_connector(
            "OrbitKVConnector",
            "orbitkv.connector",
            "OrbitKVConnector",
        )
    with contextlib.suppress(ValueError):
        KVConnectorFactory.register_connector(
            "NoopKVConnector",
            "orbitkv.connector",
            "NoopKVConnector",
        )
    with contextlib.suppress(ValueError):
        KVConnectorFactory.register_connector(
            "PdConnector",
            "orbitkv.pd_connector",
            "PdConnector",
        )
    with contextlib.suppress(ValueError):
        KVConnectorFactory.register_connector(
            "PdDecodeConnector",
            "orbitkv.pd_connector",
            "PdDecodeConnector",
        )
    with contextlib.suppress(ValueError):
        KVConnectorFactory.register_connector(
            "PdPrefillConnector",
            "orbitkv.pd_connector",
            "PdPrefillConnector",
        )
    with contextlib.suppress(ValueError):
        KVConnectorFactory.register_connector(
            "NixlConnector",
            "orbitkv.nixl_connector",
            "NixlConnector",
        )
    with contextlib.suppress(ValueError):
        KVConnectorFactory.register_connector(
            "NixlPullConnector",
            "orbitkv.nixl_connector",
            "NixlPullConnector",
        )
    with contextlib.suppress(ValueError):
        KVConnectorFactory.register_connector(
            "NixlPushConnector",
            "orbitkv.nixl_connector",
            "NixlPushConnector",
        )
    with contextlib.suppress(ValueError):
        KVConnectorFactory.register_connector(
            "OrbitKVNixlConnector",
            "orbitkv.nixl_connector",
            "OrbitKVNixlConnector",
        )
    with contextlib.suppress(ValueError):
        KVConnectorFactory.register_connector(
            "OrbitKVNixlPullConnector",
            "orbitkv.nixl_connector",
            "OrbitKVNixlPullConnector",
        )
