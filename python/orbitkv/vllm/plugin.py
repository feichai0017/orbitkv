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


def register() -> None:
    from vllm.distributed.kv_transfer.kv_connector.factory import (
        KVConnectorFactory,
    )

    KVConnectorFactory.register_connector(
        "OrbitKVConnector",
        "orbitkv.vllm",
        "OrbitKVConnector",
    )
