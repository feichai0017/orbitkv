from __future__ import annotations

from enum import Enum

import pytest

from orbitkv.sglang import OrbitKVSGLangConfig, component_for_pool


class Pool(Enum):
    KV = "kv"
    MAMBA = "mamba"


def test_pool_mapping_does_not_require_sglang_import():
    assert component_for_pool(Pool.KV) == "attention_kv"
    assert component_for_pool(Pool.MAMBA) == "recurrent_checkpoint"
    assert component_for_pool("vendor_state") == "opaque:vendor_state"


def test_sglang_config_requires_shared_host_allocator():
    config = OrbitKVSGLangConfig.from_extra_config({"endpoint": "unix:///tmp/orbitkv.sock"})
    assert config.allocator == "shm"
    with pytest.raises(ValueError, match="allocator='shm'"):
        OrbitKVSGLangConfig.from_extra_config({"allocator": "default"})
