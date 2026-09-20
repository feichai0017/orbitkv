"""SGLang integration contracts and lazily loaded cache backends."""

from orbitkv.sglang.config import OrbitKVSGLangConfig
from orbitkv.sglang.pools import component_for_pool

__all__ = ["OrbitKVSGLangConfig", "component_for_pool"]
