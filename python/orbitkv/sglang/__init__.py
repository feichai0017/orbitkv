"""SGLang integration contracts for OrbitKV.

The executable HiCache backend will live here. The current module deliberately
exports only stable configuration and pool mapping helpers until the shared
host-page registration path is implemented.
"""

from orbitkv.sglang.config import OrbitKVSGLangConfig
from orbitkv.sglang.pools import component_for_pool

__all__ = ["OrbitKVSGLangConfig", "component_for_pool"]
