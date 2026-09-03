from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
source_root = REPOSITORY_ROOT / "compat/sglang/bridge/src"
value = str(source_root)
if value not in sys.path:
    sys.path.insert(0, value)
