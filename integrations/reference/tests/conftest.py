from __future__ import annotations

import sys
from pathlib import Path


INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = INTEGRATION_ROOT.parents[1]
RUNTIME_SOURCE = REPOSITORY_ROOT / "python" / "orbitkv-runtime" / "src"
REFERENCE_SOURCE = INTEGRATION_ROOT / "src"
sys.path[:0] = [str(RUNTIME_SOURCE), str(REFERENCE_SOURCE)]
