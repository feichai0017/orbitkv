"""Test-only engine plugin for real P/D fault gates; never a production entry point."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch


def register_sglang() -> None:
    from sglang.srt.disaggregation.lifecycle import register_pd_transfer_observer

    from orbitkv.sglang import pd, plugin

    root = Path(os.environ["ORBITKV_TEST_PD_FAULT_DIR"])
    parent = pd.SGLangTentTransferEngine

    class GatedTent(parent):
        def batch_transfer_sync(self, session_id, buffers, destinations, lengths):
            arm = root / "arm.json"
            claimed = root / "claimed.json"
            try:
                arm.rename(claimed)
            except FileNotFoundError:
                return super().batch_transfer_sync(session_id, buffers, destinations, lengths)
            case = json.loads(claimed.read_text())
            case_root = root / case["name"]
            first = min(int(lengths[0]), 4096)
            status = super().batch_transfer_sync(session_id, buffers[:1], destinations[:1], [first])
            assert status == 0, "the fault gate needs an actual successful GPU WRITE"
            (case_root / "first-write.json").write_text(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "bytes": first,
                        "endpoint": session_id,
                        "total_bytes": sum(lengths),
                        "source": int(buffers[0]),
                        "destination": int(destinations[0]),
                    }
                )
            )
            deadline = time.monotonic() + 30
            while not (case_root / "release").exists():
                assert time.monotonic() < deadline, "fault gate was not released"
                time.sleep(0.01)
            if case["name"] == "partial":
                return -1
            sources, targets, sizes = list(buffers), list(destinations), list(lengths)
            sources[0] += first
            targets[0] += first
            sizes[0] -= first
            if sizes[0] == 0:
                sources, targets, sizes = sources[1:], targets[1:], sizes[1:]
            return super().batch_transfer_sync(session_id, sources, targets, sizes)

    # Select the test payload through the same public factory registration.
    # Only this isolated test plugin imports the instrumented constructor.
    with patch.object(pd, "SGLangTentTransferEngine", GatedTent):
        plugin.register()

    def record(event):
        value = {**asdict(event), "pid": os.getpid()}
        encoded = (json.dumps(value) + "\n").encode()
        fd = os.open(root / "events.jsonl", os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        try:
            os.write(fd, encoded)
        finally:
            os.close(fd)

    register_pd_transfer_observer("orbitkv-test-evidence", record)
