import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aletheia_providers import probe as provider_probe  # noqa: E402


class ProbeTests(unittest.TestCase):
    def test_reports_module_outside_pinned_source(self):
        with tempfile.TemporaryDirectory() as directory:
            module = types.ModuleType("fake_provider")
            module.__file__ = str(Path(directory, "fake_provider.py"))
            sys.modules["fake_provider"] = module
            provider_probe.EXPECTED["fake_provider"] = (ROOT / "not-the-module", "fake-distribution")
            try:
                report = provider_probe.probe("fake_provider")
            finally:
                provider_probe.EXPECTED.pop("fake_provider")
                sys.modules.pop("fake_provider")
        self.assertFalse(report["from_pinned_source"])
        self.assertIsNone(report["provenance"])


if __name__ == "__main__":
    unittest.main()
