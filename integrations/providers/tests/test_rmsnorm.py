import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aletheia_providers.rmsnorm import exact_domain, semantics  # noqa: E402


class RmsNormContractTests(unittest.TestCase):
    def test_semantics_changes_with_geometry(self):
        self.assertNotEqual(semantics(1, 4096, 1e-6, "bfloat16"), semantics(4, 4096, 1e-6, "bfloat16"))

    def test_workload_domain_is_an_exact_bucket(self):
        domain = exact_domain(4, 2048)
        self.assertEqual(domain["batch"], {"min": 4, "max": 4})
        self.assertEqual(domain["context_tokens"], {"min": 2048, "max": 2048})


if __name__ == "__main__":
    unittest.main()
