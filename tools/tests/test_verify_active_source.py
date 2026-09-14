from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import verify_active_source as verifier


class SourceBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.crate = self.root / "crates/component"
        self.source = self.crate / "src/lib.rs"
        self.source.parent.mkdir(parents=True)
        (self.crate / "Cargo.toml").write_text('[package]\nname = "component"\n')
        self.target = self.crate / "tests/unit/state.rs"
        self.target.parent.mkdir(parents=True)
        self.target.write_text("// Private tests\n")
        patcher = patch.object(verifier, "ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def check(self, source):
        return verifier.check_rust_layout(self.source, source, production=True)

    def test_private_named_bridge_remains_inside_owning_crate(self):
        source = '#[cfg(test)]\n#[path = "../tests/unit/state.rs"]\npub(crate) mod state_tests;'
        self.assertEqual(self.check(source), [])

    def test_unguarded_bridge_is_rejected(self):
        self.assertTrue(self.check('#[path = "../tests/unit/state.rs"]\nmod tests;'))

    def test_bridge_cannot_escape_its_test_tree(self):
        outside = self.crate / "state.rs"
        outside.write_text("// Not in tests\n")
        self.assertTrue(self.check('#[cfg(test)]\n#[path = "../state.rs"]\nmod tests;'))

    def test_missing_bridge_target_is_rejected(self):
        self.target.unlink()
        self.assertTrue(self.check('#[cfg(test)]\n#[path = "../tests/unit/state.rs"]\nmod tests;'))

    def test_inline_regressions_do_not_belong_in_production(self):
        self.assertTrue(self.check('#[cfg(test)]\nmod tests { #[test] fn regression() {} }'))

    def test_target_specific_alias_does_not_hide_dependency_direction(self):
        manifest = {"target": {"cfg(unix)": {"dependencies": {
            "alias": {"package": "orbitkv-engine", "workspace": True}
        }}}}
        dependencies = verifier.direct_dependencies(manifest)
        self.assertEqual(dependencies, {"orbitkv-engine"})
        self.assertTrue(dependencies.intersection(verifier.LAYER_DEPENDENCIES)
                        - verifier.LAYER_DEPENDENCIES["orbitkv-compiler"])


if __name__ == "__main__":
    unittest.main()
