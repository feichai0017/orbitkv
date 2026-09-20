"""Stable source checkout locations for tests and their subprocesses."""

from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = TESTS_ROOT.parent
REPO_ROOT = PYTHON_ROOT.parent
