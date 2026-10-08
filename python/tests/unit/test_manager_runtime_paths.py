"""Manager test launches keep source and installed package modes distinct."""

from tests.support.cache_manager import manager_pythonpath
from tests.support.paths import PYTHON_ROOT


def test_manager_pythonpath_defaults_to_source_then_inherited_paths():
    paths = manager_pythonpath(
        ["/caller/python", "/installed/site"],
        ["/installed/site", "/system/site"],
        None,
    )
    assert paths == (
        str(PYTHON_ROOT),
        "/caller/python",
        "/installed/site",
        "/system/site",
    )


def test_manager_pythonpath_installed_mode_excludes_source_and_inherited_paths():
    paths = manager_pythonpath(
        [str(PYTHON_ROOT), "/unrelated/source"],
        ["/installed/site", "/system/site"],
        ("/installed/site",),
    )
    assert paths == ("/installed/site", "/system/site")
    assert str(PYTHON_ROOT) not in paths
