"""Installed-package provenance for release gates; runnable in isolated Python."""

import base64
import hashlib
import importlib
import importlib.metadata as metadata
import importlib.util
import json
import os
import sys
import sysconfig
from pathlib import Path

ENGINE_VERSIONS = {"vllm": "0.30.0", "sglang": "0.5.20"}
NATIVE_LIBRARIES = {"libtent_shared.so", "libmooncake_common.so", "libasio.so"}


def isolated_environment(environment: dict[str, str], directory: Path) -> dict[str, str]:
    env = {
        key: value
        for key, value in environment.items()
        if not key.startswith(("PYTHON", "ORBITKV_"))
    }
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        part
        for part in env.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        if part
        and "/.orbitkv/" not in part
        and not part.endswith("/python/orbitkv")
        and not any((Path(part) / name).exists() for name in NATIVE_LIBRARIES)
    )
    env.update(
        ORBITKV_CACHE_SCOPE=directory.name,
        MC_FORCE_TCP="1",
        VLLM_USE_V2_MODEL_RUNNER="0",
        VLLM_BATCH_INVARIANT="1",
        HF_HUB_OFFLINE="1",
        PYTHONDONTWRITEBYTECODE="1",
        TORCHINDUCTOR_CACHE_DIR=str(directory / "inductor"),
        TRITON_CACHE_DIR=str(directory / "triton"),
        VLLM_CACHE_ROOT=str(directory / "vllm"),
    )
    return env


def distribution_snapshot(distribution, package: str) -> dict:
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    assert not direct_url.get("dir_info", {}).get("editable", False), direct_url
    files = distribution.files
    assert files, f"{distribution.metadata['Name']} has no installed RECORD"
    hashes = {}
    recorded = set()
    for item in files:
        path = Path(distribution.locate_file(item)).resolve()
        recorded.add(path)
        if item.hash is None and item.name != "RECORD":
            assert path.suffix not in {".py", ".so"}, f"Runtime file has no RECORD hash: {path}"
            continue
        assert path.is_file(), f"Missing installed file: {path}"
        data = path.read_bytes()
        if item.hash:
            actual = base64.urlsafe_b64encode(hashlib.new(item.hash.mode, data).digest())
            assert actual.rstrip(b"=").decode() == item.hash.value, (
                f"Installed file differs from RECORD: {path}"
            )
        hashes[str(path)] = hashlib.sha256(data).hexdigest()
    package_root = Path(distribution.locate_file(package)).resolve()
    assert package_root.is_dir(), package_root
    unrecorded = [
        str(path)
        for path in package_root.rglob("*")
        if path.is_file() and path.suffix in {".py", ".so"} and path.resolve() not in recorded
    ]
    assert not unrecorded, f"Unrecorded runtime files: {unrecorded}"
    return {
        "name": distribution.metadata["Name"],
        "version": distribution.version,
        "package_root": str(package_root),
        "files": hashes,
    }


def installation_snapshot(engine: str, initialize_transfer: bool = False) -> dict:
    purelib = Path(sysconfig.get_path("purelib")).resolve()
    owners = metadata.packages_distributions().get("orbitkv", [])
    assert len(owners) == 1, f"Expected one OrbitKV distribution, got {owners}"
    distribution = metadata.distribution(owners[0])
    assert distribution.metadata["Name"] in {"orbitkv-llm", "orbitkv-llm-cu13"}
    engine_distribution = metadata.distribution(engine)
    assert engine_distribution.version == ENGINE_VERSIONS[engine], engine_distribution.version
    snapshots = {}
    for package, installed in (("orbitkv", distribution), (engine, engine_distribution)):
        spec = importlib.util.find_spec(package)
        assert spec and spec.origin, f"Missing installed package: {package}"
        origin = Path(spec.origin).resolve()
        assert origin.is_relative_to(purelib), f"Package outside this environment: {origin}"
        snapshot = distribution_snapshot(installed, package)
        assert origin.parent == Path(snapshot["package_root"]), origin
        snapshots[package] = snapshot
    for group, target in (
        ("vllm.general_plugins", "orbitkv.vllm.plugin:register"),
        ("sglang.srt.plugins", "orbitkv.sglang.plugin:register"),
    ):
        entries = list(metadata.entry_points(group=group, name="orbitkv"))
        assert len(entries) == 1 and entries[0].value == target, entries
        assert entries[0].dist.metadata["Name"] == distribution.metadata["Name"], entries
    import orbitkv

    for plugin in ("orbitkv.vllm.plugin", "orbitkv.sglang.plugin"):
        importlib.import_module(plugin)
    assert "torch" not in sys.modules and "orbitkv.orbitkv" not in sys.modules, (
        "Plugin discovery initialized the GPU/native runtime"
    )
    assert orbitkv.__version__ == distribution.version
    loaded = []
    if initialize_transfer:
        transfer = orbitkv.MooncakeTransferEngine(bind_host="127.0.0.1")
        loaded = sorted(
            {
                str(Path(line.split()[-1]).resolve())
                for line in Path("/proc/self/maps").read_text().splitlines()
                if Path(line.split()[-1]).name in NATIVE_LIBRARIES
            }
        )
        assert {Path(path).name for path in loaded} == NATIVE_LIBRARIES, loaded
        assert all(Path(path).parent == Path(orbitkv.__file__).parent for path in loaded), loaded
        del transfer
    return {
        "distributions": snapshots,
        "native_libraries": loaded,
        "python": sys.executable,
        "python_libdir": sysconfig.get_config_var("LIBDIR"),
        "pythonhome": os.environ.get("PYTHONHOME"),
        "pythonpath": os.environ.get("PYTHONPATH"),
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
    }


if __name__ == "__main__":
    print(json.dumps(installation_snapshot(sys.argv[1], "--native" in sys.argv[2:])))
