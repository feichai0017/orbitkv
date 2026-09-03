from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from orbitkv_sglang import cli


def _config(*, chunk_tokens: int | None = None) -> SimpleNamespace:
    chunked_class = (
        None
        if chunk_tokens is None
        else SimpleNamespace(chunk_tokens=chunk_tokens)
    )
    return SimpleNamespace(page_tokens=16, chunked_class=chunked_class)


def _profile(cache_policy: str) -> SimpleNamespace:
    return SimpleNamespace(cache_policy=cache_policy)


def test_compile_uses_built_binary_and_validates_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compiler = tmp_path / "orbitkv"
    compiler.write_text("binary", encoding="utf-8")
    compiler.chmod(0o755)
    model = tmp_path / "model"
    model.mkdir()
    config = model / "config.json"
    config.write_text("{}", encoding="utf-8")
    output = tmp_path / "runtime-manifest.json"
    manifest = {"schema": "fixture"}
    binding = {"schema": "orbitkv.runtime-binding"}
    payload = json.dumps(manifest).encode() + b"\n"
    calls: list[tuple[str, ...]] = []

    def run(command, **kwargs):
        calls.append(tuple(command))
        assert kwargs == {
            "check": True,
            "capture_output": True,
            "timeout": 120,
        }
        return SimpleNamespace(
            stdout=payload if len(calls) == 1 else json.dumps(binding).encode()
        )

    validated: list[object] = []
    monkeypatch.setattr(cli.subprocess, "run", run)
    monkeypatch.setattr(
        cli,
        "validate_runtime_manifest",
        lambda raw: validated.append(raw) or raw,
    )
    monkeypatch.setattr(cli, "runtime_binding_from_manifest", lambda _raw: binding)

    result = cli.compile_manifest(
        str(model),
        output,
        environ={"ORBITKV_BINARY": str(compiler)},
    )

    assert result == output
    assert output.read_bytes() == payload
    assert validated == [manifest]
    assert binding["schema"] == "orbitkv.runtime-binding"
    assert calls[0] == (
        str(compiler),
        "compile-hf-runtime-manifest",
        str(config),
        "--page-tokens",
        "16",
        "--kv-dtype-bytes",
        "2",
    )
    assert calls[1][0:2] == (str(compiler), "bind-runtime-manifest")
    assert Path(calls[1][2]).parent == tmp_path
    assert len(calls[1]) == 3
    assert not Path(calls[1][2]).exists()


def test_compile_does_not_publish_when_rust_binding_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compiler = tmp_path / "orbitkv"
    compiler.write_text("binary", encoding="utf-8")
    compiler.chmod(0o755)
    model = tmp_path / "config.json"
    model.write_text("{}", encoding="utf-8")
    output = tmp_path / "runtime-manifest.json"
    output.write_text("old", encoding="utf-8")
    outputs = iter((b'{"schema": "fixture"}', b'{"binding": "rust"}'))
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=next(outputs)),
    )
    monkeypatch.setattr(cli, "validate_runtime_manifest", lambda raw: raw)
    monkeypatch.setattr(
        cli, "runtime_binding_from_manifest", lambda _raw: {"binding": "python"}
    )

    with pytest.raises(RuntimeError, match="different SGLang runtime bindings"):
        cli.compile_manifest(
            str(model),
            output,
            environ={"ORBITKV_BINARY": str(compiler)},
        )
    assert output.read_text(encoding="utf-8") == "old"


def test_compile_publishes_manifest_that_loads_with_a_runtime_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compiler = tmp_path / "orbitkv"
    compiler.write_text("binary", encoding="utf-8")
    compiler.chmod(0o755)
    model = tmp_path / "config.json"
    model.write_text("{}", encoding="utf-8")
    library = tmp_path / "liborbitkv_ffi.so"
    library.write_bytes(b"library")
    output = tmp_path / "runtime-manifest.json"
    manifest = {"schema": "orbitkv.runtime-manifest"}
    binding = {"schema": "orbitkv.runtime-binding"}
    outputs = iter((json.dumps(manifest).encode(), json.dumps(binding).encode()))
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=next(outputs)),
    )
    monkeypatch.setattr(cli, "validate_runtime_manifest", lambda raw: raw)
    monkeypatch.setattr(cli, "runtime_binding_from_manifest", lambda _raw: binding)

    cli.compile_manifest(
        str(model),
        output,
        environ={"ORBITKV_BINARY": str(compiler)},
    )

    def load(environment):
        assert json.loads(Path(environment["ORBITKV_RUNTIME_MANIFEST"]).read_text()) == manifest
        assert environment["ORBITKV_LIBRARY"] == str(library)
        return SimpleNamespace(runtime_binding=binding)

    monkeypatch.setattr(cli, "load_config", load)
    loaded = cli.load_config(
        {
            "ORBITKV_RUNTIME_MANIFEST": str(output),
            "ORBITKV_LIBRARY": str(library),
        }
    )
    assert loaded.runtime_binding == binding


def test_build_compile_command_defaults() -> None:
    assert cli.build_compile_command(Path("orbitkv"), Path("config.json")) == (
        "orbitkv",
        "compile-hf-runtime-manifest",
        "config.json",
        "--page-tokens",
        "16",
        "--kv-dtype-bytes",
        "2",
    )


def test_compile_does_not_publish_invalid_compiler_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compiler = tmp_path / "orbitkv"
    compiler.write_text("binary", encoding="utf-8")
    compiler.chmod(0o755)
    model = tmp_path / "config.json"
    model.write_text("{}", encoding="utf-8")
    output = tmp_path / "runtime-manifest.json"
    output.write_text("old", encoding="utf-8")
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=b"{}"),
    )

    def reject(_raw):
        raise ValueError("invalid fixture")

    monkeypatch.setattr(cli, "validate_runtime_manifest", reject)
    with pytest.raises(RuntimeError, match="invalid runtime manifest"):
        cli.compile_manifest(
            str(model),
            output,
            environ={"ORBITKV_BINARY": str(compiler)},
        )
    assert output.read_text(encoding="utf-8") == "old"


@pytest.mark.parametrize(
    ("cache_policy", "has_disable_radix"),
    (("shared_prefix", False), ("request_private", True)),
)
def test_serve_command_uses_safe_contract_and_admitted_radix_policy(
    cache_policy: str, has_disable_radix: bool
) -> None:
    command = cli.build_serve_command(
        "org/model",
        _config(),
        _profile(cache_policy),
        ("--", "--port", "31000"),
        python_executable="/python",
    )

    assert command[:5] == (
        "/python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        "org/model",
    )
    for required in (
        "--disable-overlap-schedule",
        "--disable-cuda-graph",
        "--radix-cache-backend",
        "--page-size",
        "--kv-cache-dtype",
        "--max-running-requests",
    ):
        assert required in command
    assert ("--disable-radix-cache" in command) is has_disable_radix
    assert command[-2:] == ("--port", "31000")


def test_chunked_serve_command_uses_compiled_chunk_geometry() -> None:
    command = cli.build_serve_command(
        "model",
        _config(chunk_tokens=3072),
        _profile("request_private"),
    )
    assert command[command.index("--chunked-prefill-size") + 1] == "3072"
    assert command[command.index("--max-prefill-tokens") + 1] == "3072"
    assert command[command.index("--prefill-max-requests") + 1] == "1"


def test_serve_rejects_overrides_of_locked_contract_options() -> None:
    with pytest.raises(ValueError, match="fixed by the OrbitKV runtime contract"):
        cli.build_serve_command(
            "model",
            _config(),
            _profile("shared_prefix"),
            ("--page-size=32",),
        )


def test_prepare_serve_launch_validates_and_scrubs_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "product"
    root.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    library = tmp_path / "liborbitkv_ffi.so"
    library.write_bytes(b"library")
    resolved_config = _config()
    resolved_config.runtime_manifest_path = manifest.resolve()
    resolved_config.library_path = library.resolve()
    events: list[tuple[str, object]] = []

    def validate_source(value):
        events.append(("source", value))
        return root.resolve()

    def load_config(environment):
        events.append(("config", dict(environment)))
        return resolved_config

    def load_library(value):
        events.append(("library", value))

    def admit(config):
        events.append(("admit", config))
        return _profile("request_private")

    monkeypatch.setattr(cli, "validate_patched_source", validate_source)
    monkeypatch.setattr(cli, "load_config", load_config)
    monkeypatch.setattr(cli, "LoadedLibrary", load_library)
    monkeypatch.setattr(cli, "admit_product_takeover_config", admit)

    launch = cli.prepare_serve_launch(
        model="org/model",
        manifest=manifest,
        library=library,
        sglang_root=root,
        environ={
            "PATH": "/bin",
            "PYTHONPATH": "/prior",
            "ORBITKV_STALE": "must-be-removed",
        },
        python_executable="/python",
    )

    assert [name for name, _value in events] == [
        "source",
        "config",
        "library",
        "admit",
    ]
    assert launch.cache_policy == "request_private"
    assert launch.environment["ORBITKV_RUNTIME_MANIFEST"] == str(
        manifest.resolve()
    )
    assert launch.environment["ORBITKV_LIBRARY"] == str(library.resolve())
    assert launch.environment["ORBITKV_SGLANG_ROOT"] == str(root.resolve())
    assert launch.environment["PYTHONPATH"] == (
        str(root.resolve() / "python") + os.pathsep + "/prior"
    )
    assert "ORBITKV_STALE" not in launch.environment
    assert launch.environment["SGLANG_USE_HND_KVCACHE"] == "0"


def test_dry_run_prints_without_exec(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    launch = cli.ServeLaunch(
        command=("/python", "-m", "sglang.launch_server"),
        environment={
            "ORBITKV_RUNTIME_MANIFEST": "/manifest.json",
            "ORBITKV_LIBRARY": "/lib.so",
            "ORBITKV_SGLANG_ROOT": "/source",
            "PYTHONPATH": "/source/python",
            **cli._RUNTIME_ENVIRONMENT,
        },
        cache_policy="shared_prefix",
    )
    monkeypatch.setattr(cli, "prepare_serve_launch", lambda **_kwargs: launch)
    monkeypatch.setattr(
        cli.os, "execvpe", lambda *_args: pytest.fail("dry run executed server")
    )

    assert (
        cli.main(
            [
                "serve",
                "--model",
                "model",
                "--manifest",
                "manifest",
                "--library",
                "library",
                "--sglang-root",
                "source",
                "--print-command",
            ]
        )
        == 0
    )
    assert "python -m sglang.launch_server" in capsys.readouterr().out
