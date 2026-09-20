"""Cache identities must survive relocation and reject changes to computation."""

from __future__ import annotations

import copy
import shutil

import pytest

from orbitkv.identity import artifact_identity, model_identity, state_namespace


@pytest.fixture(autouse=True)
def _clean_identity_environment(monkeypatch):
    monkeypatch.delenv("ORBITKV_MODEL_FINGERPRINT", raising=False)
    monkeypatch.delenv("ORBITKV_CACHE_SCOPE", raising=False)


def test_artifact_contents_survive_relocation_but_detect_same_path_replacement(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"original")
    (model / "tokenizer.json").write_text('{"vocab": [1]}')
    identity = model_identity(str(model))
    replica = tmp_path / "replica"
    shutil.copytree(model, replica)
    assert model_identity(str(replica)) == identity

    # Same path and byte length must not hide a weight replacement.
    (model / "model.safetensors").write_bytes(b"replaced")
    changed_weights = model_identity(str(model))
    assert changed_weights != identity
    (model / "tokenizer.json").write_text('{"vocab": [2]}')
    assert model_identity(str(model)) != changed_weights


def test_hub_identity_requires_a_pinned_commit_and_tracks_separate_tokenizer():
    with pytest.raises(ValueError, match="immutable"):
        model_identity("org/model", revision="main")
    with pytest.raises(ValueError, match="immutable"):
        model_identity("org/model", revision="a" * 40, tokenizer="org/tokenizer")
    original = model_identity("org/model", revision="a" * 40)
    assert model_identity("org/model", revision="b" * 40) != original
    assert (
        model_identity(
            "org/model", revision="a" * 40, tokenizer="org/tokenizer", tokenizer_revision="b" * 40
        )
        != original
    )


def test_explicit_fingerprint_is_checked_and_does_not_read_model_files(monkeypatch):
    monkeypatch.setenv("ORBITKV_MODEL_FINGERPRINT", "name-is-not-an-identity")
    with pytest.raises(ValueError, match="SHA-256"):
        model_identity("org/model")
    monkeypatch.setenv("ORBITKV_MODEL_FINGERPRINT", "c" * 64)
    assert model_identity("org/model") == {"deployment": "c" * 64}


def test_namespace_binds_engine_computation_format_and_scope(monkeypatch):
    kwargs = {
        "engine": "vllm",
        "engine_version": "0.29.0",
        "model": {"deployment": "a" * 64},
        "computation": {"rope": 10000, "quantization": "none"},
        "representation": {"dtype": "bf16", "block_tokens": 16, "tp": [0, 2]},
    }
    identity = state_namespace(**kwargs)
    assert len(identity) == len("orbitkv:v1:") + 64
    reordered = copy.deepcopy(kwargs)
    reordered["computation"] = dict(reversed(list(kwargs["computation"].items())))
    assert state_namespace(**reordered) == identity
    for key, value in (
        ("engine", "sglang"),
        ("engine_version", "0.30.0"),
        ("model", {"deployment": "b" * 64}),
        ("computation", {"rope": 100000, "quantization": "none"}),
        ("representation", {"dtype": "fp8", "block_tokens": 16, "tp": [0, 2]}),
        ("representation", {"dtype": "bf16", "block_tokens": 32, "tp": [0, 2]}),
        ("representation", {"dtype": "bf16", "block_tokens": 16, "tp": [1, 2]}),
    ):
        assert state_namespace(**(kwargs | {key: value})) != identity, key
    monkeypatch.setenv("ORBITKV_CACHE_SCOPE", "other-tenant")
    assert state_namespace(**kwargs) != identity


def test_unknown_local_artifact_set_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="no model/tokenizer artifacts"):
        artifact_identity(str(tmp_path))
