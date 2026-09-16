use orbitkv_compiler::lower::KernArtifact;

const MINIMAL_MANIFEST: &str = r#"
{
  "schema_version": 5,
  "model": "orbitkv-next-boundary-test",
  "vars": {"tokens": {"max": 8}},
  "states": {"state": {"bytes_per_token": 16}},
  "buffers": {
    "x": {"dtype": "bf16", "shape": ["tokens", 16], "kind": "input"},
    "y": {"dtype": "bf16", "shape": ["tokens", 16], "kind": "output"}
  },
  "modules": {
    "test": {
      "source": "test.cubin",
      "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    }
  },
  "ops": {
    "step": {
      "params": ["in buffer<bf16>", "out buffer<bf16>", "inout state"],
      "impl": {
        "launches": [{
          "module": "test", "entry": "step",
          "block": [32, 1, 1], "grid": ["tokens", 1, 1]
        }]
      }
    }
  },
  "programs": {
    "decode": {
      "calls": [{
        "op": "step",
        "args": [{"buf": "x"}, {"buf": "y"}, {"state": "state"}]
      }]
    }
  }
}
"#;

#[test]
fn accepts_only_a_manifest_verified_by_pinned_kern() {
    let artifact = KernArtifact::from_json(MINIMAL_MANIFEST).unwrap();
    assert_eq!(artifact.model(), "orbitkv-next-boundary-test");
    assert!(artifact.to_json().contains("\"schema_version\": 5"));
}

#[test]
fn rejects_an_old_schema_before_runtime() {
    let old = MINIMAL_MANIFEST.replace("\"schema_version\": 5", "\"schema_version\": 4");
    let error = KernArtifact::from_json(&old).unwrap_err();
    assert!(error.to_string().contains("unsupported schema_version 4"));
}
