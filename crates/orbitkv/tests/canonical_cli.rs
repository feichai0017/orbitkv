use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

use orbitkv::{RUNTIME_MANIFEST_VERSION, RuntimeCapability, RuntimeManifest};

static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

struct TempJson(PathBuf);

impl TempJson {
    fn new(contents: &[u8]) -> Self {
        let suffix = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "orbitkv-canonical-cli-{}-{suffix}.json",
            std::process::id()
        ));
        std::fs::write(&path, contents).unwrap();
        Self(path)
    }
}

impl Drop for TempJson {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn run(arguments: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_orbitkv"))
        .args(arguments)
        .output()
        .unwrap()
}

#[test]
fn compile_plan_accepts_only_the_canonical_source_shape() {
    let plan = TempJson::new(
        br#"{
          "page_tokens": 16,
          "classes": [{
            "name": "swa",
            "layers": [0, 1],
            "retention": "sliding",
            "bytes_per_token_per_layer": 2048,
            "window_tokens": 18
          }]
        }"#,
    );
    let output = run(&["compile-plan", plan.0.to_str().unwrap()]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["page_tokens"], 16);
    assert_eq!(value["classes"][0]["slot_count"], 3);

    let removed_shape = TempJson::new(
        br#"{
          "schema": "orbitkv.retention-ir.v1",
          "page_tokens": 16,
          "states": []
        }"#,
    );
    let output = run(&["compile-plan", removed_shape.0.to_str().unwrap()]);
    assert!(!output.status.success());
}

#[test]
fn hf_token_manager_plan_is_directly_consumable_by_compile_plan() {
    let config = TempJson::new(
        br#"{
          "architectures": ["GenericCausalLM"],
          "num_hidden_layers": 2,
          "sliding_window": 18,
          "use_sliding_window": true,
          "num_key_value_heads": 8,
          "head_dim": 64
        }"#,
    );
    let output = run(&[
        "compile-hf-token-manager-plan",
        config.0.to_str().unwrap(),
        "--page-tokens",
        "16",
        "--kv-dtype-bytes",
        "2",
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let plan: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(plan["page_tokens"], 16);
    assert_eq!(plan["classes"][0]["name"], "swa");
    assert_eq!(plan["classes"][0]["retention"], "sliding");
    assert_eq!(plan["classes"][0]["window_tokens"], 18);

    let generated = TempJson::new(&output.stdout);
    let compiled = run(&["compile-plan", generated.0.to_str().unwrap()]);
    assert!(
        compiled.status.success(),
        "{}",
        String::from_utf8_lossy(&compiled.stderr)
    );
}

#[test]
fn hf_token_manager_plan_emits_full_and_hybrid_classes() {
    let full = TempJson::new(
        br#"{
          "architectures": ["GenericCausalLM"],
          "num_hidden_layers": 2,
          "sliding_window": 131072,
          "use_sliding_window": false,
          "num_key_value_heads": 4,
          "head_dim": 128
        }"#,
    );
    let full_output = run(&[
        "compile-hf-token-manager-plan",
        full.0.to_str().unwrap(),
        "--page-tokens",
        "16",
        "--kv-dtype-bytes",
        "2",
    ]);
    assert!(
        full_output.status.success(),
        "{}",
        String::from_utf8_lossy(&full_output.stderr)
    );
    let full_plan: serde_json::Value = serde_json::from_slice(&full_output.stdout).unwrap();
    assert_eq!(full_plan["classes"][0]["retention"], "full");
    assert_eq!(full_plan["classes"][0]["layers"], serde_json::json!([0, 1]));

    let hybrid = TempJson::new(
        br#"{
          "architectures": ["GptOssForCausalLM"],
          "num_hidden_layers": 4,
          "layer_types": [
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention"
          ],
          "sliding_window": 128,
          "num_key_value_heads": 8,
          "head_dim": 64
        }"#,
    );
    let hybrid_output = run(&[
        "compile-hf-token-manager-plan",
        hybrid.0.to_str().unwrap(),
        "--page-tokens",
        "16",
        "--kv-dtype-bytes",
        "2",
    ]);
    assert!(
        hybrid_output.status.success(),
        "{}",
        String::from_utf8_lossy(&hybrid_output.stderr)
    );
    let hybrid_plan: serde_json::Value = serde_json::from_slice(&hybrid_output.stdout).unwrap();
    assert_eq!(hybrid_plan["classes"].as_array().unwrap().len(), 2);
    assert_eq!(hybrid_plan["classes"][0]["retention"], "full");
    assert_eq!(
        hybrid_plan["classes"][0]["layers"],
        serde_json::json!([1, 3])
    );
    assert_eq!(hybrid_plan["classes"][1]["retention"], "sliding");
    assert_eq!(
        hybrid_plan["classes"][1]["layers"],
        serde_json::json!([0, 2])
    );
}

#[test]
fn hf_token_manager_plan_rejects_unproven_layer_semantics() {
    let config = TempJson::new(
        br#"{
          "architectures": ["UnknownForCausalLM"],
          "num_hidden_layers": 2,
          "sliding_window": 18,
          "num_key_value_heads": 8,
          "head_dim": 64
        }"#,
    );
    let output = run(&[
        "compile-hf-token-manager-plan",
        config.0.to_str().unwrap(),
        "--page-tokens",
        "16",
        "--kv-dtype-bytes",
        "2",
    ]);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("does not prove"));
}

#[test]
fn hybrid_gdn_hf_state_input_uses_the_consumable_storage_schema() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let config = root.join("fixtures/hybrid-fixed-state/config.json");
    let common = [
        config.to_str().unwrap(),
        "--page-tokens",
        "16",
        "--kv-dtype-bytes",
        "2",
    ];

    let state_input_output = run(&[
        "compile-hf-state-input",
        common[0],
        common[1],
        common[2],
        common[3],
        common[4],
    ]);
    assert!(
        state_input_output.status.success(),
        "{}",
        String::from_utf8_lossy(&state_input_output.stderr)
    );
    let state_input: serde_json::Value =
        serde_json::from_slice(&state_input_output.stdout).unwrap();
    assert_eq!(state_input["states"][0]["storage"]["kind"], "token_kv");
    assert_eq!(
        state_input["states"][2]["storage"]["state_bytes_per_layer"],
        36_864
    );
    assert!(state_input["states"][0].get("backend").is_none());
    let generated_state_input = TempJson::new(&state_input_output.stdout);
    let compiled_state_input = run(&[
        "compile-state-plan",
        generated_state_input.0.to_str().unwrap(),
    ]);
    assert!(
        compiled_state_input.status.success(),
        "{}",
        String::from_utf8_lossy(&compiled_state_input.stderr)
    );
}

#[test]
fn hybrid_gdn_hf_frontend_compiles_state_and_token_manager_plans() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let config = root.join("fixtures/hybrid-fixed-state/config.json");
    let common = [
        config.to_str().unwrap(),
        "--page-tokens",
        "16",
        "--kv-dtype-bytes",
        "2",
    ];
    let state_output = run(&[
        "compile-hf-state-plan",
        common[0],
        common[1],
        common[2],
        common[3],
        common[4],
    ]);
    assert!(
        state_output.status.success(),
        "{}",
        String::from_utf8_lossy(&state_output.stderr)
    );
    let state_plan: serde_json::Value = serde_json::from_slice(&state_output.stdout).unwrap();
    assert_eq!(state_plan["schema"], "orbitkv.attention-state-plan.v1");
    assert_eq!(state_plan["states"].as_array().unwrap().len(), 3);
    assert_eq!(
        state_plan["states"][0]["layers"],
        serde_json::json!([3, 7, 11, 15, 19, 23])
    );
    assert_eq!(
        state_plan["states"][0]["backend"]["components"][0]["bytes_per_token_per_layer"],
        1_024
    );
    assert_eq!(
        state_plan["states"][0]["backend"]["components"][1]["bytes_per_token_per_layer"],
        1_024
    );
    assert_eq!(
        state_plan["states"][1]["backend"]["state_bytes_per_layer"],
        1_048_576
    );
    assert_eq!(
        state_plan["states"][1]["backend"]["checkpoint_slots_per_request"],
        2
    );
    assert_eq!(
        state_plan["states"][1]["layers"].as_array().unwrap().len(),
        18
    );
    assert_eq!(
        state_plan["states"][2]["backend"]["state_bytes_per_layer"],
        36_864
    );
    assert_eq!(state_plan["states"][2]["backend"]["kernel_width"], 4);

    let manager_output = run(&[
        "compile-hf-token-manager-plan",
        common[0],
        common[1],
        common[2],
        common[3],
        common[4],
    ]);
    assert!(
        manager_output.status.success(),
        "{}",
        String::from_utf8_lossy(&manager_output.stderr)
    );
    let manager: serde_json::Value = serde_json::from_slice(&manager_output.stdout).unwrap();
    assert_eq!(manager["classes"].as_array().unwrap().len(), 1);
    assert_eq!(manager["classes"][0]["name"], "full_attention_kv");
    assert_eq!(
        manager["classes"][0]["layers"],
        serde_json::json!([3, 7, 11, 15, 19, 23])
    );
    assert_eq!(manager["classes"][0]["bytes_per_token_per_layer"], 2_048);
    assert!(manager_output.stderr.is_empty());
    let generated = TempJson::new(&manager_output.stdout);
    let compiled = run(&["compile-plan", generated.0.to_str().unwrap()]);
    assert!(
        compiled.status.success(),
        "{}",
        String::from_utf8_lossy(&compiled.stderr)
    );
}

#[test]
fn renamed_hybrid_hf_frontend_compiles_every_public_artifact() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source =
        std::fs::read_to_string(root.join("fixtures/hybrid-fixed-state/config.json")).unwrap();
    let mut config: serde_json::Value = serde_json::from_str(&source).unwrap();
    config["architectures"] = serde_json::json!(["RenamedArchitecture"]);
    config["model_type"] = serde_json::json!("renamed_envelope");
    config["text_config"]["model_type"] = serde_json::json!("renamed_text");
    let config = TempJson::new(&serde_json::to_vec(&config).unwrap());

    for command in [
        "compile-hf-state-plan",
        "compile-hf-token-manager-plan",
        "compile-hf-runtime-manifest",
    ] {
        let output = run(&[
            command,
            config.0.to_str().unwrap(),
            "--page-tokens",
            "16",
            "--kv-dtype-bytes",
            "2",
        ]);
        assert!(
            output.status.success(),
            "{command}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn compile_state_plan_separates_mla_recurrent_and_convolution_contracts() {
    let plan = TempJson::new(
        br#"{
          "page_tokens": 16,
          "states": [
            {
              "name": "mla",
              "layers": [0, 1],
              "storage": {
                "kind": "latent_kv",
                "latent_bytes_per_token_per_layer": 1024,
                "rope_bytes_per_token_per_layer": 128,
                "retention": "full",
                "window_tokens": null
              }
            },
            {
              "name": "gdn",
              "layers": [2],
              "storage": {
                "kind": "recurrent",
                "family": "gdn",
                "state_bytes_per_layer": 4096,
                "checkpoint_slots_per_request": 2
              }
            },
            {
              "name": "shortconv",
              "layers": [2],
              "storage": {
                "kind": "convolution",
                "state_bytes_per_layer": 2048,
                "kernel_width": 4,
                "checkpoint_slots_per_request": 2
              }
            }
          ]
        }"#,
    );
    let output = run(&["compile-state-plan", plan.0.to_str().unwrap()]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let compiled: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(compiled["schema"], "orbitkv.attention-state-plan.v1");
    assert_eq!(compiled["states"][0]["backend"]["kind"], "token_slots");
    assert_eq!(
        compiled["states"][1]["backend"]["kind"],
        "recurrent_checkpoints"
    );
    assert_eq!(compiled["states"][2]["backend"]["kind"], "convolution_ring");

    let manager = run(&["compile-state-manager-plan", plan.0.to_str().unwrap()]);
    assert!(
        manager.status.success(),
        "{}",
        String::from_utf8_lossy(&manager.stderr)
    );
    let manager: serde_json::Value = serde_json::from_slice(&manager.stdout).unwrap();
    assert_eq!(manager["page_tokens"], 16);
    assert_eq!(manager["classes"].as_array().unwrap().len(), 1);
    assert_eq!(manager["classes"][0]["name"], "mla");
    assert_eq!(manager["classes"][0]["bytes_per_token_per_layer"], 1_152);
    assert_eq!(manager["classes"][0]["storage"], "latent_kv");
    assert_eq!(manager["classes"][0]["components"][0]["name"], "latent");
    assert_eq!(
        manager["classes"][0]["components"][1]["bytes_per_token_per_layer"],
        128
    );
    let generated = TempJson::new(&serde_json::to_vec(&manager).unwrap());
    let compiled_manager = run(&["compile-plan", generated.0.to_str().unwrap()]);
    assert!(
        compiled_manager.status.success(),
        "{}",
        String::from_utf8_lossy(&compiled_manager.stderr)
    );
}

#[test]
fn state_manager_plan_rejects_a_checkpoint_only_model() {
    let plan = TempJson::new(
        br#"{
          "page_tokens": 16,
          "states": [{
            "name": "mamba",
            "layers": [0],
            "storage": {
              "kind": "recurrent",
              "family": "mamba",
              "state_bytes_per_layer": 4096,
              "checkpoint_slots_per_request": 2
            }
          }]
        }"#,
    );
    let output = run(&["compile-state-manager-plan", plan.0.to_str().unwrap()]);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("no token-addressable state"));
}

#[test]
fn latent_kv_examples_compile_to_the_same_manager_plan() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let state_plan = root.join("examples/latent-kv-attention-state-plan.json");
    let manager_plan = root.join("examples/latent-kv-token-manager-plan.json");
    let projected = run(&["compile-state-manager-plan", state_plan.to_str().unwrap()]);
    assert!(
        projected.status.success(),
        "{}",
        String::from_utf8_lossy(&projected.stderr)
    );
    let expected: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manager_plan).unwrap()).unwrap();
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&projected.stdout).unwrap(),
        expected
    );
    let compiled = run(&["compile-plan", manager_plan.to_str().unwrap()]);
    assert!(
        compiled.status.success(),
        "{}",
        String::from_utf8_lossy(&compiled.stderr)
    );
}

#[test]
fn removed_cli_commands_have_no_compatibility_aliases() {
    for command in [
        "compile",
        "compile-hf-config",
        "compile-hf-manager-plan",
        "serve-dense-runtime",
    ] {
        let output = run(&[command]);
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("usage:"));
    }
}

#[test]
fn usage_discovers_canonical_runtime_manifest_commands() {
    let output = run(&["unknown"]);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("compile-runtime-manifest <state-plan.json>"));
    assert!(stderr.contains("compile-retention-runtime-manifest <retention-ir.json>"));
    assert!(stderr.contains("compile-hf-runtime-manifest <config.json>"));
    assert!(!stderr.contains("bind-runtime-manifest"));
    assert!(!stderr.contains("check-runtime-manifest"));
    assert!(stderr.contains("canonical executable artifact"));
    assert!(stderr.contains("same canonical artifact"));
    assert!(stderr.contains("supported HF config into the canonical artifact"));
}

#[test]
fn retention_runtime_manifest_cli_emits_region_aware_canonical_manifest() {
    let program = TempJson::new(
        br#"{
          "schema": "orbitkv.retention-ir.v1",
          "page_tokens": 4,
          "states": [{
            "name": "attention",
            "layers": [0],
            "bytes_per_token_per_layer": 128,
            "may_read": {
              "op": "or",
              "terms": [
                {"op": "less_than", "lhs": {"op": "key_position"}, "rhs": {"op": "constant", "value": 4}},
                {"op": "less_than", "lhs": {"op": "sub", "lhs": {"op": "query_position"}, "rhs": {"op": "key_position"}}, "rhs": {"op": "constant", "value": 8}}
              ]
            }
          }]
        }"#,
    );
    let output = run(&[
        "compile-retention-runtime-manifest",
        program.0.to_str().unwrap(),
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let manifest = RuntimeManifest::from_json(&output.stdout).unwrap();
    assert_eq!(manifest.version, RUNTIME_MANIFEST_VERSION);
    assert_eq!(
        serde_json::to_value(&manifest).unwrap()["token_manager_plan"]["layout"]["classes"][0]["address"]
            ["kind"],
        "pinned"
    );
    assert_eq!(
        serde_json::to_value(&manifest).unwrap()["token_manager_plan"]["layout"]["classes"][1]["address"]
            ["kind"],
        "periodic_from"
    );
}

#[test]
fn runtime_manifest_cli_emits_one_validated_executable_artifact() {
    let plan = TempJson::new(
        br#"{
          "page_tokens": 16,
          "states": [
            {
              "name": "full",
              "layers": [0],
              "storage": {
                "kind": "token_kv",
                "key_bytes_per_token_per_layer": 256,
                "value_bytes_per_token_per_layer": 256,
                "retention": "full"
              }
            },
            {
              "name": "state",
              "layers": [1],
              "storage": {
                "kind": "recurrent",
                "family": "linear_attention",
                "state_bytes_per_layer": 4096,
                "checkpoint_slots_per_request": 2
              }
            }
          ]
        }"#,
    );
    let output = run(&["compile-runtime-manifest", plan.0.to_str().unwrap()]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty());
    assert!(output.stdout.ends_with(b"\n"));

    let manifest = RuntimeManifest::from_json(&output.stdout).unwrap();
    assert_eq!(manifest.schema, "orbitkv.runtime-manifest");
    assert_eq!(manifest.version, RUNTIME_MANIFEST_VERSION);
    assert!(manifest.token_manager_plan.is_some());
    assert!(
        manifest
            .capability_requirements
            .contains(&RuntimeCapability::RecurrentState)
    );
}

#[test]
fn runtime_manifest_cli_preserves_fixed_only_state() {
    let plan = TempJson::new(
        br#"{
          "page_tokens": 16,
          "states": [{
            "name": "state",
            "layers": [0],
            "storage": {
              "kind": "recurrent",
              "family": "mamba",
              "state_bytes_per_layer": 4096,
              "checkpoint_slots_per_request": 2
            }
          }]
        }"#,
    );
    let output = run(&["compile-runtime-manifest", plan.0.to_str().unwrap()]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let manifest = RuntimeManifest::from_json(&output.stdout).unwrap();
    assert!(manifest.token_manager_plan.is_none());
    assert_eq!(
        manifest.attention_state_plan.as_ref().unwrap().states.len(),
        1
    );
}

#[test]
fn hf_runtime_manifest_cli_supports_token_only_and_heterogeneous_configs() {
    let token_only = TempJson::new(
        br#"{
          "architectures": ["GenericCausalLM"],
          "num_hidden_layers": 2,
          "sliding_window": 18,
          "use_sliding_window": true,
          "num_key_value_heads": 8,
          "head_dim": 64
        }"#,
    );
    let token_output = run(&[
        "compile-hf-runtime-manifest",
        token_only.0.to_str().unwrap(),
        "--page-tokens",
        "16",
        "--kv-dtype-bytes",
        "2",
    ]);
    assert!(
        token_output.status.success(),
        "{}",
        String::from_utf8_lossy(&token_output.stderr)
    );
    let token_manifest = RuntimeManifest::from_json(&token_output.stdout).unwrap();
    assert_eq!(
        token_manifest
            .attention_state_plan
            .as_ref()
            .unwrap()
            .states
            .len(),
        1
    );
    assert!(token_manifest.token_manager_plan.is_some());

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let config = root.join("fixtures/hybrid-fixed-state/config.json");
    let hybrid_output = run(&[
        "compile-hf-runtime-manifest",
        config.to_str().unwrap(),
        "--page-tokens",
        "16",
        "--kv-dtype-bytes",
        "2",
    ]);
    assert!(
        hybrid_output.status.success(),
        "{}",
        String::from_utf8_lossy(&hybrid_output.stderr)
    );
    let hybrid_manifest = RuntimeManifest::from_json(&hybrid_output.stdout).unwrap();
    assert_eq!(
        hybrid_manifest
            .attention_state_plan
            .as_ref()
            .unwrap()
            .states
            .len(),
        3
    );
    assert!(
        hybrid_manifest
            .capability_requirements
            .contains(&RuntimeCapability::ConvolutionState)
    );
}

#[test]
fn runtime_manifest_cli_rejects_invalid_arguments_and_input() {
    let plan = TempJson::new(br#"{"page_tokens":16,"states":[]}"#);
    let extra = run(&[
        "compile-runtime-manifest",
        plan.0.to_str().unwrap(),
        "unexpected",
    ]);
    assert!(!extra.status.success());
    assert!(String::from_utf8_lossy(&extra.stderr).contains("unexpected argument"));

    let invalid = run(&["compile-runtime-manifest", plan.0.to_str().unwrap()]);
    assert!(!invalid.status.success());
    assert!(String::from_utf8_lossy(&invalid.stderr).contains("must not be empty"));
}

#[test]
fn canonical_runtime_manifest_cli_output_is_byte_stable() {
    let plan = TempJson::new(
        br#"{
          "page_tokens": 16,
          "states": [{
            "name": "full", "layers": [0],
            "storage": {
              "kind": "token_kv",
              "key_bytes_per_token_per_layer": 64,
              "value_bytes_per_token_per_layer": 64,
              "retention": "full", "window_tokens": null
            }
          }]
        }"#,
    );
    let first = run(&["compile-runtime-manifest", plan.0.to_str().unwrap()]);
    let second = run(&["compile-runtime-manifest", plan.0.to_str().unwrap()]);
    assert!(first.status.success());
    assert!(second.status.success());
    assert_eq!(first.stdout, second.stdout);
    assert_eq!(
        RuntimeManifest::from_json(&first.stdout).unwrap().version,
        RUNTIME_MANIFEST_VERSION
    );
}
