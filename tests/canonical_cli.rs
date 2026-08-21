use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

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
fn hf_manager_plan_is_directly_consumable_by_compile_plan() {
    let config = TempJson::new(
        br#"{
          "architectures": ["MistralForCausalLM"],
          "num_hidden_layers": 2,
          "sliding_window": 18,
          "num_key_value_heads": 8,
          "head_dim": 64
        }"#,
    );
    let output = run(&[
        "compile-hf-manager-plan",
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
fn hf_manager_plan_emits_full_and_hybrid_classes() {
    let full = TempJson::new(
        br#"{
          "architectures": ["Qwen2ForCausalLM"],
          "num_hidden_layers": 2,
          "sliding_window": 131072,
          "use_sliding_window": false,
          "num_key_value_heads": 4,
          "head_dim": 128
        }"#,
    );
    let full_output = run(&[
        "compile-hf-manager-plan",
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
        "compile-hf-manager-plan",
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
fn hf_manager_plan_rejects_unproven_layer_semantics() {
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
        "compile-hf-manager-plan",
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
fn removed_cli_commands_have_no_compatibility_aliases() {
    for command in ["compile", "compile-hf-config", "serve-dense-runtime"] {
        let output = run(&[command]);
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("usage:"));
    }
}
