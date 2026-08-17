use std::path::PathBuf;
use std::process::Command;

fn binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_orbitkv"))
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn run(arguments: &[&str]) -> serde_json::Value {
    let output = output(arguments);
    assert!(
        output.status.success(),
        "orbitkv failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("valid JSON output")
}

fn output(arguments: &[&str]) -> std::process::Output {
    Command::new(binary())
        .args(arguments)
        .output()
        .expect("run orbitkv CLI")
}

#[test]
fn legacy_and_retention_frontends_emit_identical_artifacts() {
    let legacy = root().join("examples/full_swa.json");
    let retention = root().join("examples/full_swa_retention.json");
    for command in ["emit-layout", "emit-sglang-policy"] {
        let legacy = run(&[command, legacy.to_str().unwrap()]);
        let retention = run(&[command, retention.to_str().unwrap()]);
        assert_eq!(legacy, retention);
    }
}

#[test]
fn retention_analysis_reports_derived_window_and_address() {
    let retention = root().join("examples/full_swa_retention.json");
    let report = run(&["analyze-retention", retention.to_str().unwrap()]);
    assert_eq!(report["schema"], "orbitkv.retention-analysis.v1");
    assert_eq!(report["analyses"][0]["inferred"]["kind"], "unbounded");
    assert_eq!(report["analyses"][1]["inferred"]["window_tokens"], 1024);
    assert_eq!(
        report["analyses"][1]["proven_query_key_delta_upper_bound"],
        1023
    );
    assert_eq!(
        report["layout"]["classes"][1]["address"]["period_blocks"],
        65
    );
}

#[test]
fn sink_sliding_relation_emits_two_regions_without_new_retention_kind() {
    let source = root().join("examples/sink_sliding_retention.json");
    let report = run(&["analyze-retention", source.to_str().unwrap()]);
    let regions = &report["analyses"][0]["inferred"]["regions"];
    assert_eq!(regions[0]["label"], "sink");
    assert_eq!(regions[0]["retention"]["kind"], "unbounded");
    assert_eq!(regions[1]["label"], "local");
    assert_eq!(regions[1]["retention"]["window_tokens"], 8);
    assert_eq!(report["layout"]["classes"][0]["address"]["kind"], "pinned");
    assert_eq!(
        report["layout"]["classes"][1]["address"]["kind"],
        "periodic_from"
    );
    assert_eq!(
        report["layout"]["classes"][1]["address"]["period_blocks"],
        3
    );
}

#[test]
fn sink_sliding_sglang_lowering_fails_closed() {
    let source = root().join("examples/sink_sliding_retention.json");
    let output = output(&["emit-sglang-policy", source.to_str().unwrap()]);
    assert!(!output.status.success());
    assert_eq!(output.stdout, b"");
    assert_eq!(
        String::from_utf8_lossy(&output.stderr).trim(),
        "orbitkv: SGLang lowering does not support partitioned block domains"
    );
}

#[test]
fn hf_config_frontend_compiles_hybrid_fixture() {
    let config = root().join("fixtures/gpt-oss-hybrid-tiny/config.json");
    let report = run(&[
        "compile-hf-config",
        config.to_str().unwrap(),
        "--page-tokens",
        "16",
        "--kv-dtype-bytes",
        "2",
    ]);
    assert_eq!(
        report["compilation"]["layer_inference"],
        "explicit_layer_types"
    );
    assert_eq!(report["compilation"]["bytes_per_token_per_layer"], 512);
    assert_eq!(
        report["compilation"]["program"]["states"][0]["name"],
        "full"
    );
    assert_eq!(
        report["compilation"]["program"]["states"][0]["layers"],
        serde_json::json!([1, 3, 5, 7])
    );
    assert_eq!(report["compilation"]["program"]["states"][1]["name"], "swa");
    assert_eq!(
        report["layout"]["classes"][1]["address"]["period_blocks"],
        65
    );
    let legacy = root().join("examples/gpt_oss_hybrid_tiny.json");
    assert_eq!(
        report["layout"],
        run(&["emit-layout", legacy.to_str().unwrap()])
    );
}

#[test]
fn hf_applicability_reports_full_uniform_and_hybrid_models() {
    let cases = [
        (
            "fixtures/qwen2.5-full-tiny/config.json",
            "fallback_all_full",
            "safe_fallback",
            serde_json::json!(["append_only"]),
            0,
        ),
        (
            "fixtures/mistral-uniform-swa-tiny/config.json",
            "architecture_uniform_sliding",
            "uniform_bounded",
            serde_json::json!(["periodic"]),
            49_000,
        ),
        (
            "fixtures/gpt-oss-hybrid-tiny/config.json",
            "explicit_layer_types",
            "hybrid_lifetimes",
            serde_json::json!(["append_only", "periodic"]),
            43_000,
        ),
    ];
    for (path, inference, applicability, layouts, minimum_reduction) in cases {
        let config = root().join(path);
        let report = run(&[
            "analyze-hf-applicability",
            config.to_str().unwrap(),
            "--page-tokens",
            "16",
            "--kv-dtype-bytes",
            "2",
            "--boundary",
            "8192",
        ]);
        assert_eq!(report["schema"], "orbitkv.hf-applicability-compilation.v1");
        assert_eq!(report["compilation"]["layer_inference"], inference);
        assert_eq!(report["applicability"]["applicability"], applicability);
        assert_eq!(report["applicability"]["generated_layouts"], layouts);
        assert!(
            report["applicability"]["static_reduction_percent_milli"]
                .as_u64()
                .unwrap()
                >= minimum_reduction
        );
        assert_eq!(
            report["applicability"]["claim_boundary"][1],
            "not a kernel, scheduler, admission, or end-to-end speedup prediction"
        );
    }
}

#[test]
fn hf_state_plan_emits_uniform_swa_execution_contract() {
    let config = root().join("fixtures/mistral-uniform-swa-tiny/config.json");
    let report = run(&[
        "compile-hf-state-plan",
        config.to_str().unwrap(),
        "--page-tokens",
        "1",
        "--kv-dtype-bytes",
        "2",
        "--boundary",
        "8192",
        "--max-running-requests",
        "4",
        "--chunked-prefill-tokens",
        "2048",
        "--eviction-interval",
        "128",
        "--decode-headroom-tokens",
        "32",
    ]);
    assert_eq!(report["schema"], "orbitkv.hf-state-plan.v3");
    assert_eq!(report["sglang_lowering"]["status"], "enabled");
    assert_eq!(report["sglang_lowering"]["kind"], "uniform_swa");
    assert_eq!(
        report["sglang_lowering"]["contract"]["architecture"],
        "MistralForCausalLM"
    );
    assert_eq!(
        report["sglang_lowering"]["contract"]["kernel_window_left"],
        4095
    );
    assert_eq!(
        report["sglang_lowering"]["contract"]["maximum_running_requests"],
        4
    );
    assert_eq!(
        report["sglang_lowering"]["contract"]["minimum_pool_tokens"],
        19_077
    );
    assert_eq!(
        report["sglang_lowering"]["contract"]["physical_backend"],
        "direct_periodic"
    );
    assert_eq!(
        report["layout"]["classes"][0]["address"]["kind"],
        "periodic"
    );
}

#[test]
fn hf_state_plan_emits_paged_periodic_execution_contract() {
    let config = root().join("fixtures/mistral-uniform-swa-tiny/config.json");
    let report = run(&[
        "compile-hf-state-plan",
        config.to_str().unwrap(),
        "--page-tokens",
        "16",
        "--kv-dtype-bytes",
        "2",
        "--boundary",
        "8192",
        "--max-running-requests",
        "4",
        "--chunked-prefill-tokens",
        "2048",
        "--eviction-interval",
        "128",
        "--decode-headroom-tokens",
        "32",
    ]);
    assert_eq!(report["schema"], "orbitkv.hf-state-plan.v3");
    assert_eq!(
        report["sglang_lowering"]["contract"]["physical_backend"],
        "paged_periodic"
    );
    assert_eq!(
        report["sglang_lowering"]["contract"]["minimum_pool_tokens"],
        19_152
    );
    assert_eq!(
        report["sglang_lowering"]["contract"]["logical_index_tokens"],
        32_768
    );
}

#[test]
fn hf_physical_optimizer_selects_capacity_plan() {
    let config = root().join("fixtures/gpt-oss-hybrid-tiny/config.json");
    let report = run(&[
        "compile-hf-physical-plan",
        config.to_str().unwrap(),
        "--page-tokens",
        "16",
        "--kv-dtype-bytes",
        "2",
        "--available-kv-bytes",
        "120881152",
        "--max-running-requests",
        "8",
        "--attention-dp-size",
        "1",
        "--chunked-prefill-tokens",
        "2048",
        "--workload-requests",
        "8",
        "--prompt-tokens",
        "6000",
        "--decode-tokens",
        "32",
        "--candidate-intervals",
        "16,32,64,128",
        "--max-reclamation-calls",
        "4",
        "--min-admitted-requests",
        "8",
        "--objective",
        "capacity",
    ]);
    assert_eq!(
        report["physical_plan"]["selected_eviction_interval_tokens"],
        32
    );
    let candidates = report["physical_plan"]["candidates"].as_array().unwrap();
    assert_eq!(candidates[0]["eviction_interval_tokens"], 16);
    assert_eq!(candidates[0]["feasible"], false);
    assert_eq!(
        candidates[0]["rejection_reasons"][0],
        "estimated reclamation calls 5 exceed maximum 4"
    );
    assert_eq!(
        report["physical_plan"]["selected"]["cost"]["admission_waves"],
        1
    );
}

#[test]
fn chunked_local_relation_emits_resettable_arena() {
    let source = root().join("examples/chunked_local_retention.json");
    let report = run(&["analyze-retention", source.to_str().unwrap()]);
    assert_eq!(report["analyses"][0]["inferred"]["kind"], "chunked");
    assert_eq!(report["analyses"][0]["inferred"]["chunk_tokens"], 16);
    assert_eq!(
        report["layout"]["classes"][0]["address"]["kind"],
        "resettable_arena"
    );
    assert_eq!(
        report["layout"]["classes"][0]["address"]["blocks_per_epoch"],
        4
    );
    assert_eq!(
        report["layout"]["classes"][0]["retirement"]["kind"],
        "epoch_end"
    );
    let output = output(&["emit-sglang-policy", source.to_str().unwrap()]);
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("SGLang lowering does not support retention class")
    );
}

#[test]
fn multi_scale_heads_report_lifetime_normalization_savings() {
    let source = root().join("examples/multi_scale_head_windows.json");
    let report = run(&["analyze-lifetime-normalization", source.to_str().unwrap()]);
    assert_eq!(report["schema"], "orbitkv.lifetime-normalization.v1");
    assert_eq!(report["normalized_classes"].as_array().unwrap().len(), 3);
    assert_eq!(report["savings_percent_milli"], 42_105);
    assert_eq!(report["retention_amplification_milli"], 1727);
    assert_eq!(
        report["normalized_classes"][0]["kv_head_range"],
        serde_json::json!({"start": 0, "end_exclusive": 8})
    );
    let output = output(&["emit-sglang-policy", source.to_str().unwrap()]);
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("SGLang lowering does not support retention class")
    );
}
