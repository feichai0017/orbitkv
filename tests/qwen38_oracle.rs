use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use kern_manifest::Verified;
use kern_manifest::types::Manifest;
use orbitkv_compiler::lower::{Qwen38Declarations, Qwen38Fp8WeightPlan, WeightTransformKind};
use orbitkv_compiler::model::Qwen38Contract;
use orbitkv_compiler::oracle::{ManifestInventory, OracleError, Qwen38Fp8Checkpoint, Qwen38Oracle, RowsInventory};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn qwen_manifest() -> PathBuf {
    root().join("examples/qwen3.8-27b.json")
}

fn dflash_manifest() -> PathBuf {
    root().join("examples/qwen3.8-27b-dflash2.json")
}

fn verified(path: &Path) -> Verified {
    Verified::from_json(&fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn imported_qwen_manifest_matches_the_pinned_contract() {
    let oracle = Qwen38Oracle::from_path(qwen_manifest()).unwrap();
    oracle.validate_contract().unwrap();
    let report = oracle.report();

    assert_eq!(report.contract_model, Qwen38Contract::MODEL);
    assert_eq!(report.oracle_role, "bf16_execution_oracle");
    assert!(report.not_authoritative_for.contains(&"official_fp8_weight_dtype".to_string()));
    assert_eq!(report.inventory.buffers.weights, 659);
    assert_eq!(report.inventory.buffers.weight_dtypes.get("bf16"), Some(&659));
    assert_eq!(report.inventory.buffers.weight_segments, 851);
    assert_eq!(report.inventory.modules, 22);
    assert_eq!(report.inventory.ops, 40);
    assert_eq!(report.state_packing.physical_state, "gdn");
    assert_eq!(report.state_packing.logical_states, ["gdn.recurrent", "gdn.convolution"]);
    assert_eq!(report.state_packing.bytes_per_sequence, 154_140_672);
    assert!(oracle.skeleton_differences().is_empty());
    let declarations = Qwen38Declarations::lower();
    assert_eq!(
        serde_json::to_value(declarations.variables).unwrap(),
        serde_json::to_value(&oracle.manifest().vars).unwrap()
    );
    assert_eq!(
        serde_json::to_value(declarations.states).unwrap(),
        serde_json::to_value(&oracle.manifest().states).unwrap()
    );
}

#[test]
fn generated_fp8_text_contract_has_the_expected_families() {
    let tensors = Qwen38Fp8Checkpoint::expected_text_tensors();
    let fp8 = tensors.values().filter(|tensor| tensor.dtype == "F8_E4M3").count();
    let scales = tensors.keys().filter(|name| name.ends_with("_scale_inv")).count();
    let bf16 = tensors.values().filter(|tensor| tensor.dtype == "BF16").count();

    assert_eq!(tensors.len(), 1_251);
    assert_eq!(fp8, 400);
    assert_eq!(scales, 400);
    assert_eq!(bf16, 851);
    assert_eq!(tensors["model.language_model.layers.0.mlp.gate_proj.weight"].shape, [17_408, 5_120]);
    assert_eq!(tensors["model.language_model.layers.0.mlp.gate_proj.weight_scale_inv"].shape, [136, 40]);
}

#[test]
fn fp8_physical_weight_plan_consumes_every_text_tensor_once() {
    let checkpoint = Qwen38Fp8Checkpoint::expected_text_tensors();
    let plan = Qwen38Fp8WeightPlan::lower();
    plan.validate(&checkpoint).unwrap();

    assert_eq!(plan.bound.len(), 915);
    assert_eq!(plan.fp8_buffers(), 256);
    assert_eq!(plan.derived.len(), 465);
    assert_eq!(plan.transforms.len(), 465);
    assert_eq!(
        plan.transforms.iter().filter(|transform| transform.kind == WeightTransformKind::CastBf16ToF32).count(),
        304
    );
    assert_eq!(
        plan.transforms.iter().filter(|transform| transform.kind == WeightTransformKind::AddOneBf16ToF32).count(),
        161
    );
}

#[test]
#[ignore = "requires the official Qwen3.8-27B-FP8 checkpoint headers"]
fn official_fp8_checkpoint_matches_the_generated_tensor_contract() {
    let root = std::env::var_os("ORBITKV_QWEN38_FP8_DIR").expect("set ORBITKV_QWEN38_FP8_DIR");
    let report = Qwen38Fp8Checkpoint::inspect(root).unwrap();
    assert_eq!(report.text_tensors, 1_251);
    assert_eq!(report.fp8_matrices, 400);
    assert_eq!(report.scale_tensors, 400);
    assert_eq!(report.bf16_tensors, 851);
}

#[test]
fn an_unaligned_physical_state_layout_is_rejected_by_the_manifest_verifier() {
    let text = fs::read_to_string(qwen_manifest()).unwrap();
    let mut manifest = Manifest::from_json(&text).unwrap();
    manifest.states.get_mut("gdn").unwrap().bytes_per_seq -= 4_096;

    let OracleError::InvalidManifest(error) = Qwen38Oracle::from_json(&manifest.to_json()).unwrap_err() else {
        panic!("expected a manifest verification error");
    };
    assert!(error.contains("do not divide its"));
}

#[test]
fn an_aligned_but_wrong_physical_state_layout_is_rejected_by_the_model_contract() {
    let text = fs::read_to_string(qwen_manifest()).unwrap();
    let mut manifest = Manifest::from_json(&text).unwrap();
    let line = manifest.states["gdn"].bytes_per_seq / Qwen38Contract::GATED_DELTA_LAYERS as u64;
    manifest.states.get_mut("gdn").unwrap().bytes_per_seq -= line;

    let oracle = Qwen38Oracle::from_json(&manifest.to_json()).unwrap();
    let OracleError::Contract(errors) = oracle.validate_contract().unwrap_err() else {
        panic!("expected a model-contract error");
    };
    assert!(errors.iter().any(|error| error.contains("gdn.bytes_per_sequence")));
}

#[test]
fn a_manifest_is_structurally_identical_to_itself() {
    let oracle = Qwen38Oracle::from_path(qwen_manifest()).unwrap();
    assert!(oracle.diff(oracle.manifest()).is_empty());
}

#[test]
fn structural_diff_names_a_changed_program() {
    let oracle = Qwen38Oracle::from_path(qwen_manifest()).unwrap();
    let mut candidate = Manifest::from_json(&fs::read_to_string(qwen_manifest()).unwrap()).unwrap();
    candidate.programs.get_mut("decode").unwrap().graph = false;
    let candidate = Verified::from_json(&candidate.to_json()).unwrap();

    let diff = oracle.diff(&candidate);
    assert!(diff.programs.changed.contains(&"decode".to_string()));
    let decode = diff.program_details.iter().find(|program| program.name == "decode").unwrap();
    assert!(decode.graph_changed);
    assert!(!decode.calls_changed);
}

#[test]
fn dflash2_is_a_verified_speculative_oracle() {
    let manifest = verified(&dflash_manifest());
    let inventory = ManifestInventory::from_verified(&manifest);

    assert_eq!(inventory.model, "qwen3.8-27b-dflash2");
    assert_eq!(inventory.states.len(), 3);
    assert_eq!(inventory.buffers.weights, 730);
    assert_eq!(inventory.modules, 33);
    assert_eq!(inventory.ops, 58);
    let round = &inventory.programs["round"];
    assert_eq!(round.calls, 1_252);
    assert!(round.graph);
    assert_eq!(round.batch.as_ref().unwrap().groups, 128);
    assert_eq!(round.batch.as_ref().unwrap().rows, RowsInventory::Constant(8));
}

#[test]
fn cli_validates_and_self_diffs_the_qwen_oracle() {
    let binary = env!("CARGO_BIN_EXE_orbitkv");
    let validate = Command::new(binary).args(["oracle", "validate"]).current_dir(root()).output().unwrap();
    assert!(validate.status.success(), "{}", String::from_utf8_lossy(&validate.stderr));
    assert!(String::from_utf8_lossy(&validate.stdout).contains("\"physical_state\": \"gdn\""));

    let diff = Command::new(binary)
        .args(["oracle", "diff", "examples/qwen3.8-27b.json"])
        .current_dir(root())
        .output()
        .unwrap();
    assert!(diff.status.success(), "{}", String::from_utf8_lossy(&diff.stderr));
    assert!(String::from_utf8_lossy(&diff.stdout).contains("\"program_details\": []"));
}
