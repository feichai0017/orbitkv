use std::collections::BTreeMap;

use orbitkv_compiler::compiler::provider::{
    DEEPGEMM_REVISION, DeepGemmSm90Capability, Fp8ProjectionShape, HistoricalDeepGemmSourceEvidence,
    LOW_LATENCY_ROW_BUCKETS, PACKED_ACTIVATION_ABI, QualificationState, render_aot_source,
};
use orbitkv_compiler::lower::Qwen38Fp8WeightPlan;

#[test]
fn qwen38_fp8_weights_collapse_to_five_model_neutral_shape_families() {
    let families = Qwen38Fp8WeightPlan::lower().fp8_projection_families().unwrap();
    let counts: BTreeMap<_, _> = families
        .iter()
        .map(|family| ((family.shape.output_features, family.shape.input_features), family.uses))
        .collect();

    assert_eq!(families.iter().map(|family| family.uses).sum::<usize>(), 256);
    assert_eq!(
        counts,
        BTreeMap::from([
            ((5_120, 6_144), 64),
            ((5_120, 17_408), 64),
            ((14_336, 5_120), 16),
            ((16_384, 5_120), 48),
            ((34_816, 5_120), 64),
        ])
    );
}

#[test]
fn every_qwen38_shape_and_low_row_bucket_has_a_legal_h20_contract() {
    let capability = DeepGemmSm90Capability::h20();
    let families = Qwen38Fp8WeightPlan::lower().fp8_projection_families().unwrap();

    for family in families {
        for rows in LOW_LATENCY_ROW_BUCKETS {
            let contract = capability.preferred_candidate(rows, family.shape).unwrap();
            capability.validate_contract(&contract).unwrap();
            assert_eq!(contract.qualification, QualificationState::Legal);
            assert_eq!(contract.row_limit, rows);
            assert_eq!(contract.layout.quantized_activation_fp8_e4m3, [rows, family.shape.input_features]);
            assert_eq!(contract.layout.activation_scale_f32[0], family.shape.input_features / 128);
            assert_eq!(contract.layout.activation_scale_f32[1] % 4, 0);
            assert_eq!(contract.layout.scratch.scale_offset % 16, 0);
            assert_eq!(
                contract.layout.scratch.total_bytes,
                contract.layout.scratch.quantized_bytes + contract.layout.scratch.scale_bytes
            );
            assert!(contract.tile.shared_memory_bytes <= 232_448);
        }
    }
}

#[test]
fn provider_and_numerical_identity_are_explicit_but_model_identity_is_not() {
    let capability = DeepGemmSm90Capability::h20();
    let contract = capability
        .preferred_candidate(1, Fp8ProjectionShape { output_features: 14_336, input_features: 5_120 })
        .unwrap();

    assert_eq!(contract.provider_revision, DEEPGEMM_REVISION);
    assert_eq!(contract.numerical_abi, PACKED_ACTIVATION_ABI);
    assert_eq!(contract.target, "nvidia-h20-sm90a");
    assert_eq!(contract.num_sms, 78);
    assert_eq!(contract.contract_version, 1);
    let decoded: orbitkv_compiler::compiler::provider::DeepGemmKernelContract =
        serde_json::from_str(&serde_json::to_string(&contract).unwrap()).unwrap();
    assert_eq!(decoded, contract);
    let serialized = serde_json::to_string(&contract).unwrap();
    assert!(!serialized.to_ascii_lowercase().contains("qwen"));
    assert!(!serialized.to_ascii_lowercase().contains("model"));
}

#[test]
fn aot_source_is_fully_specialized_and_embeds_both_identities() {
    let contract = DeepGemmSm90Capability::h20()
        .preferred_candidate(8, Fp8ProjectionShape { output_features: 14_336, input_features: 5_120 })
        .unwrap();
    let source = render_aot_source(&contract).unwrap();

    assert!(!source.contains('@'));
    assert!(source.contains(DEEPGEMM_REVISION));
    assert!(source.contains(PACKED_ACTIVATION_ABI));
    assert!(source.contains("0, 14336, 5120"));
    assert!(source.contains("block_scaled_quantize"));
    assert!(source.contains("static GemmKernel orbitkv_deepgemm_kernel()"));
}

#[test]
fn qualification_state_only_advances_from_legal_with_real_measurements() {
    let capability = DeepGemmSm90Capability::h20();
    let contract = capability
        .preferred_candidate(1, Fp8ProjectionShape { output_features: 14_336, input_features: 5_120 })
        .unwrap();

    let source_sha = "1".repeat(64);
    let cubin_sha = "a".repeat(64);
    assert!(contract.clone().record_benchmark(&source_sha, &cubin_sha, 0, 100).is_err());
    assert!(contract.clone().record_benchmark(&source_sha, &cubin_sha, 10, 0).is_err());
    assert!(contract.clone().record_benchmark("not-a-digest", &cubin_sha, 10, 100).is_err());
    assert_eq!(
        contract.record_benchmark(&source_sha, &cubin_sha, 123_456, 100).unwrap().qualification,
        QualificationState::Benchmarked {
            source_sha256: source_sha,
            cubin_sha256: cubin_sha,
            median_nanoseconds: 123_456,
            samples: 100,
        }
    );
}

#[test]
fn qualified_artifact_hashes_are_verified_against_the_actual_bytes() {
    use sha2::{Digest, Sha256};

    let legal = DeepGemmSm90Capability::h20()
        .preferred_candidate(1, Fp8ProjectionShape { output_features: 14_336, input_features: 5_120 })
        .unwrap();
    let source = render_aot_source(&legal).unwrap();
    let source_digest = hex::encode(Sha256::digest(source.as_bytes()));
    let empty_digest = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    let contract = legal.record_benchmark(source_digest, empty_digest, 50_000, 100).unwrap();

    contract.verify_artifacts(source.as_bytes(), b"").unwrap();
    assert!(contract.verify_artifacts(b"tampered", b"").is_err());
    assert!(contract.verify_artifacts(source.as_bytes(), b"tampered").is_err());
}

#[test]
fn capability_rejects_a_tampered_tile_or_unqualified_contract() {
    let capability = DeepGemmSm90Capability::h20();
    let contract = capability
        .preferred_candidate(8, Fp8ProjectionShape { output_features: 14_336, input_features: 5_120 })
        .unwrap();
    let mut tampered = contract.clone();
    tampered.tile.shared_memory_bytes += 1;
    assert!(capability.validate_contract(&tampered).is_err());

    let mut unqualified = contract;
    unqualified.qualification = QualificationState::Unqualified;
    assert!(capability.validate_contract(&unqualified).is_err());
}

#[test]
fn legacy_cache_source_is_parseable_evidence_not_an_artifact_identity() {
    let source = r#"
static constexpr char kDeepGemmRevision[] = "559d79fb6994a58b8a15b4b93bf13ccc16edf247";
// Packed activation ABI: fp8-e4m3-row128-f32-kmajor-align4-rne-reciprocal-v2
using GemmKernel = decltype(&sm90_fp8_gemm_1d2d_impl<
    cute::UMMA::Major::K,
    0, 12288, 5120,
    1,
    128, 160, 128,
    128, 128, 64,
    5,
    128, 256,
    2, true,
    78, GemmType::Normal>);
cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 228416);
"#;
    let evidence = HistoricalDeepGemmSourceEvidence::parse(source).unwrap();

    assert_eq!(evidence.provider_revision, DEEPGEMM_REVISION);
    assert_eq!(evidence.numerical_abi, PACKED_ACTIVATION_ABI);
    assert_eq!(evidence.shape, Fp8ProjectionShape { output_features: 12_288, input_features: 5_120 });
    assert_eq!(evidence.tile.block_m, 128);
    assert_eq!(evidence.tile.block_n, 160);
    assert_eq!(evidence.tile.shared_memory_bytes, 228_416);
    assert_eq!(evidence.num_sms, 78);
    // Evidence intentionally has neither qualification state nor a binary digest.
    let serialized = serde_json::to_value(evidence).unwrap();
    assert!(serialized.get("qualification").is_none());
    assert!(serialized.get("artifact_digest").is_none());
}
