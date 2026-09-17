use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use half::bf16;
use kern_runtime::{Capacity, Runtime};
use orbitkv_compiler::compiler::provider::{DeepGemmKernelContract, DeepGemmSm90Capability};
use orbitkv_compiler::lower::lower_deepgemm_projection_probe;

#[test]
#[ignore = "requires an H20 and a qualification record produced by tools/qualify_deepgemm_h20.py"]
fn qualified_projection_executes_through_kern_runtime() {
    let report = PathBuf::from(
        std::env::var_os("ORBITKV_DEEPGEMM_QUALIFICATION")
            .expect("set ORBITKV_DEEPGEMM_QUALIFICATION to qualification.json"),
    );
    let value: serde_json::Value = serde_json::from_str(&fs::read_to_string(&report).unwrap()).unwrap();
    let contract: DeepGemmKernelContract = serde_json::from_value(value["contract"].clone()).unwrap();
    let source = Path::new(value["artifacts"]["source"].as_str().unwrap());
    let cubin = Path::new(value["artifacts"]["cubin"].as_str().unwrap());
    let capability = DeepGemmSm90Capability::h20();
    capability.validate_contract(&contract).unwrap();
    contract.verify_artifacts(&fs::read(source).unwrap(), &fs::read(cubin).unwrap()).unwrap();

    let artifact = lower_deepgemm_projection_probe(&contract, cubin.file_name().unwrap().to_str().unwrap()).unwrap();
    let mut runtime = Runtime::load(
        artifact.verified(),
        cubin.parent().unwrap(),
        0,
        Some(Capacity { tokens: Some(1), seqs: 0 }),
        None,
    )
    .unwrap();

    let rows = contract.row_limit as usize;
    let n = contract.shape.output_features as usize;
    let k = contract.shape.input_features as usize;
    // Ones exercise the real quantizer but make its reconstructed value exact:
    // FP8 448 times the emitted 1/448 scale is one. The alternating weight
    // pattern and per-block scales then have an independent closed-form result.
    let input: Vec<u8> = (0..rows * k).flat_map(|_| bf16::ONE.to_bits().to_le_bytes()).collect();
    let weight: Vec<u8> = (0..n * k).map(|index| if (index % k).is_multiple_of(3) { 0xb0 } else { 0x30 }).collect();
    let scales: Vec<f32> = (0..n.div_ceil(128) * k.div_ceil(128))
        .map(|index| 0.25 * (1 + (index / (k / 128) + index % (k / 128)) % 4) as f32)
        .collect();
    let scale_bytes: Vec<u8> = scales.iter().flat_map(|value| value.to_le_bytes()).collect();
    runtime.write_input("input", &input).unwrap();
    runtime.write_input("weight", &weight).unwrap();
    runtime.write_input("weight_scale", &scale_bytes).unwrap();
    runtime.issue("projection", &BTreeMap::new()).unwrap();
    runtime.synchronize().unwrap();
    assert!(runtime.is_captured("projection", &BTreeMap::new()));
    runtime.run_captured("projection", &BTreeMap::new()).unwrap();

    let output = runtime.read_output("output").unwrap();
    let values: Vec<_> = output
        .chunks_exact(2)
        .map(|bytes| bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32())
        .collect();
    assert_eq!(values.len(), rows * n);
    for row in values.chunks_exact(n) {
        for (n_block, block) in row.chunks_exact(128).enumerate() {
            let expected = (0..k.div_ceil(128))
                .map(|k_block| {
                    let negatives = (k_block * 128..(k_block + 1) * 128).filter(|column| column % 3 == 0).count();
                    let unscaled = 64.0 - negatives as f32;
                    let scale = 0.25 * (1 + (n_block + k_block) % 4) as f32;
                    unscaled * scale
                })
                .sum::<f32>();
            let expected = bf16::from_f32(expected).to_f32();
            assert!(block.iter().all(|actual| *actual == expected));
        }
    }
}
