use serde_json::{Value, json};

use crate::compiler::provider::{DeepGemmKernelContract, DeepGemmSm90Capability};

use super::{KernArtifact, LoweringError};

/// Lower one qualified block-scaled projection to an executable schema-v6
/// probe. The same op shape is used when the full model decoder is emitted;
/// the probe keeps weight and scale as inputs so qualification needs no model
/// checkpoint or weight-loader special case.
pub fn lower_deepgemm_projection_probe(
    contract: &DeepGemmKernelContract,
    module_source: &str,
) -> Result<KernArtifact, LoweringError> {
    DeepGemmSm90Capability::h20().validate_contract(contract).map_err(|error| LoweringError(error.to_string()))?;
    let cubin_sha256 = contract.cubin_sha256().map_err(|error| LoweringError(error.to_string()))?;
    let kernel_entry = contract.kernel_entry().map_err(|error| LoweringError(error.to_string()))?;
    if module_source.is_empty() {
        return Err(LoweringError("projection module source must not be empty".into()));
    }

    let m = u64::from(contract.row_limit);
    let n = u64::from(contract.shape.output_features);
    let k = u64::from(contract.shape.input_features);
    let tile = contract.tile;
    let scale_rows = u64::from(contract.layout.activation_scale_f32[1]);
    let scale_columns = u64::from(contract.layout.activation_scale_f32[0]);
    let cluster = (tile.cluster_m * tile.cluster_n > 1).then(|| json!([tile.cluster_m * tile.cluster_n, 1, 1]));

    let mut gemm_launch = json!({
        "module": "deepgemm",
        "entry": kernel_entry,
        "params": [
            "in buffer<f32>", "i64", "i32", "i32", "i32",
            "bytes<128>", "bytes<128>", "bytes<128>", "bytes<128>"
        ],
        "block": [128 + tile.math_threads, 1, 1],
        "grid": [contract.num_sms, 1, 1],
        "shared_mem": tile.shared_memory_bytes,
        "args": [
            {"param": 2},
            {"i64": 0},
            {"i32": contract.row_limit},
            {"i32": contract.shape.output_features},
            {"i32": contract.shape.input_features},
            tma_scratch("quantized_activation", "u8", [k, m], k, [128, u64::from(tile.block_m)], 128),
            tma_param(1, "u8", [k, n], k, [128, u64::from(tile.block_n)], 128),
            tma_param(3, "bf16", [n, m], n * 2, [u64::from(tile.swizzle_d / 2), u64::from(tile.block_m)], tile.swizzle_d),
            tma_scratch("activation_scale", "f32", [scale_rows, scale_columns], scale_rows * 4, [u64::from(tile.block_m), 1], 0),
        ]
    });
    if let Some(cluster) = cluster {
        gemm_launch["cluster"] = cluster;
    }

    let manifest = json!({
        "schema_version": 6,
        "model": "orbitkv-fp8-projection-probe",
        "vars": {},
        "states": {},
        "buffers": {
            "input": {"dtype": "bf16", "shape": [m, k], "kind": "input"},
            "weight": {"dtype": "fp8e4m3", "shape": [n, k], "kind": "input"},
            "weight_scale": {"dtype": "f32", "shape": [n.div_ceil(128), k.div_ceil(128)], "kind": "input"},
            "output": {"dtype": "bf16", "shape": [m, n], "kind": "output"}
        },
        "modules": {
            "deepgemm": {"source": module_source, "sha256": cubin_sha256}
        },
        "ops": {
            "fp8_projection": {
                "params": [
                    "in buffer<bf16>", "in buffer<fp8e4m3>",
                    "in buffer<f32>", "out buffer<bf16>"
                ],
                "impl": {
                    "scratch": {
                        "quantized_activation": {"dtype": "u8", "shape": [m, k]},
                        "activation_scale": {"dtype": "f32", "shape": [scale_columns, scale_rows]}
                    },
                    "launches": [
                        {
                            "module": "deepgemm",
                            "entry": "block_scaled_quantize",
                            "params": [
                                "out buffer<u8>", "out buffer<f32>", "in buffer<bf16>", "i32", "i32"
                            ],
                            "block": [32, 1, 1],
                            "grid": [scale_columns, m, 1],
                            "args": [
                                {"scratch": "quantized_activation"},
                                {"scratch": "activation_scale"},
                                {"param": 0},
                                {"i32": contract.row_limit},
                                {"i32": contract.shape.input_features}
                            ]
                        },
                        gemm_launch
                    ]
                }
            }
        },
        "programs": {
            "projection": {
                "batch": {"groups": m, "rows": 1},
                "graph": true,
                "calls": [{
                    "op": "fp8_projection",
                    "args": [
                        {"buf": "input"}, {"buf": "weight"},
                        {"buf": "weight_scale"}, {"buf": "output"}
                    ]
                }]
            }
        }
    });
    KernArtifact::from_json(&manifest.to_string())
}

fn tma_param(param: usize, dtype: &str, dims: [u64; 2], stride: u64, box_: [u64; 2], swizzle: u32) -> Value {
    tma(json!({"param": param}), dtype, dims, stride, box_, swizzle)
}

fn tma_scratch(scratch: &str, dtype: &str, dims: [u64; 2], stride: u64, box_: [u64; 2], swizzle: u32) -> Value {
    tma(json!({"scratch": scratch}), dtype, dims, stride, box_, swizzle)
}

fn tma(mut source: Value, dtype: &str, dims: [u64; 2], stride: u64, box_: [u64; 2], swizzle: u32) -> Value {
    let source = source.as_object_mut().expect("TMA source is an object");
    source.insert("dtype".into(), dtype.into());
    source.insert("dims".into(), json!(dims));
    source.insert("strides".into(), json!([stride]));
    source.insert("box".into(), json!(box_));
    source.insert("l2_promotion".into(), 256.into());
    if swizzle != 0 {
        source.insert("swizzle".into(), swizzle.into());
    }
    json!({"pack": {"size": 128, "fields": [{"at": 0, "tensormap": source}]}})
}
