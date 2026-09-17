use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Component, Path};

use serde::{Deserialize, Serialize};

use crate::model::{LayerKind, Qwen38Contract};

const MAX_HEADER_BYTES: usize = 16 * 1024 * 1024;
const INDEX_TENSORS: usize = 1_606;
const INDEX_SHARDS: usize = 66;

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct TensorContract {
    pub dtype: String,
    pub shape: Vec<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Qwen38Fp8Report {
    pub model: String,
    /// Pinned source revision expected by the project. The local HF snapshot
    /// does not carry cryptographically verifiable revision metadata.
    pub expected_checkpoint_revision: String,
    pub quantization: String,
    pub block_shape: [u64; 2],
    pub index_tensors: usize,
    pub index_shards: usize,
    pub text_tensors: usize,
    pub fp8_matrices: usize,
    pub scale_tensors: usize,
    pub bf16_tensors: usize,
    pub ignored_non_text_tensors: usize,
}

#[derive(Debug)]
pub enum CheckpointError {
    Io(String),
    Json(String),
    Contract(Vec<String>),
}

impl fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "checkpoint I/O: {error}"),
            Self::Json(error) => write!(f, "checkpoint JSON: {error}"),
            Self::Contract(errors) => {
                f.write_str("Qwen3.8 FP8 checkpoint violates the pinned contract:")?;
                for error in errors {
                    write!(f, "\n  - {error}")?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for CheckpointError {}

#[derive(Deserialize)]
struct Index {
    weight_map: BTreeMap<String, String>,
}

#[derive(Deserialize)]
struct HeaderTensor {
    dtype: String,
    shape: Vec<u64>,
}

pub struct Qwen38Fp8Checkpoint;

impl Qwen38Fp8Checkpoint {
    pub fn expected_text_tensors() -> BTreeMap<String, TensorContract> {
        expected_text_tensors()
    }

    /// Validate config, shard index, and safetensors headers without reading
    /// any tensor payload. The text tower is exact; vision and MTP are counted
    /// but remain outside the target-only M1 gate.
    pub fn inspect(root: impl AsRef<Path>) -> Result<Qwen38Fp8Report, CheckpointError> {
        let root = root.as_ref();
        let config = read_json(&root.join("config.json"))?;
        let index: Index = serde_json::from_value(read_json(&root.join("model.safetensors.index.json"))?)
            .map_err(|error| CheckpointError::Json(error.to_string()))?;
        let expected = expected_text_tensors();
        let mut errors = Vec::new();
        validate_config(&config, &mut errors);

        check_eq(&mut errors, "index tensor count", index.weight_map.len(), INDEX_TENSORS);
        let shards: BTreeSet<_> = index.weight_map.values().cloned().collect();
        check_eq(&mut errors, "index shard count", shards.len(), INDEX_SHARDS);
        let actual_text: BTreeSet<_> = index
            .weight_map
            .keys()
            .filter(|name| name.starts_with("model.language_model.") || name.as_str() == "lm_head.weight")
            .cloned()
            .collect();
        let expected_names: BTreeSet<_> = expected.keys().cloned().collect();
        for name in expected_names.difference(&actual_text) {
            errors.push(format!("missing text tensor `{name}`"));
        }
        for name in actual_text.difference(&expected_names) {
            errors.push(format!("unexpected text tensor `{name}`"));
        }

        let mut by_shard: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for name in expected.keys() {
            if let Some(shard) = index.weight_map.get(name) {
                by_shard.entry(shard).or_default().push(name);
            }
        }
        for (shard, names) in by_shard {
            if !safe_file_name(shard) {
                errors.push(format!("unsafe shard path `{shard}`"));
                continue;
            }
            let header = read_safetensors_header(&root.join(shard))?;
            for name in names {
                let Some(actual) = header.get(name) else {
                    errors.push(format!("shard `{shard}` does not contain indexed tensor `{name}`"));
                    continue;
                };
                let expected = &expected[name];
                if actual.dtype != expected.dtype || actual.shape != expected.shape {
                    errors.push(format!(
                        "tensor `{name}`: expected {} {:?}, found {} {:?}",
                        expected.dtype, expected.shape, actual.dtype, actual.shape
                    ));
                }
            }
        }
        if !errors.is_empty() {
            return Err(CheckpointError::Contract(errors));
        }

        let fp8_matrices = expected.values().filter(|tensor| tensor.dtype == "F8_E4M3").count();
        let scale_tensors = expected.keys().filter(|name| name.ends_with("_scale_inv")).count();
        let bf16_tensors = expected.values().filter(|tensor| tensor.dtype == "BF16").count();
        Ok(Qwen38Fp8Report {
            model: Qwen38Contract::MODEL.into(),
            expected_checkpoint_revision: Qwen38Contract::REVISION.into(),
            quantization: "dynamic_e4m3".into(),
            block_shape: [Qwen38Contract::FP8_SCALE_ROWS as u64, Qwen38Contract::FP8_SCALE_COLUMNS as u64],
            index_tensors: index.weight_map.len(),
            index_shards: shards.len(),
            text_tensors: expected.len(),
            fp8_matrices,
            scale_tensors,
            bf16_tensors,
            ignored_non_text_tensors: index.weight_map.len() - expected.len(),
        })
    }
}

fn expected_text_tensors() -> BTreeMap<String, TensorContract> {
    let mut tensors = BTreeMap::new();
    bf16(&mut tensors, "model.language_model.embed_tokens.weight", &[248_320, 5_120]);
    bf16(&mut tensors, "model.language_model.norm.weight", &[5_120]);
    bf16(&mut tensors, "lm_head.weight", &[248_320, 5_120]);
    for layer in 0..Qwen38Contract::LAYERS {
        let prefix = format!("model.language_model.layers.{layer}");
        bf16(&mut tensors, &format!("{prefix}.input_layernorm.weight"), &[5_120]);
        bf16(&mut tensors, &format!("{prefix}.post_attention_layernorm.weight"), &[5_120]);
        fp8(&mut tensors, &format!("{prefix}.mlp.gate_proj.weight"), 17_408, 5_120);
        fp8(&mut tensors, &format!("{prefix}.mlp.up_proj.weight"), 17_408, 5_120);
        fp8(&mut tensors, &format!("{prefix}.mlp.down_proj.weight"), 5_120, 17_408);
        match Qwen38Contract::layer_kind(layer) {
            LayerKind::GatedDelta => {
                let prefix = format!("{prefix}.linear_attn");
                fp8(&mut tensors, &format!("{prefix}.in_proj_qkv.weight"), 10_240, 5_120);
                fp8(&mut tensors, &format!("{prefix}.in_proj_z.weight"), 6_144, 5_120);
                fp8(&mut tensors, &format!("{prefix}.out_proj.weight"), 5_120, 6_144);
                bf16(&mut tensors, &format!("{prefix}.in_proj_b.weight"), &[48, 5_120]);
                bf16(&mut tensors, &format!("{prefix}.in_proj_a.weight"), &[48, 5_120]);
                bf16(&mut tensors, &format!("{prefix}.conv1d.weight"), &[10_240, 1, 4]);
                bf16(&mut tensors, &format!("{prefix}.dt_bias"), &[48]);
                bf16(&mut tensors, &format!("{prefix}.A_log"), &[48]);
                bf16(&mut tensors, &format!("{prefix}.norm.weight"), &[128]);
            }
            LayerKind::FullAttention => {
                let prefix = format!("{prefix}.self_attn");
                fp8(&mut tensors, &format!("{prefix}.q_proj.weight"), 12_288, 5_120);
                fp8(&mut tensors, &format!("{prefix}.k_proj.weight"), 1_024, 5_120);
                fp8(&mut tensors, &format!("{prefix}.v_proj.weight"), 1_024, 5_120);
                fp8(&mut tensors, &format!("{prefix}.o_proj.weight"), 5_120, 6_144);
                bf16(&mut tensors, &format!("{prefix}.q_norm.weight"), &[256]);
                bf16(&mut tensors, &format!("{prefix}.k_norm.weight"), &[256]);
            }
        }
    }
    tensors
}

fn bf16(tensors: &mut BTreeMap<String, TensorContract>, name: &str, shape: &[u64]) {
    let old = tensors.insert(name.into(), TensorContract { dtype: "BF16".into(), shape: shape.into() });
    assert!(old.is_none(), "duplicate tensor contract `{name}`");
}

fn fp8(tensors: &mut BTreeMap<String, TensorContract>, name: &str, rows: u64, columns: u64) {
    let old = tensors.insert(name.into(), TensorContract { dtype: "F8_E4M3".into(), shape: vec![rows, columns] });
    assert!(old.is_none(), "duplicate tensor contract `{name}`");
    bf16(tensors, &format!("{name}_scale_inv"), &[rows.div_ceil(128), columns.div_ceil(128)]);
}

fn validate_config(config: &serde_json::Value, errors: &mut Vec<String>) {
    check_json(errors, "model_type", &config["model_type"], &serde_json::json!("qwen3_5"));
    let text = &config["text_config"];
    for (name, expected) in [
        ("hidden_size", serde_json::json!(5_120)),
        ("intermediate_size", serde_json::json!(17_408)),
        ("num_hidden_layers", serde_json::json!(64)),
        ("num_attention_heads", serde_json::json!(24)),
        ("num_key_value_heads", serde_json::json!(4)),
        ("head_dim", serde_json::json!(256)),
        ("linear_num_key_heads", serde_json::json!(16)),
        ("linear_num_value_heads", serde_json::json!(48)),
        ("linear_key_head_dim", serde_json::json!(128)),
        ("linear_value_head_dim", serde_json::json!(128)),
        ("linear_conv_kernel_dim", serde_json::json!(4)),
        ("mamba_ssm_dtype", serde_json::json!("float32")),
    ] {
        check_json(errors, &format!("text_config.{name}"), &text[name], &expected);
    }
    let expected_layers: Vec<_> = (0..Qwen38Contract::LAYERS)
        .map(|layer| match Qwen38Contract::layer_kind(layer) {
            LayerKind::GatedDelta => serde_json::json!("linear_attention"),
            LayerKind::FullAttention => serde_json::json!("full_attention"),
        })
        .collect();
    check_json(errors, "text_config.layer_types", &text["layer_types"], &serde_json::json!(expected_layers));
    let quant = &config["quantization_config"];
    for (name, expected) in [
        ("quant_method", serde_json::json!("fp8")),
        ("fmt", serde_json::json!("e4m3")),
        ("activation_scheme", serde_json::json!("dynamic")),
        ("weight_block_size", serde_json::json!([128, 128])),
    ] {
        check_json(errors, &format!("quantization_config.{name}"), &quant[name], &expected);
    }
}

fn check_json(errors: &mut Vec<String>, name: &str, actual: &serde_json::Value, expected: &serde_json::Value) {
    if actual != expected {
        errors.push(format!("{name}: expected {expected}, found {actual}"));
    }
}

fn check_eq<T: fmt::Debug + PartialEq>(errors: &mut Vec<String>, name: &str, actual: T, expected: T) {
    if actual != expected {
        errors.push(format!("{name}: expected {expected:?}, found {actual:?}"));
    }
}

fn read_json(path: &Path) -> Result<serde_json::Value, CheckpointError> {
    let bytes = fs::read(path).map_err(|error| CheckpointError::Io(format!("{}: {error}", path.display())))?;
    serde_json::from_slice(&bytes).map_err(|error| CheckpointError::Json(format!("{}: {error}", path.display())))
}

fn read_safetensors_header(path: &Path) -> Result<BTreeMap<String, HeaderTensor>, CheckpointError> {
    let mut file = File::open(path).map_err(|error| CheckpointError::Io(format!("{}: {error}", path.display())))?;
    let mut size = [0_u8; 8];
    file.read_exact(&mut size).map_err(|error| CheckpointError::Io(format!("{}: {error}", path.display())))?;
    let size = usize::try_from(u64::from_le_bytes(size))
        .ok()
        .filter(|size| *size <= MAX_HEADER_BYTES)
        .ok_or_else(|| CheckpointError::Json(format!("{}: invalid safetensors header size", path.display())))?;
    let mut bytes = vec![0_u8; size];
    file.read_exact(&mut bytes).map_err(|error| CheckpointError::Io(format!("{}: {error}", path.display())))?;
    let values: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&bytes)
        .map_err(|error| CheckpointError::Json(format!("{}: {error}", path.display())))?;
    values
        .into_iter()
        .filter(|(name, _)| name != "__metadata__")
        .map(|(name, value)| {
            serde_json::from_value(value)
                .map(|tensor| (name.clone(), tensor))
                .map_err(|error| CheckpointError::Json(format!("{} tensor `{name}`: {error}", path.display())))
        })
        .collect()
}

fn safe_file_name(path: &str) -> bool {
    let path = Path::new(path);
    path.components().all(|component| matches!(component, Component::Normal(_)))
}
