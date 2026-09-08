use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::{DecoderBlockLayout, DecoderConfig, DecoderError};

const MAX_WEIGHT_HEADER_BYTES: usize = 16 * 1024 * 1024;
type WeightCatalog = BTreeMap<String, (Vec<usize>, String)>;

#[derive(Deserialize)]
struct HeaderTensor {
    dtype: String,
    shape: Vec<usize>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Deserialize, Serialize)]
pub(super) struct DecoderWeightFeatures {
    pub(super) qkv_bias: bool,
    pub(super) qk_norm: bool,
}

pub(super) fn inspect_weight_features(
    weight_files: &[PathBuf],
    config: &DecoderConfig,
) -> Result<DecoderWeightFeatures, DecoderError> {
    let catalog = read_catalog(weight_files)?;
    require(
        &catalog,
        &format!("{}.embed_tokens.weight", config.tensor_prefix),
        &[config.vocabulary_size, config.hidden_size],
    )?;
    require(
        &catalog,
        &format!("{}.norm.weight", config.tensor_prefix),
        &[config.hidden_size],
    )?;
    if !config.tied_embeddings {
        require(
            &catalog,
            "lm_head.weight",
            &[config.vocabulary_size, config.hidden_size],
        )?;
    }
    let qkv_bias = family_presence(
        &catalog,
        &config.tensor_prefix,
        config.layers,
        &["q_proj.bias", "k_proj.bias", "v_proj.bias"],
    )?;
    let qk_norm = family_presence(
        &catalog,
        &config.tensor_prefix,
        config.layers,
        &["q_norm.weight", "k_norm.weight"],
    )?;
    for layer in 0..config.layers {
        validate_layer(&catalog, config, layer, qkv_bias, qk_norm)?;
    }
    Ok(DecoderWeightFeatures { qkv_bias, qk_norm })
}

fn read_catalog(weight_files: &[PathBuf]) -> Result<WeightCatalog, DecoderError> {
    let mut catalog = BTreeMap::new();
    for path in weight_files {
        for (name, tensor) in read_metadata(path)? {
            if name == "__metadata__" {
                continue;
            }
            let info = serde_json::from_value::<HeaderTensor>(tensor)
                .map_err(|_| DecoderError::InvalidGeometry("weights metadata"))?;
            if catalog.insert(name, (info.shape, info.dtype)).is_some() {
                return Err(DecoderError::InvalidGeometry("duplicate weight tensor"));
            }
        }
    }
    Ok(catalog)
}

fn validate_layer(
    catalog: &WeightCatalog,
    config: &DecoderConfig,
    layer: usize,
    qkv_bias: bool,
    qk_norm: bool,
) -> Result<(), DecoderError> {
    let prefix = format!("{}.layers.{layer}", config.tensor_prefix);
    let q_width = config.query_heads * config.head_dim;
    let kv_width = config.kv_heads * config.head_dim;
    for (suffix, shape) in [
        ("input_layernorm.weight", vec![config.hidden_size]),
        ("post_attention_layernorm.weight", vec![config.hidden_size]),
        ("self_attn.q_proj.weight", vec![q_width, config.hidden_size]),
        (
            "self_attn.k_proj.weight",
            vec![kv_width, config.hidden_size],
        ),
        (
            "self_attn.v_proj.weight",
            vec![kv_width, config.hidden_size],
        ),
        ("self_attn.o_proj.weight", vec![config.hidden_size, q_width]),
        (
            "mlp.gate_proj.weight",
            vec![config.intermediate_size, config.hidden_size],
        ),
        (
            "mlp.up_proj.weight",
            vec![config.intermediate_size, config.hidden_size],
        ),
        (
            "mlp.down_proj.weight",
            vec![config.hidden_size, config.intermediate_size],
        ),
    ] {
        require(catalog, &format!("{prefix}.{suffix}"), &shape)?;
    }
    if config.block_layout == DecoderBlockLayout::SandwichNorm {
        require(
            catalog,
            &format!("{prefix}.pre_feedforward_layernorm.weight"),
            &[config.hidden_size],
        )?;
        require(
            catalog,
            &format!("{prefix}.post_feedforward_layernorm.weight"),
            &[config.hidden_size],
        )?;
    }
    for suffix in optional_suffixes(qkv_bias, &["q_proj.bias", "k_proj.bias", "v_proj.bias"]) {
        let width = if suffix.starts_with('q') {
            q_width
        } else {
            kv_width
        };
        require(catalog, &format!("{prefix}.self_attn.{suffix}"), &[width])?;
    }
    for suffix in optional_suffixes(qk_norm, &["q_norm.weight", "k_norm.weight"]) {
        require(
            catalog,
            &format!("{prefix}.self_attn.{suffix}"),
            &[config.head_dim],
        )?;
    }
    Ok(())
}

fn optional_suffixes<'a>(enabled: bool, suffixes: &'a [&str]) -> &'a [&'a str] {
    if enabled { suffixes } else { &[] }
}

fn family_presence(
    catalog: &WeightCatalog,
    tensor_prefix: &str,
    layers: usize,
    suffixes: &[&str],
) -> Result<bool, DecoderError> {
    let families = suffixes
        .iter()
        .map(|suffix| {
            (0..layers)
                .filter(|layer| {
                    catalog.contains_key(&format!(
                        "{tensor_prefix}.layers.{layer}.self_attn.{suffix}"
                    ))
                })
                .collect::<BTreeSet<_>>()
        })
        .collect::<Vec<_>>();
    let family_refs = families.iter().collect::<Vec<_>>();
    complete_family(layers, &family_refs)
}

fn complete_family(layers: usize, families: &[&BTreeSet<usize>]) -> Result<bool, DecoderError> {
    let counts = families
        .iter()
        .map(|family| family.len())
        .collect::<Vec<_>>();
    if counts.iter().all(|&count| count == 0) {
        return Ok(false);
    }
    if counts.iter().all(|&count| count == layers) {
        return Ok(true);
    }
    Err(DecoderError::InvalidGeometry("incomplete weight family"))
}

fn require(catalog: &WeightCatalog, name: &str, shape: &[usize]) -> Result<(), DecoderError> {
    let Some((actual_shape, dtype)) = catalog.get(name) else {
        return Err(DecoderError::InvalidGeometry("missing weight tensor"));
    };
    if actual_shape != shape || !matches!(dtype.as_str(), "F32" | "F16" | "BF16") {
        return Err(DecoderError::InvalidGeometry("weight tensor geometry"));
    }
    Ok(())
}

fn read_metadata(path: &PathBuf) -> Result<BTreeMap<String, serde_json::Value>, DecoderError> {
    let mut file = File::open(path).map_err(|_| DecoderError::InvalidGeometry("weights path"))?;
    let mut size_bytes = [0_u8; 8];
    file.read_exact(&mut size_bytes)
        .map_err(|_| DecoderError::InvalidGeometry("weights metadata"))?;
    let header_bytes = usize::try_from(u64::from_le_bytes(size_bytes))
        .ok()
        .filter(|size| *size <= MAX_WEIGHT_HEADER_BYTES)
        .ok_or(DecoderError::InvalidGeometry("weights metadata"))?;
    let mut header = Vec::with_capacity(header_bytes + size_bytes.len());
    header.extend_from_slice(&size_bytes);
    header.resize(header_bytes + size_bytes.len(), 0);
    file.read_exact(&mut header[size_bytes.len()..])
        .map_err(|_| DecoderError::InvalidGeometry("weights metadata"))?;
    serde_json::from_slice(&header[size_bytes.len()..])
        .map_err(|_| DecoderError::InvalidGeometry("weights metadata"))
}
