use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::{
    DecoderBlockLayout, DecoderConfig, DecoderError, DecoderLayerKind, DecoderWeightFormat,
};

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
    require_float(
        &catalog,
        &format!("{}.embed_tokens.weight", config.tensor_prefix),
        &[config.vocabulary_size, config.hidden_size],
    )?;
    require_float(
        &catalog,
        &format!("{}.norm.weight", config.tensor_prefix),
        &[config.hidden_size],
    )?;
    if !config.tied_embeddings {
        require_float(
            &catalog,
            "lm_head.weight",
            &[config.vocabulary_size, config.hidden_size],
        )?;
    }
    let token_layers = (0..config.layers)
        .filter(|&layer| config.layer_kind(layer) != DecoderLayerKind::Linear)
        .collect::<Vec<_>>();
    let qkv_bias = family_presence(
        &catalog,
        &config.tensor_prefix,
        &token_layers,
        &["q_proj.bias", "k_proj.bias", "v_proj.bias"],
    )?;
    let qk_norm = family_presence(
        &catalog,
        &config.tensor_prefix,
        &token_layers,
        &["q_norm.weight", "k_norm.weight"],
    )?;
    for layer in 0..config.layers {
        validate_common_layer(&catalog, config, layer)?;
        match config.layer_kind(layer) {
            DecoderLayerKind::Full | DecoderLayerKind::Sliding => {
                validate_attention_layer(&catalog, config, layer, qkv_bias, qk_norm)?;
            }
            DecoderLayerKind::Linear => validate_gated_delta_layer(&catalog, config, layer)?,
        }
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

fn validate_common_layer(
    catalog: &WeightCatalog,
    config: &DecoderConfig,
    layer: usize,
) -> Result<(), DecoderError> {
    let prefix = format!("{}.layers.{layer}", config.tensor_prefix);
    for (suffix, shape) in [
        ("input_layernorm.weight", vec![config.hidden_size]),
        ("post_attention_layernorm.weight", vec![config.hidden_size]),
    ] {
        require_float(catalog, &format!("{prefix}.{suffix}"), &shape)?;
    }
    for (suffix, shape) in [
        (
            "mlp.gate_proj.weight",
            [config.intermediate_size, config.hidden_size],
        ),
        (
            "mlp.up_proj.weight",
            [config.intermediate_size, config.hidden_size],
        ),
        (
            "mlp.down_proj.weight",
            [config.hidden_size, config.intermediate_size],
        ),
    ] {
        require_matrix(catalog, config, &format!("{prefix}.{suffix}"), shape)?;
    }
    if config.block_layout == DecoderBlockLayout::SandwichNorm {
        require_float(
            catalog,
            &format!("{prefix}.pre_feedforward_layernorm.weight"),
            &[config.hidden_size],
        )?;
        require_float(
            catalog,
            &format!("{prefix}.post_feedforward_layernorm.weight"),
            &[config.hidden_size],
        )?;
    }
    Ok(())
}

fn validate_attention_layer(
    catalog: &WeightCatalog,
    config: &DecoderConfig,
    layer: usize,
    qkv_bias: bool,
    qk_norm: bool,
) -> Result<(), DecoderError> {
    let prefix = format!("{}.layers.{layer}.self_attn", config.tensor_prefix);
    let q_width = config.query_heads * config.head_dim;
    let q_projection_width = q_width * if config.attention_output_gate { 2 } else { 1 };
    let kv_width = config.kv_heads * config.head_dim;
    for (suffix, shape) in [
        ("q_proj.weight", [q_projection_width, config.hidden_size]),
        ("k_proj.weight", [kv_width, config.hidden_size]),
        ("v_proj.weight", [kv_width, config.hidden_size]),
        ("o_proj.weight", [config.hidden_size, q_width]),
    ] {
        require_matrix(catalog, config, &format!("{prefix}.{suffix}"), shape)?;
    }
    for suffix in optional_suffixes(qkv_bias, &["q_proj.bias", "k_proj.bias", "v_proj.bias"]) {
        let width = if suffix.starts_with('q') {
            q_projection_width
        } else {
            kv_width
        };
        require_float(catalog, &format!("{prefix}.{suffix}"), &[width])?;
    }
    for suffix in optional_suffixes(qk_norm, &["q_norm.weight", "k_norm.weight"]) {
        require_float(catalog, &format!("{prefix}.{suffix}"), &[config.head_dim])?;
    }
    Ok(())
}

fn validate_gated_delta_layer(
    catalog: &WeightCatalog,
    config: &DecoderConfig,
    layer: usize,
) -> Result<(), DecoderError> {
    let geometry = config.gated_delta.ok_or(DecoderError::InvalidGeometry(
        "missing gated-delta geometry",
    ))?;
    let key_elements = geometry
        .key_elements()
        .ok_or(DecoderError::InvalidGeometry("gated-delta key width"))?;
    let value_elements = geometry
        .value_elements()
        .ok_or(DecoderError::InvalidGeometry("gated-delta value width"))?;
    let convolution_channels =
        geometry
            .convolution_channels()
            .ok_or(DecoderError::InvalidGeometry(
                "gated-delta convolution width",
            ))?;
    if convolution_channels != key_elements * 2 + value_elements {
        return Err(DecoderError::InvalidGeometry(
            "gated-delta convolution width",
        ));
    }
    let prefix = format!("{}.layers.{layer}.linear_attn", config.tensor_prefix);
    for (suffix, shape) in [
        (
            "in_proj_qkv.weight",
            [convolution_channels, config.hidden_size],
        ),
        ("in_proj_z.weight", [value_elements, config.hidden_size]),
        ("out_proj.weight", [config.hidden_size, value_elements]),
    ] {
        require_matrix(catalog, config, &format!("{prefix}.{suffix}"), shape)?;
    }
    for (suffix, shape) in [
        (
            "in_proj_b.weight",
            vec![geometry.value_heads, config.hidden_size],
        ),
        (
            "in_proj_a.weight",
            vec![geometry.value_heads, config.hidden_size],
        ),
        (
            "conv1d.weight",
            vec![convolution_channels, 1, geometry.convolution_kernel_width],
        ),
        ("dt_bias", vec![geometry.value_heads]),
        ("A_log", vec![geometry.value_heads]),
        ("norm.weight", vec![geometry.value_width]),
    ] {
        require_float(catalog, &format!("{prefix}.{suffix}"), &shape)?;
    }
    Ok(())
}

fn optional_suffixes<'a>(enabled: bool, suffixes: &'a [&str]) -> &'a [&'a str] {
    if enabled { suffixes } else { &[] }
}

fn family_presence(
    catalog: &WeightCatalog,
    tensor_prefix: &str,
    layers: &[usize],
    suffixes: &[&str],
) -> Result<bool, DecoderError> {
    let families = suffixes
        .iter()
        .map(|suffix| {
            layers
                .iter()
                .copied()
                .filter(|layer| {
                    catalog.contains_key(&format!(
                        "{tensor_prefix}.layers.{layer}.self_attn.{suffix}"
                    ))
                })
                .collect::<BTreeSet<_>>()
        })
        .collect::<Vec<_>>();
    let family_refs = families.iter().collect::<Vec<_>>();
    complete_family(layers.len(), &family_refs)
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

fn require_float(catalog: &WeightCatalog, name: &str, shape: &[usize]) -> Result<(), DecoderError> {
    let Some((actual_shape, dtype)) = catalog.get(name) else {
        return Err(DecoderError::InvalidGeometry("missing weight tensor"));
    };
    if actual_shape != shape || !matches!(dtype.as_str(), "F32" | "F16" | "BF16") {
        return Err(DecoderError::InvalidGeometry("weight tensor geometry"));
    }
    Ok(())
}

fn require_matrix(
    catalog: &WeightCatalog,
    config: &DecoderConfig,
    name: &str,
    shape: [usize; 2],
) -> Result<(), DecoderError> {
    match config.weight_format {
        DecoderWeightFormat::Float => require_float(catalog, name, &shape),
        DecoderWeightFormat::Fp8E4M3Block { rows, columns } => {
            require_dtype(catalog, name, &shape, &["F8_E4M3"])?;
            let scale_shape = [shape[0].div_ceil(rows), shape[1].div_ceil(columns)];
            require_dtype(
                catalog,
                &format!("{name}_scale_inv"),
                &scale_shape,
                &["BF16", "F32"],
            )
        }
    }
}

fn require_dtype(
    catalog: &WeightCatalog,
    name: &str,
    shape: &[usize],
    dtypes: &[&str],
) -> Result<(), DecoderError> {
    let Some((actual_shape, dtype)) = catalog.get(name) else {
        return Err(DecoderError::InvalidGeometry("missing weight tensor"));
    };
    if actual_shape != shape || !dtypes.contains(&dtype.as_str()) {
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
