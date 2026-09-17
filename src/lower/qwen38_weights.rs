use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use kern_manifest::types::{Buffer, BufferKind, DType, Dim, Placement, Segment, TensorSource};
use serde::Serialize;

use crate::compiler::provider::{Fp8ProjectionFamily, ProviderContractError};
use crate::model::{LayerKind, Qwen38Contract};
use crate::oracle::TensorContract;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightTransformKind {
    CastBf16ToF32,
    AddOneBf16ToF32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct WeightTransform {
    pub input: String,
    pub output: String,
    pub elements: u64,
    pub kind: WeightTransformKind,
}

#[derive(Debug)]
pub struct Qwen38Fp8WeightPlan {
    /// Buffers bound directly to official checkpoint tensors.
    pub bound: BTreeMap<String, Buffer>,
    /// F32 scales, unit-offset norms, and A_log values produced by `load`.
    pub derived: BTreeMap<String, Buffer>,
    pub transforms: Vec<WeightTransform>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WeightPlanError(pub Vec<String>);

impl fmt::Display for WeightPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("invalid Qwen3.8 FP8 physical weight plan:")?;
        for error in &self.0 {
            write!(f, "\n  - {error}")?;
        }
        Ok(())
    }
}

impl std::error::Error for WeightPlanError {}

impl Qwen38Fp8WeightPlan {
    pub fn lower() -> Self {
        let mut plan = Self { bound: BTreeMap::new(), derived: BTreeMap::new(), transforms: Vec::new() };
        plan.direct_bf16("model.embed_tokens.weight", &[248_320, 5_120], "model.language_model.embed_tokens.weight");
        plan.unit_offset_norm("model.norm.weight", &[5_120], "model.language_model.norm.weight");
        plan.direct_bf16("lm_head.weight", &[248_320, 5_120], "lm_head.weight");

        for layer in 0..Qwen38Contract::LAYERS {
            let physical = format!("model.layers.{layer}");
            let checkpoint = format!("model.language_model.layers.{layer}");
            plan.unit_offset_norm(
                &format!("{physical}.input_layernorm.weight"),
                &[5_120],
                &format!("{checkpoint}.input_layernorm.weight"),
            );
            plan.fp8_matrix(
                &format!("{physical}.mlp.gate_up_proj.weight"),
                34_816,
                5_120,
                &[
                    (&format!("{checkpoint}.mlp.gate_proj.weight"), 17_408),
                    (&format!("{checkpoint}.mlp.up_proj.weight"), 17_408),
                ],
            );
            plan.fp8_matrix(
                &format!("{physical}.mlp.down_proj.weight"),
                5_120,
                17_408,
                &[(&format!("{checkpoint}.mlp.down_proj.weight"), 5_120)],
            );

            match Qwen38Contract::layer_kind(layer) {
                LayerKind::GatedDelta => plan.gdn_layer(&physical, &checkpoint),
                LayerKind::FullAttention => plan.attention_layer(&physical, &checkpoint),
            }
            plan.unit_offset_norm(
                &format!("{physical}.post_attention_layernorm.weight"),
                &[5_120],
                &format!("{checkpoint}.post_attention_layernorm.weight"),
            );
        }
        plan
    }

    pub fn validate(&self, checkpoint: &BTreeMap<String, TensorContract>) -> Result<(), WeightPlanError> {
        let mut errors = Vec::new();
        let mut consumed = BTreeMap::<String, usize>::new();
        for (name, buffer) in &self.bound {
            if buffer.kind != BufferKind::Weight {
                errors.push(format!("bound buffer `{name}` is not a weight"));
            }
            let expected_dtype = dtype_name(buffer.dtype);
            let mut rows = 0_u64;
            let mut columns = None;
            for segment in &buffer.bind {
                let TensorSource::Named(tensor) = &segment.tensor else {
                    errors.push(format!("buffer `{name}` uses an unexpected ranked tensor"));
                    continue;
                };
                *consumed.entry(tensor.clone()).or_insert(0) += 1;
                let Some(contract) = checkpoint.get(tensor) else {
                    errors.push(format!("buffer `{name}` binds unknown tensor `{tensor}`"));
                    continue;
                };
                if contract.dtype != expected_dtype {
                    errors.push(format!(
                        "buffer `{name}` is {expected_dtype} but tensor `{tensor}` is {}",
                        contract.dtype
                    ));
                }
                if contract.shape.len() == 1 {
                    rows += 1;
                    columns = Some(columns.unwrap_or(contract.shape[0]));
                    if columns != Some(contract.shape[0]) {
                        errors.push(format!("buffer `{name}` concatenates unequal vector widths"));
                    }
                } else {
                    rows += contract.shape[0];
                    let width: u64 = contract.shape[1..].iter().product();
                    columns = Some(columns.unwrap_or(width));
                    if columns != Some(width) {
                        errors.push(format!("buffer `{name}` concatenates unequal matrix widths"));
                    }
                }
            }
            let actual_shape: Vec<u64> = buffer
                .shape
                .iter()
                .map(|dim| match dim {
                    Dim::Const(value) => *value,
                    Dim::Var(var) => {
                        errors.push(format!("weight buffer `{name}` has dynamic dimension `{var}`"));
                        0
                    }
                })
                .collect();
            let packed_shape = if buffer.bind.len() == 1 {
                buffer
                    .bind
                    .first()
                    .and_then(|segment| named(checkpoint, segment))
                    .map(|tensor| tensor.shape.clone())
                    .unwrap_or_default()
            } else {
                vec![rows, columns.unwrap_or(0)]
            };
            if actual_shape != packed_shape {
                errors.push(format!("buffer `{name}` shape {actual_shape:?} does not match segments {packed_shape:?}"));
            }
        }
        for name in checkpoint.keys() {
            match consumed.get(name).copied() {
                None => errors.push(format!("checkpoint tensor `{name}` is not consumed")),
                Some(1) => {}
                Some(count) => errors.push(format!("checkpoint tensor `{name}` is consumed {count} times")),
            }
        }
        for name in consumed.keys().filter(|name| !checkpoint.contains_key(*name)) {
            errors.push(format!("unknown consumed checkpoint tensor `{name}`"));
        }

        let outputs: BTreeSet<_> = self.transforms.iter().map(|transform| transform.output.as_str()).collect();
        if outputs.len() != self.transforms.len() {
            errors.push("multiple transforms write the same output".into());
        }
        for transform in &self.transforms {
            let Some(input) = self.bound.get(&transform.input) else {
                errors.push(format!("transform input `{}` is not bound", transform.input));
                continue;
            };
            let Some(output) = self.derived.get(&transform.output) else {
                errors.push(format!("transform output `{}` is not declared", transform.output));
                continue;
            };
            if input.dtype != DType::Bf16 || output.dtype != DType::F32 || input.shape != output.shape {
                errors.push(format!(
                    "transform `{}` -> `{}` is not shape-preserving BF16 -> F32",
                    transform.input, transform.output
                ));
            }
            let elements = dims(&input.shape).unwrap_or(0);
            if elements != transform.elements {
                errors.push(format!(
                    "transform `{}` has {} elements, expected {elements}",
                    transform.output, transform.elements
                ));
            }
        }
        for name in self.bound.keys().filter(|name| name.ends_with(".weight")) {
            if self.bound[name].dtype == DType::Fp8E4m3 {
                let raw = format!("{name}_scale_inv.raw");
                let derived = format!("{name}_scale_inv");
                if !self.bound.contains_key(&raw) || !self.derived.contains_key(&derived) {
                    errors.push(format!("FP8 buffer `{name}` has no raw and derived scale pair"));
                }
            }
        }
        if errors.is_empty() { Ok(()) } else { Err(WeightPlanError(errors)) }
    }

    pub fn fp8_buffers(&self) -> usize {
        self.bound.values().filter(|buffer| buffer.dtype == DType::Fp8E4m3).count()
    }

    pub fn fp8_projection_families(&self) -> Result<Vec<Fp8ProjectionFamily>, ProviderContractError> {
        Fp8ProjectionFamily::from_weight_buffers(&self.bound)
    }

    fn gdn_layer(&mut self, physical: &str, checkpoint: &str) {
        let p = format!("{physical}.linear_attn");
        let c = format!("{checkpoint}.linear_attn");
        self.fp8_matrix(
            &format!("{p}.in_proj_qkvz.weight"),
            16_384,
            5_120,
            &[(&format!("{c}.in_proj_qkv.weight"), 10_240), (&format!("{c}.in_proj_z.weight"), 6_144)],
        );
        self.direct_bf16_many(
            &format!("{p}.in_proj_ba.weight"),
            &[96, 5_120],
            &[&format!("{c}.in_proj_b.weight"), &format!("{c}.in_proj_a.weight")],
        );
        self.direct_bf16(&format!("{p}.conv1d.weight"), &[10_240, 1, 4], &format!("{c}.conv1d.weight"));
        self.direct_bf16(&format!("{p}.dt_bias"), &[48], &format!("{c}.dt_bias"));
        self.direct_bf16(&format!("{p}.norm.weight"), &[128], &format!("{c}.norm.weight"));
        self.fp8_matrix(&format!("{p}.out_proj.weight"), 5_120, 6_144, &[(&format!("{c}.out_proj.weight"), 5_120)]);
        self.cast(
            &format!("{p}.A_log.bf16"),
            &format!("{p}.A_log"),
            &[48],
            &format!("{c}.A_log"),
            WeightTransformKind::CastBf16ToF32,
        );
    }

    fn attention_layer(&mut self, physical: &str, checkpoint: &str) {
        let p = format!("{physical}.self_attn");
        let c = format!("{checkpoint}.self_attn");
        self.fp8_matrix(
            &format!("{p}.qkv_proj.weight"),
            14_336,
            5_120,
            &[
                (&format!("{c}.q_proj.weight"), 12_288),
                (&format!("{c}.k_proj.weight"), 1_024),
                (&format!("{c}.v_proj.weight"), 1_024),
            ],
        );
        self.fp8_matrix(&format!("{p}.o_proj.weight"), 5_120, 6_144, &[(&format!("{c}.o_proj.weight"), 5_120)]);
        self.unit_offset_norm(&format!("{p}.q_norm.weight"), &[256], &format!("{c}.q_norm.weight"));
        self.unit_offset_norm(&format!("{p}.k_norm.weight"), &[256], &format!("{c}.k_norm.weight"));
    }

    fn fp8_matrix(&mut self, name: &str, rows: u64, columns: u64, sources: &[(&str, u64)]) {
        assert_eq!(sources.iter().map(|(_, rows)| rows).sum::<u64>(), rows);
        self.insert_bound(
            name,
            DType::Fp8E4m3,
            &[rows, columns],
            sources.iter().map(|(source, _)| (*source).into()).collect(),
        );
        let raw = format!("{name}_scale_inv.raw");
        let output = format!("{name}_scale_inv");
        let scale_rows = rows.div_ceil(Qwen38Contract::FP8_SCALE_ROWS as u64);
        let scale_columns = columns.div_ceil(Qwen38Contract::FP8_SCALE_COLUMNS as u64);
        let scale_sources = sources.iter().map(|(source, _)| format!("{source}_scale_inv")).collect();
        self.insert_bound(&raw, DType::Bf16, &[scale_rows, scale_columns], scale_sources);
        self.insert_derived(&output, DType::F32, &[scale_rows, scale_columns]);
        self.transforms.push(WeightTransform {
            input: raw,
            output,
            elements: scale_rows * scale_columns,
            kind: WeightTransformKind::CastBf16ToF32,
        });
    }

    fn unit_offset_norm(&mut self, name: &str, shape: &[u64], source: &str) {
        self.cast(name, &format!("{name}_p1"), shape, source, WeightTransformKind::AddOneBf16ToF32);
    }

    fn cast(&mut self, input: &str, output: &str, shape: &[u64], source: &str, kind: WeightTransformKind) {
        self.insert_bound(input, DType::Bf16, shape, vec![source.into()]);
        self.insert_derived(output, DType::F32, shape);
        self.transforms.push(WeightTransform {
            input: input.into(),
            output: output.into(),
            elements: shape.iter().product(),
            kind,
        });
    }

    fn direct_bf16(&mut self, name: &str, shape: &[u64], source: &str) {
        self.direct_bf16_many(name, shape, &[source]);
    }

    fn direct_bf16_many(&mut self, name: &str, shape: &[u64], sources: &[&str]) {
        self.insert_bound(name, DType::Bf16, shape, sources.iter().map(|source| (*source).into()).collect());
    }

    fn insert_bound(&mut self, name: &str, dtype: DType, shape: &[u64], sources: Vec<String>) {
        let buffer = Buffer {
            dtype,
            shape: shape.iter().copied().map(Dim::Const).collect(),
            kind: BufferKind::Weight,
            placement: Placement::Device,
            fill: None,
            domain: None,
            export: false,
            of: None,
            group: None,
            bind: sources
                .into_iter()
                .map(|tensor| Segment { tensor: TensorSource::Named(tensor), rows: None, cols: None })
                .collect(),
        };
        assert!(self.bound.insert(name.into(), buffer).is_none(), "duplicate bound buffer `{name}`");
    }

    fn insert_derived(&mut self, name: &str, dtype: DType, shape: &[u64]) {
        let buffer = Buffer {
            dtype,
            shape: shape.iter().copied().map(Dim::Const).collect(),
            kind: BufferKind::Carry,
            placement: Placement::Device,
            fill: None,
            domain: None,
            export: false,
            of: None,
            group: None,
            bind: Vec::new(),
        };
        assert!(self.derived.insert(name.into(), buffer).is_none(), "duplicate derived buffer `{name}`");
    }
}

fn named<'a>(checkpoint: &'a BTreeMap<String, TensorContract>, segment: &Segment) -> Option<&'a TensorContract> {
    let TensorSource::Named(name) = &segment.tensor else { return None };
    checkpoint.get(name)
}

fn dims(shape: &[Dim]) -> Option<u64> {
    shape.iter().try_fold(1_u64, |product, dim| match dim {
        Dim::Const(value) => product.checked_mul(*value),
        Dim::Var(_) => None,
    })
}

fn dtype_name(dtype: DType) -> String {
    match dtype {
        DType::Bf16 => "BF16",
        DType::Fp8E4m3 => "F8_E4M3",
        DType::F32 => "F32",
        other => return other.to_string().to_uppercase(),
    }
    .into()
}
