use std::collections::BTreeSet;
use std::fmt;
use std::fs;
use std::path::Path;

use kern_manifest::Verified;
use kern_manifest::types::TensorSource;
use serde::Serialize;

use crate::model::{LayerKind, Qwen38Contract};

use super::{ManifestDiff, ManifestInventory};
use crate::lower::{Qwen38Declarations, Qwen38ProgramSkeletons};

#[derive(Debug)]
pub struct Qwen38Oracle {
    verified: Verified,
    inventory: ManifestInventory,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Qwen38StatePacking {
    pub physical_state: String,
    pub logical_states: Vec<String>,
    pub layers: u16,
    pub recurrent_bytes_per_layer: u64,
    pub convolution_bytes_per_layer: u64,
    pub padding_bytes_per_layer: u64,
    pub bytes_per_sequence: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Qwen38OracleReport {
    pub oracle_role: String,
    pub authoritative_for: Vec<String>,
    pub not_authoritative_for: Vec<String>,
    pub contract_model: String,
    pub checkpoint_repository: String,
    pub checkpoint_revision: String,
    pub inventory: ManifestInventory,
    pub state_packing: Qwen38StatePacking,
}

#[derive(Debug)]
pub enum OracleError {
    Read(std::io::Error),
    InvalidManifest(String),
    Contract(Vec<String>),
}

impl fmt::Display for OracleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read(error) => write!(f, "reading oracle: {error}"),
            Self::InvalidManifest(error) => write!(f, "invalid oracle manifest: {error}"),
            Self::Contract(errors) => {
                f.write_str("Qwen3.8 oracle does not match the pinned model contract:")?;
                for error in errors {
                    write!(f, "\n  - {error}")?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for OracleError {}

impl Qwen38Oracle {
    pub const MODEL_LABEL: &'static str = "qwen3.8-27b-trtllm-gen";

    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, OracleError> {
        let json = fs::read_to_string(path).map_err(OracleError::Read)?;
        Self::from_json(&json)
    }

    pub fn from_json(json: &str) -> Result<Self, OracleError> {
        let verified = Verified::from_json(json).map_err(|errors| OracleError::InvalidManifest(errors.to_string()))?;
        let inventory = ManifestInventory::from_verified(&verified);
        Ok(Self { verified, inventory })
    }

    pub fn manifest(&self) -> &Verified {
        &self.verified
    }

    pub fn inventory(&self) -> &ManifestInventory {
        &self.inventory
    }

    pub fn report(&self) -> Qwen38OracleReport {
        Qwen38OracleReport {
            oracle_role: "bf16_execution_oracle".into(),
            authoritative_for: vec![
                "model_topology".into(),
                "state_packing".into(),
                "program_call_graph".into(),
                "kernel_abi".into(),
                "serving_protocol".into(),
            ],
            not_authoritative_for: vec![
                "official_fp8_weight_dtype".into(),
                "block_scale_contract".into(),
                "fp8_kernel_numerics".into(),
            ],
            contract_model: Qwen38Contract::MODEL.into(),
            checkpoint_repository: Qwen38Contract::REPOSITORY.into(),
            checkpoint_revision: Qwen38Contract::REVISION.into(),
            inventory: self.inventory.clone(),
            state_packing: Qwen38StatePacking {
                physical_state: "gdn".into(),
                logical_states: vec!["gdn.recurrent".into(), "gdn.convolution".into()],
                layers: Qwen38Contract::GATED_DELTA_LAYERS,
                recurrent_bytes_per_layer: Qwen38Contract::recurrent_bytes_per_layer(),
                convolution_bytes_per_layer: Qwen38Contract::convolution_bytes_per_layer(),
                padding_bytes_per_layer: Qwen38Contract::ORACLE_GDN_PADDING_BYTES_PER_LAYER,
                bytes_per_sequence: Qwen38Contract::oracle_gdn_bytes_per_sequence(),
            },
        }
    }

    pub fn diff(&self, candidate: &Verified) -> ManifestDiff {
        ManifestDiff::between(&self.verified, candidate)
    }

    /// Differences between compiler-lowered label/op skeletons and the
    /// imported handwritten programs. An empty map is exact topology parity.
    pub fn skeleton_differences(&self) -> std::collections::BTreeMap<String, Vec<String>> {
        let skeletons = Qwen38ProgramSkeletons::lower();
        let mut differences = std::collections::BTreeMap::new();
        for (name, skeleton) in skeletons.programs {
            let Some(program) = self.verified.programs.get(&name) else {
                differences.insert(name, vec!["program missing from oracle".into()]);
                continue;
            };
            let mut program_differences = Vec::new();
            let pairs = skeleton.calls.len().max(program.calls.len());
            for index in 0..pairs {
                let generated = skeleton.calls.get(index).map(|call| (call.label.as_str(), call.op.as_str()));
                let oracle = program
                    .calls
                    .get(index)
                    .map(|call| (call.label.as_deref().unwrap_or("<unlabeled>"), call.op.as_str()));
                if generated != oracle {
                    program_differences.push(format!("call {index}: generated {generated:?}, oracle {oracle:?}"));
                }
            }
            if !program_differences.is_empty() {
                differences.insert(name, program_differences);
            }
        }
        differences
    }

    pub fn validate_contract(&self) -> Result<(), OracleError> {
        let manifest = &self.verified;
        let inventory = &self.inventory;
        let mut errors = Vec::new();
        expect_eq(&mut errors, "model label", &manifest.model, &Self::MODEL_LABEL.to_string());
        expect_eq(&mut errors, "tokens.max", &inventory.variables.get("tokens").copied(), &Some(8_192));
        expect_eq(&mut errors, "seqs.max", &inventory.variables.get("seqs").copied(), &Some(128));
        expect_eq(&mut errors, "state count", &inventory.states.len(), &2);
        expect_eq(
            &mut errors,
            "kv.bytes_per_token",
            &inventory.states.get("kv").map(|state| state.bytes_per_token),
            &Some(Qwen38Contract::oracle_kv_bytes_per_token()),
        );
        expect_eq(
            &mut errors,
            "gdn.bytes_per_sequence",
            &inventory.states.get("gdn").map(|state| state.bytes_per_sequence),
            &Some(Qwen38Contract::oracle_gdn_bytes_per_sequence()),
        );
        expect_eq(&mut errors, "buffer count", &manifest.buffers.len(), &915);
        expect_eq(&mut errors, "weight buffers", &inventory.buffers.weights, &659);
        expect_eq(
            &mut errors,
            "weight dtypes",
            &inventory.buffers.weight_dtypes,
            &std::collections::BTreeMap::from([("bf16".into(), 659)]),
        );
        expect_eq(&mut errors, "weight segments", &inventory.buffers.weight_segments, &851);
        expect_eq(&mut errors, "unique checkpoint tensors", &inventory.buffers.checkpoint_tensors, &851);
        expect_eq(&mut errors, "module count", &inventory.modules, &22);
        expect_eq(&mut errors, "op count", &inventory.ops, &40);
        let declarations = Qwen38Declarations::lower();
        if serde_json::to_value(&declarations.variables).expect("variables serialize")
            != serde_json::to_value(&manifest.vars).expect("variables serialize")
        {
            errors.push("compiler-lowered variables differ from the oracle".into());
        }
        if serde_json::to_value(&declarations.states).expect("states serialize")
            != serde_json::to_value(&manifest.states).expect("states serialize")
        {
            errors.push("compiler-lowered states differ from the oracle".into());
        }

        check_program(&mut errors, inventory, "prefill", 1_127, false, false, Some((1, "tokens")));
        check_program(&mut errors, inventory, "decode", 646, false, true, Some((1, "1")));
        check_program(&mut errors, inventory, "decode_batch", 534, false, true, Some((128, "1")));
        check_program(&mut errors, inventory, "load", 217, true, false, None);

        let required_ops = [
            "embedding",
            "gemm",
            "gdn_conv",
            "gdn_step",
            "attn",
            "attn_batch",
            "attn_prefill",
            "gemma_fused_norm",
            "sigmoid_mul",
            "argmax",
        ];
        for op in required_ops {
            if !manifest.ops.contains_key(op) {
                errors.push(format!("missing required op `{op}`"));
            }
        }

        for program in ["prefill", "decode", "decode_batch"] {
            let layers = inventory.programs.get(program).map(|program| &program.layers);
            let expected: BTreeSet<u16> = (0..Qwen38Contract::LAYERS).collect();
            if layers != Some(&expected) {
                errors.push(format!("program `{program}` does not cover all {} layers", Qwen38Contract::LAYERS));
            }
        }
        validate_layer_ops(&mut errors, manifest);
        validate_weight_bindings(&mut errors, manifest);
        for (program, differences) in self.skeleton_differences() {
            errors.push(format!(
                "compiler skeleton for `{program}` differs at {} calls; first: {}",
                differences.len(),
                differences.first().expect("non-empty differences")
            ));
        }

        if errors.is_empty() { Ok(()) } else { Err(OracleError::Contract(errors)) }
    }
}

fn expect_eq<T: fmt::Debug + PartialEq>(errors: &mut Vec<String>, what: &str, actual: &T, expected: &T) {
    if actual != expected {
        errors.push(format!("{what}: expected {expected:?}, found {actual:?}"));
    }
}

fn check_program(
    errors: &mut Vec<String>,
    inventory: &ManifestInventory,
    name: &str,
    calls: usize,
    once: bool,
    graph: bool,
    batch: Option<(u64, &str)>,
) {
    let Some(program) = inventory.programs.get(name) else {
        errors.push(format!("missing program `{name}`"));
        return;
    };
    expect_eq(errors, &format!("{name}.calls"), &program.calls, &calls);
    expect_eq(errors, &format!("{name}.once"), &program.once, &once);
    expect_eq(errors, &format!("{name}.graph"), &program.graph, &graph);
    let actual_batch = program.batch.as_ref().map(|batch| {
        let rows = match &batch.rows {
            super::inventory::RowsInventory::Constant(rows) => rows.to_string(),
            super::inventory::RowsInventory::Variable(rows) => rows.clone(),
        };
        (batch.groups, rows)
    });
    let expected_batch = batch.map(|(groups, rows)| (groups, rows.to_string()));
    expect_eq(errors, &format!("{name}.batch"), &actual_batch, &expected_batch);
}

fn validate_layer_ops(errors: &mut Vec<String>, manifest: &Verified) {
    for layer in 0..Qwen38Contract::LAYERS {
        let expected = match Qwen38Contract::layer_kind(layer) {
            LayerKind::GatedDelta => [("prefill", "chunk_h"), ("decode", "gdn_step"), ("decode_batch", "gdn_step")],
            LayerKind::FullAttention => {
                [("prefill", "attn_prefill"), ("decode", "attn"), ("decode_batch", "attn_batch")]
            }
        };
        for (program, op) in expected {
            let prefix = format!("l{layer}.");
            let found = manifest.programs[program]
                .calls
                .iter()
                .any(|call| call.op == op && call.label.as_deref().is_some_and(|label| label.starts_with(&prefix)));
            if !found {
                errors.push(format!("program `{program}` layer {layer} missing `{op}`"));
            }
        }
    }
}

fn validate_weight_bindings(errors: &mut Vec<String>, manifest: &Verified) {
    let mut tensors = BTreeSet::new();
    for buffer in manifest.buffers.values() {
        for segment in &buffer.bind {
            match &segment.tensor {
                TensorSource::Named(name) => {
                    tensors.insert(name.as_str());
                }
                TensorSource::Ranked { tensors: ranked, .. } => tensors.extend(ranked.iter().map(String::as_str)),
            }
        }
    }
    for tensor in ["model.language_model.embed_tokens.weight", "model.language_model.norm.weight", "lm_head.weight"] {
        require_tensor(errors, &tensors, tensor);
    }
    for layer in 0..Qwen38Contract::LAYERS {
        let prefix = format!("model.language_model.layers.{layer}");
        for suffix in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ] {
            require_tensor(errors, &tensors, &format!("{prefix}.{suffix}"));
        }
        let suffixes: &[&str] = match Qwen38Contract::layer_kind(layer) {
            LayerKind::GatedDelta => &[
                "linear_attn.A_log",
                "linear_attn.conv1d.weight",
                "linear_attn.dt_bias",
                "linear_attn.in_proj_a.weight",
                "linear_attn.in_proj_b.weight",
                "linear_attn.in_proj_qkv.weight",
                "linear_attn.in_proj_z.weight",
                "linear_attn.norm.weight",
                "linear_attn.out_proj.weight",
            ],
            LayerKind::FullAttention => &[
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.q_norm.weight",
                "self_attn.k_norm.weight",
                "self_attn.o_proj.weight",
            ],
        };
        for suffix in suffixes {
            require_tensor(errors, &tensors, &format!("{prefix}.{suffix}"));
        }
    }
}

fn require_tensor(errors: &mut Vec<String>, tensors: &BTreeSet<&str>, tensor: &str) {
    if !tensors.contains(tensor) {
        errors.push(format!("missing checkpoint tensor binding `{tensor}`"));
    }
}
