use std::collections::{BTreeMap, BTreeSet};

use kern_manifest::Verified;
use kern_manifest::types::{BufferKind, Dim, Manifest, TensorSource};
use serde::Serialize;

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct StateInventory {
    pub bytes_per_token: u64,
    pub bytes_per_sequence: u64,
    pub fixed_bytes: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct BufferInventory {
    pub inputs: usize,
    pub outputs: usize,
    pub weights: usize,
    pub workspaces: usize,
    pub carries: usize,
    pub peers: usize,
    pub weight_segments: usize,
    pub checkpoint_tensors: usize,
    pub dtypes: BTreeMap<String, usize>,
    pub weight_dtypes: BTreeMap<String, usize>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum RowsInventory {
    Constant(u64),
    Variable(String),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BatchInventory {
    pub groups: u64,
    pub rows: RowsInventory,
    pub span: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ProgramInventory {
    pub batch: Option<BatchInventory>,
    pub once: bool,
    pub graph: bool,
    pub calls: usize,
    pub op_counts: BTreeMap<String, usize>,
    pub layers: BTreeSet<u16>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ManifestInventory {
    pub model: String,
    pub variables: BTreeMap<String, u64>,
    pub states: BTreeMap<String, StateInventory>,
    pub buffers: BufferInventory,
    pub modules: usize,
    pub ops: usize,
    pub programs: BTreeMap<String, ProgramInventory>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct NamedDiff {
    pub missing: Vec<String>,
    pub extra: Vec<String>,
    pub changed: Vec<String>,
}

impl NamedDiff {
    pub fn is_empty(&self) -> bool {
        self.missing.is_empty() && self.extra.is_empty() && self.changed.is_empty()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ProgramDiff {
    pub name: String,
    pub reference_calls: usize,
    pub candidate_calls: usize,
    pub batch_changed: bool,
    pub once_changed: bool,
    pub graph_changed: bool,
    pub calls_changed: bool,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct ManifestDiff {
    pub model_changed: bool,
    pub variables: NamedDiff,
    pub states: NamedDiff,
    pub buffers: NamedDiff,
    pub modules: NamedDiff,
    pub ops: NamedDiff,
    pub programs: NamedDiff,
    pub program_details: Vec<ProgramDiff>,
}

impl ManifestDiff {
    pub fn between(reference: &Manifest, candidate: &Manifest) -> Self {
        let programs = named_diff(&reference.programs, &candidate.programs);
        let common: BTreeSet<_> =
            reference.programs.keys().filter(|name| candidate.programs.contains_key(*name)).collect();
        let program_details = common
            .into_iter()
            .filter_map(|name| {
                let a = &reference.programs[name];
                let b = &candidate.programs[name];
                let calls_changed = json(a.calls.as_slice()) != json(b.calls.as_slice());
                let batch_changed = json(&a.batch) != json(&b.batch);
                let changed = batch_changed || a.once != b.once || a.graph != b.graph || calls_changed;
                changed.then(|| ProgramDiff {
                    name: name.clone(),
                    reference_calls: a.calls.len(),
                    candidate_calls: b.calls.len(),
                    batch_changed,
                    once_changed: a.once != b.once,
                    graph_changed: a.graph != b.graph,
                    calls_changed,
                })
            })
            .collect();
        Self {
            model_changed: reference.model != candidate.model,
            variables: named_diff(&reference.vars, &candidate.vars),
            states: named_diff(&reference.states, &candidate.states),
            buffers: named_diff(&reference.buffers, &candidate.buffers),
            modules: named_diff(&reference.modules, &candidate.modules),
            ops: named_diff(&reference.ops, &candidate.ops),
            programs,
            program_details,
        }
    }

    pub fn is_empty(&self) -> bool {
        !self.model_changed
            && self.variables.is_empty()
            && self.states.is_empty()
            && self.buffers.is_empty()
            && self.modules.is_empty()
            && self.ops.is_empty()
            && self.programs.is_empty()
            && self.program_details.is_empty()
    }
}

impl ManifestInventory {
    pub fn from_verified(manifest: &Verified) -> Self {
        let mut buffers = BufferInventory::default();
        let mut checkpoint_tensors = BTreeSet::new();
        for buffer in manifest.buffers.values() {
            *buffers.dtypes.entry(buffer.dtype.to_string()).or_insert(0) += 1;
            match buffer.kind {
                BufferKind::Input => buffers.inputs += 1,
                BufferKind::Output => buffers.outputs += 1,
                BufferKind::Weight => {
                    buffers.weights += 1;
                    *buffers.weight_dtypes.entry(buffer.dtype.to_string()).or_insert(0) += 1;
                }
                BufferKind::Workspace => buffers.workspaces += 1,
                BufferKind::Carry => buffers.carries += 1,
                BufferKind::Peer => buffers.peers += 1,
            }
            buffers.weight_segments += buffer.bind.len();
            for segment in &buffer.bind {
                match &segment.tensor {
                    TensorSource::Named(name) => {
                        checkpoint_tensors.insert(name.clone());
                    }
                    TensorSource::Ranked { tensors, .. } => {
                        checkpoint_tensors.extend(tensors.iter().cloned());
                    }
                }
            }
        }
        buffers.checkpoint_tensors = checkpoint_tensors.len();

        let states = manifest
            .states
            .iter()
            .map(|(name, state)| {
                (
                    name.clone(),
                    StateInventory {
                        bytes_per_token: state.bytes_per_token,
                        bytes_per_sequence: state.bytes_per_seq,
                        fixed_bytes: state.bytes,
                    },
                )
            })
            .collect();
        let programs = manifest
            .programs
            .iter()
            .map(|(name, program)| {
                let mut op_counts = BTreeMap::new();
                let mut layers = BTreeSet::new();
                for call in &program.calls {
                    *op_counts.entry(call.op.clone()).or_insert(0) += 1;
                    if let Some(layer) = call.label.as_deref().and_then(layer_from_label) {
                        layers.insert(layer);
                    }
                }
                let batch = program.batch.as_ref().map(|batch| BatchInventory {
                    groups: batch.groups,
                    rows: match &batch.rows {
                        Dim::Const(rows) => RowsInventory::Constant(*rows),
                        Dim::Var(rows) => RowsInventory::Variable(rows.clone()),
                    },
                    span: batch.span.clone(),
                });
                (
                    name.clone(),
                    ProgramInventory {
                        batch,
                        once: program.once,
                        graph: program.graph,
                        calls: program.calls.len(),
                        op_counts,
                        layers,
                    },
                )
            })
            .collect();

        Self {
            model: manifest.model.clone(),
            variables: manifest.vars.iter().map(|(name, var)| (name.clone(), var.max)).collect(),
            states,
            buffers,
            modules: manifest.modules.len(),
            ops: manifest.ops.len(),
            programs,
        }
    }
}

fn layer_from_label(label: &str) -> Option<u16> {
    label.strip_prefix('l')?.split_once('.')?.0.parse().ok()
}

fn json<T: Serialize + ?Sized>(value: &T) -> serde_json::Value {
    serde_json::to_value(value).expect("manifest declarations serialize")
}

fn named_diff<T: Serialize>(reference: &BTreeMap<String, T>, candidate: &BTreeMap<String, T>) -> NamedDiff {
    let reference_names: BTreeSet<_> = reference.keys().cloned().collect();
    let candidate_names: BTreeSet<_> = candidate.keys().cloned().collect();
    let missing = reference_names.difference(&candidate_names).cloned().collect();
    let extra = candidate_names.difference(&reference_names).cloned().collect();
    let changed = reference_names
        .intersection(&candidate_names)
        .filter(|name| json(&reference[*name]) != json(&candidate[*name]))
        .cloned()
        .collect();
    NamedDiff { missing, extra, changed }
}
