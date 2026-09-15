//! Searchable DeepGEMM provider for 128x128 block-scaled FP8 linear operators.
//!
//! `orbitkv_ops::ops::linear` owns the numerical graph contract. These egglog
//! rules contribute equivalent CUDA implementations. An independent portable
//! implementation lives under tests and is never a deployment fallback.
//! OrbitKV therefore profiles the alternatives instead of selecting a kernel
//! with a model- or device-name branch.
//!
//! `contract` owns the versioned numeric/storage ABI, `shared` exposes its
//! graph-owned intermediate, and `scratch` retains private capture resources.
//! `tiling` defines SM90 legality and candidate ordering; `selection` exposes
//! explicit, bounded tile descriptors to egglog. `jit` prepares their native
//! libraries before execution. Graph equivalences remain in egglog below.

mod contract;
pub mod jit;
mod scratch;
mod selection;
mod shared;
mod tiling;
pub use shared::BlockScaledQuantize;

/// Experimental shared activation candidate opt-in. Include this stable fact
/// in `CompileOptions::compiler_facts` and in the caller's artifact identity.
/// The existing combined provider remains selectable when this is enabled.
pub const SHARED_QUANTIZATION_COMPILER_FACT: &str = "(enable-shared-fp8-quantization)";

use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg};
use orbitkv_compiler::{
    dtype::DType,
    egglog_utils::{
        SerializedEGraph,
        api::{Rule, SortDef, sort},
        base::{EXPRESSION, OP_KIND, STRING},
        extract_expr,
        primitives::EgglogPrimitive,
    },
    op::{EgglogOp, LLIROp},
    prelude::{ENodeId, Expression, FxHashMap, NodeIndex},
};

use crate::{
    compile_module_image_for_current_device,
    providers::{DeviceBuffer, HostOp},
    resource::{HostDeviceMemoryPlan, ResourceViolation},
};

use contract::{
    BF16_BYTES, PackedActivationLayout, SCALE_BLOCK as BLOCK, SCALE_BYTES, scratch_bytes,
};
#[cfg(test)]
use orbitkv_compiler::prelude::GraphTensor;
use orbitkv_ops::ops::linear::BLOCK_SCALED_LINEAR_DECLARATIONS;
#[cfg(test)]
use orbitkv_ops::ops::linear::{BlockScaledLinearSpec, block_scaled_linear};
use scratch::Scratch;
use selection::{Selection, TileCandidate};

fn validate_buffers<const PREQUANTIZED: bool>(
    self_node: NodeIndex,
    inputs: &[NodeIndex],
    buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
    m: usize,
    n: usize,
    k: usize,
) -> anyhow::Result<()> {
    let lengths = inputs
        .iter()
        .copied()
        .chain(std::iter::once(self_node))
        .filter_map(|node| buffers.get(&node).map(|buffer| (node, buffer.len())))
        .collect();
    validate_buffer_lengths::<PREQUANTIZED>(self_node, inputs, &lengths, m, n, k)
        .map_err(|error| anyhow::anyhow!(error))
}

fn validate_buffer_lengths<const PREQUANTIZED: bool>(
    self_node: NodeIndex,
    inputs: &[NodeIndex],
    buffers: &FxHashMap<NodeIndex, usize>,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), ResourceViolation> {
    let required = [
        if PREQUANTIZED {
            Some(PackedActivationLayout::new(m, k)?.total_bytes)
        } else {
            m.checked_mul(k).and_then(|v| v.checked_mul(BF16_BYTES))
        },
        n.checked_mul(k),
        n.div_ceil(BLOCK)
            .checked_mul(k.div_ceil(BLOCK))
            .and_then(|v| v.checked_mul(SCALE_BYTES)),
        m.checked_mul(n).and_then(|v| v.checked_mul(BF16_BYTES)),
    ];
    let required = required.into_iter().collect::<Option<Vec<_>>>().ok_or(
        ResourceViolation::ArithmeticOverflow {
            resource: "BlockScaledLinear tensor bytes",
        },
    )?;
    let nodes = [
        inputs.first().copied(),
        inputs.get(1).copied(),
        inputs.get(2).copied(),
        Some(self_node),
    ];
    if inputs.len() != 3
        || nodes
            .into_iter()
            .flatten()
            .zip(required)
            .any(|(node, required)| buffers.get(&node).is_none_or(|bytes| *bytes < required))
    {
        return Err(ResourceViolation::HostResourcePlanning {
            name: "BlockScaledLinear buffers",
        });
    }
    Ok(())
}

#[derive(Clone)]
pub struct DeepGemmImpl<const PREQUANTIZED: bool> {
    rows: Expression,
    selection: Selection,
    prepared: Arc<OnceLock<&'static jit::Library>>,
    provider: String,
    scratch: Arc<Mutex<Option<Arc<Scratch>>>>,
}

pub type DeepGemm = DeepGemmImpl<false>;
pub type PrequantizedDeepGemm = DeepGemmImpl<true>;

impl<const PREQUANTIZED: bool> std::fmt::Debug for DeepGemmImpl<PREQUANTIZED> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct(Self::op_name())
            .field("rows", &self.rows)
            .field("selection", &self.selection)
            .field("provider", &self.provider)
            .finish()
    }
}

impl<const PREQUANTIZED: bool> Default for DeepGemmImpl<PREQUANTIZED> {
    fn default() -> Self {
        Self {
            rows: Expression::default(),
            selection: Selection::default(),
            prepared: Arc::new(OnceLock::new()),
            provider: String::new(),
            scratch: Arc::new(Mutex::new(None)),
        }
    }
}

impl<const PREQUANTIZED: bool> EgglogOp for DeepGemmImpl<PREQUANTIZED> {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            Self::op_name(),
            &[
                ("rows", EXPRESSION),
                ("selection", STRING),
                ("provider", STRING),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        3
    }

    fn egglog_declarations(&self) -> Vec<String> {
        vec![
            BLOCK_SCALED_LINEAR_DECLARATIONS.to_owned(),
            crate::target::DECLARATIONS.to_owned(),
        ]
    }

    fn egglog_primitives(&self) -> Vec<EgglogPrimitive> {
        vec![EgglogPrimitive::new::<TileCandidate>()]
    }

    fn rewrites(&self) -> Vec<Rule> {
        let Ok(provider) = jit::provider_identity() else {
            return vec![];
        };
        Self::provider_rewrites(&provider)
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        _list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        assert_eq!(
            kind_children.len(),
            3,
            "DeepGEMM schedule lacks an explicit tile; rebuild the selected schedule"
        );
        let string = |node: &ENodeId| {
            serde_json::from_str::<String>(&egraph.enodes[node].0)
                .expect("DeepGEMM string metadata")
        };
        let provider = string(kind_children[2]);
        let current_provider = jit::provider_identity().unwrap_or_else(|error| panic!("{error}"));
        crate::providers::provider_source::validate_provider_identity(&provider, &current_provider)
            .unwrap_or_else(|error| panic!("{error}"));
        let selection: Selection = serde_json::from_str(&string(kind_children[1]))
            .expect("DeepGEMM explicit tile descriptor");
        selection
            .validate()
            .unwrap_or_else(|error| panic!("{error}"));
        (
            LLIROp::new::<dyn HostOp>(Box::new(Self {
                rows: extract_expr(egraph, kind_children[0], expr_cache).unwrap(),
                selection,
                prepared: Arc::new(OnceLock::new()),
                provider,
                scratch: Arc::new(Mutex::new(None)),
            }) as Box<dyn HostOp>),
            input_enodes,
        )
    }
}

impl<const PREQUANTIZED: bool> DeepGemmImpl<PREQUANTIZED> {
    fn op_name() -> &'static str {
        if PREQUANTIZED {
            "DeepGemmPrequantized"
        } else {
            "DeepGemm"
        }
    }

    fn provider_rewrites(provider: &str) -> Vec<Rule> {
        let name = Self::op_name();
        let (gate, prepare, activation) = if PREQUANTIZED {
            (
                SHARED_QUANTIZATION_COMPILER_FACT,
                format!(
                    "(let ?packed (Op (BlockScaledQuantize ?m ?k \"{}\" \"{provider}\")
                        (ICons ?x (INil))))
                     (set (dtype ?packed) (U8))",
                    contract::PACKED_ACTIVATION_ABI,
                ),
                "?packed",
            )
        } else {
            ("", String::new(), "?x")
        };
        // Constants work without interval analysis. Dynamic expressions must
        // have an explicit finite bucket bound; missing bounds admit no tile.
        [
            ("constant", "(= ?m (MNum ?row_limit))"),
            ("bounded", "(= ?row_limit (upper ?m))"),
        ]
        .into_iter()
        .flat_map(|(bound_name, row_bound)| {
            let prepare = &prepare;
            (0..selection::SEARCH_VARIANTS).map(move |rank| {
                Rule::raw(format!(
                    include_str!("deepgemm/provider_rewrite.egg.in"),
                    BLOCK = BLOCK,
                    activation = activation,
                    gate = gate,
                    name = name,
                    prepare = prepare,
                    provider = provider,
                    rank = rank,
                    row_bound = row_bound,
                    bound_name = bound_name,
                ))
            })
        })
        .collect()
    }

    fn dimensions(
        &self,
        dyn_map: &orbitkv_compiler::prelude::DynMap,
    ) -> anyhow::Result<(usize, usize, usize)> {
        let rows = self
            .rows
            .exec(dyn_map)
            .ok_or_else(|| anyhow::anyhow!("unresolved DeepGEMM row dimension"))?;
        Ok((
            rows,
            self.selection.config.output_features,
            self.selection.config.input_features,
        ))
    }
}

impl<const PREQUANTIZED: bool> HostOp for DeepGemmImpl<PREQUANTIZED> {
    fn provider_dependencies(&self) -> Vec<super::registry::ProviderId> {
        vec![super::registry::ProviderId::DeepGemm]
    }

    fn prepare_compilation(
        &self,
        stream: &Arc<CudaStream>,
        dyn_map: &orbitkv_compiler::prelude::DynMap,
    ) -> anyhow::Result<()> {
        stream.context().bind_to_thread()?;
        self.selection.validate()?;
        let (m, _, _) = self.dimensions(dyn_map)?;
        self.selection.validate_rows(m)?;
        let config = self.selection.config;
        let target = crate::target::CudaTarget::from_context(stream.context())?;
        target.hopper_architecture()?;
        let num_sms = usize::try_from(stream.context().attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        )?)?;
        anyhow::ensure!(
            config.num_sms == num_sms,
            "DeepGEMM selected SM count {} differs from execution device {num_sms}; rebuild the schedule",
            config.num_sms
        );
        if self.prepared.get().is_none() {
            let _stage = tracing::info_span!(target: "orbitkv::stage", "cuda.deepgemm.prepare", row_limit = self.selection.row_limit, config = ?config).entered();
            let library = jit::ensure_compiled(target, config)?;
            // Racing preparation can only install the same globally cached
            // library for this immutable descriptor.
            let _ = self.prepared.set(library);
        }
        Ok(())
    }

    fn execute(
        &self,
        stream: &Arc<CudaStream>,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &orbitkv_compiler::prelude::DynMap,
    ) -> anyhow::Result<()> {
        stream.context().bind_to_thread()?;
        let (m, n, k) = self.dimensions(dyn_map)?;
        if m == 0 {
            return Ok(());
        }
        self.selection.validate_rows(m)?;
        validate_buffers::<PREQUANTIZED>(self_node, inputs, buffers, m, n, k)?;
        let library = self.prepared.get().ok_or_else(|| {
            anyhow::anyhow!(
                "DeepGEMM kernel is not prepared; call prepare_compilation before execution"
            )
        })?;
        let status = if PREQUANTIZED {
            let layout =
                PackedActivationLayout::new(m, k).map_err(|error| anyhow::anyhow!(error))?;
            let packed = buffers[&inputs[0]].ptr();
            let scales = layout.scale_pointer(packed)?;
            unsafe {
                (library.run_prequantized)(
                    buffers[&inputs[1]].ptr() as *const std::ffi::c_void,
                    buffers[&inputs[2]].ptr() as *const std::ffi::c_void,
                    packed as *const std::ffi::c_void,
                    scales as *const std::ffi::c_void,
                    buffers[&self_node].ptr() as *mut std::ffi::c_void,
                    m as i32,
                    stream.cu_stream().cast(),
                )
            }
        } else {
            let mut scratch = self.scratch.lock().unwrap();
            let scratch = Scratch::prepare(&mut scratch, stream, m, k)?;
            unsafe {
                (library.run)(
                    buffers[&inputs[0]].ptr() as *const std::ffi::c_void,
                    buffers[&inputs[1]].ptr() as *const std::ffi::c_void,
                    buffers[&inputs[2]].ptr() as *const std::ffi::c_void,
                    scratch.quantized_ptr as *mut std::ffi::c_void,
                    scratch.scales_ptr as *mut std::ffi::c_void,
                    buffers[&self_node].ptr() as *mut std::ffi::c_void,
                    m as i32,
                    stream.cu_stream().cast(),
                )
            }
        };
        if status != 0 {
            anyhow::bail!(
                "DeepGEMM launch failed for M={m}, N={n}, K={k}, tile={:?}: {}",
                self.selection.config,
                library.last_error()
            );
        }
        Ok(())
    }

    fn output_size(&self) -> Expression {
        self.rows * self.selection.config.output_features
    }

    fn output_bytes(&self) -> Expression {
        self.output_size() * BF16_BYTES
    }

    fn output_dtype(&self) -> DType {
        DType::Bf16
    }

    fn cuda_graph_capture_arity(&self) -> Option<usize> {
        Some(3)
    }

    fn cuda_graph_capture_dyn_dims(&self) -> Vec<orbitkv_compiler::prelude::Symbol> {
        self.rows.dyn_vars()
    }

    fn prepare_cuda_graph_capture(
        &self,
        stream: &Arc<CudaStream>,
        _self_node: NodeIndex,
        _inputs: &[NodeIndex],
        _buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &orbitkv_compiler::prelude::DynMap,
    ) -> anyhow::Result<()> {
        let (m, _, k) = self.dimensions(dyn_map)?;
        if PREQUANTIZED {
            return Ok(());
        }
        Scratch::prepare(&mut self.scratch.lock().unwrap(), stream, m, k)?;
        Ok(())
    }

    fn cuda_graph_capture_resources(&self) -> Vec<super::CudaGraphCaptureResource> {
        self.scratch
            .lock()
            .unwrap()
            .iter()
            .cloned()
            .map(|scratch| scratch as super::CudaGraphCaptureResource)
            .collect()
    }

    fn device_memory_plan(
        &self,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffer_lengths: &FxHashMap<NodeIndex, usize>,
        dyn_map: &orbitkv_compiler::prelude::DynMap,
    ) -> Result<HostDeviceMemoryPlan, ResourceViolation> {
        let (m, n, k) =
            self.dimensions(dyn_map)
                .map_err(|_| ResourceViolation::HostResourcePlanning {
                    name: "DeepGEMM dimensions",
                })?;
        validate_buffer_lengths::<PREQUANTIZED>(self_node, inputs, buffer_lengths, m, n, k)?;
        self.selection
            .validate_rows(m)
            .map_err(|_| ResourceViolation::HostResourcePlanning {
                name: "DeepGEMM SM90 1D2D legality",
            })?;
        if PREQUANTIZED {
            // The packed result belongs to the graph arena, stays live until
            // its last consumer, and is already included in arena accounting.
            return Ok(HostDeviceMemoryPlan::default());
        }
        let (quantized_bytes, scale_bytes) = scratch_bytes(m, k)?;
        Ok(HostDeviceMemoryPlan {
            persistent_bytes: quantized_bytes
                .checked_add(scale_bytes)
                .ok_or(ResourceViolation::ArithmeticOverflow {
                    resource: "DeepGEMM activation scratch",
                })?
                .max(
                    self.scratch
                        .lock()
                        .unwrap()
                        .as_ref()
                        .map_or(0, |scratch| scratch.bytes()),
                ),
            ..Default::default()
        })
    }

    fn resource_buffer_nodes(&self, inputs: &[NodeIndex]) -> Vec<NodeIndex> {
        inputs.to_vec()
    }

    fn stats_name(&self) -> Option<&'static str> {
        Some(Self::op_name())
    }
}

#[cfg(test)]
#[path = "../../tests/unit/providers/deepgemm/mod.rs"]
mod tests;
