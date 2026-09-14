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
//! `tiling` defines SM90 legality and candidate ordering; `jit` renders and
//! loads those kernels. Graph equivalences remain in egglog below.

mod contract;
pub mod jit;
mod scratch;
mod shared;
mod tiling;
pub use shared::BlockScaledQuantize;

/// Experimental shared activation candidate opt-in. Include this stable fact
/// in `CompileOptions::compiler_facts` and in the caller's artifact identity.
/// The existing combined provider remains selectable when this is enabled.
pub const SHARED_QUANTIZATION_COMPILER_FACT: &str = "(enable-shared-fp8-quantization)";

use std::sync::{Arc, Mutex};

use cudarc::driver::{CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg};
use orbitkv_compiler::{
    dtype::DType,
    egglog_utils::{
        SerializedEGraph,
        api::{Rule, SortDef, sort},
        base::{EXPRESSION, I64, OP_KIND},
        extract_expr,
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

#[allow(dead_code)]
#[derive(Clone)]
pub struct DeepGemmImpl<const PREQUANTIZED: bool> {
    rows: Expression,
    output_features: usize,
    input_features: usize,
    variant: usize,
    provider: String,
    scratch: Arc<Mutex<Option<Arc<Scratch>>>>,
}

pub type DeepGemm = DeepGemmImpl<false>;
pub type PrequantizedDeepGemm = DeepGemmImpl<true>;

impl<const PREQUANTIZED: bool> std::fmt::Debug for DeepGemmImpl<PREQUANTIZED> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct(Self::op_name())
            .field("rows", &self.rows)
            .field("output_features", &self.output_features)
            .field("input_features", &self.input_features)
            .field("variant", &self.variant)
            .field("provider", &self.provider)
            .finish()
    }
}

impl<const PREQUANTIZED: bool> Default for DeepGemmImpl<PREQUANTIZED> {
    fn default() -> Self {
        Self {
            rows: Expression::default(),
            output_features: 0,
            input_features: 0,
            variant: 0,
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
                ("output_features", EXPRESSION),
                ("input_features", EXPRESSION),
                ("variant", I64),
                ("provider", orbitkv_compiler::egglog_utils::base::STRING),
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
            5,
            "DeepGEMM schedule lacks provider identity; rebuild the selected schedule"
        );
        let integer = |node: &ENodeId| {
            egraph.enodes[node]
                .0
                .replace('\"', "")
                .parse::<usize>()
                .unwrap()
        };
        let provider = egraph.enodes[kind_children[4]].0.replace('\"', "");
        let current_provider = jit::provider_identity().unwrap_or_else(|error| panic!("{error}"));
        crate::providers::provider_source::validate_provider_identity(&provider, &current_provider)
            .unwrap_or_else(|error| panic!("{error}"));
        let output_features = extract_expr(egraph, kind_children[1], expr_cache)
            .unwrap()
            .exec(&Default::default())
            .unwrap();
        let input_features = extract_expr(egraph, kind_children[2], expr_cache)
            .unwrap()
            .exec(&Default::default())
            .unwrap();
        let variant = integer(kind_children[3]);
        (
            LLIROp::new::<dyn HostOp>(Box::new(Self {
                rows: extract_expr(egraph, kind_children[0], expr_cache).unwrap(),
                output_features,
                input_features,
                variant,
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
        (0..jit::SEARCH_VARIANTS)
            .map(|variant| {
                Rule::raw(format!(
                    include_str!("deepgemm/provider_rewrite.egg.in"),
                    BLOCK = BLOCK,
                    activation = activation,
                    gate = gate,
                    name = name,
                    prepare = prepare,
                    provider = provider,
                    variant = variant,
                ))
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
        Ok((rows, self.output_features, self.input_features))
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
        let (m, n, k) = self.dimensions(dyn_map)?;
        let config = jit::Config::for_variant(stream, m, n, k, self.variant)?;
        let _ = jit::ensure_compiled(
            crate::target::CudaTarget::from_context(stream.context())?,
            config,
        )?;
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
        validate_buffers::<PREQUANTIZED>(self_node, inputs, buffers, m, n, k)?;
        let config = jit::Config::for_variant(stream, m, n, k, self.variant)?;
        let library = jit::ensure_compiled(
            crate::target::CudaTarget::from_context(stream.context())?,
            config,
        )?;
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
                "DeepGEMM launch failed for M={m}, N={n}, K={k}, variant={}: {}",
                self.variant,
                library.last_error()
            );
        }
        Ok(())
    }

    fn output_size(&self) -> Expression {
        self.rows * self.output_features
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
        jit::Config::validate_shape(m, n, k, self.variant).map_err(|_| {
            ResourceViolation::HostResourcePlanning {
                name: "DeepGEMM SM90 1D2D legality",
            }
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
        if PREQUANTIZED {
            return Some(match self.variant {
                0 => "DeepGemmPrequantizedV0",
                1 => "DeepGemmPrequantizedV1",
                2 => "DeepGemmPrequantizedV2",
                3 => "DeepGemmPrequantizedV3",
                _ => "DeepGemmPrequantized",
            });
        }
        Some(match self.variant {
            0 => "DeepGemmV0",
            1 => "DeepGemmV1",
            2 => "DeepGemmV2",
            3 => "DeepGemmV3",
            _ => "DeepGemm",
        })
    }
}

#[cfg(test)]
#[path = "../../tests/unit/providers/deepgemm/mod.rs"]
mod tests;
