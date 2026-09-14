//! Graph-visible activation preparation for reusable block-scaled GEMMs.
//!
//! The output is one opaque byte allocation, not a generally usable FP8 tensor.
//! The matching GEMM consumes both planes using this versioned ABI. Ordinary
//! graph edges preserve the allocation through every consumer; no input is
//! mutated, aliased, or retained as hidden per-operation device scratch.

use super::*;
use orbitkv_compiler::egglog_utils::base::STRING;

use super::contract::PACKED_ACTIVATION_ABI;

/// A graph-owned packed FP8/scale intermediate, emitted only by egglog.
#[derive(Clone, Default)]
pub struct BlockScaledQuantize {
    pub(super) rows: Expression,
    pub(super) input_features: usize,
    pub(super) provider: String,
}

impl std::fmt::Debug for BlockScaledQuantize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockScaledQuantize")
            .field("rows", &self.rows)
            .field("input_features", &self.input_features)
            .field("packed_abi", &PACKED_ACTIVATION_ABI)
            .field("provider", &self.provider)
            .finish()
    }
}

impl EgglogOp for BlockScaledQuantize {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "BlockScaledQuantize",
            &[
                ("rows", EXPRESSION),
                ("input_features", EXPRESSION),
                ("packed_abi", STRING),
                ("provider", STRING),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        1
    }

    fn egglog_declarations(&self) -> Vec<String> {
        vec!["(relation enable-shared-fp8-quantization ())".to_owned()]
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
            4,
            "rebuild the packed activation schedule"
        );
        let abi = egraph.enodes[kind_children[2]].0.replace('"', "");
        assert_eq!(
            abi, PACKED_ACTIVATION_ABI,
            "packed activation ABI changed; rebuild the selected schedule"
        );
        let provider = egraph.enodes[kind_children[3]].0.replace('"', "");
        let current = jit::provider_identity().unwrap_or_else(|error| panic!("{error}"));
        crate::host::provider_source::validate_provider_identity(&provider, &current)
            .unwrap_or_else(|error| panic!("{error}"));
        (
            LLIROp::new::<dyn HostOp>(Box::new(Self {
                rows: extract_expr(egraph, kind_children[0], expr_cache).unwrap(),
                input_features: extract_expr(egraph, kind_children[1], expr_cache)
                    .unwrap()
                    .exec(&Default::default())
                    .unwrap(),
                provider,
            }) as Box<dyn HostOp>),
            input_enodes,
        )
    }
}

impl BlockScaledQuantize {
    fn layout(
        &self,
        dyn_map: &orbitkv_compiler::prelude::DynMap,
    ) -> Result<(usize, PackedActivationLayout), ResourceViolation> {
        let m = self
            .rows
            .exec(dyn_map)
            .ok_or(ResourceViolation::HostResourcePlanning {
                name: "BlockScaledQuantize unresolved rows",
            })?;
        Ok((m, PackedActivationLayout::new(m, self.input_features)?))
    }

    fn validate_buffers(
        &self,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        lengths: &FxHashMap<NodeIndex, usize>,
        dyn_map: &orbitkv_compiler::prelude::DynMap,
    ) -> Result<(), ResourceViolation> {
        let (m, layout) = self.layout(dyn_map)?;
        let input_bytes = m
            .checked_mul(self.input_features)
            .and_then(|v| v.checked_mul(BF16_BYTES))
            .ok_or(ResourceViolation::ArithmeticOverflow {
                resource: "BlockScaledQuantize BF16 input",
            })?;
        if inputs.len() != 1
            || lengths
                .get(&inputs[0])
                .is_none_or(|bytes| *bytes < input_bytes)
            || lengths
                .get(&self_node)
                .is_none_or(|bytes| *bytes < layout.total_bytes)
        {
            return Err(ResourceViolation::HostResourcePlanning {
                name: "BlockScaledQuantize input and packed output buffers",
            });
        }
        Ok(())
    }
}

impl HostOp for BlockScaledQuantize {
    fn prepare_compilation(
        &self,
        stream: &Arc<CudaStream>,
        dyn_map: &orbitkv_compiler::prelude::DynMap,
    ) -> anyhow::Result<()> {
        self.layout(dyn_map)
            .map_err(|error| anyhow::anyhow!(error))?;
        let _ = quantize_module(stream)?;
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
        let lengths = inputs
            .iter()
            .copied()
            .chain(std::iter::once(self_node))
            .filter_map(|node| buffers.get(&node).map(|buffer| (node, buffer.len())))
            .collect();
        self.validate_buffers(self_node, inputs, &lengths, dyn_map)
            .map_err(|error| anyhow::anyhow!(error))?;
        let (m, layout) = self
            .layout(dyn_map)
            .map_err(|error| anyhow::anyhow!(error))?;
        if m == 0 {
            return Ok(());
        }
        let module = quantize_module(stream)?;
        let output = buffers[&self_node].ptr();
        let scales = layout.scale_pointer(output)?;
        let input = buffers[&inputs[0]].ptr();
        let (m, k) = (m as i32, self.input_features as i32);
        unsafe {
            stream
                .launch_builder(&module.quantize)
                .arg(&output)
                .arg(&scales)
                .arg(&input)
                .arg(&m)
                .arg(&k)
                .launch(LaunchConfig {
                    grid_dim: ((k / BLOCK as i32) as u32, m as u32, 1),
                    block_dim: (BLOCK as u32, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    fn output_size(&self) -> Expression {
        self.output_bytes()
    }

    fn output_bytes(&self) -> Expression {
        PackedActivationLayout::output_bytes(self.rows, self.input_features)
    }

    fn output_dtype(&self) -> DType {
        DType::U8
    }

    fn cuda_graph_capture_arity(&self) -> Option<usize> {
        Some(1)
    }

    fn cuda_graph_capture_dyn_dims(&self) -> Vec<orbitkv_compiler::prelude::Symbol> {
        self.rows.dyn_vars()
    }

    fn prepare_cuda_graph_capture(
        &self,
        _stream: &Arc<CudaStream>,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &orbitkv_compiler::prelude::DynMap,
    ) -> anyhow::Result<()> {
        let lengths = buffers
            .iter()
            .map(|(node, buffer)| (*node, buffer.len()))
            .collect();
        self.validate_buffers(self_node, inputs, &lengths, dyn_map)
            .map_err(|error| anyhow::anyhow!(error))
    }

    fn device_memory_plan(
        &self,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffer_lengths: &FxHashMap<NodeIndex, usize>,
        dyn_map: &orbitkv_compiler::prelude::DynMap,
    ) -> Result<HostDeviceMemoryPlan, ResourceViolation> {
        self.validate_buffers(self_node, inputs, buffer_lengths, dyn_map)?;
        Ok(HostDeviceMemoryPlan::default())
    }

    fn resource_buffer_nodes(&self, inputs: &[NodeIndex]) -> Vec<NodeIndex> {
        inputs.to_vec()
    }

    fn stats_name(&self) -> Option<&'static str> {
        Some("BlockScaledQuantize")
    }
}

struct QuantizeModule {
    _module: Arc<CudaModule>,
    quantize: CudaFunction,
}

fn quantize_module(stream: &Arc<CudaStream>) -> anyhow::Result<&'static QuantizeModule> {
    static MODULES: std::sync::OnceLock<
        Mutex<std::collections::HashMap<usize, &'static QuantizeModule>>,
    > = std::sync::OnceLock::new();
    let mut modules = MODULES.get_or_init(Default::default).lock().unwrap();
    let key = stream.context().cu_ctx() as usize;
    if let Some(module) = modules.get(&key) {
        crate::artifact::observe_cached_module(stream.context(), contract::quantizer_source())?;
        return Ok(module);
    }
    let image =
        compile_module_image_for_current_device(stream.context(), contract::quantizer_source())?;
    let module = stream.context().load_module(image)?;
    let quantize = Box::leak(Box::new(QuantizeModule {
        quantize: module.load_function("block_scaled_quantize")?,
        _module: module,
    }));
    modules.insert(key, quantize);
    Ok(quantize)
}

#[cfg(test)]
#[path = "../../../tests/unit/host/deepgemm/shared/mod.rs"]
mod tests;
