//! Explicit portable oracle; never registered in production search.
use super::*;

// Eight warps reduce each output element in the portable semantic reference.
const REFERENCE_THREADS: usize = 256;
const REFERENCE_SOURCE: &str = include_str!("reference.cu");
#[derive(Debug)]
struct ReferenceModule {
    _module: Arc<CudaModule>,
    quantize: CudaFunction,
    matmul: CudaFunction,
}

pub(super) struct BlockScaledLinearReference {
    spec: BlockScaledLinearSpec,
    scratch: Mutex<Option<Arc<Scratch>>>,
}

impl std::fmt::Debug for BlockScaledLinearReference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockScaledLinearReference")
            .field("spec", &self.spec)
            .finish()
    }
}

impl BlockScaledLinearReference {
    pub(super) fn new(spec: BlockScaledLinearSpec) -> Self {
        Self {
            spec,
            scratch: Mutex::new(None),
        }
    }

    fn dimensions(
        &self,
        dyn_map: &orbitkv_compiler::prelude::DynMap,
    ) -> anyhow::Result<(usize, usize, usize)> {
        let rows = self
            .spec
            .rows
            .exec(dyn_map)
            .ok_or_else(|| anyhow::anyhow!("unresolved block-scaled linear row dimension"))?;
        Ok((rows, self.spec.output_features, self.spec.input_features))
    }
}

impl Clone for BlockScaledLinearReference {
    fn clone(&self) -> Self {
        Self::new(self.spec)
    }
}

impl EgglogOp for BlockScaledLinearReference {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "BlockScaledLinearReference",
            &[
                ("rows", EXPRESSION),
                ("output_features", EXPRESSION),
                ("input_features", EXPRESSION),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        3
    }

    fn cleanup(&self) -> bool {
        false
    }
}

impl HostOp for BlockScaledLinearReference {
    fn deployment_eligible(&self) -> bool {
        false
    }

    fn prepare_compilation(
        &self,
        stream: &Arc<CudaStream>,
        _dyn_map: &orbitkv_compiler::prelude::DynMap,
    ) -> anyhow::Result<()> {
        let _ = reference_module(stream)?;
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
        anyhow::ensure!(inputs.len() == 3, "block-scaled reference expects 3 inputs");
        let (m, n, k) = self.dimensions(dyn_map)?;
        if m == 0 {
            return Ok(());
        }
        validate_buffers::<false>(self_node, inputs, buffers, m, n, k)?;
        let mut scratch = self.scratch.lock().unwrap();
        let scratch = Scratch::prepare(&mut scratch, stream, m, k)?;
        let module = reference_module(stream)?;
        let output = buffers[&self_node].ptr();
        let input = buffers[&inputs[0]].ptr();
        let weight = buffers[&inputs[1]].ptr();
        let weight_scale = buffers[&inputs[2]].ptr();
        let quantized = scratch.quantized_ptr;
        let scales = scratch.scales_ptr;
        let (m, n, k) = (m as i32, n as i32, k as i32);
        unsafe {
            stream
                .launch_builder(&module.quantize)
                .arg(&quantized)
                .arg(&scales)
                .arg(&input)
                .arg(&m)
                .arg(&k)
                .launch(LaunchConfig {
                    grid_dim: (((k + BLOCK as i32 - 1) / BLOCK as i32) as u32, m as u32, 1),
                    block_dim: (BLOCK as u32, 1, 1),
                    shared_mem_bytes: 0,
                })?;
            stream
                .launch_builder(&module.matmul)
                .arg(&output)
                .arg(&quantized)
                .arg(&scales)
                .arg(&weight)
                .arg(&weight_scale)
                .arg(&m)
                .arg(&n)
                .arg(&k)
                .launch(LaunchConfig {
                    grid_dim: (n as u32, m as u32, 1),
                    block_dim: (REFERENCE_THREADS as u32, 1, 1),
                    shared_mem_bytes: (REFERENCE_THREADS / 32 * SCALE_BYTES) as u32,
                })?;
        }
        Ok(())
    }

    fn output_size(&self) -> Expression {
        self.spec.rows * self.spec.output_features
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
        self.spec.rows.dyn_vars()
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
        Scratch::prepare(&mut self.scratch.lock().unwrap(), stream, m, k)?;
        Ok(())
    }

    fn cuda_graph_capture_resources(&self) -> Vec<crate::providers::CudaGraphCaptureResource> {
        self.scratch
            .lock()
            .unwrap()
            .iter()
            .cloned()
            .map(|scratch| scratch as crate::providers::CudaGraphCaptureResource)
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
                    name: "BlockScaledLinear dimensions",
                })?;
        validate_buffer_lengths::<false>(self_node, inputs, buffer_lengths, m, n, k)?;
        let (quantized_bytes, scale_bytes) = scratch_bytes(m, k)?;
        Ok(HostDeviceMemoryPlan {
            persistent_bytes: quantized_bytes
                .checked_add(scale_bytes)
                .ok_or(ResourceViolation::ArithmeticOverflow {
                    resource: "BlockScaledLinear reference scratch",
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
        Some("BlockScaledLinearReference")
    }
}

fn reference_module(stream: &Arc<CudaStream>) -> anyhow::Result<&'static ReferenceModule> {
    static MODULES: std::sync::OnceLock<
        Mutex<std::collections::HashMap<usize, &'static ReferenceModule>>,
    > = std::sync::OnceLock::new();
    let modules = MODULES.get_or_init(Default::default);
    let key = stream.context().cu_ctx() as usize;
    let mut modules = modules.lock().unwrap();
    let source = REFERENCE_SOURCE
        .replace("@QUANTIZER@", contract::quantizer_source())
        .replace("@REFERENCE_THREADS@", &REFERENCE_THREADS.to_string());
    if let Some(module) = modules.get(&key) {
        crate::artifact::observe_cached_module(stream.context(), &source)?;
        return Ok(module);
    }
    let image = compile_module_image_for_current_device(stream.context(), &source)?;
    let module = stream.context().load_module(image)?;
    let reference = Box::leak(Box::new(ReferenceModule {
        quantize: module.load_function("block_scaled_quantize")?,
        matmul: module.load_function("block_scaled_reference")?,
        _module: module,
    }));
    modules.insert(key, reference);
    Ok(reference)
}
