use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule, CudaStream};
use orbitkv_compiler::{
    egglog_utils::{
        api::{Rule, SortDef, sort},
        base::OP_KIND,
    },
    op::EgglogOp,
    prelude::GraphTensor,
};

mod convolution;
mod delta_scan;
#[cfg(test)]
#[path = "../../tests/unit/kernel/sequence_state/mod.rs"]
mod tests;

pub use convolution::{
    PackedConvolutionOutput, PackedConvolutionPlan, PackedConvolutionSpec,
    packed_causal_convolution,
};
pub use delta_scan::{
    PackedDeltaScanOutput, PackedDeltaScanPlan, PackedDeltaScanSpec, packed_delta_scan,
};

const THREADS: usize = 256;

#[derive(Debug, Default)]
pub struct SequenceStateRules;

impl EgglogOp for SequenceStateRules {
    fn sort(&self) -> SortDef {
        sort(OP_KIND, "SequenceStateRules", &[])
    }

    fn egglog_declarations(&self) -> Vec<String> {
        vec![include_str!("sequence_state/declarations.egg").to_owned()]
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![Rule::raw(include_str!("sequence_state/state_commit.egg"))]
    }

    fn cleanup(&self) -> bool {
        false
    }
}

fn contiguous(input: GraphTensor) -> GraphTensor {
    if input.shape.is_contiguous() {
        input
    } else {
        input.gather(input.graph().iota('z', input.dims()))
    }
}

fn render_source<const N: usize>(template: &str, replacements: [(&str, String); N]) -> String {
    let mut source = template.to_owned();
    for (placeholder, value) in replacements {
        assert!(!value.contains('@'));
        assert!(
            source.contains(placeholder),
            "missing CUDA parameter {placeholder}"
        );
        source = source.replace(placeholder, &value);
    }
    assert!(!source.contains('@'), "unresolved CUDA parameter");
    source
}

fn compile_kernel(
    stream: &Arc<CudaStream>,
    cache: &mut orbitkv_compiler::prelude::FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    source: &str,
    name: &str,
) -> (Arc<CudaModule>, CudaFunction) {
    if let Some((module, function)) = cache.get(source) {
        return (Arc::clone(module), function.clone());
    }
    let image = crate::compile_module_image_for_current_device(stream.context(), source).unwrap();
    let module = stream.context().load_module(image).unwrap();
    let function = module.load_function(name).unwrap();
    cache.insert(source.to_owned(), (Arc::clone(&module), function.clone()));
    (module, function)
}
