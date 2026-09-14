use super::*;
use crate::kernel::hlir::KernelScatter;
use orbitkv_compiler::op::LLIROp;

#[test]
#[ignore = "requires CUDA; verifies rejection precedes compilation and runtime replacement"]
fn invalid_state_alias_preserves_the_loaded_bucket() {
    let mut runtime = CudaRuntime::new().unwrap();
    let input = NodeIndex::new(11);
    let output = NodeIndex::new(23);
    runtime.required_state_aliases.push((output, input));

    let mut llir = LLIRGraph::default();
    let source = llir.add_node(LLIROp::new::<Input>(Box::new(Input {
        node: input.index(),
        ..Default::default()
    })));
    let target = llir.add_node(LLIROp::new::<Output>(Box::new(Output {
        node: output.index(),
        persist_only: false,
    })));
    let edge = llir.add_edge(source, target, ());
    runtime.try_load_llir(&llir).unwrap();
    assert!(runtime.output_aliases_input_in_all_buckets(output, input));

    llir.remove_edge(edge);
    let copy = llir.add_node(LLIROp::new::<dyn KernelOp>(Box::new(
        KernelScatter::default(),
    )));
    llir.add_edge(source, copy, ());
    llir.add_edge(copy, target, ());
    let expected = format!(
        "required state alias output {} does not resolve to input {}",
        output.index(),
        input.index()
    );
    let error = runtime.try_load_llir(&llir).err().unwrap();
    assert!(error.to_string().contains(&expected));
    assert!(runtime.kernel_cache.is_empty());
    assert!(runtime.output_aliases_input_in_all_buckets(output, input));

    let dimensions = DynMap::default();
    let candidate = BucketLLIRRef {
        bucket_indices: &dimensions,
        representative_dyn_map: &dimensions,
        llir: &llir,
    };
    let error = runtime
        .compile_and_validate_bucket_set_with_allocation_maps(
            &[candidate],
            std::slice::from_ref(&dimensions),
        )
        .err()
        .unwrap();
    assert!(error.to_string().contains(&expected));
    assert!(runtime.kernel_cache.is_empty());
    assert!(runtime.output_aliases_input_in_all_buckets(output, input));
}
