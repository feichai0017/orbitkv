use super::*;
use cudarc::driver::{CudaFunction, CudaModule, CudaSlice};
use orbitkv_compiler::{op::LLIROp, prelude::Symbol};
use std::sync::Arc;

// Logical IDs deliberately differ from the extracted graph's node indices.
const STATE_INPUT: usize = 103;
const STATE_OUTPUT: usize = 211;
const OTHER_INPUT: usize = 307;

fn required(pairs: &[(usize, usize)]) -> Vec<(NodeIndex, NodeIndex)> {
    pairs
        .iter()
        .map(|&(output, input)| (NodeIndex::new(output), NodeIndex::new(input)))
        .collect()
}

#[derive(Debug)]
struct PreparationProbe {
    alias: Option<usize>,
    mutates: bool,
}

impl KernelOp for PreparationProbe {
    fn compile(
        &self,
        _: &Arc<CudaStream>,
        _: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
        String,
        (Expression, Expression, Expression),
        (Expression, Expression, Expression),
        Expression,
        FxHashMap<Symbol, CudaSlice<u8>>,
    ) {
        panic!("invalid state must be rejected before CUDA compilation")
    }

    fn output_size(&self) -> Expression {
        panic!("invalid state must be rejected before kernel preparation")
    }

    fn output_bytes(&self) -> Expression {
        panic!("invalid state must be rejected before resource planning")
    }

    fn output_aliases_input(&self) -> Option<usize> {
        self.alias
    }

    fn output_data_input(&self) -> Option<usize> {
        // Copies have data ancestry but do not preserve storage identity.
        Some(self.alias.unwrap_or(0))
    }

    fn mutates_aliased_input(&self) -> bool {
        self.mutates
    }
}

fn input(llir: &mut LLIRGraph, logical: usize) -> NodeIndex {
    llir.add_node(LLIROp::new::<Input>(Box::new(Input {
        node: logical,
        ..Default::default()
    })))
}

fn output(llir: &mut LLIRGraph, producer: NodeIndex, logical: usize) -> NodeIndex {
    let output = llir.add_node(LLIROp::new::<Output>(Box::new(Output {
        node: logical,
        persist_only: false,
    })));
    llir.add_edge(producer, output, ());
    output
}

fn kernel(
    llir: &mut LLIRGraph,
    inputs: &[NodeIndex],
    alias: Option<usize>,
    mutates: bool,
) -> NodeIndex {
    let node = llir.add_node(LLIROp::new::<dyn KernelOp>(Box::new(PreparationProbe {
        alias,
        mutates,
    })));
    for &input in inputs {
        llir.add_edge(input, node, ());
    }
    node
}

fn assert_required_alias_rejected(llir: &LLIRGraph) {
    let expected = ResourceViolation::RequiredStateAlias {
        output: NodeIndex::new(STATE_OUTPUT),
        input: NodeIndex::new(STATE_INPUT),
    };
    // Direct/stored loading and search/finalist preparation enforce the same
    // contract. Neither path needs a CUDA context or may inspect kernel sizes.
    assert_eq!(
        validate_static_llir_semantics(llir, &required(&[(STATE_OUTPUT, STATE_INPUT)])),
        Err(expected.clone())
    );
    let result = prepare_static_llir_resources(
        llir,
        &required(&[(STATE_OUTPUT, STATE_INPUT)]),
        &DynMap::default(),
        &mut RegionSourceCache::default(),
    );
    assert_eq!(result.err(), Some(expected));
}

#[test]
fn materialized_state_is_rejected_before_preparation() {
    let mut llir = LLIRGraph::default();
    let state = input(&mut llir, STATE_INPUT);
    let copy = kernel(&mut llir, &[state], None, false);
    output(&mut llir, copy, STATE_OUTPUT);
    assert_required_alias_rejected(&llir);
    // Optional state aliases continue to admit copying implementations.
    assert!(validate_static_llir_semantics(&llir, &[]).is_ok());
}

#[test]
fn alias_chains_preserve_logical_input_identity() {
    let mut llir = LLIRGraph::default();
    let state = input(&mut llir, STATE_INPUT);
    let view = kernel(&mut llir, &[state], Some(0), false);
    let update = kernel(&mut llir, &[view], Some(0), true);
    let view = kernel(&mut llir, &[update], Some(0), false);
    output(&mut llir, view, STATE_OUTPUT);
    assert!(
        validate_static_llir_semantics(&llir, &required(&[(STATE_OUTPUT, STATE_INPUT)])).is_ok()
    );
}

#[test]
fn alias_input_uses_argument_order() {
    let mut llir = LLIRGraph::default();
    let state = input(&mut llir, STATE_INPUT);
    let other = input(&mut llir, OTHER_INPUT);
    let update = kernel(&mut llir, &[other, state], Some(1), true);
    output(&mut llir, update, STATE_OUTPUT);
    assert!(
        validate_static_llir_semantics(&llir, &required(&[(STATE_OUTPUT, STATE_INPUT)])).is_ok()
    );
    assert!(matches!(
        validate_static_llir_semantics(&llir, &required(&[(STATE_OUTPUT, OTHER_INPUT)])),
        Err(ResourceViolation::RequiredStateAlias { .. })
    ));
}

#[test]
fn wrong_or_missing_state_endpoints_fail_closed() {
    assert_required_alias_rejected(&LLIRGraph::default());
    let mut llir = LLIRGraph::default();
    let other = input(&mut llir, OTHER_INPUT);
    output(&mut llir, other, STATE_OUTPUT);
    assert_required_alias_rejected(&llir);
    let state = input(&mut llir, STATE_INPUT);
    output(&mut llir, state, OTHER_INPUT);
    assert_required_alias_rejected(&llir);
}

#[test]
fn conflicting_output_bindings_fail_closed() {
    let mut llir = LLIRGraph::default();
    let state = input(&mut llir, STATE_INPUT);
    let other = input(&mut llir, OTHER_INPUT);
    let first = output(&mut llir, state, STATE_OUTPUT);
    let second = output(&mut llir, other, STATE_OUTPUT);
    assert_required_alias_rejected(&llir);
    llir.remove_node(second);
    llir.add_edge(other, first, ());
    assert_required_alias_rejected(&llir);
}

#[test]
fn required_storage_does_not_relax_mutation_ordering() {
    let mut llir = LLIRGraph::default();
    let state = input(&mut llir, STATE_INPUT);
    let update = kernel(&mut llir, &[state], Some(0), true);
    output(&mut llir, update, STATE_OUTPUT);
    output(&mut llir, state, OTHER_INPUT);
    assert!(matches!(
        validate_static_llir_semantics(&llir, &required(&[(STATE_OUTPUT, STATE_INPUT)])),
        Err(ResourceViolation::AliasingHazard { .. })
    ));
}

#[test]
fn independent_required_arenas_are_all_validated() {
    let mut llir = LLIRGraph::default();
    let state = input(&mut llir, STATE_INPUT);
    let other = input(&mut llir, OTHER_INPUT);
    output(&mut llir, state, STATE_OUTPUT);
    output(&mut llir, other, OTHER_INPUT);
    let aliases = required(&[(STATE_OUTPUT, STATE_INPUT), (OTHER_INPUT, OTHER_INPUT)]);
    assert!(
        prepare_static_llir_resources(
            &llir,
            &aliases,
            &DynMap::default(),
            &mut RegionSourceCache::default(),
        )
        .is_ok()
    );
    let aliases = required(&[(STATE_OUTPUT, STATE_INPUT), (OTHER_INPUT, STATE_INPUT)]);
    assert!(matches!(
        validate_static_llir_semantics(&llir, &aliases),
        Err(ResourceViolation::RequiredStateAlias {
            output, ..
        }) if output.index() == OTHER_INPUT
    ));
}
