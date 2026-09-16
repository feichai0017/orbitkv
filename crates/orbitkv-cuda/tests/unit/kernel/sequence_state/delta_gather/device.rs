use cudarc::driver::CudaContext;
use half::bf16;
use orbitkv_compiler::prelude::*;

use crate::{
    kernel::{
        KernelOp,
        sequence_state::{PackedDeltaScanPlan, PackedDeltaScanSpec, packed_delta_scan},
    },
    runtime::CudaRuntime,
    tests::utilities::{
        ForcedExtractionConfig, llir_kernel_names, try_extract_forced_op_llir_where,
    },
};

const EXTRACTION: ForcedExtractionConfig = ForcedExtractionConfig::new(0xD317_A319)
    .attempts_per_node(256)
    .node_seed_stride(256);

struct Case {
    lengths: Vec<usize>,
    slots: Vec<usize>,
    dynamic: bool,
    strided_indices: bool,
    strided_arena: bool,
    commit: bool,
}

struct Problem {
    graph: Graph,
    inputs: [GraphTensor; 6],
    indices: GraphTensor,
    indptr: GraphTensor,
    packed: GraphTensor,
    previous: GraphTensor,
    committed: Option<GraphTensor>,
    values: [Vec<f32>; 6],
    index_values: Vec<i32>,
    indptr_values: Vec<i32>,
    logical_arena: Vec<f32>,
    logical_indices: Vec<usize>,
    token_elements: usize,
    round_final_state: bool,
}

fn samples(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|index| {
            let value = ((index * 37 + seed * 19) % 127) as f32 - 63.0;
            bf16::from_f32(value / 64.0).to_f32()
        })
        .collect()
}

fn problem(spec: PackedDeltaScanSpec, case: &Case) -> Problem {
    assert_eq!(case.lengths.len(), case.slots.len());
    let requests = case.lengths.len();
    let tokens: usize = case.lengths.iter().sum();
    let state_elements = spec.value_heads * spec.key_width * spec.value_width;
    // Each slot owns several layers and padding after the selected layer.
    let layers_per_slot = 3;
    let selected_layer = 1;
    let slots = case.slots.iter().max().unwrap() + 2;
    let logical_arena = samples(slots * layers_per_slot * state_elements, 31)
        .into_iter()
        .map(|value| value + 1.0 / 8192.0)
        .collect::<Vec<_>>();
    let logical_indices = case
        .slots
        .iter()
        .flat_map(|slot| {
            let layer_start = (slot * layers_per_slot + selected_layer) * state_elements;
            (0..state_elements).map(move |offset| layer_start + offset)
        })
        .collect::<Vec<_>>();
    let index_values = logical_indices
        .iter()
        .flat_map(|&index| {
            let index = i32::try_from(index).unwrap();
            if case.strided_indices {
                vec![index, 0]
            } else {
                vec![index]
            }
        })
        .collect();
    let arena_values = logical_arena
        .iter()
        .flat_map(|&value| {
            if case.strided_arena {
                vec![value, 19.0]
            } else {
                vec![value]
            }
        })
        .collect();
    let mut graph = Graph::new();
    let (token_dim, request_dim): (Expression, Expression) = if case.dynamic {
        graph.set_dim_interval('s', 1, i64::try_from(tokens.max(1) + 1).unwrap());
        graph.set_dim_interval('b', 1, i64::try_from(requests + 1).unwrap());
        graph.set_dim('s', tokens);
        graph.set_dim('b', requests);
        ('s'.into(), 'b'.into())
    } else {
        (tokens.into(), requests.into())
    };
    let arena_pitch_value = if case.strided_arena { 2 } else { 1 };
    let arena_pitch: Expression = if case.dynamic {
        // This dimension belongs only to the source layout. The gathered scan
        // must collect and pass it even though its output geometry omits it.
        graph.set_dim('p', arena_pitch_value);
        graph.set_dim_interval('p', arena_pitch_value as i64, arena_pitch_value as i64 + 1);
        'p'.into()
    } else {
        arena_pitch_value.into()
    };
    let query = graph.named_tensor("query", (token_dim, spec.key_heads, spec.key_width));
    let key = graph.named_tensor("key", (token_dim, spec.key_heads, spec.key_width));
    let value = graph.named_tensor("value", (token_dim, spec.value_heads, spec.value_width));
    let decay = graph.named_tensor("decay", (token_dim, spec.value_heads));
    let beta = graph.named_tensor("beta", (token_dim, spec.value_heads));
    let arena = graph.named_tensor("arena", (logical_arena.len(), arena_pitch));
    arena.output();
    let arena_view = GraphTensor {
        shape: ShapeTracker::new_strided(logical_arena.len(), Expression::from('z') * arena_pitch),
        ..arena
    };
    let indices = graph
        .named_tensor(
            "indices",
            (
                request_dim,
                spec.value_heads,
                spec.key_width,
                spec.value_width,
                if case.strided_indices { 2 } else { 1 },
            ),
        )
        .as_dtype(DType::Int);
    let index_view = indices.slice((.., .., .., .., ..1)).squeeze(4);
    let previous = arena_view.gather(index_view).output();
    let indptr = graph
        .named_tensor("indptr", request_dim + 1)
        .as_dtype(DType::Int);
    let output = packed_delta_scan(
        PackedDeltaScanPlan {
            query,
            key,
            value,
            log_decay: decay,
            update_gate: beta,
            state: previous,
            query_indptr: indptr,
        },
        spec,
    );
    // Observe the complete custom operation, including its final-state suffix.
    let packed_id = graph.get_sources(output.values.id)[1];
    let token_elements = tokens * spec.value_heads * spec.value_width;
    let packed = GraphTensor::from_id(
        packed_id,
        ShapeTracker::new(
            token_dim * spec.value_heads * spec.value_width + request_dim * state_elements,
        ),
        &mut graph,
        DType::F32,
    )
    .output();
    let committed = case
        .commit
        .then(|| output.state.scatter(index_view, arena_view).output());
    let mut indptr_values = vec![0];
    for &length in &case.lengths {
        indptr_values.push(indptr_values.last().unwrap() + i32::try_from(length).unwrap());
    }
    Problem {
        graph,
        inputs: [query, key, value, decay, beta, arena],
        indices,
        indptr,
        packed,
        previous,
        committed,
        values: [
            samples(tokens * spec.key_heads * spec.key_width, 1),
            samples(tokens * spec.key_heads * spec.key_width, 7),
            samples(token_elements, 13),
            vec![-0.125; tokens * spec.value_heads],
            vec![0.625; tokens * spec.value_heads],
            arena_values,
        ],
        index_values,
        indptr_values,
        logical_arena,
        logical_indices,
        token_elements,
        round_final_state: spec.round_final_state_to_bf16,
    }
}

fn assert_bits(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{label} element {index}"
        );
    }
}

fn compare(mut problem: Problem) -> Vec<f32> {
    problem
        .graph
        .build_search_space::<CudaRuntime>(CompileOptions::default());
    let program = |kind: &str, name: &str, gathered: bool| {
        try_extract_forced_op_llir_where(&problem.graph, &[kind], EXTRACTION, |llir| {
            let names = llir_kernel_names(llir);
            names.contains(&name)
                && (gathered || !names.contains(&"PackedDeltaGather"))
                && !names.contains(&"ScatterNoCopy")
                && (problem.committed.is_none() || names.contains(&"Scatter"))
        })
        .unwrap_or_else(|error| panic!("cannot force {kind}: {error}"))
    };
    let baseline = program("KernelDeltaRegisters", "PackedDeltaRegisters", false);
    let candidate = program("KernelDeltaGather", "PackedDeltaGather", true);
    if problem.graph.dyn_map.contains_key(&Symbol::from('p')) {
        let kernel = candidate
            .node_weights()
            .filter_map(|op| op.to_dialect::<dyn KernelOp>())
            .find(|kernel| kernel.kernel_name() == "PackedDeltaGather")
            .unwrap();
        assert!(
            kernel.all_dyn_vars().contains(&Symbol::from('p')),
            "the source-only row pitch must be present in the launch ABI"
        );
    }
    let context = CudaContext::new(0).unwrap();
    let previous = problem
        .logical_indices
        .iter()
        .map(|&index| problem.logical_arena[index])
        .collect::<Vec<_>>();
    let mut outputs = Vec::new();
    for program in [&baseline, &candidate] {
        let mut runtime = CudaRuntime::initialize(context.new_stream().unwrap());
        for (&input, values) in problem.inputs.iter().zip(&problem.values) {
            runtime.set_data(input, values.clone());
        }
        runtime.set_data(problem.indices, problem.index_values.clone());
        runtime.set_data(problem.indptr, problem.indptr_values.clone());
        runtime.load_llir(program);
        runtime.execute(&problem.graph.dyn_map);
        let packed = runtime.get_f32(problem.packed);
        assert!(packed.iter().all(|value| value.is_finite()));
        assert_eq!(packed.len(), problem.token_elements + previous.len());
        let state_elements = previous.len() / (problem.indptr_values.len() - 1);
        for (request, range) in problem.indptr_values.windows(2).enumerate() {
            if range[0] == range[1] {
                let start = request * state_elements;
                let expected = previous[start..start + state_elements]
                    .iter()
                    .map(|&value| {
                        if problem.round_final_state {
                            bf16::from_f32(value).to_f32()
                        } else {
                            value
                        }
                    })
                    .collect::<Vec<_>>();
                if problem.round_final_state {
                    assert!(
                        expected
                            .iter()
                            .zip(&previous[start..start + state_elements])
                            .any(|(&rounded, &initial)| rounded.to_bits() != initial.to_bits()),
                        "empty-request fixture must exercise final-state rounding"
                    );
                }
                let output_start = problem.token_elements + start;
                assert_bits(
                    &packed[output_start..output_start + state_elements],
                    &expected,
                    "empty request carries its initial state",
                );
            }
        }
        assert_bits(
            &runtime.get_f32(problem.previous),
            &previous,
            "observed old state",
        );
        assert_bits(
            &runtime.get_f32(problem.inputs[5]),
            &problem.values[5],
            "immutable source arena",
        );
        if let Some(committed) = problem.committed {
            let mut expected = problem.logical_arena.clone();
            for (&index, &value) in problem
                .logical_indices
                .iter()
                .zip(&packed[problem.token_elements..])
            {
                expected[index] = value;
            }
            assert_bits(
                &runtime.get_f32(committed),
                &expected,
                "explicit scatter commit",
            );
        }
        outputs.push(packed);
    }
    assert_bits(
        &outputs[1],
        &outputs[0],
        "gather scan versus materialized scan",
    );
    outputs.pop().unwrap()
}

#[test]
#[ignore = "requires CUDA; validates gathered state, ragged batches and observed old versions"]
fn gathered_state_matches_materialized_scan_for_layouts_and_rounding() {
    for (key_width, value_width, dynamic, strided_indices, strided_arena, round_qk, round_state) in [
        (2, 3, false, false, false, false, false),
        (65, 35, true, true, false, true, false),
        (128, 8, false, false, true, false, true),
        (128, 8, true, true, true, true, true),
    ] {
        let spec = PackedDeltaScanSpec {
            key_heads: 1,
            value_heads: 2,
            key_width,
            value_width,
            normalization_epsilon: 1e-6,
            round_normalized_qk_to_bf16: round_qk,
            round_final_state_to_bf16: round_state,
        };
        compare(problem(
            spec,
            &Case {
                lengths: vec![2, 0, 3],
                slots: vec![2, 0, 2],
                dynamic,
                strided_indices,
                strided_arena,
                commit: false,
            },
        ));
    }
}

#[test]
#[ignore = "requires CUDA; validates explicit scatter remains separate from gathered-state reads"]
fn explicit_commit_preserves_observed_arena_and_previous_state() {
    compare(problem(
        PackedDeltaScanSpec {
            key_heads: 1,
            value_heads: 2,
            key_width: 65,
            value_width: 35,
            normalization_epsilon: 1e-6,
            round_normalized_qk_to_bf16: true,
            round_final_state_to_bf16: true,
        },
        &Case {
            lengths: vec![1, 0, 2],
            slots: vec![2, 0, 1],
            dynamic: true,
            strided_indices: true,
            strided_arena: true,
            commit: true,
        },
    ));
}

#[derive(serde::Deserialize)]
struct FrozenDeltaScan {
    key_width: usize,
    value_width: usize,
    indptr: Vec<i32>,
    query: Vec<u16>,
    key: Vec<u16>,
    value: Vec<u16>,
    log_decay_f32_bits: Vec<i32>,
    beta: Vec<u16>,
    initial_state: Vec<u16>,
    output: Vec<u16>,
    final_state: Vec<u16>,
}

#[test]
#[ignore = "requires CUDA; independent frozen Torch oracle for gathered nonzero recurrent state"]
fn gathered_state_matches_frozen_torch_output_and_cache_bits() {
    let fixture: FrozenDeltaScan = serde_json::from_str(include_str!(
        "../../../../fixtures/sequence_state/delta_scan.json"
    ))
    .unwrap();
    let from_bf16 = |values: &[u16]| {
        values
            .iter()
            .map(|&bits| bf16::from_bits(bits).to_f32())
            .collect::<Vec<_>>()
    };
    let mut problem = problem(
        PackedDeltaScanSpec {
            key_heads: 1,
            value_heads: 1,
            key_width: fixture.key_width,
            value_width: fixture.value_width,
            normalization_epsilon: 1e-6,
            round_normalized_qk_to_bf16: true,
            round_final_state_to_bf16: true,
        },
        &Case {
            lengths: fixture
                .indptr
                .windows(2)
                .map(|pair| usize::try_from(pair[1] - pair[0]).unwrap())
                .collect(),
            slots: vec![2],
            dynamic: true,
            strided_indices: true,
            strided_arena: true,
            commit: true,
        },
    );
    problem.values[0] = from_bf16(&fixture.query);
    problem.values[1] = from_bf16(&fixture.key);
    problem.values[2] = from_bf16(&fixture.value);
    problem.values[3] = fixture
        .log_decay_f32_bits
        .iter()
        .map(|&bits| f32::from_bits(bits as u32))
        .collect();
    problem.values[4] = from_bf16(&fixture.beta);
    assert_eq!(problem.logical_indices.len(), fixture.initial_state.len());
    for (&index, value) in problem
        .logical_indices
        .iter()
        .zip(from_bf16(&fixture.initial_state))
    {
        problem.logical_arena[index] = value;
        problem.values[5][index * 2] = value;
    }
    let actual = compare(problem);
    assert_eq!(
        actual[..fixture.output.len()]
            .iter()
            .map(|&value| bf16::from_f32(value).to_bits())
            .collect::<Vec<_>>(),
        fixture.output
    );
    assert_bits(
        &actual[fixture.output.len()..],
        &from_bf16(&fixture.final_state),
        "frozen final state",
    );
}
