use super::*;
use crate::kernel::fusion::{CudaUnaryElementwise, FusionEnd, FusionStart};
use orbitkv_compiler::hlir::Input;

#[test]
#[ignore = "requires CUDA to construct the graph owner; resource planning allocates no GPU buffers"]
fn flashinfer_recapture_budgets_both_plan_generations() {
    let context = cudarc::driver::CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let mut lengths = FxHashMap::default();
    let ops = (0..2)
        .map(|index| {
            let inputs = (0..7)
                .map(|input| NodeIndex::new(index * 8 + input))
                .collect_vec();
            for input in &inputs[1..3] {
                lengths.insert(*input, 64 * 16 * 2 * 64 * 2);
            }
            lengths.insert(inputs[4], 2 * std::mem::size_of::<i32>());
            lengths.insert(inputs[5], 2 * std::mem::size_of::<i32>());
            lengths.insert(inputs[6], std::mem::size_of::<i32>());
            CompiledFlashInferDecode::new(
                NodeIndex::new(index * 8 + 7),
                inputs,
                Arc::new(Box::new(FlashInferAttention::paged(
                    crate::host::flashinfer::FlashInferAlgorithm::CudaCoreDecode,
                    4,
                    2,
                    64,
                    16,
                    1.into(),
                    1.into(),
                    1.into(),
                    DType::Bf16,
                    0.0,
                    None,
                ))),
            )
        })
        .collect();
    let state = CudaGraphOpState::new(
        vec![],
        vec![],
        ops,
        vec![],
        vec![
            CompiledStep::FlashInferDecode(0),
            CompiledStep::FlashInferDecode(1),
        ],
    );
    let graph = CudaGraphOp::new(vec![], FxHashMap::default(), vec![], stream, None, state);
    let plan = graph
        .host_device_memory_plan(&lengths, &DynMap::default())
        .unwrap();
    // Each explicit-CSR plan owns 8 MiB metadata and a one-row BF16 output.
    let generation_bytes = 2 * (8 * 1024 * 1024 + 4 * 64 * 2);
    assert_eq!(plan.active_bucket_bytes, generation_bytes);
    assert_eq!(plan.transient_peak_bytes, generation_bytes);
    assert_eq!(plan.persistent_bytes, 0);
    assert_eq!(
        plan.shared_allocations,
        vec![crate::host::flashinfer::shared_device_memory_allocation()]
    );
}

fn test_kernel_op(op: impl KernelOp + 'static) -> LLIROp {
    LLIROp::new::<dyn KernelOp>(Box::new(op) as Box<dyn KernelOp>)
}

#[test]
fn prepared_fusion_sources_share_the_subgraph_dyn_dims_abi() {
    let mut llir = LLIRGraph::default();
    let input = llir.add_node(LLIROp::new::<Input>(Box::new(Input {
        node: 0,
        label: String::new(),
        dtype: DType::F32,
    })));
    let mut add_region = |llir: &mut LLIRGraph, producer, dim: char| {
        let shape = vec![Expression::from(dim)];
        let strides = vec![Expression::from('z')];
        let start = llir.add_node(test_kernel_op(FusionStart {
            shape: shape.clone(),
            strides: strides.clone(),
            dtype: DType::F32,
        }));
        let unary = llir.add_node(test_kernel_op(CudaUnaryElementwise {
            op: "Sin".to_string(),
            shape: shape.clone(),
            in_strides: strides.clone(),
            out_strides: strides.clone(),
            dtype: DType::F32,
        }));
        let end = llir.add_node(test_kernel_op(FusionEnd {
            shape,
            strides,
            dtype: DType::F32,
        }));
        llir.add_edge(producer, start, ());
        llir.add_edge(start, unary, ());
        llir.add_edge(unary, end, ());
        end
    };
    let b_end = add_region(&mut llir, input, 'b');
    let a_end = add_region(&mut llir, b_end, 'a');

    clear_global_dyn_dims();
    let prepared = prepare_kernel_to_host_plan(&llir);
    assert_eq!(prepared.subgraphs.len(), 1);
    assert_eq!(
        prepared.subgraphs[0].global_dyn_dims,
        vec![Symbol::from('a'), Symbol::from('b')]
    );
    assert!(
        prepared
            .fusion
            .region_kernel(b_end)
            .unwrap()
            .source
            .contains("dyn_dims[1]")
    );
    assert!(
        prepared
            .fusion
            .region_kernel(a_end)
            .unwrap()
            .source
            .contains("dyn_dims[0]")
    );
    assert!(prepared.materialized_kernel_nodes().contains(&a_end));
    assert!(prepared.materialized_kernel_nodes().contains(&b_end));
    for unit in prepared.fusion.compile_units() {
        if let CompileUnit::Region(region) = unit {
            assert!(
                region
                    .elementwise_topo
                    .iter()
                    .all(|node| !prepared.materialized_kernel_nodes().contains(node)),
                "register-resident fusion interiors must not receive device buffers"
            );
            assert!(
                region
                    .fs_nodes
                    .iter()
                    .all(|node| !prepared.materialized_kernel_nodes().contains(node)),
                "pure FusionStart aliases must not receive device buffers"
            );
        }
    }
    assert_eq!(get_global_dyn_dims(), None);
}

#[test]
fn dependency_steps_use_only_real_data_producers() {
    let a = NodeIndex::new(1);
    let b = NodeIndex::new(2);
    let external = NodeIndex::new(3);

    let mut producers = FxHashMap::default();
    producers.insert(a, 4);
    producers.insert(b, 9);

    let deps = dependency_steps_for_inputs(&producers, &[a, external, b, a], 12);
    assert_eq!(deps, vec![4, 9]);
}

#[test]
fn prepare_sharing_rejects_unordered_steps_and_distinct_keys() {
    // 0 feeds both siblings, but neither sibling reaches the other.
    let reachable = transitive_step_reachability(&[vec![1, 2], vec![], vec![]]);
    assert!(steps_are_dependency_ordered(&reachable, 0, 1));
    assert!(steps_are_dependency_ordered(&reachable, 0, 2));
    assert!(!steps_are_dependency_ordered(&reachable, 1, 2));
    assert!(prepare_cache_group_accepts(&7, &[0, 1], &7, 1, &reachable));
    assert!(!prepare_cache_group_accepts(&7, &[1], &7, 2, &reachable));
    assert!(!prepare_cache_group_accepts(&7, &[0], &8, 1, &reachable));
}

#[test]
fn arena_ordering_requires_real_dependency_order_after_every_use() {
    // Sibling steps 0 and 1 both feed step 2. Neither sibling may reuse
    // the other's storage, while both may reuse with a buffer produced at
    // step 2 after their final use.
    let reachability = transitive_step_reachability(&[vec![2], vec![2], vec![]]);
    let make_buffer = |node: usize, users: &[usize], producers: Vec<usize>| {
        let mut after_all_uses = FixedBitSet::with_capacity(reachability.len());
        after_all_uses.insert_range(..);
        for &user in users {
            after_all_uses.intersect_with(&reachability[user]);
        }
        ArenaBufferOrder {
            node: NodeIndex::new(node),
            first: *users.first().unwrap(),
            last: *users.last().unwrap(),
            after_all_uses,
            producers,
        }
    };
    let buffers = vec![
        make_buffer(0, &[0], vec![0]),
        make_buffer(1, &[1], vec![1]),
        make_buffer(2, &[2], vec![2]),
    ];
    let ordering = CudaGraphArenaOrdering {
        buffers,
        node_to_buffer: vec![0, 1, 2],
        span: 3,
    };

    assert!(!ordering.precedes(NodeIndex::new(0), NodeIndex::new(1)));
    assert!(!ordering.precedes(NodeIndex::new(1), NodeIndex::new(0)));
    assert!(ordering.precedes(NodeIndex::new(0), NodeIndex::new(2)));
    assert!(ordering.precedes(NodeIndex::new(1), NodeIndex::new(2)));
}

#[test]
fn arena_ordering_keeps_a_buffer_live_through_its_consumer_step() {
    let reachability = transitive_step_reachability(&[vec![1], vec![]]);
    let mut after_all_uses = reachability[0].clone();
    after_all_uses.intersect_with(&reachability[1]);
    let buffers = vec![
        ArenaBufferOrder {
            node: NodeIndex::new(0),
            first: 0,
            last: 1,
            after_all_uses,
            producers: vec![0],
        },
        ArenaBufferOrder {
            node: NodeIndex::new(1),
            first: 1,
            last: 1,
            after_all_uses: reachability[1].clone(),
            producers: vec![1],
        },
    ];
    let ordering = CudaGraphArenaOrdering {
        buffers,
        node_to_buffer: vec![0, 1],
        span: 2,
    };

    assert!(
        !ordering.precedes(NodeIndex::new(0), NodeIndex::new(1)),
        "an input and output used by the same step must not share storage"
    );
}

#[test]
fn flashinfer_global_workspaces_serialize_every_island() {
    let steps = vec![
        CompiledStep::FlashInferDecode(0),
        CompiledStep::Kernel(0),
        CompiledStep::FlashInferDecode(1),
        CompiledStep::CuBlasLt(0),
        CompiledStep::FlashInferDecode(2),
    ];
    let mut dependencies = vec![Vec::new(); steps.len()];
    add_flashinfer_workspace_serial_dependencies(&steps, &mut dependencies);

    assert_eq!(dependencies[2], vec![0]);
    assert_eq!(dependencies[4], vec![2]);
    assert!(dependencies[0].is_empty());
    assert!(dependencies[1].is_empty());
    assert!(dependencies[3].is_empty());

    let mut successors = vec![Vec::new(); steps.len()];
    for (step, deps) in dependencies.iter().enumerate() {
        for &dependency in deps {
            successors[dependency].push(step);
        }
    }
    let reachable = transitive_step_reachability(&successors);
    assert!(steps_are_dependency_ordered(&reachable, 0, 4));
}
