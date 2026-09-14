use super::*;

fn reference_partition(
    graph: &StableGraph<(), (), Directed>,
    marked: &FxHashSet<NodeIndex>,
    topo: &[NodeIndex],
) -> Vec<FxHashSet<NodeIndex>> {
    let bound = graph.node_bound();
    let mut reachable = vec![vec![false; bound]; bound];
    for &node in topo.iter().rev() {
        for successor in graph.neighbors_directed(node, Direction::Outgoing) {
            reachable[node.index()][successor.index()] = true;
            let successor_reachability = reachable[successor.index()].clone();
            for (target, successor_reachable) in reachable[node.index()]
                .iter_mut()
                .zip(successor_reachability)
            {
                *target |= successor_reachable;
            }
        }
    }
    let mut position = vec![usize::MAX; bound];
    for (index, &node) in topo.iter().enumerate() {
        position[node.index()] = index;
    }

    let mut result = Vec::new();
    for mut component in marked_weak_components(graph, marked) {
        component.sort_by_key(|node| position[node.index()]);
        let mut current = FxHashSet::default();
        for node in component {
            let violates = current.iter().copied().any(|prior: NodeIndex| {
                graph.node_indices().any(|witness| {
                    !marked.contains(&witness)
                        && ((reachable[prior.index()][witness.index()]
                            && reachable[witness.index()][node.index()])
                            || (reachable[node.index()][witness.index()]
                                && reachable[witness.index()][prior.index()]))
                })
            });
            if violates && !current.is_empty() {
                result.push(std::mem::take(&mut current));
            }
            current.insert(node);
        }
        if !current.is_empty() {
            result.push(current);
        }
    }
    result
}

#[test]
fn sparse_convex_partition_matches_exhaustive_reference() {
    const NODES: usize = 5;
    let possible_edges = (0..NODES)
        .flat_map(|source| ((source + 1)..NODES).map(move |target| (source, target)))
        .collect::<Vec<_>>();

    for edge_mask in 0..(1usize << possible_edges.len()) {
        let mut graph = StableGraph::<(), (), Directed>::default();
        let nodes = (0..NODES).map(|_| graph.add_node(())).collect::<Vec<_>>();
        for (bit, &(source, target)) in possible_edges.iter().enumerate() {
            if edge_mask & (1 << bit) != 0 {
                graph.add_edge(nodes[source], nodes[target], ());
            }
        }
        let topo = toposort(&graph, None).unwrap();
        for marked_mask in 1..(1usize << NODES) {
            let marked = nodes
                .iter()
                .enumerate()
                .filter_map(|(bit, &node)| (marked_mask & (1 << bit) != 0).then_some(node))
                .collect::<FxHashSet<_>>();
            assert_eq!(
                partition_marked_convex(&graph, &marked, &topo),
                reference_partition(&graph, &marked, &topo),
                "edge mask {edge_mask:#x}, marked mask {marked_mask:#x}"
            );
        }
    }
}

#[test]
fn arena_release_plan_retains_and_deduplicates_allocation_pools() {
    let pool = std::ptr::NonNull::<sys::CUmemPoolHandle_st>::dangling().as_ptr();
    let mut releases = ArenaReleasePlan::default();

    releases.record_arena(Some(pool));
    releases.record_arena(Some(pool));
    releases.record_arena(None);

    assert_eq!(releases.arenas_released, 3);
    assert_eq!(releases.pools_to_trim, vec![pool]);
}

#[test]
fn search_intermediate_cap_reserves_unplanned_device_headroom() {
    let gib = 1024 * 1024 * 1024;
    assert_eq!(
        bounded_search_intermediate_bytes(Some(12 * gib), 10 * gib, 100 * gib),
        10 * gib - 512 * 1024 * 1024
    );
    assert_eq!(
        bounded_search_intermediate_bytes(Some(6 * gib), 10 * gib, 100 * gib),
        6 * gib
    );
    assert_eq!(
        bounded_search_intermediate_bytes(None, 700 * 1024 * 1024, 24 * gib),
        188 * 1024 * 1024
    );
}

#[test]
fn search_cache_is_evicted_only_inside_device_pressure_margin() {
    let gib = 1024 * 1024 * 1024;
    assert!(!search_cache_under_pressure(3 * gib, 100 * gib));
    assert!(search_cache_under_pressure(gib, 100 * gib));
    assert!(!search_cache_under_pressure(2 * gib, 24 * gib));
    assert!(search_cache_under_pressure(900 * 1024 * 1024, 24 * gib));
}

#[test]
fn materialized_bucket_capacity_evicts_lru_before_new_bucket() {
    let materialized = [true, true, false];
    let lru = VecDeque::from([1, 0, 2]);

    assert_eq!(
        materialized_bucket_evictions(&materialized, &lru, 2, 1),
        vec![1, 0]
    );
    assert_eq!(
        materialized_bucket_evictions(&materialized, &lru, 2, 2),
        vec![1]
    );
    assert_eq!(
        materialized_bucket_evictions(&materialized, &lru, 1, 1),
        vec![0]
    );
}

#[test]
fn search_planning_node_limit_allows_growth_but_rejects_graph_explosion() {
    assert_eq!(search_candidate_node_limit(3_500), 4_524);
    assert!(10_311 > search_candidate_node_limit(3_500));
    assert_eq!(search_candidate_node_limit(10), 1_034);
}

#[test]
fn compiled_buckets_bind_different_layouts_to_one_shared_arena() {
    let Ok(mut rt) = CudaRuntime::new() else {
        return;
    };
    let first_node = NodeIndex::new(1);
    let second_node = NodeIndex::new(2);
    rt.compiled_buckets = vec![CompiledBucket::new(), CompiledBucket::new()];
    rt.compiled_buckets[0].arena_bytes = 1024;
    rt.compiled_buckets[0]
        .logical_buffer_offsets
        .insert(first_node, 0);
    rt.compiled_buckets[0]
        .logical_buffer_bytes
        .insert(first_node, 256);
    rt.compiled_buckets[1].arena_bytes = 2048;
    rt.compiled_buckets[1]
        .logical_buffer_offsets
        .insert(second_node, 512);
    rt.compiled_buckets[1]
        .logical_buffer_bytes
        .insert(second_node, 512);

    assert!(rt.ensure_shared_arena_capacity(4096));
    let (ptr, len) = rt.shared_arena_ptr_and_len().unwrap();
    CudaRuntime::bind_intermediate_buffers(
        &mut rt.compiled_buckets[0],
        Some(ptr),
        len,
        &FxHashSet::default(),
    );
    CudaRuntime::bind_intermediate_buffers(
        &mut rt.compiled_buckets[1],
        Some(ptr),
        len,
        &FxHashSet::default(),
    );

    assert_eq!(rt.intermediate_buffer_bytes(), 4096);
    assert_eq!(
        rt.compiled_buckets[0]
            .cached_device_buffers
            .get(&first_node)
            .copied()
            .map(DeviceBuffer::ptr),
        Some(ptr)
    );
    assert_eq!(
        rt.compiled_buckets[1]
            .cached_device_buffers
            .get(&second_node)
            .copied()
            .map(DeviceBuffer::ptr),
        Some(ptr + 512)
    );

    rt.clear_intermediate_buffers();
    assert!(rt.shared_arena.is_none());
    assert!(
        rt.compiled_buckets
            .iter()
            .all(|bucket| bucket.bound_arena_ptr.is_none())
    );
    assert_eq!(rt.intermediate_buffer_bytes(), 0);
}

#[test]
fn search_cleanup_releases_candidate_arenas_and_bucket_state() {
    let Ok(mut rt) = CudaRuntime::new() else {
        return;
    };
    rt.shared_arena = Some(SharedArena {
        allocation: unsafe { rt.cuda_stream.alloc::<u8>(4096).unwrap() },
        pool: None,
    });
    rt.compiled_buckets[0].arena_bytes = 4096;

    rt.release_search_candidate_allocations();

    assert_eq!(rt.compiled_buckets.len(), 1);
    assert!(rt.shared_arena.is_none());
    assert_eq!(rt.intermediate_buffer_bytes(), 0);

    rt.discard_search_bucket_compilation_state();

    assert!(rt.compiled_buckets.is_empty());
    assert!(rt.validated_resource_signatures.is_empty());
    assert!(rt.resource_length_sensitive_hlir.is_empty());
}

#[test]
fn resource_input_footprint_tracks_lengths_and_owned_capacity_only() {
    let owned = ResourceInputFootprint::owned(64, Some(8));
    assert_eq!(owned, ResourceInputFootprint::owned(64, Some(8)));
    assert_ne!(owned, ResourceInputFootprint::owned(64, Some(16)));
    assert_ne!(owned, ResourceInputFootprint::owned(128, Some(8)));
    assert_ne!(owned, ResourceInputFootprint::owned(64, None));
    assert_ne!(owned, ResourceInputFootprint::external(8));

    // External pointer identity and payload contents are intentionally not
    // represented. An equally sized replacement cannot change a HostOp
    // resource plan or the runtime-owned allocation total.
    assert_eq!(
        ResourceInputFootprint::external(32),
        ResourceInputFootprint::external(32)
    );
    assert_ne!(
        ResourceInputFootprint::external(32),
        ResourceInputFootprint::external(64)
    );
}

#[test]
fn identical_external_pointer_binding_is_a_noop() {
    assert!(device_pointer_binding_matches(
        Some(0x1000),
        Some(64),
        0x1000,
        64
    ));
    assert!(!device_pointer_binding_matches(
        Some(0x2000),
        Some(64),
        0x1000,
        64
    ));
    assert!(!device_pointer_binding_matches(
        Some(0x1000),
        Some(32),
        0x1000,
        64
    ));
    assert!(!device_pointer_binding_matches(None, None, 0x1000, 64));
}

#[test]
fn set_device_ptr_dirties_only_changed_external_bindings() {
    let mut rt = CudaRuntime::new().unwrap();
    let input = NodeIndex::new(125);
    let allocation = rt.cuda_stream.alloc_zeros::<u8>(64).unwrap();
    let ptr = allocation.device_ptr(&rt.cuda_stream).0;

    unsafe { rt.set_device_ptr(input, ptr, 64) };
    assert_eq!(rt.changed_hlir, FxHashSet::from_iter([input]));

    rt.changed_hlir.clear();
    unsafe { rt.set_device_ptr(input, ptr, 64) };
    assert!(rt.changed_hlir.is_empty());

    unsafe { rt.set_device_ptr(input, ptr, 32) };
    assert_eq!(rt.changed_hlir, FxHashSet::from_iter([input]));
}

#[test]
fn set_output_device_ptr_dirties_only_changed_registrations() {
    let mut rt = CudaRuntime::new().unwrap();
    let output = NodeIndex::new(126);
    let allocation = rt.cuda_stream.alloc_zeros::<u8>(64).unwrap();
    let ptr = allocation.device_ptr(&rt.cuda_stream).0;

    unsafe { rt.set_output_device_ptr(output, ptr, 64) };
    assert_eq!(
        rt.dirty_output_ptr_registrations,
        FxHashSet::from_iter([output])
    );

    rt.dirty_output_ptr_registrations.clear();
    unsafe { rt.set_output_device_ptr(output, ptr, 64) };
    assert!(rt.dirty_output_ptr_registrations.is_empty());

    unsafe { rt.set_output_device_ptr(output, ptr, 32) };
    assert_eq!(
        rt.dirty_output_ptr_registrations,
        FxHashSet::from_iter([output])
    );

    rt.dirty_output_ptr_registrations.clear();
    rt.clear_output_device_ptr(output);
    assert!(!rt.output_ptr_registrations.contains_key(&output));
    assert_eq!(
        rt.dirty_output_ptr_registrations,
        FxHashSet::from_iter([output])
    );
}

#[test]
#[ignore = "requires a CUDA device"]
fn allocated_required_state_preserves_alias_contract_after_rebinding() {
    let mut runtime = CudaRuntime::new().expect("CUDA device");
    let input = NodeIndex::new(125);
    let output = NodeIndex::new(126);
    let first = runtime.alias_state_required(input, output, 64);
    assert_eq!(runtime.required_state_aliases, [(output, input)]);
    let second = runtime.alias_state_required(input, output, 64);
    assert_eq!(runtime.required_state_aliases, [(output, input)]);
    assert_eq!(runtime.input_allocation(input).unwrap().1, 64);
    drop((first, second));
}

#[test]
fn shared_required_state_keeps_ownership_and_clears_both_bindings() {
    let Ok(mut rt) = CudaRuntime::new() else {
        return;
    };
    let input = NodeIndex::new(127);
    let output = NodeIndex::new(128);
    let allocation = Arc::new(rt.cuda_stream.alloc_zeros::<u8>(64).unwrap());
    let pointer = allocation.device_ptr(&rt.cuda_stream).0;

    let binding = rt
        .alias_shared_state_required(input, output, Arc::clone(&allocation), 64)
        .unwrap();
    assert_eq!(Arc::strong_count(&allocation), 3);
    assert_eq!(rt.input_allocation(input), Some((pointer, 64)));
    assert_eq!(rt.output_ptr_registrations[&output], (pointer, 64));
    assert!(rt.required_state_aliases.contains(&(output, input)));

    rt.set_data(input, vec![1_u8; 16]);
    assert_eq!(Arc::strong_count(&allocation), 2);
    assert!(!rt.shared_external_buffers.contains_key(&input));
    assert!(!rt.shared_external_outputs.contains_key(&input));
    assert!(!rt.required_state_aliases.contains(&(output, input)));
    assert!(!rt.output_ptr_registrations.contains_key(&output));
    assert!(rt.dirty_output_ptr_registrations.contains(&output));

    let receipt =
        CudaExecutionReceipt::record(&rt.cuda_stream, vec![binding.clone()].into_boxed_slice())
            .unwrap();
    let replacement = rt
        .alias_shared_state_required(input, output, Arc::clone(&allocation), 64)
        .unwrap();
    assert!(
        receipt
            .synchronize_required_states(&rt.cuda_stream, [&replacement])
            .is_err()
    );
}

#[test]
fn shared_copy_back_state_keeps_identity_without_requiring_alias() {
    let Ok(mut rt) = CudaRuntime::new() else {
        return;
    };
    let input = NodeIndex::new(129);
    let output = NodeIndex::new(130);
    let allocation = Arc::new(rt.cuda_stream.alloc_zeros::<u8>(64).unwrap());
    let pointer = allocation.device_ptr(&rt.cuda_stream).0;

    let binding = rt
        .bind_shared_state(
            input,
            output,
            Arc::clone(&allocation),
            64,
            CudaSharedStatePolicy::CopyBackAllowed,
        )
        .unwrap();
    assert_eq!(rt.input_allocation(input), Some((pointer, 64)));
    assert_eq!(rt.output_ptr_registrations[&output], (pointer, 64));
    assert!(!rt.required_state_aliases.contains(&(output, input)));
    let receipt =
        CudaExecutionReceipt::record(&rt.cuda_stream, vec![binding.clone()].into_boxed_slice())
            .unwrap();
    assert!(
        receipt
            .synchronize_required_states(&rt.cuda_stream, [&binding])
            .is_ok()
    );

    let replacement = rt
        .bind_shared_state(
            input,
            output,
            Arc::clone(&allocation),
            64,
            CudaSharedStatePolicy::RequiredInPlace,
        )
        .unwrap();
    assert!(rt.required_state_aliases.contains(&(output, input)));
    rt.bind_shared_state(
        input,
        output,
        Arc::clone(&allocation),
        64,
        CudaSharedStatePolicy::CopyBackAllowed,
    )
    .unwrap();
    assert!(!rt.required_state_aliases.contains(&(output, input)));
    let old_receipt =
        CudaExecutionReceipt::record(&rt.cuda_stream, vec![replacement].into_boxed_slice())
            .unwrap();
    assert!(
        old_receipt
            .synchronize_required_states(&rt.cuda_stream, [&binding])
            .is_err()
    );
}

#[test]
fn dirty_external_output_is_detached_before_dynamic_arena_refresh() {
    let mut rt = CudaRuntime::new().unwrap();
    let output = NodeIndex::new(126);
    let data_node = NodeIndex::new(7);
    let allocation = rt.cuda_stream.alloc_zeros::<u8>(16).unwrap();
    let ptr = allocation.device_ptr(&rt.cuda_stream).0;

    rt.resolved_output_bucket = Some(rt.active_bucket);
    rt.output_ptr_registrations.insert(output, (ptr, 16));
    rt.dirty_output_ptr_registrations.insert(output);
    rt.resolved_output_registrations
        .insert(output, ResolvedOutputRegistration::External { data_node });
    let external = unsafe { rt.cuda_stream.upgrade_device_ptr::<u8>(ptr, 16) };
    rt.external_output_buffers
        .insert(data_node, std::mem::ManuallyDrop::new(external));
    CudaRuntime::cache_bucket_device_buffer(rt.active_mut(), data_node, DeviceBuffer::new(ptr, 16));

    rt.detach_dirty_external_output_bindings();

    assert!(!rt.external_output_buffers.contains_key(&data_node));
    assert!(!rt.active().cached_device_buffers.contains_key(&data_node));
    assert!(!rt.resolved_output_registrations.contains_key(&output));
    assert!(rt.dirty_output_ptr_registrations.contains(&output));
}

#[test]
fn cached_device_buffer_tracks_exact_materialization_changes() {
    let mut bucket = CompiledBucket::new();
    let node = NodeIndex::new(7);

    bucket.materialization_fully_dirty = false;
    CudaRuntime::cache_bucket_device_buffer(&mut bucket, node, DeviceBuffer::new(0x1000, 64));
    assert_eq!(
        bucket.materialization_dirty_nodes,
        FxHashSet::from_iter([node])
    );

    bucket.materialization_dirty_nodes.clear();
    CudaRuntime::cache_bucket_device_buffer(&mut bucket, node, DeviceBuffer::new(0x1000, 64));
    assert!(bucket.materialization_dirty_nodes.is_empty());

    CudaRuntime::cache_bucket_device_buffer(&mut bucket, node, DeviceBuffer::new(0x1000, 32));
    assert_eq!(
        bucket.materialization_dirty_nodes,
        FxHashSet::from_iter([node])
    );
}

#[test]
fn dynamic_length_refresh_preserves_physical_arena_capacity() {
    let mut bucket = CompiledBucket::new();
    let node = NodeIndex::new(8);
    bucket.buffer_specs.insert(
        node,
        BufferSpec {
            bytes: Expression::from('s') * 4,
            dtype: DType::F32,
        },
    );
    bucket.cached_buffer_ptrs.insert(node, 0x2000);
    bucket
        .cached_device_buffers
        .insert(node, DeviceBuffer::new(0x2000, 16).with_capacity(16));
    bucket.last_dyn_map.insert(Symbol::from('s'), 4);

    let mut shrunk = DynMap::default();
    shrunk.insert(Symbol::from('s'), 3);
    CudaRuntime::refresh_intermediate_buffer_lengths_for_changed_dims(&mut bucket, &shrunk);
    assert_eq!(bucket.cached_device_buffers[&node].len(), 12);
    assert_eq!(bucket.cached_device_buffers[&node].capacity(), 16);

    let mut regrown = DynMap::default();
    regrown.insert(Symbol::from('s'), 4);
    CudaRuntime::refresh_intermediate_buffer_lengths_for_changed_dims(&mut bucket, &regrown);
    assert_eq!(bucket.cached_device_buffers[&node].len(), 16);
    assert_eq!(bucket.cached_device_buffers[&node].capacity(), 16);
}

#[test]
fn external_pointer_inputs_are_persistent_but_owned_inputs_remain_consumable() {
    assert!(!should_consume_hlir_input(true, false));
    assert!(!should_consume_hlir_input(true, true));
    assert!(should_consume_hlir_input(false, false));
    assert!(!should_consume_hlir_input(false, true));
}

#[test]
fn device_range_overlap_detects_hidden_output_input_aliases() {
    assert!(device_ranges_overlap(0x1000, 64, 0x1000, 64));
    assert!(device_ranges_overlap(0x1000, 64, 0x1020, 64));
    assert!(!device_ranges_overlap(0x1000, 64, 0x1040, 64));
    assert!(!device_ranges_overlap(0x1000, 0, 0x1000, 64));
}

#[test]
fn required_state_aliases_fail_closed_per_bucket() {
    let input = NodeIndex::new(1);
    let other_input = NodeIndex::new(2);
    let producer = NodeIndex::new(3);
    let output = NodeIndex::new(4);
    let required = [(output, input)];

    let mut in_place = CompiledBucket::new();
    in_place.output_producers.insert(output, producer);
    in_place.llir_to_hlir.insert(producer, input);
    assert!(validate_required_state_aliases(&required, &in_place).is_ok());

    let mut materialized = CompiledBucket::new();
    materialized.output_producers.insert(output, producer);
    assert!(validate_required_state_aliases(&required, &materialized).is_err());

    let mut wrong_input = CompiledBucket::new();
    wrong_input.output_producers.insert(output, producer);
    wrong_input.llir_to_hlir.insert(producer, other_input);
    assert!(validate_required_state_aliases(&required, &wrong_input).is_err());
}

#[test]
fn resource_validation_cache_reuses_nonconsecutive_exact_signatures() {
    let signature = |a, bytes| ResourceValidationSignature {
        allocation_dyn_maps: vec![vec![(Symbol::from('a'), a)]],
        input_footprints: vec![(7, ResourceInputFootprint::external(bytes))],
    };
    let a17 = signature(17, 64);
    let a18 = signature(18, 64);
    let a17_larger_input = signature(17, 128);
    let mut validated = FxHashSet::default();

    assert!(validated.insert(a17.clone()));
    assert!(validated.insert(a18));
    assert!(validated.contains(&a17));
    assert!(!validated.contains(&a17_larger_input));
}

#[test]
fn external_lengths_only_enter_signature_for_resource_sensitive_inputs() {
    let mut rt = CudaRuntime::new().unwrap();
    let ordinary = NodeIndex::new(126);
    let resource_sensitive = NodeIndex::new(127);
    let allocation = rt.cuda_stream.alloc_zeros::<u8>(128).unwrap();
    let ptr = allocation.device_ptr(&rt.cuda_stream).0;

    unsafe {
        rt.set_device_ptr(ordinary, ptr, 32);
        rt.set_device_ptr(resource_sensitive, ptr, 64);
    }
    rt.resource_length_sensitive_hlir.insert(resource_sensitive);

    let signature = rt.current_resource_input_signature();
    assert!(!signature.contains_key(&ordinary));
    assert_eq!(
        signature.get(&resource_sensitive),
        Some(&ResourceInputFootprint::external(64))
    );

    let original_signature = signature;
    rt.changed_hlir.clear();
    unsafe { rt.set_device_ptr(ordinary, ptr, 16) };
    assert_eq!(rt.current_resource_input_signature(), original_signature);

    unsafe { rt.set_device_ptr(resource_sensitive, ptr, 32) };
    assert_ne!(rt.current_resource_input_signature(), original_signature);
}

#[test]
fn set_data_reuses_hlir_buffer_when_payload_fits() {
    let mut rt = CudaRuntime::new().unwrap();
    let input = NodeIndex::new(123);

    rt.set_data(input, vec![1i32, 2, 3, 4]);
    let (first_ptr, first_capacity, first_len) = match rt.hlir_buffers.get(&input).unwrap() {
        CudaInput::Buffer { buf, len } => (buf.device_ptr(&rt.cuda_stream).0, buf.len(), *len),
        CudaInput::Ptr(_) => panic!("set_data must create an owned CUDA buffer"),
    };
    assert_eq!(first_capacity, 16);
    assert_eq!(first_len, 16);

    rt.set_data(input, vec![9i32, 8]);
    let (second_ptr, second_capacity, second_len) = match rt.hlir_buffers.get(&input).unwrap() {
        CudaInput::Buffer { buf, len } => (buf.device_ptr(&rt.cuda_stream).0, buf.len(), *len),
        CudaInput::Ptr(_) => panic!("set_data must keep an owned CUDA buffer"),
    };

    assert_eq!(second_ptr, first_ptr);
    assert_eq!(second_capacity, first_capacity);
    assert_eq!(second_len, 8);

    let bytes = DeviceBuffer::new(second_ptr, second_len)
        .clone_dtoh(&rt.cuda_stream)
        .unwrap();
    assert_eq!(bytemuck::cast_slice::<u8, i32>(&bytes), &[9, 8]);
}

#[test]
fn host_mirrors_are_opt_in_and_cleared_by_ordinary_rebinding() {
    let mut rt = CudaRuntime::new().unwrap();
    let input = NodeIndex::new(125);

    rt.set_data_with_host_mirror(input, vec![1i32, 2, 3]);
    assert_eq!(
        bytemuck::cast_slice::<u8, i32>(&rt.hlir_host_mirrors[&input]),
        &[1, 2, 3]
    );

    rt.set_data(input, vec![4i32, 5]);
    assert!(!rt.hlir_host_mirrors.contains_key(&input));

    rt.set_data_with_host_mirror(input, vec![6i32]);
    rt.set_zeros(input, 4);
    assert!(!rt.hlir_host_mirrors.contains_key(&input));
}

#[test]
fn set_data_mutates_reserved_hlir_buffer_in_place() {
    let mut rt = CudaRuntime::new().unwrap();
    let input = NodeIndex::new(124);

    rt.set_data_with_capacity(input, vec![1i32, 2], 16);
    let first_ptr = match rt.hlir_buffers.get(&input).unwrap() {
        CudaInput::Buffer { buf, len } => {
            assert_eq!(buf.len(), 16);
            assert_eq!(*len, 8);
            buf.device_ptr(&rt.cuda_stream).0
        }
        CudaInput::Ptr(_) => panic!("set_data_with_capacity must create an owned buffer"),
    };

    rt.set_data(input, vec![3i32, 4, 5, 6]);
    let (second_ptr, second_len) = match rt.hlir_buffers.get(&input).unwrap() {
        CudaInput::Buffer { buf, len } => (buf.device_ptr(&rt.cuda_stream).0, *len),
        CudaInput::Ptr(_) => panic!("set_data must keep an owned buffer"),
    };
    assert_eq!(second_ptr, first_ptr);
    assert_eq!(second_len, 16);

    let bytes = DeviceBuffer::new(second_ptr, second_len)
        .clone_dtoh(&rt.cuda_stream)
        .unwrap();
    assert_eq!(bytemuck::cast_slice::<u8, i32>(&bytes), &[3, 4, 5, 6]);

    rt.set_data(input, vec![0i32; 5]);
    let (third_ptr, third_len) = match rt.hlir_buffers.get(&input).unwrap() {
        CudaInput::Buffer { buf, len } => (buf.device_ptr(&rt.cuda_stream).0, *len),
        CudaInput::Ptr(_) => panic!("set_data must keep an owned buffer"),
    };
    assert_ne!(third_ptr, first_ptr);
    assert_eq!(third_len, 20);
}

#[test]
fn free_intermediate_buffers_invalidates_hlir_sync() {
    let mut rt = CudaRuntime::new().unwrap();
    let mut bucket = CompiledBucket::new();
    let llir_input = NodeIndex::new(0);
    bucket.hlir_synced = true;
    bucket.cached_buffer_ptrs.insert(llir_input, 0x1000);
    bucket
        .cached_device_buffers
        .insert(llir_input, DeviceBuffer::new(0x1000, 16));
    rt.compiled_buckets.push(bucket);

    rt.free_intermediate_buffers();

    let bucket = &rt.compiled_buckets[0];
    assert!(!bucket.hlir_synced);
    assert!(bucket.cached_buffer_ptrs.is_empty());
    assert!(bucket.cached_device_buffers.is_empty());
}

#[test]
fn bucket_memory_dry_plan_uses_bucket_capacity_dims() {
    let data = NodeIndex::new(1);
    let mut bucket = CompiledBucket::new();
    bucket.bucket_indices.insert(Symbol::from('s'), 1);
    bucket.buffer_specs.insert(
        data,
        BufferSpec {
            bytes: Expression::from('s') * 4,
            dtype: DType::F32,
        },
    );
    bucket.output_producers.insert(NodeIndex::new(99), data);

    let mut dim_buckets = FxHashMap::default();
    dim_buckets.insert(
        Symbol::from('s'),
        vec![DimBucket::new(1, 1), DimBucket::new(2, 64)],
    );

    let mut representative_dyn_map = FxHashMap::default();
    representative_dyn_map.insert(Symbol::from('s'), 16);
    let capacity_dyn_map = CudaRuntime::bucket_capacity_dyn_map_from_context(
        &representative_dyn_map,
        &bucket.bucket_indices,
        &dim_buckets,
    );

    CudaRuntime::dry_plan_intermediate_buffers(&mut bucket, &capacity_dyn_map);

    assert_eq!(capacity_dyn_map[&Symbol::from('s')], 64);
    assert_eq!(bucket.arena_bytes, align_up(64 * 4, ARENA_ALIGNMENT));
    assert_eq!(
        CudaRuntime::planned_allocation_bytes(&bucket),
        bucket.arena_bytes
    );
}

#[test]
fn retained_bucket_plan_uses_peak_live_arena_before_allocation() {
    let mut buckets = Vec::new();
    for (node, bytes) in [
        (NodeIndex::new(1), 64usize),
        (NodeIndex::new(2), ARENA_ALIGNMENT * 2),
    ] {
        let mut bucket = CompiledBucket::new();
        bucket.buffer_specs.insert(
            node,
            BufferSpec {
                bytes: bytes.into(),
                dtype: DType::F32,
            },
        );
        bucket.output_producers.insert(NodeIndex::new(100), node);
        buckets.push(bucket);
    }
    let dyn_maps = vec![FxHashMap::default(), FxHashMap::default()];

    let aggregate = CudaRuntime::retained_bucket_resource_plan(
        &mut buckets,
        &dyn_maps,
        &FxHashMap::default(),
        &mut CompiledFunctionResourceCache::default(),
        CandidateResourceCaps::default(),
        None,
    )
    .unwrap();
    let individual_bytes = buckets
        .iter()
        .map(CudaRuntime::planned_allocation_bytes)
        .collect_vec();

    assert_eq!(
        aggregate.planned_intermediate_bytes,
        individual_bytes.iter().copied().max()
    );
    let limit = *individual_bytes.iter().max().unwrap();
    assert!(individual_bytes.iter().sum::<usize>() > limit);
    assert!(individual_bytes.iter().all(|bytes| *bytes <= limit));
    assert!(
        validate_resource_plan(
            &aggregate,
            CandidateResourceCaps {
                max_intermediate_bytes: Some(limit),
                max_kernel_source_bytes: None,
            },
            None,
        )
        .is_ok()
    );
    assert!(matches!(
        validate_resource_plan(
            &aggregate,
            CandidateResourceCaps {
                max_intermediate_bytes: Some(limit - 1),
                max_kernel_source_bytes: None,
            },
            None,
        ),
        Err(ResourceViolation::IntermediateMemory { .. })
    ));
}

#[test]
fn host_memory_aggregation_preserves_lifetimes_and_shared_dedup() {
    let shared = SharedDeviceMemoryAllocation {
        key: "shared-workspace",
        bytes: 64,
    };
    let buckets = vec![
        vec![
            HostDeviceMemoryPlan {
                persistent_bytes: 10,
                active_bucket_bytes: 7,
                transient_peak_bytes: 100,
                shared_allocations: vec![shared.clone()],
            },
            HostDeviceMemoryPlan {
                persistent_bytes: 20,
                active_bucket_bytes: 13,
                transient_peak_bytes: 40,
                shared_allocations: vec![shared.clone()],
            },
        ],
        vec![HostDeviceMemoryPlan {
            persistent_bytes: 30,
            active_bucket_bytes: 50,
            transient_peak_bytes: 80,
            shared_allocations: vec![shared],
        }],
    ];

    let (retained, transient_peak, shared) =
        CudaRuntime::aggregate_host_device_memory(&buckets, &[]).unwrap();

    assert_eq!(
        retained, 130,
        "compiled and materialized plans coexist across buckets"
    );
    assert_eq!(
        transient_peak, 100,
        "per-execution allocations peak across stream-ordered buckets"
    );
    assert_eq!(shared.len(), 1, "the keyed workspace is counted once");
    assert_eq!(shared[0].bytes, 64);
}

#[test]
fn resident_shared_memory_survives_into_non_host_and_flash_plans() {
    let resident = crate::providers::flashinfer::shared_device_memory_allocation();

    let (_, _, non_flash_shared) =
        CudaRuntime::aggregate_host_device_memory(&[Vec::new()], std::slice::from_ref(&resident))
            .unwrap();
    assert_eq!(non_flash_shared, vec![resident.clone()]);

    let flash_plan = HostDeviceMemoryPlan {
        shared_allocations: vec![resident.clone()],
        ..Default::default()
    };
    let (_, _, flash_shared) = CudaRuntime::aggregate_host_device_memory(
        &[vec![flash_plan]],
        std::slice::from_ref(&resident),
    )
    .unwrap();
    assert_eq!(flash_shared, vec![resident], "shared key is charged once");
}

#[test]
fn fixed_arena_slot_refresh_grows_capacity_without_reassigning_slots() {
    let a = NodeIndex::new(1);
    let b = NodeIndex::new(2);
    let mut bucket = CompiledBucket::new();
    bucket.stabilize_intermediate_pointers = true;
    bucket.buffer_specs.insert(
        a,
        BufferSpec {
            bytes: Expression::from('s') * 4,
            dtype: DType::F32,
        },
    );
    bucket.buffer_specs.insert(
        b,
        BufferSpec {
            bytes: Expression::from('s') * 8,
            dtype: DType::F32,
        },
    );
    bucket.logical_buffer_slots.insert(a, 0);
    bucket.logical_buffer_slots.insert(b, 0);
    bucket.arena_slots.push(ArenaSlot {
        members: vec![
            PlannedBuffer {
                node: a,
                bytes: 1,
                start: 0,
                end: 0,
            },
            PlannedBuffer {
                node: b,
                bytes: 1,
                start: 1,
                end: 1,
            },
        ],
        offset: 0,
        capacity_bytes: 0,
    });

    let mut dyn_map = FxHashMap::default();
    dyn_map.insert(Symbol::from('s'), 4);
    CudaRuntime::refresh_fixed_intermediate_buffer_plan(&mut bucket, &dyn_map);
    let first_offset_a = bucket.logical_buffer_offsets[&a];
    let first_offset_b = bucket.logical_buffer_offsets[&b];
    let first_arena_bytes = bucket.arena_bytes;

    dyn_map.insert(Symbol::from('s'), 32);
    CudaRuntime::refresh_fixed_intermediate_buffer_plan(&mut bucket, &dyn_map);

    assert_eq!(bucket.logical_buffer_slots[&a], 0);
    assert_eq!(bucket.logical_buffer_slots[&b], 0);
    assert_eq!(bucket.logical_buffer_offsets[&a], first_offset_a);
    assert_eq!(bucket.logical_buffer_offsets[&b], first_offset_b);
    assert!(bucket.arena_bytes >= first_arena_bytes);
    assert_eq!(bucket.arena_slots.len(), 1);
}

#[test]
fn fixed_arena_slot_assignment_respects_lifetime_overlap() {
    let a = NodeIndex::new(1);
    let b = NodeIndex::new(2);
    let planned = vec![
        PlannedBuffer {
            node: a,
            bytes: 16,
            start: 0,
            end: 0,
        },
        PlannedBuffer {
            node: b,
            bytes: 16,
            start: 1,
            end: 1,
        },
    ];

    let mut disjoint = CompiledBucket::new();
    CudaRuntime::assign_fixed_arena_slots(&mut disjoint, planned.clone());
    assert_eq!(disjoint.arena_slots.len(), 1);

    let mut overlapping = CompiledBucket::new();
    let mut planned = planned;
    planned[1].start = 0;
    CudaRuntime::assign_fixed_arena_slots(&mut overlapping, planned);
    assert_eq!(overlapping.arena_slots.len(), 2);
}
