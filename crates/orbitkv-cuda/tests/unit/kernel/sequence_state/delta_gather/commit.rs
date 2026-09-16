use super::*;
use cudarc::driver::{CudaContext, DevicePtr};

#[test]
#[ignore = "requires CUDA; forces gathered reads with required in-place commits across steps"]
fn gathered_reads_and_required_in_place_commits_reuse_updated_slots() {
    const REQUESTS: usize = 2;
    const SLOTS: usize = 5;
    let spec = spec(2);
    let state_elements = spec.value_heads * spec.key_width * spec.value_width;
    let mut graph = Graph::new();
    let query = graph.tensor((REQUESTS, spec.key_heads, spec.key_width));
    let key = graph.tensor((REQUESTS, spec.key_heads, spec.key_width));
    let value = graph.tensor((REQUESTS, spec.value_heads, spec.value_width));
    let decay = graph.tensor((REQUESTS, spec.value_heads));
    let beta = graph.tensor((REQUESTS, spec.value_heads));
    let arena = graph.tensor(SLOTS * state_elements).persist();
    let indices = graph
        .tensor((REQUESTS, spec.value_heads, spec.key_width, spec.value_width))
        .as_dtype(DType::Int);
    let indptr = graph.tensor(REQUESTS + 1).as_dtype(DType::Int);
    let output = packed_delta_scan(
        PackedDeltaScanPlan {
            query,
            key,
            value,
            log_decay: decay,
            update_gate: beta,
            state: arena.gather(indices),
            query_indptr: indptr,
        },
        spec,
    );
    output.values.output();
    output.state.output();
    let committed = output.state.scatter(indices, arena).output();
    assert!(has_candidate(&mut graph));
    let candidate = extract(&graph, true);
    let baseline =
        try_extract_forced_op_llir_where(&graph, &["KernelDeltaRegisters"], EXTRACTION, |llir| {
            let names = llir_kernel_names(llir);
            names.contains(&"PackedDeltaRegisters")
                && names.contains(&"ScatterNoCopy")
                && !names.contains(&"PackedDeltaGather")
        })
        .expect("the materialized scan and in-place commit must remain reachable");
    let initial = (0..SLOTS * state_elements)
        .map(|index| (index as f32 - 23.0) / 64.0 + 1.0 / 8192.0)
        .collect::<Vec<_>>();
    let context = CudaContext::new(0).unwrap();
    let mut snapshots = Vec::new();
    for program in [&baseline, &candidate] {
        let stream = context.new_stream().unwrap();
        let mut runtime = CudaRuntime::initialize(stream.clone());
        let mut allocation = runtime.alias_state_required(
            arena,
            committed,
            initial.len() * std::mem::size_of::<f32>(),
        );
        let address = allocation.device_ptr(&stream).0;
        stream
            .memcpy_htod(bytemuck::cast_slice(&initial), &mut allocation)
            .unwrap();
        let mut previous = initial
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        let mut steps = Vec::new();
        // Slot 3 is updated again using its first-step state; slots 1 and 0
        // change membership, while slots 2 and 4 must stay untouched throughout.
        for (step, slots) in [[1, 3], [3, 0]].into_iter().enumerate() {
            let positions = slots
                .iter()
                .flat_map(|&slot| {
                    (0..state_elements).map(move |offset| slot * state_elements + offset)
                })
                .collect::<Vec<_>>();
            runtime.set_data(
                indices,
                positions
                    .iter()
                    .map(|&index| index as i32)
                    .collect::<Vec<_>>(),
            );
            runtime.set_data(indptr, vec![0_i32, 1, 2]);
            runtime.set_data(query, vec![0.5_f32, -0.25, -0.5, 0.75]);
            runtime.set_data(key, vec![0.25_f32, 0.75, -0.5, 0.25]);
            runtime.set_data(
                value,
                (0..REQUESTS * spec.value_heads * spec.value_width)
                    .map(|index| (index as f32 - 4.0) / 8.0 + step as f32 * 0.25)
                    .collect::<Vec<_>>(),
            );
            runtime.set_data(decay, vec![-0.125_f32; REQUESTS * spec.value_heads]);
            runtime.set_data(beta, vec![0.625_f32; REQUESTS * spec.value_heads]);
            if step == 0 {
                runtime.load_llir(program);
                assert!(runtime.output_aliases_input_in_all_buckets(committed, arena));
            }
            runtime.execute(&graph.dyn_map);
            assert_eq!(allocation.device_ptr(&stream).0, address);
            let values = runtime
                .get_f32(output.values)
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>();
            let selected = runtime
                .get_f32(output.state)
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>();
            assert_eq!(values.len(), REQUESTS * spec.value_heads * spec.value_width);
            assert_eq!(selected.len(), positions.len());
            assert!(
                selected
                    .iter()
                    .zip(&positions)
                    .any(|(&next, &index)| next != previous[index]),
                "each step must actually update persistent state"
            );
            for (&index, &next) in positions.iter().zip(&selected) {
                previous[index] = next;
            }
            let actual = stream
                .clone_dtoh(&allocation)
                .unwrap()
                .chunks_exact(std::mem::size_of::<u32>())
                .map(|bytes| u32::from_ne_bytes(bytes.try_into().unwrap()))
                .collect::<Vec<_>>();
            assert_eq!(
                actual, previous,
                "step {step}: selected updates and untouched slots"
            );
            steps.push((values, selected, actual));
        }
        snapshots.push(steps);
    }
    assert_eq!(
        snapshots[1], snapshots[0],
        "gathered and materialized scans must agree at both steps"
    );
}
