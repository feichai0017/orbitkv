use super::*;
use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};

use crate::tests::utilities::{llir_kernel_names, try_extract_forced_op_llir_where};

#[test]
#[ignore = "requires CUDA; compares every finite BF16 gate against the original GPU graph"]
fn all_finite_bf16_gates_match_the_unfused_gpu_graph() {
    let finite = (0..=u16::MAX)
        .map(bf16::from_bits)
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    // A permutation of every finite BF16, alternating signs and starting at
    // ordinary magnitudes, so the tiny dynamic-row cases are meaningful too.
    let half = finite.len() / 2;
    let values = (0..finite.len())
        .map(|i| finite[(i / 2 * 251 + 0x3f80) % half + i % 2 * half])
        .collect::<Vec<_>>();
    compare_graphs(
        VECTOR_ELEMENTS,
        &[1, 3, values.len() / VECTOR_ELEMENTS],
        &values,
    );
    // The current model's MLP shape also exercises many vector blocks, using
    // the same generic rule and arithmetic as the small exhaustive case.
    compare_graphs(17_408, &[1, 8, 32], &values);
}

fn compare_graphs(width: usize, row_counts: &[usize], values: &[bf16]) {
    let mut graph = Graph::new();
    graph.set_dim_interval('s', 1, *row_counts.iter().max().unwrap() as i64);
    let (gate, up) = inputs(&mut graph, 's'.into(), width);
    let output = activation_product(gate, up).output();
    assert!(contains_candidate(&mut graph));
    let candidate =
        extract_forced_kernel_llir(&graph, "KernelSiluMul", "SiluMul", EXTRACTION, false);
    let baseline = try_extract_forced_op_llir_where(&graph, &["KernelCast"], EXTRACTION, |llir| {
        !llir_kernel_names(llir).contains(&"SiluMul")
    })
    .expect("the existing primitive implementation must remain reachable");
    let context = CudaContext::new(0).unwrap();
    let mut baseline_runtime = CudaRuntime::initialize(context.new_stream().unwrap());
    let mut candidate_runtime = CudaRuntime::initialize(context.new_stream().unwrap());
    for &rows in row_counts {
        graph.set_dim('s', rows);
        let values = values
            .iter()
            .copied()
            .cycle()
            .take(rows * width)
            .collect::<Vec<_>>();
        let multipliers = (0..values.len())
            .map(|i| bf16::from_f32([0.5, 1.0, 1.5, -0.75][i % 4]))
            .collect::<Vec<_>>();
        for (runtime, program) in [
            (&mut baseline_runtime, &baseline),
            (&mut candidate_runtime, &candidate),
        ] {
            runtime.set_data(gate, values.clone());
            runtime.set_data(up, multipliers.clone());
            runtime.load_llir(program);
            runtime.execute(&graph.dyn_map);
        }
        let expected = baseline_runtime.get_bf16(output.id);
        let actual = candidate_runtime.get_bf16(output.id);
        assert_eq!(expected.len(), rows * width);
        assert_eq!(actual.len(), rows * width);
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "width={width}, rows={rows}, index={index}, gate={:?}, up={:?}",
                values[index],
                multipliers[index]
            );
        }
    }
}

#[test]
#[ignore = "requires CUDA; exercises unaligned external pointers and partial blocks"]
fn unaligned_external_buffers_use_the_scalar_access_fallback() {
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let count = VECTOR_ELEMENTS * 3;
    let kernel = KernelSiluMul { size: count.into() };
    let (function, _module, _, _, _, _, _) = kernel.compile(&stream, &mut FxHashMap::default());
    let gates = (0..count)
        .map(|i| bf16::from_f32((i as f32 - 12.0) * 0.5))
        .collect::<Vec<_>>();
    let ups = vec![bf16::from_f32(0.5); count];
    let guard = bf16::from_f32(19.0);
    let mut gate_storage = vec![guard.to_bits()];
    gate_storage.extend(gates.iter().map(|value| value.to_bits()));
    let mut up_storage = vec![guard.to_bits()];
    up_storage.extend(ups.iter().map(|value| value.to_bits()));
    let gate_buffer = stream.clone_htod(&gate_storage).unwrap();
    let up_buffer = stream.clone_htod(&up_storage).unwrap();
    let mut out_buffer = stream
        .clone_htod(&vec![guard.to_bits(); count + 2])
        .unwrap();
    unsafe {
        stream
            .launch_builder(&function)
            .arg(&mut out_buffer.slice_mut(1..count + 1))
            .arg(&gate_buffer.slice(1..))
            .arg(&up_buffer.slice(1..))
            .launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (THREADS as u32, 1, 1),
                shared_mem_bytes: 0,
            })
            .unwrap();
    }
    let actual = stream.clone_dtoh(&out_buffer).unwrap();
    assert_eq!(actual[0], guard.to_bits());
    assert_eq!(actual[count + 1], guard.to_bits());
    let expected = gates
        .into_iter()
        .zip(ups)
        .map(|(gate, up)| reference(gate, up).to_bits())
        .collect::<Vec<_>>();
    assert_eq!(actual[1..count + 1], expected);
}
