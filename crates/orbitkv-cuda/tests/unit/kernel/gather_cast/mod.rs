use half::{bf16, f16};
use orbitkv_compiler::prelude::*;

use crate::{
    runtime::CudaRuntime,
    tests::utilities::{
        ForcedExtractionConfig, get_cuda_stream, llir_kernel_names,
        try_extract_forced_op_llir_where,
    },
};

fn rounded(value: f32, dtype: DType) -> f32 {
    match dtype {
        DType::Bf16 => bf16::from_f32(value).to_f32(),
        DType::F16 => f16::from_f32(value).to_f32(),
        DType::F32 => value,
        _ => unreachable!(),
    }
}

fn read(runtime: &CudaRuntime, tensor: GraphTensor) -> Vec<f32> {
    match tensor.dtype {
        DType::Bf16 => runtime
            .get_bf16(tensor)
            .into_iter()
            .map(bf16::to_f32)
            .collect(),
        DType::F16 => runtime
            .get_f16(tensor)
            .into_iter()
            .map(f16::to_f32)
            .collect(),
        DType::F32 => runtime.get_f32(tensor),
        _ => unreachable!(),
    }
}

#[test]
fn both_fusion_directions_preserve_strides_cast_rounding_and_shared_inputs() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    for (from, to) in [
        (DType::Bf16, DType::F32),
        (DType::F16, DType::F32),
        (DType::F32, DType::Bf16),
        (DType::F32, DType::F16),
    ] {
        for before in [false, true] {
            let mut graph = Graph::new();
            // Each row has a leading guard element outside the selected view.
            let data = graph.tensor((3, Expression::from('w') + 1)).as_dtype(from);
            let indices = graph.tensor(('r', 4)).as_dtype(DType::Int);
            let view = data.permute((1, 0)).slice((1.., ..));
            let picks = indices.permute((1, 0));
            let selected = if before {
                view.cast(to).gather(picks)
            } else {
                // Casting the prefix must retain its smaller addressed span.
                view.gather(picks).slice((..2, ..)).cast(to)
            };
            let output = selected.output();
            let observer = data.cast(DType::F32).output();
            graph.set_dim('w', 5);
            graph.set_dim('r', 2);
            graph.build_search_space::<CudaRuntime>(CompileOptions::default());
            let llir = try_extract_forced_op_llir_where(
                &graph,
                &["KernelGatherCast"],
                ForcedExtractionConfig::new(0x06A7_CA57),
                |llir| llir_kernel_names(llir).contains(&"GatherCast"),
            )
            .expect("the fused candidate must remain reachable");
            let mut runtime = CudaRuntime::initialize(stream.clone());
            // One extracted program must honor both runtime geometries.
            for (width, rows) in [(5, 2), (7, 3)] {
                graph.set_dim('w', width);
                graph.set_dim('r', rows);
                let values = (0..3 * (width + 1))
                    .map(|i| rounded((i as f32 - 9.0) * 0.137, from))
                    .collect::<Vec<_>>();
                let ids = (0..rows * 4)
                    .map(|i| ((i * 7 + 3) % (width * 3)) as i32)
                    .collect::<Vec<_>>();
                runtime.set_data(indices, ids.clone());
                match from {
                    DType::F32 => runtime.set_data(data, values.clone()),
                    DType::F16 => runtime.set_data(
                        data,
                        values
                            .iter()
                            .copied()
                            .map(f16::from_f32)
                            .collect::<Vec<_>>(),
                    ),
                    DType::Bf16 => runtime.set_data(
                        data,
                        values
                            .iter()
                            .copied()
                            .map(bf16::from_f32)
                            .collect::<Vec<_>>(),
                    ),
                    _ => unreachable!(),
                }
                runtime.load_llir(&llir);
                runtime.execute(&graph.dyn_map);
                let columns = if before { 4 } else { 2 };
                let mut expected = Vec::new();
                for column in 0..columns {
                    for row in 0..rows {
                        let index = ids[row * 4 + column] as usize;
                        expected.push(rounded(values[1 + index / 3 + index % 3 * (width + 1)], to));
                    }
                }
                assert_eq!(
                    read(&runtime, output),
                    expected,
                    "{from:?} -> {to:?}, before={before}"
                );
                assert_eq!(
                    read(&runtime, observer),
                    values,
                    "shared source was modified"
                );
            }
        }
    }
}

#[test]
fn unsupported_conversion_stays_on_the_original_path() {
    let mut graph = Graph::new();
    let data = graph.tensor(8).as_dtype(DType::Int);
    let indices = graph.tensor(3).as_dtype(DType::Int);
    data.gather(indices).cast(DType::F32).output();
    graph.build_search_space::<CudaRuntime>(CompileOptions::default());
    assert!(
        !graph
            .egraph()
            .unwrap()
            .enodes
            .values()
            .any(|(kind, _)| kind == "KernelGatherCast")
    );
}
