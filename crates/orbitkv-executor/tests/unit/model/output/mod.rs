use super::*;
use orbitkv_compiler::prelude::{CompileOptions, DType, Graph, ReferenceRuntime, Runtime};

mod checkpoint;

#[test]
fn selects_last_hidden_rows_in_ragged_request_order() {
    let mut graph = Graph::new();
    let hidden = graph.tensor(('s', 3)).as_dtype(DType::Bf16);
    let indptr = graph
        .tensor(orbitkv_compiler::prelude::Expression::from('b') + 1)
        .as_dtype(DType::Int);
    let selected = DecoderOutputRows::LastTokenPerRequest
        .select(&hidden, &indptr)
        .cast(DType::F32)
        .output();
    let all = DecoderOutputRows::AllTokens
        .select(&hidden, &indptr)
        .cast(DType::F32)
        .output();
    let mut runtime = graph.compile(ReferenceRuntime::default(), CompileOptions::default());
    // Decode, mixed decode/prefill, and a single longer prompt share the same
    // symbolic graph. Distinct row values catch request/terminal-offset errors.
    for offsets in [&[0_i32, 1, 2, 3][..], &[0, 2, 3, 7], &[0, 9]] {
        let tokens = usize::try_from(*offsets.last().unwrap()).unwrap();
        graph.set_dim('s', tokens);
        graph.set_dim('b', offsets.len() - 1);
        let values = (0..tokens * 3)
            .map(|index| half::bf16::from_f32(f32::from(u16::try_from(index).unwrap())))
            .collect::<Vec<_>>();
        runtime.set_data(hidden, values.clone());
        runtime.set_data(indptr, offsets.to_vec());
        runtime.execute(&graph.dyn_map);
        let expected = offsets[1..]
            .iter()
            .flat_map(|&end| {
                let start = (usize::try_from(end).unwrap() - 1) * 3;
                values[start..start + 3].iter().map(|value| value.to_f32())
            })
            .collect::<Vec<_>>();
        assert_eq!(runtime.get_f32(selected), &expected);
        assert_eq!(
            runtime.get_f32(all),
            &values
                .iter()
                .map(|value| value.to_f32())
                .collect::<Vec<_>>()
        );
    }
}
