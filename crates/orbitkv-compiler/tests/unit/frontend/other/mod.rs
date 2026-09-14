use crate::{hlir::ReferenceData, prelude::*, tests::assert_close};
use candle_core::{Device, Tensor};
use proptest::prelude::*;

pub fn test_init(func: impl Fn(&mut Graph) -> GraphTensor, ref_func: impl Fn(&Device) -> Tensor) {
    let mut cx = Graph::new();
    let b = func(&mut cx).output();

    cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
    let mut rt = cx.search(
        ReferenceRuntime::default(),
        CompileOptions::default().search_graph_limit(1),
    );

    rt.execute(&cx.dyn_map);

    // Reference
    let device = Device::Cpu;
    let ref_b = ref_func(&device).flatten_all().unwrap();

    // need to assert close because some unaries (exp and log) are (good) approximations
    assert_close(rt.get_f32(b.id), &ref_b.to_vec1::<f32>().unwrap())
}

#[test]
fn constant_i64_preserves_full_width_values() {
    let values = [i64::MIN, -(1i64 << 40) + 7, -1, 0, 1i64 << 40, i64::MAX];
    let mut cx = Graph::new();
    for &value in &values {
        cx.constant_i64(value).output();
    }

    cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
    let mut runtime = cx.search(
        ReferenceRuntime::default(),
        CompileOptions::default().search_graph_limit(1),
    );
    runtime.execute(&cx.dyn_map);

    let mut actual = runtime
        .buffers
        .values()
        .map(|data| match data {
            ReferenceData::I64(values) => values[0],
            other => panic!("expected I64 output, got {other:?}"),
        })
        .collect::<Vec<_>>();
    actual.sort();
    let mut expected = values.to_vec();
    expected.sort();
    assert_eq!(actual, expected);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]
    #[test]
    fn test_arange(end in 1i32..64) {
        test_init(
            |cx| cx.arange(end).cast(DType::F32) * 1.0,
            |dev| Tensor::arange(0_f32, end as f32, dev).unwrap(),
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]
    #[test]
    fn test_arange_options(start in -16i32..16, step in 1i32..6, count in 1i32..20) {
        let end = start + step * count;
        test_init(
            |cx| cx.arange_options(start, end, step).cast(DType::F32) * 1.0,
            |dev| {
                let values = (0..count)
                    .map(|i| (start + step * i) as f32)
                    .collect::<Vec<f32>>();
                Tensor::from_vec(values, count as usize, dev).unwrap()
            },
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]
    #[test]
    fn test_gather(base_len in 5usize..64, count in 1usize..16) {
        prop_assume!(base_len >= 2 * count - 1);
        test_init(
            |cx| {
                cx.arange(base_len as i32)
                    .cast(DType::F32)
                    .gather(cx.iota(Expression::from('z') * 2, count as i32))
            },
            |dev| {
                let values = (0..count).map(|i| (2 * i) as f32).collect::<Vec<f32>>();
                Tensor::new(values, dev).unwrap()
            },
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]
    #[test]
    fn test_triangle_mask(size in 1usize..64) {
        test_init(
            |cx| cx.tril(size as i32, 0).cast(DType::F32),
            |dev| Tensor::tril2(size, candle_core::DType::F32, dev).unwrap(),
        );
        test_init(
            |cx| cx.triu(size as i32, 0).cast(DType::F32),
            |dev| Tensor::triu2(size, candle_core::DType::F32, dev).unwrap(),
        );
    }
}

#[test]
fn test_stack() {
    use crate::tests::random_vec;

    let mut cx = Graph::new();
    let a = cx.tensor((2, 3));
    let b = cx.tensor((2, 3));
    let c = cx.tensor((2, 3));
    let stacked = cx.stack(&[a, b, c], 0).output();

    cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
    let mut rt = cx.search(
        ReferenceRuntime::default(),
        CompileOptions::default().search_graph_limit(1),
    );

    let a_data = random_vec(6);
    let b_data = random_vec(6);
    let c_data = random_vec(6);
    rt.set_data(a.id, a_data.clone());
    rt.set_data(b.id, b_data.clone());
    rt.set_data(c.id, c_data.clone());
    rt.execute(&cx.dyn_map);

    let ref_a = Tensor::new(a_data, &Device::Cpu)
        .unwrap()
        .reshape((2, 3))
        .unwrap();
    let ref_b = Tensor::new(b_data, &Device::Cpu)
        .unwrap()
        .reshape((2, 3))
        .unwrap();
    let ref_c = Tensor::new(c_data, &Device::Cpu)
        .unwrap()
        .reshape((2, 3))
        .unwrap();
    let ref_stacked = Tensor::stack(&[&ref_a, &ref_b, &ref_c], 0).unwrap();

    assert_close(
        rt.get_f32(stacked.id),
        &ref_stacked.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
    );
}
