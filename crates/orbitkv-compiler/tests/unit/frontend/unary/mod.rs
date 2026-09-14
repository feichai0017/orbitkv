use std::collections::BinaryHeap;

use crate::{
    prelude::*,
    tests::{assert_close, random_vec},
};
use candle_core::{Device, Tensor};
use candle_nn::ops::softmax;
use itertools::Itertools;
use ordered_float::NotNan;
use proptest::prelude::*;

fn cummax_ref_2d(a: Tensor) -> Tensor {
    let v = a.to_vec2::<f32>().unwrap();
    let mut out = vec![vec![0.0; v[0].len()]; v.len()];
    for (i, row) in v.iter().enumerate() {
        let mut acc = f32::NEG_INFINITY;
        for (j, val) in row.iter().enumerate() {
            acc = acc.max(*val);
            out[i][j] = acc;
        }
    }
    Tensor::new(out, a.device()).unwrap()
}

fn cumprod_ref_2d(a: Tensor) -> Tensor {
    let v = a.to_vec2::<f32>().unwrap();
    let mut out = vec![vec![0.0; v[0].len()]; v.len()];
    for (i, row) in v.iter().enumerate() {
        let mut acc = 1.0;
        for (j, val) in row.iter().enumerate() {
            acc *= val;
            out[i][j] = acc;
        }
    }
    Tensor::new(out, a.device()).unwrap()
}

pub fn test_unary(
    shape: impl ToShape,
    func: impl Fn(GraphTensor) -> GraphTensor,
    ref_func: impl Fn(Tensor) -> Tensor,
) {
    let shape = shape
        .to_shape()
        .into_iter()
        .map(|e| e.to_usize().unwrap())
        .collect_vec();
    let mut cx = Graph::new();
    let a = cx.tensor(shape.clone());
    let b = func(a).output();

    cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
    let mut rt = cx.search(
        ReferenceRuntime::default(),
        CompileOptions::default().search_graph_limit(1),
    );

    let v = random_vec(shape.iter().copied().product());
    rt.set_data(a.id, v.clone());
    rt.execute(&cx.dyn_map);

    // Reference
    let device = Device::Cpu;
    let ref_a = Tensor::new(v, &device).unwrap().reshape(shape).unwrap();
    let ref_b = ref_func(ref_a).flatten_all().unwrap();

    // need to assert close because some unaries (exp and log) are (good) approximations
    assert_close(rt.get_f32(b.id), &ref_b.to_vec1::<f32>().unwrap())
}

#[test]
fn extrema_ties_choose_highest_index_on_each_axis() {
    let data = [
        [-2.0, -2.0, -5.0, -5.0],
        [-2.0, -7.0, -5.0, -9.0],
        [-8.0, -7.0, -5.0, -9.0],
    ];
    let mut graph = Graph::new();
    let input = graph.tensor((data.len(), data[0].len()));
    let outputs = [
        input.argmax(0).cast(DType::F32).output(),
        input.argmin(0).cast(DType::F32).output(),
        input.argmax(1).cast(DType::F32).output(),
        input.argmin(1).cast(DType::F32).output(),
    ];
    graph.build_search_space::<ReferenceRuntime>(CompileOptions::default());
    let mut runtime = graph.search(
        ReferenceRuntime::default(),
        CompileOptions::default().search_graph_limit(1),
    );
    runtime.set_data(input.id, data.into_iter().flatten().collect::<Vec<_>>());
    runtime.execute(&graph.dyn_map);
    let expected: [&[f32]; 4] = [
        &[1.0, 0.0, 2.0, 0.0],
        &[2.0, 2.0, 2.0, 2.0],
        &[1.0, 0.0, 2.0],
        &[3.0, 3.0, 3.0],
    ];
    for (output, expected) in outputs.into_iter().zip(expected) {
        assert_eq!(runtime.get_f32(output.id), expected);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]

    #[test]
    fn test_exp(size in 1usize..128) {
        test_unary(size, |a| a.exp(), |a| a.exp().unwrap());
    }

    #[test]
    fn test_log(size in 1usize..128) {
        test_unary(size, |a| a.log(), |a| a.log().unwrap());
    }

    #[test]
    fn test_sin(size in 1usize..128) {
        test_unary(size, |a| a.sin(), |a| a.sin().unwrap());
    }

    #[test]
    fn test_cos(size in 1usize..128) {
        test_unary(size, |a| a.cos(), |a| a.cos().unwrap());
    }

    #[test]
    fn test_activations(size in 1usize..128) {
        test_unary(size, |a| a.relu(), |a| a.relu().unwrap());
        // Exact GELU vs candle's exact erf GELU; tanh approximation vs candle's tanh GELU.
        test_unary(size, |a| a.gelu(), |a| a.gelu_erf().unwrap());
        test_unary(
            size,
            |a| a.gelu_fast_tanh_approximation(),
            |a| a.gelu().unwrap(),
        );
        test_unary(size, |a| a.swish(), |a| a.silu().unwrap());
        test_unary(size, |a| a.tanh(), |a| a.tanh().unwrap());
    }

    #[test]
    fn test_recip(size in 1usize..128) {
        test_unary(size, |a| a.reciprocal(), |a| a.recip().unwrap());
    }

    #[test]
    fn test_sqrt(size in 1usize..128) {
        test_unary(size, |a| a.sqrt(), |a| a.sqrt().unwrap());
    }

    #[test]
    fn test_square(size in 1usize..128) {
        test_unary(size, |a| a.square(), |a| a.powf(2.0).unwrap());
    }

    #[test]
    fn test_softmax(size in 1usize..128, rows in 1usize..16, cols in 1usize..16) {
        test_unary(size, |a| a.softmax(0), |a| softmax(&a, 0).unwrap());
        test_unary((rows, cols), |a| a.softmax(1), |a| softmax(&a, 1).unwrap());
    }

    #[test]
    fn test_layer_norm(size in 2usize..128) {
        test_unary(
            size,
            |a| a.layer_norm(0, 1e-5),
            |a| {
                let meaned = (a.clone() - a.mean(0).unwrap().broadcast_as(size)).unwrap();
                meaned
                    .powf(2.0)
                    .unwrap()
                    .mean(0)
                    .unwrap()
                    .add(&Tensor::new(1e-5_f32, a.device()).unwrap())
                    .unwrap()
                    .sqrt()
                    .unwrap()
                    .recip()
                    .unwrap()
                    .broadcast_as(size)
                    .unwrap()
                    .mul(&meaned)
                    .unwrap()
            },
        );
    }

    #[test]
    fn test_cumulative(rows in 1usize..16, cols in 1usize..16) {
        test_unary(rows, |a| a.cumsum(0), |a| a.cumsum(0).unwrap());
        test_unary((rows, cols), |a| a.cumsum(1), |a| a.cumsum(1).unwrap());
        test_unary((rows, cols), |a| a.cumsum(0), |a| a.cumsum(0).unwrap());
        test_unary(
            (rows, cols),
            |a| a.cumsum((0, 1)),
            |a| a.cumsum(0).unwrap().cumsum(1).unwrap(),
        );
        test_unary(
            (rows, cols),
            |a| a.cumsum((1, 0)),
            |a| a.cumsum(1).unwrap().cumsum(0).unwrap(),
        );
        test_unary((rows, cols), |a| a.cummax(1), cummax_ref_2d);
        test_unary((rows, cols), |a| a.cumprod(1), cumprod_ref_2d);
    }

    #[test]
    fn test_argmax(rows in 1usize..16, cols in 1usize..16) {
        test_unary((rows, cols), |a| a.argmax(0).cast(DType::F32), |a| a.argmax(0).unwrap().to_dtype(candle_core::DType::F32).unwrap());
        test_unary((rows, cols), |a| a.argmax(1).cast(DType::F32), |a| a.argmax(1).unwrap().to_dtype(candle_core::DType::F32).unwrap());
    }

    #[test]
    fn test_argmin(rows in 1usize..16, cols in 1usize..16) {
        test_unary((rows, cols), |a| a.argmin(0).cast(DType::F32), |a| a.argmin(0).unwrap().to_dtype(candle_core::DType::F32).unwrap());
        test_unary((rows, cols), |a| a.argmin(1).cast(DType::F32), |a| a.argmin(1).unwrap().to_dtype(candle_core::DType::F32).unwrap());
    }

    #[test]
    fn test_var(rows in 2usize..16, cols in 2usize..16) {
        test_unary((rows, cols), |a| a.var(1), |a| a.var(1).unwrap());
        test_unary((rows, cols), |a| a.var(0), |a| a.var(0).unwrap());
    }

    #[test]
    fn test_std(rows in 2usize..16, cols in 2usize..16) {
        test_unary((rows, cols), |a| a.std(1), |a| a.var(1).unwrap().sqrt().unwrap());
    }

    #[test]
    fn test_topk(rows in 1usize..12, cols in 1usize..12, k in 1usize..12) {
        prop_assume!(k <= cols);
        pub fn topk_sorted_indices(x: &[f32], k: usize) -> Vec<usize> {
            if k == 0 {
                return Vec::new();
            }

            let mut heap: BinaryHeap<std::cmp::Reverse<(NotNan<f32>, usize)>> =
                BinaryHeap::with_capacity(k);

            for (i, &v) in x.iter().enumerate() {
                let v = NotNan::new(v).expect("NaN encountered in topk");
                if heap.len() < k {
                    heap.push(std::cmp::Reverse((v, i)));
                } else if let Some(&std::cmp::Reverse((min_v, _))) = heap.peek()
                    && v > min_v {
                        heap.pop();
                        heap.push(std::cmp::Reverse((v, i)));
                    }
            }

            let mut out: Vec<(NotNan<f32>, usize)> =
                heap.into_iter().map(|std::cmp::Reverse(t)| t).collect();

            out.sort_unstable_by_key(|b| std::cmp::Reverse(b.0));
            out.into_iter().map(|(_, i)| i).collect()
        }
        test_unary(
            (rows, cols),
            |a| a.topk_indexes(k, 1).cast(DType::F32) * 1.0,
            |a| {
                let data = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                let topk = data
                    .chunks_exact(cols)
                    .flat_map(|c| topk_sorted_indices(c, k))
                    .map(|i| i as f32)
                    .collect_vec();
                Tensor::new(topk, a.device()).unwrap()
            },
        );
    }
}
