use crate::{
    frontend::{binary::tests::test_binary, unary::tests::test_unary},
    prelude::*,
    tests::assert_exact,
};
use candle_core::{IndexOp, Tensor};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]
    #[test]
    fn test_pad_1d(len in 1usize..64, left in 0usize..6, right in 0usize..6) {
        test_unary(
            len,
            |a| a.pad((left, right), 0.),
            |a| a.pad_with_zeros(0, left, right).unwrap(),
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]
    #[test]
    fn test_pad_2d(rows in 1usize..32, cols in 1usize..32, top in 0usize..6, bottom in 0usize..6, left in 0usize..6, right in 0usize..6) {
        test_unary(
            (rows, cols),
            |a| a.pad(((top, bottom), (left, right)), 0.),
            |a| {
                a.pad_with_zeros(0, top, bottom)
                    .unwrap()
                    .pad_with_zeros(1, left, right)
                    .unwrap()
            },
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]
    #[test]
    fn test_slice_pad(
        rows in 3usize..32,
        cols in 3usize..32,
        start_row in 0usize..32,
        end_row in 1usize..32,
        start_col in 0usize..32,
        end_col in 1usize..32,
        pad_top in 0usize..6,
        pad_bottom in 0usize..6,
        pad_left in 0usize..6,
        pad_right in 0usize..6,
    ) {
        prop_assume!(start_row < end_row && end_row <= rows);
        prop_assume!(start_col < end_col && end_col <= cols);
        test_unary(
            (rows, cols),
            |a| a.slice((start_row..end_row, start_col..end_col)).pad(((pad_top, pad_bottom), (pad_left, pad_right)), 0.),
            |a| {
                a.i((start_row..end_row, start_col..end_col))
                    .unwrap()
                    .pad_with_zeros(0, pad_top, pad_bottom)
                    .unwrap()
                    .pad_with_zeros(1, pad_left, pad_right)
                    .unwrap()
            },
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]
    #[test]
    fn test_transpose(rows in 1usize..32, cols in 1usize..32) {
        test_unary(
            (rows, cols),
            |a| a.transpose(0, 1) * 1.0,
            |a| a.transpose(0, 1).unwrap(),
        );
    }
}

#[test]
fn test_unfold() {
    // Need all this code because candle doesnt do unfold
    #[allow(clippy::too_many_arguments)]
    pub fn unfold_nd_f32(
        x: &[f32],
        shape: &[usize],
        strides: &[usize],
        kernel: &[usize],
        step: &[usize],
        dilation: &[usize],
        pad_before: &[usize],
        pad_after: &[usize],
    ) -> Vec<f32> {
        let n = shape.len();
        assert!(n > 0);
        assert_eq!(strides.len(), n);
        assert_eq!(kernel.len(), n);
        assert_eq!(step.len(), n);
        assert_eq!(dilation.len(), n);
        assert_eq!(pad_before.len(), n);
        assert_eq!(pad_after.len(), n);

        for d in 0..n {
            assert!(kernel[d] > 0);
            assert!(step[d] > 0);
            assert!(dilation[d] > 0);
            assert!(shape[d] > 0);
        }

        // Effective kernel size per dim: (K-1)*d + 1
        let eff_kernel: Vec<usize> = (0..n).map(|d| (kernel[d] - 1) * dilation[d] + 1).collect();

        // Output spatial shape (number of windows) per dim
        let mut out_shape = vec![0usize; n];
        for d in 0..n {
            let padded = shape[d] + pad_before[d] + pad_after[d];
            if padded < eff_kernel[d] {
                return Vec::new();
            }
            out_shape[d] = (padded - eff_kernel[d]) / step[d] + 1;
        }

        let windows = prod(&out_shape);
        let window_elems = prod(kernel);
        let mut out = vec![0.0f32; windows * window_elems];

        // Precompute helpers
        let k_mul = row_major_multipliers(kernel);

        // Current output window position (row-major)
        let mut out_pos = vec![0usize; n];

        for w in 0..windows {
            if w > 0 {
                incr_row_major(&mut out_pos, &out_shape);
            }

            // Window start in padded coordinates
            let start_padded: Vec<usize> = (0..n).map(|d| out_pos[d] * step[d]).collect();

            let base_out = w * window_elems;

            // Iterate kernel elements (flattened)
            for ke in 0..window_elems {
                let k_idx = unravel_row_major(ke, kernel, &k_mul);

                let mut flat: isize = 0;
                let mut in_bounds = true;

                for d in 0..n {
                    let p = start_padded[d] + k_idx[d] * dilation[d];
                    let logical = p as isize - pad_before[d] as isize;

                    if logical < 0 || logical >= shape[d] as isize {
                        in_bounds = false;
                        break;
                    }
                    flat += logical * strides[d] as isize;
                }

                let out_idx = base_out + ke;
                out[out_idx] = if in_bounds { x[flat as usize] } else { 0.0 };
            }
        }

        out
    }

    // -------- helpers --------

    fn prod(xs: &[usize]) -> usize {
        xs.iter().copied().product()
    }

    fn row_major_multipliers(shape: &[usize]) -> Vec<usize> {
        let n = shape.len();
        let mut mul = vec![1usize; n];
        let mut acc = 1usize;
        for d in (0..n).rev() {
            mul[d] = acc;
            acc *= shape[d];
        }
        mul
    }

    fn unravel_row_major(mut idx: usize, shape: &[usize], mul: &[usize]) -> Vec<usize> {
        let n = shape.len();
        let mut coords = vec![0usize; n];
        for d in 0..n {
            coords[d] = idx / mul[d];
            idx %= mul[d];
        }
        coords
    }

    fn incr_row_major(pos: &mut [usize], shape: &[usize]) {
        for d in (0..pos.len()).rev() {
            pos[d] += 1;
            if pos[d] < shape[d] {
                return;
            }
            pos[d] = 0;
        }
    }

    test_unary(
        5,
        |a| a.unfold(3, 1, 1),
        |a| {
            Tensor::new(
                unfold_nd_f32(
                    &a.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                    a.dims(),
                    a.stride(),
                    &[3],
                    &[1],
                    &[1],
                    &[0],
                    &[0],
                ),
                a.device(),
            )
            .unwrap()
        },
    );
    test_unary(
        (8, 10),
        |a| a.pad(((0, 2), (4, 4)), 0.).unfold((2, 3), (1, 2), (2, 1)),
        |a| {
            Tensor::new(
                unfold_nd_f32(
                    &a.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                    a.dims(),
                    a.stride(),
                    &[2, 3],
                    &[1, 2],
                    &[2, 1],
                    &[0, 4],
                    &[2, 3],
                ),
                a.device(),
            )
            .unwrap()
        },
    );
}

#[test]
fn test_unfold_floor_div_shape_for_odd_window_numerator() {
    let mut cx = Graph::new();
    let inp = cx.tensor((80, 3000));
    let out = inp.pad(((0, 0), (1, 1)), 0.).unfold((1, 3), (1, 2), (1, 1));
    assert_eq!(out.dims(), &[80, 1500, 1, 3]);
}

#[test]
fn test_unsqueeze() {
    let mut cx = Graph::new();
    let inp = cx.tensor((2, 2, 3));
    let out1 = inp.unsqueeze(1);
    let out2 = inp.unsqueeze(3);
    assert_eq!(out1.dims(), &[2, 1, 2, 3]);
    assert_eq!(out2.dims(), &[2, 2, 3, 1]);
    test_unary(
        (1, 3),
        |a| a.squeeze(0).expand_dim(0, 2) * 1.,
        |a| a.broadcast_as((2, 3)).unwrap(),
    );
    test_unary((2, 1, 3), |a| a.squeeze(1), |a| a.reshape((2, 3)).unwrap());
}

#[test]
fn test_concat() {
    test_binary(
        17,
        32,
        |a, b| a.concat_along(b, 0),
        |a, b| Tensor::cat(&[a, b], 0).unwrap(),
    );
    test_binary(
        (10, 4),
        (10, 6),
        |a, b| a.concat_along(b, 1),
        |a, b| Tensor::cat(&[a, b], 1).unwrap(),
    );
    test_binary(
        (4, 10),
        (6, 10),
        |a, b| a.concat_along(b, 0),
        |a, b| Tensor::cat(&[a, b], 0).unwrap(),
    );
    test_unary(
        (4, 10),
        |a| a.concat_along(a, 0),
        |a| Tensor::cat(&[a.clone(), a], 0).unwrap(),
    );
}

#[test]
fn empty_offset_slice_is_a_metadata_only_view() {
    let mut cx = Graph::new();
    let input = cx.tensor((1, 4, 64));
    let empty = input.slice_along(64.., 2);

    assert_eq!(empty.dims(), &[1, 4, 0]);
    assert_eq!(empty.id, input.id, "an empty slice must not create an Iota");
}

#[test]
fn test_gather_and_scatter_inverse() {
    let mut cx = Graph::new();
    let data = cx.tensor((2, 3));
    let indexes = cx.tensor(4).as_dtype(DType::Int);
    let gathered = data.gather(indexes).output();
    // Inverse permutation via scatter: scatter arange at perm positions into zeros
    let perm = cx.tensor(6).as_dtype(DType::Int);
    let values = cx.arange(6);
    let zeros = cx.iota(Expression::from(0usize), 6);
    let inv = values.scatter(perm, zeros).cast(DType::F32).output();
    cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
    let mut rt = cx.search(
        ReferenceRuntime::default(),
        CompileOptions::default().search_graph_limit(1),
    );
    rt.set_data(data.id, vec![0., 1., 2., 3., 4., 5.]);
    rt.set_data(indexes.id, vec![5, 0, 3, 2]);
    rt.set_data(perm.id, vec![3, 2, 4, 1, 5, 0]);
    rt.execute(&cx.dyn_map);
    assert_eq!(*rt.get_f32(gathered.id), vec![5., 0., 3., 2.]);
    assert_eq!(*rt.get_f32(inv.id), vec![5., 3., 1., 0., 2., 4.]);
}

#[test]
fn test_scatter_basic() {
    let mut cx = Graph::new();
    let src = cx.tensor(3);
    let indexes = cx.tensor(3).as_dtype(DType::Int);
    let dest = cx.tensor(5);
    let result = src.scatter(indexes, dest).output();
    cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
    let mut rt = cx.search(
        ReferenceRuntime::default(),
        CompileOptions::default().search_graph_limit(1),
    );
    rt.set_data(src.id, vec![10., 20., 30.]);
    rt.set_data(indexes.id, vec![1, 3, 4]);
    rt.set_data(dest.id, vec![0., 0., 0., 0., 0.]);
    rt.execute(&cx.dyn_map);
    assert_eq!(*rt.get_f32(result.id), vec![0., 10., 0., 20., 30.]);
}

#[test]
fn test_scatter_into_nonzero_dest() {
    let mut cx = Graph::new();
    let src = cx.tensor(1);
    let indexes = cx.tensor(1).as_dtype(DType::Int);
    let dest = cx.tensor(5);
    let result = src.scatter(indexes, dest).output();
    cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
    let mut rt = cx.search(
        ReferenceRuntime::default(),
        CompileOptions::default().search_graph_limit(1),
    );
    rt.set_data(src.id, vec![99.]);
    rt.set_data(indexes.id, vec![2]);
    rt.set_data(dest.id, vec![1., 2., 3., 4., 5.]);
    rt.execute(&cx.dyn_map);
    assert_eq!(*rt.get_f32(result.id), vec![1., 2., 99., 4., 5.]);
}

#[test]
fn test_scatter_all_positions() {
    let mut cx = Graph::new();
    let src = cx.tensor(4);
    let indexes = cx.tensor(4).as_dtype(DType::Int);
    let dest = cx.tensor(4);
    let result = src.scatter(indexes, dest).output();
    cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
    let mut rt = cx.search(
        ReferenceRuntime::default(),
        CompileOptions::default().search_graph_limit(1),
    );
    rt.set_data(src.id, vec![40., 30., 20., 10.]);
    rt.set_data(indexes.id, vec![3, 2, 1, 0]);
    rt.set_data(dest.id, vec![1., 2., 3., 4.]);
    rt.execute(&cx.dyn_map);
    assert_eq!(*rt.get_f32(result.id), vec![10., 20., 30., 40.]);
}

#[test]
fn test_repeat_is_view_only() {
    let mut cx = Graph::new();
    let a = cx.tensor((2, 3));
    let repeated = a.repeat((2, 2));

    assert_eq!(repeated.id, a.id);
    assert_eq!(
        repeated.dims(),
        vec![Expression::from(4usize), Expression::from(6usize)]
    );
}

#[test]
fn test_repeat_runtime_values() {
    let mut cx = Graph::new();
    let a = cx.tensor((2, 3));
    let repeated = (a.repeat((2, 2)) * 1.0).output();

    cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
    let mut rt = cx.search(
        ReferenceRuntime::default(),
        CompileOptions::default().search_graph_limit(1),
    );
    rt.set_data(a.id, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    rt.execute(&cx.dyn_map);

    assert_exact(
        rt.get_f32(repeated.id),
        &[
            1.0, 2.0, 3.0, 1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0, 4.0, 5.0, 6.0, //
            1.0, 2.0, 3.0, 1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0, 4.0, 5.0, 6.0,
        ],
    );
}

//     // #[test]
//     // fn test_cumsum() {
//     //     let mut cx = Graph::new();
//     //     let a = cx.constant(1.).expand_dim(0, 3);
//     //     let b = a.cumsum_last_dim().retrieve();
//     //     let c = a
//     //         .expand_dim(1, 3)
//     //         .permute((1, 0))
//     //         .cumsum_last_dim()
//     //         .permute((1, 0))
//     //         .retrieve();
//     //     cx.execute();

//     //     assert_exact(&b.data(), &[1., 2., 3.]);
//     //     assert_exact(&c.data(), &[1., 1., 1., 2., 2., 2., 3., 3., 3.]);
//     // }

//     // #[test]
//     // fn test_pool_1d() {
//     //     let mut cx = Graph::new();

//     //     let inp1 = cx.tensor(5).set([1., 2., 3., 4., 5.]);
//     //     let inp2 = cx
//     //         .tensor((2, 5))
//     //         .set([[15., 14., 13., 12., 11.], [1., 2., 3., 4., 5.]]);
//     //     // Stride 1
//     //     let out1 = inp1.pool_last_dim(3, 1, 1).retrieve();
//     //     // Stride 2
//     //     let out2 = inp1.pool_last_dim(3, 2, 1).retrieve();
//     //     // Stride 3
//     //     let out3 = inp1.pool_last_dim(3, 3, 1).retrieve();
//     //     // Dilation 2
//     //     let out4 = inp1.pool_last_dim(3, 1, 2).retrieve();
//     //     // Dilation 2 Padding 1
//     //     let out5 = inp1.pad(((1, 1),)).pool_last_dim(3, 1, 2).retrieve();
//     //     // Stride 1 Batch 2
//     //     let out6 = inp2.pool_last_dim(3, 1, 1).retrieve();
//     //     // Stride 3
//     //     let out7 = inp2.pool_last_dim(3, 3, 1).retrieve();
//     //     // Dilation 2
//     //     let out8 = inp2.pool_last_dim(3, 1, 2).retrieve();
//     //     // Dilation 2 Padding 1
//     //     let out9 = inp2.pad(((0, 0), (1, 1))).pool_last_dim(3, 1, 2).retrieve();

//     //     cx.execute();

//     //     assert_exact(&out1.data(), &[1., 2., 3., 2., 3., 4., 3., 4., 5.]);
//     //     assert_exact(&out2.data(), &[1., 2., 3., 3., 4., 5.]);
//     //     assert_exact(&out3.data(), &[1., 2., 3.]);
//     //     assert_exact(&out4.data(), &[1., 3., 5.]);
//     //     assert_exact(&out5.data(), &[0., 2., 4., 1., 3., 5., 2., 4., 0.]);
//     //     assert_exact(
//     //         &out6.data(),
//     //         &[
//     //             15., 14., 13., 14., 13., 12., 13., 12., 11., 1., 2., 3., 2., 3., 4., 3., 4., 5.,
//     //         ],
//     //     );
//     //     assert_exact(&out7.data(), &[15., 14., 13., 1., 2., 3.]);
//     //     assert_exact(&out8.data(), &[15., 13., 11., 1., 3., 5.]);
//     //     assert_exact(
//     //         &out9.data(),
//     //         &[
//     //             0., 14., 12., 15., 13., 11., 14., 12., 0., 0., 2., 4., 1., 3., 5., 2., 4., 0.,
//     //         ],
//     //     );
//     // }

//     // #[test]
//     // fn test_pool_1d_dims() {
//     //     let mut cx = Graph::new();

//     //     let inp1 = cx.tensor((4, 4)).set(vec![
//     //         1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
//     //     ]);
//     //     // Stride 1
//     //     let out1 = inp1.pool_last_dim(3, 1, 1).retrieve();

//     //     cx.execute();

//     //     assert_exact(
//     //         &out1.data(),
//     //         &[
//     //             1., 2., 3., 2., 3., 4., 5., 6., 7., 6., 7., 8., 9., 10., 11., 10., 11., 12., 13.,
//     //             14., 15., 14., 15., 16.,
//     //         ],
//     //     );
//     // }

//     // #[test]
//     // fn test_pool_2d() {
//     //     let mut cx = Graph::new();

//     //     let inp1 = cx.tensor((4, 4)).set(vec![
//     //         1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
//     //     ]);
//     //     // 3x3 kernel
//     //     let out1 = inp1
//     //         // Pool first dim first by moving it to end
//     //         .permute((1, 0))
//     //         .pool_last_dim(3, 1, 1)
//     //         // Now move other dim to end
//     //         .permute((1, 2, 0))
//     //         .pool_last_dim(3, 1, 1)
//     //         // Now swap middle two dims
//     //         .permute((0, 2, 1, 3))
//     //         // Now merge both pooled dimensions
//     //         .reshape((4, 3, 3))
//     //         .retrieve();

//     //     cx.execute();

//     //     assert_exact(
//     //         &out1.data(),
//     //         &[
//     //             1.00, 2.00, 3.00, 5.00, 6.00, 7.00, 9.00, 10.00, 11.00, 2.00, 3.00, 4.00, 6.00,
//     //             7.00, 8.00, 10.00, 11.00, 12.00, 5.00, 6.00, 7.00, 9.00, 10.00, 11.00, 13.00,
//     //             14.00, 15.00, 6.00, 7.00, 8.00, 10.00, 11.00, 12.00, 14.00, 15.00, 16.00,
//     //         ],
//     //     );
//     // }

//     // #[test]
//     // fn test_pool_1d_dilation() {
//     //     let mut cx = Graph::new();

//     //     let inp1 = cx.tensor(5).set(vec![1., 2., 3., 4., 5.]);
//     //     // Stride 1
//     //     let out1 = inp1.pool_last_dim(2, 1, 2).retrieve();
//     //     // Stride 2
//     //     let out2 = inp1.pool_last_dim(2, 2, 2).retrieve();
//     //     // Stride 3
//     //     let out3 = inp1.pool_last_dim(2, 3, 2).retrieve();

//     //     cx.execute();

//     //     assert_exact(&out1.data(), &[1., 3., 2., 4., 3., 5.]);
//     //     assert_exact(&out2.data(), &[1., 3., 3., 5.]);
//     //     assert_exact(&out3.data(), &[1., 3.]);
//     // }

//     // #[test]
//     // fn test_rotate_half() {
//     //     let mut cx = Graph::new();
//     //     let a = cx.tensor((3, 2));
//     //     a.set(vec![1.4325, 2.492428, 3.127365, 33.2834, 4.18734, 23.854]);
//     //     let x1 = a.slice((.., ..1)).contiguous();
//     //     let x2 = a.slice((.., 1..)).contiguous();
//     //     let c = (-x2).concat_along(x1, 1);
//     //     c.retrieve();
//     //     cx.execute();

//     //     let d_dev = Cpu::default();
//     //     let d_a = d_dev.tensor_from_vec(
//     //         vec![1.4325, 2.492428, 3.127365, 33.2834, 4.18734, 23.854],
//     //         (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<2>),
//     //     );
//     //     let d_x1 = d_a.clone().slice((.., ..1));
//     //     let d_x2 = d_a.slice((.., 1..));
//     //     let d_c = (-d_x2, d_x1)
//     //         .concat_along(dfdx::shapes::Axis::<1>)
//     //         .realize::<Rank2<3, 2>>();

//     //     assert_close(&c.data(), &d_c.as_vec());
//     // }
