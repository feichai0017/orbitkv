use crate::prelude::*;
use proptest::prelude::*;
#[test]
fn test_contiguous_physical_span_equals_n_elements() {
    // A fully contiguous tensor addresses exactly n_elements offsets, so
    // physical_span must equal n_elements (modulo the symbolic max guard).
    let s = expr('s');
    let tracker = ShapeTracker::new([s, expr(2), expr(12), expr(16)]);
    assert!(tracker.is_contiguous());
    let span = tracker.physical_span();
    let n_elem = tracker.n_elements();
    // A contiguous view addresses exactly n_elements offsets. Evaluate at a
    // few concrete sizes: span and n_elements must agree (regression for the
    // Add fold bug that over-counted the constant as 533 instead of 383).
    for s in 1usize..=4 {
        let map: crate::prelude::DynMap = [(sym("s"), s)].into_iter().collect();
        assert_eq!(
            span.exec(&map),
            n_elem.exec(&map),
            "span and n_elements disagree at s={s}: span={span:?} n_elem={n_elem:?}"
        );
        assert_eq!(span.exec(&map), Some(384 * s));
    }
}

#[test]
fn test_idx_expr() {
    let mut tracker = ShapeTracker::new([expr(10), expr(5), expr(3)]);
    tracker.permute(&[2, 0, 1]);
    println!("Shape: [10, 5, 3]");
    println!("Strides: {:?}", tracker.strides);
    println!("Ind: {:?}", tracker.index_expression());
    println!("Val: {:?}", tracker.valid_expression());
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]
    #[test]
    fn test_permute_and_expand(a in 1usize..10, b in 1usize..10, c in 1usize..10, expand_a in 2usize..10) {
        let z = expr('z');
        // Build expected strides the same way new() does: z, z*c, z*c*b
        let zc = z * expr(c);
        let zcb = zc * expr(b);
        let mut tracker = ShapeTracker::new((a, b, c));
        assert!(tracker.is_contiguous());
        assert_eq!(
            tracker.strides.as_slice(),
            &[zcb, zc, z]
        );
        tracker.permute((1, 2, 0));
        assert_eq!(
            tracker.dims.as_slice(),
            &[
                expr(b),
                expr(c),
                expr(a)
            ]
        );
        assert_eq!(
            tracker.strides.as_slice(),
            &[zc, z, zcb]
        );
        tracker.expand_dim(1, 1);
        assert_eq!(
            tracker.dims.as_slice(),
            &[
                expr(b),
                expr(1),
                expr(c),
                expr(a)
            ]
        );
        assert_eq!(
            tracker.strides.as_slice(),
            &[zc, expr(0), z, zcb]
        );
        let removed = tracker.remove_dim(1);
        assert_eq!(removed, expr(1));
        assert_eq!(
            tracker.dims.as_slice(),
            &[
                expr(b),
                expr(c),
                expr(a)
            ]
        );
        let mut tracker = ShapeTracker::new((1, c));
        tracker.expand((expand_a, c));
        assert_eq!(
            tracker.dims.as_slice(),
            &[expr(expand_a), expr(c)]
        );
        assert_eq!(
            tracker.strides.as_slice(),
            &[expr(0), z]
        );
    }
}

#[test]
fn test_merge_dims() {
    let z = expr('z');
    let mut tracker = ShapeTracker::new((10, 5, 3));
    assert_eq!(tracker.dims.len(), 3);
    tracker.merge_dims(1, 2);
    // merged: dims [10, 15], strides [z*15, z]
    assert_eq!(tracker.dims.len(), 2);
    assert_eq!(tracker.dims[0], expr(10));
    assert_eq!(tracker.dims[1].simplify(), expr(15));
    assert_eq!(tracker.strides[1], z);
    // stride[0] should evaluate to z*15 (check numerically)
    let s0 = tracker.strides[0].simplify();
    for val in [0, 1, 5, 10] {
        assert_eq!(
            s0.substitute('z', val).to_usize(),
            Some(val * 15),
            "stride[0] failed for z={val}: got {s0}"
        );
    }
}

#[test]
fn test_merge_dims_non_adjacent() {
    // Shape [A, B, C] = [4, 3, 5], merge dims 0 and 2
    // This should permute to [A, C, B] then merge A and C
    let mut tracker = ShapeTracker::new((4, 3, 5));
    tracker.merge_dims(0, 2);
    // Result: dims [4*5, 3] = [20, 3]
    assert_eq!(tracker.dims.len(), 2);
    assert_eq!(tracker.dims[0].simplify(), expr(20));
    assert_eq!(tracker.dims[1], expr(3));
    // Verify index mapping numerically
    let idx = tracker.index_expression();
    for a in 0..4 {
        for c in 0..5 {
            for b in 0..3 {
                let merged_idx = (a * 5 + c) * 3 + b;
                let physical = a * 15 + b * 5 + c; // original [A,B,C] layout
                let result = idx
                    .substitute('z', merged_idx)
                    .simplify()
                    .to_usize()
                    .unwrap();
                assert_eq!(
                    result, physical,
                    "Failed for a={a}, b={b}, c={c}: merged_idx={merged_idx}"
                );
            }
        }
    }
}

#[test]
fn test_repeat_index_mapping() {
    let mut tracker = ShapeTracker::new((2, 3));
    tracker.repeat((2, 2));

    assert_eq!(tracker.dims.as_slice(), &[expr(4), expr(6)]);

    let idx = tracker.index_expression();
    for row in 0..4 {
        for col in 0..6 {
            let logical = row * 6 + col;
            let physical = (row % 2) * 3 + (col % 3);
            let result = idx.substitute('z', logical).to_usize().unwrap();
            assert_eq!(
                result, physical,
                "Failed for row={row}, col={col}: logical={logical}"
            );
        }
    }
}

#[test]
fn test_split_dims_preserves_merged_index_mapping() {
    let mut tracker = ShapeTracker::new((2, 3, 5));
    let original_idx = tracker.index_expression();

    tracker.permute((1, 0, 2));
    tracker.merge_dims(0, 1);
    tracker.split_dims(0, 2);

    assert_eq!(tracker.dims.as_slice(), &[expr(3), expr(2), expr(5)]);

    let split_idx = tracker.index_expression();
    for b in 0..3 {
        for a in 0..2 {
            for c in 0..5 {
                let split_logical = (b * 2 + a) * 5 + c;
                let original_logical = (a * 3 + b) * 5 + c;
                let split_physical = split_idx
                    .substitute('z', split_logical)
                    .simplify()
                    .to_usize()
                    .unwrap();
                let original_physical = original_idx
                    .substitute('z', original_logical)
                    .simplify()
                    .to_usize()
                    .unwrap();
                assert_eq!(
                    split_physical, original_physical,
                    "Failed for a={a}, b={b}, c={c}"
                );
            }
        }
    }
}

#[test]
#[should_panic(expected = "split_dims cannot represent stride")]
fn test_split_dims_rejects_non_separable_stride() {
    let mut tracker = ShapeTracker::new((6,));
    tracker.repeat((2,));
    tracker.split_dims(0, 4);
}

// #[test]
// fn test_symbolic_idx() {
//     let mut cx = Graph::new();
//     let seq = 2;
//     let head_dim = 4;
//     let a = cx.named_tensor("a", (seq, head_dim)).keep();
//     let _b = cx.tensor((seq, head_dim / 2, 1)).keep();
//     // Split input into evens and odds
//     let split = a.reshape((seq, head_dim / 2, 2));
//     let x0 = split.slice((.., .., ..1));
//     let _x = split.slice((.., .., 1..));

//     println!("x0: {:?}", x0.shape.index_expression());
// }
