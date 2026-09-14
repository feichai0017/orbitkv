use crate::frontend::binary::tests::test_binary;
use crate::prelude::{DType, Graph};
use proptest::prelude::*;

#[test]
fn fp8_matmul_promotes_products_and_accumulation_to_f32() {
    let mut cx = Graph::new();
    let lhs = cx.tensor((2, 4)).as_dtype(DType::F8E4M3);
    let rhs = cx.tensor((4, 3)).as_dtype(DType::F8E4M3);

    let out = lhs.matmul(rhs);

    assert_eq!(out.dtype, DType::F32);
    let promoted_casts = cx
        .graph
        .node_indices()
        .filter(|&node| {
            cx.try_get_op::<crate::hlir::Cast>(node)
                .is_some_and(|cast| cast.1 == DType::F32)
        })
        .count();
    assert_eq!(
        promoted_casts, 2,
        "both FP8 operands must have explicit F32 semantic fallbacks"
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]
    #[test]
    fn test_matrix_vector(m in 1usize..6, k in 1usize..6, n in 1usize..6) {
        test_binary(
            (m, k),
            (k, n),
            |a, b| a.matmul(b),
            |a, b| a.matmul(&b).unwrap(),
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]
    #[test]
    fn test_matmul(m in 1usize..6, k in 1usize..6, n in 1usize..6) {
        test_binary(
            (m, k),
            (k, n),
            |a, b| a.matmul(b),
            |a, b| a.matmul(&b).unwrap(),
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]
    #[test]
    fn test_batch_matmul(batch in 1usize..4, m in 1usize..6, k in 1usize..6, n in 1usize..6) {
        test_binary(
            (batch, m, k),
            (k, n),
            |a, b| a.matmul(b),
            |a, b| {
                a.reshape((batch * m, k))
                    .unwrap()
                    .matmul(&b)
                    .unwrap()
                    .reshape((batch, m, n))
                    .unwrap()
            },
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]
    #[test]
    fn test_batch_batch_matmul(batch in 1usize..4, m in 1usize..6, k in 1usize..6, n in 1usize..6) {
        test_binary(
            (batch, m, k),
            (batch, m, k),
            |a, b| a.matmul(b.permute((0, 2, 1))),
            |a, b| a.matmul(&b.permute((0, 2, 1)).unwrap()).unwrap(),
        );
        test_binary(
            (batch, m, k),
            (batch, k, n),
            |a, b| a.matmul(b),
            |a, b| a.matmul(&b).unwrap(),
        );
    }
}
