use crate::frontend::unary::tests::test_unary;
use candle_core::{Device, Tensor};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]
    #[test]
    fn test_sum(rows in 1usize..8, cols in 1usize..8, depth in 1usize..6) {
        test_unary((rows, cols), |a| a.sum(1), |a| a.sum(1).unwrap());
        test_unary(
            (rows, cols, depth),
            |a| a.sum((0, 2)),
            |a| a.sum((0, 2)).unwrap(),
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]
    #[test]
    fn test_max(rows in 1usize..8, cols in 1usize..8) {
        test_unary((rows, cols), |a| a.max(1), |a| a.max(1).unwrap());
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]
    #[test]
    fn test_min(rows in 1usize..8, cols in 1usize..8) {
        test_unary((rows, cols), |a| a.min(1), |a| a.min(1).unwrap());
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]
    #[test]
    fn test_mean(rows in 1usize..8, cols in 1usize..8, depth in 1usize..6) {
        test_unary((rows, cols), |a| a.mean(1), |a| a.mean(1).unwrap());
        let denom = (rows * depth) as f32;
        test_unary(
            (rows, cols, depth),
            |a| a.mean((0, 2)),
            |a| {
                let denom = Tensor::from_vec(vec![denom; cols], cols, a.device()).unwrap();
                (a.sum(2).unwrap().sum(0).unwrap() / denom).unwrap()
            },
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]
    #[test]
    fn test_prod(rows in 1usize..8, cols in 1usize..8) {
        test_unary(
            (rows, cols),
            |a| a.prod(1),
            |a| {
                let v = a.to_vec2::<f32>().unwrap();
                let out: Vec<f32> = v.iter().map(|row| row.iter().product()).collect();
                Tensor::from_vec(out, v.len(), &Device::Cpu).unwrap()
            },
        );
    }
}
