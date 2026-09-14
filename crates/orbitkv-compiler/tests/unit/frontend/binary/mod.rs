use crate::{
    prelude::*,
    tests::{assert_close, random_vec},
};
use candle_core::{DType, Device, Tensor};
use itertools::Itertools;
use proptest::prelude::*;

pub fn identity(v: Vec<f32>) -> Vec<f32> {
    v
}

pub fn shift_from_zero(v: Vec<f32>) -> Vec<f32> {
    v.into_iter()
        .map(|x| if x >= 0.0 { x + 1.0 } else { x - 1.0 })
        .collect()
}

pub fn test_binary(
    a_shape: impl ToShape,
    b_shape: impl ToShape,
    func: impl Fn(GraphTensor, GraphTensor) -> GraphTensor,
    ref_func: impl Fn(Tensor, Tensor) -> Tensor,
) {
    test_binary_transforms(a_shape, b_shape, func, ref_func, identity, identity);
}

pub fn test_binary_transforms(
    a_shape: impl ToShape,
    b_shape: impl ToShape,
    func: impl Fn(GraphTensor, GraphTensor) -> GraphTensor,
    ref_func: impl Fn(Tensor, Tensor) -> Tensor,
    lhs_transform: impl Fn(Vec<f32>) -> Vec<f32>,
    rhs_transform: impl Fn(Vec<f32>) -> Vec<f32>,
) {
    let a_shape = a_shape
        .to_shape()
        .into_iter()
        .map(|e| e.to_usize().unwrap())
        .collect_vec();
    let b_shape = b_shape
        .to_shape()
        .into_iter()
        .map(|e| e.to_usize().unwrap())
        .collect_vec();
    let mut cx = Graph::new();
    let a = cx.tensor(a_shape.clone());
    let b = cx.tensor(b_shape.clone());
    let c = func(a, b).output();

    cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
    let mut rt = cx.search(
        ReferenceRuntime::default(),
        CompileOptions::default().search_graph_limit(1),
    );

    let lhs_values = lhs_transform(random_vec(a_shape.iter().copied().product()));
    let rhs_values = rhs_transform(random_vec(b_shape.iter().copied().product()));
    rt.set_data(a.id, lhs_values.clone());
    rt.set_data(b.id, rhs_values.clone());
    rt.execute(&cx.dyn_map);

    // Reference
    let device = Device::Cpu;
    let ref_a = Tensor::from_vec(lhs_values, a_shape, &device).unwrap();
    let ref_b = Tensor::from_vec(rhs_values, b_shape, &device).unwrap();
    let ref_c = ref_func(ref_a, ref_b).flatten_all().unwrap();

    assert_close(rt.get_f32(c.id), &ref_c.to_vec1::<f32>().unwrap())
}

#[test]
#[should_panic(expected = "Dims must match to add tensors.")]
fn test_add_rejects_implicit_broadcast() {
    let mut cx = Graph::new();
    let a = cx.tensor((2, 3));
    let b = cx.tensor((1, 3));
    let _ = a + b;
}

#[test]
#[should_panic(expected = "Dims must match to multiply tensors.")]
fn test_mul_rejects_implicit_broadcast() {
    let mut cx = Graph::new();
    let a = cx.tensor((2, 3));
    let b = cx.tensor((1, 3));
    let _ = a * b;
}

#[test]
#[should_panic(expected = "Dims must match to mod tensors.")]
fn test_mod_rejects_implicit_broadcast() {
    let mut cx = Graph::new();
    let a = cx.tensor((2, 3));
    let b = cx.tensor((1, 3));
    let _ = a % b;
}

#[test]
#[should_panic(expected = "Dims must match to lt tensors.")]
fn test_lt_rejects_implicit_broadcast() {
    let mut cx = Graph::new();
    let a = cx.tensor((2, 3));
    let b = cx.tensor((1, 3));
    let _ = a.lt(b);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]
    #[test]
    fn test_add(x in 1..100, y in 1..5) {
        test_binary(x, x, |a, b| a + b, |a, b| (&a + &b).unwrap());
        test_binary((y, x), (y, x), |a, b| a + b, |a, b| (&a + &b).unwrap());
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]
    #[test]
    fn test_sub(x in 1..100, y in 1..5) {
        test_binary(x, x, |a, b| a - b, |a, b| (&a - &b).unwrap());
        test_binary((y, x), (y, x), |a, b| a - b, |a, b| (&a - &b).unwrap());
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]
    #[test]
    fn test_mul(x in 1..100, y in 1..5) {
        test_binary(x, x, |a, b| a * b, |a, b| (&a * &b).unwrap());
        test_binary(
            (2, y, x),
            (2, y, x),
            |a, b| a * b,
            |a, b| (&a * &b).unwrap(),
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]
    #[test]
    fn test_div(x in 1..100) {
        test_binary_transforms(
            x,
            x,
            |a, b| a / b,
            |a, b| (&a / &b).unwrap(),
            identity,
            shift_from_zero,
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]
    #[test]
    fn test_maximum(x in 1..100) {
        test_binary(x, x, |a, b| a.maximum(b), |a, b| a.maximum(&b).unwrap());
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]
    #[test]
    fn test_minimum(x in 1..100) {
        test_binary(x, x, |a, b| a.minimum(b), |a, b| a.minimum(&b).unwrap());
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]
    #[test]
    fn test_mod(size in 1usize..64) {
        test_binary_transforms(
            size,
            size,
            |a, b| a % b,
            |a, b| {
                let lhs = a.to_vec1::<f32>().unwrap();
                let rhs = b.to_vec1::<f32>().unwrap();
                let remainder: Vec<f32> = lhs.iter().zip(rhs.iter()).map(|(x, y)| x % y).collect();
                Tensor::from_vec(remainder, size, &Device::Cpu).unwrap()
            },
            identity,
            shift_from_zero,
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]
    #[test]
    fn test_mod_scalar_broadcast(size in 1usize..64) {
        // rank-0 RHS expanded against rank-N LHS, mirroring `x % torch.tensor(c)`.
        test_binary_transforms(
            size,
            (),
            |a, b| a % b.expand_rhs(a.shape),
            |a, b| {
                let lhs = a.to_vec1::<f32>().unwrap();
                let rhs_scalar = b.to_scalar::<f32>().unwrap();
                let remainder: Vec<f32> = lhs.iter().map(|x| x % rhs_scalar).collect();
                Tensor::from_vec(remainder, size, &Device::Cpu).unwrap()
            },
            identity,
            shift_from_zero,
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]
    #[test]
    fn test_lt(size in 1usize..64) {
        test_binary(
            size,
            size,
            |a, b| a.lt(b).cast(crate::dtype::DType::F32),
            |a, b| a.lt(&b).unwrap().to_dtype(DType::F32).unwrap(),
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]
    #[test]
    fn test_lt_scalar_broadcast(size in 1usize..64) {
        // rank-0 RHS expanded against rank-N LHS for `lt`.
        test_binary(
            size,
            (),
            |a, b| a.lt(b.expand_rhs(a.shape)).cast(crate::dtype::DType::F32),
            |a, b| {
                let scalar = b.to_scalar::<f32>().unwrap();
                let lhs = a.to_vec1::<f32>().unwrap();
                let result: Vec<f32> = lhs
                    .iter()
                    .map(|x| if *x < scalar { 1.0f32 } else { 0.0f32 })
                    .collect();
                Tensor::from_vec(result, size, &Device::Cpu).unwrap()
            },
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]
    #[test]
    fn test_gt(size in 1usize..64) {
        test_binary(
            size,
            size,
            |a, b| a.gt(b).cast(crate::dtype::DType::F32),
            |a, b| a.gt(&b).unwrap().to_dtype(DType::F32).unwrap(),
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]
    #[test]
    fn test_le(size in 1usize..64) {
        test_binary(
            size,
            size,
            |a, b| a.le(b).cast(crate::dtype::DType::F32),
            |a, b| a.le(&b).unwrap().to_dtype(DType::F32).unwrap(),
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]
    #[test]
    fn test_ge(size in 1usize..64) {
        test_binary(
            size,
            size,
            |a, b| a.ge(b).cast(crate::dtype::DType::F32),
            |a, b| a.ge(&b).unwrap().to_dtype(DType::F32).unwrap(),
        );
    }
}

#[test]
fn test_ne() {
    test_binary(
        27,
        27,
        |a, b| {
            let result = a.ne(b);
            assert_eq!(result.dtype, crate::dtype::DType::Bool);
            result.cast(crate::dtype::DType::F32)
        },
        |a, b| a.ne(&b).unwrap().to_dtype(DType::F32).unwrap(),
    );
}

#[test]
fn test_eq() {
    test_binary(
        27,
        27,
        |a, b| a.eq(b).cast(crate::dtype::DType::F32),
        |a, b| a.eq(&b).unwrap().to_dtype(DType::F32).unwrap(),
    );
}

#[test]
fn test_pow() {
    test_binary_transforms(
        27,
        27,
        |a, _| a.pow(2.5f32),
        |a, _| a.powf(2.5f64).unwrap(),
        shift_from_zero,
        identity,
    );
}

#[test]
fn test_clip() {
    test_binary_transforms(
        27,
        27,
        |a, _| a.clip(-0.25, 0.25),
        |a, _| a.clamp(-0.25, 0.25).unwrap(),
        identity,
        identity,
    );
}

#[test]
fn test_maximum_f32() {
    test_binary_transforms(
        27,
        27,
        |a, _| a.maximum_f32(0.1),
        |a, _| {
            a.maximum(&Tensor::new(vec![0.1f32; 27], &Device::Cpu).unwrap())
                .unwrap()
        },
        identity,
        identity,
    );
}

#[test]
fn test_minimum_f32() {
    test_binary_transforms(
        27,
        27,
        |a, _| a.minimum_f32(-0.1),
        |a, _| {
            a.minimum(&Tensor::new(vec![-0.1f32; 27], &Device::Cpu).unwrap())
                .unwrap()
        },
        identity,
        identity,
    );
}

#[test]
fn test_cond() {
    test_binary(
        27,
        27,
        |a, b| {
            // gt() returns Bool, cast to F32 for cond which expects F32
            let cond = a
                .gt(b.graph().constant_float(0.0).expand_rhs(a.shape))
                .cast(crate::dtype::DType::F32);
            a.cond(cond, b)
        },
        |a, b| {
            let refer = a.gt(&Tensor::zeros_like(&a).unwrap()).unwrap();
            refer.where_cond(&a, &b).unwrap()
        },
    );
}
