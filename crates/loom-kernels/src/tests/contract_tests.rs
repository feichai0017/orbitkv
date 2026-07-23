use crate::*;

#[test]
fn contiguous_tensor_has_expected_strides() {
    let tensor = TensorSpec::contiguous(DType::Bf16, vec![2, 3, 5]).unwrap();
    assert_eq!(tensor.shape(), &[2, 3, 5]);
    assert_eq!(tensor.strides(), &[15, 5, 1]);
    assert_eq!(tensor.numel(), 30);
    assert_eq!(tensor.size_in_bytes(), 60);
}

#[test]
fn invalid_shapes_are_rejected() {
    assert_eq!(
        TensorSpec::contiguous(DType::F32, vec![]),
        Err(ContractError::EmptyShape)
    );
    assert_eq!(
        TensorSpec::contiguous(DType::F32, vec![2, 0]),
        Err(ContractError::ZeroDimension)
    );
}
