use crate::providers::{HostOp, cublaslt};

#[cfg(test)]
pub(crate) type CublasLtTypeTuple = (
    orbitkv_compiler::dtype::DType,
    orbitkv_compiler::dtype::DType,
    orbitkv_compiler::dtype::DType,
    orbitkv_compiler::dtype::DType,
    &'static str,
    orbitkv_compiler::dtype::DType,
);

#[cfg(test)]
pub(crate) fn cublaslt_type_tuple(op: &dyn HostOp) -> Option<CublasLtTypeTuple> {
    op.as_any()
        .downcast_ref::<cublaslt::CuBlasLt>()
        .map(cublaslt::CuBlasLt::type_tuple)
}

#[cfg(test)]
pub(crate) type CublasLtScaleValues = (f64, f64);

#[cfg(test)]
pub(crate) fn cublaslt_scale_values(op: &dyn HostOp) -> Option<CublasLtScaleValues> {
    op.as_any()
        .downcast_ref::<cublaslt::CuBlasLt>()
        .map(cublaslt::CuBlasLt::scale_values)
}

#[cfg(test)]
pub(crate) fn cublaslt_epilogue(op: &dyn HostOp) -> Option<&'static str> {
    op.as_any()
        .downcast_ref::<cublaslt::CuBlasLt>()
        .map(cublaslt::CuBlasLt::epilogue)
}

#[cfg(test)]
pub(crate) type CublasLtMatrixOrders = (&'static str, &'static str, &'static str, &'static str);

#[cfg(test)]
pub(crate) fn cublaslt_matrix_orders(op: &dyn HostOp) -> Option<CublasLtMatrixOrders> {
    op.as_any()
        .downcast_ref::<cublaslt::CuBlasLt>()
        .map(cublaslt::CuBlasLt::matrix_orders)
}

#[cfg(test)]
pub(crate) type CublasLtTransposeOps = (&'static str, &'static str);

#[cfg(test)]
pub(crate) fn cublaslt_transpose_ops(op: &dyn HostOp) -> Option<CublasLtTransposeOps> {
    op.as_any()
        .downcast_ref::<cublaslt::CuBlasLt>()
        .map(cublaslt::CuBlasLt::transpose_ops)
}

#[cfg(test)]
pub(crate) fn cublaslt_c_d_layouts_match(op: &dyn HostOp) -> Option<bool> {
    op.as_any()
        .downcast_ref::<cublaslt::CuBlasLt>()
        .map(cublaslt::CuBlasLt::c_d_layouts_match)
}

#[cfg(test)]
pub(crate) type CublasLtTensorScaleInputs = (bool, bool);

#[cfg(test)]
pub(crate) fn cublaslt_tensor_scale_inputs(op: &dyn HostOp) -> Option<CublasLtTensorScaleInputs> {
    op.as_any()
        .downcast_ref::<cublaslt::CuBlasLt>()
        .map(cublaslt::CuBlasLt::tensor_scale_inputs)
}
