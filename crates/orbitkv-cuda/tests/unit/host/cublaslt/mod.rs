mod identity;
mod recapture;

use super::*;

#[derive(Debug, Clone)]
struct TestArityTwoFp8(crate::kernel::quant_f8::KernelQuantF8);

impl orbitkv_compiler::op::CustomOp for TestArityTwoFp8 {
    fn to_llir_op(&self) -> LLIROp {
        LLIROp::new::<dyn crate::kernel::KernelOp>(
            Box::new(self.0.clone()) as Box<dyn crate::kernel::KernelOp>
        )
    }
}

#[test]
fn matmul_spec_resolution_is_shared_with_resource_prepare_key() {
    let op = CuBlasLt {
        m: Expression::from('m'),
        n: Expression::from(5),
        k: Expression::from(7),
        a_order: cublasLtOrder_t::CUBLASLT_ORDER_ROW,
        lda: Expression::from(1),
        ldb: Expression::from(1),
        ldc: Expression::from(1),
        ldd: Expression::from(1),
        ..Default::default()
    };
    let dyn_map = FxHashMap::from_iter([(Symbol::from('m'), 3)]);

    let spec = op.resolve_matmul_spec(&dyn_map).unwrap();
    let prepare_key = op.prepare_key_for_resources(&dyn_map).unwrap();

    assert_eq!(prepare_key.spec, spec);
    assert_eq!(spec.problem.m, 3);
    assert_eq!(spec.problem.n, 5);
    assert_eq!(spec.problem.k, 7);
    assert_eq!(spec.a.ld, 7, "row-order A must clamp ld to columns");
    assert_eq!(spec.b.ld, 5, "column-order B must clamp ld to rows");
    assert_eq!(spec.c.ld, 3);
    assert_eq!(spec.d.ld, 3);
    assert_eq!(spec.workspace_size, CuBlasLt::WORKSPACE_SIZE_BYTES);

    let changed_dyn_map = FxHashMap::from_iter([(Symbol::from('m'), 11)]);
    let changed_key = op.prepare_key_for_resources(&changed_dyn_map).unwrap();
    assert_eq!(changed_key.spec.problem.m, 11);

    let restored_key = op.prepare_key_for_resources(&dyn_map).unwrap();
    assert_eq!(restored_key, prepare_key);
}

#[test]
fn matmul_spec_resolution_reports_missing_dynamic_dimension() {
    let op = CuBlasLt {
        m: Expression::from('m'),
        ..Default::default()
    };

    let error = op
        .resolve_matmul_spec(&FxHashMap::default())
        .unwrap_err()
        .to_string();

    assert!(error.contains("unresolved cuBLASLt dimension m"), "{error}");
}

#[test]
fn staged_arity_two_fp8_activation_reaches_scaled_cublaslt() {
    let mut cx = Graph::default();
    let activation = cx.tensor((1usize, 32usize)).as_dtype(DType::Bf16);
    let input_scale = cx.tensor(());
    let weight_scale = cx.tensor(());
    let weight = cx.tensor((8usize, 16usize)).as_dtype(DType::F8E4M3);
    // The consumer contract is intentionally generic: any arity-two F8
    // custom op whose last input is the activation scale is eligible.
    // Wrap Lite's real FP8 quant primitive so the graph and LLIR contracts
    // remain semantically aligned without depending on a full-only fused
    // producer.
    let quantized = cx.custom_op(
        TestArityTwoFp8(crate::kernel::quant_f8::KernelQuantF8::from_size(
            16usize.into(),
        )),
        vec![activation, input_scale],
        (1usize, 16usize),
        DType::F8E4M3,
    );
    let matmul = quantized.matmul(weight.t()).cast(DType::F32);
    (matmul * (input_scale * weight_scale).expand_rhs(matmul.dims())).output();

    cx.build_search_space::<crate::runtime::CudaRuntime>(CompileOptions::default());

    assert!(
        cx.egraph()
            .unwrap()
            .enodes
            .values()
            .any(|(label, _)| label == "cublaslt_scaled"),
        "arity-two FP8 custom activation should reach the shared scaled-cuBLASLt consumer"
    );
}

#[test]
fn lt_scalar_packs_f32_scale_values() {
    match LtScalar::one(DType::F32).unwrap() {
        LtScalar::F32(value) => assert_eq!(value, 1.0),
        other => panic!("expected f32 scalar, got {other:?}"),
    }

    match LtScalar::zero(DType::F32).unwrap() {
        LtScalar::F32(value) => assert_eq!(value, 0.0),
        other => panic!("expected f32 scalar, got {other:?}"),
    }
}

#[test]
fn lt_scalar_packs_f64_scale_values() {
    match LtScalar::one(DType::F64).unwrap() {
        LtScalar::F64(value) => assert_eq!(value, 1.0),
        other => panic!("expected f64 scalar, got {other:?}"),
    }

    match LtScalar::zero(DType::F64).unwrap() {
        LtScalar::F64(value) => assert_eq!(value, 0.0),
        other => panic!("expected f64 scalar, got {other:?}"),
    }
}

#[test]
fn lt_scalar_packs_low_precision_scale_values() {
    match LtScalar::one(DType::F16).unwrap() {
        LtScalar::F16(value) => assert_eq!(f32::from(value), 1.0),
        other => panic!("expected f16 scalar, got {other:?}"),
    }

    match LtScalar::zero(DType::Bf16).unwrap() {
        LtScalar::Bf16(value) => assert_eq!(f32::from(value), 0.0),
        other => panic!("expected bf16 scalar, got {other:?}"),
    }
}

#[test]
fn lt_scalar_rejects_non_host_scalar_scale_dtypes() {
    assert!(LtScalar::one(DType::TF32).is_err());
    assert!(LtScalar::zero(DType::F8E4M3).is_err());
}

#[test]
fn fp8_cuda_dtypes_request_tensorwide_scales() {
    assert!(cuda_dtype_needs_tensorwide_scale(
        cudaDataType::CUDA_R_8F_E4M3
    ));
    assert!(cuda_dtype_needs_tensorwide_scale(
        cudaDataType::CUDA_R_8F_E5M2
    ));
    assert!(!cuda_dtype_needs_tensorwide_scale(cudaDataType::CUDA_R_32F));
}

#[test]
fn cublaslt_pointers_alias_output_as_c_for_two_input_beta_zero() {
    let output = NodeIndex::new(0);
    let a = NodeIndex::new(1);
    let b = NodeIndex::new(2);
    let buffers = buffers_for(&[(output, 0xD000), (a, 0xA000), (b, 0xB000)]);

    let ptrs = resolve_cublaslt_pointers(
        output,
        &[a, b],
        &buffers,
        0.0,
        cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DEFAULT,
        false,
        false,
    )
    .unwrap();

    assert_eq!(ptrs.a, 0xA000);
    assert_eq!(ptrs.b, 0xB000);
    assert_eq!(ptrs.c, 0xD000);
    assert_eq!(ptrs.d, 0xD000);
    assert_eq!(ptrs.bias, None);
}

#[test]
fn cublaslt_pointers_ignore_extra_inputs_for_beta_zero() {
    let output = NodeIndex::new(0);
    let a = NodeIndex::new(1);
    let b = NodeIndex::new(2);
    let extra = NodeIndex::new(3);
    let buffers = buffers_for(&[(output, 0xD000), (a, 0xA000), (b, 0xB000), (extra, 0xEEEE)]);

    let ptrs = resolve_cublaslt_pointers(
        output,
        &[a, b, extra],
        &buffers,
        0.0,
        cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DEFAULT,
        false,
        false,
    )
    .unwrap();

    assert_eq!(ptrs.a, 0xA000);
    assert_eq!(ptrs.b, 0xB000);
    assert_eq!(ptrs.c, 0xD000);
    assert_eq!(ptrs.d, 0xD000);
    assert_eq!(ptrs.bias, None);
}

#[test]
fn cublaslt_pointers_use_distinct_c_input_when_present() {
    let output = NodeIndex::new(0);
    let a = NodeIndex::new(1);
    let b = NodeIndex::new(2);
    let c = NodeIndex::new(3);
    let buffers = buffers_for(&[(output, 0xD000), (a, 0xA000), (b, 0xB000), (c, 0xC000)]);

    let ptrs = resolve_cublaslt_pointers(
        output,
        &[a, b, c],
        &buffers,
        1.0,
        cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DEFAULT,
        false,
        false,
    )
    .unwrap();

    assert_eq!(ptrs.a, 0xA000);
    assert_eq!(ptrs.b, 0xB000);
    assert_eq!(ptrs.c, 0xC000);
    assert_eq!(ptrs.d, 0xD000);
    assert_eq!(ptrs.bias, None);
}

#[test]
fn cublaslt_pointers_use_bias_input_for_bias_epilogue() {
    let output = NodeIndex::new(0);
    let a = NodeIndex::new(1);
    let b = NodeIndex::new(2);
    let bias = NodeIndex::new(3);
    let buffers = buffers_for(&[(output, 0xD000), (a, 0xA000), (b, 0xB000), (bias, 0xB1A5)]);

    let ptrs = resolve_cublaslt_pointers(
        output,
        &[a, b, bias],
        &buffers,
        0.0,
        cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS,
        false,
        false,
    )
    .unwrap();

    assert_eq!(ptrs.a, 0xA000);
    assert_eq!(ptrs.b, 0xB000);
    assert_eq!(ptrs.c, 0xD000);
    assert_eq!(ptrs.d, 0xD000);
    assert_eq!(ptrs.bias, Some(0xB1A5));
}

#[test]
fn cublaslt_pointers_use_tensor_scale_inputs_after_base_inputs() {
    let output = NodeIndex::new(0);
    let a = NodeIndex::new(1);
    let b = NodeIndex::new(2);
    let a_scale = NodeIndex::new(3);
    let b_scale = NodeIndex::new(4);
    let buffers = buffers_for(&[
        (output, 0xD000),
        (a, 0xA000),
        (b, 0xB000),
        (a_scale, 0xA5A5),
        (b_scale, 0xB5B5),
    ]);

    let ptrs = resolve_cublaslt_pointers(
        output,
        &[a, b, a_scale, b_scale],
        &buffers,
        0.0,
        cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DEFAULT,
        true,
        true,
    )
    .unwrap();

    assert_eq!(ptrs.a, 0xA000);
    assert_eq!(ptrs.b, 0xB000);
    assert_eq!(ptrs.c, 0xD000);
    assert_eq!(ptrs.d, 0xD000);
    assert_eq!(ptrs.bias, None);
    assert_eq!(ptrs.a_scale, Some(0xA5A5));
    assert_eq!(ptrs.b_scale, Some(0xB5B5));
}

#[test]
fn cublaslt_pointers_reject_two_input_nonzero_beta() {
    let output = NodeIndex::new(0);
    let a = NodeIndex::new(1);
    let b = NodeIndex::new(2);
    let buffers = buffers_for(&[(output, 0xD000), (a, 0xA000), (b, 0xB000)]);

    let err = resolve_cublaslt_pointers(
        output,
        &[a, b],
        &buffers,
        1.0,
        cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DEFAULT,
        false,
        false,
    )
    .unwrap_err();

    assert!(
        err.to_string().contains("requires a third C input"),
        "unexpected error: {err}"
    );
}

#[test]
fn cublaslt_pointers_reject_missing_bias_input() {
    let output = NodeIndex::new(0);
    let a = NodeIndex::new(1);
    let b = NodeIndex::new(2);
    let buffers = buffers_for(&[(output, 0xD000), (a, 0xA000), (b, 0xB000)]);

    let err = resolve_cublaslt_pointers(
        output,
        &[a, b],
        &buffers,
        0.0,
        cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS,
        false,
        false,
    )
    .unwrap_err();

    assert!(
        err.to_string().contains("requires a bias input"),
        "unexpected error: {err}"
    );
}

fn buffers_for(entries: &[(NodeIndex, u64)]) -> FxHashMap<NodeIndex, DeviceBuffer> {
    entries
        .iter()
        .map(|(node, ptr)| (*node, DeviceBuffer::new(*ptr, 16)))
        .collect()
}
