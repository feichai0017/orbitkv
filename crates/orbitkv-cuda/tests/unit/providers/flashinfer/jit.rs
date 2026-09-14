use crate::tests::utilities::get_cuda_stream;

#[test]
#[ignore = "one-time JIT compile check for the gemma variants (~2 min each cold)"]
fn jit_compiles_gemma_variants() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    let target = crate::target::CudaTarget::from_context(stream.context()).unwrap();
    // sliding layers: head_dim 256 with the sliding-window kernel variant
    let _ = crate::providers::flashinfer::jit::ensure_compiled(target, 256, true, 2).unwrap();
    // full layers: head_dim 512 (16-bit only; f32 instantiation is gated out)
    let _ = crate::providers::flashinfer::jit::ensure_compiled(target, 512, false, 2).unwrap();
}

#[test]
#[ignore = "one-time JIT compile check for a non-power-of-two GQA ratio"]
fn jit_compiles_non_power_of_two_gqa_geometry() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    let target = crate::target::CudaTarget::from_context(stream.context()).unwrap();
    let _ = crate::providers::flashinfer::jit::ensure_compiled(target, 64, false, 7).unwrap();
}
