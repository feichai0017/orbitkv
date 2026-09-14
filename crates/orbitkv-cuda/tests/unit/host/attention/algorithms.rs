use super::reference::Case;
use crate::host::{
    HostOp,
    flashinfer::{FlashInferAlgorithm, FlashInferAttention},
};
use cudarc::driver::CudaContext;
use orbitkv_compiler::dtype::DType;

#[test]
fn both_flashinfer_decode_algorithms_match_independent_reference() {
    let stream = CudaContext::new(0).unwrap().new_stream().unwrap();
    for (dimension, dtype, window) in [
        (64, DType::F16, None),
        (128, DType::Bf16, Some(5)),
        (256, DType::Bf16, None),
    ] {
        let case = Case::new(dimension, 16, &[1, 1], &[35, 7]);
        let buffers = case.upload(&stream, dtype);
        for algorithm in [
            FlashInferAlgorithm::CudaCoreDecode,
            FlashInferAlgorithm::TensorCore,
        ] {
            let op = FlashInferAttention::paged(
                algorithm,
                case.query_heads,
                case.kv_heads,
                dimension,
                case.page_size,
                's'.into(),
                'c'.into(),
                'b'.into(),
                dtype,
                case.scale(),
                window,
            );
            op.prepare_compilation(&stream, &case.dimensions()).unwrap();
            op.execute(
                &stream,
                buffers.nodes[7],
                &buffers.nodes[..7],
                &buffers.map,
                &case.dimensions(),
            )
            .unwrap();
            buffers.check(&stream, dtype, &case.expected(window));
        }
    }
}
