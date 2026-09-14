use super::*;
use crate::{
    cudarc::driver::CudaContext,
    kernel::{CudaGraphExecHandle, CudaGraphHandle},
};
use half::bf16;

// Zero queries/keys give uniform softmax. The independent reference is the
// average of exactly the causal/window-visible values for each ragged request.
struct Case {
    exec: CudaGraphExecHandle,
    _graph: CudaGraphHandle,
    prepared: PreparedFlashInferAttention,
    storage: Vec<CudaSlice<u8>>,
    expected: Vec<f32>,
}

impl Case {
    fn new(
        stream: &Arc<CudaStream>,
        algorithm: FlashInferAlgorithm,
        queries: &[usize],
        lengths: &[usize],
        window: Option<usize>,
    ) -> Self {
        assert_eq!(queries.len(), lengths.len());
        let query_heads = 4;
        let kv_heads = 2;
        let head_dim = 64;
        let page_size = 16;
        let total_queries: usize = queries.iter().sum();
        let page_counts = lengths
            .iter()
            .map(|length| length.div_ceil(page_size))
            .collect_vec();
        let page_count: usize = page_counts.iter().sum();
        let mut qo = vec![0_i32];
        let mut kv = vec![0_i32];
        for (&q, &pages) in queries.iter().zip(&page_counts) {
            qo.push(qo.last().unwrap() + q as i32);
            kv.push(kv.last().unwrap() + pages as i32);
        }
        let last = lengths
            .iter()
            .map(|length| ((length - 1) % page_size + 1) as i32)
            .collect_vec();
        let indices = (0..page_count as i32).rev().collect_vec();
        let mut values = vec![bf16::ZERO; page_count * page_size * kv_heads * head_dim];
        let mut expected = vec![0_f32; total_queries * query_heads * head_dim];
        for (request, (&query_count, &length)) in queries.iter().zip(lengths).enumerate() {
            for token in 0..length {
                let page = indices[kv[request] as usize + token / page_size] as usize;
                for head in 0..kv_heads {
                    for dim in 0..head_dim {
                        let index = ((page * page_size + token % page_size) * kv_heads + head)
                            * head_dim
                            + dim;
                        values[index] = bf16::from_f32(
                            (token % 13) as f32 / 8.0
                                + request as f32 / 4.0
                                + head as f32 / 2.0
                                + (dim % 3) as f32 / 32.0,
                        );
                    }
                }
            }
            for query in 0..query_count {
                let end = length - query_count + query + 1;
                let start = window.map_or(0, |left| end.saturating_sub(left + 1));
                for head in 0..query_heads {
                    for dim in 0..head_dim {
                        let sum: f32 = (start..end)
                            .map(|token| {
                                let page =
                                    indices[kv[request] as usize + token / page_size] as usize;
                                let index = ((page * page_size + token % page_size) * kv_heads
                                    + head / (query_heads / kv_heads))
                                    * head_dim
                                    + dim;
                                values[index].to_f32()
                            })
                            .sum();
                        let index =
                            (head * total_queries + qo[request] as usize + query) * head_dim + dim;
                        expected[index] = sum / (end - start) as f32;
                    }
                }
            }
        }
        let inputs = [
            vec![0_u8; total_queries * query_heads * head_dim * std::mem::size_of::<bf16>()],
            vec![0_u8; values.len() * std::mem::size_of::<bf16>()],
            bytemuck::cast_slice(&values).to_vec(),
            bytemuck::cast_slice(&indices).to_vec(),
            bytemuck::cast_slice(&qo).to_vec(),
            bytemuck::cast_slice(&kv).to_vec(),
            bytemuck::cast_slice(&last).to_vec(),
            vec![0_u8; expected.len() * std::mem::size_of::<bf16>()],
        ];
        let storage = inputs
            .iter()
            .map(|bytes| stream.clone_htod(bytes).unwrap())
            .collect_vec();
        let nodes = (0..storage.len()).map(NodeIndex::new).collect_vec();
        let buffers = storage
            .iter()
            .zip(&nodes)
            .map(|(buffer, &node)| {
                (
                    node,
                    DeviceBuffer::new(buffer.device_ptr(stream).0, buffer.len()),
                )
            })
            .collect();
        let op = FlashInferAttention::paged(
            algorithm,
            query_heads,
            kv_heads,
            head_dim,
            page_size,
            total_queries.into(),
            page_count.into(),
            queries.len().into(),
            DType::Bf16,
            0.0,
            window,
        );
        let resolved = op
            .resolve_for_graph(nodes[7], &nodes[..7], &buffers, &DynMap::default())
            .unwrap();
        let ptrs = resolved.ptrs;
        let prepared = op
            .prepare_resolved_for_graph(stream, resolved, true)
            .unwrap();
        CudaGraphHandle::begin_standalone_capture(stream).unwrap();
        prepared.enqueue(stream, ptrs, true).unwrap();
        let graph = CudaGraphHandle::end_standalone_capture(stream).unwrap();
        let exec = graph.instantiate().unwrap();
        Self {
            exec,
            _graph: graph,
            prepared,
            storage,
            expected,
        }
    }

    fn check(&self, stream: &Arc<CudaStream>) {
        self.exec.launch(stream).unwrap();
        let bytes = stream.clone_dtoh(self.storage.last().unwrap()).unwrap();
        for (actual, expected) in bytes
            .chunks_exact(2)
            .map(|bytes| bf16::from_le_bytes(bytes.try_into().unwrap()).to_f32())
            .zip(&self.expected)
        {
            assert!(
                (actual - expected).abs() < 0.016,
                "actual={actual}, expected={expected}"
            );
        }
    }
}

#[test]
#[ignore = "requires CUDA and FlashInfer; retained decode/prefill plans must preserve their own metadata"]
fn retained_plans_survive_other_shapes_and_graph_retirement() {
    let context = CudaContext::new(0).expect("CUDA required");
    let stream = context.new_stream().unwrap();
    let decode = Case::new(
        &stream,
        FlashInferAlgorithm::CudaCoreDecode,
        &[1],
        &[257],
        None,
    );
    decode.check(&stream);
    let metadata: &CudaSlice<u8> = &decode.prepared.workspace.metadata;
    let original = stream.clone_dtoh(metadata).unwrap();
    let prefill = Case::new(
        &stream,
        FlashInferAlgorithm::TensorCore,
        &[3, 2],
        &[23, 17],
        Some(11),
    );
    prefill.check(&stream);
    assert!(
        original == stream.clone_dtoh(metadata).unwrap(),
        "preparing another shape overwrote retained plan metadata"
    );
    let ragged_decode = Case::new(
        &stream,
        FlashInferAlgorithm::CudaCoreDecode,
        &[1, 1, 1],
        &[37, 513, 19],
        None,
    );
    let tensor_decode = Case::new(
        &stream,
        FlashInferAlgorithm::TensorCore,
        &[1, 1],
        &[65, 19],
        None,
    );
    for _ in 0..3 {
        decode.check(&stream);
        ragged_decode.check(&stream);
        prefill.check(&stream);
        tensor_decode.check(&stream);
    }
    drop(prefill);
    stream.synchronize().unwrap();
    for _ in 0..3 {
        ragged_decode.check(&stream);
        decode.check(&stream);
        tensor_decode.check(&stream);
    }
}
