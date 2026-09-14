use super::*;
use crate::kernel::{CudaGraphExecHandle, CudaGraphHandle};
use crate::providers::attention::tests::reference::{Buffers, Case};
use cudarc::driver::CudaContext;

mod search;

fn operation(case: &Case, dtype: DType, window: Option<usize>) -> FlashAttention {
    FlashAttention {
        query_heads: case.query_heads,
        kv_heads: case.kv_heads,
        head_dim: case.head_dim,
        page_size: case.page_size,
        query_tokens: 's'.into(),
        context_pages: 'c'.into(),
        requests: 'b'.into(),
        dtype,
        scale: case.scale(),
        window_left: window.map_or(-1, |n| n as i64),
        provider: jit::provider_identity().unwrap(),
        prepared: Mutex::default(),
    }
}

#[test]
fn scratch_planning_is_pointer_free_and_rejects_invalid_ranges() {
    let case = Case::new(128, 16, &[2, 1], &[19, 7]);
    let op = FlashAttention {
        query_heads: case.query_heads,
        kv_heads: case.kv_heads,
        head_dim: case.head_dim,
        page_size: case.page_size,
        query_tokens: 's'.into(),
        context_pages: 'c'.into(),
        requests: 'b'.into(),
        dtype: DType::Bf16,
        scale: case.scale(),
        window_left: -1,
        ..Default::default()
    };
    let plan = Plan::new(&op, &case.dimensions()).unwrap();
    assert!(op.prepared.lock().unwrap().is_none());
    assert!(plan.bytes >= case.page_indices.len() * case.last_page_len.len() * size_of::<i32>());
    assert!(
        plan.bytes < case.key.len() * 2,
        "scratch must not copy K/V payload"
    );
    let mut invalid = case.dimensions();
    invalid.insert('s'.into(), 1);
    assert!(Plan::new(&op, &invalid).is_err());
    invalid.insert('c'.into(), usize::MAX);
    assert!(Plan::new(&op, &invalid).is_err());
    assert!(Plan::new(&op, &DynMap::default()).is_err());
}

#[test]
fn paged_decode_and_prefill_match_independent_reference() {
    let context = CudaContext::new(0).expect("CUDA required");
    let stream = context.new_stream().unwrap();
    for (dimension, dtype, queries, lengths, window, page_size) in [
        (64, DType::F16, vec![1, 1], vec![19, 7], None, 16),
        (128, DType::F16, vec![3, 2], vec![35, 7], None, 16),
        (256, DType::F16, vec![1], vec![65], None, 32),
        (64, DType::Bf16, vec![1, 1], vec![257, 37], None, 16),
        (128, DType::Bf16, vec![2, 1], vec![19, 7], Some(5), 16),
        (256, DType::Bf16, vec![3, 2], vec![35, 7], None, 16),
        (256, DType::Bf16, vec![1, 1], vec![513, 19], Some(17), 32),
    ] {
        let case = Case::new(dimension, page_size, &queries, &lengths);
        let buffers = case.upload(&stream, dtype);
        let op = operation(&case, dtype, window);
        let key_before = stream.clone_dtoh(&buffers.storage[1]).unwrap();
        let value_before = stream.clone_dtoh(&buffers.storage[2]).unwrap();
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
        assert_eq!(stream.clone_dtoh(&buffers.storage[1]).unwrap(), key_before);
        assert_eq!(
            stream.clone_dtoh(&buffers.storage[2]).unwrap(),
            value_before
        );
    }
}

struct Captured {
    exec: CudaGraphExecHandle,
    _graph: CudaGraphHandle,
    _resources: Vec<CudaGraphCaptureResource>,
    buffers: Buffers,
    expected: Vec<f32>,
}

impl Captured {
    fn new(op: &FlashAttention, case: &Case, stream: &Arc<CudaStream>) -> Self {
        let buffers = case.upload(stream, DType::Bf16);
        let dimensions = case.dimensions();
        op.prepare_cuda_graph_capture(
            stream,
            buffers.nodes[7],
            &buffers.nodes[..7],
            &buffers.map,
            &dimensions,
        )
        .unwrap();
        let before = format!("{op:?}");
        CudaGraphHandle::begin_standalone_capture(stream).unwrap();
        op.execute(
            stream,
            buffers.nodes[7],
            &buffers.nodes[..7],
            &buffers.map,
            &dimensions,
        )
        .unwrap();
        let graph = CudaGraphHandle::end_standalone_capture(stream).unwrap();
        let resources = op.cuda_graph_capture_resources();
        assert_eq!(before, format!("{op:?}"));
        let exec = graph.instantiate().unwrap();
        Self {
            exec,
            _graph: graph,
            _resources: resources,
            buffers,
            expected: case.expected(None),
        }
    }
    fn check(&self, stream: &Arc<CudaStream>) {
        self.exec.launch(stream).unwrap();
        self.buffers.check(stream, DType::Bf16, &self.expected);
    }
}

#[test]
fn captured_workspace_survives_replanning_and_other_graph_retirement() {
    let stream = CudaContext::new(0).unwrap().new_stream().unwrap();
    let first = Case::new(256, 16, &[1], &[19]);
    let op = operation(&first, DType::Bf16, None);
    let decode = Captured::new(&op, &first, &stream);
    decode.check(&stream);
    let second = Case::new(256, 16, &[2, 1], &[35, 7]);
    let prefill = Captured::new(&op, &second, &stream);
    for _ in 0..3 {
        prefill.check(&stream);
        decode.check(&stream);
    }
    drop(prefill);
    decode.check(&stream);
    op.prepared.lock().unwrap().take();
    decode.check(&stream);
}

#[test]
fn graph_replay_reads_updated_page_metadata() {
    let stream = CudaContext::new(0).unwrap().new_stream().unwrap();
    let mut case = Case::new(256, 16, &[1, 1], &[35, 19]);
    let op = operation(&case, DType::Bf16, None);
    let mut captured = Captured::new(&op, &case, &stream);
    captured.check(&stream);
    case.page_indices.reverse();
    case.last_page_len = vec![9, 7];
    for (index, values) in [(3, &case.page_indices), (6, &case.last_page_len)] {
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        stream
            .memcpy_htod(&bytes, &mut captured.buffers.storage[index])
            .unwrap();
    }
    captured.expected = case.expected(None);
    captured.check(&stream);
}
