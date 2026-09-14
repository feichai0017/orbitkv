use super::*;
use crate::{cudarc::driver::CudaContext, runtime::CudaRuntime};
use half::bf16;
use orbitkv_compiler::{op::Runtime, prelude::*};
use safetensors::tensor::{Dtype, TensorView};
use std::path::PathBuf;

struct Fixture(PathBuf);
impl Fixture {
    fn new(tensors: &[(&str, TensorView<'_>)]) -> Self {
        let path = std::env::temp_dir().join(format!(
            "orbitkv-weights-{}.safetensors",
            uuid::Uuid::new_v4()
        ));
        std::fs::write(
            &path,
            safetensors::serialize(tensors.iter().map(|(name, tensor)| (*name, tensor)), None)
                .unwrap(),
        )
        .unwrap();
        Self(path)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn read_input(runtime: &CudaRuntime, input: GraphTensor) -> Vec<u8> {
    let CudaInput::Buffer { buf, len } = &runtime.hlir_buffers[&input.id] else {
        panic!("owned input expected")
    };
    let bytes = runtime.cuda_stream.clone_dtoh(buf).unwrap();
    assert_eq!(bytes.len(), *len);
    bytes
}

#[test]
#[ignore = "requires CUDA; verifies raw FP8 upload, BF16 promotion and mapping lifetime"]
fn loaded_weights_survive_mapping_release_and_clear_stale_host_mirrors() {
    let context = CudaContext::new(0).expect("CUDA required");
    let mut graph = Graph::default();
    let promoted = graph.named_tensor("promoted", 2);
    let raw = graph.named_tensor("raw", 256).as_dtype(DType::F8E4M3);
    let source = [bf16::from_f32(1.5), bf16::from_f32(-2.25)];
    let bits = (0..=u8::MAX).collect::<Vec<_>>();
    let file = Fixture::new(&[
        (
            "promoted",
            TensorView::new(Dtype::BF16, vec![2], bytemuck::cast_slice(&source)).unwrap(),
        ),
        (
            "raw",
            TensorView::new(Dtype::F8_E4M3, vec![256], &bits).unwrap(),
        ),
        (
            "unused",
            TensorView::new(Dtype::U8, vec![256], &bits).unwrap(),
        ),
    ]);
    let mut runtime = CudaRuntime::initialize(context.default_stream());
    runtime.set_data_with_host_mirror(promoted, vec![0.0_f32; 2]);
    let report = runtime.load_safetensors(&graph, &file.0).unwrap();
    drop(file);
    assert_eq!(
        report,
        WeightLoadReport {
            tensors: 2,
            converted_tensors: 1,
            source_bytes: 260,
            device_bytes: 264
        }
    );
    assert!(runtime.hlir_host_mirrors.is_empty());
    assert!(runtime.changed_hlir.contains(&promoted.id));
    assert_eq!(read_input(&runtime, raw), bits);
    assert_eq!(
        read_input(&runtime, promoted),
        bytemuck::cast_slice::<_, u8>(&[1.5_f32, -2.25])
    );
}

#[test]
#[ignore = "requires CUDA; malformed/unsupported shards must not replace existing bindings"]
fn invalid_shards_return_contextual_errors_before_rebinding() {
    let context = CudaContext::new(0).expect("CUDA required");
    let mut graph = Graph::default();
    let valid = graph.named_tensor("valid", 1);
    let _invalid = graph.named_tensor("invalid", 1);
    let mut runtime = CudaRuntime::initialize(context.default_stream());
    runtime.set_data(valid, vec![7.0_f32]);
    let source = [0_u8; 4];
    let file = Fixture::new(&[
        (
            "valid",
            TensorView::new(Dtype::F32, vec![1], &source).unwrap(),
        ),
        (
            "invalid",
            TensorView::new(Dtype::I32, vec![1], &source).unwrap(),
        ),
    ]);
    let pointer = runtime.input_allocation(valid).unwrap();
    let error = runtime.load_safetensors(&graph, &file.0).unwrap_err();
    assert!(format!("{error:#}").contains("tensor invalid"));
    assert_eq!(runtime.input_allocation(valid).unwrap(), pointer);
    assert_eq!(
        read_input(&runtime, valid),
        bytemuck::cast_slice::<_, u8>(&[7.0_f32])
    );
    std::fs::write(&file.0, b"invalid safetensors").unwrap();
    assert!(runtime.load_safetensors(&graph, &file.0).is_err());
    drop(file);
    assert!(
        runtime
            .load_safetensors(&graph, "missing-checkpoint.safetensors")
            .is_err()
    );
    assert_eq!(runtime.input_allocation(valid).unwrap(), pointer);
}
