use super::*;
use cudarc::driver::DevicePtr;

#[test]
#[ignore = "requires SM90 and DeepGEMM; verifies preparation is mandatory"]
fn execution_requires_preparation_and_keeps_the_selected_library() {
    let context = cudarc::driver::CudaContext::new(0).expect("SM90 required");
    assert_eq!(context.compute_capability().unwrap(), (9, 0));
    let stream = context.default_stream();
    let max_rows = 24;
    let width = 128;
    let x = stream
        .clone_htod(&vec![bf16::ONE.to_bits(); max_rows * width])
        .unwrap();
    let w = stream.clone_htod(&vec![0x38_u8; width * width]).unwrap();
    let ws = stream.clone_htod(&[1.0_f32]).unwrap();
    let y = stream.alloc_zeros::<u16>(max_rows * width).unwrap();
    let inputs = [NodeIndex::new(0), NodeIndex::new(1), NodeIndex::new(2)];
    let output = NodeIndex::new(3);
    let buffers = [
        (
            inputs[0],
            DeviceBuffer::new(x.device_ptr(&stream).0, max_rows * width * 2),
        ),
        (
            inputs[1],
            DeviceBuffer::new(w.device_ptr(&stream).0, width * width),
        ),
        (inputs[2], DeviceBuffer::new(ws.device_ptr(&stream).0, 4)),
        (
            output,
            DeviceBuffer::new(y.device_ptr(&stream).0, max_rows * width * 2),
        ),
    ]
    .into_iter()
    .collect();
    let op = DeepGemm {
        rows: 's'.into(),
        selection: device_selection(&stream, max_rows, width, width, 0),
        provider: jit::provider_identity().unwrap(),
        ..Default::default()
    };
    let mut dims = [('s'.into(), 4)].into_iter().collect();
    let error = op
        .execute(&stream, output, &inputs, &buffers, &dims)
        .unwrap_err();
    assert!(error.to_string().contains("kernel is not prepared"));
    assert!(op.prepared.get().is_none());
    op.prepare_compilation(&stream, &dims).unwrap();
    let library = *op.prepared.get().unwrap();
    for rows in [4, 12, 24, 1, 4] {
        dims.insert('s'.into(), rows);
        op.execute(&stream, output, &inputs, &buffers, &dims)
            .unwrap();
        let values = stream.clone_dtoh(&y).unwrap();
        assert!(
            values[..rows * width]
                .iter()
                .all(|v| *v == bf16::from_f32(width as f32).to_bits())
        );
        assert!(std::ptr::eq(*op.prepared.get().unwrap(), library));
    }
    dims.insert('s'.into(), max_rows + 1);
    assert!(
        op.execute(&stream, output, &inputs, &buffers, &dims)
            .unwrap_err()
            .to_string()
            .contains("outside selected kernel range")
    );
    context.check_err().unwrap();
}
