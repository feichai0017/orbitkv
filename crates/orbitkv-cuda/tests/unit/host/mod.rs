use super::DeviceBuffer;
use crate::runtime::CudaRuntime;
use orbitkv_compiler::op::{IntoEgglogOp, Runtime};

#[test]
fn device_buffer_host_mirror_is_explicit_and_borrowed() {
    let plain = DeviceBuffer::new(0x1000, 16);
    assert!(plain.host_bytes().is_none());

    let host = [1u8, 2, 3, 4];
    let mirrored = plain.with_host_bytes(&host);
    assert_eq!(mirrored.host_bytes(), Some(host.as_slice()));
    assert_eq!(mirrored.ptr(), 0x1000);
    assert_eq!(mirrored.len(), 16);
}

#[test]
fn lite_registers_generic_attention_but_not_sink_attention() {
    let ops = <CudaRuntime as Runtime>::Ops::into_vec();
    assert_eq!(
        ops.iter()
            .filter(|op| op.sort().name == "FlashInferAttention")
            .count(),
        1
    );
    assert!(ops.iter().all(|op| op.sort().name != "SinkAttention"));
}
