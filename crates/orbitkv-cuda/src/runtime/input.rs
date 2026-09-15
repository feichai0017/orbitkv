//! Owned and caller-registered inputs, including their physical capacity.

use crate::providers::DeviceBuffer;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use std::sync::Arc;

pub enum CudaInput {
    Buffer { buf: CudaSlice<u8>, len: usize },
    Ptr(u64),
}

impl CudaInput {
    pub(super) fn from_bytes(stream: &Arc<CudaStream>, bytes: &[u8]) -> Self {
        Self::from_bytes_with_capacity(stream, bytes, bytes.len())
    }

    pub(super) fn from_bytes_with_capacity(
        stream: &Arc<CudaStream>,
        bytes: &[u8],
        capacity: usize,
    ) -> Self {
        assert!(capacity >= bytes.len());
        if capacity == bytes.len() {
            return CudaInput::Buffer {
                buf: stream.clone_htod(bytes).unwrap(),
                len: bytes.len(),
            };
        }
        let mut buf = stream.alloc_zeros::<u8>(capacity).unwrap();
        if !bytes.is_empty() {
            let mut view = buf.slice_mut(..bytes.len());
            stream.memcpy_htod(bytes, &mut view).unwrap();
        }
        CudaInput::Buffer {
            buf,
            len: bytes.len(),
        }
    }

    /// A borrowed descriptor keeps both logical length and backing capacity.
    /// Provider plans may use a proved capacity across several logical shapes.
    pub(super) fn device_buffer(
        &self,
        stream: &Arc<CudaStream>,
        external: Option<&std::mem::ManuallyDrop<CudaSlice<u8>>>,
    ) -> Option<DeviceBuffer> {
        match self {
            Self::Buffer { buf, len } => {
                Some(DeviceBuffer::new(buf.device_ptr(stream).0, *len).with_capacity(buf.len()))
            }
            Self::Ptr(ptr) => external.map(|buffer| DeviceBuffer::new(*ptr, buffer.len())),
        }
    }
}
