use std::ptr::NonNull;

use crate::{MemoryRegion, Result, TransferDesc, TransferEngine, TransferOp};

/// Physical transfer backend selected after OrbitKV has produced a state plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RemoteBackendKind {
    NativeRdma,
    Mooncake,
}

/// Backend-private address in an established peer session.
///
/// Native RDMA interprets this value as a virtual address. A Mooncake adapter
/// interprets it as a Segment offset. Callers must obtain it from an
/// authorized, generation-qualified capability rather than minting it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RemoteAddress(u64);

impl RemoteAddress {
    pub fn new(value: u64) -> Self {
        Self(value)
    }

    pub fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RemoteSlice {
    pub local_ptr: NonNull<u8>,
    pub remote: RemoteAddress,
    pub len: usize,
}

pub type RemoteCompletion = mea::oneshot::Receiver<Result<usize>>;

/// Byte-movement boundary beneath OrbitKV's state planner.
///
/// Replica discovery, bundle completeness, leases, and page generations stay
/// above this trait. A mover receives an already-authorized physical plan and
/// reports only transfer completion.
pub trait RemoteMover: Send + Sync {
    fn backend_kind(&self) -> RemoteBackendKind;

    fn register_memory(&self, regions: &[MemoryRegion]) -> Result<()>;

    fn unregister_memory(&self, ptrs: &[NonNull<u8>]) -> Result<()>;

    fn submit(
        &self,
        operation: TransferOp,
        peer: &str,
        slices: &[RemoteSlice],
    ) -> Result<Vec<RemoteCompletion>>;

    fn invalidate_peer(&self, peer: &str);
}

impl RemoteMover for TransferEngine {
    fn backend_kind(&self) -> RemoteBackendKind {
        RemoteBackendKind::NativeRdma
    }

    fn register_memory(&self, regions: &[MemoryRegion]) -> Result<()> {
        TransferEngine::register_memory(self, regions)
    }

    fn unregister_memory(&self, ptrs: &[NonNull<u8>]) -> Result<()> {
        TransferEngine::unregister_memory(self, ptrs)
    }

    fn submit(
        &self,
        operation: TransferOp,
        peer: &str,
        slices: &[RemoteSlice],
    ) -> Result<Vec<RemoteCompletion>> {
        let descriptors = slices
            .iter()
            .map(|slice| {
                let remote_ptr = NonNull::new(slice.remote.get() as *mut u8).ok_or(
                    crate::TransferError::InvalidArgument("remote address is zero"),
                )?;
                Ok(TransferDesc {
                    local_ptr: slice.local_ptr,
                    remote_ptr,
                    len: slice.len,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        self.batch_transfer_async(operation, peer, &descriptors)
    }

    fn invalidate_peer(&self, peer: &str) {
        self.invalidate_connection(peer);
    }
}
