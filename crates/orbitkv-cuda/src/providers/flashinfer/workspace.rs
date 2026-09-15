//! Separate persistent plan metadata from serialized execution scratch and
//! temporary host staging. Retained graphs must never share mutable plan data.

use std::sync::{Arc, Mutex, OnceLock};

use crate::{
    cudarc::driver::{CudaSlice, CudaStream, DevicePtr, PinnedHostSlice},
    resource::SharedDeviceMemoryAllocation,
};

// Provider workspace budgets, independent of model/layer geometry. FlashInfer
// reports a planning error when a workload exceeds these supplied capacities.
pub(super) const FLOAT_WORKSPACE_SIZE: usize = 128 * 1024 * 1024;
pub(super) const INT_WORKSPACE_SIZE: usize = 8 * 1024 * 1024;

static FLOAT_WORKSPACE: OnceLock<CudaSlice<u8>> = OnceLock::new();
static PLAN_STAGING: Mutex<Option<PinnedHostSlice<u8>>> = Mutex::new(None);

pub(super) struct PlanWorkspace {
    _float: &'static CudaSlice<u8>,
    pub(super) float_ptr: u64,
    pub(super) metadata: CudaSlice<u8>,
    pub(super) int_ptr: u64,
}

impl PlanWorkspace {
    pub(super) fn new(stream: &Arc<CudaStream>) -> anyhow::Result<Self> {
        let float = FLOAT_WORKSPACE.get_or_init(|| {
            // The execution kernels initialize the scratch they consume.
            unsafe { stream.alloc::<u8>(FLOAT_WORKSPACE_SIZE).unwrap() }
        });
        // FlashInfer's planner initializes the metadata consumed by its kernels.
        let metadata = unsafe { stream.alloc::<u8>(INT_WORKSPACE_SIZE)? };
        let int_ptr = metadata.device_ptr(stream).0;
        Ok(Self {
            _float: float,
            float_ptr: float.device_ptr(stream).0,
            int_ptr,
            metadata,
        })
    }
}

/// Serialize planner writes and retire its asynchronous host-to-device copy
/// before another plan can reuse the pinned staging memory, including errors.
pub(super) fn with_plan_staging<T>(
    stream: &Arc<CudaStream>,
    plan: impl FnOnce(*mut std::ffi::c_void) -> T,
) -> anyhow::Result<T> {
    let _stage = tracing::info_span!(target: "orbitkv::stage", "cuda.provider.plan", provider = "flashinfer").entered();
    let mut staging = PLAN_STAGING
        .lock()
        .map_err(|_| anyhow::anyhow!("FlashInfer plan staging lock poisoned"))?;
    if staging.is_none() {
        // The native planner writes all ranges it copies to the device.
        *staging = Some(unsafe { stream.context().alloc_pinned::<u8>(INT_WORKSPACE_SIZE)? });
    }
    let pointer = staging.as_mut().unwrap().as_mut_ptr()?.cast();
    let result = plan(pointer);
    stream.synchronize()?;
    Ok(result)
}

/// Float scratch is shared only by dependency-ordered execution. Plan metadata
/// is charged separately to each prepared allocation, including inactive graphs.
pub fn shared_device_memory_allocation() -> SharedDeviceMemoryAllocation {
    SharedDeviceMemoryAllocation {
        key: "flashinfer-global-workspaces",
        bytes: FLOAT_WORKSPACE_SIZE,
    }
}

pub(crate) fn resident_shared_device_memory_allocations() -> Vec<SharedDeviceMemoryAllocation> {
    FLOAT_WORKSPACE
        .get()
        .is_some()
        .then(shared_device_memory_allocation)
        .into_iter()
        .collect()
}
