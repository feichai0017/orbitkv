//! cuBLASLt provider for dense BF16 GEMM.

use crate::command::{CommandScope, ExternalCommandError, ResolvedRrww};
use crate::driver::bind_context_for_cleanup;
use crate::gemm::plan::{Bf16DenseGemmEnqueueError, Bf16DenseGemmOperands, Bf16DenseGemmPlanError};
use crate::memory::{DeviceRegion, ReadWriteDeviceRegion};
use cuda_core::{CudaContext, DeviceCopy};
use cudarc::cublaslt::{result, sys};
use half::bf16;
use loom_infer::Bf16DenseGemmSpec;
use std::ffi::c_void;
use std::mem::size_of;
use std::sync::Arc;
use std::sync::atomic::{AtomicI32, Ordering};

const CUBLASLT_PROVIDER: &str = "cuBLASLt";
const HOPPER_WORKSPACE_LIMIT_BYTES: usize = 32 * 1024 * 1024;
const TENSOR_ALIGNMENT_BYTES: u64 = 16;
const WORKSPACE_ALIGNMENT_BYTES: u64 = 256;

/// A cuBLASLt provider bound to one exact CUDA context.
#[derive(Clone)]
pub(crate) struct CublasLtProvider {
    inner: Arc<ProviderInner>,
}

impl CublasLtProvider {
    /// Creates one long-lived cuBLASLt handle in `context`.
    pub(crate) fn load(context: &Arc<CudaContext>) -> Result<Self, Bf16DenseGemmPlanError> {
        context.bind_to_thread()?;
        let handle = result::create_handle()?;
        // SAFETY: cuBLASLt is loaded and initialized because handle creation
        // succeeded. This query has no pointer or lifetime arguments.
        let library_version = unsafe { sys::cublasLtGetVersion() };
        Ok(Self {
            inner: Arc::new(ProviderInner {
                context: context.clone(),
                handle,
                library_version,
                sticky_status: AtomicI32::new(0),
            }),
        })
    }

    pub(crate) const fn workspace_limit_bytes(&self) -> usize {
        HOPPER_WORKSPACE_LIMIT_BYTES
    }

    pub(crate) fn library_version(&self) -> usize {
        self.inner.library_version
    }

    /// Selects and freezes one BF16 algorithm for `spec`.
    pub(crate) fn plan_bf16_dense(
        &self,
        spec: Bf16DenseGemmSpec,
    ) -> Result<CublasLtBf16DensePlan, Bf16DenseGemmPlanError> {
        self.inner.require_healthy()?;
        self.inner.context.bind_to_thread()?;
        let dimensions = CheckedDimensions::new(spec)?;

        let matmul_desc = MatmulDesc::new(self.inner.clone())?;
        matmul_desc.set_transpose(
            sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
            false,
        )?;
        matmul_desc.set_transpose(
            sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
            true,
        )?;

        let a_layout = MatrixLayout::row_major(
            self.inner.clone(),
            dimensions.m,
            dimensions.k,
            dimensions.k_ld,
        )?;
        let weight_layout = MatrixLayout::row_major(
            self.inner.clone(),
            dimensions.n,
            dimensions.k,
            dimensions.k_ld,
        )?;
        let output_layout = MatrixLayout::row_major(
            self.inner.clone(),
            dimensions.m,
            dimensions.n,
            dimensions.n_ld,
        )?;

        let preference = MatmulPreference::new(self.inner.clone())?;
        preference.set_workspace_limit(HOPPER_WORKSPACE_LIMIT_BYTES)?;
        for attribute in [
            sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
            sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
            sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
            sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,
        ] {
            preference.set_minimum_alignment(attribute, TENSOR_ALIGNMENT_BYTES as u32)?;
        }

        // SAFETY: every descriptor belongs to this live provider, is fully
        // initialized, and remains owned until after the call returns.
        let heuristic = unsafe {
            result::get_matmul_algo_heuristic(
                self.inner.handle,
                matmul_desc.raw,
                a_layout.raw,
                weight_layout.raw,
                output_layout.raw,
                output_layout.raw,
                preference.raw,
            )?
        };
        if heuristic.workspaceSize > HOPPER_WORKSPACE_LIMIT_BYTES {
            return Err(Bf16DenseGemmPlanError::WorkspaceRequirementExceedsLimit {
                required: heuristic.workspaceSize,
                limit: HOPPER_WORKSPACE_LIMIT_BYTES,
            });
        }

        Ok(CublasLtBf16DensePlan {
            inner: Arc::new(CublasLtBf16DensePlanInner {
                provider: self.inner.clone(),
                spec,
                matmul_desc,
                a_layout,
                weight_layout,
                output_layout,
                algorithm: heuristic.algo,
                workspace_required_bytes: heuristic.workspaceSize,
                waves_count: heuristic.wavesCount,
            }),
        })
    }
}

struct ProviderInner {
    context: Arc<CudaContext>,
    handle: sys::cublasLtHandle_t,
    library_version: usize,
    sticky_status: AtomicI32,
}

// SAFETY: cuBLASLt handles may be called concurrently. Every call binds the
// owning cuda-oxide context first. The handle is immutable after construction
// and is retained by every plan until all submitted work is quiescent.
unsafe impl Send for ProviderInner {}
// SAFETY: the same invariants permit shared access from multiple host threads.
unsafe impl Sync for ProviderInner {}

impl ProviderInner {
    fn require_healthy(&self) -> Result<(), Bf16DenseGemmPlanError> {
        let status = self.sticky_status.load(Ordering::Acquire);
        if status == 0 {
            Ok(())
        } else {
            Err(Bf16DenseGemmPlanError::ProviderPoisoned { status })
        }
    }

    fn record_cleanup_result(&self, cleanup: Result<(), result::CublasError>) {
        if let Err(error) = cleanup {
            let _ = self.sticky_status.compare_exchange(
                0,
                cublas_status(error),
                Ordering::AcqRel,
                Ordering::Acquire,
            );
        }
    }

    fn run_cleanup(&self, cleanup: impl FnOnce() -> Result<(), result::CublasError>) {
        match bind_context_for_cleanup(&self.context) {
            Ok(()) => self.record_cleanup_result(cleanup()),
            Err(error) => self.context.record_err::<()>(Err(error)),
        }
    }
}

impl Drop for ProviderInner {
    fn drop(&mut self) {
        let handle = self.handle;
        // SAFETY: this is the final Arc, so no plan or completion can still
        // use the handle. cuBLASLt owns the pointed-to handle allocation.
        self.run_cleanup(|| unsafe { result::destroy_handle(handle) });
    }
}

struct MatmulDesc {
    provider: Arc<ProviderInner>,
    raw: sys::cublasLtMatmulDesc_t,
}

impl MatmulDesc {
    fn new(provider: Arc<ProviderInner>) -> Result<Self, result::CublasError> {
        let raw = result::create_matmul_desc(
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cudaDataType_t::CUDA_R_32F,
        )?;
        Ok(Self { provider, raw })
    }

    fn set_transpose(
        &self,
        attribute: sys::cublasLtMatmulDescAttributes_t,
        transpose: bool,
    ) -> Result<(), result::CublasError> {
        // cublasOperation_t is a 32-bit C enum: 0 is N and 1 is T.
        let operation = i32::from(transpose);
        // SAFETY: `raw` is live, `operation` has the ABI size expected by the
        // attribute, and cuBLASLt copies the value before returning.
        unsafe {
            result::set_matmul_desc_attribute(
                self.raw,
                attribute,
                (&operation as *const i32).cast(),
                size_of::<i32>(),
            )
        }
    }
}

impl Drop for MatmulDesc {
    fn drop(&mut self) {
        let raw = self.raw;
        // SAFETY: this wrapper uniquely owns the live descriptor.
        self.provider
            .run_cleanup(|| unsafe { result::destroy_matmul_desc(raw) });
    }
}

struct MatrixLayout {
    provider: Arc<ProviderInner>,
    raw: sys::cublasLtMatrixLayout_t,
}

impl MatrixLayout {
    fn row_major(
        provider: Arc<ProviderInner>,
        rows: u64,
        columns: u64,
        leading_dimension: i64,
    ) -> Result<Self, result::CublasError> {
        let raw = result::create_matrix_layout(
            sys::cudaDataType_t::CUDA_R_16BF,
            rows,
            columns,
            leading_dimension,
        )?;
        let layout = Self { provider, raw };
        let order = sys::cublasLtOrder_t::CUBLASLT_ORDER_ROW;
        // SAFETY: `layout.raw` is live, and `order` has the exact enum type
        // and size required by CUBLASLT_MATRIX_LAYOUT_ORDER.
        unsafe {
            result::set_matrix_layout_attribute(
                layout.raw,
                sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_ORDER,
                (&order as *const sys::cublasLtOrder_t).cast(),
                size_of::<sys::cublasLtOrder_t>(),
            )?;
        }
        Ok(layout)
    }
}

impl Drop for MatrixLayout {
    fn drop(&mut self) {
        let raw = self.raw;
        // SAFETY: this wrapper uniquely owns the live layout.
        self.provider
            .run_cleanup(|| unsafe { result::destroy_matrix_layout(raw) });
    }
}

struct MatmulPreference {
    provider: Arc<ProviderInner>,
    raw: sys::cublasLtMatmulPreference_t,
}

impl MatmulPreference {
    fn new(provider: Arc<ProviderInner>) -> Result<Self, result::CublasError> {
        let raw = result::create_matmul_pref()?;
        Ok(Self { provider, raw })
    }

    fn set_workspace_limit(&self, bytes: usize) -> Result<(), result::CublasError> {
        let bytes = bytes as u64;
        // SAFETY: `raw` is live and cuBLASLt copies the uint64 value before
        // returning.
        unsafe {
            result::set_matmul_pref_attribute(
                self.raw,
                sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                (&bytes as *const u64).cast(),
                size_of::<u64>(),
            )
        }
    }

    fn set_minimum_alignment(
        &self,
        attribute: sys::cublasLtMatmulPreferenceAttributes_t,
        bytes: u32,
    ) -> Result<(), result::CublasError> {
        // SAFETY: `raw` is live and alignment attributes consume a u32 value.
        unsafe {
            result::set_matmul_pref_attribute(
                self.raw,
                attribute,
                (&bytes as *const u32).cast(),
                size_of::<u32>(),
            )
        }
    }
}

impl Drop for MatmulPreference {
    fn drop(&mut self) {
        let raw = self.raw;
        // SAFETY: this wrapper uniquely owns the live preference.
        self.provider
            .run_cleanup(|| unsafe { result::destroy_matmul_pref(raw) });
    }
}

/// One immutable BF16 cuBLASLt algorithm and its exact layouts.
#[derive(Clone)]
pub(crate) struct CublasLtBf16DensePlan {
    inner: Arc<CublasLtBf16DensePlanInner>,
}

impl CublasLtBf16DensePlan {
    pub(crate) fn spec(&self) -> Bf16DenseGemmSpec {
        self.inner.spec
    }

    pub(crate) fn workspace_required_bytes(&self) -> usize {
        self.inner.workspace_required_bytes
    }

    pub(crate) const fn tensor_alignment_bytes(&self) -> u64 {
        TENSOR_ALIGNMENT_BYTES
    }

    pub(crate) const fn workspace_alignment_bytes(&self) -> u64 {
        WORKSPACE_ALIGNMENT_BYTES
    }

    pub(crate) fn estimated_waves_count(&self) -> f32 {
        self.inner.waves_count
    }

    /// Enqueues the frozen algorithm on the scope's caller-owned stream.
    pub(crate) fn enqueue_into(
        &self,
        scope: &mut CommandScope<'_>,
        operands: Bf16DenseGemmOperands,
    ) -> Result<(), Bf16DenseGemmEnqueueError> {
        let permit = scope.prepare_command()?;
        if let Err(error) = self.inner.provider.context.bind_to_thread() {
            scope.record_preflight_driver_failure(error);
            return Err(error.into());
        }
        let matmul_result = {
            let resolved = scope.resolve_rrww(
                operands.activation(),
                operands.weight(),
                operands.output(),
                operands.workspace(),
            )?;
            self.validate_resolved(&resolved)?;
            let alpha = 1.0_f32;
            let beta = 0.0_f32;
            let output = mutable_device_ptr(resolved.third);
            // SAFETY: the immutable plan fixes compatible descriptors and one
            // heuristic algorithm. Exact spans, access modes, context,
            // alignment, workspace size, and stream identity were validated
            // above. The command scope retains every buffer and provider plan
            // until CUDA quiescence is confirmed.
            unsafe {
                result::matmul(
                    self.inner.provider.handle,
                    self.inner.matmul_desc.raw,
                    (&alpha as *const f32).cast(),
                    (&beta as *const f32).cast(),
                    const_device_ptr(resolved.first),
                    self.inner.a_layout.raw,
                    const_device_ptr(resolved.second),
                    self.inner.weight_layout.raw,
                    output.cast_const(),
                    self.inner.output_layout.raw,
                    output,
                    self.inner.output_layout.raw,
                    &self.inner.algorithm,
                    mutable_device_ptr(resolved.fourth),
                    self.inner.workspace_required_bytes,
                    resolved.stream.cu_stream() as sys::cudaStream_t,
                )
            }
        };

        match matmul_result {
            Ok(()) => {
                scope.record_external_submission(permit, self.inner.clone());
                Ok(())
            }
            Err(error) => {
                scope.record_failed_external_submission(
                    permit,
                    self.inner.clone(),
                    ExternalCommandError::new(CUBLASLT_PROVIDER, cublas_status(error)),
                );
                Err(error.into())
            }
        }
    }

    fn validate_resolved(
        &self,
        resolved: &ResolvedRrww<'_, bf16, bf16, bf16, u8>,
    ) -> Result<(), Bf16DenseGemmEnqueueError> {
        let spec = self.inner.spec;
        require_exact_len("A", resolved.first.len(), spec.a_numel())?;
        require_exact_len("W", resolved.second.len(), spec.weight_numel())?;
        require_exact_len("D", resolved.third.len(), spec.output_numel())?;
        if resolved.fourth.num_bytes() < self.inner.workspace_required_bytes {
            return Err(Bf16DenseGemmEnqueueError::WorkspaceTooSmall {
                required: self.inner.workspace_required_bytes,
                actual: resolved.fourth.num_bytes(),
            });
        }
        require_alignment("A", resolved.first.cu_deviceptr(), TENSOR_ALIGNMENT_BYTES)?;
        require_alignment("W", resolved.second.cu_deviceptr(), TENSOR_ALIGNMENT_BYTES)?;
        require_alignment("D", resolved.third.cu_deviceptr(), TENSOR_ALIGNMENT_BYTES)?;
        require_alignment(
            "workspace",
            resolved.fourth.cu_deviceptr(),
            WORKSPACE_ALIGNMENT_BYTES,
        )?;

        let stream_context = resolved.stream.context();
        if stream_context.cu_ctx() != self.inner.provider.context.cu_ctx() {
            return Err(Bf16DenseGemmEnqueueError::ContextMismatch {
                plan_device: self.inner.provider.context.ordinal(),
                stream_device: stream_context.ordinal(),
            });
        }
        let sticky_status = self.inner.provider.sticky_status.load(Ordering::Acquire);
        if sticky_status != 0 {
            return Err(Bf16DenseGemmEnqueueError::ProviderPoisoned {
                status: sticky_status,
            });
        }
        Ok(())
    }
}

struct CublasLtBf16DensePlanInner {
    provider: Arc<ProviderInner>,
    spec: Bf16DenseGemmSpec,
    matmul_desc: MatmulDesc,
    a_layout: MatrixLayout,
    weight_layout: MatrixLayout,
    output_layout: MatrixLayout,
    algorithm: sys::cublasLtMatmulAlgo_t,
    workspace_required_bytes: usize,
    waves_count: f32,
}

// SAFETY: the plan contains immutable cuBLASLt descriptors and an immutable
// algorithm. cuBLASLt permits concurrent calls, each call supplies its own
// stream and workspace, and CommandScope enforces unique mutable output and
// workspace leases. Drop binds the owning context before releasing resources.
unsafe impl Send for CublasLtBf16DensePlanInner {}
// SAFETY: sharing does not mutate descriptors or algorithm state.
unsafe impl Sync for CublasLtBf16DensePlanInner {}

struct CheckedDimensions {
    m: u64,
    n: u64,
    k: u64,
    n_ld: i64,
    k_ld: i64,
}

impl CheckedDimensions {
    fn new(spec: Bf16DenseGemmSpec) -> Result<Self, Bf16DenseGemmPlanError> {
        Ok(Self {
            m: checked_u64("M", spec.m())?,
            n: checked_u64("N", spec.n())?,
            k: checked_u64("K", spec.k())?,
            n_ld: checked_i64("N", spec.n())?,
            k_ld: checked_i64("K", spec.k())?,
        })
    }
}

fn checked_u64(name: &'static str, value: usize) -> Result<u64, Bf16DenseGemmPlanError> {
    u64::try_from(value).map_err(|_| Bf16DenseGemmPlanError::DimensionOutOfRange { name, value })
}

fn checked_i64(name: &'static str, value: usize) -> Result<i64, Bf16DenseGemmPlanError> {
    i64::try_from(value).map_err(|_| Bf16DenseGemmPlanError::DimensionOutOfRange { name, value })
}

fn require_exact_len(
    operand: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), Bf16DenseGemmEnqueueError> {
    if actual == expected {
        Ok(())
    } else {
        Err(Bf16DenseGemmEnqueueError::LengthMismatch {
            operand,
            expected,
            actual,
        })
    }
}

fn require_alignment(
    operand: &'static str,
    address: u64,
    alignment: u64,
) -> Result<(), Bf16DenseGemmEnqueueError> {
    if address.is_multiple_of(alignment) {
        Ok(())
    } else {
        Err(Bf16DenseGemmEnqueueError::MisalignedBuffer {
            operand,
            address,
            alignment,
        })
    }
}

fn const_device_ptr<T: DeviceCopy>(region: &DeviceRegion<T>) -> *const c_void {
    region.cu_deviceptr() as usize as *const c_void
}

fn mutable_device_ptr<T: DeviceCopy>(region: &mut ReadWriteDeviceRegion<T>) -> *mut c_void {
    region.cu_deviceptr() as usize as *mut c_void
}

fn cublas_status(error: result::CublasError) -> i32 {
    error.0 as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alignment_gate_distinguishes_tensor_and_workspace_requirements() {
        assert!(require_alignment("A", 0x1010, TENSOR_ALIGNMENT_BYTES).is_ok());
        assert!(require_alignment("workspace", 0x1100, WORKSPACE_ALIGNMENT_BYTES).is_ok());
        assert!(matches!(
            require_alignment("workspace", 0x1010, WORKSPACE_ALIGNMENT_BYTES),
            Err(Bf16DenseGemmEnqueueError::MisalignedBuffer {
                alignment: WORKSPACE_ALIGNMENT_BYTES,
                ..
            })
        ));
    }
}
