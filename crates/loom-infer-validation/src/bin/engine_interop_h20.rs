#![deny(unsafe_code)]

use half::bf16;
use loom_infer::{Bf16SingleDecodeSpec, SINGLE_DECODE_HEAD_DIM, single_decode_bf16_reference};
use loom_infer_cuda::attention::{Bf16SingleDecodeArgs, DecodeProvider};
use loom_infer_cuda::interop::{
    EngineInteropQueue, EngineSingleDecodeAlgorithm, EngineStreamHandoff,
};
use loom_infer_validation::comparison::{compare_bf16, compare_f32};
use loom_infer_validation::fixture::deterministic_bf16;
use std::error::Error;

const OUTPUT_MAX_ABS_LIMIT: f32 = 0.015_625;
const LSE_MAX_ABS_LIMIT: f32 = 0.01;

fn main() -> Result<(), Box<dyn Error>> {
    let context = cuda_core::CudaContext::new(0)?;
    let mut engine_stream = raw_engine::EngineStream::new(context.clone())?;
    let external_stream = engine_stream.external_lease()?;
    let mut queue = EngineInteropQueue::new(external_stream, 1)?;
    let provider = DecodeProvider::load(&context)?;

    let spec = Bf16SingleDecodeSpec::new(127, 16, 4, SINGLE_DECODE_HEAD_DIM)?;
    let plan = provider.plan_bf16(spec)?;
    let query_host = deterministic_bf16(spec.query_numel(), 0xe001);
    let key_host = deterministic_bf16(spec.kv_numel(), 0xe002);
    let value_host = deterministic_bf16(spec.kv_numel(), 0xe003);
    let mut expected_output = vec![bf16::ZERO; spec.output_numel()];
    let mut expected_lse = vec![0.0_f32; spec.lse_numel()];
    single_decode_bf16_reference(
        &query_host,
        &key_host,
        &value_host,
        &mut expected_output,
        &mut expected_lse,
        spec,
    )?;

    let query = raw_engine::EngineBuffer::from_host(&engine_stream, &query_host)?;
    let key = raw_engine::EngineBuffer::from_host(&engine_stream, &key_host)?;
    let value = raw_engine::EngineBuffer::from_host(&engine_stream, &value_host)?;
    // These asynchronous initializations are prior engine-stream work. The
    // adapter's pre-event orders them before the Loom launch.
    let output = raw_engine::EngineBuffer::<bf16>::zeroed(&engine_stream, spec.output_numel())?;
    let lse = raw_engine::EngineBuffer::<f32>::zeroed(&engine_stream, spec.lse_numel())?;
    let pointers = [
        query.pointer(),
        key.pointer(),
        value.pointer(),
        output.pointer(),
        lse.pointer(),
    ];

    let mut bindings = queue.bindings(5)?;
    let (query, query_guard) = query.into_read_region();
    let (key, key_guard) = key.into_read_region();
    let (value, value_guard) = value.into_read_region();
    let (output, output_guard) = output.into_read_write_region();
    let (lse, lse_guard) = lse.into_read_write_region();
    let query_handle = bindings.bind_read_region(query)?;
    let key_handle = bindings.bind_read_region(key)?;
    let value_handle = bindings.bind_read_region(value)?;
    let output_handle = bindings.bind_read_write_region(output)?;
    let lse_handle = bindings.bind_read_write_region(lse)?;

    let authority = engine_stream.take_authority([
        query_guard,
        key_guard,
        value_guard,
        output_guard,
        lse_guard,
    ])?;
    match raw_engine::EngineBuffer::<u8>::zeroed(&engine_stream, 1) {
        Err(raw_engine::EngineBufferError::SubmissionAuthorityUnavailable) => {}
        Err(error) => {
            return Err(format!("unexpected guarded stream submission error: {error}").into());
        }
        Ok(_) => return Err("engine submitted without stream authority".into()),
    }
    let external_bindings = raw_engine::couple_authority(bindings, authority)?;
    let decode_args = Bf16SingleDecodeArgs::new(
        query_handle,
        key_handle,
        value_handle,
        output_handle.write(),
        lse_handle.write(),
    );

    let submission = queue.enqueue_bf16_single_decode(&plan, external_bindings, decode_args)?;
    let (completion, authority) = submission.into_parts();
    if completion.submitted() != 1 {
        return Err("engine interop completion covered the wrong command count".into());
    }
    let trace = completion.trace().clone();
    if trace.provider() != "loom-infer-cuda"
        || trace.operator() != "bf16_single_decode"
        || trace.algorithm() != EngineSingleDecodeAlgorithm::Direct
        || trace.stream_handoff() != EngineStreamHandoff::ExternalEventBridge
        || trace.memory().external_regions() != 5
        || trace.memory().device_buffers() != 0
        || trace.buffer_addresses() != pointers
        || trace.adapter_device_to_device_copies() != 0
        || !trace.is_adapter_zero_copy()
    {
        return Err(format!("engine interop returned an invalid provider trace: {trace:?}").into());
    }

    let outcome = completion.wait()?;
    if outcome.trace() != &trace {
        return Err("engine interop trace changed across completion".into());
    }
    let external_bindings = outcome.rejoin(authority)?;
    let submission = queue.enqueue_bf16_single_decode(&plan, external_bindings, decode_args)?;
    let (completion, authority) = submission.into_parts();
    if completion.submitted() != 1 || completion.trace() != &trace {
        return Err("rejoined engine submission changed its command trace".into());
    }

    // These engine-owned D2H reads are enqueued before host-side Loom
    // completion. The completion still owns checked bindings and independent
    // allocation leases. Returned authority supplies separate engine guards.
    let output_readback = raw_engine::PendingReadback::enqueue_after_command(
        authority.authority(),
        pointers[3],
        spec.output_numel(),
    )?;
    let lse_readback = raw_engine::PendingReadback::enqueue_after_command(
        authority.authority(),
        pointers[4],
        spec.lse_numel(),
    )?;
    let outcome = completion.wait()?;
    let (authority, second_trace) = outcome.release(authority)?;
    if second_trace != trace {
        return Err("engine interop trace changed while releasing authority".into());
    }
    engine_stream.return_authority(authority)?;

    // The readbacks retain their own allocation leases. Drop Loom's opaque
    // bindings and the engine storage guards before waiting to prove that the
    // pending DMA does not depend on either owner for allocation lifetime.
    let actual_output = output_readback.wait()?;
    let actual_lse = lse_readback.wait()?;
    let output_comparison = compare_bf16(&actual_output, &expected_output, "interop BF16")?;
    let lse_comparison = compare_f32(&actual_lse, &expected_lse, "interop F32 LSE")?;
    if output_comparison.max_abs > OUTPUT_MAX_ABS_LIMIT {
        return Err(format!(
            "engine interop output max abs {:.9e} exceeds {:.9e}",
            output_comparison.max_abs, OUTPUT_MAX_ABS_LIMIT
        )
        .into());
    }
    if lse_comparison.max_abs > LSE_MAX_ABS_LIMIT {
        return Err(format!(
            "engine interop LSE max abs {:.9e} exceeds {:.9e}",
            lse_comparison.max_abs, LSE_MAX_ABS_LIMIT
        )
        .into());
    }

    println!(
        "gate=engine_interop_h20 case=single_decode status=pass provider={} operator={} \
         algorithm=direct stream_handoff=external_event_bridge external_regions={} \
         adapter_zero_copy={} adapter_d2d_copies={} pointers_unchanged=true \
         authority_returned_before_completion=true submission_guard_enforced=true \
         authority_rejoined=true logical_calls=2 post_wait_output_read=true \
         boundary=simulated_engine kv_len={} \
         output_max_abs={:.9e} lse_max_abs={:.9e}",
        trace.provider(),
        trace.operator(),
        trace.memory().external_regions(),
        trace.is_adapter_zero_copy(),
        trace.adapter_device_to_device_copies(),
        spec.kv_len(),
        output_comparison.max_abs,
        lse_comparison.max_abs,
    );
    Ok(())
}

// This private harness simulates an engine that creates allocations and a
// stream before Loom sees them. All raw-pointer construction is confined here.
#[allow(unsafe_code)]
mod raw_engine {
    use cuda_core::sys::CUdeviceptr;
    use cuda_core::{
        CudaContext, CudaStream, DeviceBuffer, DeviceCopy, DriverError, PinnedHostBuffer,
    };
    use loom_infer_cuda::command::CheckedBindings;
    use loom_infer_cuda::interop::{
        EngineExternalBindings, EngineExternalBindingsError, ExternalCudaStream,
        ExternalCudaStreamError,
    };
    use loom_infer_cuda::memory::{ReadDeviceRegion, ReadWriteDeviceRegion};
    use std::any::Any;
    use std::error::Error;
    use std::fmt::{self, Display, Formatter};
    use std::mem::size_of;
    use std::sync::Arc;

    pub struct EngineStream {
        raw: Arc<CudaStream>,
        submission_guard: Option<StreamSubmissionGuard>,
    }

    struct StreamSubmissionGuard;

    #[derive(Clone, Copy)]
    enum StorageAccess {
        Read,
        ReadWrite,
    }

    pub struct EngineStorageGuard {
        allocation: Arc<dyn Any + Send + Sync>,
        pointer: CUdeviceptr,
        num_bytes: usize,
        access: StorageAccess,
    }

    pub struct EngineAuthority {
        stream: Arc<CudaStream>,
        stream_guard: StreamSubmissionGuard,
        storage: [EngineStorageGuard; 5],
    }

    pub fn couple_authority(
        bindings: CheckedBindings,
        authority: EngineAuthority,
    ) -> Result<EngineExternalBindings<EngineAuthority>, EngineExternalBindingsError<EngineAuthority>>
    {
        // SAFETY: the simulated engine consumed its sole stream token and all
        // five storage guards. Their ordered spans and access modes exactly
        // match the checked regions in `bindings`.
        unsafe { EngineExternalBindings::assume_engine_authority(bindings, authority) }
    }

    impl EngineStream {
        pub fn new(context: Arc<CudaContext>) -> Result<Self, DriverError> {
            Ok(Self {
                raw: context.new_stream()?,
                submission_guard: Some(StreamSubmissionGuard),
            })
        }

        pub fn external_lease(&self) -> Result<ExternalCudaStream, ExternalCudaStreamError> {
            // SAFETY: `self.raw` is the live stream and its Arc clone is the
            // explicit lease retained by ExternalCudaStream.
            unsafe {
                ExternalCudaStream::from_raw_parts(
                    self.raw.cu_stream(),
                    self.raw.context().clone(),
                    self.raw.clone(),
                )
            }
        }

        fn submission_stream(&self) -> Result<&Arc<CudaStream>, EngineBufferError> {
            self.submission_guard
                .as_ref()
                .ok_or(EngineBufferError::SubmissionAuthorityUnavailable)?;
            Ok(&self.raw)
        }

        pub fn take_authority(
            &mut self,
            storage: [EngineStorageGuard; 5],
        ) -> Result<EngineAuthority, &'static str> {
            let stream_guard = self
                .submission_guard
                .take()
                .ok_or("engine stream submission authority is already in flight")?;
            Ok(EngineAuthority {
                stream: self.raw.clone(),
                stream_guard,
                storage,
            })
        }

        pub fn return_authority(&mut self, authority: EngineAuthority) -> Result<(), &'static str> {
            if self.submission_guard.is_some() || !Arc::ptr_eq(&self.raw, &authority.stream) {
                return Err("engine stream authority does not belong to this stream");
            }
            self.submission_guard = Some(authority.stream_guard);
            drop(authority.storage);
            Ok(())
        }
    }

    pub struct EngineBuffer<T: DeviceCopy + Send + Sync + 'static> {
        allocation: Arc<DeviceBuffer<T>>,
    }

    #[derive(Debug)]
    pub enum EngineBufferError {
        SubmissionAuthorityUnavailable,
        Driver(DriverError),
    }

    impl Display for EngineBufferError {
        fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
            match self {
                Self::SubmissionAuthorityUnavailable => {
                    formatter.write_str("engine stream submission authority is in flight")
                }
                Self::Driver(error) => Display::fmt(error, formatter),
            }
        }
    }

    impl Error for EngineBufferError {
        fn source(&self) -> Option<&(dyn Error + 'static)> {
            match self {
                Self::SubmissionAuthorityUnavailable => None,
                Self::Driver(error) => Some(error),
            }
        }
    }

    impl From<DriverError> for EngineBufferError {
        fn from(error: DriverError) -> Self {
            Self::Driver(error)
        }
    }

    impl<T: DeviceCopy + Send + Sync + 'static> EngineBuffer<T> {
        pub fn from_host(stream: &EngineStream, data: &[T]) -> Result<Self, EngineBufferError> {
            let stream = stream.submission_stream()?;
            Ok(Self {
                allocation: Arc::new(DeviceBuffer::from_host(stream, data)?),
            })
        }

        pub fn zeroed(stream: &EngineStream, len: usize) -> Result<Self, EngineBufferError> {
            let stream = stream.submission_stream()?;
            Ok(Self {
                allocation: Arc::new(DeviceBuffer::zeroed(stream, len)?),
            })
        }

        pub fn pointer(&self) -> CUdeviceptr {
            self.allocation.cu_deviceptr()
        }

        pub fn into_read_region(self) -> (ReadDeviceRegion<T>, EngineStorageGuard) {
            let pointer = self.allocation.cu_deviceptr();
            let len = self.allocation.len();
            let context = self.allocation.context().clone();
            let region_lease: Arc<dyn Any + Send + Sync> = self.allocation.clone();
            let storage_guard = EngineStorageGuard {
                allocation: self.allocation,
                pointer,
                num_bytes: len * size_of::<T>(),
                access: StorageAccess::Read,
            };
            // SAFETY: the consumed wrapper transfers its only public access to
            // the returned region and authority guard. The region has an
            // independent allocation lease.
            let region = unsafe {
                ReadDeviceRegion::from_external_parts(pointer, len, context, region_lease)
                    .expect("a cuda-core allocation is a valid typed external region")
            };
            (region, storage_guard)
        }

        pub fn into_read_write_region(self) -> (ReadWriteDeviceRegion<T>, EngineStorageGuard) {
            let pointer = self.allocation.cu_deviceptr();
            let len = self.allocation.len();
            let context = self.allocation.context().clone();
            let region_lease: Arc<dyn Any + Send + Sync> = self.allocation.clone();
            let storage_guard = EngineStorageGuard {
                allocation: self.allocation,
                pointer,
                num_bytes: len * size_of::<T>(),
                access: StorageAccess::ReadWrite,
            };
            // SAFETY: consuming the wrapper transfers exclusive public access
            // to the returned region and authority guard. The region has an
            // independent allocation lease.
            let region = unsafe {
                ReadWriteDeviceRegion::from_external_parts(pointer, len, context, region_lease)
                    .expect("a cuda-core allocation is a valid typed external region")
            };
            (region, storage_guard)
        }
    }

    pub struct PendingReadback<T: DeviceCopy + Send + Sync + 'static> {
        host: Option<PinnedHostBuffer<T>>,
        stream: Arc<CudaStream>,
        _source_lease: Arc<dyn Any + Send + Sync>,
        complete: bool,
    }

    impl<T: DeviceCopy + Send + Sync + 'static> PendingReadback<T> {
        pub fn enqueue_after_command(
            authority: &EngineAuthority,
            source: CUdeviceptr,
            len: usize,
        ) -> Result<Self, DriverError> {
            let num_bytes = len
                .checked_mul(size_of::<T>())
                .expect("engine readback byte extent fits usize");
            let source_lease = authority
                .storage
                .iter()
                .find(|guard| {
                    matches!(guard.access, StorageAccess::Read | StorageAccess::ReadWrite)
                        && guard.pointer == source
                        && guard.num_bytes >= num_bytes
                })
                .expect("engine authority covers the readback source")
                .allocation
                .clone();
            let mut host = PinnedHostBuffer::zeroed(authority.stream.context(), len)?;
            authority.stream.context().bind_to_thread()?;
            // SAFETY: `_source_lease` keeps the source allocation live.
            // PinnedHostBuffer owns exactly `len` writable T values, and the
            // engine stream orders this read after Loom's post-event wait.
            let enqueue_result = unsafe {
                cuda_core::memory::memcpy_dtoh_async(
                    host.as_mut_ptr(),
                    source,
                    host.num_bytes(),
                    authority.stream.cu_stream(),
                )
            };
            if let Err(error) = enqueue_result {
                if let Some(settle_error) = synchronize_stream_or_abort(&authority.stream) {
                    authority
                        .stream
                        .context()
                        .record_err::<()>(Err(settle_error));
                }
                return Err(error);
            }
            Ok(Self {
                host: Some(host),
                stream: authority.stream.clone(),
                _source_lease: source_lease,
                complete: false,
            })
        }

        pub fn wait(mut self) -> Result<Vec<T>, DriverError> {
            let synchronize_error = synchronize_stream_or_abort(&self.stream);
            self.complete = true;
            match synchronize_error {
                None => Ok(self
                    .host
                    .take()
                    .expect("live readback retains pinned storage")
                    .as_slice()
                    .to_vec()),
                Some(error) => Err(error),
            }
        }
    }

    impl<T: DeviceCopy + Send + Sync + 'static> Drop for PendingReadback<T> {
        fn drop(&mut self) {
            if self.complete {
                return;
            }
            if let Some(error) = synchronize_stream_or_abort(&self.stream) {
                self.stream.context().record_err::<()>(Err(error));
            }
            self.complete = true;
        }
    }

    fn synchronize_stream_or_abort(stream: &CudaStream) -> Option<DriverError> {
        match stream.synchronize() {
            Ok(()) => None,
            Err(stream_error) => match stream.context().synchronize() {
                Ok(()) => Some(stream_error),
                Err(context_error) => {
                    eprintln!(
                        "engine gate cannot confirm CUDA quiescence after stream and context \
                         synchronization failed; aborting to preserve pinned-memory safety: \
                         stream={stream_error}; context={context_error}"
                    );
                    std::process::abort()
                }
            },
        }
    }
}
