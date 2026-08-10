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
    let engine_stream = raw_engine::EngineStream::new(context.clone())?;
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
    let query_handle = bindings.bind_read_region(query.into_read_region())?;
    let key_handle = bindings.bind_read_region(key.into_read_region())?;
    let value_handle = bindings.bind_read_region(value.into_read_region())?;
    let output_handle = bindings.bind_read_write_region(output.into_read_write_region())?;
    let lse_handle = bindings.bind_read_write_region(lse.into_read_write_region())?;

    let completion = queue.enqueue_bf16_single_decode(
        &plan,
        bindings,
        Bf16SingleDecodeArgs::new(
            query_handle,
            key_handle,
            value_handle,
            output_handle.write(),
            lse_handle.write(),
        ),
    )?;
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

    // These engine-owned D2H reads are enqueued before host-side Loom
    // completion. They can observe correct output only if the adapter placed
    // its post-event wait on the external stream. The opaque completion borrow
    // keeps the output bindings alive until both reads settle.
    let output_readback = raw_engine::PendingReadback::enqueue_after_command(
        &engine_stream,
        pointers[3],
        spec.output_numel(),
        &completion,
    )?;
    let lse_readback = raw_engine::PendingReadback::enqueue_after_command(
        &engine_stream,
        pointers[4],
        spec.lse_numel(),
        &completion,
    )?;
    let actual_output = output_readback.wait()?;
    let actual_lse = lse_readback.wait()?;
    let output_comparison = compare_bf16(&actual_output, &expected_output, "interop BF16")?;
    let lse_comparison = compare_f32(&actual_lse, &expected_lse, "interop F32 LSE")?;

    let outcome = completion.wait()?;
    if outcome.trace() != &trace {
        return Err("engine interop trace changed across completion".into());
    }
    let mut bindings = outcome.into_bindings();
    let output_region = bindings.take_read_write_region(output_handle)?;
    let lse_region = bindings.take_read_write_region(lse_handle)?;
    if output_region.cu_deviceptr() != pointers[3] || lse_region.cu_deviceptr() != pointers[4] {
        return Err("engine-owned output pointers changed across Loom execution".into());
    }
    drop((bindings, output_region, lse_region));
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
         post_wait_output_read=true boundary=simulated_engine kv_len={} \
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
    use loom_infer_cuda::interop::{
        EngineCommandCompletion, ExternalCudaStream, ExternalCudaStreamError,
    };
    use loom_infer_cuda::memory::{ReadDeviceRegion, ReadWriteDeviceRegion};
    use std::marker::PhantomData;
    use std::sync::Arc;

    pub struct EngineStream {
        raw: Arc<CudaStream>,
    }

    impl EngineStream {
        pub fn new(context: Arc<CudaContext>) -> Result<Self, DriverError> {
            Ok(Self {
                raw: context.new_stream()?,
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

        fn raw(&self) -> &Arc<CudaStream> {
            &self.raw
        }
    }

    pub struct EngineBuffer<T: DeviceCopy + Send + Sync + 'static> {
        allocation: Arc<DeviceBuffer<T>>,
    }

    impl<T: DeviceCopy + Send + Sync + 'static> EngineBuffer<T> {
        pub fn from_host(stream: &EngineStream, data: &[T]) -> Result<Self, DriverError> {
            Ok(Self {
                allocation: Arc::new(DeviceBuffer::from_host(stream.raw(), data)?),
            })
        }

        pub fn zeroed(stream: &EngineStream, len: usize) -> Result<Self, DriverError> {
            Ok(Self {
                allocation: Arc::new(DeviceBuffer::zeroed(stream.raw(), len)?),
            })
        }

        pub fn pointer(&self) -> CUdeviceptr {
            self.allocation.cu_deviceptr()
        }

        pub fn into_read_region(self) -> ReadDeviceRegion<T> {
            let pointer = self.allocation.cu_deviceptr();
            let len = self.allocation.len();
            let context = self.allocation.context().clone();
            // SAFETY: the consumed wrapper transfers its only public access to
            // the region, and the Arc allocation is the retained lease.
            unsafe {
                ReadDeviceRegion::from_external_parts(pointer, len, context, self.allocation)
                    .expect("a cuda-core allocation is a valid typed external region")
            }
        }

        pub fn into_read_write_region(self) -> ReadWriteDeviceRegion<T> {
            let pointer = self.allocation.cu_deviceptr();
            let len = self.allocation.len();
            let context = self.allocation.context().clone();
            // SAFETY: consuming the wrapper transfers exclusive public access
            // to the non-cloneable region. The Arc remains an opaque lease.
            unsafe {
                ReadWriteDeviceRegion::from_external_parts(pointer, len, context, self.allocation)
                    .expect("a cuda-core allocation is a valid typed external region")
            }
        }
    }

    pub struct PendingReadback<'source, T: DeviceCopy + Send + Sync + 'static> {
        host: Option<PinnedHostBuffer<T>>,
        stream: Arc<CudaStream>,
        source: PhantomData<&'source T>,
        complete: bool,
    }

    impl<'source, T: DeviceCopy + Send + Sync + 'static> PendingReadback<'source, T> {
        pub fn enqueue_after_command(
            stream: &EngineStream,
            source: CUdeviceptr,
            len: usize,
            _completion: &'source EngineCommandCompletion<'_>,
        ) -> Result<Self, DriverError> {
            let mut host = PinnedHostBuffer::zeroed(stream.raw().context(), len)?;
            stream.raw().context().bind_to_thread()?;
            // SAFETY: the lifetime ties the live source to this pending copy.
            // PinnedHostBuffer owns exactly `len` writable T values, and the
            // engine stream orders this read after Loom's post-event wait.
            let enqueue_result = unsafe {
                cuda_core::memory::memcpy_dtoh_async(
                    host.as_mut_ptr(),
                    source,
                    host.num_bytes(),
                    stream.raw().cu_stream(),
                )
            };
            if let Err(error) = enqueue_result {
                if let Some(settle_error) = synchronize_stream_or_abort(stream.raw()) {
                    stream.raw().context().record_err::<()>(Err(settle_error));
                }
                return Err(error);
            }
            Ok(Self {
                host: Some(host),
                stream: stream.raw().clone(),
                source: PhantomData,
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

    impl<T: DeviceCopy + Send + Sync + 'static> Drop for PendingReadback<'_, T> {
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
