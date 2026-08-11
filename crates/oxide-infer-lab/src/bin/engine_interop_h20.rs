#![deny(unsafe_code)]

use half::bf16;
use oxide_infer::{
    Bf16PagedBatchDecodeSpec, Bf16SingleDecodeSpec, ContractError, PagedKvLayout,
    SINGLE_DECODE_HEAD_DIM, paged_batch_decode_bf16_reference, single_decode_bf16_reference,
};
use oxide_infer_cuda::attention::{Bf16SingleDecodeArgs, DecodeProvider};
use oxide_infer_cuda::interop::{
    EngineAlgorithm, EngineCommand, EngineCommandFailure, EngineEnqueueCause, EngineExecutionShape,
    EngineInteropQueue, EngineOperator, EngineStreamHandoff,
};
use oxide_infer_lab::comparison::{compare_bf16, compare_f32};
use oxide_infer_lab::fixture::deterministic_bf16;
use std::error::Error;

const OUTPUT_MAX_ABS_LIMIT: f32 = 0.015_625;
const LSE_MAX_ABS_LIMIT: f32 = 0.01;

fn main() -> Result<(), Box<dyn Error>> {
    let context = cuda_core::CudaContext::new(0)?;
    let mut engine_stream = raw_engine::EngineStream::new(context.clone())?;
    let external_stream = engine_stream.external_lease()?;
    let mut queue = EngineInteropQueue::new(external_stream, 3, 2)?;
    let provider = DecodeProvider::load(&context)?;

    let single_spec = Bf16SingleDecodeSpec::new(127, 16, 4, SINGLE_DECODE_HEAD_DIM)?;
    let single_plan = provider.plan_bf16(single_spec)?;
    let query_host = deterministic_bf16(single_spec.query_numel(), 0xe001);
    let key_host = deterministic_bf16(single_spec.kv_numel(), 0xe002);
    let value_host = deterministic_bf16(single_spec.kv_numel(), 0xe003);
    let mut expected_output = vec![bf16::ZERO; single_spec.output_numel()];
    let mut expected_lse = vec![0.0_f32; single_spec.lse_numel()];
    single_decode_bf16_reference(
        &query_host,
        &key_host,
        &value_host,
        &mut expected_output,
        &mut expected_lse,
        single_spec,
    )?;

    let query = raw_engine::EngineBuffer::from_host(&engine_stream, &query_host)?;
    let key = raw_engine::EngineBuffer::from_host(&engine_stream, &key_host)?;
    let value = raw_engine::EngineBuffer::from_host(&engine_stream, &value_host)?;
    // These asynchronous initializations are prior engine-stream work. The
    // adapter's pre-event orders them before the Oxide launch.
    let output =
        raw_engine::EngineBuffer::<bf16>::zeroed(&engine_stream, single_spec.output_numel())?;
    let lse = raw_engine::EngineBuffer::<f32>::zeroed(&engine_stream, single_spec.lse_numel())?;
    let single_pointers = [
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

    let authority = engine_stream.take_authority(vec![
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

    let submission = queue.enqueue(
        EngineCommand::Bf16SingleDecode {
            plan: &single_plan,
            args: decode_args,
        },
        external_bindings,
    )?;
    let (completion, authority) = submission.into_parts();
    if completion.submitted() != 1 {
        return Err("engine interop completion covered the wrong command count".into());
    }
    let single_trace = completion.trace().clone();
    if single_trace.provider() != "oxide-infer-cuda"
        || single_trace.operator() != EngineOperator::Bf16SingleDecode
        || single_trace.algorithm() != EngineAlgorithm::SingleDecodeDirect
        || single_trace.shape()
            != (EngineExecutionShape::SingleDecode {
                kv_len: single_spec.kv_len(),
                query_heads: single_spec.num_query_heads(),
                kv_heads: single_spec.num_kv_heads(),
                head_dim: single_spec.head_dim(),
            })
        || single_trace.paged_kv_layout().is_some()
        || single_trace.stream_handoff() != EngineStreamHandoff::ExternalEventBridge
        || single_trace.memory().external_regions() != 5
        || single_trace.memory().device_buffers() != 0
        || single_trace.buffer_addresses() != single_pointers
        || single_trace.adapter_device_to_device_copies() != 0
        || !single_trace.is_adapter_zero_copy()
    {
        return Err(
            format!("engine interop returned an invalid single trace: {single_trace:?}").into(),
        );
    }

    let mut authority = authority;
    let single_output_readback = raw_engine::PendingReadback::enqueue_after_command(
        &mut authority,
        single_pointers[3],
        single_spec.output_numel(),
    )?;
    let single_lse_readback = raw_engine::PendingReadback::enqueue_after_command(
        &mut authority,
        single_pointers[4],
        single_spec.lse_numel(),
    )?;
    engine_stream.return_authority(authority)?;

    let paged_spec = Bf16PagedBatchDecodeSpec::new(2, 4, 12, 2, 128, 16, PagedKvLayout::Hnd)?;
    let paged_plan = provider.plan_bf16_paged_batch(paged_spec)?;
    let page_indptr_host = [0_i32, 2, 4];
    let page_indices_host = [0_i32, 3, 1, 2];
    let last_page_len_host = [9_i32, 16];
    let paged_query_host = deterministic_bf16(paged_spec.query_numel(), 0xe101);
    let paged_key_host = deterministic_bf16(paged_spec.kv_pages_numel(), 0xe102);
    let paged_value_host = deterministic_bf16(paged_spec.kv_pages_numel(), 0xe103);
    let mut paged_expected_output = vec![bf16::ZERO; paged_spec.output_numel()];
    let mut paged_expected_lse = vec![0.0_f32; paged_spec.lse_numel()];
    paged_batch_decode_bf16_reference(
        &paged_query_host,
        &paged_key_host,
        &paged_value_host,
        &page_indptr_host,
        &page_indices_host,
        &last_page_len_host,
        &mut paged_expected_output,
        &mut paged_expected_lse,
        paged_spec,
    )?;
    let paged = raw_engine::PagedBuffers::new(
        &engine_stream,
        &paged_plan,
        paged_spec,
        &paged_query_host,
        &paged_key_host,
        &paged_value_host,
        &page_indptr_host,
        &page_indices_host,
        &last_page_len_host,
    )?;
    let paged_pointers = paged.pointers();
    let (paged_bindings, paged_args, paged_guards) = paged.bind(&queue)?;
    let paged_authority = engine_stream.take_authority(paged_guards)?;
    let paged_external = raw_engine::couple_authority(paged_bindings, paged_authority)?;
    let paged_submission = queue.enqueue(
        EngineCommand::Bf16PagedBatchDecode {
            plan: &paged_plan,
            args: paged_args,
        },
        paged_external,
    )?;
    let (paged_completion, mut paged_authority) = paged_submission.into_parts();
    let paged_trace = paged_completion.trace().clone();
    if paged_completion.submitted() != 3
        || paged_trace.operator() != EngineOperator::Bf16PagedBatchDecode
        || paged_trace.algorithm() != EngineAlgorithm::PagedBatchDecodeTokenParallel8
        || paged_trace.shape()
            != (EngineExecutionShape::PagedBatchDecode {
                batch_size: paged_spec.batch_size(),
                max_num_pages: paged_spec.max_num_pages(),
                query_heads: paged_spec.num_query_heads(),
                kv_heads: paged_spec.num_kv_heads(),
                head_dim: paged_spec.head_dim(),
                page_size: paged_spec.page_size(),
            })
        || paged_trace.paged_kv_layout() != Some(PagedKvLayout::Hnd)
        || paged_trace.buffer_addresses() != paged_pointers
        || paged_trace.memory().external_regions() != 9
        || !paged_trace.is_adapter_zero_copy()
    {
        return Err(
            format!("engine interop returned an invalid paged trace: {paged_trace:?}").into(),
        );
    }
    let paged_output_readback = raw_engine::PendingReadback::enqueue_after_command(
        &mut paged_authority,
        paged_pointers[7],
        paged_spec.output_numel(),
    )?;
    let paged_lse_readback = raw_engine::PendingReadback::enqueue_after_command(
        &mut paged_authority,
        paged_pointers[8],
        paged_spec.lse_numel(),
    )?;
    engine_stream.return_authority(paged_authority)?;

    let capacity_probe = raw_engine::SingleBuffers::zeroed(&engine_stream, single_spec)?;
    let (probe_bindings, probe_args, probe_guards) = capacity_probe.bind(&queue)?;
    let probe_authority = engine_stream.take_authority(probe_guards)?;
    let probe_external = raw_engine::couple_authority(probe_bindings, probe_authority)?;
    let capacity_error = match queue.enqueue(
        EngineCommand::Bf16SingleDecode {
            plan: &single_plan,
            args: probe_args,
        },
        probe_external,
    ) {
        Ok(_) => return Err("third in-flight interop submission was accepted".into()),
        Err(error) => error,
    };
    if !matches!(
        capacity_error.cause(),
        EngineEnqueueCause::InFlightCapacityExceeded
    ) || !capacity_error.recovery_is_coupled()
    {
        return Err(format!("interop capacity returned the wrong error: {capacity_error}").into());
    }
    let (_, capacity_recovery) = capacity_error.into_parts();
    let probe_external = match capacity_recovery.into_coupled() {
        Ok(coupled) => coupled,
        Err(authority) => {
            engine_stream.return_authority(authority)?;
            return Err("interop capacity failure lost its checked bindings".into());
        }
    };
    let paged_wait_trace = paged_completion.wait()?;
    let probe_submission = queue.enqueue(
        EngineCommand::Bf16SingleDecode {
            plan: &single_plan,
            args: probe_args,
        },
        probe_external,
    )?;
    let (probe_completion, probe_authority) = probe_submission.into_parts();
    engine_stream.return_authority(probe_authority)?;
    let probe_trace = probe_completion.wait()?;
    let single_wait_trace = std::thread::spawn(move || completion.wait())
        .join()
        .map_err(|_| "engine completion worker thread panicked")??;
    if paged_wait_trace != paged_trace
        || probe_trace.operator() != EngineOperator::Bf16SingleDecode
        || single_wait_trace != single_trace
    {
        return Err("engine interop trace changed across reverse completion waits".into());
    }

    let actual_output = single_output_readback.wait()?;
    let actual_lse = single_lse_readback.wait()?;
    let output_comparison = compare_bf16(&actual_output, &expected_output, "interop single BF16")?;
    let lse_comparison = compare_f32(&actual_lse, &expected_lse, "interop single F32 LSE")?;
    let paged_actual_output = paged_output_readback.wait()?;
    let paged_actual_lse = paged_lse_readback.wait()?;
    let paged_output_comparison = compare_bf16(
        &paged_actual_output,
        &paged_expected_output,
        "interop paged BF16",
    )?;
    let paged_lse_comparison = compare_f32(
        &paged_actual_lse,
        &paged_expected_lse,
        "interop paged F32 LSE",
    )?;
    if output_comparison.max_abs > OUTPUT_MAX_ABS_LIMIT
        || paged_output_comparison.max_abs > OUTPUT_MAX_ABS_LIMIT
    {
        return Err(format!(
            "engine interop output max abs exceeds limit: single={:.9e}, paged={:.9e}, limit={:.9e}",
            output_comparison.max_abs, paged_output_comparison.max_abs, OUTPUT_MAX_ABS_LIMIT
        )
        .into());
    }
    if lse_comparison.max_abs > LSE_MAX_ABS_LIMIT
        || paged_lse_comparison.max_abs > LSE_MAX_ABS_LIMIT
    {
        return Err(format!(
            "engine interop LSE max abs exceeds limit: single={:.9e}, paged={:.9e}, limit={:.9e}",
            lse_comparison.max_abs, paged_lse_comparison.max_abs, LSE_MAX_ABS_LIMIT
        )
        .into());
    }

    run_invalid_paged_case(&mut engine_stream, &mut queue, &paged_plan)?;
    run_command_capacity_case(&mut engine_stream, &paged_plan)?;

    println!(
        "gate=engine_interop_h20 case=unified_decode status=pass provider={} \
         operators=single_decode,paged_batch_decode paged_layout=HND paged_group_size=6 \
         stream_handoff=external_event_bridge external_regions=5,9 \
         adapter_zero_copy=true adapter_d2d_copies=0 pointers_unchanged=true \
         two_in_flight=true reverse_wait=true cross_thread_completion=true capacity_recovered=true \
         command_capacity_preflight=true \
         authority_returned_before_completion=true submission_guard_enforced=true \
         typed_invalid_metadata=true queue_reusable_after_rejection=true \
         post_wait_output_read=true boundary=simulated_engine \
         single_output_max_abs={:.9e} single_lse_max_abs={:.9e} \
         paged_output_max_abs={:.9e} paged_lse_max_abs={:.9e}",
        single_trace.provider(),
        output_comparison.max_abs,
        lse_comparison.max_abs,
        paged_output_comparison.max_abs,
        paged_lse_comparison.max_abs,
    );
    Ok(())
}

fn run_command_capacity_case(
    engine_stream: &mut raw_engine::EngineStream,
    plan: &oxide_infer_cuda::attention::Bf16PagedBatchDecodePlan,
) -> Result<(), Box<dyn Error>> {
    let external_stream = engine_stream.external_lease()?;
    let mut queue = EngineInteropQueue::new(external_stream, 1, 1)?;
    let spec = plan.spec();
    let buffers = raw_engine::PagedBuffers::new(
        engine_stream,
        plan,
        spec,
        &vec![bf16::ZERO; spec.query_numel()],
        &vec![bf16::ZERO; spec.kv_pages_numel()],
        &vec![bf16::ZERO; spec.kv_pages_numel()],
        &[0_i32, 2, 4],
        &[0_i32, 3, 1, 2],
        &[9_i32, 16],
    )?;
    let (bindings, args, guards) = buffers.bind(&queue)?;
    let authority = engine_stream.take_authority(guards)?;
    let external = raw_engine::couple_authority(bindings, authority)?;
    let error = match queue.enqueue(EngineCommand::Bf16PagedBatchDecode { plan, args }, external) {
        Ok(_) => return Err("paged interop ignored static command capacity".into()),
        Err(error) => error,
    };
    if !matches!(
        error.cause(),
        EngineEnqueueCause::Command(
            oxide_infer_cuda::command::CommandError::CommandCapacityExceeded { capacity: 1 }
        )
    ) || !error.recovery_is_coupled()
    {
        return Err(format!("paged command capacity returned the wrong error: {error}").into());
    }
    engine_stream.return_authority(error.into_parts().1.into_authority())?;
    Ok(())
}

fn run_invalid_paged_case(
    engine_stream: &mut raw_engine::EngineStream,
    queue: &mut EngineInteropQueue,
    plan: &oxide_infer_cuda::attention::Bf16PagedBatchDecodePlan,
) -> Result<(), Box<dyn Error>> {
    let spec = plan.spec();
    let invalid_indices = [0_i32, 4, 1, 2];
    let buffers = raw_engine::PagedBuffers::new(
        engine_stream,
        plan,
        spec,
        &vec![bf16::ZERO; spec.query_numel()],
        &vec![bf16::ZERO; spec.kv_pages_numel()],
        &vec![bf16::ZERO; spec.kv_pages_numel()],
        &[0_i32, 2, 4],
        &invalid_indices,
        &[9_i32, 16],
    )?;
    let (bindings, args, guards) = buffers.bind(queue)?;
    let authority = engine_stream.take_authority(guards)?;
    let external = raw_engine::couple_authority(bindings, authority)?;
    let submission = queue.enqueue(EngineCommand::Bf16PagedBatchDecode { plan, args }, external)?;
    let (completion, authority) = submission.into_parts();
    engine_stream.return_authority(authority)?;
    match completion.wait() {
        Err(error)
            if matches!(
                error.cause(),
                EngineCommandFailure::DeviceRejected(ContractError::PageIndexOutOfRange {
                    position: 1,
                    index: 4,
                    max_num_pages: 4,
                })
            ) => {}
        Err(error) => return Err(format!("invalid paged metadata returned {error}").into()),
        Ok(_) => return Err("invalid paged metadata was accepted".into()),
    }

    let valid = raw_engine::PagedBuffers::new(
        engine_stream,
        plan,
        spec,
        &vec![bf16::ZERO; spec.query_numel()],
        &vec![bf16::ZERO; spec.kv_pages_numel()],
        &vec![bf16::ZERO; spec.kv_pages_numel()],
        &[0_i32, 2, 4],
        &[0_i32, 3, 1, 2],
        &[9_i32, 16],
    )?;
    let (bindings, args, guards) = valid.bind(queue)?;
    let authority = engine_stream.take_authority(guards)?;
    let external = raw_engine::couple_authority(bindings, authority)?;
    let submission = queue.enqueue(EngineCommand::Bf16PagedBatchDecode { plan, args }, external)?;
    let (completion, authority) = submission.into_parts();
    engine_stream.return_authority(authority)?;
    completion.wait()?;
    Ok(())
}

// This private module simulates an engine that creates allocations and a
// stream before Oxide sees them. This module contains all raw-pointer setup.
#[allow(unsafe_code)]
mod raw_engine {
    use cuda_core::sys::CUdeviceptr;
    use cuda_core::{
        CudaContext, CudaStream, DeviceBuffer, DeviceCopy, DriverError, PinnedHostBuffer,
    };
    use half::bf16;
    use oxide_infer::{Bf16PagedBatchDecodeSpec, Bf16SingleDecodeSpec};
    use oxide_infer_cuda::attention::{
        Bf16PagedBatchDecodeArgs, Bf16PagedBatchDecodePlan, Bf16SingleDecodeArgs,
    };
    use oxide_infer_cuda::command::CheckedBindings;
    use oxide_infer_cuda::interop::{
        EngineExternalBindings, EngineExternalBindingsError, EngineInteropQueue,
        ExternalCudaStream, ExternalCudaStreamError, StreamOrderedEngineAuthority,
    };
    use oxide_infer_cuda::memory::{ReadDeviceRegion, ReadWriteDeviceRegion};
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
        storage: Vec<EngineStorageGuard>,
    }

    // SAFETY: this private gate capability is linear and has no Clone or
    // storage/stream getter. Its only safe device operation is a readback
    // enqueued through the retained exact stream. Each region holds a separate
    // allocation lease, so dropping this value cannot invalidate Oxide work.
    unsafe impl StreamOrderedEngineAuthority for EngineAuthority {
        fn submission_stream(&self) -> cuda_core::sys::CUstream {
            self.stream.cu_stream()
        }
    }

    pub fn couple_authority(
        bindings: CheckedBindings,
        authority: EngineAuthority,
    ) -> Result<EngineExternalBindings<EngineAuthority>, EngineExternalBindingsError<EngineAuthority>>
    {
        // SAFETY: the simulated engine consumed its sole stream token and all
        // storage guards. Their ordered spans and access modes exactly
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
            storage: Vec<EngineStorageGuard>,
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

    pub struct SingleBuffers {
        query: EngineBuffer<bf16>,
        key: EngineBuffer<bf16>,
        value: EngineBuffer<bf16>,
        output: EngineBuffer<bf16>,
        lse: EngineBuffer<f32>,
    }

    impl SingleBuffers {
        pub fn zeroed(
            stream: &EngineStream,
            spec: Bf16SingleDecodeSpec,
        ) -> Result<Self, EngineBufferError> {
            Ok(Self {
                query: EngineBuffer::zeroed(stream, spec.query_numel())?,
                key: EngineBuffer::zeroed(stream, spec.kv_numel())?,
                value: EngineBuffer::zeroed(stream, spec.kv_numel())?,
                output: EngineBuffer::zeroed(stream, spec.output_numel())?,
                lse: EngineBuffer::zeroed(stream, spec.lse_numel())?,
            })
        }

        pub fn bind(
            self,
            queue: &EngineInteropQueue,
        ) -> Result<
            (
                CheckedBindings,
                Bf16SingleDecodeArgs,
                Vec<EngineStorageGuard>,
            ),
            Box<dyn Error>,
        > {
            let mut bindings = queue.bindings(5)?;
            let (query, query_guard) = self.query.into_read_region();
            let (key, key_guard) = self.key.into_read_region();
            let (value, value_guard) = self.value.into_read_region();
            let (output, output_guard) = self.output.into_read_write_region();
            let (lse, lse_guard) = self.lse.into_read_write_region();
            let query = bindings.bind_read_region(query)?;
            let key = bindings.bind_read_region(key)?;
            let value = bindings.bind_read_region(value)?;
            let output = bindings.bind_read_write_region(output)?;
            let lse = bindings.bind_read_write_region(lse)?;
            Ok((
                bindings,
                Bf16SingleDecodeArgs::new(query, key, value, output.write(), lse.write()),
                vec![query_guard, key_guard, value_guard, output_guard, lse_guard],
            ))
        }
    }

    pub struct PagedBuffers {
        query: EngineBuffer<bf16>,
        key_pages: EngineBuffer<bf16>,
        value_pages: EngineBuffer<bf16>,
        page_indptr: EngineBuffer<i32>,
        page_indices: EngineBuffer<i32>,
        last_page_len: EngineBuffer<i32>,
        metadata_status: EngineBuffer<i32>,
        output: EngineBuffer<bf16>,
        lse: EngineBuffer<f32>,
    }

    impl PagedBuffers {
        #[allow(clippy::too_many_arguments)]
        pub fn new(
            stream: &EngineStream,
            plan: &Bf16PagedBatchDecodePlan,
            spec: Bf16PagedBatchDecodeSpec,
            query: &[bf16],
            key_pages: &[bf16],
            value_pages: &[bf16],
            page_indptr: &[i32],
            page_indices: &[i32],
            last_page_len: &[i32],
        ) -> Result<Self, EngineBufferError> {
            Ok(Self {
                query: EngineBuffer::from_host(stream, query)?,
                key_pages: EngineBuffer::from_host(stream, key_pages)?,
                value_pages: EngineBuffer::from_host(stream, value_pages)?,
                page_indptr: EngineBuffer::from_host(stream, page_indptr)?,
                page_indices: EngineBuffer::from_host(stream, page_indices)?,
                last_page_len: EngineBuffer::from_host(stream, last_page_len)?,
                metadata_status: EngineBuffer::zeroed(
                    stream,
                    plan.metadata_status_required_numel(),
                )?,
                output: EngineBuffer::zeroed(stream, spec.output_numel())?,
                lse: EngineBuffer::zeroed(stream, spec.lse_numel())?,
            })
        }

        pub fn pointers(&self) -> [CUdeviceptr; 9] {
            [
                self.query.pointer(),
                self.key_pages.pointer(),
                self.value_pages.pointer(),
                self.page_indptr.pointer(),
                self.page_indices.pointer(),
                self.last_page_len.pointer(),
                self.metadata_status.pointer(),
                self.output.pointer(),
                self.lse.pointer(),
            ]
        }

        #[allow(clippy::type_complexity)]
        pub fn bind(
            self,
            queue: &EngineInteropQueue,
        ) -> Result<
            (
                CheckedBindings,
                Bf16PagedBatchDecodeArgs,
                Vec<EngineStorageGuard>,
            ),
            Box<dyn Error>,
        > {
            let mut bindings = queue.bindings(9)?;
            let (query, query_guard) = self.query.into_read_region();
            let (key_pages, key_guard) = self.key_pages.into_read_region();
            let (value_pages, value_guard) = self.value_pages.into_read_region();
            let (page_indptr, indptr_guard) = self.page_indptr.into_read_region();
            let (page_indices, indices_guard) = self.page_indices.into_read_region();
            let (last_page_len, last_page_guard) = self.last_page_len.into_read_region();
            let (metadata_status, status_guard) = self.metadata_status.into_read_write_region();
            let (output, output_guard) = self.output.into_read_write_region();
            let (lse, lse_guard) = self.lse.into_read_write_region();
            let query = bindings.bind_read_region(query)?;
            let key_pages = bindings.bind_read_region(key_pages)?;
            let value_pages = bindings.bind_read_region(value_pages)?;
            let page_indptr = bindings.bind_read_region(page_indptr)?;
            let page_indices = bindings.bind_read_region(page_indices)?;
            let last_page_len = bindings.bind_read_region(last_page_len)?;
            let metadata_status = bindings.bind_read_write_region(metadata_status)?;
            let output = bindings.bind_read_write_region(output)?;
            let lse = bindings.bind_read_write_region(lse)?;
            Ok((
                bindings,
                Bf16PagedBatchDecodeArgs::new(
                    query,
                    key_pages,
                    value_pages,
                    page_indptr,
                    page_indices,
                    last_page_len,
                    metadata_status,
                    output.write(),
                    lse.write(),
                ),
                vec![
                    query_guard,
                    key_guard,
                    value_guard,
                    indptr_guard,
                    indices_guard,
                    last_page_guard,
                    status_guard,
                    output_guard,
                    lse_guard,
                ],
            ))
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
            authority: &mut EngineAuthority,
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
            // engine stream orders this read after Oxide's post-event wait.
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
