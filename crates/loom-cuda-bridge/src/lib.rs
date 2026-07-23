//! Checked C entrypoints into Loom Kernels' safe Rust CUDA runtime.
//!
//! Framework adapters own tensor allocations and stream lifetime. This crate
//! converts that raw boundary into [`loom_cuda`] borrowed resources and keeps
//! panics, validation errors, and CUDA submission failures behind a stable C
//! status ABI.

#![deny(unsafe_op_in_unsafe_fn)]

/// Whether this build contains the checked CUDA bridge.
pub const fn compiled_with_cuda() -> bool {
    cfg!(feature = "cuda")
}

#[cfg(feature = "cuda")]
mod cuda {
    use half::{bf16, f16};
    use loom_cuda::runtime::{
        CudaDeviceRead, CudaDeviceWrite, CudaStreamHandle, CudaStreamRef, DeviceSlice,
        DeviceSliceMut,
    };
    use loom_cuda::{CudaBackend, CudaExecutorError};
    use loom_kernels::{AddRmsNormSpec, DType, GreedySampleLogprobsSpec, RmsNormDynamicFp8Spec};
    use std::cell::RefCell;
    use std::ffi::{c_char, c_int, c_void, CString};
    use std::mem::{align_of, size_of};
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::atomic::{AtomicU64, Ordering};

    const SUCCESS: c_int = 0;
    const INVALID_ARGUMENT: c_int = 1;
    const UNSUPPORTED: c_int = 2;
    const LAUNCH_ERROR: c_int = 3;
    const UNAVAILABLE: c_int = 4;

    static ADD_RMS_NORM_LAUNCHES: AtomicU64 = AtomicU64::new(0);
    static RMS_NORM_DYNAMIC_FP8_LAUNCHES: AtomicU64 = AtomicU64::new(0);
    static GREEDY_SAMPLE_LOGPROBS_LAUNCHES: AtomicU64 = AtomicU64::new(0);

    thread_local! {
        static LAST_ERROR: RefCell<CString> = RefCell::new(
            CString::new("no bridge error has been recorded")
                .expect("static bridge message contains no NUL")
        );
    }

    trait AddRmsNormScalar: Copy {
        const DTYPE: DType;

        fn launch<S, I, R, W>(
            backend: &CudaBackend<S>,
            input: &mut I,
            residual: &mut R,
            weight: &W,
            spec: AddRmsNormSpec,
        ) -> Result<(), CudaExecutorError>
        where
            S: CudaStreamHandle,
            I: CudaDeviceWrite<Self>,
            R: CudaDeviceWrite<Self>,
            W: CudaDeviceRead<Self>;
    }

    impl AddRmsNormScalar for f32 {
        const DTYPE: DType = DType::F32;

        fn launch<S, I, R, W>(
            backend: &CudaBackend<S>,
            input: &mut I,
            residual: &mut R,
            weight: &W,
            spec: AddRmsNormSpec,
        ) -> Result<(), CudaExecutorError>
        where
            S: CudaStreamHandle,
            I: CudaDeviceWrite<Self>,
            R: CudaDeviceWrite<Self>,
            W: CudaDeviceRead<Self>,
        {
            backend.add_rms_norm_f32(input, residual, weight, spec)
        }
    }

    impl AddRmsNormScalar for f16 {
        const DTYPE: DType = DType::F16;

        fn launch<S, I, R, W>(
            backend: &CudaBackend<S>,
            input: &mut I,
            residual: &mut R,
            weight: &W,
            spec: AddRmsNormSpec,
        ) -> Result<(), CudaExecutorError>
        where
            S: CudaStreamHandle,
            I: CudaDeviceWrite<Self>,
            R: CudaDeviceWrite<Self>,
            W: CudaDeviceRead<Self>,
        {
            backend.add_rms_norm_f16(input, residual, weight, spec)
        }
    }

    impl AddRmsNormScalar for bf16 {
        const DTYPE: DType = DType::Bf16;

        fn launch<S, I, R, W>(
            backend: &CudaBackend<S>,
            input: &mut I,
            residual: &mut R,
            weight: &W,
            spec: AddRmsNormSpec,
        ) -> Result<(), CudaExecutorError>
        where
            S: CudaStreamHandle,
            I: CudaDeviceWrite<Self>,
            R: CudaDeviceWrite<Self>,
            W: CudaDeviceRead<Self>,
        {
            backend.add_rms_norm_bf16(input, residual, weight, spec)
        }
    }

    trait RmsNormDynamicFp8Scalar: Copy {
        const DTYPE: DType;

        fn launch<S, I, W, O, Q>(
            backend: &CudaBackend<S>,
            input: &I,
            weight: &W,
            output: &mut O,
            scales: &mut Q,
            spec: RmsNormDynamicFp8Spec,
        ) -> Result<(), CudaExecutorError>
        where
            S: CudaStreamHandle,
            I: CudaDeviceRead<Self>,
            W: CudaDeviceRead<Self>,
            O: CudaDeviceWrite<u8>,
            Q: CudaDeviceWrite<f32>;
    }

    impl RmsNormDynamicFp8Scalar for f32 {
        const DTYPE: DType = DType::F32;

        fn launch<S, I, W, O, Q>(
            backend: &CudaBackend<S>,
            input: &I,
            weight: &W,
            output: &mut O,
            scales: &mut Q,
            spec: RmsNormDynamicFp8Spec,
        ) -> Result<(), CudaExecutorError>
        where
            S: CudaStreamHandle,
            I: CudaDeviceRead<Self>,
            W: CudaDeviceRead<Self>,
            O: CudaDeviceWrite<u8>,
            Q: CudaDeviceWrite<f32>,
        {
            backend.rms_norm_dynamic_fp8_f32(input, weight, output, scales, spec)
        }
    }

    impl RmsNormDynamicFp8Scalar for f16 {
        const DTYPE: DType = DType::F16;

        fn launch<S, I, W, O, Q>(
            backend: &CudaBackend<S>,
            input: &I,
            weight: &W,
            output: &mut O,
            scales: &mut Q,
            spec: RmsNormDynamicFp8Spec,
        ) -> Result<(), CudaExecutorError>
        where
            S: CudaStreamHandle,
            I: CudaDeviceRead<Self>,
            W: CudaDeviceRead<Self>,
            O: CudaDeviceWrite<u8>,
            Q: CudaDeviceWrite<f32>,
        {
            backend.rms_norm_dynamic_fp8_f16(input, weight, output, scales, spec)
        }
    }

    impl RmsNormDynamicFp8Scalar for bf16 {
        const DTYPE: DType = DType::Bf16;

        fn launch<S, I, W, O, Q>(
            backend: &CudaBackend<S>,
            input: &I,
            weight: &W,
            output: &mut O,
            scales: &mut Q,
            spec: RmsNormDynamicFp8Spec,
        ) -> Result<(), CudaExecutorError>
        where
            S: CudaStreamHandle,
            I: CudaDeviceRead<Self>,
            W: CudaDeviceRead<Self>,
            O: CudaDeviceWrite<u8>,
            Q: CudaDeviceWrite<f32>,
        {
            backend.rms_norm_dynamic_fp8_bf16(input, weight, output, scales, spec)
        }
    }

    trait GreedySampleLogprobsScalar: Copy {
        const DTYPE: DType;

        fn launch<S, I, T, L, R>(
            backend: &CudaBackend<S>,
            logits: &I,
            token_ids: &mut T,
            logprobs: &mut L,
            ranks: &mut R,
            spec: GreedySampleLogprobsSpec,
        ) -> Result<(), CudaExecutorError>
        where
            S: CudaStreamHandle,
            I: CudaDeviceRead<Self>,
            T: CudaDeviceWrite<i32>,
            L: CudaDeviceWrite<f32>,
            R: CudaDeviceWrite<i64>;
    }

    impl GreedySampleLogprobsScalar for f32 {
        const DTYPE: DType = DType::F32;

        fn launch<S, I, T, L, R>(
            backend: &CudaBackend<S>,
            logits: &I,
            token_ids: &mut T,
            logprobs: &mut L,
            ranks: &mut R,
            spec: GreedySampleLogprobsSpec,
        ) -> Result<(), CudaExecutorError>
        where
            S: CudaStreamHandle,
            I: CudaDeviceRead<Self>,
            T: CudaDeviceWrite<i32>,
            L: CudaDeviceWrite<f32>,
            R: CudaDeviceWrite<i64>,
        {
            backend.greedy_sample_logprobs_f32(logits, token_ids, logprobs, ranks, spec)
        }
    }

    impl GreedySampleLogprobsScalar for f16 {
        const DTYPE: DType = DType::F16;

        fn launch<S, I, T, L, R>(
            backend: &CudaBackend<S>,
            logits: &I,
            token_ids: &mut T,
            logprobs: &mut L,
            ranks: &mut R,
            spec: GreedySampleLogprobsSpec,
        ) -> Result<(), CudaExecutorError>
        where
            S: CudaStreamHandle,
            I: CudaDeviceRead<Self>,
            T: CudaDeviceWrite<i32>,
            L: CudaDeviceWrite<f32>,
            R: CudaDeviceWrite<i64>,
        {
            backend.greedy_sample_logprobs_f16(logits, token_ids, logprobs, ranks, spec)
        }
    }

    impl GreedySampleLogprobsScalar for bf16 {
        const DTYPE: DType = DType::Bf16;

        fn launch<S, I, T, L, R>(
            backend: &CudaBackend<S>,
            logits: &I,
            token_ids: &mut T,
            logprobs: &mut L,
            ranks: &mut R,
            spec: GreedySampleLogprobsSpec,
        ) -> Result<(), CudaExecutorError>
        where
            S: CudaStreamHandle,
            I: CudaDeviceRead<Self>,
            T: CudaDeviceWrite<i32>,
            L: CudaDeviceWrite<f32>,
            R: CudaDeviceWrite<i64>,
        {
            backend.greedy_sample_logprobs_bf16(logits, token_ids, logprobs, ranks, spec)
        }
    }

    #[derive(Clone, Copy)]
    struct ByteRange {
        start: usize,
        end: usize,
    }

    fn element_count(value: u64, name: &str) -> Result<usize, CudaExecutorError> {
        usize::try_from(value).map_err(|_| {
            CudaExecutorError::InvalidContract(format!("{name} element count exceeds the host ABI"))
        })
    }

    fn checked_byte_range<T>(
        pointer: *const T,
        elements: usize,
        name: &str,
    ) -> Result<ByteRange, CudaExecutorError> {
        if pointer.is_null() {
            return Err(CudaExecutorError::InvalidContract(format!(
                "{name} pointer is null"
            )));
        }
        if elements == 0 {
            return Err(CudaExecutorError::InvalidContract(format!(
                "{name} region is empty"
            )));
        }
        if !(pointer as usize).is_multiple_of(align_of::<T>()) {
            return Err(CudaExecutorError::InvalidContract(format!(
                "{name} pointer is not aligned to {} bytes",
                align_of::<T>()
            )));
        }
        let bytes = elements.checked_mul(size_of::<T>()).ok_or_else(|| {
            CudaExecutorError::InvalidContract(format!("{name} byte size overflows usize"))
        })?;
        let start = pointer as usize;
        let end = start.checked_add(bytes).ok_or_else(|| {
            CudaExecutorError::InvalidContract(format!("{name} address range overflows usize"))
        })?;
        Ok(ByteRange { start, end })
    }

    fn ranges_overlap(left: ByteRange, right: ByteRange) -> bool {
        left.start < right.end && right.start < left.end
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_add_rms_norm<T: AddRmsNormScalar>(
        input: *mut T,
        input_elements: u64,
        residual: *mut T,
        residual_elements: u64,
        weight: *const T,
        weight_elements: u64,
        rows: u32,
        hidden_size: u32,
        epsilon: f32,
        stream: *mut c_void,
    ) -> Result<(), CudaExecutorError> {
        let input_elements = element_count(input_elements, "Add+RMSNorm input")?;
        let residual_elements = element_count(residual_elements, "Add+RMSNorm residual")?;
        let weight_elements = element_count(weight_elements, "Add+RMSNorm weight")?;

        let input_range =
            checked_byte_range(input.cast_const(), input_elements, "Add+RMSNorm input")?;
        let residual_range = checked_byte_range(
            residual.cast_const(),
            residual_elements,
            "Add+RMSNorm residual",
        )?;
        let weight_range = checked_byte_range(weight, weight_elements, "Add+RMSNorm weight")?;
        if ranges_overlap(input_range, residual_range)
            || ranges_overlap(input_range, weight_range)
            || ranges_overlap(residual_range, weight_range)
        {
            return Err(CudaExecutorError::InvalidContract(
                "Add+RMSNorm input, residual, and weight regions must not overlap".into(),
            ));
        }

        let spec = AddRmsNormSpec::new(rows as usize, hidden_size as usize, epsilon, T::DTYPE)
            .map_err(|error| CudaExecutorError::InvalidContract(error.to_string()))?;
        let stream = unsafe { CudaStreamRef::from_raw(stream) };
        let backend = CudaBackend::from_stream(stream);
        let mut input = unsafe { DeviceSliceMut::from_raw_parts(input, input_elements) }?;
        let mut residual = unsafe { DeviceSliceMut::from_raw_parts(residual, residual_elements) }?;
        let weight = unsafe { DeviceSlice::from_raw_parts(weight, weight_elements) }?;

        T::launch(&backend, &mut input, &mut residual, &weight, spec)?;
        ADD_RMS_NORM_LAUNCHES.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_rms_norm_dynamic_fp8<T: RmsNormDynamicFp8Scalar>(
        input: *const T,
        input_elements: u64,
        weight: *const T,
        weight_elements: u64,
        output: *mut u8,
        output_elements: u64,
        scales: *mut f32,
        scale_elements: u64,
        rows: u32,
        hidden_size: u32,
        epsilon: f32,
        stream: *mut c_void,
    ) -> Result<(), CudaExecutorError> {
        let input_elements = element_count(input_elements, "RMSNorm+FP8 input")?;
        let weight_elements = element_count(weight_elements, "RMSNorm+FP8 weight")?;
        let output_elements = element_count(output_elements, "RMSNorm+FP8 output")?;
        let scale_elements = element_count(scale_elements, "RMSNorm+FP8 scales")?;

        let input_range = checked_byte_range(input, input_elements, "RMSNorm+FP8 input")?;
        let weight_range = checked_byte_range(weight, weight_elements, "RMSNorm+FP8 weight")?;
        let output_range =
            checked_byte_range(output.cast_const(), output_elements, "RMSNorm+FP8 output")?;
        let scale_range =
            checked_byte_range(scales.cast_const(), scale_elements, "RMSNorm+FP8 scales")?;
        if ranges_overlap(output_range, input_range)
            || ranges_overlap(output_range, weight_range)
            || ranges_overlap(output_range, scale_range)
            || ranges_overlap(scale_range, input_range)
            || ranges_overlap(scale_range, weight_range)
        {
            return Err(CudaExecutorError::InvalidContract(
                "RMSNorm+FP8 output and scales must not overlap input, weight, or each other"
                    .into(),
            ));
        }

        let spec =
            RmsNormDynamicFp8Spec::new(rows as usize, hidden_size as usize, epsilon, T::DTYPE)
                .map_err(|error| CudaExecutorError::InvalidContract(error.to_string()))?;
        let stream = unsafe { CudaStreamRef::from_raw(stream) };
        let backend = CudaBackend::from_stream(stream);
        let input = unsafe { DeviceSlice::from_raw_parts(input, input_elements) }?;
        let weight = unsafe { DeviceSlice::from_raw_parts(weight, weight_elements) }?;
        let mut output = unsafe { DeviceSliceMut::from_raw_parts(output, output_elements) }?;
        let mut scales = unsafe { DeviceSliceMut::from_raw_parts(scales, scale_elements) }?;

        T::launch(&backend, &input, &weight, &mut output, &mut scales, spec)?;
        RMS_NORM_DYNAMIC_FP8_LAUNCHES.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_greedy_sample_logprobs<T: GreedySampleLogprobsScalar>(
        logits: *const T,
        logits_elements: u64,
        token_ids: *mut i32,
        token_id_elements: u64,
        logprobs: *mut f32,
        logprob_elements: u64,
        ranks: *mut i64,
        rank_elements: u64,
        rows: u32,
        vocab_size: u32,
        stream: *mut c_void,
    ) -> Result<(), CudaExecutorError> {
        let logits_elements = element_count(logits_elements, "greedy-sampling logits")?;
        let token_id_elements = element_count(token_id_elements, "greedy-sampling token IDs")?;
        let logprob_elements = element_count(logprob_elements, "greedy-sampling logprobs")?;
        let rank_elements = element_count(rank_elements, "greedy-sampling ranks")?;

        let logits_range = checked_byte_range(logits, logits_elements, "greedy-sampling logits")?;
        let token_id_range = checked_byte_range(
            token_ids.cast_const(),
            token_id_elements,
            "greedy-sampling token IDs",
        )?;
        let logprob_range = checked_byte_range(
            logprobs.cast_const(),
            logprob_elements,
            "greedy-sampling logprobs",
        )?;
        let rank_range =
            checked_byte_range(ranks.cast_const(), rank_elements, "greedy-sampling ranks")?;
        if ranges_overlap(logits_range, token_id_range)
            || ranges_overlap(logits_range, logprob_range)
            || ranges_overlap(logits_range, rank_range)
            || ranges_overlap(token_id_range, logprob_range)
            || ranges_overlap(token_id_range, rank_range)
            || ranges_overlap(logprob_range, rank_range)
        {
            return Err(CudaExecutorError::InvalidContract(
                "greedy-sampling logits and output regions must not overlap".into(),
            ));
        }

        let spec = GreedySampleLogprobsSpec::new(rows as usize, vocab_size as usize, T::DTYPE)
            .map_err(|error| CudaExecutorError::InvalidContract(error.to_string()))?;
        let stream = unsafe { CudaStreamRef::from_raw(stream) };
        let backend = CudaBackend::from_stream(stream);
        let logits = unsafe { DeviceSlice::from_raw_parts(logits, logits_elements) }?;
        let mut token_ids =
            unsafe { DeviceSliceMut::from_raw_parts(token_ids, token_id_elements) }?;
        let mut logprobs = unsafe { DeviceSliceMut::from_raw_parts(logprobs, logprob_elements) }?;
        let mut ranks = unsafe { DeviceSliceMut::from_raw_parts(ranks, rank_elements) }?;

        T::launch(
            &backend,
            &logits,
            &mut token_ids,
            &mut logprobs,
            &mut ranks,
            spec,
        )?;
        GREEDY_SAMPLE_LOGPROBS_LAUNCHES.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn status_for_error(error: &CudaExecutorError) -> c_int {
        match error {
            CudaExecutorError::InvalidContract(_) => INVALID_ARGUMENT,
            CudaExecutorError::BackendUnavailable => UNAVAILABLE,
            CudaExecutorError::KernelSubmission { status, .. }
                if matches!(
                    *status,
                    INVALID_ARGUMENT | UNSUPPORTED | LAUNCH_ERROR | UNAVAILABLE
                ) =>
            {
                *status
            }
            CudaExecutorError::KernelSubmission { .. } => LAUNCH_ERROR,
        }
    }

    fn set_last_error(message: impl AsRef<str>) {
        let sanitized = message.as_ref().replace('\0', "\\0");
        let message = CString::new(sanitized).unwrap_or_else(|_| {
            CString::new("bridge error contained an invalid NUL")
                .expect("static bridge message contains no NUL")
        });
        LAST_ERROR.with(|slot| {
            *slot.borrow_mut() = message;
        });
    }

    fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
        if let Some(message) = payload.downcast_ref::<&str>() {
            (*message).to_owned()
        } else if let Some(message) = payload.downcast_ref::<String>() {
            message.clone()
        } else {
            "non-string panic payload".to_owned()
        }
    }

    fn run_ffi(operation: impl FnOnce() -> Result<(), CudaExecutorError>) -> c_int {
        match catch_unwind(AssertUnwindSafe(operation)) {
            Ok(Ok(())) => SUCCESS,
            Ok(Err(error)) => {
                let status = status_for_error(&error);
                set_last_error(error.to_string());
                status
            }
            Err(payload) => {
                set_last_error(format!(
                    "panic inside Loom Rust CUDA bridge: {}",
                    panic_message(payload)
                ));
                LAUNCH_ERROR
            }
        }
    }

    /// Return the detailed error for the most recent failed bridge call on
    /// this host thread.
    #[no_mangle]
    pub extern "C" fn loom_cuda_bridge_last_error_message() -> *const c_char {
        LAST_ERROR.with(|message| message.borrow().as_ptr())
    }

    /// Launch checked F32 Add+RMSNorm through borrowed Rust CUDA resources.
    ///
    /// # Safety
    ///
    /// Device pointers, element counts, active CUDA context, stream lifetime,
    /// and asynchronous allocation lifetime must satisfy the bridge header.
    #[no_mangle]
    pub unsafe extern "C" fn loom_cuda_bridge_add_rms_norm_f32(
        input: *mut f32,
        input_elements: u64,
        residual: *mut f32,
        residual_elements: u64,
        weight: *const f32,
        weight_elements: u64,
        rows: u32,
        hidden_size: u32,
        epsilon: f32,
        stream: *mut c_void,
    ) -> c_int {
        run_ffi(|| unsafe {
            launch_add_rms_norm(
                input,
                input_elements,
                residual,
                residual_elements,
                weight,
                weight_elements,
                rows,
                hidden_size,
                epsilon,
                stream,
            )
        })
    }

    /// Launch checked FP16 Add+RMSNorm through borrowed Rust CUDA resources.
    ///
    /// # Safety
    ///
    /// Device pointers, element counts, active CUDA context, stream lifetime,
    /// and asynchronous allocation lifetime must satisfy the bridge header.
    #[no_mangle]
    pub unsafe extern "C" fn loom_cuda_bridge_add_rms_norm_f16(
        input: *mut u16,
        input_elements: u64,
        residual: *mut u16,
        residual_elements: u64,
        weight: *const u16,
        weight_elements: u64,
        rows: u32,
        hidden_size: u32,
        epsilon: f32,
        stream: *mut c_void,
    ) -> c_int {
        run_ffi(|| unsafe {
            launch_add_rms_norm(
                input.cast::<f16>(),
                input_elements,
                residual.cast::<f16>(),
                residual_elements,
                weight.cast::<f16>(),
                weight_elements,
                rows,
                hidden_size,
                epsilon,
                stream,
            )
        })
    }

    /// Launch checked BF16 Add+RMSNorm through borrowed Rust CUDA resources.
    ///
    /// # Safety
    ///
    /// Device pointers, element counts, active CUDA context, stream lifetime,
    /// and asynchronous allocation lifetime must satisfy the bridge header.
    #[no_mangle]
    pub unsafe extern "C" fn loom_cuda_bridge_add_rms_norm_bf16(
        input: *mut u16,
        input_elements: u64,
        residual: *mut u16,
        residual_elements: u64,
        weight: *const u16,
        weight_elements: u64,
        rows: u32,
        hidden_size: u32,
        epsilon: f32,
        stream: *mut c_void,
    ) -> c_int {
        run_ffi(|| unsafe {
            launch_add_rms_norm(
                input.cast::<bf16>(),
                input_elements,
                residual.cast::<bf16>(),
                residual_elements,
                weight.cast::<bf16>(),
                weight_elements,
                rows,
                hidden_size,
                epsilon,
                stream,
            )
        })
    }

    /// Launch checked F32 RMSNorm plus dynamic per-token FP8 quantization.
    ///
    /// # Safety
    ///
    /// Device pointers, element counts, active CUDA context, stream lifetime,
    /// and asynchronous allocation lifetime must satisfy the bridge header.
    #[no_mangle]
    pub unsafe extern "C" fn loom_cuda_bridge_rms_norm_dynamic_fp8_f32(
        input: *const f32,
        input_elements: u64,
        weight: *const f32,
        weight_elements: u64,
        output: *mut u8,
        output_elements: u64,
        scales: *mut f32,
        scale_elements: u64,
        rows: u32,
        hidden_size: u32,
        epsilon: f32,
        stream: *mut c_void,
    ) -> c_int {
        run_ffi(|| unsafe {
            launch_rms_norm_dynamic_fp8(
                input,
                input_elements,
                weight,
                weight_elements,
                output,
                output_elements,
                scales,
                scale_elements,
                rows,
                hidden_size,
                epsilon,
                stream,
            )
        })
    }

    /// Launch checked FP16 RMSNorm plus dynamic per-token FP8 quantization.
    ///
    /// # Safety
    ///
    /// Device pointers, element counts, active CUDA context, stream lifetime,
    /// and asynchronous allocation lifetime must satisfy the bridge header.
    #[no_mangle]
    pub unsafe extern "C" fn loom_cuda_bridge_rms_norm_dynamic_fp8_f16(
        input: *const u16,
        input_elements: u64,
        weight: *const u16,
        weight_elements: u64,
        output: *mut u8,
        output_elements: u64,
        scales: *mut f32,
        scale_elements: u64,
        rows: u32,
        hidden_size: u32,
        epsilon: f32,
        stream: *mut c_void,
    ) -> c_int {
        run_ffi(|| unsafe {
            launch_rms_norm_dynamic_fp8(
                input.cast::<f16>(),
                input_elements,
                weight.cast::<f16>(),
                weight_elements,
                output,
                output_elements,
                scales,
                scale_elements,
                rows,
                hidden_size,
                epsilon,
                stream,
            )
        })
    }

    /// Launch checked BF16 RMSNorm plus dynamic per-token FP8 quantization.
    ///
    /// # Safety
    ///
    /// Device pointers, element counts, active CUDA context, stream lifetime,
    /// and asynchronous allocation lifetime must satisfy the bridge header.
    #[no_mangle]
    pub unsafe extern "C" fn loom_cuda_bridge_rms_norm_dynamic_fp8_bf16(
        input: *const u16,
        input_elements: u64,
        weight: *const u16,
        weight_elements: u64,
        output: *mut u8,
        output_elements: u64,
        scales: *mut f32,
        scale_elements: u64,
        rows: u32,
        hidden_size: u32,
        epsilon: f32,
        stream: *mut c_void,
    ) -> c_int {
        run_ffi(|| unsafe {
            launch_rms_norm_dynamic_fp8(
                input.cast::<bf16>(),
                input_elements,
                weight.cast::<bf16>(),
                weight_elements,
                output,
                output_elements,
                scales,
                scale_elements,
                rows,
                hidden_size,
                epsilon,
                stream,
            )
        })
    }

    /// Launch checked contiguous F32 greedy selection, logprobs, and ranks.
    ///
    /// # Safety
    ///
    /// Device pointers, exact element counts, active CUDA context, stream
    /// lifetime, and asynchronous allocation lifetime must satisfy the bridge
    /// header.
    #[no_mangle]
    pub unsafe extern "C" fn loom_cuda_bridge_greedy_sample_logprobs_f32(
        logits: *const f32,
        logits_elements: u64,
        token_ids: *mut i32,
        token_id_elements: u64,
        logprobs: *mut f32,
        logprob_elements: u64,
        ranks: *mut i64,
        rank_elements: u64,
        rows: u32,
        vocab_size: u32,
        stream: *mut c_void,
    ) -> c_int {
        run_ffi(|| unsafe {
            launch_greedy_sample_logprobs(
                logits,
                logits_elements,
                token_ids,
                token_id_elements,
                logprobs,
                logprob_elements,
                ranks,
                rank_elements,
                rows,
                vocab_size,
                stream,
            )
        })
    }

    /// Launch checked contiguous FP16 greedy selection, logprobs, and ranks.
    ///
    /// # Safety
    ///
    /// Device pointers, exact element counts, active CUDA context, stream
    /// lifetime, and asynchronous allocation lifetime must satisfy the bridge
    /// header.
    #[no_mangle]
    pub unsafe extern "C" fn loom_cuda_bridge_greedy_sample_logprobs_f16(
        logits: *const u16,
        logits_elements: u64,
        token_ids: *mut i32,
        token_id_elements: u64,
        logprobs: *mut f32,
        logprob_elements: u64,
        ranks: *mut i64,
        rank_elements: u64,
        rows: u32,
        vocab_size: u32,
        stream: *mut c_void,
    ) -> c_int {
        run_ffi(|| unsafe {
            launch_greedy_sample_logprobs(
                logits.cast::<f16>(),
                logits_elements,
                token_ids,
                token_id_elements,
                logprobs,
                logprob_elements,
                ranks,
                rank_elements,
                rows,
                vocab_size,
                stream,
            )
        })
    }

    /// Launch checked contiguous BF16 greedy selection, logprobs, and ranks.
    ///
    /// # Safety
    ///
    /// Device pointers, exact element counts, active CUDA context, stream
    /// lifetime, and asynchronous allocation lifetime must satisfy the bridge
    /// header.
    #[no_mangle]
    pub unsafe extern "C" fn loom_cuda_bridge_greedy_sample_logprobs_bf16(
        logits: *const u16,
        logits_elements: u64,
        token_ids: *mut i32,
        token_id_elements: u64,
        logprobs: *mut f32,
        logprob_elements: u64,
        ranks: *mut i64,
        rank_elements: u64,
        rows: u32,
        vocab_size: u32,
        stream: *mut c_void,
    ) -> c_int {
        run_ffi(|| unsafe {
            launch_greedy_sample_logprobs(
                logits.cast::<bf16>(),
                logits_elements,
                token_ids,
                token_id_elements,
                logprobs,
                logprob_elements,
                ranks,
                rank_elements,
                rows,
                vocab_size,
                stream,
            )
        })
    }

    /// Return successful Add+RMSNorm submissions through the Rust bridge.
    #[no_mangle]
    pub extern "C" fn loom_cuda_bridge_add_rms_norm_launch_count() -> u64 {
        ADD_RMS_NORM_LAUNCHES.load(Ordering::Relaxed)
    }

    /// Reset Add+RMSNorm bridge launch telemetry.
    #[no_mangle]
    pub extern "C" fn loom_cuda_bridge_reset_add_rms_norm_launch_count() {
        ADD_RMS_NORM_LAUNCHES.store(0, Ordering::Relaxed);
    }

    /// Return successful RMSNorm+FP8 submissions through the Rust bridge.
    #[no_mangle]
    pub extern "C" fn loom_cuda_bridge_rms_norm_dynamic_fp8_launch_count() -> u64 {
        RMS_NORM_DYNAMIC_FP8_LAUNCHES.load(Ordering::Relaxed)
    }

    /// Reset RMSNorm+FP8 bridge launch telemetry.
    #[no_mangle]
    pub extern "C" fn loom_cuda_bridge_reset_rms_norm_dynamic_fp8_launch_count() {
        RMS_NORM_DYNAMIC_FP8_LAUNCHES.store(0, Ordering::Relaxed);
    }

    /// Return successful greedy-sampling submissions through the Rust bridge.
    #[no_mangle]
    pub extern "C" fn loom_cuda_bridge_greedy_sample_logprobs_launch_count() -> u64 {
        GREEDY_SAMPLE_LOGPROBS_LAUNCHES.load(Ordering::Relaxed)
    }

    /// Reset greedy-sampling bridge launch telemetry.
    #[no_mangle]
    pub extern "C" fn loom_cuda_bridge_reset_greedy_sample_logprobs_launch_count() {
        GREEDY_SAMPLE_LOGPROBS_LAUNCHES.store(0, Ordering::Relaxed);
    }

    #[cfg(test)]
    mod tests {
        use super::{checked_byte_range, launch_greedy_sample_logprobs, ranges_overlap};
        use loom_cuda::CudaExecutorError;

        #[test]
        fn byte_ranges_reject_overflow_and_detect_overlap() {
            let first = checked_byte_range(0x1000_usize as *const f32, 8, "first").unwrap();
            let overlapping = checked_byte_range(0x1010_usize as *const f32, 8, "overlap").unwrap();
            let disjoint = checked_byte_range(0x1020_usize as *const f32, 8, "disjoint").unwrap();
            assert!(ranges_overlap(first, overlapping));
            assert!(!ranges_overlap(first, disjoint));

            assert!(checked_byte_range::<f32>(std::ptr::null(), 8, "null").is_err());
            assert!(checked_byte_range(0x1000_usize as *const f32, 0, "empty").is_err());
            assert!(checked_byte_range(
                0x1000_usize as *const f32,
                usize::MAX / std::mem::size_of::<f32>() + 1,
                "overflow",
            )
            .is_err());
        }

        #[test]
        fn greedy_regions_reject_bad_lengths_and_overlap_before_submission() {
            let short_logits = unsafe {
                launch_greedy_sample_logprobs::<f32>(
                    0x1000_usize as *const f32,
                    7,
                    0x2000_usize as *mut i32,
                    2,
                    0x3000_usize as *mut f32,
                    2,
                    0x4000_usize as *mut i64,
                    2,
                    2,
                    4,
                    std::ptr::null_mut(),
                )
            };
            assert!(matches!(
                short_logits,
                Err(CudaExecutorError::InvalidContract(_))
            ));

            let overlapping_output = unsafe {
                launch_greedy_sample_logprobs::<f32>(
                    0x1000_usize as *const f32,
                    8,
                    0x1000_usize as *mut i32,
                    2,
                    0x3000_usize as *mut f32,
                    2,
                    0x4000_usize as *mut i64,
                    2,
                    2,
                    4,
                    std::ptr::null_mut(),
                )
            };
            assert!(matches!(
                overlapping_output,
                Err(CudaExecutorError::InvalidContract(_))
            ));
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda::*;
