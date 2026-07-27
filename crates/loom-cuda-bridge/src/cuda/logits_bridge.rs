//! Checked C bridge entrypoints for logits processing.

use super::*;

#[allow(clippy::too_many_arguments)]
unsafe fn launch_logits_preprocess(
    logits: *mut f32,
    logits_elements: u64,
    temperatures: *const f32,
    temperature_elements: u64,
    blocked_mask: *const u8,
    blocked_mask_elements: u64,
    bias_row_ids: *const i32,
    bias_row_id_elements: u64,
    bias_token_ids: *const i32,
    bias_token_id_elements: u64,
    bias_values: *const f32,
    bias_value_elements: u64,
    suppressed_row_ids: *const i32,
    suppressed_row_id_elements: u64,
    suppressed_token_ids: *const i32,
    suppressed_token_id_elements: u64,
    rows: u32,
    vocab_size: u32,
    row_stride: u64,
    stream: *mut c_void,
) -> Result<(), CudaExecutorError> {
    let (mut logits, logits_range) =
        unsafe { write_slice(logits, logits_elements, "logits-preprocessing logits") }?;
    let (temperatures, temperature_range) = unsafe {
        read_slice(
            temperatures,
            temperature_elements,
            "logits-preprocessing temperatures",
        )
    }?;
    let (blocked_mask, blocked_mask_range) = unsafe {
        read_optional_slice(
            blocked_mask,
            blocked_mask_elements,
            "logits-preprocessing blocked mask",
        )
    }?;
    let (bias_row_ids, bias_row_id_range) = unsafe {
        read_optional_slice(
            bias_row_ids,
            bias_row_id_elements,
            "logits-preprocessing bias row IDs",
        )
    }?;
    let (bias_token_ids, bias_token_id_range) = unsafe {
        read_optional_slice(
            bias_token_ids,
            bias_token_id_elements,
            "logits-preprocessing bias token IDs",
        )
    }?;
    let (bias_values, bias_value_range) = unsafe {
        read_optional_slice(
            bias_values,
            bias_value_elements,
            "logits-preprocessing bias values",
        )
    }?;
    let (suppressed_row_ids, suppressed_row_id_range) = unsafe {
        read_optional_slice(
            suppressed_row_ids,
            suppressed_row_id_elements,
            "logits-preprocessing suppressed row IDs",
        )
    }?;
    let (suppressed_token_ids, suppressed_token_id_range) = unsafe {
        read_optional_slice(
            suppressed_token_ids,
            suppressed_token_id_elements,
            "logits-preprocessing suppressed token IDs",
        )
    }?;

    require_disjoint_from(
        "logits",
        logits_range,
        &[("temperatures", temperature_range)],
        "logits preprocessing",
    )?;
    for (name, range) in [
        ("blocked mask", blocked_mask_range),
        ("bias row IDs", bias_row_id_range),
        ("bias token IDs", bias_token_id_range),
        ("bias values", bias_value_range),
        ("suppressed row IDs", suppressed_row_id_range),
        ("suppressed token IDs", suppressed_token_id_range),
    ] {
        if let Some(range) = range {
            require_disjoint_from(
                "logits",
                logits_range,
                &[(name, range)],
                "logits preprocessing",
            )?;
        }
    }

    let spec = LogitsPreprocessSpec::new(
        rows as usize,
        vocab_size as usize,
        blocked_mask.is_some(),
        bias_row_ids.as_ref().map_or(0, DeviceSlice::len),
        suppressed_row_ids.as_ref().map_or(0, DeviceSlice::len),
    )
    .map_err(invalid_contract)?;
    let layout = RowStridedLayout::new(
        vocab_size as usize,
        element_count(row_stride, "logits-preprocessing row stride")?,
    )?;
    stream_backend(stream).logits_preprocess_f32(
        &mut logits,
        &temperatures,
        blocked_mask
            .as_ref()
            .map(|values| values as &dyn loom_cuda::runtime::CudaDeviceRead<u8>),
        bias_row_ids
            .as_ref()
            .map(|values| values as &dyn loom_cuda::runtime::CudaDeviceRead<i32>),
        bias_token_ids
            .as_ref()
            .map(|values| values as &dyn loom_cuda::runtime::CudaDeviceRead<i32>),
        bias_values
            .as_ref()
            .map(|values| values as &dyn loom_cuda::runtime::CudaDeviceRead<f32>),
        suppressed_row_ids
            .as_ref()
            .map(|values| values as &dyn loom_cuda::runtime::CudaDeviceRead<i32>),
        suppressed_token_ids
            .as_ref()
            .map(|values| values as &dyn loom_cuda::runtime::CudaDeviceRead<i32>),
        spec,
        layout,
    )?;
    record_launch(OP_LOGITS_PREPROCESS);
    Ok(())
}

/// Checked fused F32 logits preprocessing.
///
/// Optional pointer/count pairs must both be zero/null or both be present.
/// Sparse bias tensors form one group and sparse suppression tensors another.
///
/// # Safety
///
/// Every present pointer must identify the declared CUDA storage on the active
/// context and remain alive until work on `stream` completes.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn loom_cuda_bridge_logits_preprocess(
    logits: *mut f32,
    logits_elements: u64,
    temperatures: *const f32,
    temperature_elements: u64,
    blocked_mask: *const u8,
    blocked_mask_elements: u64,
    bias_row_ids: *const i32,
    bias_row_id_elements: u64,
    bias_token_ids: *const i32,
    bias_token_id_elements: u64,
    bias_values: *const f32,
    bias_value_elements: u64,
    suppressed_row_ids: *const i32,
    suppressed_row_id_elements: u64,
    suppressed_token_ids: *const i32,
    suppressed_token_id_elements: u64,
    rows: u32,
    vocab_size: u32,
    row_stride: u64,
    stream: *mut c_void,
) -> c_int {
    bridge_call(|| unsafe {
        launch_logits_preprocess(
            logits,
            logits_elements,
            temperatures,
            temperature_elements,
            blocked_mask,
            blocked_mask_elements,
            bias_row_ids,
            bias_row_id_elements,
            bias_token_ids,
            bias_token_id_elements,
            bias_values,
            bias_value_elements,
            suppressed_row_ids,
            suppressed_row_id_elements,
            suppressed_token_ids,
            suppressed_token_id_elements,
            rows,
            vocab_size,
            row_stride,
            stream,
        )
    })
}

#[allow(clippy::too_many_arguments)]
unsafe fn launch_min_p_filter<T: Scalar>(
    logits: *mut T,
    logits_elements: u64,
    min_p: *const f32,
    min_p_elements: u64,
    rows: u32,
    vocab_size: u32,
    row_stride: u64,
    stream: *mut c_void,
) -> Result<(), CudaExecutorError> {
    let (mut logits, logits_range) =
        unsafe { write_slice(logits, logits_elements, "min-p logits") }?;
    let (min_p, min_p_range) = unsafe { read_slice(min_p, min_p_elements, "min-p values") }?;
    require_disjoint(
        &[("logits", logits_range), ("min-p", min_p_range)],
        "min-p filtering",
    )?;
    let spec = MinPFilterSpec::new(rows as usize, vocab_size as usize, T::DTYPE)
        .map_err(invalid_contract)?;
    let layout = RowStridedLayout::new(
        vocab_size as usize,
        element_count(row_stride, "min-p row stride")?,
    )?;
    T::min_p_filter(&stream_backend(stream), &mut logits, &min_p, spec, layout)?;
    record_launch(OP_MIN_P_FILTER);
    Ok(())
}

/// Checked in-place min-p filtering.
///
/// # Safety
///
/// Every pointer must identify the declared CUDA storage on the active
/// context and remain alive until work on `stream` completes.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn loom_cuda_bridge_min_p_filter(
    dtype: u32,
    logits: *mut c_void,
    logits_elements: u64,
    min_p: *const f32,
    min_p_elements: u64,
    rows: u32,
    vocab_size: u32,
    row_stride: u64,
    stream: *mut c_void,
) -> c_int {
    bridge_call(|| {
        let kind = scalar_kind(dtype)?;
        dispatch_scalar!(
            kind,
            launch_min_p_filter(
                logits.cast(),
                logits_elements,
                min_p,
                min_p_elements,
                rows,
                vocab_size,
                row_stride,
                stream,
            )
        )
    })
}
