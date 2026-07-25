//! Checked C bridge entrypoints for token selection and logprobs.

use super::*;

#[allow(clippy::too_many_arguments)]
unsafe fn launch_greedy_sample_logprobs<T: Scalar>(
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
    row_stride: u64,
    stream: *mut c_void,
) -> Result<(), CudaExecutorError> {
    let (logits, logits_range) = unsafe { read_slice(logits, logits_elements, "greedy logits") }?;
    let (mut token_ids, token_ids_range) =
        unsafe { write_slice(token_ids, token_id_elements, "greedy token IDs") }?;
    let (mut logprobs, logprobs_range) =
        unsafe { write_slice(logprobs, logprob_elements, "greedy logprobs") }?;
    let (mut ranks, ranks_range) = unsafe { write_slice(ranks, rank_elements, "greedy ranks") }?;
    require_disjoint(
        &[
            ("logits", logits_range),
            ("token IDs", token_ids_range),
            ("logprobs", logprobs_range),
            ("ranks", ranks_range),
        ],
        "greedy sampling",
    )?;
    let spec = GreedySampleLogprobsSpec::new(rows as usize, vocab_size as usize, T::DTYPE)
        .map_err(invalid_contract)?;
    let layout = RowStridedLayout::new(
        vocab_size as usize,
        element_count(row_stride, "greedy row stride")?,
    )?;
    T::greedy_sample_logprobs(
        &stream_backend(stream),
        &logits,
        &mut token_ids,
        &mut logprobs,
        &mut ranks,
        spec,
        layout,
    )?;
    record_launch(OP_GREEDY_SAMPLE_LOGPROBS);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
unsafe fn launch_selected_token_logprobs<T: Scalar>(
    logits: *const T,
    logits_elements: u64,
    token_ids: *const i64,
    token_id_elements: u64,
    logprobs: *mut f32,
    logprob_elements: u64,
    ranks: *mut i64,
    rank_elements: u64,
    rows: u32,
    vocab_size: u32,
    row_stride: u64,
    stream: *mut c_void,
) -> Result<(), CudaExecutorError> {
    let (logits, logits_range) =
        unsafe { read_slice(logits, logits_elements, "selected-token logits") }?;
    let (token_ids, token_ids_range) =
        unsafe { read_slice(token_ids, token_id_elements, "selected-token IDs") }?;
    let (mut logprobs, logprobs_range) =
        unsafe { write_slice(logprobs, logprob_elements, "selected-token logprobs") }?;
    let (mut ranks, ranks_range) =
        unsafe { write_slice(ranks, rank_elements, "selected-token ranks") }?;
    require_disjoint(
        &[
            ("logits", logits_range),
            ("token IDs", token_ids_range),
            ("logprobs", logprobs_range),
            ("ranks", ranks_range),
        ],
        "selected-token logprobs",
    )?;
    let spec = SelectedTokenLogprobsSpec::new(rows as usize, vocab_size as usize, T::DTYPE)
        .map_err(invalid_contract)?;
    let layout = RowStridedLayout::new(
        vocab_size as usize,
        element_count(row_stride, "selected-token row stride")?,
    )?;
    T::selected_token_logprobs(
        &stream_backend(stream),
        &logits,
        &token_ids,
        &mut logprobs,
        &mut ranks,
        spec,
        layout,
    )?;
    record_launch(OP_SELECTED_TOKEN_LOGPROBS);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
unsafe fn launch_topk_sampled_logprobs<T: Scalar>(
    logits: *const T,
    logits_elements: u64,
    sampled_token_ids: *const i64,
    sampled_token_id_elements: u64,
    output_token_ids: *mut i32,
    output_token_id_elements: u64,
    output_logprobs: *mut f32,
    output_logprob_elements: u64,
    sampled_token_ranks: *mut i64,
    sampled_token_rank_elements: u64,
    workspace: *mut u8,
    workspace_elements: u64,
    rows: u32,
    vocab_size: u32,
    top_k: u32,
    row_stride: u64,
    stream: *mut c_void,
) -> Result<(), CudaExecutorError> {
    let (logits, logits_range) =
        unsafe { read_slice(logits, logits_elements, "top-k sampled-logprob logits") }?;
    let (sampled_token_ids, sampled_ids_range) = unsafe {
        read_slice(
            sampled_token_ids,
            sampled_token_id_elements,
            "sampled token IDs",
        )
    }?;
    let (mut output_token_ids, output_ids_range) = unsafe {
        write_slice(
            output_token_ids,
            output_token_id_elements,
            "top-k output token IDs",
        )
    }?;
    let (mut output_logprobs, output_logprobs_range) = unsafe {
        write_slice(
            output_logprobs,
            output_logprob_elements,
            "top-k output logprobs",
        )
    }?;
    let (mut sampled_token_ranks, sampled_ranks_range) = unsafe {
        write_slice(
            sampled_token_ranks,
            sampled_token_rank_elements,
            "sampled token ranks",
        )
    }?;
    if !(workspace as usize).is_multiple_of(align_of::<f32>()) {
        return Err(CudaExecutorError::InvalidContract(
            "top-k sampled-logprob workspace must be aligned to 4 bytes".into(),
        ));
    }
    let (mut workspace, workspace_range) = unsafe {
        write_slice(
            workspace,
            workspace_elements,
            "top-k sampled-logprob workspace",
        )
    }?;
    require_disjoint(
        &[
            ("logits", logits_range),
            ("sampled token IDs", sampled_ids_range),
            ("output token IDs", output_ids_range),
            ("output logprobs", output_logprobs_range),
            ("sampled token ranks", sampled_ranks_range),
            ("workspace", workspace_range),
        ],
        "top-k sampled logprobs",
    )?;
    let spec =
        TopKSampledLogprobsSpec::new(rows as usize, vocab_size as usize, top_k as usize, T::DTYPE)
            .map_err(invalid_contract)?;
    let layout = RowStridedLayout::new(
        spec.vocab_size(),
        element_count(row_stride, "top-k sampled-logprob row stride")?,
    )?;
    T::topk_sampled_logprobs(
        &stream_backend(stream),
        &logits,
        &sampled_token_ids,
        &mut output_token_ids,
        &mut output_logprobs,
        &mut sampled_token_ranks,
        &mut workspace,
        spec,
        layout,
    )?;
    record_launch(OP_TOPK_SAMPLED_LOGPROBS);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
unsafe fn launch_token_penalties(
    logits: *mut f32,
    logits_elements: u64,
    prompt_token_ids: *const i64,
    prompt_token_id_elements: u64,
    output_token_ids: *const i64,
    output_token_id_elements: u64,
    presence_penalties: *const f32,
    presence_penalty_elements: u64,
    frequency_penalties: *const f32,
    frequency_penalty_elements: u64,
    repetition_penalties: *const f32,
    repetition_penalty_elements: u64,
    workspace: *mut u64,
    workspace_elements: u64,
    rows: u32,
    vocab_size: u32,
    prompt_tokens: u32,
    output_tokens: u32,
    workspace_capacity: u32,
    logits_row_stride: u64,
    prompt_row_stride: u64,
    output_row_stride: u64,
    workspace_row_stride: u64,
    stream: *mut c_void,
) -> Result<(), CudaExecutorError> {
    let (mut logits, logits_range) =
        unsafe { write_slice(logits, logits_elements, "token-penalty logits") }?;
    let (prompt_token_ids, prompt_range) = unsafe {
        read_slice(
            prompt_token_ids,
            prompt_token_id_elements,
            "token-penalty prompt IDs",
        )
    }?;
    let (output_token_ids, output_range) = unsafe {
        read_slice(
            output_token_ids,
            output_token_id_elements,
            "token-penalty output IDs",
        )
    }?;
    let (presence_penalties, presence_range) = unsafe {
        read_slice(
            presence_penalties,
            presence_penalty_elements,
            "presence penalties",
        )
    }?;
    let (frequency_penalties, frequency_range) = unsafe {
        read_slice(
            frequency_penalties,
            frequency_penalty_elements,
            "frequency penalties",
        )
    }?;
    let (repetition_penalties, repetition_range) = unsafe {
        read_slice(
            repetition_penalties,
            repetition_penalty_elements,
            "repetition penalties",
        )
    }?;
    let (mut workspace, workspace_range) =
        unsafe { write_slice(workspace, workspace_elements, "token-penalty workspace") }?;
    let read_regions = [
        ("prompt IDs", prompt_range),
        ("output IDs", output_range),
        ("presence penalties", presence_range),
        ("frequency penalties", frequency_range),
        ("repetition penalties", repetition_range),
    ];
    require_disjoint_from(
        "logits",
        logits_range,
        &[
            read_regions[0],
            read_regions[1],
            read_regions[2],
            read_regions[3],
            read_regions[4],
            ("workspace", workspace_range),
        ],
        "token penalties",
    )?;
    require_disjoint_from(
        "workspace",
        workspace_range,
        &read_regions,
        "token penalties",
    )?;

    let spec = TokenPenaltiesSpec::new(
        rows as usize,
        vocab_size as usize,
        prompt_tokens as usize,
        output_tokens as usize,
        workspace_capacity as usize,
    )
    .map_err(invalid_contract)?;
    let logits_layout = RowStridedLayout::new(
        spec.vocab_size(),
        element_count(logits_row_stride, "token-penalty logits row stride")?,
    )?;
    let prompt_layout = RowStridedLayout::new(
        spec.prompt_tokens(),
        element_count(prompt_row_stride, "token-penalty prompt row stride")?,
    )?;
    let output_layout = RowStridedLayout::new(
        spec.output_tokens(),
        element_count(output_row_stride, "token-penalty output row stride")?,
    )?;
    let workspace_layout = RowStridedLayout::new(
        spec.workspace_capacity(),
        element_count(workspace_row_stride, "token-penalty workspace row stride")?,
    )?;
    stream_backend(stream).apply_token_penalties_f32(
        &mut logits,
        &prompt_token_ids,
        &output_token_ids,
        &presence_penalties,
        &frequency_penalties,
        &repetition_penalties,
        &mut workspace,
        spec,
        logits_layout,
        prompt_layout,
        output_layout,
        workspace_layout,
    )?;
    record_launch(OP_TOKEN_PENALTIES);
    Ok(())
}

/// Checked greedy selection, sampled-token logprob, and rank.
///
/// # Safety
///
/// Every pointer must identify the declared CUDA storage on the active
/// context and remain alive until work on `stream` completes.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn loom_cuda_bridge_greedy_sample_logprobs(
    dtype: u32,
    logits: *const c_void,
    logits_elements: u64,
    token_ids: *mut i32,
    token_id_elements: u64,
    logprobs: *mut f32,
    logprob_elements: u64,
    ranks: *mut i64,
    rank_elements: u64,
    rows: u32,
    vocab_size: u32,
    row_stride: u64,
    stream: *mut c_void,
) -> c_int {
    bridge_call(|| {
        let kind = scalar_kind(dtype)?;
        dispatch_scalar!(
            kind,
            launch_greedy_sample_logprobs(
                logits.cast(),
                logits_elements,
                token_ids,
                token_id_elements,
                logprobs,
                logprob_elements,
                ranks,
                rank_elements,
                rows,
                vocab_size,
                row_stride,
                stream,
            )
        )
    })
}

/// Checked selected-token logprob and rank.
///
/// # Safety
///
/// Every pointer must identify the declared CUDA storage on the active
/// context and remain alive until work on `stream` completes.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn loom_cuda_bridge_selected_token_logprobs(
    dtype: u32,
    logits: *const c_void,
    logits_elements: u64,
    token_ids: *const i64,
    token_id_elements: u64,
    logprobs: *mut f32,
    logprob_elements: u64,
    ranks: *mut i64,
    rank_elements: u64,
    rows: u32,
    vocab_size: u32,
    row_stride: u64,
    stream: *mut c_void,
) -> c_int {
    bridge_call(|| {
        let kind = scalar_kind(dtype)?;
        dispatch_scalar!(
            kind,
            launch_selected_token_logprobs(
                logits.cast(),
                logits_elements,
                token_ids,
                token_id_elements,
                logprobs,
                logprob_elements,
                ranks,
                rank_elements,
                rows,
                vocab_size,
                row_stride,
                stream,
            )
        )
    })
}

/// Checked sampled-token plus deterministic top-k logprobs and rank.
///
/// # Safety
///
/// Every pointer must identify the declared CUDA storage on the active
/// context and remain alive until work on `stream` completes. `workspace`
/// must be aligned to at least four bytes.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn loom_cuda_bridge_topk_sampled_logprobs(
    dtype: u32,
    logits: *const c_void,
    logits_elements: u64,
    sampled_token_ids: *const i64,
    sampled_token_id_elements: u64,
    output_token_ids: *mut i32,
    output_token_id_elements: u64,
    output_logprobs: *mut f32,
    output_logprob_elements: u64,
    sampled_token_ranks: *mut i64,
    sampled_token_rank_elements: u64,
    workspace: *mut u8,
    workspace_elements: u64,
    rows: u32,
    vocab_size: u32,
    top_k: u32,
    row_stride: u64,
    stream: *mut c_void,
) -> c_int {
    bridge_call(|| {
        let kind = scalar_kind(dtype)?;
        dispatch_scalar!(
            kind,
            launch_topk_sampled_logprobs(
                logits.cast(),
                logits_elements,
                sampled_token_ids,
                sampled_token_id_elements,
                output_token_ids,
                output_token_id_elements,
                output_logprobs,
                output_logprob_elements,
                sampled_token_ranks,
                sampled_token_rank_elements,
                workspace,
                workspace_elements,
                rows,
                vocab_size,
                top_k,
                row_stride,
                stream,
            )
        )
    })
}

/// Return the caller-owned byte workspace required by top-k logprobs.
///
/// # Safety
///
/// `workspace_bytes` must be a valid aligned writable host pointer.
#[no_mangle]
pub unsafe extern "C" fn loom_cuda_bridge_topk_sampled_logprobs_workspace_size(
    rows: u32,
    vocab_size: u32,
    top_k: u32,
    workspace_bytes: *mut u64,
) -> c_int {
    bridge_call(|| {
        if workspace_bytes.is_null()
            || !(workspace_bytes as usize).is_multiple_of(align_of::<u64>())
        {
            return Err(CudaExecutorError::InvalidContract(
                "top-k workspace-size output is null or misaligned".into(),
            ));
        }
        let spec = TopKSampledLogprobsSpec::new(
            rows as usize,
            vocab_size as usize,
            top_k as usize,
            DType::F32,
        )
        .map_err(invalid_contract)?;
        unsafe {
            *workspace_bytes = u64::try_from(spec.workspace_bytes()).map_err(|_| {
                CudaExecutorError::InvalidContract(
                    "top-k workspace size exceeds the bridge ABI".into(),
                )
            })?;
        }
        Ok(())
    })
}

/// Checked in-place sparse token penalties.
///
/// # Safety
///
/// Every pointer must identify the declared CUDA storage on the active
/// context and remain alive until work on `stream` completes. Mutable logits
/// and workspace storage must not overlap any input.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn loom_cuda_bridge_apply_token_penalties(
    logits: *mut f32,
    logits_elements: u64,
    prompt_token_ids: *const i64,
    prompt_token_id_elements: u64,
    output_token_ids: *const i64,
    output_token_id_elements: u64,
    presence_penalties: *const f32,
    presence_penalty_elements: u64,
    frequency_penalties: *const f32,
    frequency_penalty_elements: u64,
    repetition_penalties: *const f32,
    repetition_penalty_elements: u64,
    workspace: *mut u64,
    workspace_elements: u64,
    rows: u32,
    vocab_size: u32,
    prompt_tokens: u32,
    output_tokens: u32,
    workspace_capacity: u32,
    logits_row_stride: u64,
    prompt_row_stride: u64,
    output_row_stride: u64,
    workspace_row_stride: u64,
    stream: *mut c_void,
) -> c_int {
    bridge_call(|| unsafe {
        launch_token_penalties(
            logits,
            logits_elements,
            prompt_token_ids,
            prompt_token_id_elements,
            output_token_ids,
            output_token_id_elements,
            presence_penalties,
            presence_penalty_elements,
            frequency_penalties,
            frequency_penalty_elements,
            repetition_penalties,
            repetition_penalty_elements,
            workspace,
            workspace_elements,
            rows,
            vocab_size,
            prompt_tokens,
            output_tokens,
            workspace_capacity,
            logits_row_stride,
            prompt_row_stride,
            output_row_stride,
            workspace_row_stride,
            stream,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::{launch_greedy_sample_logprobs, launch_token_penalties};
    use loom_cuda::CudaExecutorError;

    #[test]
    fn greedy_rejects_bad_storage_before_submission() {
        let result = unsafe {
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
                4,
                std::ptr::null_mut(),
            )
        };
        assert!(matches!(result, Err(CudaExecutorError::InvalidContract(_))));
    }

    #[test]
    fn token_penalties_reject_a_short_workspace_before_submission() {
        let result = unsafe {
            launch_token_penalties(
                0x1000_usize as *mut f32,
                12,
                0x2000_usize as *const i64,
                10,
                0x3000_usize as *const i64,
                8,
                0x4000_usize as *const f32,
                2,
                0x5000_usize as *const f32,
                2,
                0x6000_usize as *const f32,
                2,
                0x7000_usize as *mut u64,
                32,
                2,
                6,
                5,
                4,
                32,
                6,
                5,
                4,
                32,
                std::ptr::null_mut(),
            )
        };
        assert!(matches!(result, Err(CudaExecutorError::InvalidContract(_))));
    }
}
